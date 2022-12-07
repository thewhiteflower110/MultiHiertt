import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from utils.new_retriever_utils import *
from utils.utils import *
from transformers import AutoModel, AutoTokenizer, AutoConfig,TapasConfig, TapasForQuestionAnswering
from typing import Optional, Dict, Any, Tuple, List
from transformers.optimization import AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_schedule_with_warmup
torch.cuda.empty_cache()


class RetrieverModel(LightningModule):
    
    def __init__(self, 
                 transformer_model_name: str,
                 topn: int, 
                 dropout_rate: float, 
                 warmup_steps: int = 0,
                 optimizer: Dict[str, Any] = None,
                 lr_scheduler: Dict[str, Any] = None,
                 ) -> None:

        super().__init__()
        
        self.topn = topn
        self.dropout_rate = dropout_rate

        self.transformer_model_name = "ProsusAI/finbert"
        self.model_finbert = AutoModel.from_pretrained(self.transformer_model_name)
        self.model_config_finbert = AutoConfig.from_pretrained(self.transformer_model_name)

        #self.transformer_model_name = "google/tapas-base-finetuned-wtq"
        #self.model_tapas = AutoModel.from_pretrained(self.transformer_model_name)
        self.model_config_tapas = TapasConfig.from_pretrained("google/tapas-base-finetuned-wtq")
        self.model_tapas = TapasForQuestionAnswering.from_pretrained("google/tapas-base", config=self.model_config_tapas)

        self.warmup_steps = warmup_steps

        self.criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        self.predictions: List[Dict[str, Any]] = []

        self.opt_params = optimizer["init_args"]
        self.lrs_params = lr_scheduler

        new_hidden_size = self.model_config_tapas.hidden_size + self.model_config_finbert.hidden_size
        hidden_size = self.model_config_finbert.hidden_size #728
        
        self.cmbn_prj = nn.Linear(new_hidden_size, hidden_size, bias=True)
        self.cmbn_dropout = nn.Dropout(self.dropout_rate)
        
        self.cls_prj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.cls_dropout = nn.Dropout(self.dropout_rate)
        self.cls_final = nn.Linear(hidden_size, 2, bias=True)
        
        self.predictions = []

    def forward(self, text_input_ids, text_attention_mask, text_segment_ids,table_input_ids, table_attention_mask, table_segment_ids, metadata) -> List[Dict[str, Any]]:

        table_input_ids = torch.tensor(table_input_ids).to("cuda")
        table_attention_mask = torch.tensor(table_attention_mask).to("cuda")
        table_segment_ids = torch.tensor(table_segment_ids).to("cuda")
        table_input_ids = torch.unsqueeze(table_input_ids.detach(),dim=1).to("cuda")
        table_segment_ids = torch.unsqueeze(table_segment_ids.detach(),dim=1).to("cuda")
        #print(table_segment_ids.shape, table_input_ids.shape)        
        tabert_outputs = self.model_tapas(
            input_ids=table_input_ids, attention_mask=table_attention_mask, token_type_ids=table_segment_ids)

        tabert_sequence_output = tabert_outputs.last_hidden_state

        tabert_pooled_output = tabert_sequence_output[:, 0, :]
        
        text_input_ids = torch.tensor(text_input_ids).to("cuda")
        text_attention_mask = torch.tensor(text_attention_mask).to("cuda")
        text_segment_ids = torch.tensor(text_segment_ids).to("cuda")

        finbert_outputs = self.model_finbert(
            input_ids=text_input_ids, attention_mask=text_attention_mask, token_type_ids=text_segment_ids)

        finbert_sequence_output = finbert_outputs.last_hidden_state

        finbert_pooled_output = finbert_sequence_output[:, 0, :]

        new_pooled_output = torch.cat((finbert_pooled_output,tabert_pooled_output),0)
        cmbn_pooled_output = self.cmbn_prj(new_pooled_output)
        cmbn_pooled_output = self.cmbn_dropout(cmbn_pooled_output)
        
        pooled_output = self.cls_prj(cmbn_pooled_output)
        pooled_output = self.cls_dropout(pooled_output)

        logits = self.cls_final(pooled_output)
        output_dicts = []
        for i in range(len(metadata)):
            output_dicts.append({"logits": logits[i], "filename_id": metadata[i]["filename_id"], "tab_ind": metadata[i]["tab_ind"], "text_ind": metadata[i]["text_ind"]})
        print("made output dicts")
        return output_dicts


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        batch_table, batch_text = batch
        table_input_ids = batch_table["input_ids"]
        table_attention_mask = batch_table["input_mask"]
        table_segment_ids = batch_table["segment_ids"]
        table_labels = batch_table["label"]
        print("table_input_ids",table_input_ids.shape)
        print("batch",batch.shape)
        print("batch_table",batch_table.shape)

        text_input_ids = batch_text["input_ids"]
        text_attention_mask = batch_text["input_mask"]
        text_segment_ids = batch_text["segment_ids"]
        text_labels = batch_text["label"]
        
        table_labels = torch.tensor(table_labels).to("cuda")
        text_labels = torch.tensor(text_labels).to("cuda")
        
        metadata = [{"filename_id": filename_id, "tab_ind": tab_ind, "text_ind":text_ind} for filename_id, tab_ind, text_ind in zip(batch_table["filename_id"], batch_table["ind"], batch_text["ind"])]
        
        output_dicts = self(text_input_ids, text_attention_mask, text_segment_ids, table_input_ids, table_attention_mask, table_segment_ids, metadata)
        
        #labels currently adding both
        labels=torch.cat((text_labels,text_labels),0)

        logits = []
        for output_dict in output_dicts:
            logits.append(output_dict["logits"])
        logits = torch.stack(logits)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        self.log("loss", loss.sum(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss.sum()}

    def on_fit_start(self) -> None:
        # save the code using wandb
        #if self.logger: 
        #    # if logger is initialized, save the code
        #    self.logger[0].log_code()
        #else:
        #    print("logger is not initialized, code will not be saved")  
        
        return super().on_fit_start()

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        batch_table, batch_text = batch
        table_input_ids = batch_table["input_ids"]
        table_attention_mask = batch_table["input_mask"]
        table_segment_ids = batch_table["segment_ids"]
        table_labels = batch_table["label"]

        text_input_ids = batch_text["input_ids"]
        text_attention_mask = batch_text["input_mask"]
        text_segment_ids = batch_text["segment_ids"]
        text_labels = batch_text["label"]
        
        table_labels = torch.tensor(table_labels).to("cuda")
        text_labels = torch.tensor(text_labels).to("cuda")
        
        metadata = [{"filename_id": filename_id, "tab_ind": tab_ind, "text_ind":text_ind} for filename_id, tab_ind, text_ind in zip(batch_table["filename_id"], batch_table["ind"], batch_text["ind"])]
        
        output_dicts = self(text_input_ids, text_attention_mask, text_segment_ids, table_input_ids, table_attention_mask, table_segment_ids, metadata)
        labels=torch.cat((text_labels,text_labels),0)
        logits = []
        for output_dict in output_dicts:
            logits.append(output_dict["logits"])
        logits = torch.stack(logits)
        loss = self.criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        self.log("val_loss", loss)
        return output_dicts

    def predict_step(self, batch: torch.Tensor, batch_idx: int):
        batch_table, batch_text = batch
        table_input_ids = batch_table["input_ids"]
        table_attention_mask = batch_table["input_mask"]
        table_segment_ids = batch_table["segment_ids"]
        table_labels = batch_table["label"]

        text_input_ids = batch_text["input_ids"]
        text_attention_mask = batch_text["input_mask"]
        text_segment_ids = batch_text["segment_ids"]
        text_labels = batch_text["label"]
        
        table_labels = torch.tensor(table_labels).to("cuda")
        text_labels = torch.tensor(text_labels).to("cuda")
        
        metadata = [{"filename_id": filename_id, "tab_ind": tab_ind, "text_ind":text_ind} for filename_id, tab_ind, text_ind in zip(batch_table["filename_id"], batch_table["ind"], batch_text["ind"])]
        
        output_dicts = self(text_input_ids, text_attention_mask, text_segment_ids, table_input_ids, table_attention_mask, table_segment_ids, metadata)
        return output_dicts
    

    def predict_step_end(self, outputs: List[Dict[str, Any]]) -> None:
        self.predictions.extend(outputs)

        
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.opt_params)
        if self.lrs_params["name"] == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, **self.lrs_params["init_args"])
        elif self.lrs_params["name"] == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, **self.lrs_params["init_args"])
        elif self.lrs_params["name"] == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer, **self.lrs_params["init_args"])
        else:
            raise ValueError(f"lr_scheduler {self.lrs_params} is not supported")

        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step"
                    }
                }
