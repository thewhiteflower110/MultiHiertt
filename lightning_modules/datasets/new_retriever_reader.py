import json
import logging
import sys
import os
import torch

from typing import Dict, Iterable, List, Any, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from utils.datasets_util import right_pad_sequences
import pandas as pd
from transformers import AutoTokenizer,TapasTokenizer
import utils.new_retriever_utils as retriever_utils
from utils.new_retriever_utils import *
from utils.utils import *
from torch.utils.data import DataLoader

os.environ['TOKENIZERS_PARALLELISM']='0'

class TabertRetrieverDataset(Dataset):
    def __init__(
        self, 
        transformer_model_name: str,
        file_path: str,
        inventory_file_path: str,
        max_instances: int,
        mode: str = "train", 
        **kwargs):
        super().__init__(**kwargs)

        assert mode in ["train", "test", "valid"]

        self.tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
        self.df = pd.read_csv(inventory_file_path, header=None, delimiter='|')
        self.max_instances = max_instances
        self.mode = mode
        self.instances = self.read(file_path, self.tokenizer)
        
        #self.df = pd.read_csv(inventory_file_path, header=None, delimiter='|')
        #elif mode =="valid":
        #  self.df = pd.read_csv(dev_inventory_file_path, header=None, delimiter='|')
        #elif mode=="test":
        #  self.df = pd.read_csv(test_inventory_file_path, header=None, delimiter='|')

    def read(self, input_path: str, tokenizer) -> Iterable[Dict[str, Any]]:
        with open(input_path) as input_file:
            if self.max_instances > 0:
                input_data = json.load(input_file)[:self.max_instances]
            else:
                input_data = json.load(input_file)

        examples = []
        for entry in input_data:
            examples.append(retriever_utils.read_mathqa_entry(self.df, entry, tokenizer))

        if self.mode == "train":
            kwargs = {"examples": examples,
            "tokenizer": tokenizer,
            "option": "rand",
            "is_training": True,
            "max_seq_length": 512,
            }
        else:
            kwargs = {"examples": examples,
            "tokenizer": tokenizer,
            "option": "rand",
            "is_training": False,
            "max_seq_length": 512,
            }

        features = convert_table_examples_to_features(**kwargs)#get just the tokenized output of the table
        #data_pos, neg_sent, irrelevant_neg_table, relevant_neg_table = features[0], features[1], features[2], features[3]
        pos_table, neg_table = features[0], features[1]
        
        if self.mode == "train":
            random.shuffle(pos_table)
            random.shuffle(neg_table)
            data = pos_table + neg_table[:min(len(neg_table),len(pos_table))]
        else:
            data = pos_table + neg_table
        print(self.mode, len(data))
        return data

    def __getitem__(self, idx: int):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def truncate(self, max_instances):
        truncated_instances = self.instances[max_instances:]
        self.instances = self.instances[:max_instances]
        return truncated_instances

    def extend(self, instances):
        self.instances.extend(instances)

class FinbertRetrieverDataset(Dataset):
    def __init__(
        self, 
        transformer_model_name: str,
        file_path: str,
        max_instances: int,
        mode: str = "train", 
        **kwargs):
        super().__init__(**kwargs)

        assert mode in ["train", "test", "valid"]

        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

        self.max_instances = max_instances
        self.mode = mode
        self.instances = self.read(file_path, self.tokenizer)


    def read(self, input_path: str, tokenizer) -> Iterable[Dict[str, Any]]:
        with open(input_path) as input_file:
            if self.max_instances > 0:
                input_data = json.load(input_file)[:self.max_instances]
            else:
                input_data = json.load(input_file)

        examples = []
        for entry in input_data:
            examples.append(retriever_utils.read_text_mathqa_entry(entry, tokenizer))

        if self.mode == "train":
            kwargs = {"examples": examples,
            "tokenizer": tokenizer,
            "option": "rand",
            "is_training": True,
            "max_seq_length": 512,
            }
        else:
            kwargs = {"examples": examples,
            "tokenizer": tokenizer,
            "option": "rand",
            "is_training": False,
            "max_seq_length": 512,
            }

        features = convert_text_examples_to_features(**kwargs)
        data_pos, neg_sent = features[0], features[1]

        
        if self.mode == "train":
            random.shuffle(data_pos)
            random.shuffle(neg_sent)
            data = data_pos + neg_sent[:min(len(neg_sent),len(data_pos))]
        else:
            data = data_pos + neg_sent
        print(self.mode, len(data))
        return data

    def __getitem__(self, idx: int):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def truncate(self, max_instances):
        truncated_instances = self.instances[max_instances:]
        self.instances = self.instances[:max_instances]
        return truncated_instances

    def extend(self, instances):
        self.instances.extend(instances)

def customized_collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    result_text_dict = {}
    table_examples = examples[0]
    text_examples = examples[1]
    for k in text_examples[0].keys():
        try:
            result_text_dict[k] = right_pad_sequences([torch.tensor(ex[k]) for ex in text_examples], 
                                    batch_first=True, padding_value=0)
        except:
            result_text_dict[k] = [ex[k] for ex in text_examples]
    result_table_dict = {}
    for k in table_examples[0].keys():
        try:
            result_table_dict[k] = right_pad_sequences([torch.tensor(ex[k]) for ex in table_examples],
                                    batch_first=True, padding_value=0)
        except:
            result_table_dict[k] = [ex[k] for ex in table_examples]
    return (result_table_dict, result_text_dict)


class RetrieverDataModule(LightningDataModule):
    def __init__(self, 
                transformer_model_name: str,
                batch_size: int = 1, 
                val_batch_size: int = 1,
                train_file_path: str = None,
                num_workers: int = 8,
                val_file_path: str = None,
                train_inventory_file_path: str = None,
                dev_inventory_file_path: str = None,
                #test_inventory_file_path: str = None,
                train_max_instances: int = sys.maxsize,
                val_max_instances: int = sys.maxsize):
        super().__init__()
        self.transformer_model_name = transformer_model_name

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.num_workers = num_workers
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path

        self.train_inventory_file_path=train_inventory_file_path
        self.dev_inventory_file_path=dev_inventory_file_path
        #self.test_inventory_file_path=test_inventory_file_path

        self.train_max_instances = train_max_instances
        self.val_max_instances = val_max_instances
        
        self.train_data = None
        self.val_data = None

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        assert stage in ["fit", "validate", "test"]

        tabert_train_data = TabertRetrieverDataset(transformer_model_name = "google/tapas-base-finetuned-wtq",
                                file_path=self.train_file_path,
                                inventory_file_path = self.train_inventory_file_path,
                                max_instances=self.train_max_instances, 
                                mode="train")
        finbert_train_data = FinbertRetrieverDataset(transformer_model_name = "ProsusAI/finbert",
                                file_path=self.train_file_path,
                                max_instances=self.train_max_instances, 
                                mode="train")
        self.train_data = (tabert_train_data,finbert_train_data)

        tabert_val_data = TabertRetrieverDataset(transformer_model_name = "google/tapas-base-finetuned-wtq",
                                file_path=self.val_file_path,
                                inventory_file_path = self.dev_inventory_file_path,
                                max_instances=self.val_max_instances, 
                                mode="valid")
        finbert_val_data = FinbertRetrieverDataset(transformer_model_name = "ProsusAI/finbert",
                                file_path=self.val_file_path,
                                max_instances=self.val_max_instances, 
                                mode="valid")
        self.val_data = ( tabert_val_data , finbert_val_data )
        

    def train_dataloader(self):
        if self.train_data is None:
            self.setup(stage="fit")

        dtloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=customized_collate_fn, num_workers = self.num_workers)
        return dtloader

    def val_dataloader(self):
        if self.val_data is None:
            self.setup(stage="validate")

        dtloader = DataLoader(self.val_data, batch_size=self.val_batch_size, shuffle=True, drop_last=False, collate_fn=customized_collate_fn, num_workers = self.num_workers)
        return dtloader

class RetrieverPredictionDataModule(LightningDataModule):
    def __init__(self, 
                transformer_model_name: str,
                batch_size: int = 1, 
                num_workers: int = 8,
                test_file_path: str = None,
                test_max_instances: int = sys.maxsize):
        super().__init__()
        self.transformer_model_name = transformer_model_name

        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.test_file_path = test_file_path

        self.test_max_instances = test_max_instances
        
        self.test_data = None

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        assert stage in ["test", "predict"]
        
        tabert_test_data = TabertRetrieverDataset(transformer_model_name = "google/tapas-base-finetuned-wtq",
                                file_path=self.test_file_path,
                                max_instances=self.test_max_instances, 
                                mode="test")
        finbert_test_data = FinbertRetrieverDataset(transformer_model_name = "ProsusAI/finbert",
                                file_path=self.test_file_path,
                                max_instances=self.test_max_instances, 
                                mode="test")
                                
        self.test_data = (tabert_test_data,finbert_test_data)

    def test_dataloader(self):
        if self.test_data is None:
            self.setup(stage="test")
            
        dtloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=customized_collate_fn, num_workers = self.num_workers)
        return dtloader
    
    def predict_dataloader(self):
        if self.test_data is None:
            self.setup(stage="predict")
            
        dtloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=customized_collate_fn, num_workers = self.num_workers)
        return dtloader
