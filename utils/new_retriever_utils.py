import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import random
import enum
import six
import copy
from six.moves import map
from six.moves import range
from six.moves import zip
from tqdm import tqdm
from utils.utils import *
import pandas as pd

_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)

class MathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "filename_id question paragraphs \
            pos_sent_ids"
        )):
    def convert_single_example(self, *args, **kwargs):
        return convert_single_text_mathqa_example(self, *args, **kwargs)

class MathTableQAExample( 
        collections.namedtuple(
            "MathTableQAExample",
            "filename_id question tables answer_coords \
            answer_text float_answer tokenized_table pos_table_ids"
        )):
    def convert_single_table_example(self, *args, **kwargs):
        return convert_single_table_mathqa_example(self, *args, **kwargs)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 filename_id,
                 retrieve_ind,
                 tokens,
                 input_ids,
                 segment_ids,
                 input_mask,
                 label):

        self.filename_id = filename_id
        self.retrieve_ind = retrieve_ind
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label

def text_tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.

    Args:
      tokenizer: a tokenizer from bert.tokenization.FullTokenizer
      text: text to tokenize
      apply_basic_tokenization: If True, apply the basic tokenization. If False,
        apply the full tokenization (basic + wordpiece).

    Returns:
      tokenized text.

    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """

    _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize

    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.get_vocab():
                tokens.append(token)
            else:
                tokens.append(tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))
    return tokens

def remove_space(text_in):
    res = []

    for tmp in text_in.split(" "):
        if tmp != "":
            res.append(tmp)

    return " ".join(res)

def wrap_single_text_pair(tokenizer, question, context, label, max_seq_length,
                     cls_token, sep_token):
    '''
    single pair of question, context, label feature
    '''

    question_tokens = text_tokenize(tokenizer, question)
    this_gold_tokens = text_tokenize(tokenizer, context)

    tokens = [cls_token] + question_tokens + [sep_token]
    segment_ids = [0] * len(tokens)

    tokens += this_gold_tokens
    segment_ids.extend([0] * len(this_gold_tokens))

    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length-1]
        tokens += [sep_token]
        segment_ids = segment_ids[:max_seq_length]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    this_input_feature = {
        "context": context,
        "tokens": tokens,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "label": label
    }

    return this_input_feature


def convert_single_text_mathqa_example(example, option, is_training, tokenizer, max_seq_length,
                                  cls_token, sep_token):
    """Converts a single MathQAExample into Multiple Retriever Features."""
    """ option: tf idf or all"""
    """train: 1:3 pos neg. Test: all"""

    pos_features, neg_sent_features= [], []

    question = example.question

    # positive examples
    # tables = example.tables
    paragraphs = example.paragraphs
    pos_text_ids = example.pos_sent_ids
    
    for sent_idx, sent in enumerate(paragraphs):
        if sent_idx in pos_text_ids:
            this_input_feature = wrap_single_text_pair(
                tokenizer, example.question, sent, 1, max_seq_length,
                cls_token, sep_token)
        else:
            this_input_feature = wrap_single_text_pair(
                tokenizer, example.question, sent, 0, max_seq_length,
                cls_token, sep_token)
        this_input_feature["ind"] = sent_idx
        this_input_feature["filename_id"] = example.filename_id
        
        if sent_idx in pos_text_ids:
            pos_features.append(this_input_feature)
        else:
            neg_sent_features.append(this_input_feature)
        
    return pos_features, neg_sent_features

def read_text_examples(input_path, tokenizer, op_list, const_list, log_file):
    """Read a json file into a list of examples."""

    write_log(log_file, "Reading " + input_path)
    with open(input_path) as input_file:
        input_data = json.load(input_file)

    examples = []
    for entry in input_data:
        examples.append(read_text_mathqa_entry(entry, tokenizer))

    return input_data, examples, op_list, const_list

def read_text_mathqa_entry(entry, tokenizer):

    question = entry["qa"]["question"]
    
    paragraphs = entry["paragraphs"]
    # tables = entry["tables"]
    
    if 'text_evidence' in entry["qa"]:
        pos_sent_ids = entry["qa"]['text_evidence']
    else: # test set
        pos_sent_ids = []
    filename_id = entry["uid"]

    return MathQAExample(
        filename_id=filename_id,
        question=question,
        paragraphs=paragraphs,
        # tables=tables,
        pos_sent_ids=pos_sent_ids,
    )

def convert_text_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 option,
                                 is_training,
                                 ):
    """Converts a list of DropExamples into InputFeatures."""
    res, res_neg_sent  = [], []
    for (example_index, example) in tqdm(enumerate(examples)):
        pos_features, neg_sent_features = example.convert_single_example(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            option=option,
            is_training=is_training,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token)

        res.extend(pos_features)
        res_neg_sent.extend(neg_sent_features)
        
    return res, res_neg_sent

### ------------------------------------------------------------
def wrap_table_single_pair(tokenizer, question, table, label, answer_coords, answer_text, float_answer, max_seq_length,
                     cls_token, sep_token):
    '''
    single pair of question, context, label feature
    '''
    path="./dataset/"
    df0 = pd.read_csv(path+table).astype(str)
    #print("queries",question)
    #print("coords",answer_coords)
    #print("ans text",answer_text)
    tokens = tokenizer(
        table=df0,
        queries=list(question),
        #answer_coordinates=answer_coords,
        #answer_text=answer_text,
        padding="max_length",
        return_tensors="pt",
        #float_answer = float_answer
        )
    #print("queries",len(queries))
    #print("coords",len(answer_coordinates)
    #print("ans text",answer_text)
    #tokens = [cls_token] + tokens + [sep_token]
    segment_ids = [0] * len(tokens)

    #segment_ids.extend([0] * len(tokens))

    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length-1]
        tokens += [sep_token]
        segment_ids = segment_ids[:max_seq_length]

    #print("tokens",len(tokens))
    #print(type(tokens))
    input_ids = tokenizer.convert_tokens_to_ids(tokens) #check if this works here
    #print("input_ids",len(input_ids))
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    this_input_feature = {
        "context": table,
        "tokens": tokens,
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "label": label
    }
    return this_input_feature

def convert_single_table_mathqa_example(example, option, is_training, tokenizer, max_seq_length,
                                  cls_token, sep_token):
    """Converts a single MathQAExample into Multiple Retriever Features."""
    """ option: tf idf or all"""
    """train: 1:3 pos neg. Test: all"""

    pos_features, neg_features= [], []

    question = example.question

    # positive examples
    tables = example.tables
    #paragraphs = example.paragraphs
    pos_table_ids = example.pos_table_ids
    relevant_table_ids = set([i.split("-")[0] for i in pos_table_ids])


    for table_idx,table in enumerate(tables):
        if table_idx in relevant_table_ids:
          this_input_feature = wrap_table_single_pair(
                tokenizer, example.question, table, 1, example.answer_coords[table_idx],example.answer_text[table_idx], example.float_answer[table_idx], max_seq_length,
                cls_token, sep_token)
        else:
          this_input_feature = wrap_table_single_pair(
                tokenizer, example.question, table, 0, example.answer_coords[table_idx], example.answer_text[table_idx], example.float_answer[table_idx], max_seq_length,
                cls_token, sep_token)
        this_input_feature["ind"] = table_idx
        this_input_feature["filename_id"] = example.filename_id
        if table_idx in relevant_table_ids:
            pos_features.append(this_input_feature)
        else:
            neg_features.append(this_input_feature)
        
    return pos_features, neg_features

def read_mathqa_entry(df, entry, tokenizer):
  question = entry["qa"]["question"]
  #paragraphs = entry["paragraphs"]
  #tables = entry["tables"]
  
  if 'table_evidence' in entry["qa"]:
      pos_ids = entry["qa"]['table_evidence']
  else: # test set
      pos_ids = []
  filename_id = entry["uid"]
  current_data = df.loc[df[0] == filename_id]
  question=list(current_data[1])[0]
  answer_coords = current_data[3]
  answer_text = current_data[4]
  floats=[]
  try: 
    for i in current_data[5]:
      floats.append(float(i))
  except:
    for i in current_data[5]:
      floats.append(np.nan)
  csv = current_data[2]
  ans=[]
  for j in answer_coords:
      if j == "[]":
        ans.append([])
      else:
        l=j[2:-2].split('), (')
        l=[i.split(",") for i in l]
        l=[(int(i[0]),int(i[1])) for i in l]
        ans.append(l)
  queries=list(question)
  #print("read matqa answer coords",ans)
  return MathTableQAExample(
        filename_id=filename_id,
        question=question,
        tables=list(csv),
        answer_coords = ans,
        answer_text = list(answer_text),
        float_answer=list(floats),
        tokenized_table =[],
        pos_table_ids=pos_ids,
    )

def convert_table_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 option,
                                 is_training,
                                 ):
    """Converts a list of DropExamples into InputFeatures."""
    res_table, res_table_neg =[] ,[]
    for (example_index, example) in tqdm(enumerate(examples)):
        pos_table_features, neg_table_features = example.convert_single_table_example(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            option=option,
            is_training=is_training,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token)
        
        res_table.extend(pos_table_features)
        res_table_neg.extend(neg_table_features)
    
    return res_table, res_table_neg
