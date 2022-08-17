import re
import os
import sys
import json

import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from modeling import BertModel, BertConfig
from tokenization import FullTokenizer, convert_to_unicode
from extract_features import InputExample, convert_examples_to_features

def read_examples(str_list):
    """Read a list of `InputExample`s from a list of strings."""
    unique_id = 0
    for s in str_list:
        line = convert_to_unicode(s)
        if not line:
            continue
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
        unique_id += 1

# Convert theses features to np.arrays to use with tf.Keras.
def features_to_arrays(features):

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.input_type_ids)

    return (np.array(all_input_ids, dtype='int32'), 
            np.array(all_input_mask, dtype='int32'), 
            np.array(all_segment_ids, dtype='int32'))


# built all togehter 
def build_preprocessor(voc_path, seq_len, lower=True):
  tokenizer = FullTokenizer(vocab_file=voc_path, do_lower_case=lower)
  
  def strings_to_arrays(sents):
  
      sents = np.atleast_1d(sents).reshape((-1,))

      examples = []
      for example in read_examples(sents):
          examples.append(example)

      features = convert_examples_to_features(examples, seq_len, tokenizer)
      arrays = features_to_arrays(features)
      return arrays
  
  return strings_to_arrays

