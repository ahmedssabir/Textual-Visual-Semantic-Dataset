# -*- coding: utf-8 -*-
##!/usr/bin/env python3
#!/bin/env python
import sys
import argparse
import re
import os
import sys
import json

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from BertLayer import BertLayer
from BertLayer import build_preprocessor
from freeze_keras_model import freeze_keras_model

from data_pre import * 
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split


if not 'bert_repo' in sys.path:
    sys.path.insert(0, 'bert_repo')

from modeling import BertModel, BertConfig
from tokenization import FullTokenizer, convert_to_unicode
from extract_features import InputExample, convert_examples_to_features


# get TF logger 
log = logging.getLogger('tensorflow')
log.handlers = []


parser=argparse.ArgumentParser()
parser.add_argument('--train',  default='/home/asabir/BERT_layers-git/data/train.tsv', help='beam serach', type=str,required=False)  
parser.add_argument('--num_bert_layer', default='12', help='truned layers', type=int,required=False)  
parser.add_argument('--batch_size', default='128', help='truned layers', type=int,required=False) 
parser.add_argument('--epochs', default='5', help='', type=int,required=False) 
parser.add_argument('--seq_len', default='64', help='', type=int,required=False) 
parser.add_argument('--CNN_kernel_size', default='3', help='', type=int,required=False) 
parser.add_argument('--CNN_filters', default='32', help='', type=int,required=False)
args = parser.parse_args()


# Downlaod the pre-trained model

#!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
#!unzip uncased_L-12_H-768_A-12.zip


# tf.Module
def build_module_fn(config_path, vocab_path, do_lower_case=True):

    def bert_module_fn(is_training):
        """Spec function for a token embedding module."""

        input_ids = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_ids")
        input_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="input_mask")
        token_type = tf.placeholder(shape=[None, None], dtype=tf.int32, name="segment_ids")

        config = BertConfig.from_json_file(config_path)
        model = BertModel(config=config, is_training=is_training,
                          input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type)
          
        seq_output = model.all_encoder_layers[-1]
        pool_output = model.get_pooled_output()

        config_file = tf.constant(value=config_path, dtype=tf.string, name="config_file")
        vocab_file = tf.constant(value=vocab_path, dtype=tf.string, name="vocab_file")
        lower_case = tf.constant(do_lower_case)

        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, config_file)
        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_file)
        
        input_map = {"input_ids": input_ids,
                     "input_mask": input_mask,
                     "segment_ids": token_type}
        
        output_map = {"pooled_output": pool_output,
                      "sequence_output": seq_output}

        output_info_map = {"vocab_file": vocab_file,
                           "do_lower_case": lower_case}
                
        hub.add_signature(name="tokens", inputs=input_map, outputs=output_map)
        hub.add_signature(name="tokenization_info", inputs={}, outputs=output_info_map)

    return bert_module_fn


MODEL_DIR = "/Users/asabir/BERT_layers-main/uncased_L-12_H-768_A-12"
config_path = "/{}/bert_config.json".format(MODEL_DIR)
vocab_path = "/{}/vocab.txt".format(MODEL_DIR)


tags_and_args = []
for is_training in (True, False):
  tags = set()
  if is_training:
    tags.add("train")
  tags_and_args.append((tags, dict(is_training=is_training)))

module_fn = build_module_fn(config_path, vocab_path)
spec = hub.create_module_spec(module_fn, tags_and_args=tags_and_args)
spec.export("bert-module", 
            checkpoint_path="/{}/bert_model.ckpt".format(MODEL_DIR))

class BertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_path, seq_len=64, n_tune_layers=3, 
                 pooling="cls", do_preprocessing=True, verbose=False,
                 tune_embeddings=False, trainable=True, **kwargs):

        self.trainable = trainable
        self.n_tune_layers = n_tune_layers
        self.tune_embeddings = tune_embeddings
        self.do_preprocessing = do_preprocessing

        self.verbose = verbose
        self.seq_len = seq_len
        self.pooling = pooling
        self.bert_path = bert_path

        self.var_per_encoder = 16
        if self.pooling not in ["cls", "mean", None]:
            raise NameError(
                f"Undefined pooling type (must be either 'cls', 'mean', or None, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.bert = hub.Module(self.build_abspath(self.bert_path), 
                               trainable=self.trainable, name=f"{self.name}_module")

        trainable_layers = []
        if self.tune_embeddings:
            trainable_layers.append("embeddings")

        if self.pooling == "cls":
            trainable_layers.append("pooler")

        if self.n_tune_layers > 0:
            encoder_var_names = [var.name for var in self.bert.variables if 'encoder' in var.name]
            n_encoder_layers = int(len(encoder_var_names) / self.var_per_encoder)
            for i in range(self.n_tune_layers):
                trainable_layers.append(f"encoder/layer_{str(n_encoder_layers - 1 - i)}/")
        
        # Add module variables to layer's trainable weights
        for var in self.bert.variables:
            if any([l in var.name for l in trainable_layers]):
                self._trainable_weights.append(var)
            else:
                self._non_trainable_weights.append(var)

        if self.verbose:
            print("*** TRAINABLE VARS *** ")
            for var in self._trainable_weights:
                print(var)

        self.build_preprocessor()
        self.initialize_module()

        super(BertLayer, self).build(input_shape)

    def build_abspath(self, path):
        if path.startswith("https://") or path.startswith("gs://"):
          return path
        else:
          return os.path.abspath(path)

    def build_preprocessor(self):
        sess = tf.keras.backend.get_session()
        tokenization_info = self.bert(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                              tokenization_info["do_lower_case"]])
        self.preprocessor = build_preprocessor(vocab_file, self.seq_len, do_lower_case)

    def initialize_module(self):
        sess = tf.keras.backend.get_session()
        
        vars_initialized = sess.run([tf.is_variable_initialized(var) 
                                     for var in self.bert.variables])

        uninitialized = []
        for var, is_initialized in zip(self.bert.variables, vars_initialized):
            if not is_initialized:
                uninitialized.append(var)

        if len(uninitialized):
            sess.run(tf.variables_initializer(uninitialized))

    def call(self, input):

        if self.do_preprocessing:
          input = tf.numpy_function(self.preprocessor, 
                                    [input], [tf.int32, tf.int32, tf.int32], 
                                    name='preprocessor')
          for feature in input:
            feature.set_shape((None, self.seq_len))
        
        input_ids, input_mask, segment_ids = input
        
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        output = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)
        
        if self.pooling == "cls":
            pooled = output["pooled_output"]
        else:
            result = output["sequence_output"]
            
            input_mask = tf.cast(input_mask, tf.float32)
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            
            if self.pooling == "mean":
              pooled = masked_reduce_mean(result, input_mask)
            else:
              pooled = mul_mask(result, input_mask)

        return pooled

    def get_config(self):
        config_dict = {
            "bert_path": self.bert_path, 
            "seq_len": self.seq_len,
            "pooling": self.pooling,
            "n_tune_layers": self.n_tune_layers,
            "tune_embeddings": self.tune_embeddings,
            "do_preprocessing": self.do_preprocessing,
            "verbose": self.verbose
        }
        super(BertLayer, self).get_config()
        return config_dict


# read the train data 
#df = pd.read_csv("/home/asabir/BERT_layers-git/data/train.tsv", sep='\t')
df = pd.read_csv(args.train, sep='\t')




#labels = df.is_duplicate.values
labels = df.is_related.values

texts = []
delimiter = " ||| "

for vis, cap  in zip(df.visual.tolist(), df.caption.tolist()):
  texts.append(delimiter.join((str(vis), str(cap))))


texts = np.array(texts)

trX, tsX, trY, tsY = train_test_split(texts, labels, shuffle=True, test_size=0.2)


# Buliding the model 

embedding_size = 768

inp = tf.keras.Input(shape=(1,), dtype=tf.string)
# Three  Layers 
#encoder = BertLayer(bert_path="./bert-module/", seq_len=48, tune_embeddings=False,
#                    pooling='cls', n_tune_layers=3, verbose=False)

# All Layers 
encoder = BertLayer(bert_path="./bert-module/", seq_len=args.seq_len, tune_embeddings=False, pooling=None, n_tune_layers=args.num_bert_layer, verbose=False)



cnn_out = tf.keras.layers.Conv1D(args.CNN_filters, args.CNN_kernel_size, padding='VALID', activation=tf.nn.relu)(encoder(inp))
pool = tf.keras.layers.MaxPooling1D(pool_size=2)(cnn_out)
flat = tf.keras.layers.Flatten()(pool)
pred = tf.keras.layers.Dense(1, activation="sigmoid")(flat)


model = tf.keras.models.Model(inputs=[inp], outputs=[pred])

model.summary()

model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, ),
      loss="binary_crossentropy",
      metrics=["accuracy"])

# fit the data 
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

saver = keras.callbacks.ModelCheckpoint("bert_CNN_tuned.hdf5")

model.fit(trX, trY, validation_data=[tsX, tsY], batch_size=args.batch_size, epochs=args.epochs, callbacks=[saver])



#save the model 
model.predict(trX[:10])

import json
json.dump(model.to_json(), open("model.json", "w"))

model = tf.keras.models.model_from_json(json.load(open("model.json")), 
                                        custom_objects={"BertLayer": BertLayer})

model.load_weights("bert_CNN_tuned.hdf5")

model.predict(trX[:10])

# For fast inference and less RAM usesage as post-processing we need to "freezing" the model. 
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

def freeze_keras_model(model, export_path=None, clear_devices=True):
    sess = tf.keras.backend.get_session()
    graph = sess.graph
    
    with graph.as_default():

        input_tensors = model.inputs
        output_tensors = model.outputs
        dtypes = [t.dtype.as_datatype_enum for t in input_tensors]
        input_ops = [t.name.rsplit(":", maxsplit=1)[0] for t in input_tensors]
        output_ops = [t.name.rsplit(":", maxsplit=1)[0] for t in output_tensors]
        
        tmp_g = graph.as_graph_def()
        if clear_devices:
            for node in tmp_g.node:
                node.device = ""
        
        tmp_g = optimize_for_inference(
            tmp_g, input_ops, output_ops, dtypes, False)
        
        tmp_g = convert_variables_to_constants(sess, tmp_g, output_ops)
        
        if export_path is not None:
            with tf.gfile.GFile(export_path, "wb") as f:
                f.write(tmp_g.SerializeToString())
        
        return tmp_g


# freeze and save the model
frozen_graph = freeze_keras_model(model, export_path="frozen_graph.pb")


# inference 
#!git clone https://github.com/gaphex/bert_experimental/

import tensorflow as tf
import numpy as np
import sys

sys.path.insert(0, "bert_experimental")

from bert_experimental.finetuning.text_preprocessing import build_preprocessor
from bert_experimental.finetuning.graph_ops import load_graph


restored_graph = load_graph("frozen_graph.pb")
graph_ops = restored_graph.get_operations()
input_op, output_op = graph_ops[0].name, graph_ops[-1].name
print(input_op, output_op)

x = restored_graph.get_tensor_by_name(input_op + ':0')
y = restored_graph.get_tensor_by_name(output_op + ':0')


preprocessor = build_preprocessor("/Users/asabir/BERT_layers-main/uncased_L-12_H-768_A-12/vocab.txt", 64)
py_func = tf.numpy_function(preprocessor, [x], [tf.int32, tf.int32, tf.int32], name='preprocessor')

py_func = tf.numpy_function(preprocessor, [x], [tf.int32, tf.int32, tf.int32])

# predictions

sess = tf.Session(graph=restored_graph)

trX[:10]

y_out = sess.run(y, feed_dict={
        x: trX[:10].reshape((-1,1))
    })

print(y_out)
