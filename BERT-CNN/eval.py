import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import argparse
from sklearn.model_selection import train_test_split

sys.path.insert(0, "bert_experimental")

from bert_experimental.finetuning.text_preprocessing import build_preprocessor
from bert_experimental.finetuning.graph_ops import load_graph



parser=argparse.ArgumentParser(description='inference of the model')
parser.add_argument('--testset',  default='test.tsv', help='test file', type=str,required=True) 
parser.add_argument('--model', default='pre-trained model', help='', type=str, required=True) 
args = parser.parse_args()



df = pd.read_csv(args.testset, sep='\t')
 

texts = []
delimiter = " ||| "

for vis, cap  in zip(df.visual.tolist(), df.caption.tolist()):
  texts.append(delimiter.join((str(vis), str(cap))))


texts = np.array(texts)

trX, tsX = train_test_split(texts, shuffle=False, test_size=0.01)


restored_graph = load_graph(args.model)

graph_ops = restored_graph.get_operations()
input_op, output_op = graph_ops[0].name, graph_ops[-1].name
print(input_op, output_op)

x = restored_graph.get_tensor_by_name(input_op + ':0')
y = restored_graph.get_tensor_by_name(output_op + ':0')

# also need a flag here

#
preprocessor = build_preprocessor("uncased_L-12_H-768_A-12/vocab.txt", 64)
py_func = tf.numpy_function(preprocessor, [x], [tf.int32, tf.int32, tf.int32], name='preprocessor')

py_func = tf.numpy_function(preprocessor, [x], [tf.int32, tf.int32, tf.int32])

##predictions

sess = tf.Session(graph=restored_graph)

print(trX[:2])

y = tf.print(y, summarize=-1)
#x = tf.print(x, summarize=-1)
y_out = sess.run(y, feed_dict={
        x: trX[:2].reshape((-1,1))

    })

print(y_out)

