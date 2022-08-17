import tensorflow as tf
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, "bert_experimental")

from bert_experimental.finetuning.text_preprocessing import build_preprocessor
from bert_experimental.finetuning.graph_ops import load_graph


df = pd.read_csv("test.tsv", sep='\t')


texts = []
delimiter = " ||| "

for vis, cap  in zip(df.visual.tolist(), df.caption.tolist()):
  texts.append(delimiter.join((str(vis), str(cap))))


texts = np.array(texts)

trX, tsX = train_test_split(texts, shuffle=False, test_size=0.01)


restored_graph = load_graph("frozen_graph.pb")

graph_ops = restored_graph.get_operations()
input_op, output_op = graph_ops[0].name, graph_ops[-1].name
print(input_op, output_op)

x = restored_graph.get_tensor_by_name(input_op + ':0')
y = restored_graph.get_tensor_by_name(output_op + ':0')

preprocessor = build_preprocessor("/Bert/uncased_L-12_H-768_A-12/vocab.txt", 64)
py_func = tf.numpy_function(preprocessor, [x], [tf.int32, tf.int32, tf.int32], name='preprocessor')

py_func = tf.numpy_function(preprocessor, [x], [tf.int32, tf.int32, tf.int32])

##predictions

sess = tf.Session(graph=restored_graph)

print(trX[:4])

y = tf.print(y, summarize=-1)
#x = tf.print(x, summarize=-1)
y_out = sess.run(y, feed_dict={
        x: trX[:4].reshape((-1,1))
 	#x: trX[:90000].reshape((-1,1))
    })


print(y_out)

