from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util

import numpy as np

embedding_size = 105
nb_products = 50
model_folder = "./model/"

product_embeddings = tf.placeholder(tf.float32, shape=(nb_products, embedding_size), name="product_embeddings")
W = tf.get_variable("W", shape=(embedding_size, embedding_size), initializer=tf.zeros_initializer())
product_embeddings_w = tf.matmul(product_embeddings, W, name="product_embeddings_w")
user_embeddings = tf.reduce_mean(product_embeddings_w, axis=0, name="user_embeddings")

writer = tf.summary.FileWriter('./tf-logs/')
writer.add_graph(tf.get_default_graph())
writer.flush()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(W, np.random.uniform(size=W.shape)))
    user_embeddings_computed =\
        sess.run(user_embeddings, feed_dict={product_embeddings: np.random.uniform(size=product_embeddings.shape)})
    print(user_embeddings_computed)

    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        ["user_embeddings"]
    )
    tf.train.write_graph(output_graph_def, model_folder, "model-const.pbtxt", as_text=True)
    tf.train.write_graph(output_graph_def, model_folder, "model-const.pb", as_text=False)

