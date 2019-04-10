from __future__ import absolute_import, division, print_function

import tensorflow as tf

import numpy as np

model_filename = "./model/model-const.pb"

with tf.gfile.GFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # tensorflow adds "import/" prefix to all tensors when imports graph definition, ex: "import/input:0"
    # so we explicitly tell tensorflow to use empty string -> name=""
    tf.import_graph_def(graph_def, name="")

product_embeddings = tf.get_default_graph().get_tensor_by_name('product_embeddings:0')
user_embeddings = tf.get_default_graph().get_tensor_by_name('user_embeddings:0')

with tf.Session() as sess:
    print(sess.run(user_embeddings, feed_dict={product_embeddings: np.random.uniform(size=product_embeddings.shape)}))
