import tensorflow as tf

from datetime import *
from glob import glob
from os import path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--meta-path", help="Meta Path", default=[p for p in glob(path.abspath(path.join('.','*.meta')))][-1])
parser.add_argument("-o", "--output-graph-name", help="Output Graph Name", default="output_graph_{}.pb".format(str(datetime.now())))
parser.add_argument("-n", "--output-node-names", help="Output Node Names", action="append", default="Openpose/concat_stage7")
args = parser.parse_args()

meta_path = path.abspath(args.meta_path)
output_node_names = args.output_node_names if isinstance(args.output_node_names, list) else [args.output_node_names]
output_graph_name = args.output_graph_name

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,meta_path[:-5])

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open(output_graph_name, 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
