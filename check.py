from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow
import os
import tensorflow as tf
from ssdvgg import *
# checkpoint_path = os.path.join("/home/nyma/PycharmProjects/JointSSD/test", "final.ckpt")

# List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
# print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors='True')



checkpoint_path = os.path.join("/home/nyma/PycharmProjects/JointSSD/SSDcheckpoint", "final.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
#
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key).shape)

# def initialize_uninitialized_variables(sess):
#     """
#     Only initialize the weights that have not yet been initialized by other
#     means, such as importing a metagraph and a checkpoint. It's useful when
#     extending an existing model.
#     """
#     uninit_vars    = []
#     uninit_tensors = []
#     for var in tf.global_variables():
#         uninit_vars.append(var)
#         uninit_tensors.append(tf.is_variable_initialized(var))
#     uninit_bools = sess.run(uninit_tensors)
#     uninit = zip(uninit_bools, uninit_vars)
#     uninit = [var for init, var in uninit if not init]
#     sess.run(tf.variables_initializer(uninit))
# init_op = tf.initializers.global_variables()
# with tf.Session() as sess:
#     # last_check = tf.train.latest_checkpoint('/home/nyma/PycharmProjects/JointSSD/SSDcheckpoint')
#     initialize_uninitialized_variables(sess)
#     loader2 = tf.train.Saver()
#     loader2.restore(sess, "/home/nyma/PycharmProjects/JointSSD/test/final.ckpt")
#     sess.run(init_op)
    # saver = tf.train.import_meta_graph('/home/nyma/PycharmProjects/JointSSD/SSDcheckpoint/final.ckpt.meta')
    # saver.restore(sess,'/home/nyma/PycharmProjects/JointSSD/SSDcheckpoint/final.ckpt')
    ######
    # Model_variables = tf.GraphKeys.MODEL_VARIABLES
    # Global_Variables = tf.GraphKeys.GLOBAL_VARIABLES
    # ######
    # all_vars = tf.get_collection(Model_variables)
    # print (all_vars)
    # for i in all_vars:
    #     print (str(i) + '  -->  '+ str(i.eval()))