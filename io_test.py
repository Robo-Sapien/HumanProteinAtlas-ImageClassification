import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from io_pipeline import *

if __name__=='__main__':
    train_filename_list='*.tfrecords'
    iterator,train_iter_init_op,test_iter_init_op=parse_tfrecords_file(
                                            train_filename_list,
                                            train_filename_list,
                                            mini_batch_size=1,
                                            shuffle_buffer_size=1)

    next_element=iterator.get_next()
    with tf.Session() as sess:
        sess.run(train_iter_init_op)
        img_list=sess.run(next_element)

        print img_list[0].shape
