import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from io_pipeline import *

if __name__=='__main__':
    train_filename_list='train_train.tfrecords'
    iterator,train_iter_init_op,test_iter_init_op=parse_tfrecords_file(
                                            train_filename_list,
                                            train_filename_list,
                                            mini_batch_size=1,
                                            shuffle_buffer_size=1)

    next_element=iterator.get_next()
    with tf.Session() as sess:
        sess.run(train_iter_init_op)
        img_list=sess.run(next_element)

        print (img_list[4])
        plt.imshow(np.squeeze(img_list[0]),cmap='gray')
        plt.show()
        plt.imshow(np.squeeze(img_list[1]),cmap='gray')
        plt.show()
        plt.imshow(np.squeeze(img_list[2]),cmap='gray')
        plt.show()
        plt.imshow(np.squeeze(img_list[3]),cmap='gray')
        plt.show()
