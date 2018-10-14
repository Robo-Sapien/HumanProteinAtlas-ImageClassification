import tensorflow as tf
import os
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

def _load_the_label(directory_name,filename):
    '''
    This will read the csv and load it as dictionary for giving
    the labels for the image files.Since the file size in not large
    lets hash it for now.
    Better use pandas.
    '''
    pd.load_csv()




def _byte_feature(value):
    '''
    This function will convert the byte string to a tensorflow
    example feature.
    '''
    return tf.Train.Feature(byte_list=tf.train.ByteList(value=[value]))


def read_cell_image(directory_name,mode='train'):
    '''
    This function will read the cell image in all the four filters and
    reaturn as a list of numpy array
    '''
    #Getting the list of the filename in the directory
    filelist=os.listdir(directory_name)
    #Initializing the dict to hold the filename which have been read(all)
    done_dict={}

    #Initializing the tf records file writer
    tfrecord_filename=directory_name+'_'+mode+'.tfrecords'
    compression_options=tf.python_io.TFRecordOptions(
                            tf.python_io.TFRecordCompressionType.ZLIB)

    with tf.python_io.TFRecordWriter(tfrecord_filename,
                    options=compression_options) as record_writer:
        #Now iterating the list of files and getting the files
        filter_tags=['red','blue','yellow','green']
        for img_name in filelist:
            img_root=img_name.split('_')[0]
            #Skipping this image if already saved
            if img_root in done_dict:
                continue

            #Raising the flag for this image
            done_dict[img_root]=1
            #reading the image and converting it to tf records
            images=[]
            for filter in filter_tags:
                img_name_final=directory_name+'/'+img_root+'_'+filter+'.png'
                img=scipy.misc.imread(img_name_final)
                plt.imshow(img,cmap='gray')
                plt.show()
                images.append(img)

            #Now writing the file



if __name__=='__main__':
    directory_name='sample_images'
    read_cell_image(directory_name)
