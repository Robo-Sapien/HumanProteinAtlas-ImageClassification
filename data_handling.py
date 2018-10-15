import tensorflow as tf
import os
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

def _load_the_label(labels_path):
    '''
    This will read the csv and load it as dictionary for giving
    the labels for the image files.Since the file size in not large
    lets hash it for now.
    Better use pandas.
    '''
    df=pd.read_csv(labels_path)
    df.set_index('Id',inplace=True)
    # print(df.head())
    # print(df.dtypes)
    # print(type(df.iloc[0]['Target']))
    # print(type(df.iloc[0]['Id']))
    #print(df.index.tolist())
    return df

def _encode_many_hot(label_string,shape=(28,)):
    '''
    This function will conver the list of targets into one hot encoding
    '''
    #Converting the string to number
    label_list=map(lambda x: int(x),label_string.split(' '))
    #label_list=[label_string]
    #Creating the many hot vector initialized with zeros
    many_hot=np.zeros(shape,dtype=np.float32)
    #Filling the many hot encoding
    for label in label_list:
        many_hot[label]=1.0

    return many_hot

def _bytes_feature(value):
    '''
    This function will convert the byte string to a tensorflow
    example feature.
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_cell_image(labels_df,directory_name,mode='train'):
    '''
    This function will read the cell image in all the four filters and
    reaturn as a list of numpy array
    '''
    #Getting the list of the filename in the directory
    filelist=os.listdir(directory_name)

    #Initializing the tf records file writer
    tfrecord_filename=directory_name+'_'+mode+'.tfrecords'
    compression_options=tf.python_io.TFRecordOptions(
                            tf.python_io.TFRecordCompressionType.ZLIB)

    with tf.python_io.TFRecordWriter(tfrecord_filename,
                    options=compression_options) as record_writer:
        #Now iterating the list of files and getting the files
        filter_tags=['red','blue','yellow','green']
        for img_name in labels_df.index.tolist():
            print ("Creating examples for: ",img_name)
            #reading the image and converting it to tf records
            images={}
            flag=0
            for filter in filter_tags:
                img_name_final=directory_name+'/'+img_name+'_'+filter+'.png'
                try:
                    img=scipy.misc.imread(img_name_final)
                except:
                    flag=1
                    break
                plt.imshow(img,cmap='gray')
                plt.show()
                images[filter]=img

            #Creation of labels for the image
            if mode=='train' and flag==0:
                #Get the labels string
                label_string=labels_df.loc[img_name,'Predicted']
                many_hot=_encode_many_hot(label_string)

                #Now we will create an example feature to be saved in tfrecords
                example=tf.train.Example(features=tf.train.Features(
                    feature={
                        'red':_bytes_feature(images['red'].tobytes()),
                        'blue':_bytes_feature(images['blue'].tobytes()),
                        'yellow':_bytes_feature(images['yellow'].tobytes()),
                        'green':_bytes_feature(images['green'].tobytes()),
                        'label':_bytes_feature(many_hot.tobytes())
                    }
                ))
                #Writing the example to the tfrecords
                record_writer.write(example.SerializeToString())
            elif mode=='test' and flag==0:
                example=tf.train.Example(features=tf.train.Features(
                    feature={
                        'red':_bytes_feature(images['red'].tobytes()),
                        'blue':_bytes_feature(images['blue'].tobytes()),
                        'yellow':_bytes_feature(images['yellow'].tobytes()),
                        'green':_bytes_feature(images['green'].tobytes())
                    }
                ))
                #Writing the example to the tfrecords
                record_writer.write(example.SerializeToString())
            # else:
            #     assert False,"Please enter a valid mode: train/test"


if __name__=='__main__':
    directory_name='sample_images'
    labels_path='dataset/sample_submission.csv'

    df=_load_the_label(labels_path)
    read_cell_image(df,directory_name,mode='test')
