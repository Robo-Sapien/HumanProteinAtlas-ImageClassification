import tensorflow as tf
import multiprocessing
ncpu=multiprocessing.cpu_count()

def _binary_parse_function_example(serialized_example_protocol):
    '''
    This function will read the tf records and reconvert them to the
    appropriate format form the raw binary form
    '''
    #Parsing the binary feature
    features={
        'red':      tf.FixedLenFeature((),tf.string),
        'blue':     tf.FixedLenFeature((),tf.string),
        'yellow':   tf.FixedLenFeature((),tf.string),
        'green':    tf.FixedLenFeature((),tf.string),
        'label':    tf.FixedLenFeature((),tf.string)
    }
    parsed_feature=tf.parse_single_example(serialized_example_protocol,
                                            features)

    #Now setting the appropriate size of the images
    height=512
    width=512
    #Decoding the images from binary
    red=tf.decode_raw(parsed_feature['red'],tf.uint8)
    red.set_shape([height*width])
    red=tf.reshape(red,[height,width])

    blue=tf.decode_raw(parsed_feature['blue'],tf.uint8)
    blue.set_shape([height*width])
    blue=tf.reshape(blue,[height,width])

    yellow=tf.decode_raw(parsed_feature['yellow'],tf.uint8)
    yellow.set_shape([height*width])
    yellow=tf.reshape(yellow,[height,width])

    green=tf.decode_raw(parsed_feature['green'],tf.uint8)
    green.set_shape([height*width])
    green=tf.reshape(green,[height,width])

    #Decoding the labels
    label_len=28
    label=tf.decode_raw(parsed_feature['label'],tf.float32)
    label.set_shape([label_len])

    #Now we can do any transformation on the image here
    #Also, merge the chunks as required or do it in calculation phase
    #based on the load on the cpu

    return [red,blue,yellow,green,label]

def parse_tfrecords_file(train_filename_pattern,test_filename_pattern,
                        mini_batch_size,shuffle_buffer_size):
    '''
    This function will create the iterators for trainng the model
    from the datasets.
    '''
    #Making the tfrecords as datasets
    comp_type='ZLIB'
    #Reading the training datasets
    train_dataset=tf.data.TFRecordDataset(train_filename_pattern,
                        compression_type=comp_type,num_parallel_reads=int(ncpu/2))
    #Reading the testing datasets
    test_dataset=tf.data.TFRecordDataset(test_filename_pattern,
                        compression_type=comp_type,num_parallel_reads=int(ncpu/2))

    #Shuffling the file instead of the each examples
    train_dataset=train_dataset.shuffle(buffer_size=shuffle_buffer_size)
    test_dataset=test_dataset.shuffle(buffer_size=shuffle_buffer_size)


    #Applyting the transforamtion to convert binary data to examples
    train_dataset=train_dataset.apply(
            tf.contrib.data.map_and_batch(_binary_parse_function_example,
                                            mini_batch_size,
                                            num_parallel_batches=int(ncpu/2),
                                            drop_remainder=False))
    test_dataset=test_dataset.apply(
            tf.contrib.data.map_and_batch(_binary_parse_function_example,
                                            mini_batch_size,
                                            num_parallel_batches=int(ncpu/2),
                                            drop_remainder=False))

    #Prefetching the dataset
    train_dataset=train_dataset.prefetch(1)
    test_dataset=test_dataset.prefetch(1)

    #Now making the initializable iterator
    iterator=tf.data.Iterator.from_structure(
                                        train_dataset.output_types,
                                        train_dataset.output_shapes)
    train_iter_init_op=iterator.make_initializer(train_dataset)
    test_iter_init_op=iterator.make_initializer(test_dataset)

    return iterator,train_iter_init_op,test_iter_init_op
