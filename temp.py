from UCFdata import DataSet 
import numpy as np 
import tensorflow as tf 
from keras.utils import np_utils
import cv2 as cv 

data = DataSet()

train_data = []
train_label = []
all_labels = range(101)

def read_video_data(cap):
    data = []
    for _ in range(100):
        ret, frame = cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            data.append(frame)
        else:
            data.append(np.zeros(shape=(240, 320)))
    return data

all_labels = range(101)

def load_data_batch(data, all_labels, begin, batch_size):
    train_data = []
    train_label = []
    for i in range(begin, begin + batch_size):
        filename = 'data/' + data.data[i][0] + '/' + data.data[i][1] + '/' + data.data[i][2] + '.avi'
        cap = cv.VideoCapture(filename)
        if data.data[i][0] == 'train':
            train_data.append(read_video_data(cap))
            train_label.append(all_labels[data.classes.index(data.data[i][1])])

    train_data = np.array(train_data)
    train_data = train_data.reshape([train_data.shape[0], 100, 240, 320, 1])
    train_label = np.array(train_label)
    train_label = np_utils.to_categorical(train_label, num_classes=101)

    train_data_norm = train_data / 255.
    return train_label, train_data_norm

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')

def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[100, 240, 320, 1], name='x')

with tf.name_scope('C1_Conv'):
    W1 = weight([5, 5, 1, 16])
    b1 = bias([16])
    Conv1 = conv2d(x, W1) + b1
    C1_Conv = tf.nn.relu(Conv1)

with tf.name_scope('C1_Pool'):
    C1_Pool = max_pool_2x2(C1_Conv)

with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C1_Pool, [-1, 614400])

with tf.name_scope('Hidden_Layer'):
    W2 = weight([614400, 101])
    b2 = bias([101])
    D_Hidden = tf.nn.relu(tf.matmul(D_Flat, W2) + b2)

with tf.name_scope('CNN_Output_Layer'):
    W3 = weight([101, 1])
    b3 = bias([1])
    CNN_Output_Layer = tf.nn.softmax(tf.matmul(D_Hidden, W3) + b3)

with tf.name_scope('Output_Layer'):
    CNN_Output = tf.placeholder('float', shape=[None, 100], name='y_label')
    W4 = weight([100, 101])
    b4 = bias([101])
    y_predict = tf.nn.softmax(tf.matmul(CNN_Output, W4) + b4)

with tf.name_scope('optimizer'):
    y_label = tf.placeholder("float", shape=[None, 101], name='y_label')
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss_function)
    
with tf.name_scope('evaluate_model'):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        image_y, image_x = load_data_batch(data, all_labels, i * 50, 50)
        output = sess.run(CNN_Output, feed_dict={x: image_x[i]})