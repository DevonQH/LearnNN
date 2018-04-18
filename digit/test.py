from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import numpy as np

#inputs
with tf.name_scope('inputs'):
	x = tf.placeholder(tf.float32, [None, 64])
	y = tf.placeholder(tf.float32, [None, 10])
	keep_prob = tf.placeholder(tf.float32)

#add_layer
def add_layer(in_data, in_size, out_size, n_layer, activation_function):
	layer_name = 'layer%s'%n_layer
	with tf.name_scope(layer_name):
		w = tf.Variable(tf.random_normal([in_size, out_size]))
		b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
		t = tf.add(tf.matmul(in_data, w), b)
		t = tf.nn.dropout(t, keep_prob)
		if activation_function is None:
			output = t
		else:
			output = activation_function(t)
		tf.summary.histogram(layer_name + '/output', output)
		return output

#construct the network
l1 = add_layer(x, 64, 50, n_layer = 1, activation_function = tf.nn.tanh)
pred = add_layer(l1, 50, 10, n_layer = 2, activation_function = tf.nn.softmax)

#load data
digits = load_digits()
x_data = digits.data
y_data = digits.target
y_data = LabelBinarizer().fit_transform(y_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = .3)

#loss
with tf.name_scope('loss'):
	loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = [1]))
	tf.summary.scalar('loss', loss)

#trainer
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.4).minimize(loss)

#init
init = tf.global_variables_initializer()

with tf.Session() as sess:
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter('logs/train', sess.graph)
	test_writer = tf.summary.FileWriter('logs/test', sess.graph)
	sess.run(init)
	for _ in range(500):
		sess.run(train_step, feed_dict = {x: x_train, y: y_train, keep_prob: .5})
		if _%50:
			train_loss = sess.run(merged, feed_dict = {x: x_train, y: y_train, keep_prob: 1.})
			test_loss = sess.run(merged, feed_dict = {x: x_test, y: y_test, keep_prob: 1.})
			train_writer.add_summary(train_loss, _)
			test_writer.add_summary(test_loss, _)
