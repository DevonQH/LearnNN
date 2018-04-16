from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

minst = input_data.read_data_sets('MINST_data', one_hot = True)

def add_layer(in_data, in_size, out_size, n_layer, activation_function = None):
	layer_name = 'layer%d'%n_layer
	
	with tf.name_scope(layer_name):
		
		with tf.name_scope(layer_name+'-weights'):
			w = tf.Variable(tf.random_normal([in_size, out_size]))
			tf.summary.histogram(layer_name+'-weights', w)
		
		with tf.name_scope(layer_name+'-bias'):
			b = tf.Variable(tf.zeros([1, out_size])+1.0)
			tf.summary.histogram(layer_name+'-bias', b)

		t = tf.add(tf.matmul(in_data, w),b)
		
		if activation_function is None:
			output = t
		else:
		 	output = activation_function(t)
		return output

def compute_accuracy(x_data, y_data):
	#claim it as global variable
	global pred
	#run the pred to get the result
	y_pre = sess.run(pred, feed_dict={x: x_data})
	#use argmax to get the result
	correct_pred = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_data, 1))
	#data type casting
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	result = sess.run(accuracy, feed_dict = {x: x_data, y: y_data})
	return result

with tf.name_scope('in_data'):
	x = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.float32, [None, 10])

l1 = add_layer(x, 784, 200, n_layer = 1, activation_function = tf.nn.sigmoid)
l2 = add_layer(l1, 200, 100, n_layer = 2, activation_function = tf.nn.sigmoid)
pred = add_layer(l2, 100, 10, n_layer = 3, activation_function = tf.nn.softmax)


with tf.name_scope('loss'):
	loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = [1]))
	tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.4).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter("logs/", sess.graph)
	sess.run(init)
	for _ in range(1000):
		batch_x, batch_y = minst.train.next_batch(100)
		sess.run(train_step, feed_dict = {x: batch_x, y: batch_y})
		if _%50 == 0:
			rs = sess.run(merged, feed_dict = {x: batch_x, y: batch_y})
			writer.add_summary(rs, _)
			print(compute_accuracy(minst.test.images, minst.test.labels))
