import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#generate the raw data
x_data = np.linspace(-1., 1., 100, dtype = np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) + noise

#first, define the input
with tf.name_scope('inputs'):
	x = tf.placeholder(tf.float32, [None, 1])
	y = tf.placeholder(tf.float32, [None, 1])

#define add_layer function
def add_layer(in_vec, in_size, out_size, n_layer, activation_function = None):
	layer_name = 'layer%s'%n_layer
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			w = tf.Variable(tf.random_normal([in_size, out_size]))
			tf.summary.histogram(layer_name + '/weights', w)

		with tf.name_scope('bias'):
			b = tf.Variable(tf.zeros([1, out_size])+0.1)
			tf.summary.histogram(layer_name + '/biases', b)

		with tf.name_scope('temp'):
			t = tf.add(tf.matmul(in_vec, w),b)

		if activation_function is None:
			output = t
		else:
			output = activation_function(t)
		tf.summary.histogram(layer_name + '/output', output)
		return output

#then construct the layers
l1 = add_layer(x, 1, 10, n_layer = 1, activation_function = tf.nn.relu)
l2 = add_layer(l1, 10, 10, n_layer = 2, activation_function = tf.nn.relu)
pred = add_layer(l2, 10, 1, n_layer = 3, activation_function = None)

#compare the loss
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred-y), reduction_indices = [1]))
	tf.summary.scalar('loss', loss)

#choose the optimizer
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#initializer
init = tf.global_variables_initializer()

#plot the data
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.scatter(x_data, y_data)
#plt.ion()
#plt.show()

with tf.Session() as sess:
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter("logs/", sess.graph)
	sess.run(init)
	for _ in range(1000):
		sess.run(train_step, feed_dict = {x: x_data, y: y_data})
		if _%50 == 0:
			rs = sess.run(merged, feed_dict = {x: x_data, y: y_data})
			writer.add_summary(rs, _)
#			try:
#				ax.lines.remove(lines[0])
#			except Exception:
#				pass
#			val = sess.run(pred, feed_dict = {x: x_data})
#			lines = ax.plot(x_data, val, 'r-', lw = 5)
#			plt.pause(0.1)
