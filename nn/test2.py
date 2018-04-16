import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#generate the raw data
x_data = np.linspace(-1., 1., 100, dtype = np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) + noise

#first, define the input
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

#define add_layer function
def add_layer(in_vec, in_size, out_size, activation_function = None):
	w = tf.Variable(tf.random_normal([in_size, out_size]))
	b = tf.Variable(tf.zeros([1, out_size])+0.1)
	t = tf.matmul(in_vec, w)+b
	if activation_function is None:
		return t
	else:
		return activation_function(t)

#then construct the layers
l1 = add_layer(x, 1, 10, activation_function = tf.nn.relu)
l2 = add_layer(l1, 10, 10, activation_function = tf.nn.relu)
pred = add_layer(l2, 10, 1, activation_function = None)

#compare the loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred-y), reduction_indices = [1]))

#choose the optimizer
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#initializer
init = tf.global_variables_initializer()

#plot the data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

with tf.Session() as sess:
	sess.run(init)
	for _ in range(1000):
		sess.run(train_step, feed_dict = {x: x_data, y: y_data})
		if _%50 == 0:
			try:
				ax.lines.remove(lines[0])
			except Exception:
				pass
			val = sess.run(pred, feed_dict = {x: x_data})
			lines = ax.plot(x_data, val, 'r-', lw = 5)
			plt.pause(0.1)
