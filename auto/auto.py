import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

tf.set_random_seed(14)

#hyper
BATCH_SIZE = 64
LR = 0.01

mnist = input_data.read_data_sets("../mnist/MNIST_data/", one_hot = False)
x_data = mnist.test.images[:200]
y_data = mnist.test.labels[:200]

#input
x = tf.placeholder(tf.float32, [None, 28*28])

#layers
e1 = tf.layers.dense(x, 128, tf.nn.tanh)
e2 = tf.layers.dense(e1, 64, tf.nn.tanh)
e3 = tf.layers.dense(e2, 12, tf.nn.tanh)
encoded = tf.layers.dense(e3, 3)

d1 = tf.layers.dense(encoded, 12, tf.nn.tanh)
d2 = tf.layers.dense(d1, 64, tf.nn.tanh)
d3 = tf.layers.dense(d2, 128, tf.nn.tanh)
decoded = tf.layers.dense(d3, 28*28, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels = x, predictions = decoded)
train = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

f, a = plt.subplots(2, 4, figsize = (4, 2))
plt.ion()

view_data = mnist.test.images[:4]
for i in range(4):
	a[0][i].imshow(np.reshape(view_data[i], (28,28)), cmap = 'gray')
	a[0][i].set_xticks(())
	a[0][i].set_yticks(())

for step in range(8000):
	batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
	_, encoded_, decoded_, loss_ = sess.run([train, encoded, decoded, loss], feed_dict = {x: batch_x})
	if step%1000 == 0:
		print('train loss: %.4f' % loss_)
		decoded_data = sess.run(decoded, {x: x_data})
		for i in range(4):
			a[1][i].clear()
			a[1][i].imshow(np.reshape(decoded_data[i],(28, 28)),cmap = 'gray')
			a[1][i].set_xticks(())
			a[1][i].set_yticks(())
		plt.draw()
		plt.pause(0.01)
plt.ioff()
