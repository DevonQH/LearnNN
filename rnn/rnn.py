import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)

BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01

#data
mnist = input_data.read_data_sets('../mnist/MNIST_data', one_hot = True)
x_data = mnist.test.images[:2000]
y_data = mnist.test.labels[:2000]

#input
x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])
y = tf.placeholder(tf.float32, [None, 10])
image = tf.reshape(x, [-1, TIME_STEP, INPUT_SIZE])

#rnn
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units = 64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
		rnn_cell,
		image,
		initial_state = None,
		dtype = tf.float32,
		time_major = False,
)
pred = tf.layers.dense(outputs[:, -1, :], 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels = y, logits = pred)
train = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(labels = tf.argmax(y, axis = 1), predictions = tf.argmax(pred, axis = 1))[1]

with tf.Session() as sess:
	sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
	for _ in range(1200):
		batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
		_r, loss_ = sess.run([train, loss], {x: batch_x, y: batch_y})
		if _%50 == 0:
			accuracy_ = sess.run(accuracy, {x: x_data, y: y_data})
			print(accuracy_)
