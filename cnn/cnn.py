import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#define the input
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, 784])
	y = tf.placeholder(tf.float32, [None, 10])
	keep_prob = tf.placeholder(tf.float32)
	x_image = tf.reshape(x, [-1, 28, 28, 1])

#read data
data = input_data.read_data_sets('MNIST_data', one_hot = True)

#weight
def weight(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

#bias
def bias(shape):
	return tf.Variable(tf.constant(0.1, shape=shape))

def conv(in_data, w):
	return tf.nn.conv2d(in_data, w, strides = [1,1,1,1], padding = "SAME")

def pool(in_data):
	return tf.nn.max_pool(in_data, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")

def eval(x_data, y_data):
	with tf.name_scope('eval'):
		global pred
		predict = sess.run(pred, feed_dict = {x: x_data, y: y_data, keep_prob: 1.})
		correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y_data, 1)) 
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		result = sess.run(accuracy, feed_dict = {x: x_data, y: y_data, keep_prob: 1.})
		return result

#first conv layer
#with tf.name_scope('conv_1'):
#	w1 = weight([5, 5, 1, 32])
#	b1 = bias([32])
#	h1 = tf.nn.relu(conv(x_image, w1) + b1) #tensor size 28x28x32
#	p1 = pool(h1) #tensor size 14x14x32
#	tf.summary.histogram('w1', w1)

conv1 = tf.layers.conv2d(x_image, 16, 5, 1, 'same', activation = tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

#second conv layer
#with tf.name_scope('conv_2'):
#	w2 = weight([5, 5, 32, 64])
#	b2 = bias([64])
#	h2 = tf.nn.relu(conv(p1, w2) + b2) #tensor size 14x14x64
#	p2 = pool(h2) #tensor size 7x7x64
#	tf.summary.histogram('w2', w2)

conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation = tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

#flatten
#with tf.name_scope('flatten'):
#	p_flat = tf.reshape(p2, [-1, 7*7*64]) #vector 7*7*64

flat = tf.reshape(pool2, [-1, 7*7*32])

#fully connected layer one
#with tf.name_scope('fc_1'):
#	w_fc1 = weight([7*7*64, 1024])
#	b_fc1 = bias([1024])
#	f_fc1 = tf.nn.relu(tf.add(tf.matmul(p_flat, w_fc1), b_fc1)) #vector 7*7*64
#	f_fc1_drop = tf.nn.dropout(f_fc1, keep_prob)

pred = tf.layers.dense(flat, 10)

#fully connected layer two
#with tf.name_scope('fc_2'):
#	w_fc2 = weight([1024, 10])
#	b_fc2 = bias([10])
#	pred = tf.nn.softmax(tf.add(tf.matmul(f_fc1_drop, w_fc2), b_fc2)) #vector 10

#loss
#loss = tf.reduce_mean(
#	-tf.reduce_sum(y*tf.log(pred),
#	reduction_indices=[1]))

loss = tf.losses.softmax_cross_entropy(onehot_labels = y, logits = pred)
tf.summary.scalar('loss', loss)

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

with tf.Session() as sess:
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter("logs/", sess.graph)
	sess.run(tf.global_variables_initializer())
	for _ in range(1000):
		batch_x, batch_y = data.train.next_batch(100)
		sess.run(train_step, feed_dict = {x: batch_x, y: batch_y, keep_prob:0.5})
		if _%50:
			record = sess.run(merged, feed_dict = {x: batch_x, y: batch_y, keep_prob:1.0})
			print(eval(batch_x, batch_y))
			writer.add_summary(record, _)

