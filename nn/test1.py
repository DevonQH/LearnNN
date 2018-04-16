import tensorflow as tf

#mat1 = tf.constant(1)
#mat2 = tf.Variable(2, name='counter')

#product = tf.multiply(mat1, mat2)

#sess = tf.Session()
#res = sess.run(product)
#print(res)
#sess.close()

#init = tf.initialize_all_variables()
#assign a tf variable must use tf.assign
#return values must be stored when describing
#the procedure
#update = tf.assign(mat2, product)
#with tf.Session() as sess:
	#must initialize if variable exists
#	sess.run(init)
#	for _ in range(3):
		#only the depended actions will get taken
#		sess.run(product)
		#so only this one stores the content
		#sess.run(update)
	#variables also need to be run for result
	#one can think it as type conversion
#	print(sess.run(mat2))
	#otherwise
#	print(mat2)

#c = tf.constant([[0.4,0.5,0.1],[0.3,0.3,0.4],[0.3,0.2,0.1]])
#v = tf.Variable([[1.0],[1.0],[2.0]], name='vec')
#init = tf.initialize_all_variables()
#prod = tf.matmul(c,v)
#update = tf.assign(v, prod)
#flat = tf.reshape(v, (1,3))

#with tf.Session() as sess:
#	sess.run(init)
#	for _ in range(100):
#		val = sess.run(flat)
#		print(val)

c = tf.constant(1.)
v1 = tf.placeholder(tf.float32)
v2 = tf.placeholder(tf.float32)
output = tf.add(c,tf.multiply(v1, v2))

with tf.Session() as sess:
	print(sess.run(output, feed_dict={v1: [7.], v2: [3.]}))
