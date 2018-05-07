import tensorflow as tf

def build_model1(inputs, hidden_size, output_size, p, scope="x1"):
	with tf.variable_scope(scope):
		# inputs = tf.placeholder(name = "inputs",shape = embedding_size)
		outputs = tf.contrib.layers.fully_connected(
			inputs,
			hidden_size,
			activation_fn=tf.nn.relu,
		)
		outputs1 = tf.nn.dropout(outputs,p)
		outputs2 = tf.contrib.layers.fully_connected(
			outputs1,
			output_size,
			activation_fn=tf.nn.relu,
		)
		return outputs2

def build_model2(inputs, hidden_size1, hidden_size2, label_size, p=0.1, scope="x2"):
	with tf.variable_scope(scope):
		# inputs = tf.placeholder(name = "inputs",shape = input_size)

		outputs = tf.contrib.layers.fully_connected(
			inputs,
			hidden_size1,
			activation_fn=tf.nn.relu,
		)
		outputs1 = tf.nn.dropout(outputs,p)
		outputs2 = tf.contrib.layers.fully_connected(
			outputs1,
			hidden_size2,
			activation_fn=tf.nn.relu,
		)
		outputs3 = tf.nn.dropout(outputs2,p)
		outputs4 = tf.contrib.layers.fully_connected(
			outputs3,
			label_size,
			activation_fn=None,
		)

		return outputs4