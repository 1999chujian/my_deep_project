# =============基于NMT的中英文翻译================
# https://github.com/tensorflow/nmt#building-training-eval-and-inference-graphs

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils import GenData


class nmtModel():
	"""docstring for nmtModel"""
	def __init__(self, num_units, layer_num, input_vocab_size, output_vocab_size, batch_size, init_learning_rate):
		super(nmtModel, self).__init__()
		self.num_units = num_units
		self.layer_num = layer_num
		self.input_vocab_size = input_vocab_size
		self.output_vocab_size = output_vocab_size
		self.batch_size = batch_size
		self.init_learning_rate = init_learning_rate
		self._init_placeholder()
		self._init_embedding()
		self._init_encoder()
		self._init_decoder()
		self._init_optimizer()

	def _init_placeholder(self):
		self.encoder_inputs = tf.placeholder(tf.int32, [None, None])
		self.decoder_inputs = tf.placeholder(tf.int32, [None, None])
		self.decoder_targets = tf.placeholder(tf.int32, [None, None])
		self.target_weights = tf.placeholder(tf.float32, [None, None])
		self.encoder_lengths = tf.placeholder(tf.int32, [None,])
		self.decoder_lengths = tf.placeholder(tf.int32, [None,])
		self.keepprb = tf.placeholder(tf.float32)

	def _init_embedding(self):
		# encoder embedding
		with tf.name_scope('embedding_encoder'):
			self.encoder_embedding = tf.get_variable('embedding_encoder', [self.input_vocab_size, self.num_units])
			self.encoder_emb = tf.nn.embedding_lookup(self.encoder_embedding, self.encoder_inputs)
			self.encoder_emb = tf.nn.dropout(self.encoder_emb, self.keepprb)
		# decoder embedding
		with tf.name_scope('embedding_decoder'):
			self.decoder_embedding = tf.get_variable('embedding_decoder', [self.output_vocab_size, self.num_units])
			self.decoder_emb = tf.nn.embedding_lookup(self.decoder_embedding, self.decoder_inputs)
			self.decoder_emb = tf.nn.dropout(self.decoder_emb, self.keepprb)

	def _init_encoder(self):
		with tf.variable_scope('encoder'):
			self.encoder_cell = self.create_rnn_cell()
			self.initial_state = self.encoder_cell.zero_state(self.batch_size, tf.float32)
			self.encoder_outputs, self.final_state = tf.nn.dynamic_rnn(self.encoder_cell, 
																		self.encoder_emb, 
																		sequence_length=self.encoder_lengths, 
																		time_major=True, 
																		initial_state=self.initial_state)

	def _init_decoder(self):
		with tf.variable_scope('decoder_cell'):
			self.decoder_cell = self.create_rnn_cell()
			self.memory = tf.transpose(self.encoder_outputs, [1, 0, 2])
			# Create an attention mechanism
			self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.num_units, 
																	self.memory, 
																	memory_sequence_length=self.encoder_lengths)
			self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(self.decoder_cell, 
																self.attention_mechanism, 
																attention_layer_size=self.num_units)
			self.projection_layer = Dense(self.output_vocab_size, use_bias=False)
			# Helper
			self.helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb, self.decoder_lengths, time_major=True)
			self.init_state = self.decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.final_state)
			self.decoder_cell = tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, self.helper, self.init_state, output_layer=self.projection_layer)
			self.outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(self.decoder_cell, output_time_major=True, swap_memory=True)
			self.logits = self.outputs.rnn_output


	def _init_optimizer(self):
		with tf.variable_scope('optimizer'):
		    # ===============计算损失=============
		    self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_targets, logits=self.logits)
		    self.cost = (tf.reduce_sum((self.loss * self.target_weights) / self.batch_size))
		    # =============学习率衰减==============
		    self.global_step = tf.Variable(0)
		    self.learning_rate = tf.train.exponential_decay(self.init_learning_rate, self.global_step, 10000//self.batch_size, 0.99, staircase=True)
			# =======通过clip_by_global_norm()控制梯度大小======
		    self.trainable_variables = tf.trainable_variables()
		    self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, self.trainable_variables), 1)
		    self.opt = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.grads, self.trainable_variables))

		# ==============预测输出=============
		with tf.variable_scope('predict'):
			self.predict = tf.argmax(self.logits[:, 0, :], 1)


	def train(self, data, epochs=100):
		# 保存模型
		saver = tf.train.Saver()
		with tf.Session() as sess:
			writer = tf.summary.FileWriter('logs/tensorboard', tf.get_default_graph())
			sess.run(tf.global_variables_initializer())
			for k in range(epochs):
				total_loss = 0.
				BATCH_NUMS = len(data.data_list) // self.batch_size
				data_generator = data.generator(self.batch_size)
				for i in range(BATCH_NUMS):
					en_input, de_input, de_tg, tg_weight, en_len, de_len = next(data_generator)
					feed = {self.encoder_inputs: en_input, self.decoder_inputs: de_input, self.decoder_targets: de_tg, 
							self.target_weights: tg_weight, self.encoder_lengths: en_len, self.decoder_lengths: de_len, 
							self.keepprb: 0.8}
					costs, _ = sess.run([self.cost, self.opt], feed_dict=feed)
					total_loss += costs
					if (i+1) % 10 == 0:
						print('epochs:', k + 1, 'iter:', i + 1, 'cost:', total_loss / i + 1)
						print('output:', ''.join([data.id2out[i] for i in sess.run(self.predict, feed_dict=feed)]))
						print('label:', ''.join([data.id2out[i] for i in de_tg[:, 0]]))    
			saver.save(sess, './checkpoints/lstm.ckpt')
		writer.close()

	# 定义一个生成multi rnn的函数
	def create_rnn_cell(self):
	    def single_rnn_cell():
	        single_cell = tf.contrib.rnn.LSTMCell(self.num_units)
	        #添加dropout
	        cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keepprb)
	        return cell

	    cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.layer_num)])
	    return cell



def main():
	num_units = 256
	layer_num = 2
	batch_size = 32
	init_learning_rate = 0.001
	data = GenData('cmn.txt')
	input_vocab_size = len(data.id2inp)
	output_vocab_size = len(data.id2out)
	transmodel = nmtModel(num_units, layer_num, input_vocab_size, output_vocab_size, batch_size, init_learning_rate)
	transmodel.train(data)

if __name__ == '__main__':
	main()


