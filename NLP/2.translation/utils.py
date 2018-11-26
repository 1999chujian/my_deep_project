import numpy as np
import tensorflow as tf


def attention_mechanism_fn(attention_type, num_units, memory, encoder_length):
    if attention_type == 'luong':
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=encoder_length)
    elif attention_type == 'bahdanau':
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=encoder_length)
    else:
        raise ValueError('unkown atteion type %s' % attention_type)
    return attention_mechanism


def create_rnn_cell(unit_type, num_units, num_layers, keep_prob):
    def single_rnn_cell():
        if unit_type == 'lstm':
            single_cell = tf.contrib.rnn.LSTMCell(num_units)
        elif unit_type == 'gru':
            single_cell = tf.contrib.rnn.GRUCell(num_units)
        elif unit_type == 'rnn':
            single_cell = tf.contrib.rnn.LSTMCell(num_units)
        else:
            raise ValueError("Unknown cell type %s" % unit_type)
        cell = tf.contrib.rnn.DropoutWrapper(
            single_cell,
            output_keep_prob=keep_prob)
        return cell
    mul_cell = tf. contrib.rnn.MultiRNNCell(
        [single_rnn_cell() for _ in range(num_layers)])
    return mul_cell



class GenData():
	"""docstring for GenData"""

	def __init__(self, filepath):
		super(GenData, self).__init__()
		# 特殊字符
		self.SOURCE_CODES = ['<PAD>', '<UNK>']
		self.TARGET_CODES = ['<PAD>', '<GO>', '<EOS>', '<UNK>']  # 在target中，需要增加<GO>与<EOS>特殊字符
		self.filepath = filepath
		self._init_data()
		self._init_vocab()
		self._init_num_data()


	def _init_data(self):
		# ========读取原始数据========
		with open(self.filepath, 'r', encoding='utf-8') as f:
		    data = f.read()
		self.data_list = data.split('\n')[:500]
		self.input_list = [line.split('\t')[0] for line in self.data_list]
		self.output_list = [line.split('\t')[1] for line in self.data_list]


	def _init_vocab(self):
		# 生成输入字典
		self.input_vocab = sorted(list(set(''.join(self.input_list))))
		self.id2inp = self.SOURCE_CODES + list(self.input_vocab)
		self.inp2id = {c:i for i,c in enumerate(self.id2inp)}
		# 输出字典
		self.output_vocab = sorted(list(set(''.join(self.output_list))))
		self.id2out = self.TARGET_CODES + self.output_vocab
		self.out2id = {c:i for i,c in enumerate(self.id2out)}


	def _init_num_data(self):
		# 利用字典，映射数据
		self.en_inp_num_data = [[self.inp2id[en] for en in line] for line in self.input_list]
		self.de_inp_num = [[self.out2id['<GO>']] + [self.out2id[ch] for ch in line] for line in self.output_list]
		self.de_out_num = [[self.out2id[ch] for ch in line] + [self.out2id['<EOS>']] for line in self.output_list]


	def generator(self, batch_size):
	    batch_num = len(self.en_inp_num_data) // batch_size
	    for i in range(batch_num):
	        begin = i * batch_size
	        end = begin + batch_size
	        encoder_inputs = self.en_inp_num_data[begin:end]
	        decoder_inputs = self.de_inp_num[begin:end]
	        decoder_targets = self.de_out_num[begin:end]
	        encoder_lengths = [len(line) for line in encoder_inputs]
	        decoder_lengths = [len(line) for line in decoder_inputs]
	        encoder_max_length = max(encoder_lengths)
	        decoder_max_length = max(decoder_lengths)
	        encoder_inputs = np.array([data + [self.inp2id['<PAD>']] * (encoder_max_length - len(data)) for data in encoder_inputs]).T
	        decoder_inputs = np.array([data + [self.out2id['<PAD>']] * (decoder_max_length - len(data)) for data in decoder_inputs]).T
	        decoder_targets = np.array([data + [self.out2id['<PAD>']] * (decoder_max_length - len(data)) for data in decoder_targets]).T
	        mask = decoder_targets > 0
	        target_weights = mask.astype(np.int32)
	        yield encoder_inputs, decoder_inputs, decoder_targets, target_weights, encoder_lengths, decoder_lengths
