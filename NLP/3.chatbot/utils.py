import numpy as np
import jieba


# ==============判断char是否是乱码===================
def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True       
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
    if uchar in ('，','。','：','？','“','”','！','；','、','《','》','——','=','*','@','￥','?','.'):
            return True
    return False




class GenData():
	"""docstring for GenData"""

	def __init__(self):
		super(GenData, self).__init__()
		# 特殊字符
		self.SOURCE_CODES = ['<PAD>', '<UNK>']
		self.TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']  # 在target中，需要增加<GO>与<EOS>特殊字符
		self._init_data()
		self._init_vocab()
		self._init_num_data()


	def _init_data(self):
		# ========读取原始数据========
		with open('question', 'r', encoding='utf-8') as f:
		    data = f.read()
		    self.input_list = [list(jieba.cut(line)) for line in data.split('\n')]

		with open('answer', 'r', encoding='utf-8') as a:
			data = a.read()
			self.output_list = [list(jieba.cut(line)) for line in data.split('\n')]

		self.input_list = [[char for char in line] for line in self.input_list]
		self.output_list = [[char for char in line] for line in self.output_list]
		print(self.input_list)



	def _init_vocab(self):
		# 生成输入字典
		helper = []
		for line in self.input_list: helper += line
		self.input_vocab = set(helper)
		self.id2inp = self.SOURCE_CODES + list(self.input_vocab)
		self.inp2id = {c:i for i,c in enumerate(self.id2inp)}
		# 输出字典
		helper = []
		for line in self.output_list: helper += line
		self.output_vocab = set(helper)
		self.id2out = self.TARGET_CODES + list(self.output_vocab)
		self.out2id = {c:i for i,c in enumerate(self.id2out)}
		print(self.out2id)



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
	              




datav = GenData()

print(datav.input_vocab)

