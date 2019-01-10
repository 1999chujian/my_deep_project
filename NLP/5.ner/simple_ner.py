# -----------------------------------------------------------------------------------------------------
'''
&usage:		NER命名实体模型训练
dataset:	conll2003_v2
@author:	hongwen sun
codefrom:	deeppavlov
'''
# -----------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------
'''
&usage:		数据下载并解压至data，数据格式请查看文件，我们需要提取第一列和最后一列信息
'''
# -----------------------------------------------------------------------------------------------------
import os

if not os.path.exists('data/train.txt'):
	import wget, tarfile
	DATA_URL = 'http://files.deeppavlov.ai/deeppavlov_data/conll2003_v2.tar.gz'
	out_fname = 'data.tar.gz'
	wget.download(DATA_URL, out=out_fname)
	tar = tarfile.open(out_fname)
	tar.extractall('data/')
	tar.close()
	os.remove(out_fname)


# -----------------------------------------------------------------------------------------------------
'''
&usage:		读取数据，将下载后的数据转化为容易读取的格式
dataset datatype:

    {'train': [(['Mr.', 'Dwag', 'are', 'derping', 'around'], ['B-PER', 'I-PER', 'O', 'O', 'O']), ....],
     'valid': [...],
     'test': [...]}
'''
# -----------------------------------------------------------------------------------------------------

class NerDatasetReader:
    def read(self, data_path):
        data_parts = ['train', 'valid', 'test']
        extension = '.txt'
        dataset = {}
        for data_part in data_parts:
            file_path = data_path + data_part + extension
            dataset[data_part] = self.read_file(str(file_path))
        return dataset
            
    def read_file(self, file_path):
        fileobj = open(file_path, 'r', encoding='utf-8')
        samples = []
        tokens = []
        tags = []
        for content in fileobj:
            content = content.strip('\n')
            if content == '-DOCSTART- -X- -X- O':
                pass
            elif content == '':
                if len(tokens) != 0:
                    samples.append((tokens, tags))
                    tokens = []
                    tags = []
            else:
                tokens.append(content.split(' ', 1)[0])
                tags.append(content.split(' ')[-1])
        return samples


# 读取下载好的数据
dataset_reader = NerDatasetReader()
dataset = dataset_reader.read('data/')

for sample in dataset['train'][:4]:
    for token, tag in zip(*sample):
        print('%s\t%s' % (token, tag))
    print()


# -----------------------------------------------------------------------------------------------------
'''
&usage:		准备字典，将建立数据->index的映射关系
'''
# -----------------------------------------------------------------------------------------------------
from collections import defaultdict, Counter
from itertools import chain
import numpy as np

class Vocab:
    def __init__(self,
                 special_tokens=tuple()):
        self.special_tokens = special_tokens
        self._t2i = defaultdict(lambda: 1)
        self._i2t = []
        
    def fit(self, tokens):
        count = 0
        self.freqs = Counter(chain(*tokens))
        # The first special token will be the default token
        for special_token in self.special_tokens:
            self._t2i[special_token] = count
            self._i2t.append(special_token)
            count += 1
        for token, freq in self.freqs.most_common():
            if token not in self._t2i:
                # t2i是字典，i2t是所有字符串的列表
                self._t2i[token] = count
                self._i2t.append(token)
                count += 1

    def __call__(self, batch, **kwargs):
        # Implement the vocab() method. The input could be a batch of tokens
        # or a batch of indices. A batch is a list of utterances where each
        # utterance is a list of tokens
        indices_batch = []
        for tokens in batch:
            indices = []
            for token in tokens:
                indices.append(self[token])
            indices_batch.append(indices)
        return indices_batch

    def __getitem__(self, key):
        # Implement the vocab[] method. The input could be a token
        # (string) or an index. You have to detect what type of data
        # is key and return. 
        if isinstance(key, (int, np.integer)):
            return self._i2t[key]
        elif isinstance(key, str):
            return self._t2i[key]
        else:
            raise NotImplementedError("not implemented for type `{}`".format(type(key)))
    
    def __len__(self):
        return len(self._i2t)

# 例子：
# 增加特殊符号
special_tokens = ['<UNK>']
# 实例化
token_vocab = Vocab(special_tokens)
tag_vocab = Vocab()
# 用于进行训练的数据
all_tokens_by_sentenses = [tokens for tokens, tags in dataset['train']]
all_tags_by_sentenses = [tags for tokens, tags in dataset['train']]
# 进行训练
token_vocab.fit(all_tokens_by_sentenses)
tag_vocab.fit(all_tags_by_sentenses)
assert len(token_vocab) == 23624, print('There must be exactly 23624 in the token vocab!')
assert len(tag_vocab) == 9, print('There must be exactly 9 in the tag vocab!')

print(token_vocab([['be','to']]))
print(token_vocab([[39,6]]))
print(tag_vocab([[0,1,2,3,4,5,6,7,8]]))


# -----------------------------------------------------------------------------------------------------
'''
&usage:		Dataset Iterator，建立数据生成器
'''
# -----------------------------------------------------------------------------------------------------
class DatasetIterator:
    def __init__(self, data):
        self.data = {
            'train': data['train'],
            'valid': data['valid'],
            'test': data['test']
        }

    def gen_batches(self, batch_size, data_type='train', shuffle=True):
        n = 0
        while n < 10:
            x = []
            y = []
            for i in range(batch_size):
                num = n * batch_size + i
                datas = self.data[data_type]
                x.append(datas[num][0])
                y.append(datas[num][1])
            n = n + 1
            yield x, y


data_iterator = DatasetIterator(dataset)
x, y = next(data_iterator.gen_batches(2))
print(x,'\n',y)

# -----------------------------------------------------------------------------------------------------
'''
&usage:		Masking，构建MASK，输入训练集的batch，返回二进制的数组
egs:		in:	[['hello','wo','ai'], ['hi']]	
			out:[[1,1,1],[1,0,0]]
'''
# -----------------------------------------------------------------------------------------------------
class Mask():
	"""docstring for Mask"""
	def __init__(self):
		pass

	def __call__(self, token_batches, **kwargs):
		batches_size = len(token_batches)
		maxlen = max(len(utt) for utt in token_batches)
		mask = np.zeros([batches_size, maxlen], dtype=np.float32)
		for n, utt in enumerate(token_batches):
			mask[n, :len(utt)] = 1
		return mask

get_mask = Mask()
print(get_mask([['Try', 'to', 'get', 'the', 'mask'], ['Check', 'paddings']]))


# -----------------------------------------------------------------------------------------------------
'''
&usage:		搭建卷积神经网络
'''
# -----------------------------------------------------------------------------------------------------
import tensorflow as tf

# 词嵌入层，输入形如[batch_size, num_tokens]的数据，输出词向量信息会通过tokenindex值找到词向量
def get_embeddings(indices, vocabulary_size, emb_dim):
    # Initialize the random gaussian matrix with dimensions [vocabulary_size, embedding_dimension]
    # The **VARIANCE** of the random samples must be 1 / embedding_dimension
    emb_mat = np.random.randn(vocabulary_size, emb_dim).astype(np.float32) / np.sqrt(emb_dim)
    emb_mat = tf.Variable(emb_mat, name='Embeddings', trainable=True)
    emb = tf.nn.embedding_lookup(emb_mat, indices)
    return emb


# 卷积层
# units为输入形状，n_hidden_list为隐藏层卷积核个数，cnn_fileter_width为卷积核大小, activ激活函数
def conv_net(units, n_hidden_list, cnn_filter_width, activation=tf.nn.relu):
    # Use activation(units) to apply activation to units
    for n_hidden in n_hidden_list:
        
        units = tf.layers.conv1d(units,
                                 n_hidden,
                                 cnn_filter_width,
                                 padding='same')
        units = activation(units)
    return units


# 损失函数
# 
def masked_cross_entropy(logits, label_indices, number_of_tags, mask):
    ground_truth_labels = tf.one_hot(label_indices, depth=number_of_tags)
    loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ground_truth_labels, logits=logits)
    loss_tensor *= mask
    loss = tf.reduce_mean(loss_tensor)
    return loss


# 创建完整的神经网络
class NerNetwork:
    def __init__(self,
                 n_tokens,
                 n_tags,
                 token_emb_dim=100,
                 n_hidden_list=(128,),
                 cnn_filter_width=7,
                 use_batch_norm=False,
                 embeddings_dropout=False,
                 top_dropout=False,
                 **kwargs):
        
        # ================ Building inputs =================
        
        self.learning_rate_ph = tf.placeholder(tf.float32, [])
        self.dropout_keep_ph = tf.placeholder(tf.float32, [])
        self.token_ph = tf.placeholder(tf.int32, [None, None], name='token_ind_ph')
        self.mask_ph = tf.placeholder(tf.float32, [None, None], name='Mask_ph')
        self.y_ph = tf.placeholder(tf.int32, [None, None], name='y_ph')
        
        # ================== Building the network ==================
        
        # Now embedd the indices of tokens using token_emb_dim function
        emb = get_embeddings(self.token_ph, n_tokens, token_emb_dim)
        emb = tf.nn.dropout(emb, self.dropout_keep_ph, (tf.shape(emb)[0], 1, tf.shape(emb)[2]))
        
        # Build a multilayer CNN on top of the embeddings.
        # The number of units in the each layer must match
        # corresponding number from n_hidden_list.
        # Use ReLU activation 
        units = conv_net(emb, n_hidden_list, cnn_filter_width)
        units = tf.nn.dropout(units, self.dropout_keep_ph, (tf.shape(units)[0], 1, tf.shape(units)[2]))
        logits = tf.layers.dense(units, n_tags, activation=None)
        self.predictions = tf.argmax(logits, 2)
        
        # ================= Loss and train ops =================
        # Use cross-entropy loss. check the tf.nn.softmax_cross_entropy_with_logits_v2 function
        self.loss = masked_cross_entropy(logits, self.y_ph, n_tags, self.mask_ph)

        # Create a training operation to update the network parameters.
        # We purpose to use the Adam optimizer as it work fine for the
        # most of the cases. Check tf.train to find an implementation.
        # Put the train operation to the attribute self.train_op
        
        optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.train_op = optimizer.minimize(self.loss)

        # ================= Initialize the session =================
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __call__(self, tok_batch, mask_batch):
        feed_dict = {self.token_ph: tok_batch,
                     self.mask_ph: mask_batch,
                     self.dropout_keep_ph: 1.0}
        return self.sess.run(self.predictions, feed_dict)

    def train_on_batch(self, tok_batch, tag_batch, mask_batch, dropout_keep_prob, learning_rate):
        feed_dict = {self.token_ph: tok_batch,
                     self.y_ph: tag_batch,
                     self.mask_ph: mask_batch,
                     self.dropout_keep_ph: dropout_keep_prob,
                     self.learning_rate_ph: learning_rate}
        self.sess.run(self.train_op, feed_dict)


nernet = NerNetwork(len(token_vocab),
                    len(tag_vocab),
                    n_hidden_list=[100, 100])


# -----------------------------------------------------------------------------------------------------
'''
&usage:     测试准确率方法，还未整理
'''
# -----------------------------------------------------------------------------------------------------










# zero pad 顾名思义
# zero_pad takes a batch of lists of token indices, pad it with zeros to the
# maximal length and convert it to numpy matrix
def zero_pad(batch, dtype=np.float32):
    if len(batch) == 1 and len(batch[0]) == 0:
        return np.array([], dtype=dtype)
    batch_size = len(batch)
    max_len = max(len(utterance) for utterance in batch)
    if isinstance(batch[0][0], (int, np.int)):
        padded_batch = np.zeros([batch_size, max_len], dtype=np.int32)
        for n, utterance in enumerate(batch):
            padded_batch[n, :len(utterance)] = utterance
    else:
        n_features = len(batch[0][0])
        padded_batch = np.zeros([batch_size, max_len, n_features], dtype=dtype)
        for n, utterance in enumerate(batch):
            for k, token_features in enumerate(utterance):
                padded_batch[n, k] = token_features
    return padded_batch






batch_size = 64 # YOUR HYPERPARAMETER HERE
n_epochs = 80 # YOUR HYPERPARAMETER HERE
learning_rate = 0.001 # YOUR HYPERPARAMETER HERE
dropout_keep_prob = 0.5 # YOUR HYPERPARAMETER HERE



for epoch in range(n_epochs):
    for x, y in data_iterator.gen_batches(batch_size, 'train'):
        # Convert tokens to indices via Vocab
        x_inds = token_vocab(x) # YOUR CODE 
        # Convert tags to indices via Vocab
        y_inds = tag_vocab(y) # YOUR CODE 
        
        # Pad every sample with zeros to the maximal length
        x_batch = zero_pad(x_inds)
        y_batch = zero_pad(y_inds)

        mask = get_mask(x)
        nernet.train_on_batch(x_batch, y_batch, mask, dropout_keep_prob, learning_rate)



# 测试结果
sentence = 'EU rejects German call to boycott British lamb .'
x = [sentence.split()]

x_inds = token_vocab(x)
x_batch = zero_pad(x_inds)
mask = get_mask(x)
y_inds = nernet(x_batch, mask)
print(x[0])
print(tag_vocab(y_inds)[0])



