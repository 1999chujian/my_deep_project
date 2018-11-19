


# ========读取原始数据========
with open('cmn.txt', 'r', encoding='utf-8') as f:
    data = f.read()
data = data.split('\n')
data = data[:1000]
print(data[-5:])


# 分割英文数据和中文数据
en_data = [line.split('\t')[0] for line in data]
ch_data = [line.split('\t')[1] for line in data]
print('英文数据:\n', en_data[:10])
print('\n中文数据:\n', ch_data[:10])


# 特殊字符
SOURCE_CODES = ['<PAD>', '<UNK>']
TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']  # 在target中，需要增加<GO>与<EOS>特殊字符

# 分别生成中英文字典
en_vocab = set(''.join(en_data))
id2en = SOURCE_CODES + list(en_vocab)
en2id = {c:i for i,c in enumerate(id2en)}

# 分别生成中英文字典
ch_vocab = set(''.join(ch_data))
id2ch = TARGET_CODES + list(ch_vocab)
ch2id = {c:i for i,c in enumerate(id2ch)}

print('\n英文字典:\n', en2id)
print('\n中文字典共计\n:', ch2id)


# 利用字典，映射数据
en_num_data = [[en2id[en] for en in line] for line in en_data]
ch_num_data = [[ch2id['<GO>']] + [ch2id[ch] for ch in line] + [ch2id['<EOS>']] for line in ch_data]
de_num_data = [[ch2id[ch] for ch in line] + [ch2id['<EOS>']] for line in ch_data]

print('char:', en_data[1])
print('index:', en_num_data[1])

en_maxlength = max([len(line) for line in en_num_data])
ch_maxlength = max([len(line) for line in ch_num_data])

# 文本数据转化为数字数据
en_num_data = [data + [en2id['<PAD>']] * (en_maxlength - len(data)) for data in en_num_data]
ch_num_data = [data + [en2id['<PAD>']] * (ch_maxlength - len(data)) for data in ch_num_data]
de_num_data = [data + [en2id['<PAD>']] * (ch_maxlength - len(data)) for data in de_num_data]


# 设计数据生成器
def batch_data(en_num_data, ch_num_data, de_num_data, batch_size):
    batch_num = len(en_num_data) // batch_size
    for i in range(batch_num):
        begin = i * batch_size
        end = begin + batch_size
        x = en_num_data[begin:end]
        y = ch_num_data[begin:end]
        z = de_num_data[begin:end]
        yield x, y, z

