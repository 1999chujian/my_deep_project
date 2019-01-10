import numpy as np
#####################################################################
# 数据迭代器
def iterator():
    data = [1, 2, 3]
    for d in data:
        yield d
            
print(iterator)
    
for i in iterator():
    print(i)

#####################################################################
# mask
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
