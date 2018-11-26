from utils import GenData
from params import create_hparams
from model import BaseModel



param = create_hparams()
data = GenData('cmn.txt')
param.encoder_vocab_size = len(data.id2inp)
param.decoder_vocab_size = len(data.id2out)

model = BaseModel(param, 'train')
model.train(data)
