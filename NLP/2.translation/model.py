import tensorflow as tf
import jieba
from tensorflow.python.layers.core import Dense
import numpy as np
from utils import attention_mechanism_fn, create_rnn_cell
import os



class BaseModel():
    """docstring for BaseModel."""
    def __init__(self, params, mode):
        super(BaseModel, self).__init__()
        tf.reset_default_graph()
        self.mode = mode
        # networks
        self.num_units = params.num_units
        self.num_layers = params.num_layers
        # attention type and architecture
        self.attention_type = params.attention_type
        self.attention_architecture = params.attention_architecture
        # optimizer
        self.optimizer = params.optimizer
        self.learning_rate = params.learning_rate # sgd:1, Adam:0.0001
        self.decay_steps = params.decay_steps
        self.decay_rate = params.decay_rate
        self.epochs = params.epochs # 100000
        # Data
        self.out_dir = params.out_dir # log/model files
        # vocab
        self.encoder_vocab_size = params.encoder_vocab_size
        self.decoder_vocab_size = params.decoder_vocab_size
        self.share_vocab = params.share_vocab # False
        # Sequence lengths
        self.src_max_len = params.src_max_len # 50
        self.tgt_max_len = params.tgt_max_len # 50
        # Default settings works well (rarely need to change)
        self.unit_type = params.unit_type # lstm
        self.keep_prob = params.keep_prob # 0.8
        self.max_gradient_norm = params.max_gradient_norm # 1
        self.batch_size = params.batch_size # 32
        self.num_gpus = params.num_gpus # 1
        self.time_major = params.time_major
        # inference
        self.infer_mode = params.infer_mode # greedy / beam_search
        self.beam_width = params.beam_width # 0
        self.num_translations_per_input = params.num_translations_per_input # 1
        self._model_init()


    def _model_init(self):
        self._placeholder_init()
        self._embedding_init()
        self._encoder_init()
        self._decoder_init()


    def _placeholder_init(self):
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None])
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None])
        self.decoder_targets = tf.placeholder(tf.int32, [None, None])
        self.mask = tf.placeholder(tf.float32, [None, None])
        self.encoder_input_lengths = tf.placeholder(tf.int32, [None])
        self.decoder_input_lengths = tf.placeholder(tf.int32, [None])


    def _embedding_init(self):
        self.encoder_embedding = tf.get_variable(
            name='encoder_embedding',
            shape=[self.encoder_vocab_size, self.num_units])
        self.encoder_emb_inp = tf.nn.embedding_lookup(
            self.encoder_embedding,
            self.encoder_inputs)
        self.decoder_embedding = tf.get_variable(
            name='decoder_embedding',
            shape=[self.decoder_vocab_size, self.num_units])
        self.decoder_emb_inp = tf.nn.embedding_lookup(
            self.decoder_embedding,
            self.decoder_inputs)


    def _encoder_init(self):
        with tf.name_scope('encoder'):
            encoder_cell = create_rnn_cell(
                self.unit_type,
                self.num_units,
                self.num_layers,
                self.keep_prob)
            encoder_init_state = encoder_cell.zero_state(
                self.batch_size, tf.float32)
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                encoder_cell,
                self.encoder_emb_inp,
                sequence_length=self.encoder_input_lengths,
                time_major=self.time_major,
                initial_state=encoder_init_state)


    def _decoder_init(self):

        if self.time_major == True:
            memory = tf.transpose(self.encoder_outputs, [1, 0, 2])
        else:
            memory = self.encoder_outputs

        with tf.name_scope('decoder'):
            cell =create_rnn_cell(
                self.unit_type,
                self.num_units,
                self.num_layers,
                self.keep_prob)
            attention_mechanism = attention_mechanism_fn(
                self.attention_type,
                self.num_units,
                memory,
                self.encoder_input_lengths)
            cell = tf.contrib.seq2seq.AttentionWrapper(
                cell,
                attention_mechanism,
                attention_layer_size=self.num_units)
            init_state = cell.zero_state(self.batch_size, tf.float32).clone(
                cell_state=self.encoder_state)
            projection_layer = Dense(self.decoder_vocab_size, use_bias=False)

        if self.mode == 'train':
            train_helper = tf.contrib.seq2seq.TrainingHelper(
                self.decoder_emb_inp,
                self.decoder_input_lengths,
                time_major=True)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell,
                train_helper,
                init_state,
                output_layer=projection_layer)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                train_decoder,
                output_time_major=True,
                swap_memory=True)
            logits = outputs.rnn_output

            with tf.name_scope('optimizer'):
                # loss
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.decoder_targets,
                    logits=logits)
                self.cost = tf.reduce_sum((loss * self.mask) / self.batch_size)
                tf.summary.scalar('loss', self.cost)
                # learning_rate decay
                self.global_step = tf.Variable(0)
                self.learning_rate = tf.train.exponential_decay(
                    self.learning_rate,
                    self.global_step,
                    self.decay_steps,
                    self.decay_rate,
                    staircase=True)
                # clip_by_global_norm
                self.trainable_variables = tf.trainable_variables()
                self.grads, _ = tf.clip_by_global_norm(
                    tf.gradients(self.cost, self.trainable_variables),
                    self.max_gradient_norm)
                # OPTIMIZE: adam | sgd
                if self.optimizer == 'adam':
                    opt = tf.train.AdamOptimizer(self.learning_rate)
                elif self.optimizer == 'sgd':
                    opt = tf.train.GradientDescentOptimizer(
                        self.learning_rate)
                else:
                    raise ValueError('unkown optimizer %s' % self.optimizer)

                self.update = opt.apply_gradients(
                    zip(self.grads, self.trainable_variables),
                    global_step=self.global_step)

        elif self.mode == 'infer':
            start_tokens = tf.ones([self.batch_size,], tf.int32) * 1
            end_token = 2
            if self.infer_mode == 'greedy':
                infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self.decoder_embedding,
                    start_tokens=start_tokens,
                    end_token=end_token)
                infer_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    infer_helper,
                    init_state,
                    output_layer=projection_layer)
            elif self.infer_mode == 'beam_search':
                infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=cell,
                    embedding=self.decoder_embedding,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=init_state,
                    beam_width=self.beam_width,
                    output_layer=projection_layer)
            else:
                raise ValueError('unkown infer mode %s' % self.infer_mode)

            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=infer_decoder,
                maximum_iterations=50)
            self.translations = decoder_outputs.sample_id


    def train(self, data):
        saver =tf.train.Saver()
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())
            md_name = '_' + str(self.num_layers) + '_' + str(self.num_units)
            md_file = self.out_dir + md_name
            if os.path.exists(md_file + '/model.meta'):
                saver.restore(sess, md_file + '/model')
            writer = tf.summary.FileWriter(
                md_file + '/tensorboard', tf.get_default_graph())
            for k in range(self.epochs):
                total_loss = 0
                batch_num = len(data.data) // self.batch_size
                data_generator = data.generator(self.batch_size)
                for i in range(batch_num):
                    en_inp, de_inp, de_tg, mask, en_len, de_len = next(
                        data_generator)
                    feed = {
                        self.encoder_inputs: en_inp,
                        self.decoder_inputs: de_inp,
                        self.decoder_targets: de_tg,
                        self.mask: mask,
                        self.encoder_input_lengths: en_len,
                        self.decoder_input_lengths: de_len}
                    cost,_ = sess.run([self.cost,self.update], feed_dict=feed)
                    total_loss += cost
                    if (k * batch_num + i) % 10 == 0:
                        rs=sess.run(merged, feed_dict=feed)
                        writer.add_summary(rs, k * batch_num + i)
                if k % 5 == 0:
                    print('epochs', k, ': average loss = ', total_loss/batch_num)
            saver.save(sess, md_file + '/model')
            writer.close()


    def inference(self, data):
        saver =tf.train.Saver()
        with tf.Session() as sess:
            md_name = '_' + str(self.num_layers) + '_' + str(self.num_units)
            md_file = self.out_dir + md_name
            saver.restore(sess, md_file + '/model')
            while True:
                inputs = input('input english: ')
                if inputs == 'exit': break
                if data.mode == 'jieba': inputs = jieba.lcut(inputs)
                encoder_inputs = [[data.en2id.get(en, 3)] for en in inputs]
                encoder_length = [len(encoder_inputs)]
                feed = {
                    self.encoder_inputs: encoder_inputs,
                    self.encoder_input_lengths: encoder_length}
                predict = sess.run(self.translations, feed_dict=feed)
                outputs = ''.join([data.id2ch[i] for i in predict[0]])
                print('output chinese:', outputs)
