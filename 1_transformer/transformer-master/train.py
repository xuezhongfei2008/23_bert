# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import tensorflow as tf

from hyperparams import Hyperparams as hp
from data_load import get_batch_data, load_de_vocab, load_en_vocab
from modules import *
import os, codecs
from tqdm import tqdm

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data() # (N, T)
            else: # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # 2:<S>
            '''decoder_inputs和self.y相比，去掉了最后一个句子结束符，而在每句话最前面加了一个初始化为2的id，即 < S > ，代表开始。shape和self.y一样为[N, T]'''

            # Load vocabulary    
            de2idx, idx2de = load_de_vocab()
            en2idx, idx2en = load_en_vocab()
            
            # Encoder
            with tf.variable_scope("encoder"):
                ## Embedding
                self.enc = embedding(self.x, 
                                      vocab_size=len(de2idx), 
                                      num_units=hp.hidden_units, 
                                      scale=True,
                                      scope="enc_embed")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                      num_units=hp.hidden_units,  #词汇表最大数
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                else:
                    '''embedding 的输入的第二各维度的值不是词的 id,而是变成了该词的位置 id,
                    一共只有 maxlen 种这样的 id,位置的 id 利用了tf.range 实现,最后扩展到了 batch 中的所有句子,因为每个句子中词的位置 id都是一样的'''
                    self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=hp.maxlen,    #句子最大长度
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                    '''tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]):
                    得到的数据（32,10）的位置信息[0,1,2,3,4,5,6,7,8,9]，embedding后还是512维的一个向量'''


                ## Dropout
                self.enc = tf.layers.dropout(self.enc, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    '''默认为6个这样的block结构。所以代码循环6次。其中每个block都调用了依次multihead_attention以及feedforward函数'''
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False)
                        # print("self.enc4",self.enc)
                        
                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])

            # Decoder
            with tf.variable_scope("decoder"):
                ## Embedding
                self.dec = embedding(self.decoder_inputs, 
                                      vocab_size=len(en2idx), 
                                      num_units=hp.hidden_units,
                                      scale=True, 
                                      scope="dec_embed")
                
                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                
                ## Dropout
                self.dec = tf.layers.dropout(self.dec, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)

                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.dec, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True, 
                                                        scope="self_attention")
                        
                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training, 
                                                        causality=False,
                                                        scope="vanilla_attention")
                        
                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
                
            # Final linear projection
            self.logits = tf.layers.dense(self.dec, len(en2idx))
            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)
                
            if is_training:  
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(en2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
               
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                   
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss)
                self.merged = tf.summary.merge_all()

if __name__ == '__main__':                
    # Load vocabulary    
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # Construct graph
    g = Graph("train"); print("Graph loaded")
    
    # Start session
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)
                
            gs = sess.run(g.global_step)   
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
    
    print("Done")    
    

