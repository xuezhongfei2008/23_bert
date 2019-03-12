import argparse
import os, sys
import numpy as np

sys.path.append('../')

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = 2

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 768

    options = {
        'bidirectional': True,

        # 'char_cnn': {'activation': 'relu',
        #              'embedding': {'dim': 16},  # 每个字符的embedding表示维数
        #              'filters': [
        #                  [1, 32],
        #                  [2, 32],
        #                  [3, 64],
        #                  [4, 128],
        #                  [5, 256]
        #                  # [6, 512],
        #                  # [7, 1024]
        #              ],
        #              'max_characters_per_token': 50,  # 每个单词最大字符数
        #              'n_characters': 300000,  # 字符字典中总的字符个数，就60个?
        #              'n_highway': 2},  # 使用high way网络

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,  # if provided the cell state is clipped by this value prior to the cell output activation.
            'dim': 4096,  # 隐藏层神经元个数
            'n_layers': 2,
            'proj_clip': 3,
        # If num_proj > 0 and proj_clip is provided, then the projected values are clipped elementwise to within [-proj_clip, proj_clip].
            'projection_dim': 512,  # num_proj 投影矩阵的输出维数。 如果为None，则不执行投影。#最终维度,投影层维度
            'use_skip_connections': True},

        'all_clip_norm_val': 10.0,

        'n_epochs': 1,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,  # 输入句子的长度#最大时长,n_token
        'n_negative_samples_batch': 8192,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                  shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    # print("",)
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../log', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', default='../data/vocab_seg_words_elmo.txt', help='Vocabulary file')
    parser.add_argument('--train_prefix', default='../data/example/300_seg_words.txt', help='Prefix for train files')

    args = parser.parse_args()
    print("args", args)
    main(args)
