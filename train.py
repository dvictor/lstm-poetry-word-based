#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import os

import reader
import model
from collections import namedtuple
import json
import glob

'''
    Small config:
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

    Medium config:
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000

    Large config:
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
'''

WORK_DIR = 'data-lyrics'
#WORK_DIR = 'data-eminescu'

nn_config = {
    'init_scale': 0.1,
    'max_grad_norm': 5,
    'num_layers': 2,
    'num_steps': 30,
    'hidden_size': 400,
    'keep_prob': .6,
    'batch_size': 20,
    'vocab_size': 15000
}

train_config = {
    'max_max_epoch': 50,
    'max_epoch': 40,
    'learning_rate': 1.0,
    'lr_decay': 0.6
}


def run_epoch(session, m, data, eval_op, verbose=False):
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, y) in enumerate(reader.train_iterator(data, m.batch_size,
                                                      m.num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})
        costs += cost
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def main():

    # cleanup input dir
    ret = raw_input('Are you sure you want to clean %s [yes|no] ' % (WORK_DIR,))
    if ret == 'yes':
        for f in glob.glob(os.path.join(WORK_DIR, '*')):
            if not f.endswith('.txt'):
                os.remove(f)
                print(f + ' deleted')

    config = namedtuple('TrainConfig', train_config.keys())(*train_config.values())
    model_config = namedtuple('ModelConfig', nn_config.keys())(*nn_config.values())

    with open(os.path.join(WORK_DIR, 'config.json'), 'wb') as fh:
        json.dump(nn_config, fh)

    proc = reader.TextProcessor.from_file(os.path.join(WORK_DIR, 'input.txt'))
    proc.create_vocab(model_config.vocab_size)
    train_data = proc.get_vector()
    np.save(os.path.join(WORK_DIR, 'vocab.npy'), np.array(proc.id2word))
    proc.save_converted(os.path.join(WORK_DIR, 'input.conv.txt'))

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-model_config.init_scale,
                                                    model_config.init_scale)
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            m = model.Model(is_training=True, config=model_config)

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

            ckp_path = os.path.join(WORK_DIR, 'model.ckpt')
            saver.save(session, ckp_path, global_step=i)


if __name__ == "__main__":
    main()
