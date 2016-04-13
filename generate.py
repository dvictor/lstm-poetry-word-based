#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import reader
import os
import sys
import re
import json
from collections import namedtuple

from model import Model


WORK_DIR = 'data-lyrics'
#WORK_DIR = 'data-eminescu'

TEMPERATURE = .7
INI_TEXT = '''green people floating
the morning has'''


def weighted_pick(a):
    a = a.astype(np.float64)
    a = a.clip(min=1e-20)
    a = np.log(a) / TEMPERATURE
    a = np.exp(a) / (np.sum(np.exp(a)))
    return np.argmax(np.random.multinomial(1, a, 1))


class Printer:
    def __init__(self):
        self.re1 = re.compile(r'^\W')
        self.prev = ''

    def print_word(self, word):
        if self.prev == '\n' or self.prev == '('\
                or word[0] == "'" or word in '.,:;)?!'\
                or (word[0] == '-' and len(word) > 1):
            s = ''
        else:
            s = ' '
        s += word
        self.prev = word
        sys.stdout.write(s)
        sys.stdout.flush()


def main(_):

    with open(os.path.join(WORK_DIR, 'vocab.npy'), 'rb') as fh:
        id2word = np.load(fh).tolist()
    word2id = dict(zip(id2word, range(len(id2word))))

    with open(os.path.join(WORK_DIR, 'config.json'), 'rb') as fh:
        d = json.load(fh)
    d['batch_size'] = 1
    d['num_steps'] = 1
    config = namedtuple('ModelConfig', d.keys())(*d.values())

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = Model(is_training=False, config=config)

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(WORK_DIR)
        saver.restore(session, ckpt.model_checkpoint_path)

        state = m.initial_state.eval()

        data = reader.TextProcessor(INI_TEXT).set_vocab(word2id).get_vector()
        sys.stdout.write(INI_TEXT)

        w_id = 0
        for w_id in data:
            x = np.zeros((1, 1), dtype=np.int32)
            x[0, 0] = w_id
            logits, state = session.run([m.logits, m.final_state], {
                m.input_data: x, m.initial_state: state})

        p = Printer()
        for i in range(100):
            x = np.zeros((1, 1), dtype=np.int32)
            x[0, 0] = w_id
            logits, state = session.run([m.logits, m.final_state], {
                m.input_data: x, m.initial_state: state})

            probs = session.run(tf.nn.softmax(logits)).flatten()
            w_id = weighted_pick(probs)
            p.print_word(id2word[w_id])

        sys.stdout.write('\n\n')


if __name__ == "__main__":
    tf.app.run()
