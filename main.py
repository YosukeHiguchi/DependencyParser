"""Pivotal program"""

import os
import sys
import random
from tqdm import tqdm

from parser import Parser
from utils import DefaultList


def read_conll(loc):
    for sent_str in open(loc).read().strip().split('\n\n'):
        lines = [line.split() for line in sent_str.split('\n')]
        words = DefaultList(''); tags = DefaultList('')
        heads = [None]; labels = [None]
        for i, (_, word, _, pos, _, _, head, label, _, _) in enumerate(lines):
            words.append(word)
            tags.append(pos)
            heads.append(int(head) + 1 if head != '-1' else len(lines) + 1)
            labels.append(label)
        heads.append(None)
        pad_tokens(words); pad_tokens(tags); pad_tokens(labels); pad_tokens(heads)
        yield words, tags, heads, labels


def pad_tokens(tokens):
    tokens.insert(0, '<start>')
    tokens.append('ROOT')

def train_pos(parser, sentences, nr_iter):
    parser.tagger.start_training(sentences)

    for itn in tqdm(range(nr_iter)):
        random.shuffle(sentences)

        for words, gold_tags, gold_parse, gold_label in sentences:
            parser.tagger.train_one(words, gold_tags)

    parser.tagger.model.average_weights()

def train_dep(parser, sentences, nr_iter):
    for itn in tqdm(range(nr_iter)):
        corr = 0; total = 0
        random.shuffle(sentences)

        for words, gold_tags, gold_parse, gold_label in sentences:
            corr += parser.train_one(itn, words, gold_tags, gold_parse)
            total += len(words)

        print(itn, '%.3f' % (float(corr) / float(total)))

    print('Averaging weights')

    parser.model.average_weights()

def main(model_dir, train_loc, test_loc):
    parser = Parser(load=False)
    sentences = list(read_conll(train_loc))

    train_pos(parser, sentences, nr_iter=5)
    train_dep(parser, sentences, nr_iter=15)

    c = 0
    t = 0
    gold_sents = list(read_conll(test_loc))
    for (words, tags, gold_heads, gold_labels) in tqdm(gold_sents)
        _, heads = parser.parse(words)
        for i, w in list(enumerate(words))[1:-1]:
            if gold_labels[i] in ('P', 'punct'):
                continue
            if heads[i] == gold_heads[i]:
                c += 1
            t += 1
    print (c, t, float(c)/t)

if __name__ == '__main__':
    main('./model', './data/train.conll', './data/dev.conll')