"""Pivotal program"""

import os
import io
import random

from parser import Parser

def train(parser, samples, nr_iter):
    model = AveragedPerceptron()

    for i in range(nr_iter):
        random.shuffle(samples)

        for features, class_ in samples:
            scores = model.predict(features)
            guess, score = max(score.items(), key=lambda i: i[1])

            if guess != class_:
                model.update(class_, guess, features)

    model.average_weights()

    return model

def get_conll_list(loc):
    n = 0
    with io.open(loc, encoding='utf8') as file_:
        sent_strs = file_.read().strip().split('\n\n')

    for sent_str in sent_strs:
        lines = [line.split() for line in sent_str.split('\n') if not line.startswith('#')]
        words = []
        tags = []
        heads = []
        labels = []

        for i, pieces in enumerate(lines):
            if len(pieces) == 4:
                word, pos, head, label = pieces

            else:
                idx, word, lemma, pos1, pos, morph, head, label, _, _2 = pieces

            if '-' in idx:
                continue

            words.append(word)
            tags.append(pos)
            heads.append(head)
            labels.append(label)

        yield words, tags, heads, labels

def main():
    curdir = os.path.dirname(os.path.abspath(__file__))
    train_loc = os.path.join(curdir, 'data', 'train.conll')

    sentences = list(get_conll_list(train_loc))
    parser = Parser(load=False)

    train(parser, sentences, 15)

if __name__ == '__main__':
    main()