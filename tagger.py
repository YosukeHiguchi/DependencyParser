"""
Greedy Averaged Perceptron POS Tagger
"""

from perceptron import AveragedPerceptron
from utils import _pc

START = ['-START-', '-START2-']
END = ['-END-', '-END2-']

class PerceptronTagger(object):
    AP_MODEL_LOC = os.path.join(os.path.dirname(__file__), 'tagger.pickle')

    def __init__(self, load=True):
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        if load:
            self.load(PerceptronTagger.AP_MODEL_LOC)

    def train(self, sentences, save_loc=None, nr_iter=5):
        """
        Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.
        :param sentences: A list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        """

        self._make_tagdict(sentences)
        self.model.classes = self.classes
        prev, prev2 = START

        for iter_ in range(nr_iter):
            c = 0
            n = 0
            for words, tags in sentences:
                context = self.START + [self._normalize(w) for w in words] \
                                                                    + self.END
                for i, word in enumerate(words):
                    guess = self.tagdict.get(word)
                    if not guess:
                        feats = self._get_features(i, word, context, prev, prev2)
                        guess = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                    prev2 = prev
                    prev = guess
                    c += guess == tags[i]
                    n += 1
            random.shuffle(sentences)
            logging.info("Iter {0}: {1}/{2}={3}".format(iter_, c, n, _pc(c, n)))
        self.model.average_weights()
        # Pickle as a binary file
        if save_loc is not None:
            pickle.dump((self.model.weights, self.tagdict, self.classes),
                         open(save_loc, 'wb'), -1)
        return None

    def load(self, loc):
        """Load a pickled model."""

        try:
            w_td_c = pickle.load(open(loc, 'rb'))

        except IOError:
            msg = ("Missing tagger.pickle file.")
            raise MissingCorpusError(msg)

        self.model.weights, self.tagdict, self.classes = w_td_c
        self.model.classes = self.classes

        return None

    def _make_tagdict(self, sentences):
        """Make a tag dictionary for single-tag words."""

        counts = defaultdict(lambda: defaultdict(int))

        for sent in sentences:
            for word, tag in zip(sent[0], sent[1]):
                counts[word][tag] += 1
                self.classes.add(tag)

        freq_thresh = 20
        ambiguity_thresh = 0.97

        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            # Don't add rare words to the tag dictionary
            # Only add quite unambiguous words
            if n >= freq_thresh and (float(mode) / n) >= ambiguity_thresh:
                self.tagdict[word] = tag
