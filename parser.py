from perceptron import AveragedPerceptron
from utils import DefaultList

SHIFT = 0; RIGHT = 1; LEFT = 2
MOVES = [SHIFT, RIGHT, LEFT]

class Parse(object):
    def __init__(self, n):
        self.n = n
        self.heads = [None] * (n - 1)
        self.lefts = []
        self.rights = []

        for i in range(n + 1):
            self.lefts.append(DefaultList(0))
            self.rights.append(DefaultList(0))

    def add_arc(self, head, child):
        self.heads[child] = head

        if child < head:
            self.lefts[head].append(child)

        else:
            self.rights[head].append(child)

class Parser(object):
    def __init__(self, load=True):
        self.model = AveragedPerceptron(MOVES)
        # model_dir = os.path.dirname(__file__)
        # if load:
        #     self.model.load(path.join(model_dir, 'parser.pickle'))
        self.tagger = PerceptronTagger(load)

    def parse(self, words):
        tags = self.tagger(words)
        n = len(words)
        idx = 1
        stack = [0]
        parse = Parse(n)

        while stack or idx < n:
            features = extract_features(words, tags, idx, n, stack, parse)

            scores = self.model.score(features)

            valid_moves = get_valid_moves(i, n, len(stack))

            next_move = max(valid_moves, key=lambda move: scores[move])

            idx = transition(next_move, idx, stack, parse)

        return tags, parse

    def train_one(self, itn, words, gold_tags, gold_heads):
        n = len(words)
        i = 2; stack = [1]; parse = Parse(n)
        tags = self.tagger.tag(words)

        while stack or (i + 1) < n:
            features = extract_features(words, tags, i, n, stack, parse)
            scores = self.model.score(features)
            valid_moves = get_valid_moves(i, n, len(stack))
            gold_moves = get_gold_moves(i, n, stack, parse.heads, gold_heads)
            guess = max(valid_moves, key=lambda move: scores[move])
            best = max(gold_moves, key=lambda move: scores[move])
            self.model.update(best, guess, features)
            i = transition(guess, i, stack, parse)

        return len([i for i in range(n - 1) if parse.heads[i] == gold_heads[i]])

def get_valid_moves(i, n, stack_depth):
    moves = []
    if i < n:
        moves.append(SHIFT)

    if stack_depth <= 2:
        moves.append(RIGHT)

    if stack_depth <= 1:
        moves.append(LEFT)

    return moves

def transition(move, i, stack, parse):
    global SHIFT, RIGHT, LEFT

    if move == SHFIT:
        stack.append(i)
        return i + 1

    elif move == RIGHT:
        parse.add_arc(stack[-2], stack.pop())
        return i

    elif move == LEFT:
        parse.add_arc(i, stack.pop())
        return i

    raise GrammarError("Unknown move: %d" % move)