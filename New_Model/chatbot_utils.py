"""
Utilities for Chatbot 
"""

import unicodedata
import re
import torch
import math
import time


SOS_INDEX = 0
EOS_INDEX = 1
UNK_INDEX = 2
RMVD_INDEX = 3
PAD_INDEX = 4
ELP_INDEX = 5
NL_INDEX = 6

UNK = "<UNK>"
RMVD = "<RMVD>"
EOS = "<EOS>"
SOS = "<SOS>"
PAD = "<PAD>"
ELP = "<ELP>"
NL = "<NL>"

UNK_OLD = "UNK"
RMVD_OLD = "RMVD"
EOS_OLD = "EOS"
SOS_OLD = "SOS"
PAD_OLD = "PAD"

TOKENS = {r'\.\.\.': ' '+ELP+' '}
RESERVED_I2W = {SOS_INDEX: SOS, EOS_INDEX: EOS, UNK_INDEX: UNK, RMVD_INDEX: RMVD,
            PAD_INDEX: PAD, ELP_INDEX: ELP, NL_INDEX: NL}
RESERVED_W2I = dict((v,k) for k,v in RESERVED_I2W.items())



DATA_DIR = "current_model/"
ENC_FILE = "enc.pt"
DEC_FILE = "dec.pt"
I2W_FILE = "i2w.dict"
W2I_FILE = "w2i.dict"
INF_FILE = "info.dat"
FIG_FILE = "losses.png"

USE_CUDA = torch.cuda.is_available()

MAX_LENGTH = 10

#helper functions for cleaning data
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    for token, flag in TOKENS.items():
        s = re.sub(token, flag, s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?<>']+", r" ", s)
    return s

def clean_resp(raw_resp, rmv_tokens=list(RESERVED_W2I.keys())):
    resp = [w for w in raw_resp if not w in rmv_tokens]
    return " ".join(resp)

#helper functions for preprocessing data
def import_data(datafile, max_n=-1):
    with open(datafile, 'r') as infile:
        lines = []
        count = 0
        for line in infile:
            if max_n > 0 and count >= max_n:
                break
            lines.append(line[:-1].split(","))
            count += 1
        return lines


def filter_lines(lines):
    count = 0
    for i in range(len(lines)):
        if len(lines[i]) > MAX_LENGTH:
            lines[i] = [RMVD]
            count += 1
    return lines, count


def prepare_data(datafile, max_n=-1, unk_thresh=5):
    print("Reading input...")
    lines = import_data(datafile, max_n=max_n)
    print("Read %s sentence lines" % len(lines))
    lines, count = filter_lines(lines)
    print("Trimmed %s sentences" % count)
    print("Counting words...")
    wd = WordDict()
    for line in lines:
        wd.add_sentence(line)
    print("Counted words:")
    print(wd.n_words)

    print("Removing infrequent words...")

    unks = wd.remove_unknowns(unk_thresh)
    print(str(len(unks)), "words removed.")

    return wd, lines

def pad_seq(seq, max_length):
    seq += [PAD_INDEX for i in range(max_length - len(seq))]
    return seq


#helper functions for printing time elapsed and such
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


#training helper functions
def indexes_from_sentence(corpus, sentence):
    indices = []
    for word in sentence:
        index = -1
        try:
            index = corpus.word2index[word]
        except KeyError:
            index = UNK_INDEX
        finally:
            indices.append(index)
    return indices


def variable_from_sentence(corpus, sentence):
    indexes = indexes_from_sentence(corpus, sentence)
    indexes.append(EOS_INDEX)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def get_training_pairs(lang, sentences):
    pairs = []
    i_rmvd = -1
    msg = None
    resp = None
    addpair = False
    print("Collecting training pairs.")
    for i in range(len(sentences)):
        if i % 5000 == 0:
            print(i, "pairs collected.")
        if not RMVD in sentences[i]:
            rep = variable_from_sentence(lang, sentences[i])
            if addpair == True:
                pairs.append([msg, rep])
            msg = rep
            addpair = True
        else:
            addpair=False
    print("Training pairs collected.")
    return pairs


class WordDict(object):
    def __init__(self, dicts=None):
        if dicts == None:
            self._init_dicts()
        else:
            self.word2index, self.index2word, self.word2count, self.n_words = dicts

    def _init_dicts(self):
        self.word2index = {}
        self.index2word = {}
        self.word2count = {}
        self.index2word.update(RESERVED_I2W)
        self.word2index.update(RESERVED_W2I)

        self.n_words = len(RESERVED_I2W)  # number of words in the dictionary

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if not word in RESERVED_W2I:
            if not word in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def remove_unknowns(self, cutoff):
        # find unknown words
        unks = []
        for word, count in self.word2count.items():
            if count <= cutoff:
                unks.append(word)

        # remove unknown words
        for word in unks:
            del self.index2word[self.word2index[word]]
            del self.word2index[word]
            del self.word2count[word]

        # reformat dictionaries so keys get shifted to correspond to removed words
        old_w2i = self.word2index
        self._init_dicts()
        for word, index in old_w2i.items():
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        self.n_words = self.n_words

        return unks

    def to_indices(self, words):
        indices = []
        for word in words:
            if word in self.word2index:
                indices.append(self.word2index[word])
            else:
                indices.append(self.word2index[UNK])
        return indices

    def to_words(self, indices):
        words = []
        for index in indices:
            words.append(self.index2word[index])
        return words