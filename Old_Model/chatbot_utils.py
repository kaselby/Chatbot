"""
Utilities for Chatbot 
"""

import unicodedata
import re
import torch
from torch.autograd import Variable


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

TOKENS = {r'\.\.\.': ' '+ELP+' ', r'\\n': ' '+NL+' '}
RESERVED_I2W = {SOS_INDEX: SOS, EOS_INDEX: EOS, UNK_INDEX: UNK, RMVD_INDEX: RMVD,
            PAD_INDEX: PAD, ELP_INDEX: ELP, NL_INDEX: NL}
RESERVED_W2I = dict((v,k) for k,v in RESERVED_I2W.items())

DATA_DIR = "Current_Model/"
ENC_FILE = "enc.pt"
DEC_FILE = "dec.pt"
I2W_FILE = "i2w.dict"
W2I_FILE = "w2i.dict"
INF_FILE = "info.dat"
FIG_FILE = "losses.png"

USE_CUDA = torch.cuda.is_available()

MAX_LENGTH = 10

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

def clean_resp(raw_resp, rmv_tokens):
    resp = [w for w in raw_resp if not w in rmv_tokens]
    return " ".join(resp)


def indexesFromSentence(corpus, sentence):
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


def variableFromSentence(corpus, sentence):
    indexes = indexesFromSentence(corpus, sentence)
    indexes.append(EOS_INDEX)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if USE_CUDA:
        return result.cuda()
    else:
        return result


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
                indices.append(self.word2index[UNK_OLD])
        return indices

    def to_words(self, indices):
        words = []
        for index in indices:
            words.append(self.index2word[index])
        return words