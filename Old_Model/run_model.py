

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import pickle
import tarfile

import os
import sys

from chatbot_utils import *
from Seq2Seq import *

def load_model(model_file):
    print("Loading models.")
    cwd = os.getcwd()+'/'
    tf = tarfile.open(model_file)
    tf.extractall(path=DATA_DIR)
    info = open(cwd+DATA_DIR+INF_FILE, 'r')
    hidden_size, e_layers, d_layers, n_words = [int(i) for i in info.readlines()]

    i2w = open(cwd+DATA_DIR+I2W_FILE, 'rb')
    w2i = open(cwd+DATA_DIR+W2I_FILE, 'rb')
    i2w_dict = pickle.load(i2w)
    w2i_dict = pickle.load(w2i)
    corpus = WordDict(dicts=[w2i_dict, i2w_dict, {}, n_words])
    w2i.close()
    i2w.close()

    encoder1 = EncoderRNN(corpus.n_words, hidden_size)
    decoder1 = DecoderRNN(hidden_size, corpus.n_words)
    if not USE_CUDA:
        encoder1.load_state_dict(torch.load(cwd+DATA_DIR+ENC_FILE,map_location=lambda storage, loc: storage))
        decoder1.load_state_dict(torch.load(cwd+DATA_DIR+DEC_FILE,map_location=lambda storage, loc: storage))
    else:
        encoder1.load_state_dict(torch.load(cwd + DATA_DIR + ENC_FILE))
        decoder1.load_state_dict(torch.load(cwd + DATA_DIR + DEC_FILE))
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()
    encoder1.eval()
    decoder1.eval()

    tf.close()

    print("Loaded models.")

    return encoder1, decoder1, corpus


def evaluate(encoder, decoder, corpus, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(corpus, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if USE_CUDA else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_INDEX]]))  # SOS
    decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_INDEX:
            decoded_words.append(EOS)
            break
        else:
            decoded_words.append(corpus.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

    return decoded_words

def converse(encoder, decoder, wd, max_length = MAX_LENGTH):
    print("Enter your message:")
    end = False
    while not end:
        msg = input()
        if "exit" in msg:
            end=True
        else:
            msg = normalize_string(msg).split(" ")
            resp = evaluate(encoder, decoder, wd, msg)
            print(clean_resp(resp,[EOS]))



if __name__ == '__main__':
    model_file = sys.argv[1]
    enc, dec, wd = load_model(model_file)
    converse(enc, dec, wd)