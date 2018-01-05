

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import pickle
import tarfile

import sys
import os

from chatbot_utils import *
from Seq2Seq import *



def load_model(model_file):
    print("Loading models.")
    cwd = os.getcwd()+'/'
    tf = tarfile.open(model_file)
    tf.extractall(path=DATA_DIR)
    info = open(cwd+DATA_DIR+INF_FILE, 'r')
    lns = info.readlines()
    hidden_size, e_layers, d_layers, n_words = [int(i) for i in lns[:4]]

    i2w = open(cwd+DATA_DIR+I2W_FILE, 'rb')
    w2i = open(cwd+DATA_DIR+W2I_FILE, 'rb')
    i2w_dict = pickle.load(i2w)
    w2i_dict = pickle.load(w2i)
    wd = WordDict(dicts=[w2i_dict, i2w_dict, {}, n_words])
    w2i.close()
    i2w.close()

    encoder1 = EncoderRNN(wd.n_words, hidden_size, n_layers=e_layers)
    decoder1 = LuongAttnDecoderRNN('dot', hidden_size, wd.n_words, n_layers=d_layers)
    if not USE_CUDA:
        encoder1.load_state_dict(torch.load(cwd+DATA_DIR+ENC_FILE,map_location=lambda storage, loc: storage))
        decoder1.load_state_dict(torch.load(cwd+DATA_DIR+DEC_FILE,map_location=lambda storage, loc: storage))
    else:
        encoder1.load_state_dict(torch.load(cwd + DATA_DIR + ENC_FILE))
        decoder1.load_state_dict(torch.load(cwd + DATA_DIR + DEC_FILE))
    encoder1.eval()
    decoder1.eval()


    tf.close()

    if USE_CUDA:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()

    print("Loaded models.")

    return encoder1, decoder1, wd

def evaluate(encoder, decoder, wd, input_seq, max_length=MAX_LENGTH, attention=False):
    input_lengths = [len(input_seq)]
    input_seqs = [wd.to_indices(input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

    if USE_CUDA:
        input_batches = input_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_INDEX]), volatile=True)  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    if attention:
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1)
        # Run through decoder
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs, lengths=None
            )
            decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_INDEX:
                decoded_words.append(EOS_INDEX)
                break
            else:
                decoded_words.append(ni)

            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([ni]))
            if USE_CUDA: decoder_input = decoder_input.cuda()
    else:
        # Run through decoder
        for di in range(max_length):
            decoder_output, decoder_hidden= decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_INDEX:
                decoded_words.append(EOS_INDEX)
                break
            else:
                decoded_words.append(ni)

            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([ni]))
            if USE_CUDA: decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words

def converse(encoder, decoder, wd, max_length=MAX_LENGTH):
    print("Enter your message (press q to quit):")
    end = False
    while not end:
        msg = input()
        if "q" == msg:    #convert to curses or cmd
            end = True
        else:
            msg = normalize_string(msg).split(" ")
            raw_resp = wd.to_words(evaluate(encoder, decoder, wd, msg, attention=True))
            resp = clean_resp(raw_resp, RESERVED_W2I)
            print(raw_resp)

if __name__ == '__main__':
    model_file = sys.argv[1]
    enc, dec, wd = load_model(model_file)
    converse(enc, dec, wd)
