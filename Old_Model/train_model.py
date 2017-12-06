
import matplotlib.pyplot as plt
from masked_cross_entropy import *
from Seq2Seq import *
from chatbot_utils import *

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import time
import datetime
import random
import argparse
import os
import tarfile
import pickle



def get_plot(points):
    plt.figure(frameon=False)
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    return plt.gcf()

def train_iters(corpus, lines, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, plot=True):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = get_training_pairs(corpus, lines)
    n = len(training_pairs)
    criterion = nn.NLLLoss()

    print("Beginning training.")
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[(iter - 1) % n]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    if plot:
        return get_plot(plot_losses)


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_RESPONSE_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    #encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    #encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_INDEX]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_INDEX:
                break

    loss.backward()


def save_model(encoder, decoder, corpus, out_path="", fig=None):
    print("Saving models.")

    cwd = os.getcwd() + '/'

    enc_out = out_path+ENC_FILE
    dec_out = out_path+DEC_FILE
    i2w_out = out_path+I2W_FILE
    w2i_out = out_path+W2I_FILE
    inf_out = out_path+INF_FILE
    fig_out = out_path+FIG_FILE

    torch.save(encoder.state_dict(), enc_out)
    torch.save(decoder.state_dict(), dec_out)

    i2w = open(i2w_out, 'wb')
    pickle.dump(corpus.index2word, i2w)
    i2w.close()
    w2i = open(w2i_out, 'wb')
    pickle.dump(corpus.word2index, w2i)
    w2i.close()

    info = open(inf_out, 'w')
    info.write(str(encoder.hidden_size)+"\n"+str(encoder.n_layers)+"\n"+str(decoder.n_layers)+"\n"+str(corpus.n_words))
    info.close()

    if fig != None:
        fig.savefig(fig_out)

    print("Bundling models")
    t = datetime.datetime.now()
    timestamp = str(t.day) + "_" + str(t.hour) + "_" + str(t.minute)
    tf = tarfile.open(cwd+out_path +"s2s_" + timestamp + ".tar", mode='w')
    tf.add(enc_out)
    tf.add(dec_out)
    tf.add(i2w_out)
    tf.add(w2i_out)
    tf.add(inf_out)
    if fig != None:
        tf.add(fig_out)
    tf.close()

    os.remove(enc_out)
    os.remove(dec_out)
    os.remove(i2w_out)
    os.remove(w2i_out)
    os.remove(inf_out)
    if fig != None:
        os.remove(fig_out)

    print("Finished saving models.")


def init_model(wd, n_layers, hidden_size):
    encoder = EncoderRNN(wd.n_words, hidden_size, n_layers=n_layers)
    decoder= DecoderRNN(hidden_size, wd.n_words, n_layers=n_layers)

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    return encoder, decoder


def init_parser():
    parser = argparse.ArgumentParser(description='Sequence to sequence chatbot model.')
    parser.add_argument('-f', dest='datafile', action='store', default="data/formatted_cornell.txt")
    parser.add_argument('-m', dest='maxlines', action='store', default = -1)
    parser.add_argument('-i', dest='iters', action='store', default=1)
    parser.add_argument('-hs', dest='hidden_size', action='store', default=256)
    parser.add_argument('-lr', dest='learning_rate', action='store', default=0.05)
    parser.add_argument('-l', dest='layers', action='store', default=2)

    args = parser.parse_args()
    return args.datafile, args.maxlines, args.hidden_size, args.layers, args.iters, args.learning_rate


if __name__ == '__main__':
    datafile, max_lines, hidden_size, n_layers, iters, batch_size, learning_rate = init_parser()
    max_lines, hidden_size, n_layers, iters, batch_size, learning_rate = int(max_lines), int(hidden_size), int(n_layers), \
                                                           int(iters), int(batch_size), float(learning_rate)
    wd, lines = prepare_data(datafile, max_n=max_lines)
    encoder, decoder, enc_opt, dec_opt = init_model(wd, n_layers, hidden_size, learning_rate=learning_rate)
    plot = train_iters(wd, lines, encoder, decoder, iters)
    save_model(encoder, decoder, wd, fig=plot)
