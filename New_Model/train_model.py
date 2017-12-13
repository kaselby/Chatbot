
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

def getIndexPairs(wd, sentences):
    pairs = []
    msg = None
    resp = None
    addpair = False
    # print("Collecting pairs of index lists.")
    for i in range(len(sentences)):
        if i % 5000 == 0:
            pass  # print(i, "pairs collected.")
        if not RMVD in sentences[i]:
            resp = wd.to_indices(sentences[i])
            if addpair == True:
                pairs.append([msg, resp])
            msg = resp
            addpair = True
        else:
            addpair = False
    # print("Pairs of index lists collected.")
    return pairs

def random_batch(batch_size, wd, sentences):
    input_seqs = []
    target_seqs = []
    indexes = []
    pairs = getIndexPairs(wd, sentences)

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(pair[0])
        target_seqs.append(pair[1])

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) + 1 for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths) - 1) + [EOS_INDEX] for s in input_seqs]
    target_lengths = [len(s) + 1 for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths) - 1) + [EOS_INDEX] for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return input_var, input_lengths, target_var, target_lengths, indexes

def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder,
          encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH, clip=50.0,
          teacher_forcing_ratio=1.0, attention=False):

    #encoder_hidden = encoder.initHidden(batch_size)
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_INDEX] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()


    # Run through decoder one time step at a time
    if attention:
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t]  # Next input is current target
    else:
        for t in range(max_target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden
            )

            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t]  # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], ec, dc

#training handler
def train_epochs(wd, lines, encoder, decoder, encoder_optimizer, decoder_optimizer,
                 batch_size, n_epochs, print_every=100, plot_every=100, plot=True, attention=False):
    ecs = []
    dcs = []
    eca = 0
    dca = 0

    epoch=0
    print_loss_total = 0
    plot_loss_total = 0

    print("Beginning training...")
    start=time.time()

    while epoch < n_epochs:
        epoch += 1

        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths, index = random_batch(batch_size, wd, lines)

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer, batch_size,
            attention=attention
        )

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        dca += dc


        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (
            time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, float(print_loss_avg))
            print(print_summary)

        '''
        if epoch % evaluate_every == 0:
            evaluate_randomly()

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            # TODO: Running average helper
            ecs.append(eca / plot_every)
            dcs.append(dca / plot_every)
            ecs_win = 'encoder grad (%s)' % hostname
            dcs_win = 'decoder grad (%s)' % hostname
            vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
            vis.line(np.array(dcs), win=dcs_win, opts={'title': dcs_win})
            eca = 0
            dca = 0
        '''
def save_model(encoder, decoder, wd, out_path="", fig=None):
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
    pickle.dump(wd.index2word, i2w)
    i2w.close()
    w2i = open(w2i_out, 'wb')
    pickle.dump(wd.word2index, w2i)
    w2i.close()

    info = open(inf_out, 'w')
    info.write(str(encoder.hidden_size)+"\n"+str(encoder.n_layers)+"\n"+str(decoder.n_layers)+"\n"+str(wd.n_words))
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

def init_model(wd, n_layers, hidden_size, learning_rate=0.05, decoder_learning_ratio=5.0):
    encoder = EncoderRNN(wd.n_words, hidden_size, n_layers=n_layers)
    decoder= LuongAttnDecoderRNN('dot', hidden_size, wd.n_words, n_layers=n_layers)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

    if USE_CUDA:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    return encoder, decoder, encoder_optimizer, decoder_optimizer


def init_parser():
    parser = argparse.ArgumentParser(description='Sequence to sequence chatbot model.')
    parser.add_argument('-f', dest='datafile', action='store', default="data/formatted_cornell.txt")
    parser.add_argument('-m', dest='maxlines', action='store', default = -1)
    parser.add_argument('-e', dest='epochs', action='store', default=1)
    parser.add_argument('-hs', dest='hidden_size', action='store', default=256)
    parser.add_argument('-bs', dest='batch_size', action='store', default=32)
    parser.add_argument('-lr', dest='learning_rate', action='store', default=0.05)
    parser.add_argument('-l', dest='layers', action='store', default=2)

    args = parser.parse_args()
    return args.datafile, args.maxlines, args.hidden_size, args.layers, args.epochs, args.batch_size, args.learning_rate


if __name__ == '__main__':
    datafile, max_lines, hidden_size, n_layers, epochs, batch_size, learning_rate = init_parser()
    max_lines, hidden_size, n_layers, epochs, batch_size, learning_rate = int(max_lines), int(hidden_size), int(n_layers), \
                                                           int(epochs), int(batch_size), float(learning_rate)
    wd, lines = prepare_data(datafile, max_n=max_lines)
    encoder, decoder, enc_opt, dec_opt = init_model(wd, n_layers, hidden_size, learning_rate=learning_rate)
    train_epochs(wd, lines, encoder, decoder, enc_opt, dec_opt, batch_size, epochs, attention=True)
    save_model(encoder, decoder, wd)
