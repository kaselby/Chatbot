from chatbot_utils import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result


class Attn(nn.Module):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, hidden, encoder_hiddens):
        max_len = encoder_hiddens.size(0)
        batch_size = encoder_hiddens.size(1)

        attn_energies = Variable(torch.zeros(batch_size, max_len))  # B x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        for j in range(max_len):
            attn_energies[:,j] = self.attn(torch.cat())

        attn_weights =

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = MAX_LENGTH

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.h_reduce = nn.Linear(2*hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_hiddens):
        batch_size = word_input.size(0)
        max_len = encoder_hiddens.size(0)

        word_embedded = self.embedding(word_input).view(1, batch_size, self.hidden_size)  # S=1 x B x N
        word_embedded = self.dropout(word_embedded)

        attn_energies = Variable(torch.zeros(batch_size, max_len))  # B x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        for j in range(max_len):
            for i in range(batch_size):
                attn_energies[:,j] = encoder_hiddens[j].dot(last_hidden)

        attn_weights = F.softmax(attn_energies)
        context = attn_weights.bmm(encoder_hiddens.transpose(0, 1))  # B x S=1 x N

        effective_hidden = torch.cat(last_hidden, context)
