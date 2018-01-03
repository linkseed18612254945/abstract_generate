import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

USE_CUDA = False
MAX_POS = 5

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
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        for b in range(this_batch_size):
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        return F.softmax(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.squeeze().dot(encoder_output.squeeze())
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy


class LuongAttnDecoderRNNFeature(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, feature_size=2, num_features=1, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNNFeature, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.feature_size = feature_size
        self.num_features = num_features
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.position_embedding = nn.Embedding(MAX_POS, self.hidden_size + self.feature_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU((hidden_size + feature_size) * self.num_features, hidden_size, n_layers, dropout=dropout)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, positions, feature):
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)
        feature = feature.view(1, batch_size, self.feature_size)
        embedded = torch.cat([embedded, feature], 2)
        if positions is not None:
            embedded_positions = self.position_embedding(positions)
            embedded_positions = embedded_positions.unsqueeze(0)
            embedded = torch.cat([embedded, embedded_positions], 2)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        return output, hidden, attn_weights


class LuongAttnDecoderRNNInfo(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, info_vocab,
                 num_info_size=4, use_feature_num=3, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNNInfo, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.position_embedding = nn.Embedding(info_vocab.max_position + 1, hidden_size)
        self.year_embedding = nn.Embedding(info_vocab.max_year - info_vocab.start_year + 1, hidden_size)
        self.institution_embedding = nn.Embedding(info_vocab.institution_vocab.n_words, hidden_size)
        self.field_embedding = nn.Embedding(info_vocab.field_vocab.n_words, hidden_size)
        self.cites_Embedding = nn.Linear(num_info_size, hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size * (use_feature_num + 1), hidden_size, n_layers, dropout=dropout)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, positions, years, froms, fields, num_info):
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)
        if positions is not None:
            embedded_positions = self.position_embedding(positions)
            embedded_positions = embedded_positions.view(1, batch_size, self.hidden_size)
            embedded = torch.cat([embedded, embedded_positions], 2)
        if years is not None:
            embedded_years = self.year_embedding(years)
            embedded_years = embedded_years.view(1, batch_size, self.hidden_size)
            embedded = torch.cat([embedded, embedded_years], 2)
        if froms is not None:
            embedded_institution = self.institution_embedding(froms)
            embedded_institution = embedded_institution.view(1, batch_size, self.hidden_size)
            embedded = torch.cat([embedded, embedded_institution], 2)
        if fields is not None:
            embedded_fields = self.field_embedding(fields)
            embedded_fields = embedded_fields.view(1, batch_size, self.hidden_size)
            embedded = torch.cat([embedded, embedded_fields], 2)
        if num_info is not None:
            embedded_cites = self.cites_Embedding(num_info)
            embedded_cites = embedded_cites.view(1, batch_size, self.hidden_size)
            embedded = torch.cat([embedded, embedded_cites], 2)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        return output, hidden, attn_weights
