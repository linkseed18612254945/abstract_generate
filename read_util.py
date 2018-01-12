import random
import torch
import os
from torch.autograd import Variable

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
USE_CUDA = False
USE_CHAR = False


class Vocab:
    def __init__(self, only_unk=False):
        self.word2index = {}
        self.word2count = {}
        if only_unk:
            self.index2word = {0: "UNK"}
            self.n_words = 1
        else:
            self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: 'UNK'}
            self.n_words = 4

    def index_info(self, info):
        if info not in self.word2index:
            self.word2index[info] = self.n_words
            self.word2count[info] = 1
            self.index2word[self.n_words] = info
            self.n_words += 1
        else:
            self.word2count[info] += 1

    def text2id(self, text):
        if text in self.word2index:
            return self.word2index[text]
        else:
            return UNK_token


class LiteratureInfo:
    def __init__(self):
        self.institution_vocab = Vocab(only_unk=True)
        self.field_vocab = Vocab(only_unk=True)
        self.start_year = 2018
        self.max_year = 1900
        self.max_position = 0

    def add_info(self, information):
        if 'institution' in information:
            self.institution_vocab.index_info(information['institution'])
        if 'field' in information:
            for filed in information['field'].split('/'):
                self.field_vocab.index_info(filed)
        if 'year' in information:
            year = int(information['year'])
            self.start_year = min([self.start_year, year])
            self.max_year = max([self.max_year, year])
        if 'position' in information:
            self.max_position = max([self.max_position, int(information['position'])])


class SeqInfo:
    def __init__(self, use_char=False):
        self.vocab = Vocab()
        self.use_char = use_char

    def index_words(self, sentence):
        if self.use_char:
            words = sentence
        else:
            words = sentence.split(' ')
        for word in words:
            self.vocab.index_info(word)

    def get_index_sentence(self, sentence):
        indexes = []
        if self.use_char:
            words = sentence
        else:
            words = sentence.split(' ')
        for word in words:
            if word in self.vocab.word2index:
                indexes.append(self.vocab.word2index[word])
            else:
                indexes.append(UNK_token)
        return indexes + [EOS_token]

    def only_choose_frequent(self, frequent=1):
        temp_word2index = {"PAD": 0, "SOS": 1, "EOS": 2, 'UNK': 3}
        temp_index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: 'UNK'}
        self.vocab.n_words = len(temp_word2index)

        for word in self.vocab.word2index:
            if self.vocab.word2count[word] > frequent:
                temp_word2index[word] = self.vocab.n_words
                temp_index2word[self.vocab.n_words] = word
                self.vocab.n_words += 1
        self.vocab.word2index = temp_word2index
        self.vocab.index2word = temp_index2word


def pad_seq(seq, max_length):
    seq += [PAD_token for _ in range(max_length - len(seq))]
    return seq


def prepare_data(data_name, use_char):
    filepath = os.path.join('./data/MT', data_name + '.txt')
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()

    # 读取句对文本与相关信息
    pairs = []
    for l in lines:
        temp = l.split('\t')
        pair = temp[:2]
        pair[0] = pair[0].lower()
        pair[1] = pair[1].lower()

        # info = {'position': temp[2], 'year': temp[3], 'cite': temp[4], 'institution': temp[5], 'author_cite': temp[6],
        #         'author_production': temp[7], 'h-index': temp[8], 'field': temp[9]}
        # info = {'institution': temp[2], 'year': temp[3], 'field': temp[4], 'position': temp[5]}
        # info = {'feature': [float(temp[2]), float(temp[3])]}
        info = {'feature': [float(temp[2]), float(temp[3])], 'position': temp[4]}

        pair.append(info)
        pairs.append(pair)

    # 创建词表
    input_vocab = SeqInfo()
    output_vocab = SeqInfo(use_char)
    info_vocab = LiteratureInfo()
    for pair in pairs:
        input_vocab.index_words(pair[0])
        output_vocab.index_words(pair[1])
        # info_vocab.add_info(pair[2])
    return input_vocab, output_vocab, info_vocab, pairs


def read_test_data(data_path):
    with open(data_path, 'r') as f:
        lines = f.read().splitlines()
    data = []
    for line in lines:
        line = line.split('\t')
        cites = (float(line[4]) + float(line[6]) + float(line[7]) + float(line[8])) / 4
        data.append({'title': line[0], 'gold': line[1], 'position': int(line[2]), 'year': int(line[3]), 'institution': line[5],
                     'field': line[9].split('/')[0], 'cites': cites})
    return data


def read_feature_data(data_path):
    with open(data_path, 'r') as f:
        lines = f.read().splitlines()
    data = []
    for line in lines:
        line = line.split('\t')
        feature = [float(line[1]), float(line[2])]
        data.append({'title': line[0], 'gold': '', 'feature': feature, 'position': None, 'year': None, 'institution': None,
                     'field': None, 'cites': None})
    return data


def random_batch_info(batch_size, input_vocab, output_vocab, info_vocab, pairs):
    input_seqs = []
    target_seqs = []
    positions = []
    years = []
    institutions = []
    fields = []
    cites = []
    features = []
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(input_vocab.get_index_sentence(pair[0]))
        target_seqs.append(output_vocab.get_index_sentence(pair[1]))
        info = pair[2]
        if 'feature' in info:
            features.append(info['feature'])
        if 'position' in info:
            positions.append(int(info['position']))
        else:
            positions.append(0)
        if 'year' in info:
            years.append(int(info['year']) - info_vocab.start_year)
        else:
            years.append(0)
        if 'institution' in info:
            institutions.append(info_vocab.institution_vocab.word2index[info['institution']])
        else:
            institutions.append(0)
        if 'field' in fields:
            fields.append(info_vocab.field_vocab.word2index[info['field'].split('/')[0]])
        else:
            fields.append(0)
        if 'cites' in cites:
            cites.append([int(info['cite']), int(info['author_cite']), int(info['author_production']), int(info['h-index'])])
        else:
            cites.append(0)

    seq_pairs = sorted(zip(input_seqs, target_seqs, positions, years, institutions, fields, cites, features),
                       key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs, positions, years, institutions, fields, cites, features = zip(*seq_pairs)

    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    positions = Variable(torch.LongTensor(positions))
    years = Variable(torch.LongTensor(years))
    institutions = Variable(torch.LongTensor(institutions))
    fields = Variable(torch.LongTensor(fields))
    cites = Variable(torch.FloatTensor(cites))

    features = Variable(torch.FloatTensor(features))
    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()
        positions = positions.cuda()
        years = years.cuda()
        institutions = institutions.cuda()
        fields = fields.cuda()
        cites = cites.cuda()
        features = features.cuda()

    return input_var, input_lengths, target_var, target_lengths,\
           positions, years, institutions, fields, cites, features


def batch_info(batch_size, input_vocab, output_vocab, info_vocab, pairs, batch_num):
    input_seqs = []
    target_seqs = []
    positions = []
    years = []
    institutions = []
    fields = []
    cites = []

    for i in range(batch_size):
        pair = pairs[batch_num * batch_size + i]
        input_seqs.append(input_vocab.get_index_sentence(pair[0]))
        target_seqs.append(output_vocab.get_index_sentence(pair[1]))
        info = pair[2]
        if 'position' in info:
            positions.append(int(info['position']))
        else:
            positions.append(0)
        if 'year' in info:
            years.append(int(info['year']) - info_vocab.start_year)
        else:
            years.append(0)
        if 'institution' in info:
            institutions.append(info_vocab.institution_vocab.word2index[info['institution']])
        else:
            institutions.append(0)
        if 'field' in fields:
            fields.append(info_vocab.field_vocab.word2index[info['field'].split('/')[0]])
        else:
            fields.append(0)
        if 'cites' in cites:
            cites.append([int(info['cite']), int(info['author_cite']), int(info['author_production']), int(info['h-index'])])
        else:
            cites.append(0)

    seq_pairs = sorted(zip(input_seqs, target_seqs, positions, years, institutions, fields, cites),
                       key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs, positions, years, institutions, fields, cites = zip(*seq_pairs)

    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    positions = Variable(torch.LongTensor(positions))
    years = Variable(torch.LongTensor(years))
    institutions = Variable(torch.LongTensor(institutions))
    fields = Variable(torch.LongTensor(fields))
    cites = Variable(torch.FloatTensor(cites))
    if USE_CUDA:
        input_var = input_var.cuda()
        target_var = target_var.cuda()
        positions = positions.cuda()
        years = years.cuda()
        institutions = institutions.cuda()
        fields = fields.cuda()
        cites = cites.cuda()

    return input_var, input_lengths, target_var, target_lengths,\
           positions, years, institutions, fields, cites