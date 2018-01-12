import json
import evaluator
import shutil
from read_util import *
import pickle
import os


SOS_token = 1
EOS_token = 2

MIN_COUNT = 5
MIN_LENGTH = 3
MAX_LENGTH = 300




class ModelInfo:
    def __init__(self, model_name):
        self.model_name = model_name
        self.dir_path = os.path.join('./MT_save_models', model_name)
        self._load_parameters()
        self._load_models()
        self._load_vocabs()

    def _load_parameters(self):
        self.train_config_path = os.path.join(self.dir_path, 'train_config.json')
        self.generate_config_path = os.path.join(self.dir_path, 'generate_config.json')
        with open(self.generate_config_path) as f:
            self.generate_parameters = json.load(f)
        with open(self.train_config_path) as f:
            self.train_parameters = json.load(f)
        assert (self.model_name == self.generate_parameters['model_name'])

    def _load_models(self):
        with open(os.path.join(self.dir_path, 'encoder.pkl'), 'rb') as f:
            self.encoder = torch.load(f)
        with open(os.path.join(self.dir_path, 'decoder.pkl'), 'rb') as f:
            self.decoder = torch.load(f)

    def _load_vocabs(self):
        with open(os.path.join(self.dir_path, 'input_vocab.pkl'), 'rb') as f:
            self.input_vocab = pickle.load(f)
        with open(os.path.join(self.dir_path, 'output_vocab.pkl'), 'rb') as f:
            self.output_vocab = pickle.load(f)
        with open(os.path.join(self.dir_path, 'info_vocab.pkl'), 'rb') as f:
            self.info_vocab = pickle.load(f)


def beam(model, decoder_input, decoder_hidden, encoder_outputs, position, year, institution, field, cites, beam_size):
    decoded_words = []
    beam_best_seqs = [list(decoder_input.data)]
    beam_last_tokens = [decoder_input]
    beam_last_probs = [1]
    for di in range(MAX_LENGTH):
        beam_temp = {}
        for i, input_token in enumerate(beam_last_tokens):
            if input_token.data[0] == EOS_token:
                decoded_seq = beam_best_seqs[i]
                decoded_words.append([model.output_vocab.vocab.index2word[id] for id in decoded_seq][1:-1])
                beam_size -= 1
                continue
            decoder_output, decoder_hidden, decoder_attention = model.decoder.forward(
                input_token, decoder_hidden, encoder_outputs, position,
                year, institution, field, cites)
            topv, topi = decoder_output.data.topk(beam_size)
            for v, j in zip(topv[0], topi[0]):
                seq = beam_best_seqs[i] + [j]
                new_seq = tuple(seq)
                beam_temp[new_seq] = beam_last_probs[i] * v / 100
        if beam_size == 0:
            break
        sort_keys = sorted(beam_temp, key=lambda x: beam_temp[x], reverse=True)
        beam_best_seqs = [list(key) for key in sort_keys[:beam_size]]
        beam_last_probs = [beam_temp[key] for key in sort_keys[:beam_size]]
        beam_last_tokens = [Variable(torch.LongTensor([seq[-1]])) for seq in beam_best_seqs]
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
    return decoded_words


def greedy(model, greedy_level, decoder_input, decoder_hidden, encoder_outputs, position, year, institution, field, cites, gold, feature):
    decoded_words = []
    decoder_attentions = torch.zeros(MAX_LENGTH + 1, MAX_LENGTH + 1)
    gold_length = len(gold.split(' '))
    for di in range(MAX_LENGTH):
        # decoder_output, decoder_hidden, decoder_attention = decoder.forward(
        #     decoder_input, decoder_hidden, encoder_outputs, position,
        #     year, institution, field, cites
        # )
        decoder_output, decoder_hidden, decoder_attention = model.decoder.forward(
            decoder_input, decoder_hidden, encoder_outputs,
            position, feature)
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        topv, topi = decoder_output.data.topk(greedy_level)

        # if output_vocab.vocab.index2word[ni] == '.' and gold_length <= di:
        #     decoded_words.append('.')
        #     break
        if topi[0][0] == EOS_token:
            break
        else:
            ni = random.choice(topi[0])
            if ni == EOS_token:
                break
            else:
                decoded_words.append(model.output_vocab.vocab.index2word[ni])
        # print(decoded_words)
        decoder_input = Variable(torch.LongTensor([ni]))
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
    #print(decoded_words)
    return decoded_words


def generation(model, input_seq, position=None, year=None, institution=None, field=None, cites=None, gold='', feature=None, greedy_level=2, beam_size=2):
    input_lengths = [len(input_seq)]
    input_seqs = [model.input_vocab.get_index_sentence(input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)
    if USE_CUDA:
        input_batches = input_batches.cuda()

    encoder_outputs, encoder_hidden = model.encoder(input_batches, input_lengths, None)

    if model.generate_parameters['decoder_start'] is None:
        decoder_input = Variable(torch.LongTensor([SOS_token]))
    else:
        decoder_input = Variable(torch.LongTensor([eval(model.generate_parameters['decoder_start'])]))

    input_info_choose = model.generate_parameters['feature_choice']

    position = Variable(torch.LongTensor([position])) if input_info_choose['position'] else None
    year = Variable(torch.LongTensor([year - model.info_vocab.start_year])) if input_info_choose['year'] else None
    institution = Variable(torch.LongTensor([model.info_vocab.institution_vocab.text2id(institution)])) if input_info_choose['institution'] else None
    field = Variable(torch.LongTensor([model.info_vocab.field_vocab.text2id(field)])) if input_info_choose['field'] else None
    cites = Variable(torch.FloatTensor([cites])) if input_info_choose['cites'] else None
    feature = Variable(torch.FloatTensor([feature])) if input_info_choose['feature'] else None

    decoder_hidden = encoder_hidden
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
    decoded_words = greedy(model, greedy_level, decoder_input, decoder_hidden, encoder_outputs,
                           position, year, institution, field, cites, gold, feature)
    return decoded_words


def build_outputs(output_text):
    dir_path = os.path.join('./MT_outputs', 'topic-title-abstract-' + abstract_model.model_name.split('-')[-1])
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        shutil.copy(title_model.train_config_path, os.path.join(dir_path, 'title_train_config.json'))
        shutil.copy(abstract_model.train_config_path, os.path.join(dir_path, 'abstract_train_config.json'))
    with open(os.path.join(dir_path, 'generate.txt'), 'w') as f:
        f.write('\r\n'.join(output_text))


def same_abstract(data, last_data):
    if last_data is None:
        return True
    return data['title'] == last_data['title'] and data['institution'] == last_data['institution'] \
           and data['field'] == last_data['field'] and data['cites'] == last_data['cites']


def create_evaluate_data_with_pos(data_sets):
    write_text = []
    hypothesis = []
    reference = []
    last_data = None
    generate_data = []
    gold_data = ''

    for data in data_sets:
        decoded_words = generation(data['title'], data['position'], data['year'], data['institution'],
                                   data['field'], data['cites'], data['gold'], data['feature'])
        if same_abstract(data, last_data):
            generate_data += decoded_words
            gold_data += data['gold'] + ' '
        else:
            write_text.append('Title: ' + data['title'] + '\n' + 'Gold: ' + data['gold'].replace(' ', '') +
                              '\n' + 'Generate: ' + ''.join(generate_data) + '\n')
            hypothesis.append(' '.join(generate_data).strip())
            reference.append(gold_data.strip())
            generate_data = decoded_words
            gold_data = data['gold']
        last_data = data
    return write_text, hypothesis, reference


if __name__ == '__main__':
    max_pos = 1
    base_year = 2013
    init_topic_data = './init_data/init_topic_data.txt'
    test_data = read_feature_data(init_topic_data)
    # random.shuffle(test_data)

    title_model = ModelInfo('topic-title-joint')
    abstract_model = ModelInfo('title-abstract-joint')

    generate_outputs = []
    for data in test_data:
        decoded_title_words = generation(title_model, data['title'], feature=data['feature'], greedy_level=1)
        temp_title = ' '.join(decoded_title_words)
        decoded_abstract_words = []
        for pos in range(max_pos):
            pos_words = generation(abstract_model, temp_title, pos, feature=data['feature'], greedy_level=1)
            decoded_abstract_words += pos_words + ['.']
        generate_text = 'Topic: ' + data['title'] + \
                        '\n' + 'Year: ' + str(int(data['feature'][0]) + base_year) + \
                        '    Authority: ' + str(int(data['feature'][1])) + '\n' +\
                        '\n' + 'Generate title: ' + ' '.join(decoded_title_words) + \
                        '\n' + 'Generate abstract: ' + ' '.join(decoded_abstract_words)

        print(generate_text)
        generate_outputs.append(generate_text)
    build_outputs(generate_outputs)
