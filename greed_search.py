import json
from read_util import *
import pickle
import os


SOS_token = 1

MIN_COUNT = 5
MIN_LENGTH = 3
MAX_LENGTH = 5000


def generation(input_seq, position, year=None, author_from=None, field=None, number_info=None):
    input_lengths = [len(input_seq)]
    input_seqs = [input_vocab.get_index_sentence(input_seq)]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

    if parameters['USE_CUDA']:
        input_batches = input_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True)  # SOS
    position_input = Variable(torch.LongTensor([position]))

    year_input = None if year is None else Variable(torch.LongTensor([year - info_vocab.start_year]))
    from_input = None if author_from is None else Variable(torch.LongTensor([info_vocab.from_vocab.word2index[author_from]]))
    field_input = None if field is None else Variable(torch.LongTensor([info_vocab.field_vocab.word2index[field]]))
    number_info_input = None if number_info is None else Variable(torch.FloatTensor([number_info]))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    if parameters['USE_CUDA']:
        decoder_input = decoder_input.cuda()
        position_input = position_input.cuda()
        year_input = year_input.cuda()
        field_input = field_input.cuda()
        number_info_input = number_info_input.cuda()


    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(MAX_LENGTH + 1, MAX_LENGTH + 1)

    # Run through decoder
    for di in range(MAX_LENGTH):
        # decoder_output, decoder_hidden, decoder_attention = decoder.forward(
        #     decoder_input, decoder_hidden, encoder_outputs
        # )
        decoder_output, decoder_hidden, decoder_attention = decoder.forward(
            decoder_input, decoder_hidden, encoder_outputs, position_input,
            year_input, from_input, field_input, number_info_input
        )
        decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(5)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('。')
            break
        else:
            decoded_words.append(output_vocab.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if parameters['USE_CUDA']:
            decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]


if __name__ == '__main__':
    config_path = './generate_config.json'
    with open(config_path) as f:
        parameters = json.load(f)

    dir_path = os.path.join('./save_models', parameters['model_name'])
    with open(os.path.join(dir_path, 'encoder.pkl'), 'rb') as f:
        encoder = torch.load(f)
    with open(os.path.join(dir_path, 'decoder.pkl'), 'rb') as f:
        decoder = torch.load(f)
    with open(os.path.join(dir_path, 'input_vocab.pkl'), 'rb') as f:
        input_vocab = pickle.load(f)
    with open(os.path.join(dir_path, 'output_vocab.pkl'), 'rb') as f:
        output_vocab = pickle.load(f)
    with open(os.path.join(dir_path, 'info_vocab.pkl'), 'rb') as f:
        info_vocab = pickle.load(f)
    for position in range(parameters['position']):
        decoded_words, _ = generation(parameters['input_seq'], position, parameters['year'],
                                      parameters['author_from'], parameters['field'], parameters['number_info'])
        print(' '.join(decoded_words))