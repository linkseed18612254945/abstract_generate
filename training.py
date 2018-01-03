from read_util import *
from util import *
from model import *
from torch import optim
from masked_cross_entropy import *
import time
import torch
import pickle
import os
import shutil
import json

USE_CUDA = False
SOS_token = 1


def load_config(config_path='./train_config.json'):
    with open(config_path) as f:
        parameters = json.load(f)
    dir_path = os.path.join('./MT_save_models', parameters['model_name'])
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return parameters, dir_path


def train(parameters, input_batches, input_lengths, target_batches, target_lengths,
          position, year, institution, field, cite, feature,
          encoder, decoder, encoder_optimizer, decoder_optimizer):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    target_batches = target_batches[:parameters['max_seq_length']]
    # 构造decoder的输入
    if parameters['decoder_start'] is None:
        decoder_input = Variable(torch.LongTensor([SOS_token] * parameters['batch_size']))
    else:
        decoder_input = eval(parameters['decoder_start'])
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    max_target_length = max(target_lengths)
    all_decoder_outputs = None
    # all_decoder_outputs = Variable(torch.zeros(max_target_length, parameters['batch_size'], decoder.output_size))
    if USE_CUDA:
        target_batches = target_batches.cuda()
        decoder_input = decoder_input.cuda()
        # all_decoder_outputs = all_decoder_outputs.cuda()

    input_info_choose = parameters['feature_choice']
    position = position if input_info_choose['position'] else None
    year = year if input_info_choose['year'] else None
    institution = institution if input_info_choose['institution'] else None
    field = field if input_info_choose['field'] else None
    cite = cite if input_info_choose['cites'] else None
    feature = feature if input_info_choose['feature'] else None
    # 进行decoder端的前向传播
    for t in range(max_target_length):
        if t >= parameters['max_seq_length']:
            break
        decoder_output, decoder_hidden, decoder_attn = decoder.forward(
            decoder_input, decoder_hidden, encoder_outputs,
            position, feature)
        if t == 0:
            all_decoder_outputs = decoder_output.unsqueeze(0)
        else:
            all_decoder_outputs = torch.cat([all_decoder_outputs, decoder_output.unsqueeze(0)], 0)
        decoder_input = target_batches[t]

    # 计算loss值并进行反向传播
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),
        target_batches.transpose(0, 1).contiguous(),
        target_lengths
    )
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]


def main(load_model_name=None):
    if load_model_name is None:
        config_path = './train_config.json'
        parameters, dir_path = load_config(config_path)
        print(parameters)
        generate_parameters = {"model_name": parameters["model_name"],
                               "data_name": parameters["data_name"].split('_')[0] + '_test',
                               "decoder_start": parameters["decoder_start"],
                               "feature_choice": parameters["feature_choice"],
                               "use_char": parameters["use_char"],
                               "with_pos": parameters["with_pos"]}
        shutil.copy(config_path, os.path.join(dir_path, 'train_config.json'))
        with open(os.path.join(dir_path, 'generate_config.json'), 'w') as f:
            json.dump(generate_parameters, f)
        input_seq_class, output_seq_class, info_vocab, pairs = prepare_data(parameters['data_name'], parameters['use_char'])
        with open(os.path.join(dir_path, 'input_vocab.pkl'), 'wb') as f:
            pickle.dump(input_seq_class, f)
        with open(os.path.join(dir_path, 'output_vocab.pkl'), 'wb') as f:
            pickle.dump(output_seq_class, f)
        with open(os.path.join(dir_path, 'info_vocab.pkl'), 'wb') as f:
            pickle.dump(info_vocab, f)
        use_feature_num = [1 for key in parameters['feature_choice'] if parameters['feature_choice'][key]].count(1)
        encoder = EncoderRNN(input_seq_class.vocab.n_words, parameters['hidden_size'], parameters['n_layers'], dropout=parameters['dropout'])
        decoder = LuongAttnDecoderRNNFeature(parameters['attn_model'], parameters['hidden_size'], output_seq_class.vocab.n_words,
                                             feature_size=len(pairs[0][2]['feature']), num_features=use_feature_num,
                                             n_layers=parameters['n_layers'], dropout=parameters['dropout'])
        if USE_CUDA:
            encoder.cuda()
            decoder.cuda()
    else:
        dir_path = os.path.join('./MT_save_models', load_model_name)
        with open(os.path.join(dir_path, 'encoder.pkl'), 'rb') as f:
            encoder = torch.load(f)
        with open(os.path.join(dir_path, 'decoder.pkl'), 'rb') as f:
            decoder = torch.load(f)
        with open(os.path.join(dir_path, 'train_config.json')) as f:
            parameters = json.load(f)
        input_seq_class, output_seq_class, info_vocab, pairs = prepare_data(parameters['data_name'],
                                                                            parameters['use_char'])

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=parameters['learning_rate'])
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=parameters['learning_rate'])

    # 初始化部分参数
    plot_losses = []
    print_loss_total = 0
    start_time = time.time()
    epoch = 0
    batch_num = len(pairs) // parameters['batch_size']

    # 开始训练

    while epoch < parameters['n_epochs']:
        epoch += 1
        # 随机获取训练用的mini batch
        if parameters['train_with_one_batch']:
            input_batches, input_lengths, target_batches, target_lengths, \
            position_batches, year_batches, institution_batches, field_batches, cites_batches, feature_batches \
                = random_batch_info(parameters['batch_size'], input_seq_class, output_seq_class, info_vocab, pairs)
            loss = train(parameters, input_batches, input_lengths, target_batches, target_lengths,
                         position_batches, year_batches, institution_batches, field_batches, cites_batches, feature_batches,
                         encoder, decoder, encoder_optimizer, decoder_optimizer)
            print_loss_total += loss
        # 完整训练所有batch
        else:
            epoch_loss = 0
            random.shuffle(pairs)
            for i in range(batch_num):
                input_batches, input_lengths, target_batches, target_lengths, \
                position_batches, year_batches, institution_batches, field_batches, cites_batches, feature_batches \
                    = batch_info(parameters['batch_size'], input_seq_class, output_seq_class, info_vocab, pairs, i)
                loss = train(parameters, input_batches, input_lengths, target_batches, target_lengths,
                             position_batches, year_batches, institution_batches, field_batches, cites_batches, feature_batches,
                             encoder, decoder, encoder_optimizer, decoder_optimizer)
                epoch_loss += loss
            epoch_loss /= batch_num

        # 输出训练信息并记录损失值
        if epoch % parameters['print_every'] == 0:
            print_loss_avg = print_loss_total / parameters['print_every']
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (
                time_since(start_time, epoch / parameters['n_epochs']), epoch, (epoch / parameters['n_epochs']) * 100,
                print_loss_avg)
            plot_losses.append(str(print_loss_avg))
            print(print_summary)

        # 保存模型
        if epoch % parameters['save_every'] == 0 or epoch == parameters['n_epochs'] - 1:
            torch.save(encoder, os.path.join(dir_path, 'encoder.pkl'))
            torch.save(decoder, os.path.join(dir_path, 'decoder.pkl'))
            with open(os.path.join(dir_path, 'loss.txt'), 'a') as f:
                f.write('\n'.join(plot_losses))


if __name__ == '__main__':
    main('title-abstract-joint')