import torch
import torch.nn as nn
from torch import autograd, optim
import random
from net import Config, Generator, Discriminator
from dataset import Traffic_Dataset
from utils import *
import visualization


def prepare_generator_batch(samples, start_letter=0, gpu=False):
    batch_size, seq_len = samples.size()
    inp = torch.zeros(batch_size, seq_len)
    target = samples
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len - 1]
    inp = autograd.Variable(inp).type(torch.LongTensor)
    target = autograd.Variable(target).type(torch.LongTensor)
    if gpu:
        return inp.cuda(), target.cuda()
    return inp, target


def prepare_discriminator_batch(pos_samples, neg_samples, gpu=False):
    inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[pos_samples.size()[0]:] = 0
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]
    if gpu:
        return inp.cuda(), target.cuda()
    return inp, target


def train_generator_MLE(g, g_opt, dataset, epochs, n_samples, batch_size, cuda):
    losses = []
    for _ in range(epochs):
        total_loss = 0
        for i in range(0, n_samples, batch_size):
            idx = random.randrange(len(dataset))
            inp, target = prepare_generator_batch(dataset[idx], 0, cuda)
            g_opt.zero_grad()
            loss = g.batchNLLLoss(inp, target)
            loss.backward()
            g_opt.step()
            total_loss += loss.data.item()
        total_loss = total_loss * batch_size / n_samples
        losses.append(total_loss)
        print(f'pre_train g loss: {total_loss}')
    return losses


def get_success_rate(g_batch, dataset, service_list, protocol_service_dict, service_port_dict, transform=decode_feature_indices):
    total_nums = g_batch.shape[0]
    valid_nums = 0
    for i in range(total_nums):
        flow_dict = transform(dataset, g_batch[i])
        if flow_dict is not None and judge_protocol_port(flow_dict, service_list, protocol_service_dict, service_port_dict):
            valid_nums += 1
    success_rate = valid_nums / total_nums
    print(f'success rate is {success_rate}')
    return success_rate


def train_generator_PG(g, g_opt, d, num_batches, batch_size, cuda, dataset, service_list, protocol_service_dict, service_port_dict):
    total_loss = 0
    for batch in range(num_batches):
        s = g.sample(batch_size)
        inp, target = prepare_generator_batch(s, 0, cuda)
        reward = d.batchClassify(target)
        success_rate = get_success_rate(s, dataset, service_list, protocol_service_dict, service_port_dict)
        factor_size = 1

        g_opt.zero_grad()
        pg_loss = g.batchPGLoss(inp, target, reward)
        pg_loss = pg_loss - success_rate*factor_size
        pg_loss.backward()
        g_opt.step()
        total_loss += pg_loss.data.item()
    avg_loss = total_loss / num_batches
    print(f'g loss: {avg_loss}')
    return avg_loss


def train_discriminator(d, d_opt, dataset, g, d_steps, epochs, n_samples, batch_size, cuda):
    criterion = nn.BCELoss()
    total_loss = 0
    for d_step in range(d_steps):
        for _ in range(epochs):
            for i in range(0, n_samples, batch_size):
                idx = random.randrange(len(dataset))
                real_samples = dataset[idx]
                fake_samples = g.sample(batch_size)
                inp, target = prepare_discriminator_batch(real_samples, fake_samples, cuda)
                d_opt.zero_grad()
                out = d.batchClassify(inp)
                loss = criterion(out, target)
                loss.backward()
                d_opt.step()
                total_loss += loss.data.item()
        avg_loss = (total_loss * batch_size) / (d_steps * epochs * n_samples)
        print(f'd_loss:{avg_loss}')
        return avg_loss


def generate_traffic(g, dataset, num_traffic, path, transform):
    with open(path, 'w') as file:
        valid_traffic_count = 0 # number of network traffic generated
        while valid_traffic_count < num_traffic:
            samples = g.sample(1)
            feature_dict = transform(dataset, samples[0])
            if feature_dict is not None:
                line = ','.join(str(feature_dict[feature]) for feature in feature_dict.keys())
                file.write(line)
                file.write('\n')
                valid_traffic_count += 1


def pre_training(g, d, g_opt, d_opt, dataset, n_samples, batch_size, cuda, store=True, pre_train_generate=False):
    print('start pretraining')
    pre_g_losses = train_generator_MLE(g, g_opt, dataset, 100, n_samples, batch_size, cuda)
    train_discriminator(d, d_opt, dataset, g, 50, 3, n_samples, batch_size, cuda)

    # save pre-trained networks' parameters
    record_losses('./target/pre_g_losses.csv', pre_g_losses)
    if store:
        torch.save(g.state_dict(), './conf/pre_g.pth')
        torch.save(d.state_dict(), './conf/pre_d.pth')

    if pre_train_generate:
        generate_traffic(g, dataset, 20000, './target/pre-traffic-1.csv', decode_feature_indices)


def training(g, d, g_opt, d_opt, dataset, TRAIN_EPOCHS, n_samples, batch_size, cuda, service_list, protocol_service_dict, service_port_dict, load=False):
    # load = True if skip pre-training
    if load:
        g.load_state_dict(torch.load('./conf/pre_g.pth'))
        d.load_state_dict(torch.load('./conf/pre_d.pth'))
        print('load pre-train models')
    print('start training')

    d_losses = []
    g_losses = []
    for epoch in range(TRAIN_EPOCHS):
        print(f'Epochs: {epoch}')
        if epoch < 50:
            g_n_batches = 1
        else:
            g_n_batches = 10
        g_loss = train_generator_PG(g, g_opt, d, g_n_batches, batch_size, cuda, dataset, service_list, protocol_service_dict, service_port_dict)
        d_loss = train_discriminator(d, d_opt, dataset, g, 5, 3, n_samples, batch_size, cuda)
        d_losses.append(d_loss)
        g_losses.append(g_loss)
        print(f'---------------------------------------------------------------------------')

    print('save parameters')
    torch.save(g.state_dict(), './conf/g.pth')
    torch.save(d.state_dict(), './conf/d.pth')

    print('Generate traffic')
    generate_traffic(g, dataset, 20000, './target/traffic-1.csv', decode_feature_indices)

    print('record loss')
    record_losses('./target/d_losses.csv', d_losses)
    record_losses('./target/g_losses.csv', g_losses)


def visualize(r_columns, f_columns):
    r_traffic_path = './data/train.csv'
    f_traffic_path = './target/traffic-1.csv'
    visualization.plot_shared_shaded([r_traffic_path, f_traffic_path], [False, True], r_columns, f_columns)

    visualization.plot_losses('./target/d_losses.csv', './target/g_losses.csv')


def run_seq_gan():
    config = Config()
    n_samples = config.get('n_samples')
    batch_size = config.get('batch_size')
    gen_embedding_dim = config.get('gen_embedding_dim')
    gen_hidden_dim = config.get('gen_hidden_dim')
    dis_embedding_dim = config.get('dis_embedding_dim')
    dis_hidden_dim = config.get('dis_hidden_dim')
    dataset_features = config.get('dataset_features')
    dataset_dtypes = config.get('dataset_dtypes')
    generated_features = config.get('generated_features')
    service_list = config.get('service_list')
    protocol_service_dict = config.get('protocol_service_dict')
    service_port_dict = config.get('service_port_dict')
    file_path = config.get('file_path')
    CUDA = torch.cuda.is_available()

    dataset = Traffic_Dataset(file_path, dataset_features, dataset_dtypes, generated_features,
                              batch_size=batch_size,
                              transform=build_input_indices)
    vocab_dim = dataset.vocabulary_length
    max_seq_len = dataset.max_seq_length
    train_epochs = 100

    g = Generator(gen_embedding_dim, gen_hidden_dim, vocab_dim, max_seq_len, CUDA)
    d = Discriminator(dis_embedding_dim, dis_hidden_dim, vocab_dim, max_seq_len, CUDA)
    if CUDA:
        g.cuda()
        d.cuda()
    g_opt = optim.Adam(g.parameters())
    d_opt = optim.Adagrad(d.parameters())

    pre_training(g, d, g_opt, d_opt, dataset, n_samples, batch_size, CUDA)
    training(g, d, g_opt, d_opt, dataset, train_epochs, n_samples, batch_size, CUDA, service_list, protocol_service_dict, service_port_dict)
    visualize(dataset_features, generated_features)


run_seq_gan()


