# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

def train_what(train_label_x, train_label_value):

    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)

    if train_label_x == 1:
        config.batch_size = train_label_value
    elif train_label_x == 2:
        config.embed = train_label_value
    elif train_label_x == 3:
        config.num_filters = train_label_value
    elif train_label_x == 4:
        config.learning_rate = train_label_value
    elif train_label_x == 5:
        config.dropout = train_label_value

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)


    train_loss_iter = np.zeros(config.num_epochs)
    dev_loss_iter= np.zeros(config.num_epochs)

    (train_loss_iter, dev_loss_iter) = train(config, model, train_iter, dev_iter, test_iter)

    return train_loss_iter, dev_loss_iter

if __name__ == '__main__':
    batch_size_list = [8, 16,32, 64,128]
    embedding_size_list = [64,128,256]
    hidden_size_list = [64,128,256]
    learning_rate_list = [1e-2,5e-3,1e-3,5e-4,1e-4]
    drop_out_list = [0.1,0.2,0.3,0.4,0.5]
    '''
    j = 0
    train_loss_iter = np.zeros((np.size(batch_size_list),20))
    dev_loss_iter= np.zeros((np.size(batch_size_list),20))

    for i in batch_size_list:
        train_loss_iter[j,:], dev_loss_iter[j,:] = train_what(1,i)
        j = j + 1
    np.savetxt('batch_train_loss.txt', train_loss_iter, fmt="%.5f", delimiter=',')
    np.savetxt('batch_dev_loss.txt', dev_loss_iter, fmt="%.5f", delimiter=',')
    
    j = 0
    train_loss_iter = np.zeros((np.size(embedding_size_list),20))
    dev_loss_iter= np.zeros((np.size(embedding_size_list),20))
    for i in embedding_size_list:
        train_loss_iter[j,:], dev_loss_iter[j,:] = train_what(2,i)
        j = j + 1
    np.savetxt('emb_train_loss.txt', train_loss_iter, fmt="%.5f", delimiter=',')
    np.savetxt('emb_dev_loss.txt', dev_loss_iter, fmt="%.5f", delimiter=',')
    
    j = 0
    train_loss_iter = np.zeros((np.size(hidden_size_list),20))
    dev_loss_iter= np.zeros((np.size(hidden_size_list),20))
    for i in hidden_size_list:
        train_loss_iter[j,:], dev_loss_iter[j,:] = train_what(3,i)
        j = j + 1
    np.savetxt('hidden_train_loss.txt', train_loss_iter, fmt="%.5f", delimiter=',')
    np.savetxt('hidden_dev_loss.txt', dev_loss_iter, fmt="%.5f", delimiter=',')
    '''
    print('learning_rate')
    j = 0
    train_loss_iter = np.zeros((np.size(learning_rate_list),20))
    dev_loss_iter= np.zeros((np.size(learning_rate_list),20))
    for i in learning_rate_list:
        train_loss_iter[j,:], dev_loss_iter[j,:] = train_what(4,i)
        j = j + 1
    np.savetxt('lr_train_loss.txt', train_loss_iter, fmt="%.5f", delimiter=',')
    np.savetxt('lr_dev_loss.txt', dev_loss_iter, fmt="%.5f", delimiter=',')
    '''
    print('dropout')
    j = 0
    train_loss_iter = np.zeros((np.size(drop_out_list),20))
    dev_loss_iter= np.zeros((np.size(drop_out_list),20))
    for i in drop_out_list:
        train_loss_iter[j,:], dev_loss_iter[j,:] = train_what(5,i)
        j = j + 1
    np.savetxt('drop_train_loss.txt', train_loss_iter, fmt="%.5f", delimiter=',')
    np.savetxt('drop_dev_loss.txt', dev_loss_iter, fmt="%.5f", delimiter=',')
    '''