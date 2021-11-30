# coding: UTF-8
from operator import le, ne
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import matplotlib.pyplot as plt
import random
import math
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

#初始化种群
def init_population(p1,p2,p3,p4,p5,population_size, parameter_size):
    
    populations = np.zeros((population_size,parameter_size))
    population = np.zeros(parameter_size)
    i = 0
    while i < population_size:
        population[0] = random.choice(p1)
        population[1] = random.choice(p2)
        population[2] = random.choice(p3)
        population[3] = random.choice(p4)
        population[4] = random.choice(p5)
        populations[i,:] = population
        i = i + 1
    print(populations)
    return populations

#遍历所有fit_value,计算sum of fit_value
def sum(fit_value):
    total = 0
    for v in fit_value:
        total += v
    return total

#轮盘模型计算每个个体遗传的概率
def select(pop, fit_value):
    new_fit_value = np.zeros(len(fit_value))
    total_fit = sum(fit_value)

    for i in range(len(fit_value)):
        if i == 0:
            new_fit_value[i] = (fit_value[i] / total_fit)
        else:
            new_fit_value[i] = (fit_value[i] / total_fit) + new_fit_value[i - 1]

    print(new_fit_value)
    #产生pop数量个随机数(0~1之间)
    random_p = []
    for i in range(len(pop)):
        random_p.append(random.random())
    print(random_p)
    new_pop = []#初始化遗传至下一代的个体
    # 转轮盘选择法
    #遍历每个随机数
    for i in range(len(random_p)):
        #检测随机数所在的遗传概率分布函数对应的区间
        for j in range(len(new_fit_value)):
            if random_p[i] <= new_fit_value[j]:
                new_pop.append(pop[j,:])#找到对应的区间对应的个体，退出内层循环
                print(j + 1)
                break

    return np.array(new_pop)

def value2byte(parameter, value):
    #batch_size,embedding_size,hidden_size定义为8bit长度的染色体，1-256对应0x00-0xff
    if parameter == 0 or parameter == 1 or parameter == 2:
        byte = np.zeros(8)
        for i in range(8):
            byte[i] = (value - 1) // pow(2,7 - i)
            value = value - byte[i] * pow(2,7 - i)
        return byte
    if parameter == 3:
        byte = np.zeros(7)
        value = value * 100000
        print('here')
        for i in range (7):
            byte[i] = (value - 1) // pow(2, 6 - i)
            value = value - byte[i] * pow(2,6 - i)
        return byte
    if parameter == 4:
    #dropout 定义为3bit的染色体，0.1-0.8对应000-111
        byte = np.zeros(3)
        value = value * 10
        for i in range(3):
            byte[i] = (value - 1) // pow(2, 2-i)
            value = value - byte[i] * pow(2, 2 - i)
        return byte

def byte2value(parameter, byte):
    #batch_size,embedding_size,hidden_size定义为8bit长度的染色体，1-256对应0x00-0xff
    if parameter == 0 or parameter == 1 or parameter == 2:
        value = 0
        for i in range(len(byte)):
            value = value + byte[i] * pow(2, 7 - i)
        return value + 1
    if parameter == 3:
        value = 0
        for i in range(len(byte)):
            value = value + pow(2, 6 -i)
        value = (value + 1) / 100000
        return value
    if parameter == 4:
    #dropout 定义为3bit的染色体，0.1-0.8对应000-111
        value = 0
        for i in range(len(byte)):
            value = value + byte[i] * pow(2, 2 - i)
        return (value + 1) / 10

#个体交配
def crossover(pop, pc):
    pop_len = len(pop)
    #遍历所有遗传至下一代的个体，每两个个体互相交换染色体片段
    for i in range(0, pop_len - 1, 2):
        #每个染色体片段都可能发生互换
        for j in range(5):
            if j == 0 or j == 1 or j == 2 :
                len_para = 7
            elif j == 3:
                len_para = 6
            elif j == 4:
                len_para = 2
            r = random.random()#产生随机数(0~1)，决定是否发生交叉
            print(j)
            print(r)
            if(r < pc):#随机数小于交叉概率，则发生交叉
                cpoint = random.randint(0,len_para)#产生随机数(0~len_para),决定发生交叉的位置
                temp1 = []
                temp2 = []
                byte1 = value2byte(j,pop[i][j])
                print(byte1)
                byte2 = value2byte(j,pop[i + 1][j])
                print(byte2)
                temp1.extend(byte1[0:cpoint])#存放pop[i]交叉点前的染色体片段
                temp1.extend(byte2[cpoint:len_para])#接上pop[i + 1]的交叉点后的染色体片段
                temp2.extend(byte2[0:cpoint])#存放pop[i + 1]交叉点前的染色体片段
                temp2.extend(byte1[cpoint:len_para])#接上pop[i]的交叉点后的染色体片段
                pop[i][j] = byte2value(j,temp1)
                pop[i+1][j] = byte2value(j,temp2)
    print(pop)

#个体变异
def mutation(pop, pm):
    pop_len = len(pop)
    #遍历每个新个体，决定是否发生变异
    for i in range(pop_len):
        for j in range(5):
            if j == 0 or j == 1 or j == 2 :
                len_para = 7
            elif j == 3:
                len_para = 6
            elif j == 4:
                len_para = 2
            r = random.random()#随机产生一个数(0~1)
            if r < pm:#通过随机数决定该个体是否发生变异
                mpoint = random.randint(0, len_para)#产生随机数决定发生变异的染色体位点
                #变异改变该染色体的变异位点的二进制数，1变为0;0变为1
                byte = value2byte(j,pop[i][j])
                if byte[mpoint] == 1:
                    byte[mpoint] = 0
                else:
                    byte[mpoint] = 1
                pop[i][j] = byte2value(j,byte)
    print(pop)

def cal_accuracy(population):
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

    config.batch_size = int(population[0])
    config.embed = int(population[1])
    config.num_filters = int(population[2])
    config.learning_rate = population[3]
    config.dropout = population[4]

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
    test_acc = train(config, model, train_iter, dev_iter, test_iter)
    return test_acc

if __name__ == '__main__':
    batch_size_list = [8,16,32,64,128]
    embedding_size_list = [64,128,256]
    hidden_size_list = [64,128,256]
    learning_rate_list = [1e-2,5e-3,1e-3,5e-4,1e-4]
    drop_out_list = [0.1,0.2,0.3,0.4,0.5]

    pm = 0.3
    pc = 0.8
    iter_num = 10

    pop = init_population(batch_size_list,embedding_size_list,hidden_size_list,learning_rate_list,drop_out_list,10,5)
    
    j = 0
    for j in range(iter_num):
        print('this is the %d generation'%j)
        print(pop)
        fit_value = []
        best_pop = []
        best_acc = 0
        for i in range(len(pop)):
            fit_value_i = cal_accuracy(pop[i,:])
            print('this is %d pop' %i )
            if fit_value_i > best_acc:
                best_pop = pop[i,:]
                best_acc = fit_value_i
            fit_value.append(fit_value_i)
        print('best_parameters:',best_pop)
        print('best_acc:', best_acc)
        
        pop = select(pop, fit_value)
        crossover(pop,pc)
        mutation(pop,pm)


    

