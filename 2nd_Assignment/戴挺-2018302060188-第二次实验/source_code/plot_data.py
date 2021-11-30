import numpy as np
import matplotlib.pyplot as plt

batch_train_loss = np.loadtxt('batch_train_loss.txt', delimiter = ',')
batch_dev_loss = np.loadtxt('batch_dev_loss.txt', delimiter = ',')

emb_train_loss = np.loadtxt('emb_train_loss.txt', delimiter = ',')
emb_dev_loss = np.loadtxt('emb_dev_loss.txt', delimiter = ',')

hidden_train_loss = np.loadtxt('hidden_train_loss.txt', delimiter = ',')
hidden_dev_loss = np.loadtxt('hidden_dev_loss.txt', delimiter = ',')

lr_train_loss = np.loadtxt('lr_train_loss.txt', delimiter = ',')
lr_dev_loss = np.loadtxt('lr_dev_loss.txt', delimiter = ',')

drop_train_loss = np.loadtxt('drop_train_loss.txt', delimiter = ',')
drop_dev_loss = np.loadtxt('drop_dev_loss.txt', delimiter = ',')

epoch = np.linspace(1,20,20)
batch_size_list = [8,16,32,64,128]
embedding_size_list = [64,128,256]
hidden_size_list = [64,128,256]
learning_rate_list = [1e-2,5e-3,1e-3,5e-4,1e-4]
drop_out_list = [0.1,0.2,0.3,0.4,0.5]


for i in range(len(batch_train_loss)):
    plt.plot(epoch, batch_train_loss[i,:], label = batch_size_list[i])
plt.legend(loc='upper right')
plt.title('batch_size(train_loss)')
plt.show()

for i in range(len(batch_dev_loss)):
    plt.plot(epoch, batch_dev_loss[i,:], label = batch_size_list[i])
plt.legend(loc='upper right')
plt.title('batch_size(dev_loss)')
plt.show()

for i in range(len(emb_train_loss)):
    plt.plot(epoch, emb_train_loss[i,:], label = embedding_size_list[i])
plt.legend(loc='upper right')
plt.title('embedding_size(train_loss)')
plt.show()

for i in range(len(emb_dev_loss)):
    plt.plot(epoch, emb_dev_loss[i,:], label = embedding_size_list[i])
plt.legend(loc='upper right')
plt.title('embedding_size(dev_loss)')
plt.show()

for i in range(len(hidden_train_loss)):
    plt.plot(epoch, hidden_train_loss[i,:], label = hidden_size_list[i])
plt.legend(loc='upper right')
plt.title('hidden_size(train_loss)')
plt.show()

for i in range(len(hidden_dev_loss)):
    plt.plot(epoch, hidden_dev_loss[i,:], label = hidden_size_list[i])
plt.legend(loc='upper right')
plt.title('hidden_size(dev_loss)')
plt.show()

for i in range(len(drop_train_loss)):
    plt.plot(epoch, drop_train_loss[i,:], label = drop_out_list[i])
plt.legend(loc='upper right')
plt.title('drop_out(train_loss)')
plt.show()

for i in range(len(drop_dev_loss)):
    plt.plot(epoch, drop_dev_loss[i,:], label = drop_out_list[i])
plt.legend(loc='upper right')
plt.title('drop_out(dev_loss)')
plt.show()

for i in range(len(lr_train_loss)):
    plt.plot(epoch, lr_train_loss[i,:], label = learning_rate_list[i])
plt.legend(loc='upper right')
plt.title('learning_rate(train_loss)')
plt.show()

for i in range(len(lr_dev_loss)):
    plt.plot(epoch, lr_dev_loss[i,:], label = learning_rate_list[i])
plt.legend(loc='upper right')
plt.title('learning_rate(dev_loss)')
plt.show()
