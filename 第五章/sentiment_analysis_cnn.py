import os
import argparse
import datetime
import torch
import torchtext.data as data
# from data_loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(123)  # 设置随机种子
random.seed(123)

def load_param():
    parser = argparse.ArgumentParser(description='TextCNN text classifier')
    # data
    parser.add_argument('-train-file', type=str, default='./data/rt-polarity-train.txt', help='')
    parser.add_argument('-dev-file', type=str, default='./data/rt-polarity-dev.txt', help='')
    parser.add_argument('-test-file', type=str, default='./data/rt-polarity-test.txt', help='')
    parser.add_argument('-min-freq', type=int, default=1, help='')
    parser.add_argument('-epochs-shuffle', type=bool, default=True, help='')
    # model
    parser.add_argument('-embed-dim', type=int, default=128, help='')
    parser.add_argument('-kernel-num', type=int, default=100, help='')
    parser.add_argument('-max-norm', type=float, default=5.0, help='')
    parser.add_argument('-dropout', type=float, default=0.5, help='')

def load_data(config):
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    # 读取数据集 DataLoader为自定义的类，继承torchtext.Dataset
    train_data, dev_data, test_data = DataLoader.splits("", config.train_file, config.dev_file,
                                                        config.test_file, text_field, label_field)
    # 建立词典
    text_field.build_vocab(train_data.text, min_freq=config.min_freq)
    label_field.build_vocab(train_data.label)
    # 获取迭代器 迭代器返回按模型所需格式的数据
    train_iter, dev_iter, test_iter = data.Iterator.splits((train_data, dev_data, test_data),
                                                           batch_sizes=(config.batch_size, len(dev_data), len(test_data)),
                                                           device=-1, repeat=False, shuffle=config.epochs_shuffle, sort=False)
    # 更新超参数
    config.embed_num = len(text_field.vocab)
    config.class_num = len(label_field.vocab) - 1
    config.unkId = text_field.vocab.stoi['<unk>']
    config.paddingId = text_field.vocab.stoi['<pad>']
    print("len(text_field.vocab) {}\nlen(label_field.vocab) {}".format(config.embed_num, config.class_num))
    config.save_dir = os.path.join(""+config.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.isdir(config.save_dir):
        os.makedirs(config.save_dir)

    return train_iter, dev_iter, test_iter

def load_model(config):
    model = None
    if config.CNN:
        print("Initializing CNN model...")
        model = CNN(config)
    elif config.Deep_CNN:
        print("Initializing Deep_CNN model...")
        model = Deep_CNN(config)
    assert model is not None
    # 是否使用GPU
    if config.cuda is True:
        model = model.cuda()
    return model

class CNN(nn.Module):
    # 初始化CNN模型
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args

        V = args.embed_num  # 词表大小
        D = args.embed_dim  # 词向量维度
        class_num = args.class_num  # 分类数目
        conv_in = 1         # in_channels=1
        conv_out = args.kernel_num  # 过滤器的个数
        kernel_sizes = [int(x) for x in args.kernel_sizes.split(',')]  # 卷积核大小
        # 定义词嵌入
        self.embed = nn.Embedding(V, D, max_norm=args.max_norm, scale_grad_by_freq=True)
        # 定于dropout层 每次forward时，随机将输入张量中部分元素设置为0
        self.dropout = nn.Dropout(args.dropout)
        # 定义卷积层
        self.convs1 = nn.ModuleList([nn.Conv2d(conv_in, conv_out, (K, D)) for K in kernel_sizes])
        ''' 相当于
        self.conv1-[size3] = nn.Conv2d(Ci, Co, (3, D))
        self.conv1-[size4] = nn.Conv2d(Ci, Co, (4, D))
        self.conv1-[size5] = nn.Conv2d(Ci, Co, (5, D))
        '''
        # 定义全连接层
        layer_in = len(kernel_sizes) * conv_out
        half = len(kernel_sizes) * conv_out // 2
        self.hidden = nn.Linear(layer_in, half)
        self.out = nn.Linear(half, class_num)

    def forward(self, x):
        x = self.embed(x)   # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]    # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)  # (N， Co*len(Ks))
        '''
        相当于
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''

        x = self.dropout(x)  # (N, conv_out * len(Kernel_sizes))
        x = self.hidden(x)
        logit = self.out(x)
        return logit         # (N, Class_num)

def train(train_iter, dev_iter, test_iter, model, args):
    # Define optimizer, Adam or SGD
    optimizer = None
    if args.Adam is True:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.SGD is True:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=args.momentum_value)
    assert optimizer is not None
    batches = math.ceil(len(train_iter.dataset) / args.batch_size)
    best_eval = {'dev_accuracy': -1, 'test_accuracy': -1, 'best_epoch': -1, 'best_dev': False}
    # set to train mode, In order to use BatchNormalization and Dropout
    model.train()
    for epoch in range(1, args.epochs+1):
        print("\n\n### The epoch {}, Total epoch {} ###".format(epoch, args.epochs))
        # get batch data for training
        for steps, batch in enumerate(train_iter, start=1):
            feature, target = batch.text, batch.label
            # batch first, index align
            feature, target = feature.data.t(), target.data.sub(1)
            if args.cuda:  # GPU
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()   # clear gradients in this training step
            logit = model(feature)  # cnn output
            loss = F.cross_entropy(logit, target)  # compute loss
            loss.backward()         # backpropagation and compute gradients
            if args.clip_max_norm is not None:
                utils.clip_grad_norm(model.parameters(), max_norm=args.clip_max_norm)
            optimizer.step()  # apply gradients
            # compute current training accuracy
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = float(corrects)/batch.batch_size * 100.0
                sys.stdout.write('\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'
                                 .format(steps, batches, loss.item(), accuracy, corrects, batch.batch_size))
            # call the evaluation function to evaluate
            if steps % args.test_interval == 0:
                eval(dev_iter, model, args, best_eval, epoch, test=False)
                eval(test_iter, model, args, best_eval, epoch, test=True)
            if steps % args.save_interval == 0:  # save model
                if not os.path.isdir(args.save_dir):
                    os.makedirs(args.save_dir)
                save_path = os.path.join(args.save_dir, 'epoch{}_steps{}.pt'.format(epoch, steps))
                torch.save(model.state_dict(), save_path)

def eval(data_iter, model, args, best_eval, epoch, test=False):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature, target = feature.data.t(), target.data.sub(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=True)
        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    data_size = len(data_iter.dataset)
    avg_loss = loss.item() / data_size
    accuracy = 100.0 * corrects / data_size
    model.train()
    print('loss: {:.6f}\tacc: {:.4f}%({}/{})'.format(avg_loss, accuracy, corrects, data_size))
    # 判断是否准确率最佳的模型 更新best_eval中相关字段


if __name__ == "__main__":
    # 加载超参数
    config = load_param()
    # GPU相关
    if config.cuda is True:
        print("Using GPU To Train...")
        torch.cuda.manual_seed_all(123)
        torch.cuda.set_device(config.device_id)
    # 加载数据
    train_iter, dev_iter, test_iter = load_data(config)
    # 加载模型
    model = load_model(config)
    # 开始训练
    train(model, config, train_iter, dev_iter, test_iter)
