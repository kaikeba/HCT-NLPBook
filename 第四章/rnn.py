import torch  
from torch import nn  

class RNN(nn.Module):  
    def __init__(self):  
        super(RNN, self).__init__()  
        self.rnn = nn.RNN(       
            input_size = INPUT_SIZE     
            hidden_size = HIDDEN_SIZE  
            num_layers = NUM_LAYERS  
            batch_first = True #以batch_size为第一维度的特征集  
        )  
        self.out = nn.Linear(64, 10) #全连接层  
      
    def forward(self, x):  
        ''''' 
        x_shape (batch, time_step, input_size) 
        r_out shape (batch, time_step, output_size)  
        h_n shape (n_layers, batch, hidden_size) 
        h_c shape (n_layers, batch, hidden_size) 
        '''  
        r_out, (h_n, h_c) = self.rnn(x, None)  
          
        # 选取最后一个时间点的r_out输出  
        out = self.out(r_out[:, -1, :])  
        return out  
  
rnn = RNN()  

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   
loss_func = nn.CrossEntropyLoss()  #交叉熵  

for epoch in range(EPOCH):  
    for step, (b_x, b_y) in enumerate(train_loader):  
        #reshape b_x to shape (batch, time_step, input_size)  
        # 训练  
        b_x = b_x.view(-1, TIME_STEP, INPUT_SIZE)   
        output = rnn(b_x)    #输出预测值  
        loss = loss_func(output, b_y)   #计算loss  
        optimizer.zero_grad()     #防止梯度爆炸  
        loss.backward()     #反向传播  
        optimizer.step()    #参数更新  
        # 测试  
        if step % 50 == 0:  
            # test_x shape (samples, time_step, input_size)  
            test_output = rnn(test_x)    
            pred_y = torch.max(test_output, 1)[1].data.numpy()  
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)  
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)  
  
# 打印测试集中前10个预测值与真实值  
test_output = rnn(test_x[:10].view(-1, TIME_STEP, INPUT_SIZE))  
pred_y = torch.max(test_output, 1)[1].data.numpy()  
print(pred_y, 'prediction number')  
print(test_y[:10], 'real number')  

