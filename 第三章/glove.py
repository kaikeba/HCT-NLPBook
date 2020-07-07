from nltk.tokenize import word_tokenize  
from torch.autograd import Variable  
import numpy as np  
import torch  
import torch.optim as optim  

context_size = 3  
embed_size = 2  
xmax = 2  
alpha = 0.75  
batch_size = 20  
l_rate = 0.001  
num_epochs = 10  
  
text_file = open('  ', 'r')  
text = text_file.read().lower()  
text_file.close()  

word_list = word_tokenize(text)  
vocab = np.unique(word_list)  
w_list_size = len(word_list)  
vocab_size = len(vocab)  
w_to_i = {word: ind for ind, word in enumerate(vocab)}  

comat = np.zeros((vocab_size, vocab_size))  
for i in range(w_list_size):  
    for j in range(1, context_size+1):  
        ind = w_to_i[word_list[i]]  
        if i-j > 0:  
            lind = w_to_i[word_list[i-j]]  
            comat[ind, lind] += 1.0/j  
        if i+j < w_list_size:  
            rind = w_to_i[word_list[i+j]]  
            comat[ind, rind] += 1.0/j  
coocs = np.transpose(np.nonzero(comat))  

def wf(x):  
    if x < xmax:  
        return (x/xmax)**alpha  
    return 1  
  
l_embed, r_embed = [  
    [Variable(torch.from_numpy(np.random.normal(0, 0.01, (embed_size, 1))),  
        requires_grad = True) for j in range(vocab_size)] for i in range(2)]  
l_biases, r_biases = [  
    [Variable(torch.from_numpy(np.random.normal(0, 0.01, 1)),   
        requires_grad = True) for j in range(vocab_size)] for i in range(2)]  

optimizer = optim.Adam(l_embed + r_embed + l_biases + r_biases, lr = l_rate)  

def gen_batch():      
    sample = np.random.choice(np.arange(len(coocs)), size=batch_size, replace=False)  
    l_vecs, r_vecs, covals, l_v_bias, r_v_bias = [], [], [], [], []  
    for chosen in sample:  
        ind = tuple(coocs[chosen])  
        l_vecs.append(l_embed[ind[0]])  
        r_vecs.append(r_embed[ind[1]])  
        covals.append(comat[ind])  
        l_v_bias.append(l_biases[ind[0]])  
        r_v_bias.append(r_biases[ind[1]])  
    return l_vecs, r_vecs, covals, l_v_bias, r_v_bias  

for epoch in range(num_epochs):  
    num_batches = int(w_list_size/batch_size)  
    avg_loss = 0.0  
    for batch in range(num_batches):  
        optimizer.zero_grad()  
        l_vecs, r_vecs, covals, l_v_bias, r_v_bias = gen_batch()  
        # For PyTorch v2 use, .view(-1) in torch.dot here. Otherwise, no need to use .view(-1).  
        loss = sum([torch.mul((torch.dot(l_vecs[i].view(-1), r_vecs[i].view(-1)) +  
                l_v_bias[i] + r_v_bias[i] - np.log(covals[i]))**2,  
                wf(covals[i])) for i in range(batch_size)])  
        avg_loss += loss.data[0]/num_batches  
        loss.backward()  
        optimizer.step()  
    print("Average loss for epoch "+str(epoch+1)+": ", avg_loss)  
