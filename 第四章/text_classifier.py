import torch  
import torch.nn as nn  
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence  
 
class RNN(nn.Module):  
    def __init__(self, vocab_size, embed_size, num_output, rnn_model='LSTM', use_last=True,
                 embedding_tensor=None,  padding_index=0, hidden_size=64, num_layers=1, batch_first=True):  
        super(RNN, self).__init__()  
        self.use_last = use_last  
        # embedding  
        self.encoder = None  
        if torch.is_tensor(embedding_tensor):  
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index, _weight=embedding_tensor)  
            self.encoder.weight.requires_grad = False  
        else:  
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)  
        self.drop_en = nn.Dropout(p=0.6)  
  
        # rnn module  
        if rnn_model == 'LSTM':  
            self.rnn = nn.LSTM( input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,  batch_first=True, bidirectional=True)  
        elif rnn_model == 'GRU':  
            self.rnn = nn.GRU( input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,  batch_first=True, bidirectional=True)  
        else:  
            raise LookupError(' only support LSTM and GRU')  
  
        self.bn2 = nn.BatchNorm1d(hidden_size*2)  
        self.fc = nn.Linear(hidden_size*2, num_output)  
  
    def forward(self, x, seq_lengths):  
        x_embed = self.encoder(x)  
        x_embed = self.drop_en(x_embed)  
        packed_input = pack_padded_sequence(x_embed, seq_lengths.cpu().numpy(),batch_first=True)  
        packed_output, ht = self.rnn(packed_input, None)  
        out_rnn, _ = pad_packed_sequence(packed_output, batch_first=True)  
        row_indices = torch.arange(0, x.size(0)).long()  
        col_indices = seq_lengths - 1  
        if next(self.parameters()).is_cuda:  
            row_indices = row_indices.cuda()  
            col_indices = col_indices.cuda()  
  
        if self.use_last:  
            last_tensor=out_rnn[row_indices, col_indices, :]  
        else:  
            last_tensor = out_rnn[row_indices, :, :]  
            last_tensor = torch.mean(last_tensor, dim=1)  
  
        fc_input = self.bn2(last_tensor)  
        out = self.fc(fc_input)  
        return out  

if args.cuda:  
    torch.backends.cudnn.enabled = True  
    cudnn.benchmark = True  
    model.cuda()  
    criterion = criterion.cuda()  
  
def train(train_loader, model, criterion, optimizer, epoch):  
    batch_time = AverageMeter()  
    data_time = AverageMeter()  
    losses = AverageMeter()  
    top1 = AverageMeter()  
  
    # switch to train mode  
    model.train()  
  
    end = time.time()  
    for i, (input, target, seq_lengths) in enumerate(train_loader):  
        # measure data loading time  
        data_time.update(time.time() - end)  
  
        if args.cuda:  
            input = input.cuda(async=True)  
            target = target.cuda(async=True)  
  
        # compute output  
        output = model(input, seq_lengths)  
        loss = criterion(output, target)  
  
        # measure accuracy and record loss  
        prec1 = accuracy(output.data, target, topk=(1,))  
        losses.update(loss.data, input.size(0))  
        top1.update(prec1[0][0], input.size(0))  
  
        # compute gradient and do SGD step  
        optimizer.zero_grad()  
        loss.backward()  
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  
        optimizer.step()  
  
        # measure elapsed time  
        batch_time.update(time.time() - end)  
        end = time.time()  
  
        if i != 0 and i % args.print_freq == 0:  
            print('Epoch: [{0}][{1}/{2}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '   'Data {data_time.val:.3f} ({data_time.avg:.3f})  Loss {loss.val:.4f} ({loss.avg:.4f})  '   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(  epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))  
            gc.collect()  

def test(val_loader, model, criterion):  
    batch_time = AverageMeter()  
    losses = AverageMeter()  
    top1 = AverageMeter()  
  
    # switch to evaluate mode  
    model.eval()  
    end = time.time()  
    for i, (input, target,seq_lengths) in enumerate(val_loader):  
        if args.cuda:  
            input = input.cuda(async=True)  
            target = target.cuda(async=True)  
  
        # compute output  
        output = model(input,seq_lengths)  
        loss = criterion(output, target)  
  
        # measure accuracy and record loss  
        prec1 = accuracy(output.data, target, topk=(1,))  
        losses.update(loss.data, input.size(0))  
        top1.update(prec1[0][0], input.size(0))  
  
        # measure elapsed time  
        batch_time.update(time.time() - end)  
        end = time.time()  
  
        if i!= 0 and i % args.print_freq == 0:  
            print('Test: [{0}/{1}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '  'Loss {loss.val:.4f} ({loss.avg:.4f})  Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format( i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))  
            gc.collect()  
  
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))  
    return top1.avg  

def accuracy(output, target, topk=(1,)):  
    maxk = max(topk)  
    batch_size = target.size(0)  
  
    _, pred = output.topk(maxk, 1, True, True)  
    pred = pred.t()  
    correct = pred.eq(target.view(1, -1).expand_as(pred))  
  
    res = []  
    for k in topk:  
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)  
        res.append(correct_k.mul_(100.0 / batch_size))  
    return res  
  
def adjust_learning_rate(lr, optimizer, epoch):  
    lr = lr * (0.1 ** (epoch // 8))  
    for param_group in optimizer.param_groups:  
        param_group['lr'] = lr  

for epoch in range(1, args.epochs+1):  
    adjust_learning_rate(args.lr, optimizer, epoch)  
    train(train_loader, model, criterion, optimizer, epoch)  
    test(val_loader, model, criterion)  
  
    # save current model  
    if epoch % args.save_freq == 0:  
        name_model = 'rnn_{}.pkl'.format(epoch)  
        path_save_model = os.path.join('gen', name_model)  
        joblib.dump(model.float(), path_save_model, compress=2)  


