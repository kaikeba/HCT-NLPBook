import torch
import torch.nn as nn
import torch.optim as optim
# 使用transformers包
from transformers import BertTokenizer, BertModel
from torchtext import data, datasets
import numpy as np
import random
import time

# 参数
SEED = 1234
TRAIN = False
BATCH_SIZE = 128
N_EPOCHS = 5
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

TEXT = "I like you!"

# 固定模型用种子，便于重复试验
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# 应用transformers中Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
init_token_id = tokenizer.cls_token_id
eos_token_id  = tokenizer.sep_token_id
pad_token_id  = tokenizer.pad_token_id
unk_token_id  = tokenizer.unk_token_id

max_input_len = tokenizer.max_model_input_sizes['bert-base-uncased']

# 将句子长度切割成510长，为了加上开头和最后一个token
def tokenize_and_crop(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_len - 2]
    return tokens

# 加载PyTorch提供的IMDB数据
def load_data():
    text = data.Field(
        batch_first=True,
        use_vocab=False,
        tokenize=tokenize_and_crop,
        preprocessing=tokenizer.convert_tokens_to_ids,
        init_token=init_token_id,
        pad_token=pad_token_id,
        unk_token=unk_token_id
    )

    label = data.LabelField(dtype=torch.float)

    train_data, test_data  = datasets.IMDB.splits(text, label)
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))

    print(f"training examples count: {len(train_data)}")
    print(f"test examples count: {len(test_data)}")
    print(f"validation examples count: {len(valid_data)}")

    label.build_vocab(train_data)

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device
    )

    return train_iter, valid_iter, test_iter

# 看是否有GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 通过transformers包，建立BERT模型
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 此处用BERT做为基础模型完成情感分析任务
# 在BERT之上加两层GRU
# 最后接一层线性层用于完成分类任务
class SentimentModel(nn.Module):
    def __init__(
        self,
        bert,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout
    ):
      
        super(SentimentModel, self).__init__()
    
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0 if n_layers < 2 else dropout
            )
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
            
        _, hidden = self.rnn(embedded)
    
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
    
        output = self.out(hidden)
        return output

model = SentimentModel(
  bert_model,
  HIDDEN_DIM,
  OUTPUT_DIM,
  N_LAYERS,
  BIDIRECTIONAL,
  DROPOUT
)
print(model)

# 一个epoch需要多长时间
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 二分类问题的accuracy
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

# 一个训练步
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
  
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
  
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 验证模型
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
  
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
      
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 预测模型
def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_len - 2]
    indexed = [init_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_id]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

if __name__ == "__main__":
    # 开始训练
    if TRAIN:
        # 读取数据
        train_iter, valid_iter, test_iter = load_data()

        optimizer = optim.Adam(model.parameters())
        criterion = nn.BCEWithLogitsLoss().to(device)
        model = model.to(device)

        best_val_loss = float('inf')

        for epoch in range(N_EPOCHS):
            start_time = time.time()
            # 训练一个epoch
            train_loss, train_acc = train(model, train_iter, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, valid_iter, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'model.pt')

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        # 测试
        model.load_state_dict(torch.load('model.pt'))
        test_loss, test_acc = evaluate(model, test_iter, criterion)
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
  
    # 推理结果
    else:
        model.load_state_dict(torch.load('model.pt', map_location=device))
        sentiment = predict_sentiment(model, tokenizer, TEXT)
        print(sentiment)

 
