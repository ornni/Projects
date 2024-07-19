import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Scaled Dot Production Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask = None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype = torch.float32))
       #scores = [batch size, n heads, query len, key len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim = -1)
        #attention = [batch size, n heads, query len, key len]
        
        output = torch.matmul(attn, V)
        #x = [batch size, n heads, query len, head dim]

        return output, attn

# Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask = None):
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        attn_output, attn = ScaledDotProductAttention()(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
        #x = [batch size, query len, n heads, head dim]
        #x = [batch size, query len, hid dim]

        output = self.fc(attn_output)
        #x = [batch size, query len, hid dim]

        return output, attn
    
# Position-wise Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
        #x = [batch size, seq len, hid dim]

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask = None):
        src1, _ = self.attention(src, src, src, mask)
        src = src + self.dropout1(src1)
        src = self.norm1(src)

        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, dropout = 0.1):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, mask = None):
        src = self.embedding(src)
        src = self.dropout(src)

        for layer in self.layers:
            src = layer(src, mask)
        #src = [batch size, src len, hid dim]

        return self.norm(src)

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, src_mask = None, tgt_mask = None):
        tgt2, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, attn = self.encoder_attention(tgt, memory, memory, src_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn
    
# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, d_model, num_layers, num_heads, d_ff, dropout = 0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, src_mask = None, tgt_mask = None):
        tgt = self.embedding(tgt)
        tgt = self.dropout(tgt)

        for layer in self.layers:
            tgt, attn = layer(tgt, memory, src_mask, tgt_mask)
        
        tgt = self.fc(tgt)
        tgt = F.softmax(tgt, dim=-1)

        return tgt, attn
        



###############
####EXAMPLE####
###############

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


## 1. 데이터 준비

# 영어와 프랑스어 단어 사전
eng_vocab = {"I": 0, "am": 1, "a": 2, "student": 3, "hello": 4, "world": 5, "<sos>": 6, "<eos>": 7}
fra_vocab = {'je': 0, 'suis': 1, 'un': 2, 'étudiant': 3, 'bonjour': 4, 'monde': 5, '<sos>': 6, '<eos>': 7}

# 반대로 변환할 수 있도록 사전 생성
eng_vocab_rev = {v: k for k, v in eng_vocab.items()}
fra_vocab_rev = {v: k for k, v in fra_vocab.items()}

# 임의의 영어-프랑스어 번역 데이터셋
data = [
    ([6, 0, 1, 2, 3, 7], [6, 0, 1, 2, 3, 7]),  # "<sos> I am a student <eos>" -> "<sos> je suis un étudiant <eos>"
    ([6, 4, 5, 7], [6, 4, 5, 7]),  # "<sos> hello world <eos>" -> "<sos> bonjour monde <eos>"
    ([6, 4, 0, 1, 2, 3, 7], [6, 4, 0, 1, 2, 3, 7]),  # "<sos> hello I am a student <eos>" -> "<sos> bonjour je suis un étudiant <eos>"
]

# 입력과 출력 데이터 텐서로 변환
src_sequence = [torch.tensor(seq[0]) for seq in data]
tgt_sequence = [torch.tensor(seq[1]) for seq in data]

src_pad_idx = len(eng_vocab)
tgt_pad_idx = len(fra_vocab)

## 2. 모델 정의

# 모델 하이퍼파라미터 설정
input_dim = len(eng_vocab) + 1
output_dim = len(fra_vocab) + 1
d_model = 32
num_layers = 2
num_heads = 4
d_ff = 64
dropout = 0.1

# 모델 인스턴스 생성
encoder = TransformerEncoder(input_dim, d_model, num_layers, num_heads, d_ff, dropout)
decoder = TransformerDecoder(output_dim, d_model, num_layers, num_heads, d_ff, dropout)

## 3. 학습 과정

# 학습 설정
num_epochs = 100
learning_rate = 0.001

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss(ignore_index = tgt_pad_idx)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = learning_rate)

# 학습 루프
for epoch in range(num_epochs):
    epoch_loss = 0
    for src_seq, tgt_seq in zip(src_sequence, tgt_sequence):
        src_seq = src_seq.unsqueeze(0)
        tgt_seq = tgt_seq.unsqueeze(0)

        optimizer.zero_grad()

        memory = encoder(src_seq)

        output, _ = decoder(tgt_seq[:, :-1], memory)

        # loss
        loss = criterion(output.view(-1, output_dim), tgt_seq[:, 1:].reshape(-1))
        loss.backward()

        optimizer.step()
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data):.4f}')

## 4. 평가

def translate_sentence(sentence, encoder, decoder, src_vocab, tgt_vocab_rev):
    src_seq = torch.tensor([src_vocab[word] for word in sentence]).unsqueeze(0)
    memory = encoder(src_seq)

    tgt_seq = torch.tensor([src_vocab["<sos>"]]).unsqueeze(0)
    output_sentence = []

    for _ in range(50):
        output, _ = decoder(tgt_seq, memory)
        next_word = output.argmax(2)[:, -1].item()
        output_sentence.append(next_word)

        if next_word == src_vocab["<eos>"]:
            break

        tgt_seq = torch.cat([tgt_seq, torch.tensor([[next_word]])], dim = 1)

    return [tgt_vocab_rev[idx] for idx in output_sentence]

# 번역 테스트 1
sentence = ["<sos>", "I", "am", "a", "student"]
translated = translate_sentence(sentence, encoder, decoder, eng_vocab, fra_vocab_rev)
print(" ".join(translated))

# 번역 테스트 2
sentence = ["<sos>", "hello", "I", "am", "a", "student"]
translated = translate_sentence(sentence, encoder, decoder, eng_vocab, fra_vocab_rev)
print(" ".join(translated))

# 번역 테스트 3
sentence = ["<sos>", "hello", "student"]
translated = translate_sentence(sentence, encoder, decoder, eng_vocab, fra_vocab_rev)
print(" ".join(translated))
# 학습이 많이 되지 않아 틀리지만 결과를 확인할 수 있다
