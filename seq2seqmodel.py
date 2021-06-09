import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, emb=None):
        super(Encoder, self).__init__()
        if (emb != None):
            self.embedding = nn.Embedding.from_pretrained(emb)
            # requires_grad指定是否在训练过程中对词向量的权重进行微调
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        output = self.concat(output)
        return output, hidden

class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout=0.1, emb=None, atten_model='dot'):
        super(AttnDecoder, self).__init__()
        self.atten_model = atten_model
        if (emb != None):
            self.embedding = nn.Embedding.from_pretrained(emb)
            # requires_grad指定是否在训练过程中对词向量的权重进行微调
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.hiddenSize = hidden_size
        self.output_size = vocab_size
        self.dropout = dropout

        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.att = Attn(atten_model, hidden_size)
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
    def forward(self, y, hidden, encoder_output):
        '''
                Bahdanau等（2015）的方法：将前一隐藏层输出与编码器的输出运算得到权重，然后
                将权重乘以编码器的输出并与当前输入进行concat连接得到当前输入，经过当前神经元运算得到
                下一个神经元的输入
                :param seq_in: 输入序列
                :param state: 前一神经元的隐藏层
                :param encoder_output: 编码器的输出
                :return:
                '''
        embedded = self.embedding(y)

        onelayerhidden = hidden[0, :, :].unsqueeze(1)  # batchsize,layer,hiddensize
        # encoder_output = encoder_output.permute(0,2,1) #batchsize,vocsize,time
        # 这里计算权重是由解码器的上一时刻隐藏层与编码器的所有输出运算得出
        atten_weights = self.att(onelayerhidden, encoder_output)
        # 将注意力权重乘以编码器输出以获得新的“加权和”上下文向量
        context = atten_weights.bmm(encoder_output)

        concat_input = torch.cat((embedded, context), 2)
        input = self.concat(concat_input)

        output, hidden = self.gru(input, hidden)

        output = self.out(output)
        output = F.softmax(output, dim=1)

        return output, hidden, atten_weights


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # 根据给定的方法计算注意力（能量）
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        encoder_out, hidden = self.encoder(x)
        hidden = hidden[:1, :, :]
        output, hidden, atten_weights  = self.decoder(y=y, hidden=hidden, encoder_output=encoder_out)
        return output, hidden, atten_weights

    def translate(self, x, y, max_length=50):
        encoder_out, hid = self.encoder(x)
        preds = []
        batch_size = x.shape[0]
        for i in range(max_length):
            output = self.decoder(y=y, hidden=hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)

        return torch.cat(preds, 1)
