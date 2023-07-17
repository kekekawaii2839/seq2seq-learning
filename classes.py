import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

USE_CUDA = True

SOS_token = 0
EOS_token = 1
PAD_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 2 # Count SOS and EOS
      
    def index_words(self, sentence):
        if self.name == 'cn':
            for word in sentence:
                self.index_word(word)
        else:
            for word in sentence.split(' '):
                self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=True)
        
    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.rnn(embedded, (hidden, hidden))
        return output, hidden

    def init_hidden(self):
        hidden = torch.zeros(2*self.n_layers, 1, self.hidden_size)
        if USE_CUDA: hidden = hidden.cuda()
        return hidden

class Attention(nn.Module): #method: dot product
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
    
    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        attn_energies = torch.zeros(seq_len) # B x 1 x S
        if USE_CUDA: attn_energies = attn_energies.cuda()
        
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        
        attn = F.softmax(attn_energies, dim=0)

        output = torch.zeros(self.hidden_size)
        if USE_CUDA: output = output.cuda()
        """for i in range(seq_len):
            output += attn[i] * encoder_outputs[i][0]"""
        encoder_outputs = encoder_outputs.squeeze(1)
        #print(attn.shape, encoder_outputs.shape)
        attn = attn.unsqueeze(1)
        output += torch.mm(torch.transpose(encoder_outputs, 0, 1), attn).squeeze(1)
        return output, attn
    
    def score(self, hidden, encoder_output):
        energy = torch.dot(hidden.view(-1), encoder_output.view(-1))
        return energy

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(Decoder, self).__init__()
        
        # Keep parameters for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.attention = Attention(hidden_size)
        self.out = nn.Linear(2*hidden_size, output_size)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        rnn_output, hidden = self.rnn(word_embedded, last_hidden)

        rnn_output = rnn_output.squeeze(0)
        """print(rnn_output.shape)
        print(encoder_outputs.shape)"""
        attn_output, attn = self.attention(rnn_output, encoder_outputs)
        """print(attn_output.shape)"""
        comb_output = torch.cat((rnn_output[-1], attn_output), -1)
        #output = F.log_softmax(self.out(rnn_output), -1)
        output = F.log_softmax(self.out(comb_output), -1).unsqueeze(0).view(1, -1)

        return output, hidden, attn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, criterion):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
    
    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5, clip=5.0):
        target_length = target_seq.size()[0]
        loss = 0

        hidden = self.encoder.init_hidden()
        encoder_out, hidden = self.encoder(input_seq, hidden)
        #hidden = torch.cat([haha for haha in hidden], dim=0) #from tuple to tensor
        hidden = [i.view(self.decoder.n_layers, 1, -1) for i in hidden]

        decoder_input = torch.LongTensor([[SOS_token]])
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
        
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        if use_teacher_forcing:
            for di in range(target_length):
                """print(decoder_input.shape)
                print(hidden[0].shape)
                print(hidden[1].shape)
                print(encoder_out.shape)"""
                output, hidden, attn = self.decoder(decoder_input, hidden, encoder_out)
                """print(output.shape)
                print(target_seq.shape)"""
                loss += self.criterion(output, target_seq[di])
                decoder_input = target_seq[di]
        else:
            for di in range(target_length):
                """print(decoder_input.shape)
                print(hidden[0].shape)
                print(hidden[1].shape)
                print(encoder_out.shape)"""
                output, hidden, attn = self.decoder(decoder_input, hidden, encoder_out)
                """print(output.shape)
                print(target_seq.shape)"""
                loss += self.criterion(output, target_seq[di])
                topv, topi = output.data.topk(1)
                ni = topi[0][0]
                decoder_input = torch.LongTensor([[ni]])
                if USE_CUDA: decoder_input = decoder_input.cuda()
                if ni == EOS_token: break
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)
        
        return loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term) # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term) # (max_len, d_model/2)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class myTransformer(nn.Module): # input_size是输入词典长度
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, num_heads=8, dropout_p=0.1):
        super(myTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.input_embedding = nn.Embedding(input_size, hidden_size)
        self.output_embedding = nn.Embedding(output_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=n_layers, num_decoder_layers=n_layers, dropout=dropout_p)
        self.positional_encoding = PositionalEncoding(hidden_size, dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def get_key_padding_mask(tokens):
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == PAD_token] = True
        #print(tokens)
        #print(key_padding_mask)
        return key_padding_mask
    
    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5):
        input_embedded = self.input_embedding(input_seq)
        input_embedded = self.positional_encoding(input_embedded)
        target_embedded = self.output_embedding(target_seq)
        target_embedded = self.positional_encoding(target_embedded)
        #print(input_embedded.shape, target_embedded.shape)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_seq.size(0)).cuda()
        src_key_padding_mask=myTransformer.get_key_padding_mask(input_seq).cuda().transpose(0, 1)
        tgt_key_padding_mask=myTransformer.get_key_padding_mask(target_seq).cuda().transpose(0, 1)

        """print(tgt_mask)
        print(input_seq)
        print(src_key_padding_mask)
        print(target_seq)
        print(tgt_key_padding_mask)"""

        output = self.transformer(input_embedded, target_embedded,
                        tgt_mask=tgt_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask)
        #print("output.shape:", output.shape)
        #print(output)

        return output