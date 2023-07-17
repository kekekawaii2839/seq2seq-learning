import unicodedata
import re
import time
import math
import os

from classes import *

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z\u4e00-\u9fa5.!?，。？]+", r" ", s)
    return s

def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(os.path.dirname(os.path.realpath(__file__))+'\\%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    
    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang, output_lang, pairs

MAX_LENGTH = 10

def filter_pair(p):
    return len(p[1].split(' ')) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))
    
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    
    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepare_data('cn', 'eng', False)

# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    if lang.name == 'cn':
        return [lang.word2index[word] for word in sentence]
    else:
        return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    indexes += [PAD_token] * (MAX_LENGTH - len(indexes))
    indexes.insert(0, SOS_token)
    var = torch.LongTensor(indexes).view(-1, 1)
    if USE_CUDA: var = var.cuda()
    return var

def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)

teacher_forcing_ratio = 0.5
mix_ratio = 0.5

def train(input_variable, target_variable, model, model_optimizer, criterion):

    # Zero gradients of optimizer
    model_optimizer.zero_grad()
    loss = 0
    
    tgt = target_variable[:-1,:] # (max_length, 1)
    tgt_y = target_variable[1:,:]

    #print(tgt)

    output = model(input_variable, tgt)
    output = model.out(output) # (max_length, 1, output_vocab_size)

    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_y.contiguous().view(-1))
    else:
        _, mix_tgt = torch.max(output.squeeze(1), dim=1) # mix_tgt's size: (max_length)
        mix_tgt = torch.cat((torch.LongTensor([SOS_token]).cuda(), mix_tgt[:-1]))
        # mix tgt and mix_tgt
        for i in range(mix_tgt.size(0)):
            if random.random() < mix_ratio:
                mix_tgt[i] = tgt.squeeze(1)[i]
        output = model(input_variable, mix_tgt.unsqueeze(1))
        output = model.out(output)
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_y.contiguous().view(-1))
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    model_optimizer.step()
    
    return loss.item()

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))