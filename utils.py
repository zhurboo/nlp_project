import torch
import os
import numpy as np

def model_save(model, criterion, fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion], f)

def model_load(fn):
    with open(fn, 'rb') as f:
        model, criterion = torch.load(f)
    return model, criterion

def title_storyline_data(file, corpus):
    data = []
    lens = []
    with open(file, 'r') as f:
        for line in f:
            data.append(line)
            lens.append(len(line.split('<EOT>')[0].split()))
    lens = np.array(lens)
    data = np.array(data)
    arg = np.argsort(lens)
    lens = lens[arg]
    data = data[arg]
    title = []
    lines = []
    for title_lines in data:
        title_lines = title_lines.split('<EOT>')
        t = title_lines[0].split()
        l = ['<EOT>']+title_lines[1].split()[:-1]
        title.append([corpus.word2idx(each) for each in t])
        lines.append([corpus.word2idx(each) for each in l])
    return [title,lines]
    
def get_batch_title_lines(data, i, batch_size):
    title, lines = data[0], data[1]
    title_len = len(title[i])
    next_i = i+1
    end = min(len(title), i+batch_size)
    while next_i < end and len(title[next_i]) == title_len:
        next_i += 1
    in_title = torch.LongTensor(title[i:next_i]).permute(1,0)
    in_lines = torch.LongTensor(lines[i:next_i]).permute(1,0)[:5,:]
    out_lines = torch.LongTensor(lines[i:next_i]).permute(1,0)[1:,:]
    
    return in_title, in_lines, out_lines, next_i
    
def batchify(data, bsz, args):
    data = torch.LongTensor(data)
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
