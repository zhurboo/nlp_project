import os
import argparse
import hashlib
import torch
from data import Corpus
from utils import *

parser = argparse.ArgumentParser()

# Model parameters.
parser.add_argument('--data', type=str, default='data/title_storyline_story/')
parser.add_argument('--corpus', type=str, default='data/corpus.dt')
parser.add_argument('--conditional_data', type=str, default='data/title_storyline/test.txt')
parser.add_argument('--checkpoint', type=str, default='data/model.pt')
parser.add_argument('--outf', type=str, default='data/stroys_out3.txt')
parser.add_argument('--temperature', type=float, default=0.5, help='temperature - higher will increase diversity')
args = parser.parse_args()


print('Resuming model and criterion...')
model, criterion = model_load(args.checkpoint)
model = model.cuda()
criterion = criterion.cuda()

if os.path.exists(args.corpus):
    print('Loading cached dataset...')
    corpus = torch.load(args.corpus)
else:
    print('Building dataset...')
    corpus = Corpus(args.data)
    torch.save(corpus, args.corpus)

eos_id = corpus.word2idx('<eos>')

cond_data = torch.LongTensor(corpus.tokenize(args.conditional_data))
input = torch.rand(1, 1).mul(corpus.ntoken).long().cuda()

model.eval()
with torch.no_grad():
    with open(args.outf, 'w') as outf:
        now = 0
        cnt = 0
        while now < cond_data.size(0):
            hidden = None
            while True:
                input.data.fill_(cond_data[now])
                output, hidden = model(input, hidden)
                now += 1
                if cond_data[now] == eos_id:
                    break
            now += 1
            cnt += 1
            while True:
                word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                if word_idx == eos_id:
                    outf.write('\n')
                    break
                word = corpus.idx2word(word_idx)
                outf.write(word+' ')
                input.data.fill_(word_idx)
                output, hidden = model(input, hidden)
            if cnt %100 == 0:
                print(cnt)
            