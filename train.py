import os
import hashlib
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from data import Corpus
from model import RNNModel
from utils import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=True)

parser.add_argument('--data', type=str, default='data/title_storyline_story/')
parser.add_argument('--corpus', type=str, default='data/corpus.dt')
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--bptt', type=int, default=70)

parser.add_argument('--resume', type=str, default='')
parser.add_argument('--save', type=str, default='data/model.pt')
parser.add_argument('--emsize', type=int, default=300)
parser.add_argument('--nhid', type=int, default=1100)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--dropouti', type=float, default=0.1)
parser.add_argument('--dropoutrnn', type=float, default=0.1)
parser.add_argument('--wdrop', type=float, default=0)

parser.add_argument('--optimizer', type=str, default='sgd', help='sgd or adam')
parser.add_argument('--lr', type=float, default=10)
parser.add_argument('--clip', type=float, default=0.1)
parser.add_argument('--wdecay', type=float, default=2e-5)

parser.add_argument('--log-interval', type=int, default=200)
args = parser.parse_args()


###############################################################################
# dataset
###############################################################################
if os.path.exists(args.corpus):
    print('Loading cached dataset...')
    corpus = torch.load(args.corpus)
else:
    print('Building dataset...')
    corpus = Corpus(args.data)
    torch.save(corpus, args.corpus)
print('Num tokens: {}'.format(corpus.ntoken))

train_batch_size = args.batch_size
eval_batch_size = args.batch_size
test_batch_size = 1
train_data = batchify(corpus.train, train_batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# model, criterion and optimizer
###############################################################################
if args.resume:
    print('Resuming model and criterion...')
    model, criterion = model_load(args.resume)
else:
    print('Building model and criterion...')
    model = RNNModel(corpus.ntoken, args.emsize, corpus.weight, args.nhid, args.nlayers, args.dropouti, args.dropoutrnn, args.dropout, args.wdrop)
    
if args.cuda:
    model = model.cuda()
    
print('-' * 89)
print('Args:', args)
print('Model parameters:', count_parameters(model))

criterion = nn.CrossEntropyLoss()
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
else:
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.wdecay)
scheduler = StepLR(optimizer, step_size=30, gamma=0.2)

###############################################################################
# evaluate funcition
###############################################################################
def evaluate(model, criterion, data_source, batch_size):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        hidden = None
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / len(data_source)

###############################################################################
# train funcition
###############################################################################
def train(model, criterion, optimizer, train_data, batch_size):
    model.train()
    total_loss = 0
    hidden = None
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args)
        # data: [seq, batch]
        # targets: [seq*batch]
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        hidden = hidden[0].data,hidden[1].data
        loss = criterion(output, targets)
        loss.backward()
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | loss {:5.2f} | ppl {:8.2f}'\
                  .format(epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'], cur_loss, np.exp(cur_loss)))
            total_loss = 0

###############################################################################
# train and evaluate
###############################################################################
best_loss = 1e20
try:
    for epoch in range(1, args.epochs + 1):
        train(model, criterion, optimizer, train_data, train_batch_size)
        val_loss = evaluate(model, criterion, val_data, eval_batch_size)
        print('-' * 89)
        print('| end of epoch {:3d} | valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, val_loss, np.exp(val_loss)))
        print('-' * 89)
        scheduler.step()
        if val_loss < best_loss:
            model_save(model, criterion, args.save)
            print('Saving model (new best validation)')
            best_loss = val_loss
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
