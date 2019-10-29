import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.2):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
    
class WeightDrop(nn.Module):
    def __init__(self, module, weights, dropout: float = 0,
                 variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout, name_w))
            w = getattr(self.module, name_w)
            # Delete original weight
            delattr(self.module, name_w)
            self.module.register_parameter(name_w + '_raw', Parameter(w))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.ones(raw_w.size(0), 1, requires_grad=True)
                if raw_w.is_cuda:
                    mask = mask.cuda()
                mask = F.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = F.dropout(raw_w, p=self.dropout, training=self.training)
            w = Parameter(w)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)
        
class RNNModel(nn.Module):
    def __init__(self, ntoken, emsize, weight, nhid, nlayers, dropouti=0.2, dropoutrnn=0.2, dropout=0.2, wdrop=0.5):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding.from_pretrained(weight, freeze=False)
        self.rnns = nn.GRU(emsize, nhid, nlayers, dropout=dropoutrnn)
       
        if wdrop:
            self.rnns = WeightDrop(self.rnns, ['weight_hh_l{}'.format(i) for i in range(nlayers)], wdrop)
        print(self.rnns)
        
        self.decoder = nn.Linear(nhid, ntoken)
        nn.init.xavier_normal(self.decoder.weight)
        
        self.dropouti = dropouti
        self.dropout = dropout
        
    def forward(self, input, hidden):
        emb = self.encoder(input)
        emb = self.lockdrop(emb, self.dropouti)

        x, h_n = self.rnns(emb, hidden)
        
        x = self.lockdrop(x, self.dropout)
        x = x.view(-1, x.size(2))
        output = self.decoder(x)
        return output, h_n
