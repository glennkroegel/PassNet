'''
created_by: Glenn Kroegel
date: 14 May 2020
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
from loading import *
import numpy as np
from tqdm import tqdm
import math
from config import vocab_sz, max_len

device = torch.cuda.set_device('cuda' if torch.cuda.is_available() else 'cpu')

mode = 'fp32'
if mode == 'fp16':
    from apex import amp

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''should be (sl, bs, d)'''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=max_len):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim)/10)

    def forward(self, x):
        '''inp: (bs, sl, d) ; outp: (bs, sl, d)'''
        x = x + self.pos
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, heads, dim, n=0):
        super(MultiHeadSelfAttention, self).__init__()
        self.n = n
        self.dim = dim
        self.h = heads
        self.d_k = dim // heads
        self.scale = math.sqrt(dim)

        self.lin_q = nn.Linear(dim, dim)
        self.lin_k = nn.Linear(dim, dim)
        self.lin_v = nn.Linear(dim, dim)
        self.lin_out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, q, k, v, attn_mask):
        bs = q.size(0)
        sl = q.size(1)
        q = self.lin_q(q).view(bs, self.h, self.d_k, sl)
        k = self.lin_k(k).view(bs, self.h, self.d_k, sl)
        v = self.lin_v(v).view(bs, self.h, self.d_k, sl)

        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)

        qk = q @ k.transpose(-1, -2) / self.scale
        
        if attn_mask is not None:
            qk.transpose(1,-1)[attn_mask == 0] = torch.tensor(float('-inf'), dtype=torch.float32)
        
        w = F.softmax(qk, dim=-1)
        outp = w @ v
        outp = outp.transpose(1,2).contiguous().view(bs, -1, self.dim)
        
        return outp, w

class Norm(nn.Module):
    def __init__(self, dim):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, factor):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, factor*dim)
        self.fc2 = nn.Linear(factor*dim, dim)

        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)

        self.norm1 = Norm(dim * factor)
        self.norm2 = Norm(dim)

        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.drop1(self.fc1(x)))
        x = self.norm1(x)
        x = self.act(self.drop2(self.fc2(x)))
        x = self.norm2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dim, factor, n=0):
        super(EncoderBlock, self).__init__()
        self.attn = MultiHeadSelfAttention(heads=8, dim=dim)
        self.ff = FeedForward(dim, factor)
        self.n = n

        self.norm1 = Norm(dim)
        self.norm2 = Norm(dim)

    def forward(self, inp, attn_mask):
        x, w = self.attn(inp, inp, inp, attn_mask)
        x = self.norm1(inp + x)
        x2 = self.ff(x)
        x = self.norm2(x + x2)
        return x, w

class Encoder(nn.Module):
    def __init__(self, dim, factor, num_enc):
        super(Encoder, self).__init__()
        self.encoder_blocks = nn.ModuleList([EncoderBlock(dim=dim, factor=factor, n=i) for i in range(num_enc+1)])

    def forward(self, x, attn_mask):
        for block in self.encoder_blocks:
            x, _ = block(x, attn_mask)
        return x

class Embed(nn.Module):
    def __init__(self, vocab_sz, dim):
        super(Embed, self).__init__()
        self.embedding = nn.Embedding(vocab_sz, dim)
        self.vocab_sz = vocab_sz
        self.dim = dim
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.drop(self.embedding(x))
        return x

class LMHead(nn.Module):
    def __init__(self, vocab_sz, dim):
        super(LMHead, self).__init__()
        self.l_in = nn.Linear(dim, dim)
        self.l_out = nn.Linear(dim, vocab_sz)
        # self.norm = Norm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.l_in(x))
        x = self.l_out(x)
        return x

class Model(nn.Module):
    def __init__(self, dim, factor, num_enc, vocab_sz=vocab_sz):
        super(Model, self).__init__()
        self.embedding = Embed(vocab_sz, dim)
        self.embedding.requires_grad = True
        self.positional = PositionalEncoding(dim)
        self.decoder = Encoder(dim=dim, factor=factor, num_enc=num_enc)
        self.head = LMHead(vocab_sz, dim)

    def forward(self, inp, attn_mask):
        bs = inp.size(0)
        sl = inp.size(1)
        x = self.embedding(inp)
        x = x.view(sl, bs, -1)
        x = self.positional(x)
        x = x.view(bs, sl, -1)
        enc_outp = self.decoder(x, attn_mask)
        logits = self.head(enc_outp)
        return logits

def loss_function(outp, target):
    loss = F.cross_entropy(outp, target, ignore_index=0)
    return loss

def evaluate(loader, model, criterion):
    props = {'loss': 0}
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if i % 2000 == 0:
                print(i)
            x, attn_mask, pred_ixs = batch            
            x = x.to(device)
            attn_mask = attn_mask.to(device)
            pred_ixs = pred_ixs.to(device)
            y = x[pred_ixs[:,0], pred_ixs[:,1]]
            outp = model(x, attn_mask)
            preds = outp[pred_ixs[:,0], pred_ixs[:,1]]
            loss = criterion(preds, y)
            props['loss'] += loss.item()
    L = len(loader)
    props = {k:v/L for k,v in props.items()}
    return props

def train(loader, model, optimizer, criterion, mode):
    '''training loop'''
    props = {'loss': 0}
    model.train()
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        if batch is None:
            print('NONE')
            continue
        if i % 500 == 0:
            print(i)
        x, attn_mask, pred_ixs = batch
        x = x.to(device)
        attn_mask = attn_mask.to(device)
        pred_ixs = pred_ixs.to(device)
        y = x[pred_ixs[:,0], pred_ixs[:,1]]
        outp = model(x, attn_mask)
        preds = outp[pred_ixs[:,0], pred_ixs[:,1]]
        loss = criterion(preds, y)
        if mode == 'fp16':
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        props['loss'] += loss.item()
    L = len(loader)
    props = {k:v/L for k,v in props.items()}
    return props

def status(epoch, train_props, cv_props, epochs):
    '''generate summary during training'''
    s0 = 'epoch {0}/{1}\n'.format(epoch, epochs)
    s1, s2 = '',''
    for k,v in train_props.items():
        s1 = s1 + 'train_'+ k + ': ' + str(v) + ' '
    for k,v in cv_props.items():
        s2 = s2 + 'cv_'+ k + ': ' + str(v) + ' '
    print(s0 + s1 + s2)

def main():
    train_iter = torch.load('train_loader.pt')
    test_iter = torch.load('test_loader.pt')
    model = Model(dim=768, vocab_sz=vocab_sz, num_enc=2, factor=2).to(device)
    criterion = nn.CrossEntropyLoss()
    print(model)
    print(len(train_iter), len(test_iter))

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    epochs = 30
    best_loss = np.inf

    if mode == 'fp16':
        opt_level = 'O1'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, loss_scale='dynamic')

    opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level, loss_scale='dynamic')

    restore = False
    if restore:
        checkpoint = torch.load('encoder.pth.tar')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])
        best_loss = checkpoint['loss']

    # Training loop
    for epoch in tqdm(range(epochs)):
        train_props = train(train_iter, model, optimizer, criterion, mode)
        cv_props = evaluate(test_iter, model, criterion)
        status(epoch, train_props, cv_props, epochs)

        loss = train_props['loss']
        if loss < best_loss:
            best_loss = loss
            amp_ = amp.state_dict() if mode == 'fp16' else None
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp_,
                'loss': loss
            }
            print('checkpointing..')
            torch.save(checkpoint, 'encoder.pth.tar')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('cancelling..')