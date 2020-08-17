'''
created_by: Glenn Kroegel
date: 14 May 2020
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import glob
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tokenizer import *

def collate_dynamic(batch):
    if len(batch) == 1:
        batch = batch[0]
    else:
        batch = [x for a in batch for x in a]
    encodings = tokenizer.encode_batch(batch)
    xs = []
    attn_mask = []
    pred_ixs = []
    for i, enc in enumerate(encodings):
        x = torch.LongTensor(enc.ids)
        l = x.nonzero()
        x = x.unsqueeze(0)
        xs.append(x)
        m = enc.attention_mask
        attn_mask.append(m)
        col = torch.arange(1,l+1)
        row = torch.zeros_like(col) + i
        loc = torch.cat([row.unsqueeze(1),col.unsqueeze(1)], dim=1)
        pred_ixs.append(loc)
    xs = torch.cat(xs)
    attn_mask = torch.cat(attn_mask)
    pred_ixs = torch.cat(pred_ixs)
    if len(xs) > 3000:
        print(len(xs))
        xs = xs[:3000]
        attn_mask = attn_mask[:3000]
        pred_ixs = pred_ixs[:3000]
    return xs, attn_mask, pred_ixs

def collate_fn(batch):
    encodings = tokenizer.encode_batch(batch)
    tokens = []
    attn_mask = []
    spec_tokens = []
    for x in encodings:
        tokens.append(x.ids)
        attn_mask.append(x.attention_mask)
        spec_tokens.append(x.special_tokens_mask)
    x = torch.LongTensor(tokens)
    attn_mask = torch.LongTensor(attn_mask)
    spec_tokens = torch.LongTensor(spec_tokens)
    valid = torch.zeros_like(x)
    valid[spec_tokens == 0] = 1
    valid_ixs = valid.nonzero()
    masked = dist.sample((len(valid_ixs), )).long()
    attn_mask[valid_ixs[:,0], valid_ixs[:,1]] = masked
    masked_ixs = valid_ixs[masked == 0]
    pred_ixs = torch.zeros_like(attn_mask)
    pred_ixs[masked_ixs[:,0], masked_ixs[:,1]] = 1
    return x, attn_mask, pred_ixs

class FileDataset(Dataset):
    def __init__(self, fname, num=None, split_token=False):
        self.data = []
        with open(fname, 'r') as infile:
            while True:
                line = infile.readline()
                if not line:
                    break
                line = line.replace('\n', '')
                if len(line) < 5:
                    continue
                if split_token:
                    pair = tuple(line.split(split_token))
                    if len(pair) != 2:
                        continue
                    self.data.append(pair)
                else:
                    self.data.append(line)
        if num:
            self.data = self.data[:num]

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

class FileListDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __getitem__(self, i):
        fname = self.files[i]
        with open(fname, 'r') as infile:
            data = []
            while True:
                line = infile.readline()
                if not line:
                    break
                line = line.replace('\n', '')
                if len(line) < 2:
                    continue
                else:
                    data.append(line)
            if len(data) == 0:
                print(1)
        return data

    def __len__(self):
        return len(self.files)

if __name__ == '__main__':

    pw_files = glob.glob('data/batches_250/*')
    n = 10000
    print(n)
    pw_files = random.sample(pw_files, k=n)
    L = len(pw_files)
    n = int(0.95*L)
    train_files = random.sample(pw_files, k=n)
    test_files = set(pw_files) - set(train_files)
    test_files = list(test_files)
    print(len(train_files), len(test_files))
    train_ds = FileListDataset(train_files)
    test_ds = FileListDataset(test_files)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    torch.save(train_loader, 'train_loader.pt')
    torch.save(test_loader, 'test_loader.pt')    