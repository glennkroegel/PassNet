import torch
import spacy
import pandas as pd
import numpy as np

# https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial

def accuracy(input, targs):
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    with torch.no_grad():
        # n = targs.shape[0]
        inp = input.argmax(dim=-1).view(-1)
        ys = targs.view(-1)
        return (inp==ys).float().mean()

def bce_accuracy(input, targs):
    with torch.no_grad():
        preds = torch.round(input)
        acc = (preds == targs).float().mean()
    return acc

def acc_part(input, targs, ix=0):
    with torch.no_grad():
        import pdb; pdb.set_trace()
        ixs = targs == ix
        res = input[ixs].argmax(dim=-1).view(-1)
        return (res == targs[ixs]).float().mean()

def get_masks(src, tgt, pad_ix=torch.LongTensor([1]), no_peak=True):
    mask_src = (src == pad_ix).unsqueeze(1)
    mask_tgt = (tgt == pad_ix).unsqueeze(1)

    if no_peak:
        sz = mask_tgt.size(1)
        temp = torch.ones(1, sz, sz)
        nopeak_mask = (torch.triu(temp, diagonal=1) == 0)
        mask_tgt = mask_tgt & nopeak_mask
        
    return mask_src, mask_tgt

def make_dataset(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    func = lambda x: tokenizer.encode(x, return_tensors='pt', max_length=512, pad_to_max_length=True).view(-1)
    
    pad_ix = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    TEXT = data.Field(use_vocab=False, tokenize=func, pad_token=pad_ix)
    # abstract = data.Field(use_vocab=False, tokenize=func, pad_token=pad_ix)
    
    fields = (('title', TEXT),('abstract', TEXT))
    train, valid = data.TabularDataset.splits(path='.', train='train.csv', validation='cv.csv', format='csv', fields=fields)

    train_iter, test_iter = data.BucketIterator.splits((train, valid), batch_sizes=(16, 16), device=device, sort_key=lambda x: len(x.abstract))

    return train_iter, test_iter, tokenizer