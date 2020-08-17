import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from config import encoder_path, max_len, vocab_sz
from encoder import *

class Mutate(nn.Module):
    def __init__(self, model, max_len):
        super(Mutate, self).__init__()
        self.model = model
        self.model.eval()
        self.max_len = torch.tensor(max_len)

    def forward(self, x: Tensor, k: Tensor) -> Tensor:
        k = int(k)
        attn_mask = torch.ones_like(x)
        attn_mask[x == 0] = torch.tensor(0)
        outp = self.model(x, attn_mask)
        pred_mask = (x != 2) & (x != 3) & (x != 0)
        pred_mask_ixs = pred_mask.nonzero()
        logits_at_mask = outp[pred_mask_ixs[:,0], pred_mask_ixs[:,1]]
        preds = logits_at_mask.topk(k).indices
        l = pred_mask.sum(dim=1)
        mutated = x.repeat_interleave(l*k, 0)
        pred_locs = pred_mask_ixs.repeat_interleave(k, 0)[:,1]
        pred_locs = torch.cat([torch.arange(len(mutated)).unsqueeze(1), pred_locs.unsqueeze(1)], dim=1)
        mutated[pred_locs[:,0], pred_locs[:,1]] = preds.view(-1)
        return mutated

def main():
    device = torch.device('cpu')
    mdl = Model(dim=768, vocab_sz=vocab_sz, num_enc=2, factor=2).to(device)
    # state = torch.load(encoder_path)
    # mdl.load_state_dict(state)
    mdl.eval()

    mutate = Mutate(mdl, max_len)
    scripted_module = torch.jit.script(mutate)
    torch.jit.save(scripted_module, 'sm_mutate.pt')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('cancelling..')