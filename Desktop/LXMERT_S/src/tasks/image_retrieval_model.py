# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
import torch
from param import args
from lxrt.entry import LXRTEncoder_new
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_TRIEVAL_LENGTH = 20

class RetrievalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder_new(
            args,
            max_seq_length=MAX_TRIEVAL_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        if args.zero_shot:

            self.seq_relationship_new=nn.Linear(hid_dim, 2)
        else:
            self.seq_relationship_new=nn.Linear(hid_dim, 2)
    def forward(self, input_ids,token_position, segment_ids, input_mask, feats,boxes):
        x = self.lxrt_encoder( input_ids, token_position,segment_ids, input_mask, (feats, boxes))
        logit = self.seq_relationship_new(x)
        if args.zero_shot:
            return logit
        else:
            logit = torch.softmax(logit,dim=1)
            return logit


