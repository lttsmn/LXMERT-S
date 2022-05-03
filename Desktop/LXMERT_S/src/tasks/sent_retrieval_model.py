# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder_sent
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20

class RetrievalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder_sent(
            args,
            max_seq_length=MAX_VQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim

        if args.zero_shot:
            self.seq_relationship_new=nn.Linear(hid_dim, 2)
        else:
            self.seq_relationship_new=nn.Linear(hid_dim, 2)
    def forward(self,sents, feats,boxes):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sents, (feats, boxes))
        logit = self.seq_relationship_new(x)
        return logit


