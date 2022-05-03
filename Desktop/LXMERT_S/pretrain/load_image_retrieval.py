# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import torch

def load_lxmert_qa(path, model):
    # Load state_dict from snapshot file
    print("Load LXMERT pre-trained model from %s" % path)
    state_dict = torch.load("%s_LXRT.pth" % path)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[len("module."):]] = value
        else:
            new_state_dict[key] = value
    state_dict = new_state_dict
    bert_state_dict={}
    for key, value in state_dict.items():
        if key.startswith('bert.'):
            bert_state_dict[key] = value
    answer_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("cls.seq_relationship."):
            answer_state_dict[key.replace('cls.seq_relationship.', 'seq_relationship_new.')] = value


    model.lxrt_encoder.model.load_state_dict( bert_state_dict, strict=False)

    model.load_state_dict(answer_state_dict, strict=False)



