# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np
from param import args
from pretrain.load_image_retrieval import load_lxmert_qa
from tasks.image_retrieval_data import RetrievalDataset_val, RetrievalTorchDataset_val, RetrievalEvaluator
from tasks.image_retrieval_model import RetrievalModel
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:

    dset = RetrievalDataset_val()
    tset = RetrievalTorchDataset_val(dset)
    evaluator = RetrievalEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=False, num_workers=0,  #num_workers=0
        drop_last=drop_last, pin_memory=True
    )
    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)
class Retrieval:
    def __init__(self):
        self.test_tuple=get_data_tuple(
            args.test, bs=1,
            shuffle=False, drop_last=False
        )
        self.model = RetrievalModel()
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model)
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
    def evaluate(self, test_tuple: DataTuple):
        score_matrix = np.zeros((5000, 1000))
        target_matrix = np.zeros((5000, 1000))
        rank_matrix = np.ones((5000)) * 1000
        count=0
        results = []
        dset, loader, evaluator = test_tuple
        for i, (feats, boxes, input_ids,token_position,input_mask,segment_ids, target,image_idx,caption_idx) in enumerate(loader):
            feats, boxes, input_ids,token_position,input_mask,segment_ids, target = feats.cuda(),boxes.cuda(), input_ids.cuda(),token_position.cuda() ,input_mask.cuda(),segment_ids.cuda(),target.cuda()
            feats = feats.squeeze(0)
            boxes = boxes.squeeze(0)
            input_ids=input_ids.repeat(500,1)
            input_mask=input_mask.repeat(500,1)
            segment_ids=segment_ids.repeat(500,1)
            with torch.no_grad():
                logit = self.model(input_ids, token_position, segment_ids, input_mask, feats,boxes)
                score_matrix[
                        caption_idx, image_idx * 500 : (image_idx + 1) * 500
                    ] = (torch.softmax(logit, dim=1)[:,1].view(-1).cpu().numpy())
                target_matrix[
                        caption_idx, image_idx * 500 : (image_idx + 1) * 500
                    ] = (target.view(-1).float().cpu().numpy())
                if image_idx.item() == 1:
                    rank = np.where(
                        (
                            np.argsort(-score_matrix[caption_idx])
                            == np.where(target_matrix[caption_idx] == 1)[0][0]
                        )
                        == 1
                    )[0][0]
                    rank_matrix[caption_idx] = rank

                    rank_matrix_tmp = rank_matrix[: caption_idx + 1]
                    r1 = 100.0 * np.sum(rank_matrix_tmp < 1) / len(rank_matrix_tmp)
                    r5 = 100.0 * np.sum(rank_matrix_tmp < 5) / len(rank_matrix_tmp)
                    r10 = 100.0 * np.sum(rank_matrix_tmp < 10) / len(rank_matrix_tmp)

                    medr = np.floor(np.median(rank_matrix_tmp) + 1)
                    meanr = np.mean(rank_matrix_tmp) + 1
                    print(
                        "%d Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
                        % (count, r1, r5, r10, medr, meanr)
                    )

                    results.append(np.argsort(-score_matrix[caption_idx]).tolist()[:20])
            count += 1
        r1 = 100.0 * np.sum(rank_matrix < 1) / len(rank_matrix)
        r5 = 100.0 * np.sum(rank_matrix < 5) / len(rank_matrix)
        r10 = 100.0 * np.sum(rank_matrix < 10) / len(rank_matrix)

        medr = np.floor(np.median(rank_matrix) + 1)
        meanr = np.mean(rank_matrix) + 1

        print("************************************************")
        print(
            "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
            % (r1, r5, r10, medr, meanr)
        )
        print("************************************************")
    def load(self, path):
        print("Load BERT extractor from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        self.model.load_state_dict(state_dict)

    def load_lxmert(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        self.model.load_state_dict(state_dict, strict=False)
if __name__ == "__main__":
    retrieval = Retrieval()
    retrieval.evaluate(retrieval.test_tuple)



