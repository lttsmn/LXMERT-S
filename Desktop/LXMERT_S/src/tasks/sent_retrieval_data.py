# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from param import args
from utils import load_obj_tsv_new,load_obj_tsv
import jsonlines
import platform
import platform
# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 200
FAST_IMG_NUM = 5000

# The path to data and image features.
data_root = 'data/image_retrieval'

SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'val2014',
    }
class RetrievalDataset_val:
    def __init__(self):
        self.image_entries = []
        self.data = []
        self.all_data=[]
        annotations_jsonpath=os.path.join(data_root,"flickr30k_test2014.json")
        self.all_data.extend(json.load(open(annotations_jsonpath)))
        for annotation in self.all_data:
            image_id=annotation['img_path']
            self.image_entries.append(image_id)
            self.data.append({"caption": annotation['sentences'][0], "image_id": image_id})
        print("Load %d data from test set." % (len(self.data)))
        print(len(self.image_entries))

    def __len__(self):
        return len(self.data)
class RetrievalTorchDataset_val(Dataset):
    def __init__(self, dataset: RetrievalDataset_val):
        super().__init__()
        self.raw_dataset = dataset
        self.image_entries, self.data= self.raw_dataset.image_entries,self.raw_dataset.data
        self.img_data=[]
        self.img_data.extend(load_obj_tsv('data/flickr30k/flickr30k_test2014.tsv'))
        self.imgid2img = {}
        for img_datum in self.img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        self.sent_all = []
        for i, datum in enumerate(self.data):
            sent=datum["caption"]
            self.sent_all.append(sent)
        print("Load %d data from image." % (len(self.img_data)))
    def __len__(self):
        return len(self.img_data)*2
    def __getitem__(self, index: int):

        image_idx=int(index/2)
        caption_idx = index%2
        if caption_idx == 0:
            image_entries = self.image_entries[:500]
            sent_all = self.sent_all[:500]
        else:
            image_entries = self.image_entries[500:]
            sent_all = self.sent_all[500:]
        img_info =self.img_data[image_idx]
        img_info =self.img_data[image_idx]
        feats = img_info['features'].copy()
        img_id=img_info['img_id']
        boxes = img_info['boxes'].copy()
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        target_all = torch.zeros(500)
        for i, image_id in enumerate(image_entries):
            if image_id == img_id:
                target_all[i] = 1
        return sent_all, feats, boxes, target_all,image_idx,caption_idx
class RetrievalEvaluator:
    def __init__(self, dataset: RetrievalDataset_val):
        self.dataset = dataset
        self.output = args.output

    def evaluate(self, quesid2ans: dict):
        score = 0.
        item=0
        for quesid, ans in quesid2ans.items():
            item+=1
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)



