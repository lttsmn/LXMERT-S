# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from lxrt.entry import convert_sents_to_features_new
from param import args
from utils import load_obj_tsv,load_obj_tsv_new
import jsonlines
import platform
import _pickle as cPickle
# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
#TINY_IMG_NUM = 512
TINY_IMG_NUM = 200
FAST_IMG_NUM = 5000

# The path to data and image features.
data_root = 'data/image_retrieval/'
SPLIT2NAME = {
    'train': 'flickr30k_test2014',
    'valid': 'flickr30k_tran2014',
}
class RetrievalDataset:
    def __init__(self,splits):
        self.image_entries = []
        self.splits=splits
        self.imgid2entry = {}
        count = 0
        self.data = []
        self.all_data=[]
        if splits=='train':
            annotations_jsonpath=os.path.join(data_root,"flickr30k_train2014.json")
        else:
            annotations_jsonpath=os.path.join(data_root,"flickr30k_test2014.json")

        self.all_data.extend(json.load(open(annotations_jsonpath)))
        for annotation in self.all_data:
            image_id=annotation['img_path']
            self.image_entries.append(image_id)
            self.imgid2entry[int(image_id.split('.')[0])]=[]
            for sent in annotation['sentences']:
                self.data.append({"caption": sent, "image_id": image_id})
                self.imgid2entry[int(image_id.split('.')[0])].append(count)
                count += 1
        print("Load %d data from test set." % (len(self.data)))
        print(len(self.image_entries))

    def __len__(self):
        return len(self.data)
class RetrievalTorchDataset(Dataset):
    def __init__(self, dataset: RetrievalDataset):
        super().__init__()
        self.raw_dataset = dataset
        self.image_entries,self.id2entry,self.data= self.raw_dataset.image_entries,self.raw_dataset.imgid2entry,self.raw_dataset.data
        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None
        img_data=[]
        if self.raw_dataset.splits=='train':
            img_data.extend(load_obj_tsv('lxmert_data/flickr30k/flickr30k_train2014.tsv',topk=topk))
        else:
            img_data.extend(load_obj_tsv('lxmert_data/flickr30k/flickr30k_test2014.tsv',topk=topk))
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum
        if args.tiny:
            self.data = []
            for datum in self.raw_dataset.data:
                if datum['image_id'] in self.imgid2img:
                    self.data.append(datum)
        image_info = cPickle.load(open(os.path.join(data_root, "hard_negative.pkl"), "rb"))
        for key, value in image_info.items():
            setattr(self, key, value)
        self.train_imgId2pool = {
                imageId: i for i, imageId in enumerate(self.train_image_list)
            }
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: int):
        datum1 = self.data[index]
        input_ids1,token_position1,input_mask1,segment_ids1 = convert_sents_to_features_new(datum1['caption'])
        img_id1 = datum1['image_id']
        img_info1 = self.imgid2img[img_id1]
        obj_num1 = img_info1['num_boxes']
        feats1 = img_info1['features'].copy()
        boxes1 = img_info1['boxes'].copy()
        assert obj_num1 == len(boxes1) == len(feats1)
        img_h1, img_w1 = img_info1['img_h'], img_info1['img_w']
        boxes1 = boxes1.copy()
        boxes1[:, (0, 2)] /= img_w1
        boxes1[:, (1, 3)] /= img_h1
        np.testing.assert_array_less(boxes1, 1+1e-5)
        np.testing.assert_array_less(-boxes1, 0+1e-5)
        while True:
            # sample a random image:
            datum2 = random.choice(self.data)
            if datum2['image_id'] != img_id1:
                break
        img_id2 = datum2['image_id']
        img_info2 = self.imgid2img[img_id2]
        obj_num2 = img_info2['num_boxes']
        feats2 = img_info2['features'].copy()
        boxes2 = img_info2['boxes'].copy()
        assert obj_num2 == len(boxes2) == len(feats2)
        img_h2, img_w2 = img_info2['img_h'], img_info2['img_w']
        boxes2 = boxes2.copy()
        boxes2[:, (0, 2)] /= img_w2
        boxes2[:, (1, 3)] /= img_h2
        np.testing.assert_array_less(boxes2, 1+1e-5)
        np.testing.assert_array_less(-boxes2, 0+1e-5)
        input_ids2,token_position2,input_mask2,segment_ids2=input_ids1.copy(),token_position1.copy(),input_mask1.copy(),segment_ids1.copy()
        if self.raw_dataset.splits=="train":
            rand_img_id_pool = self.train_hard_pool[self.train_imgId2pool[int(img_id1.split('.')[0])]]
            pool_img_idx = int(
                    rand_img_id_pool[np.random.randint(1, len(rand_img_id_pool))]
            )
            img_id3 = self.train_image_list[pool_img_idx]
            datum3=self.data[random.choice(self.id2entry.get(img_id3))]
        else:
            while True:
                datum3 = random.choice(self.data)
                if datum3['image_id'] != img_id1:
                    break
        feats3 = feats1.copy()
        boxes3=boxes1.copy()
        input_ids3,token_position3,input_mask3,segment_ids3 = convert_sents_to_features_new(datum3['caption'])
        while True:
            datum4 = random.choice(self.data)
            if datum4['image_id'] != img_id1:
                break
        input_ids4,token_position4,input_mask4,segment_ids4 = convert_sents_to_features_new(datum4['caption'])
        feats4 = feats1.copy()
        boxes4=boxes1.copy()
        feats_all=[feats1, feats2, feats3, feats4]
        boxes_all=[boxes1, boxes2, boxes3, boxes4]
        input_ids_all=[input_ids1, input_ids2, input_ids3, input_ids4]
        input_mask_all=[input_mask1, input_mask2, input_mask3, input_mask4]
        segment_ids_all=[segment_ids1, segment_ids2, segment_ids3, segment_ids4]
        token_position_all=[token_position1, token_position2, token_position3, token_position4]
        feats=torch.tensor(feats_all)
        boxes=torch.tensor(boxes_all)
        input_ids=torch.tensor(input_ids_all)
        input_mask=torch.tensor(input_mask_all)
        segment_ids=torch.tensor(segment_ids_all)
        token_position=torch.tensor(token_position_all)
        target = torch.tensor([[0,1],[1,0],[1,0],[1,0]]).float()
        return feats, boxes, input_ids,token_position,input_mask,segment_ids, target
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
            for sent in annotation['sentences']:
                self.data.append({"caption": sent, "image_id": image_id})
        print("Load %d data from test set." % (len(self.data)))
        print(len(self.image_entries))

    def __len__(self):
        return len(self.data)
class RetrievalTorchDataset_val(Dataset):
    def __init__(self, dataset: RetrievalDataset_val):
        super().__init__()
        self.raw_dataset = dataset
        self.image_entries, self.data= self.raw_dataset.image_entries,self.raw_dataset.data
        img_ids=self.image_entries.copy()
        img_data=[]
        img_data.extend(load_obj_tsv('lxmert_data/flickr30k/flickr30k_test2014.tsv'))
        print(len(img_data))
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        self.features_all = np.zeros((1000,36, 2048))
        self.boxes_all=np.zeros((1000,36,4))
        #  feats, boxes, input_ids,input_mask,segment_ids,target
        for i, image_id in enumerate(self.image_entries):
            img_info = self.imgid2img[image_id]
            feats=img_info['features'].copy()
            boxes=img_info['boxes'].copy()
            img_h, img_w = img_info['img_h'], img_info['img_w']
            boxes = boxes.copy()
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            self.features_all[i] =feats
            self.boxes_all[i]=boxes
        self.features_all = torch.Tensor(self.features_all).float()
        self.boxes_all = torch.Tensor(self.boxes_all).float()
        print("Load %d data from image." % (len(self.features_all)))
    def __len__(self):
        return len(self.data)*2
    def __getitem__(self, index: int):

        caption_idx=int(index/2)
        image_idx = index%2
        if image_idx == 0:
            image_entries = self.image_entries[:500]
            features_all = self.features_all[:500]
            boxes_all = self.boxes_all[:500]
        else:
            image_entries = self.image_entries[500:]
            features_all = self.features_all[500:]
            boxes_all = self.boxes_all[500:]
        entry=self.data[caption_idx]
        input_ids,token_position,input_mask,segment_ids = convert_sents_to_features_new(entry['caption'])
        target_all = torch.zeros(500)
        for i, image_id in enumerate(image_entries):
            if image_id == entry["image_id"]:
                target_all[i] = 1
        input_ids=torch.tensor(input_ids)
        input_mask=torch.tensor(input_mask)
        segment_ids=torch.tensor(segment_ids)
        token_position=torch.tensor(token_position)
        return features_all, boxes_all, input_ids,token_position,input_mask,segment_ids, target_all,image_idx,caption_idx
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



