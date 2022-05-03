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
from tasks.image_retrieval_model import RetrievalModel
from tasks.image_retrieval_data import RetrievalDataset, RetrievalTorchDataset,RetrievalDataset_val, RetrievalTorchDataset_val, RetrievalEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:

    dset = RetrievalDataset(splits)
    tset = RetrievalTorchDataset(dset)
    evaluator = RetrievalEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,  #num_workers=0
        drop_last=drop_last, pin_memory=True
    )
    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)
def get_data_tuple_val(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:

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
        if args.train!='':
            self.train_tuple = get_data_tuple(
                args.train, bs=args.batch_size, shuffle=True, drop_last=True
            )
        if args.valid  != '':
            self.valid_tuple = get_data_tuple(
                args.valid, bs=args.batch_size,
                shuffle=False, drop_last=False
            )
        self.model = RetrievalModel()
        self.margin=0.2
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model)
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
        self.ce_loss =nn.BCELoss()
        if args.train!='':
            if 'bert' in args.optim:
                batch_per_epoch = len(self.train_tuple.loader)
                t_total = int(batch_per_epoch * args.epochs)
                print("BertAdam Total Iters: %d" % t_total)
                from lxrt.optimization import BertAdam
                self.optim = BertAdam(list(self.model.parameters()),
                                      lr=args.lr,
                                      warmup=0.1,
                                      t_total=t_total)
            else:
                self.optim = args.optimizer(self.model.parameters(), args.lr)
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)
    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        for epoch in range(args.epochs):
            losses=0
            item_num=len(loader)
            for i, (feats, boxes, input_ids,token_position,input_mask,segment_ids, target) in iter_wrapper(enumerate(loader)):
                self.model.train()
                self.optim.zero_grad()
                feats, boxes, input_ids,token_position,input_mask,segment_ids, target = feats.cuda(),boxes.cuda(), input_ids.cuda(),token_position.cuda(),input_mask.cuda(),segment_ids.cuda(),target.cuda()
                feats=feats.view(-1,feats.size(2),feats.size(3))
                boxes=boxes.view(-1,boxes.size(2),boxes.size(3))
                input_ids=input_ids.view(-1,input_ids.size(2))
                input_mask=input_mask.view(-1,input_mask.size(2))
                segment_ids=segment_ids.view(-1,segment_ids.size(2))
                token_position=token_position.view(-1,token_position.size(2))
                logit = self.model(input_ids,token_position, segment_ids, input_mask, feats,boxes)
                logit = logit.view(-1,2)
                target = target.view(-1,2)
                loss = self.ce_loss(logit, target)
                loss = loss * logit.size(1)
                loss.backward()
                losses += loss.detach()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                _, preds = torch.max(logit, 1)
            losses=float(losses)/float(item_num)
            log_str = "\nEpoch %d: loss is %0.2f \n" % (epoch, losses)
            if self.valid_tuple is not None:  # Do Validation
                loss,score = self.evaluate(self.valid_tuple,epoch)
                log_str += "\nEpoch %d: loss is %0.2f \n" % (epoch,loss)
            print(log_str, end='')
            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()
        self.save("LAST")
    def evaluate(self, eval_tuple: DataTuple,epoch):
        dset, loader, evaluator = eval_tuple
        self.model.eval()
        losses=0
        batch_score=0
        item_num=len(loader)
        for i, (feats, boxes, input_ids,token_position,input_mask,segment_ids, target) in enumerate(loader):
            feats, boxes, input_ids,token_position,input_mask,segment_ids, target = feats.cuda(),boxes.cuda(), input_ids.cuda(),token_position.cuda(),input_mask.cuda(),segment_ids.cuda(),target.cuda()
            num_options=input_ids.size(1)
            feats=feats.view(-1,feats.size(2),feats.size(3))
            boxes=boxes.view(-1,boxes.size(2),boxes.size(3))
            input_ids=input_ids.view(-1,input_ids.size(2))
            segment_ids=segment_ids.view(-1,segment_ids.size(2))
            input_mask=input_mask.view(-1,input_mask.size(2))
            token_position=token_position.view(-1,token_position.size(2))
            with torch.no_grad():
                logit = self.model(input_ids,token_position, segment_ids, input_mask, feats,boxes)
                logit=logit.view(-1,2)
                target=target.view(-1,2)
                loss = self.ce_loss(logit, target)
                losses+=loss.detach()
            _, preds = torch.max(logit, 1)
            self.save("Epoch0%d"%(epoch))
        losses=float(losses)/float(item_num)
        batch_score=0.0
        return losses,batch_score
    def evaluate_test(self, test_tuple: DataTuple):
        eval_str=""
        score_matrix = np.zeros((5000, 1000))
        target_matrix = np.zeros((5000, 1000))
        rank_matrix = np.ones((5000)) * 1000
        count=0
        results = []
        dset, loader, evaluator = test_tuple
        for i, (feats, boxes, input_ids,token_position,input_mask,segment_ids, target,image_idx,caption_idx) in enumerate(loader):
            feats, boxes, input_ids,token_position,input_mask,segment_ids, target = feats.cuda(),boxes.cuda(), input_ids.cuda(),token_position.cuda(),input_mask.cuda(),segment_ids.cuda(),target.cuda()
            feats = feats.squeeze(0)
            boxes = boxes.squeeze(0)
            input_ids=input_ids.repeat(500,1)
            input_mask=input_mask.repeat(500,1)
            segment_ids=segment_ids.repeat(500,1)
            token_position = token_position.repeat(500,1)
            #print(image_entries)
            with torch.no_grad():
                logit = self.model(input_ids, token_position,segment_ids, input_mask, feats,boxes)
                score_matrix[
                        caption_idx, image_idx * 500 : (image_idx + 1) * 500
                    ] = (logit[:,1].view(-1).cpu().numpy())
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
                    eval_str+=(
                        "%d Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f\n"
                        % (count, r1, r5, r10, medr, meanr)
                    )
                    results.append(np.argsort(-score_matrix[caption_idx]).tolist()[:20])
            count += 1
        r1 = 100.0 * np.sum(rank_matrix < 1) / len(rank_matrix)
        r5 = 100.0 * np.sum(rank_matrix < 5) / len(rank_matrix)
        r10 = 100.0 * np.sum(rank_matrix < 10) / len(rank_matrix)

        medr = np.floor(np.median(rank_matrix) + 1)
        meanr = np.mean(rank_matrix) + 1
        eval_str+="\n\n************************************************\n"
        print("************************************************")
        print(
            "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
            % (r1, r5, r10, medr, meanr)
        )
        eval_str+=(
            "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f\n"
            % (r1, r5, r10, medr, meanr)
        )
        print("************************************************")
        eval_str+="************************************************"
        with open(self.output + "/eval.log", 'a') as f:
            f.write(eval_str)
            f.flush()
    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    retrieval = Retrieval()
    if args.load is not None:
        retrieval.load(args.load)
    if args.test!=None:
        print("----------------------")
        data_test=get_data_tuple_val(args.test, bs=1,
                           shuffle=False, drop_last=False)
        retrieval.evaluate_test(data_test)
    elif 'train' in args.train:
        retrieval.train(retrieval.train_tuple, retrieval.valid_tuple)


