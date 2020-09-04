import argparse
import json
import logging
import os
import random
import time as Time
from typing import List

import numpy as np
import torch
from scipy import spatial

from utils import *
from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator


class Tester:
    left: List[int] = []
    right: List[int] = []
    seeds = []
    linkEmbedding = []
    kg1E = []
    kg2E = []
    EA_results = {}

    def read_entity_align_list(self, entity_align_file_path):
        ret = []
        with open(entity_align_file_path, encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                self.left.append(int(th[0]))
                self.right.append(int(th[1]))
                ret.append((int(th[0]), int(th[1])))
            self.seeds = ret

    def XRA(self, entity_embedding_file_path):
        self.linkEmbedding = []
        with open(entity_embedding_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            aline = lines[i].strip()
            aline_list = aline.split()
            self.linkEmbedding.append(aline_list)

    @staticmethod
    def get_vec(entities_embedding, id_list, device="cuda"):
        tensor = torch.LongTensor(id_list).view(-1, 1).to(device)
        return entities_embedding(tensor).view(-1, 200).cpu().detach().numpy()

    @staticmethod
    def get_vec2(entities_embedding, id_list, device="cuda"):
        tensor = torch.LongTensor(id_list).view(-1).to(device)
        vec = torch.index_select(
            entities_embedding,
            dim=0,
            index=tensor
        ).view(-1, 200).cpu().detach().numpy()
        return vec

    # def calculate(self, top_k=(1, 10, 50, 100)):
    #     Lvec = np.array([self.linkEmbedding[e1] for e1, e2 in self.seeds])
    #     Rvec = np.array([self.linkEmbedding[e2] for e1, e2 in self.seeds])
    #     return self.get_hits(Lvec, Rvec, top_k)

    def get_hit(self, left_entity_ids, right_entity_ids, left_entity_vec, all_entity_vec, top_k=(1, 10, 50, 100)):
        distance_left_i_to_all_j = spatial.distance.cdist(left_entity_vec, all_entity_vec, metric='euclidean')
        top_lr = [0] * len(top_k)
        for i in range(len(left_entity_ids)):  # 对于每个KG1实体
            rank = distance_left_i_to_all_j[i, :].argsort()
            rank_index = np.where(rank == right_entity_ids[i])[0][0]
            for k in range(len(top_k)):
                if rank_index < top_k[k]:
                    top_lr[k] += 1
        return top_lr

    def get_hits(self, left_entity_ids, right_entity_ids, left_entity_vec, right_entity_vec, all_entity_vec, top_k=(1, 10, 50, 100)):
        # Lvec nxd, Rvec mxd, sim nxm
        # sim[i, j]为Lvec第i个实体和Rvec第j个实体的距离
        top_lr = self.get_hit(left_entity_ids, right_entity_ids, left_entity_vec, all_entity_vec, top_k)
        top_rl = self.get_hit(right_entity_ids, left_entity_ids, right_entity_vec, all_entity_vec, top_k)
        print('For each left:')
        left = []
        for i in range(len(top_lr)):
            hits = top_k[i]
            hits_value = top_lr[i] / len(self.seeds) * 100
            left.append((hits, hits_value))
            print('Hits@%d: %.2f%%' % (hits, hits_value))
        print('For each right:')
        right = []
        for i in range(len(top_rl)):
            hits = top_k[i]
            hits_value = top_rl[i] / len(self.seeds) * 100
            right.append((hits, hits_value))
            print('Hits@%d: %.2f%%' % (hits, hits_value))

        return {
            "left": left,
            "right": right,
        }


t = Tester()
t.read_entity_align_list('data/fr_en/ref_ent_ids')  # 得到已知对齐实体


class run():
    def __init__(self, isCUDA):
        self.isCUDA = isCUDA

    def save_model(self, model, optimizer):

        save_path = "./result/fr_en/model.pth"
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}

        torch.save(state, save_path)

    def save_entitylist(self, model):
        dir = "./result/fr_en/ATentsembed.txt"
        entityVectorFile = open(dir, 'w')
        temparray = model.entity_embedding.cpu().detach().numpy()

        for i in range(len(temparray)):
            entityVectorFile.write(
                str(temparray[i].tolist()).replace('[', '').replace(']', '').replace(',', ' '))
            entityVectorFile.write("\n")

        entityVectorFile.close()

    def train(self, train_triples, entity2id, att2id, value2id):
        self.nentity = len(entity2id)
        self.nattribute = len(att2id)
        self.nvalue = len(value2id)

        self.kge_model = KGEModel(
            nentity=self.nentity,
            nrelation=self.nattribute,
            nvalue=self.nvalue,
            hidden_dim=200,
            gamma=24.0,
        )
        current_learning_rate = 0.001

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.kge_model.parameters()),
            lr=current_learning_rate
        )

        # self.optimizer = torch.optim.SGD(self.kge_model.parameters(), lr=current_learning_rate)

        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, self.nentity, self.nattribute, self.nvalue, 256, 'head-batch'),
            batch_size=1024,
            shuffle=False,
            num_workers=4,
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, self.nentity, self.nattribute, self.nvalue, 256, 'tail-batch'),
            batch_size=1024,
            shuffle=False,
            num_workers=4,
            collate_fn=TrainDataset.collate_fn
        )
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        if self.isCUDA == 1:
            self.kge_model = self.kge_model.cuda()

        # start training
        print("start training")
        init_step = 0
        # Training Loop
        starttime = Time.time()

        steps = 20001
        printnum = 1000
        lastscore = 100

        for step in range(init_step, steps):
            loss = self.kge_model.train_step(self.kge_model, self.optimizer, train_iterator, self.isCUDA)

            if step % printnum == 0:
                endtime = Time.time()
                print("step:%d, cost time: %s, loss is %.6f" % (step, round((endtime - starttime), 3), loss))
                print("属性消融实验")
                left_ids = torch.LongTensor(t.left).view(-1).to("cuda")
                left_vec = torch.index_select(
                    self.kge_model.entity_embedding,
                    dim=0,
                    index=left_ids
                ).view(-1, 200).cpu().detach().numpy()
                right_ids = torch.LongTensor(t.right).view(-1).to("cuda")
                right_vec = torch.index_select(
                    self.kge_model.entity_embedding,
                    dim=0,
                    index=right_ids
                ).view(-1, 200).cpu().detach().numpy()
                all_entity_vec = t.get_vec2(self.kge_model.entity_embedding, entity2id)
                hits = t.get_hits(left_ids.cpu().detach().numpy(), right_ids.cpu().detach().numpy(), left_vec, right_vec, all_entity_vec)
                left_hits_10 = hits["left"][2][1]
                right_hits_10 = hits["right"][2][1]
                score = (left_hits_10 + right_hits_10) / 2
                print("score=", score)
                if loss < lastscore:
                    lastscore = loss
                    self.save_entitylist(self.kge_model)
                    self.save_model(self.kge_model, self.optimizer)


def openDetailsAndId(dir, sp="\t"):
    idNum = 0
    list = []
    with open(dir, encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            DetailsAndId = line.strip().split(sp)
            list.append(DetailsAndId[0])
            idNum += 1
    return idNum, list


if __name__ == '__main__':
    print('initial')
    train_SKG = load_static_graph('data/fr_en', 'att_triple_all', 0)
    dirEntity = "data/fr_en/ent_ids_all"
    entityIdNum, entityList = openDetailsAndId(dirEntity)
    dirAttr = 'data/fr_en/att2id_all'
    attrIdNum, attrList = openDetailsAndId(dirAttr)
    dirValue = "data/fr_en/att_value2id_all"
    valueIdNum, valueList = openDetailsAndId(dirValue)
    print("entity:", entityIdNum, "attr:", attrIdNum, "value:", valueIdNum)

    Run = run(1)
    Run.train(train_SKG, entityList, attrList, valueList)
