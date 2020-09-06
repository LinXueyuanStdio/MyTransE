import argparse
import json
import logging
import os
import random
import time as Time
from typing import List, Tuple

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
    seeds: List[Tuple[int, int]] = []
    linkEmbedding = []
    kg1E = []
    kg2E = []
    EA_results = {}
    train_seeds: List[Tuple[int, int]] = []
    test_seeds: List[Tuple[int, int]] = []

    def read_entity_align_list(self, entity_align_file_path):
        ret = []
        with open(entity_align_file_path, encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                self.left.append(int(th[0]))
                self.right.append(int(th[1]))
                ret.append((int(th[0]), int(th[1])))
            self.seeds = ret
        # 80%训练集，20%测试集
        train_percent = 0.8
        train_max_idx = int(train_percent * len(self.seeds))
        self.train_seeds = self.seeds[:train_max_idx]
        self.test_seeds = self.seeds[train_max_idx + 1:]

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
        all_entity_ids = torch.LongTensor(id_list).view(-1).to(device)
        all_entity_vec = torch.index_select(
            entities_embedding,
            dim=0,
            index=all_entity_ids
        ).view(-1, 200).cpu().detach().numpy()
        return all_entity_vec

    def calculate(self, top_k=(1, 10, 50, 100)):
        Lvec = np.array([self.linkEmbedding[e1] for e1, e2 in self.test_seeds])
        Rvec = np.array([self.linkEmbedding[e2] for e1, e2 in self.test_seeds])
        return self.get_hits(Lvec, Rvec, top_k)

    def get_hits(self, Lvec, Rvec, top_k=(1, 10, 50, 100)):
        sim = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
        top_lr = [0] * len(top_k)
        for i in range(Lvec.shape[0]):  # 对于每个KG1实体
            rank = sim[i, :].argsort()
            rank_index = np.where(rank == i)[0][0]
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_lr[j] += 1
        top_rl = [0] * len(top_k)
        for i in range(Rvec.shape[0]):
            rank = sim[:, i].argsort()
            rank_index = np.where(rank == i)[0][0]
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_rl[j] += 1
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

    def init_by_train_seeds(self, model: KGEModel, train_seeds: List[Tuple[int, int]], device="cuda"):
        for left_entity, right_entity in train_seeds:
            model.entity_embedding[left_entity] = model.entity_embedding[right_entity]

    def train(self, train_triples, entity2id, att2id, value2id):
        self.nentity = len(entity2id)
        self.nattribute = len(att2id)
        self.nvalue = len(value2id)

        self.kge_model = KGEModel(
            nentity=self.nentity,
            nrelation=self.nattribute,
            nvalue=self.nentity,
            hidden_dim=200,
            gamma=24.0,
        )
        self.init_by_train_seeds(self.kge_model, t.train_seeds)
        current_learning_rate = 0.001

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.kge_model.parameters()),
            lr=current_learning_rate
        )

        # self.optimizer = torch.optim.SGD(self.kge_model.parameters(), lr=current_learning_rate)

        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, self.nentity, self.nattribute, self.nentity, 256, 'head-batch'),
            batch_size=1024,
            shuffle=False,
            num_workers=4,
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, self.nentity, self.nattribute, self.nentity, 256, 'tail-batch'),
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
                left_vec = t.get_vec2(self.kge_model.entity_embedding, t.left)
                right_vec = t.get_vec2(self.kge_model.entity_embedding, t.right)
                hits = t.get_hits(left_vec, right_vec)
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
            list.append(int(DetailsAndId[0]))
            idNum += 1
    return idNum, list


if __name__ == '__main__':
    print('initial')
    train_SKG = load_static_graph('data/fr_en', 'triples_all', 0)
    dirEntity = "data/fr_en/ent_ids_all"
    entityIdNum, entityList = openDetailsAndId(dirEntity)
    dirAttr = 'data/fr_en/att2id_all'
    attrIdNum, attrList = openDetailsAndId(dirAttr)
    dirValue = "data/fr_en/att_value2id_all"
    valueIdNum, valueList = openDetailsAndId(dirValue)
    print("entity:", entityIdNum, "attr:", attrIdNum, "value:", valueIdNum)

    Run = run(1)
    Run.train(train_SKG, entityList, attrList, valueList)
