import argparse
import json
import logging
import os
import random
import sys
import time
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


class Progbar(object):
    """
    Progbar class inspired by keras

    进度条

    ```
    progbar = Progbar(max_step=100)
    for i in range(100):
        progbar.update(i, [("step", i), ("next", i+1)])
    ```
    """

    def __init__(self, max_step, width=30):
        self.max_step = max_step
        self.width = width
        self.last_width = 0

        self.sum_values = {}

        self.start = time.time()
        self.last_step = 0

        self.info = ""
        self.bar = ""

    def _update_values(self, curr_step, values):
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (curr_step - self.last_step), curr_step - self.last_step]
            else:
                self.sum_values[k][0] += v * (curr_step - self.last_step)
                self.sum_values[k][1] += (curr_step - self.last_step)

    def _write_bar(self, curr_step):
        last_width = self.last_width
        sys.stdout.write("\b" * last_width)
        sys.stdout.write("\r")

        numdigits = int(np.floor(np.log10(self.max_step))) + 1
        barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
        bar = barstr % (curr_step, self.max_step)
        prog = float(curr_step) / self.max_step
        prog_width = int(self.width * prog)
        if prog_width > 0:
            bar += ('=' * (prog_width - 1))
            if curr_step < self.max_step:
                bar += '>'
            else:
                bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
        sys.stdout.write(bar)

        return bar

    def _get_eta(self, curr_step):
        now = time.time()
        if curr_step:
            time_per_unit = (now - self.start) / curr_step
        else:
            time_per_unit = 0
        eta = time_per_unit * (self.max_step - curr_step)

        if curr_step < self.max_step:
            info = ' - ETA: %ds' % eta
        else:
            info = ' - %ds' % (now - self.start)

        return info

    def _get_values_sum(self):
        info = ""
        for name, value in self.sum_values.items():
            info += ' - %s: %.6f' % (name, value[0] / max(1, value[1]))
        return info

    def _write_info(self, curr_step):
        info = ""
        info += self._get_eta(curr_step)
        info += self._get_values_sum()

        sys.stdout.write(info)

        return info

    def _update_width(self, curr_step):
        curr_width = len(self.bar) + len(self.info)
        if curr_width < self.last_width:
            sys.stdout.write(" " * (self.last_width - curr_width))

        if curr_step >= self.max_step:
            sys.stdout.write("\n")

        sys.stdout.flush()

        self.last_width = curr_width

    def update(self, curr_step, values):
        """Updates the progress bar.

        Args:
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.

        """
        self._update_values(curr_step, values)
        self.bar = self._write_bar(curr_step)
        self.info = self._write_info(curr_step)
        self._update_width(curr_step)
        self.last_step = curr_step


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
                ret.append((int(th[0]), int(th[1])))
            self.seeds = ret
        # 80%训练集，20%测试集
        train_percent = 0.8
        train_max_idx = int(train_percent * len(self.seeds))
        self.train_seeds = self.seeds[:train_max_idx]
        self.test_seeds = self.seeds[train_max_idx + 1:]
        for i in self.test_seeds:
            self.left.append(i[0])  # 对齐的左边的实体
            self.right.append(i[1])  # 对齐的右边的实体

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
            hits_value = top_lr[i] / len(self.test_seeds) * 100
            left.append((hits, hits_value))
            print('Hits@%d: %.2f%%' % (hits, hits_value))
        print('For each right:')
        right = []
        for i in range(len(top_rl)):
            hits = top_k[i]
            hits_value = top_rl[i] / len(self.test_seeds) * 100
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
            t.train_seeds,
            nentity=self.nentity,
            nrelation=self.nattribute,
            nvalue=self.nvalue,
            hidden_dim=200,
            gamma=24.0,
        )
        # self.init_by_train_seeds(self.kge_model, t.train_seeds)
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
        init_step = 1
        steps = 50001
        printnum = 10000
        lastscore = 100
        progbar = Progbar(max_step=steps-init_step)
        # Training Loop
        starttime = Time.time()
        for step in range(init_step, steps):
            loss = self.kge_model.train_step(self.kge_model, self.optimizer, train_iterator, self.isCUDA)
            progbar.update(step-init_step, [
                ("step", step-init_step),
                ("loss", loss),
                ("cost", round((Time.time() - starttime)))
            ])
            if step > init_step and step % printnum == 0:
                print("\n属性消融实验")
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
