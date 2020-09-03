
import argparse
import json
import logging
import os
import random
import time as Time
import numpy as np
import torch
from utils import *
from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator


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

        #self.optimizer = torch.optim.SGD(self.kge_model.parameters(), lr=current_learning_rate)

        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, self.nentity, self.nattribute, self.nvalue, 256, 'head-batch'),
            batch_size=1024,
            shuffle=False,
            num_workers=0,
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, self.nentity, self.nattribute, self.nvalue, 256, 'tail-batch'),
            batch_size=1024,
            shuffle=False,
            num_workers=0,
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
