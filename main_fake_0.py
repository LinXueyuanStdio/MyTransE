from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _thread
import sys
import os
import time
import random
from math import exp
from typing import List, Tuple, Set, Dict

from scipy import spatial
import numpy as np
import torch
from torch import nn
from torch.optim import optimizer
from torch.utils import tensorboard
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from dataloader import BidirectionalOneShotIterator
import tensorflow as tf
import tensorboard as tb
import logging
import click

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)


# modified from main10.py

# region dataset
class TripleTrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, nvalue, negative_sample_size, mode, single_mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.nvalue = nvalue
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.single_mode = single_mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:

            if self.mode.endswith('head-batch'):
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode.endswith('tail-batch'):
                negative_sample = np.random.randint(self.nvalue, size=self.negative_sample_size * 2)
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode, self.single_mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        single_mode = data[0][4]
        return positive_sample, negative_sample, subsample_weight, mode, single_mode

    @staticmethod
    def count_frequency(triples, start=4):
        """
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        """
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        """
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        """

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class AVDistanceDataset(Dataset):
    def __init__(self,
                 seeds: List[Tuple[int, int]], triples: List[Tuple[int, int, int]],
                 kg1_entity_list: List[int], kg2_entity_list: List[int],
                 nentity, negative_sample_size, mode, single_mode):
        self.seeds: List[Tuple[int, int]] = seeds
        self.len: int = len(seeds)

        self.kg1_entity_list: List[int] = kg1_entity_list
        self.kg1_entity_size: int = len(kg1_entity_list)

        self.kg2_entity_list: List[int] = kg2_entity_list
        self.kg2_entity_size: int = len(kg2_entity_list)

        self.triple_mapper: Dict[int, List[Tuple[int, int]]] = self.build_triple_mapper(triples)

        self.nentity = nentity
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.single_mode = single_mode
        self.count = self.count_frequency(seeds)
        self.true_head, self.true_tail = self.get_true_head_and_tail(seeds)

    @staticmethod
    def build_triple_mapper(triples) -> Dict[int, List[Tuple[int, int]]]:
        triple_mapper: Dict[int, List[Tuple[int, int]]] = {}
        for e, a, v in triples:
            if e in triple_mapper:
                triple_mapper[e].append((a, v))
            else:
                triple_mapper[e] = [(a, v)]
        return triple_mapper

    def random_get_av(self, e) -> Tuple[int, int]:
        if e in self.triple_mapper:
            result = self.triple_mapper[e]
            return random.choice(result)
        else:
            print("error")
            return 1, 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.seeds[idx]

        head, tail = positive_sample

        subsampling_weight = self.count[head] + self.count[tail]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:

            if self.mode.endswith('head-batch'):
                # 1. 随机生成 index
                negative_sample_index = np.random.randint(self.kg1_entity_size, size=self.negative_sample_size * 2)
                # 2. 将 index 映射为 entity
                negative_sample = np.array(list(map(lambda x: self.kg1_entity_list[x], negative_sample_index)))
                mask = np.in1d(
                    negative_sample,
                    self.true_head[tail],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode.endswith('tail-batch'):
                negative_sample_index = np.random.randint(self.kg2_entity_size, size=self.negative_sample_size * 2)
                negative_sample = np.array(list(map(lambda x: self.kg2_entity_list[x], negative_sample_index)))
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[head],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            # 3. 根据 entity 随机搜索其 (attr, value)
            # sample_size x 2
            negative_sample = np.array(list(map(lambda x: self.random_get_av(x), negative_sample)))
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        # sample_size x 2
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        positive_sample = list(map(lambda x: self.random_get_av(x), positive_sample))

        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode, self.single_mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        single_mode = data[0][4]
        return positive_sample, negative_sample, subsample_weight, mode, single_mode

    @staticmethod
    def count_frequency(seeds, start=4):
        """
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        """
        count = {}
        for a, b in seeds:
            if a not in count:
                count[a] = start
            else:
                count[a] += 1

            if b not in count:
                count[b] = start
            else:
                count[b] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(seeds):
        """
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        """

        true_head = {}
        true_tail = {}

        for a, b in seeds:
            if a not in true_tail:
                true_tail[a] = []
            true_tail[a].append(b)
            if b not in true_head:
                true_head[b] = []
            true_head[b].append(a)

        for b in true_head:
            true_head[b] = np.array(list(set(true_head[b])))
        for a in true_tail:
            true_tail[a] = np.array(list(set(true_tail[a])))

        return true_head, true_tail


class AlignDataset(Dataset):
    def __init__(self,
                 seeds: List[Tuple[int, int]],
                 kg1_entity_list: List[int], kg2_entity_list: List[int],
                 nentity, negative_sample_size, mode, single_mode):
        self.seeds = seeds
        self.len = len(seeds)
        self.kg1_entity_list = kg1_entity_list
        self.kg2_entity_list = kg2_entity_list
        self.kg1_entity_size = len(kg1_entity_list)
        self.kg2_entity_size = len(kg2_entity_list)
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size
        self.mode: str = mode
        self.single_mode: str = single_mode
        self.count = self.count_frequency(seeds)
        self.true_head, self.true_tail = self.get_true_head_and_tail(seeds)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.seeds[idx]

        head, tail = positive_sample

        subsampling_weight = self.count[head] + self.count[tail]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:

            if self.mode.endswith('head-batch'):
                negative_sample = np.random.randint(self.kg1_entity_size, size=self.negative_sample_size * 2)
                negative_sample = np.array(list(map(lambda x: self.kg1_entity_list[x], negative_sample)))
                mask = np.in1d(
                    negative_sample,
                    self.true_head[tail],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode.endswith('tail-batch'):
                negative_sample = np.random.randint(self.kg2_entity_size, size=self.negative_sample_size * 2)
                negative_sample = np.array(list(map(lambda x: self.kg2_entity_list[x], negative_sample)))
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[head],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode, self.single_mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        single_mode = data[0][4]
        return positive_sample, negative_sample, subsample_weight, mode, single_mode

    def count_frequency(self, seeds: List[Tuple[int, int]], start=4):
        """
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        """
        count = {}
        for a, b in seeds:
            if a not in count:
                count[a] = start
            else:
                count[a] += 1

            if b not in count:
                count[b] = start
            else:
                count[b] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(seeds):
        """
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        """

        true_head = {}
        true_tail = {}

        for a, b in seeds:
            if a not in true_tail:
                true_tail[a] = []
            true_tail[a].append(b)
            if b not in true_head:
                true_head[b] = []
            true_head[b].append(a)

        for b in true_head:
            true_head[b] = np.array(list(set(true_head[b])))
        for a in true_tail:
            true_tail[a] = np.array(list(set(true_tail[a])))

        return true_head, true_tail


class DataManager(object):
    def __init__(self,
                 # input paths
                 entity_align_file="data/fr_en/ref_ent_ids",
                 all_entity_file="data/fr_en/ent_ids_all",
                 all_attr_file="data/fr_en/att2id_all",
                 all_relation_file="data/fr_en/att2id_all",
                 all_value_file="data/fr_en/att_value2id_all",
                 all_relation_triple_file="data/fr_en/att_triple_all",
                 all_attr_triple_file="data/fr_en/att_triple_all",
                 kg1_entity_file="data/fr_en/ent_ids_1",
                 kg2_entity_file="data/fr_en/ent_ids_2",
                 ):
        self.entity_align_file = entity_align_file
        self.all_entity_file = all_entity_file
        self.all_attr_file = all_attr_file
        self.all_relation_file = all_relation_file
        self.all_value_file = all_value_file
        self.all_relation_triple_file = all_relation_triple_file
        self.all_attr_triple_file = all_attr_triple_file
        self.kg1_entity_file = kg1_entity_file
        self.kg2_entity_file = kg2_entity_file

        self.relation_count = 0
        self.attr_count = 0
        self.value_count = 0
        self.entity_count = 0
        self.kg1_entity_count = 0
        self.kg2_entity_count = 0
        self.seeds_count = 0
        self.train_seeds_count = 0
        self.test_seeds_count = 0
        self.relation_triples_count = 0
        self.attr_triples_count = 0

    def load_relation(self):
        self.relation_list, _ = read_ids_and_names(self.all_relation_file)
        self.relation_count = len(self.relation_list)

    def load_attr(self):
        self.attr_list, _ = read_ids_and_names(self.all_attr_file)
        self.attr_count = len(self.attr_list)

    def load_value(self):
        self.value_list, _ = read_ids_and_names(self.all_value_file)
        self.value_count = len(self.value_list)

    def load_entity(self):
        self.entity_list, self.entity_name_list = read_ids_and_names(self.all_entity_file)
        self.kg1_entity_list, _ = read_ids_and_names(self.kg1_entity_file)
        self.kg2_entity_list, _ = read_ids_and_names(self.kg2_entity_file)

        self.entity_count = len(self.entity_list)
        self.kg1_entity_count = len(self.kg1_entity_list)
        self.kg2_entity_count = len(self.kg2_entity_list)

    def summary(self):
        logger.info("entity: %s attr: %s relation: %s value: %s" %
                    (self.entity_count, self.attr_count, self.relation_count, self.value_count))
        logger.info("kg1_entity: %s kg2_entity: %s" %
                    (self.kg1_entity_count, self.kg2_entity_count))
        logger.info("seeds: %s train-seeds: %s test-seeds: %s" %
                    (self.seeds_count, self.train_seeds_count, self.test_seeds_count))
        logger.info("relation triples: %s attr triples: %s" %
                    (self.relation_triples_count, self.attr_triples_count))

    def load_seeds(self):
        self.seeds: List[Tuple[int, int]] = read_seeds(self.entity_align_file)
        # 80%训练集，20%测试集
        train_percent = 0.3
        train_max_idx = int(train_percent * len(self.seeds))
        self.train_seeds: List[Tuple[int, int]] = self.seeds[:train_max_idx]
        self.test_seeds: List[Tuple[int, int]] = self.seeds[train_max_idx:]

        self.seeds_count = len(self.seeds)
        self.train_seeds_count = len(self.train_seeds)
        self.test_seeds_count = len(self.test_seeds)

    def load_relation_triple(self, enhance=True):
        self.relation_triples = read_relation_triple(self.all_relation_triple_file)
        if enhance:
            self._enhance_relation_triple()
        self.relation_triples_count = len(self.relation_triples)

    def load_attr_triple(self, enhance=True):
        self.attr_triples = read_attr_triple(self.all_attr_triple_file)
        if enhance:
            self._enhance_attr_triple()
        self.attr_triples_count = len(self.attr_triples)

    def _enhance_relation_triple(self):
        logger.info("关系三元组 数据增强")
        all_triple_file_ext = self.all_relation_triple_file + "_enhance"
        if os.path.exists(all_triple_file_ext):
            self.relation_triples = read_relation_triple(all_triple_file_ext)
            self.relation_count = read_relation_count(all_triple_file_ext)
        else:
            self.relation_triples = append_align_triple(self.relation_triples, self.train_seeds)
            save_relation_triple(self.relation_triples, all_triple_file_ext)
            self.relation_count = read_relation_count(all_triple_file_ext)

    def _enhance_attr_triple(self):
        logger.info("属性三元组 数据增强")
        all_triple_file_ext = self.all_attr_triple_file + "_enhance"
        if os.path.exists(all_triple_file_ext):
            self.attr_triples = read_relation_triple(all_triple_file_ext)
        else:
            self.attr_triples = append_align_triple(self.attr_triples, self.train_seeds)
            save_relation_triple(self.attr_triples, all_triple_file_ext)


class ModelManager(object):
    def __init__(self):
        pass


"""
可配置项分类
1. 数据集相关，如文件路径、数据集id等
2. 模型相关，如隐藏层大小、嵌入维数等
3. 训练测试相关，如lr、训练批次、保存加载模型等
"""


# endregion

# region model
# region sub model
class SubModel(object):
    def __init__(self):
        pass

    def score(self, sample):
        raise NotImplementedError


class NegSampleModel(SubModel):
    def __init__(self):
        super(NegSampleModel, self).__init__()

    def score(self, sample):
        subsample, mode = sample
        return self.forward(subsample, mode)

    def forward(self, sample, mode:str):
        raise NotImplementedError

    @staticmethod
    def get_loss(model, model_key, positive_sample, negative_sample, subsampling_weight, mode, single_mode="xxx-single"):
        negative_score = model(((positive_sample, negative_sample), mode), model_key=model_key)
        negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model((positive_sample, single_mode), model_key=model_key)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        loss = (positive_sample_loss + negative_sample_loss) / 2
        return loss


class RelationTripleModel(NegSampleModel):
    def __init__(self, entity_embedding, relation_embedding):
        super(RelationTripleModel, self).__init__()
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

    def forward(self, sample, mode:str):
        if mode.endswith("single"):
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
        elif mode.endswith("head-batch"):
            tail_part, head_part = sample
            # head_part : batch_size x sample_size
            # tail_part : batch_size x 3
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
        elif mode.endswith("tail-batch"):

            head_part, tail_part = sample
            # head_part : batch_size x 3
            # tail_part : batch_size x sample_size
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        else:
            raise ValueError('mode %s not supported' % mode)
        score = self.loss(head, relation, tail, mode)
        return score

    def loss(self, head, relation, tail, mode: str):
        raise NotImplementedError


class TransERelationModel(RelationTripleModel):
    def __init__(self, entity_embedding, relation_embedding, gamma):
        super(TransERelationModel, self).__init__(entity_embedding, relation_embedding)
        self.gamma = gamma

    def loss(self, head, relation, tail, mode: str):
        if mode.endswith('head-batch'):
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score


class RotatERelationModel(RelationTripleModel):
    def __init__(self, entity_embedding, relation_embedding, embedding_range, gamma):
        super(RotatERelationModel, self).__init__(entity_embedding, relation_embedding)
        self.embedding_range = embedding_range
        self.gamma = gamma

    def loss(self, head, relation, tail, mode: str):

        pi = 3.14159265358979323846

        re_head, im_head = head, head  # torch.chunk(head, 2, dim=2)
        re_tail, im_tail = tail, tail  # torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode.endswith('head-batch'):
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score


class AttrTripleModel(NegSampleModel):
    def __init__(self, entity_embedding, attr_embedding, value_embedding):
        super(AttrTripleModel, self).__init__()
        self.entity_embedding = entity_embedding
        self.attr_embedding = attr_embedding
        self.value_embedding = value_embedding

    def forward(self, sample, mode):
        if mode.endswith("single"):
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.attr_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.value_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode.endswith("head-batch"):
            tail_part, head_part = sample
            # head_part : batch_size x sample_size
            # tail_part : batch_size x 3
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.attr_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.value_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode.endswith("tail-batch"):

            head_part, tail_part = sample
            # head_part : batch_size x 3
            # tail_part : batch_size x sample_size
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.attr_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.value_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)
        score = self.loss(head, relation, tail, mode)
        return score

    def loss(self, head, relation, tail, mode: str):
        raise NotImplementedError


class TransEAttrModel(AttrTripleModel):
    def __init__(self, entity_embedding, attr_embedding, value_embedding, gamma):
        super(TransEAttrModel, self).__init__(entity_embedding, attr_embedding, value_embedding)
        self.gamma = gamma

    def loss(self, head, relation, tail, mode: str):
        if mode.endswith('head-batch'):
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score


class RotatEAttrModel(AttrTripleModel):
    def __init__(self, entity_embedding, attr_embedding, value_embedding, embedding_range, gamma):
        super(RotatEAttrModel, self).__init__(entity_embedding, attr_embedding, value_embedding)
        self.embedding_range = embedding_range
        self.gamma = gamma

    def loss(self, head, relation, tail, mode: str):

        pi = 3.14159265358979323846

        re_head, im_head = head, head  # torch.chunk(head, 2, dim=2)
        re_tail, im_tail = tail, tail  # torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode.endswith('head-batch'):
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score


class AlignModel(NegSampleModel):
    def __init__(self, entity_embedding):
        super(AlignModel, self).__init__()
        self.entity_embedding = entity_embedding

    def forward(self, sample, mode):
        if mode.endswith("single"):
            batch_size, negative_sample_size = sample.size(0), 1
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
        elif mode.endswith("head-batch"):
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
        elif mode.endswith("tail-batch"):
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        else:
            raise ValueError('mode %s not supported' % mode)
        score = self.loss(head, tail, mode)
        return score

    def loss(self, head, tail, mode: str):
        raise NotImplementedError


class GcnAlignModel(AlignModel):
    def __init__(self, entity_embedding, gamma):
        super(GcnAlignModel, self).__init__(entity_embedding)
        self.gamma = gamma

    def loss(self, head, tail, mode: str):
        # print(mode, head.size(), tail.size())
        score = head - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        # score = torch.norm(score, p=1, dim=2)
        return score


class AvModel(NegSampleModel):
    def __init__(self, entity_embedding, relation_embedding):
        super(AvModel, self).__init__()
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

    def forward(self, sample, mode):
        if mode == "av-single":
            batch_size, negative_sample_size = sample.size(0), 1
            # print(mode, sample[0].size(), sample[1].size())
            a = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 0, 0].view(-1)
            ).unsqueeze(1)

            v = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0, 1].view(-1)
            ).unsqueeze(1)

            a_ = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1, 0].view(-1)
            ).unsqueeze(1)

            v_ = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 1, 1].view(-1)
            ).unsqueeze(1)
        elif mode == 'av-head-batch':  # 负例是头
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            # tail_part : batch_size x 2 x 2 第一个2是实体对的2，第二个2是实体对应的(a,v)的2
            # head_part : batch_size x sample_size x 2
            # print(mode, tail_part.size(), head_part.size())

            a = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, :, 0].view(-1)
            ).view(batch_size, negative_sample_size, -1)

            v = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, :, 1].view(-1)
            ).view(batch_size, negative_sample_size, -1)

            a_ = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1, 0].view(-1)
            ).unsqueeze(1)

            v_ = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 1, 1].view(-1)
            ).unsqueeze(1)
        elif mode == 'av-tail-batch':  # 负例是尾
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            # print(mode, tail_part.size(), head_part.size(), head_part[:, 1, 0].view(batch_size, -1))
            # head_part : batch_size x 2 x 2
            # tail_part : batch_size x sample_size x 2
            a = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, :, 0].view(-1)
            ).view(batch_size, negative_sample_size, -1)

            v = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, :, 1].view(-1)
            ).view(batch_size, negative_sample_size, -1)

            a_ = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1, 0].view(-1)
            ).unsqueeze(1)

            v_ = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 1, 1].view(-1)
            ).unsqueeze(1)
        else:
            raise ValueError('mode %s not supported' % mode)
        score = self.loss(a, v, a_, v_, mode)
        return score

    def loss(self, a, v, a_, v_, mode: str):
        raise NotImplementedError


class MyAvModel(AvModel):
    def __init__(self, entity_embedding, relation_embedding, gamma):
        super(MyAvModel, self).__init__(entity_embedding, relation_embedding)
        self.gamma = gamma

    def loss(self, a, v, a_, v_, mode: str):
        score = (a - v) - (a_ - v_)
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        # score = torch.norm(score, p=1, dim=2)
        return score


# endregion

# region context
class Context(object):
    def __init__(self):
        pass

    def inject(self, model):
        raise NotImplementedError


class NegSampleContext(Context):
    def __init__(self,
                 dataset, model: NegSampleModel,
                 dataset_key: str, model_key: str,
                 device="cuda"):
        super(NegSampleContext, self).__init__()
        self.dataset = dataset
        self.model = model
        self.dataset_key = dataset_key
        self.model_key = model_key
        self.device = device

    def inject(self, model):
        positive_sample, negative_sample, subsampling_weight, mode, single_mode = next(self.dataset)
        positive_sample = positive_sample.to(self.device)
        negative_sample = negative_sample.to(self.device)
        subsampling_weight = subsampling_weight.to(self.device)
        return NegSampleModel.get_loss(model, self.model_key,
                                   positive_sample, negative_sample, subsampling_weight, mode, single_mode)


# endregion
class KGEModel(nn.Module):
    def __init__(self,
                 train_seeds,
                 entity_count, relation_count, attr_count, value_count,
                 hidden_dim, gamma):
        super(KGEModel, self).__init__()
        # self.model_name = model_name
        self.sub_model_dict = None
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        self.entity_dim = hidden_dim
        self.relation_dim = hidden_dim
        self.attr_dim = hidden_dim
        self.value_dim = hidden_dim

        # region 知识图谱的嵌入：实体、属性、属性值
        entity_weight = torch.zeros(entity_count, self.entity_dim)
        nn.init.uniform_(
            tensor=entity_weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        for left_entity, right_entity in train_seeds:
            entity_weight[left_entity] = entity_weight[right_entity]
        self.entity_embedding = nn.Parameter(entity_weight)
        # nn.init.normal_(self.entity_embedding)

        self.relation_embedding = nn.Parameter(torch.zeros(relation_count, self.relation_dim))
        # nn.init.normal_(self.relation_embedding)
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.attr_embedding = nn.Parameter(torch.zeros(attr_count, self.attr_dim))
        # nn.init.normal_(self.attr_embedding)
        nn.init.uniform_(
            tensor=self.attr_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.value_embedding = nn.Parameter(torch.zeros(value_count, self.value_dim))
        # nn.init.normal_(self.value_embedding)
        nn.init.uniform_(
            tensor=self.value_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        # endregion

        self.sub_model_dict = dict()

    def forward(self, sample, model_key):
        if self.sub_model_dict is None:
            raise ValueError('sub_model_dict is None')
        elif model_key in self.sub_model_dict:
            model = self.sub_model_dict[model_key]
            if isinstance(model, NegSampleModel):
                score = model.score(sample)
                return score
            else:
                raise ValueError('model %s not supported' % model_key)
        else:
            raise ValueError('model %s not installed' % model_key)

    def use(self, sub_model_dict):
        self.sub_model_dict = sub_model_dict


# endregion

# region 日志
def get_logger(filename):
    """
    Return instance of logger
    统一的日志样式
    """
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.INFO)

    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))

    logging.getLogger().addHandler(handler)

    return logger


logger = get_logger("./train.log")


# endregion

# region 进度条
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


# endregion

# region 测试对齐实体
class Tester:
    left_ids: List[int] = []  # test_seeds 中对齐实体的左实体id
    right_ids: List[int] = []  # test_seeds 中对齐实体的右实体id
    test_seeds: List[Tuple[int, int]] = []  # (0.2m, 2)
    linkEmbedding = []
    kg1E = []
    kg2E = []
    EA_results = {}

    def __init__(self, test_seeds):
        self.test_seeds = test_seeds
        self.left_ids = []
        self.right_ids = []
        for left_entity, right_entity in self.test_seeds:
            self.left_ids.append(left_entity)  # 对齐的左边的实体
            self.right_ids.append(right_entity)  # 对齐的右边的实体

    def XRA(self, entity_embedding_file_path):
        self.linkEmbedding = []
        with open(entity_embedding_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            aline = lines[i].strip()
            aline_list = aline.split()
            self.linkEmbedding.append(aline_list)

    @staticmethod
    def get_vec(entities_embedding, id_list: List[int], device="cuda"):
        tensor = torch.LongTensor(id_list).view(-1, 1).to(device)
        return entities_embedding(tensor).view(-1, 200).cpu().detach().numpy()

    @staticmethod
    def get_vec2(entities_embedding, id_list: List[int], device="cuda"):
        all_entity_ids = torch.LongTensor(id_list).view(-1).to(device)
        all_entity_vec = torch.index_select(
            entities_embedding,
            dim=0,
            index=all_entity_ids
        ).view(-1, 200).cpu().detach().numpy()
        return all_entity_vec

    @staticmethod
    def get_vec3(entities_embedding, orth: torch.Tensor, id_list: List[int], device="cuda"):
        all_entity_ids = torch.LongTensor(id_list).view(-1).to(device)
        all_entity_vec = torch.index_select(
            entities_embedding,
            dim=0,
            index=all_entity_ids
        ).view(-1, 200)
        all_entity_vec = all_entity_vec.matmul(orth.transpose(0, 1))
        return all_entity_vec.cpu().detach().numpy()

    def calculate(self, top_k=(1, 10, 50, 100)):
        Lvec = np.array([self.linkEmbedding[e1] for e1, e2 in self.test_seeds])
        Rvec = np.array([self.linkEmbedding[e2] for e1, e2 in self.test_seeds])
        return self.get_hits(Lvec, Rvec, top_k)

    def get_hits2(self, Lvec, Rvec, top_k=(1, 10, 50, 100)):
        sim = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
        return self.get_hits(Lvec, Rvec, sim, top_k)

    def get_hits(self, Lvec, Rvec, sim, top_k=(1, 10, 50, 100)):
        # Lvec (m, d), Rvec (m, d)
        # Lvec和Rvec分别是对齐的左右实体的嵌入组成的列表，d是嵌入维度，m是实体个数
        # sim=distance(Lvec, Rvec) (m, m)
        # sim[i, j] 表示在 Lvec 的实体 i 到 Rvec 的实体 j 的距离
        top_lr = [0] * len(top_k)
        for i in range(Lvec.shape[0]):  # 对于每个KG1实体
            rank = sim[i, :].argsort()
            # sim[i, :] 是一个行向量，表示将 Lvec 中的实体 i 到 Rvec 的所有实体的距离
            # argsort 表示将距离按大小排序，返回排序后的下标。比如[6,3,5]下标[0,1,2]，排序后[3,5,6]，则返回[1,2,0]
            rank_index = np.where(rank == i)[0][0]
            # 对于一维向量，np.where(rank == i) 等价于 list(rank).index(i)，即查找元素 i 在 rank 中的下标
            # 这里的 i 不是代表 Lvec 中的实体 i 的下标，而是代表 Rvec 中和 i 对齐的实体的下标。
            for j in range(len(top_k)):
                if rank_index < top_k[j]:  # index 从 0 开始，因此用 '<' 号
                    top_lr[j] += 1
        top_rl = [0] * len(top_k)
        for i in range(Rvec.shape[0]):
            rank = sim[:, i].argsort()
            rank_index = np.where(rank == i)[0][0]
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_rl[j] += 1
        logger.info('For each left:')
        left = []
        for i in range(len(top_lr)):
            hits = top_k[i]
            hits_value = top_lr[i] / len(self.test_seeds) * 100
            left.append((hits, hits_value))
            logger.info('Hits@%d: %.2f%%' % (hits, hits_value))
        logger.info('For each right:')
        right = []
        for i in range(len(top_rl)):
            hits = top_k[i]
            hits_value = top_rl[i] / len(self.test_seeds) * 100
            right.append((hits, hits_value))
            logger.info('Hits@%d: %.2f%%' % (hits, hits_value))

        return {
            "left": left,
            "right": right,
        }

    @staticmethod
    def get_score(hits):
        hits_left = hits["left"]
        hits_right = hits["right"]
        left_hits_10 = hits_left[2][1]
        right_hits_10 = hits_right[2][1]
        score = (left_hits_10 + right_hits_10) / 2
        return score


# endregion

# region 保存与加载模型，恢复训练状态
_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_MODEL_STATE_DICT2 = "model_state_dict2"
_OPTIMIZER_STATE_DICT2 = "optimizer_state_dict2"
_EPOCH = "epoch"
_STEP = "step"
_BEST_SCORE = "best_score"
_LOSS = "loss"


def load_checkpoint(model: nn.Module, optim: optimizer.Optimizer,
                    checkpoint_path="./result/fr_en/checkpoint.tar") -> Tuple[int, int, float]:
    """Loads training checkpoint.

    :param checkpoint_path: path to checkpoint
    :param model: model to update state
    :param optim: optimizer to  update state
    :return tuple of starting epoch id, starting step id, best checkpoint score
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint[_MODEL_STATE_DICT])
    optim.load_state_dict(checkpoint[_OPTIMIZER_STATE_DICT])
    start_epoch_id = checkpoint[_EPOCH] + 1
    step = checkpoint[_STEP] + 1
    best_score = checkpoint[_BEST_SCORE]
    return start_epoch_id, step, best_score


def save_checkpoint(model: nn.Module, optim: optimizer.Optimizer,
                    epoch_id: int, step: int, best_score: float,
                    save_path="./result/fr_en/checkpoint.tar"):
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optim.state_dict(),
        _EPOCH: epoch_id,
        _STEP: step,
        _BEST_SCORE: best_score,
    }, save_path)


def save_entity_embedding_list(entity_embedding, embedding_path="./result/fr_en/ATentsembed.txt"):
    with open(embedding_path, 'w') as f:
        d = entity_embedding.data.detach().cpu().numpy()
        for i in range(len(d)):
            f.write(" ".join([str(j) for j in d[i].tolist()]))
            f.write("\n")


# endregion

# region 数据集
def read_ids_and_names(dir_path, sp="\t"):
    ids = []
    names = []
    with open(dir_path, encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            id_to_name = line.strip().split(sp)
            ids.append(int(id_to_name[0]))
            names.append(id_to_name[1])
    return ids, names


def read_relation_count(triple_path):
    with open(triple_path, 'r') as fr:
        r = []
        for line in fr:
            line_split = line.split()
            rel = int(line_split[1])
            r.append(rel)
        r = list(set(r))
    return len(r)


def read_relation_triple(triple_path):
    with open(triple_path, 'r') as fr:
        triple = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            rel = int(line_split[1])
            tail = int(line_split[2])
            triple.add((head, rel, tail))
    return list(triple)


def read_attr_triple(triple_path):
    with open(triple_path, 'r') as fr:
        triple = set()
        for line in fr:
            line_split = line.split()
            entity = int(line_split[0])
            value = int(line_split[1])
            attr = int(line_split[2])
            triple.add((entity, attr, value))
    return list(triple)


def read_seeds(seed_path):
    data = []
    with open(seed_path, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            data.append((int(th[0]), int(th[1])))
    return data


def save_relation_triple(triples, triple_path):
    with open(triple_path, 'w') as fr:
        for triple in triples:
            fr.write("%d\t%d\t%d\n" % (triple[0], triple[1], triple[2]))


def save_attr_triple(triples, triple_path):
    with open(triple_path, 'w') as fr:
        for triple in triples:
            fr.write("%d\t%d\t%d\n" % (triple[0], triple[2], triple[1]))


def append_align_triple(triple: List[Tuple[int, int, int]], entity_align_list: List[Tuple[int, int]]):
    # 使用对齐实体替换头节点，构造属性三元组数据，从而达到利用对齐实体数据的目的
    align_set = {}
    for i in entity_align_list:
        align_set[i[0]] = i[1]
        align_set[i[1]] = i[0]
    triple_replace_with_align = []
    bar = Progbar(max_step=len(triple))
    count = 0
    for entity, attr, value in triple:
        if entity in align_set:
            triple_replace_with_align.append((align_set[entity], attr, value))
        if value in align_set:
            triple_replace_with_align.append((entity, attr, align_set[value]))
        if (entity in align_set) and (value in align_set):
            triple_replace_with_align.append((align_set[entity], attr, align_set[value]))
        count += 1
        bar.update(count, [("step", count)])
    return triple + triple_replace_with_align


# endregion

class MTransE:
    def __init__(self,
                 data_manager: DataManager,
                 # output paths
                 checkpoint_path="./result/TransE/fr_en/checkpoint.tar",
                 embedding_path="./result/TransE/fr_en/ATentsembed.txt",
                 tensorboard_log_dir="./result/TransE/fr_en/log/",
                 # model config
                 device="cuda",
                 learning_rate=0.001,
                 visualize=False,
                 ):
        self.data_manager = data_manager

        self.tensorboard_log_dir = tensorboard_log_dir
        self.checkpoint_path = checkpoint_path
        self.embedding_path = embedding_path

        self.learning_rate = learning_rate

        self.device = device
        self.visualize = visualize

        self.t = Tester(self.data_manager.test_seeds)

        self.dataset_dict = dict()
        self.sub_model_dict = dict()
        self.context_list = list()

    def init_relation_triple_dataset(self,
                                     name,
                                     negative_sample_size=1024,
                                     batch_size=512,
                                     shuffle=False,
                                     num_workers=4):
        train_dataloader_head = DataLoader(
            TripleTrainDataset(self.data_manager.relation_triples,
                               self.data_manager.entity_count,
                               self.data_manager.relation_count,
                               self.data_manager.entity_count,
                               negative_sample_size,
                               '%s-head-batch' % name, "%s-single" % name),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=TripleTrainDataset.collate_fn
        )
        train_dataloader_tail = DataLoader(
            TripleTrainDataset(self.data_manager.relation_triples,
                               self.data_manager.entity_count,
                               self.data_manager.relation_count,
                               self.data_manager.entity_count,
                               negative_sample_size,
                               '%s-tail-batch' % name, "%s-single" % name),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=TripleTrainDataset.collate_fn
        )
        relation_triples_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        self.install_dataset(name, relation_triples_iterator)

    def init_attr_triple_dataset(self,
                                 name,
                                 negative_sample_size=1024,
                                 batch_size=512,
                                 shuffle=False,
                                 num_workers=4):
        train_dataloader_head = DataLoader(
            TripleTrainDataset(self.data_manager.attr_triples,
                               self.data_manager.entity_count,
                               self.data_manager.attr_count,
                               self.data_manager.value_count,
                               negative_sample_size,
                               '%s-head-batch' % name, "%s-single" % name),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=TripleTrainDataset.collate_fn
        )
        train_dataloader_tail = DataLoader(
            TripleTrainDataset(self.data_manager.attr_triples,
                               self.data_manager.entity_count,
                               self.data_manager.attr_count,
                               self.data_manager.value_count,
                               negative_sample_size,
                               '%s-tail-batch' % name, "%s-single" % name),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=TripleTrainDataset.collate_fn
        )
        attr_triples_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        self.install_dataset(name, attr_triples_iterator)

    def init_align_dataset(self,
                           name,
                           negative_sample_size=512,
                           batch_size=512,
                           shuffle=True,
                           num_workers=4):
        align_dataloader_head = DataLoader(
            AlignDataset(self.data_manager.train_seeds,
                         self.data_manager.kg1_entity_list,
                         self.data_manager.kg2_entity_list,
                         self.data_manager.entity_count,
                         negative_sample_size,
                         '%s-head-batch' % name, "%s-single" % name),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=AlignDataset.collate_fn
        )
        align_dataloader_tail = DataLoader(
            AlignDataset(self.data_manager.train_seeds,
                         self.data_manager.kg1_entity_list,
                         self.data_manager.kg2_entity_list,
                         self.data_manager.entity_count,
                         negative_sample_size,
                         '%s-tail-batch' % name, "%s-single" % name),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=AlignDataset.collate_fn
        )
        align_iterator = BidirectionalOneShotIterator(align_dataloader_head, align_dataloader_tail)
        self.install_dataset(name, align_iterator)

    def init_av_dataset(self,
                        name,
                        negative_sample_size=512,
                        batch_size=512,
                        shuffle=True,
                        num_workers=4):
        av_dataloader_head = DataLoader(
            AVDistanceDataset(self.data_manager.train_seeds,
                              self.data_manager.attr_triples,
                              self.t.left_ids,
                              self.t.right_ids,
                              self.data_manager.entity_count,
                              negative_sample_size,
                              '%s-head-batch' % name, "%s-single" % name),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=AVDistanceDataset.collate_fn
        )
        av_dataloader_tail = DataLoader(
            AVDistanceDataset(self.data_manager.train_seeds,
                              self.data_manager.attr_triples,
                              self.t.left_ids,
                              self.t.right_ids,
                              self.data_manager.entity_count,
                              negative_sample_size,
                              '%s-tail-batch' % name, "%s-single" % name),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=AVDistanceDataset.collate_fn
        )
        av_iterator = BidirectionalOneShotIterator(av_dataloader_head, av_dataloader_tail)
        self.install_dataset(name, av_iterator)

    def init_dataset(self, gcn=False, my=False):
        self.init_relation_triple_dataset("relation-triple")
        self.init_attr_triple_dataset("attr-triple")
        if gcn:
            self.init_align_dataset("align")
        if my:
            self.init_av_dataset("av")

    def install_model(self, key, model: SubModel):
        self.sub_model_dict[key] = model

    def install_dataset(self, key, dataset):
        self.dataset_dict[key] = dataset

    def bind_neg_sample_context(self, dataset_key: str, model_key: str):
        if dataset_key not in self.dataset_dict:
            raise ValueError("dataset %s not installed" % dataset_key)
        dataset = self.dataset_dict[dataset_key]

        if model_key not in self.sub_model_dict:
            raise ValueError("model %s not installed" % model_key)
        model = self.sub_model_dict[model_key]

        self.context_list.append(NegSampleContext(dataset, model, dataset_key, model_key, self.device))

    def bind_neg_sample_soft_align_context(self, dataset_key: str, model_key: str):
        if dataset_key not in self.dataset_dict:
            raise ValueError("dataset %s not installed" % dataset_key)
        dataset = self.dataset_dict[dataset_key]

        if model_key not in self.sub_model_dict:
            raise ValueError("model %s not installed" % model_key)
        model = self.sub_model_dict[model_key]

        self.context_list.append(NegSampleContext(dataset, model, dataset_key, model_key, self.device))

    def init_model(self, hidden_dim=200):
        self.model = KGEModel(
            self.data_manager.train_seeds,
            entity_count=self.data_manager.entity_count,
            relation_count=self.data_manager.relation_count,
            attr_count=self.data_manager.attr_count,
            value_count=self.data_manager.value_count,
            hidden_dim=hidden_dim,
            gamma=24.0,
        ).to(self.device)
        self.install_model("GCN-align", GcnAlignModel(self.model.entity_embedding, self.model.gamma))
        self.install_model("TransE-relation", TransERelationModel(self.model.entity_embedding,
                                                                  self.model.relation_embedding,
                                                                  self.model.gamma))
        self.install_model("TransE-attr", TransEAttrModel(self.model.entity_embedding,
                                                          self.model.attr_embedding,
                                                          self.model.value_embedding,
                                                          self.model.gamma))
        self.install_model("RotatE-relation", RotatERelationModel(self.model.entity_embedding,
                                                                  self.model.relation_embedding,
                                                                  self.model.embedding_range,
                                                                  self.model.gamma))
        self.install_model("RotatE-attr", RotatEAttrModel(self.model.entity_embedding,
                                                          self.model.attr_embedding,
                                                          self.model.value_embedding,
                                                          self.model.embedding_range,
                                                          self.model.gamma))
        self.install_model("av", MyAvModel(self.model.entity_embedding,
                                           self.model.relation_embedding,
                                           self.model.gamma))
        self.model.use(self.sub_model_dict)
        self.bind_neg_sample_context("align", "GCN-align")
        self.bind_neg_sample_context("relation-triple", "TransE-relation")
        self.bind_neg_sample_context("attr-triple", "TransE-attr")
        self.bind_neg_sample_context("relation-triple", "RotatE-relation")
        self.bind_neg_sample_context("attr-triple", "RotatE-attr")

    def init_optimizer(self):
        self.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )

    def init_visualize(self):
        self.summary_writer = tensorboard.SummaryWriter(log_dir=self.tensorboard_log_dir)

    def train_step(self):
        self.model.train()
        self.optim.zero_grad()
        loss = None
        for context in self.context_list:
            score = context.inject(self.model)
            if loss is None:
                loss = score
            else:
                loss += score
        loss /= len(self.context_list)
        loss.backward()
        self.optim.step()
        return loss

    def run_train(self, need_to_load_checkpoint=True):
        logger.info("start training")
        init_step = 1
        total_steps = 500001
        test_steps = 5000
        score = 0
        last_score = score

        if need_to_load_checkpoint:
            _, init_step, score = load_checkpoint(self.model, self.optim, self.checkpoint_path)
            last_score = score
            logger.info("恢复模型后，查看一下模型状态")
            self.run_test()

        progbar = Progbar(max_step=total_steps)
        start_time = time.time()
        for step in range(init_step, total_steps):
            loss = self.train_step()
            progbar.update(step + 1, [
                ("loss", loss),
                ("cost", round((time.time() - start_time)))
            ])
            if self.visualize:
                self.summary_writer.add_scalar(tag='Loss/loss', scalar_value=loss, global_step=step)

            if step > init_step and step % test_steps == 0:
                logger.info("")
                hits, score = self.run_test()

                if self.visualize:
                    hits_left = hits["left"]
                    hits_right = hits["right"]
                    self.summary_writer.add_embedding(tag='Embedding',
                                                      mat=self.model.entity_embedding,
                                                      metadata=self.data_manager.entity_name_list,
                                                      global_step=step)
                    self.summary_writer.add_scalar(tag='Hits@1/left', scalar_value=hits_left[0][1], global_step=step)
                    self.summary_writer.add_scalar(tag='Hits@10/left', scalar_value=hits_left[1][1], global_step=step)
                    self.summary_writer.add_scalar(tag='Hits@50/left', scalar_value=hits_left[2][1], global_step=step)
                    self.summary_writer.add_scalar(tag='Hits@100/left', scalar_value=hits_left[3][1], global_step=step)

                    self.summary_writer.add_scalar(tag='Hits@1/right', scalar_value=hits_right[0][1], global_step=step)
                    self.summary_writer.add_scalar(tag='Hits@10/right', scalar_value=hits_right[1][1], global_step=step)
                    self.summary_writer.add_scalar(tag='Hits@50/right', scalar_value=hits_right[2][1], global_step=step)
                    self.summary_writer.add_scalar(tag='Hits@100/right', scalar_value=hits_right[3][1],
                                                   global_step=step)
                if score > last_score:
                    logger.info("保存 (" + str(score) + ">" + str(last_score) + ")")
                    last_score = score
                    save_checkpoint(self.model, self.optim,
                                    1, step, score,
                                    self.checkpoint_path)
                    save_entity_embedding_list(self.model.entity_embedding, self.embedding_path)
                save_entity_embedding_list(self.model.entity_embedding,
                                           self.embedding_path + "_score_" + str(int(score)))

    def run_test(self):
        computing_time = time.time()
        left_vec = self.t.get_vec2(self.model.entity_embedding, self.t.left_ids)
        # left_vec2 = self.t.get_vec3(self.model.entity_embedding, self.model.M, self.t.left_ids)
        right_vec = self.t.get_vec2(self.model.entity_embedding, self.t.right_ids)
        sim = spatial.distance.cdist(left_vec, right_vec, metric='euclidean')
        # sim2 = spatial.distance.cdist(left_vec2, right_vec, metric='euclidean')
        logger.info("计算距离完成，用时 " + str(int(time.time() - computing_time)) + " 秒")
        logger.info("属性消融实验")
        hits = self.t.get_hits(left_vec, right_vec, sim)
        score = self.t.get_score(hits)
        # hits2 = self.t.get_hits(left_vec2, right_vec, sim2)
        # score2 = self.t.get_score(hits2)
        # logger.info("score = " + str(score) + ", score = " + str(score2))
        logger.info("score = " + str(score))
        return hits, score


@click.command()
@click.option('--recover', default=False, help='使用上一次训练的模型')
@click.option('--lang', default='fr_en', help='使用的数据集')
@click.option('--output', default='./result/TransE2', help='输出目录，将在此目录下保存权重文件、嵌入文件')
@click.option('--data_enhance', default=True, help='训练时使用数据增强')
@click.option('--visualize', default=False, help='训练时可视化')
@click.option('--gcn', default=False, help='GCN-Align的对齐模块')
@click.option('--my', default=False, help='我设计的对齐模块')
def main(recover, lang, output, data_enhance, visualize, gcn, my):
    result_path = (output + "/%s/") % lang
    data_path = "./data/%s/" % lang
    data_manager = DataManager(
        entity_align_file=data_path + "ref_ent_ids",
        all_entity_file=data_path + "ent_ids_all",
        all_attr_file=data_path + "att2id_all",
        all_relation_file=data_path + "att2id_all",
        all_value_file=data_path + "att_value2id_all",
        all_relation_triple_file=data_path + "triples_struct_all",
        all_attr_triple_file=data_path + "att_triple_all",
        kg1_entity_file=data_path + "ent_ids_1",
        kg2_entity_file=data_path + "ent_ids_2",
    )
    data_manager.load_seeds()
    data_manager.load_entity()
    data_manager.load_relation()
    data_manager.load_attr()
    data_manager.load_value()
    data_manager.load_relation_triple(data_enhance)
    data_manager.load_attr_triple(data_enhance)
    data_manager.summary()
    m = MTransE(
        data_manager,
        checkpoint_path=result_path + "checkpoint.tar",
        embedding_path=result_path + "ATentsembed.txt",
        tensorboard_log_dir=result_path + "log/",

        visualize=visualize
    )
    if visualize:
        m.init_visualize()
    m.init_dataset(gcn, my)
    m.init_model()
    m.init_optimizer()
    m.run_train(need_to_load_checkpoint=recover)


# git pull && CUDA_VISIBLE_DEVICES=1 python main10.py --data_enhance true --gcn true --lang fr_en --output ./result/struct_TransE
if __name__ == '__main__':
    main()
    # main10.py 用 TransE 来做结构嵌入
