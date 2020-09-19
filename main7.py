from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _thread
import sys
import time
from math import exp
from random import random
from typing import List, Tuple, Set

from scipy import spatial
import numpy as np
import torch
from torch import nn
from torch.optim import optimizer
from torch.utils import tensorboard
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataloader import BidirectionalOneShotIterator
from dataloader import TrainDataset
from dataloader import TestDataset
import tensorflow as tf
import tensorboard as tb
import logging

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
torch.random.manual_seed(123456)


# region model
class KGEModel(nn.Module):
    def __init__(self, train_seeds, nentity, nrelation, nvalue, hidden_dim, gamma, double_entity_embedding=False,
                 double_relation_embedding=False):
        super(KGEModel, self).__init__()
        # self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.nvalue = nvalue
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
        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim
        self.value_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim

        entity_weight = torch.zeros(nentity, self.entity_dim)
        nn.init.uniform_(
            tensor=entity_weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        for left_entity, right_entity in train_seeds:
            entity_weight[left_entity] = entity_weight[right_entity]
        self.entity_embedding = nn.Parameter(entity_weight)
        # nn.init.normal_(self.entity_embedding)

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        # nn.init.normal_(self.relation_embedding)
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.value_embedding = nn.Parameter(torch.zeros(nvalue, self.value_dim))
        # nn.init.normal_(self.value_embedding)
        nn.init.uniform_(
            tensor=self.value_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def forward(self, sample, mode='single'):
        if mode == 'single':
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
                self.value_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
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
                self.value_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':

            head_part, tail_part = sample
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
                self.value_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        score = self.TransE(head, relation, tail, mode)

        return score

    def TransE(self, head, relation, tail, mode):

        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):

        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
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

    @staticmethod
    def train_step(model, optimizer, positive_sample, negative_sample, subsampling_weight, mode, device="cuda"):

        model.train()
        optimizer.zero_grad()

        positive_sample = positive_sample.to(device)
        negative_sample = negative_sample.to(device)
        subsampling_weight = subsampling_weight.to(device)

        negative_score = model((positive_sample, negative_sample), mode=mode)
        negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        loss.backward()
        optimizer.step()

        return loss.item()


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
    seeds: List[Tuple[int, int]] = []  # (m, 2) 对齐的实体对(a,b)称a为左实体，b为右实体
    train_seeds: List[Tuple[int, int]] = []  # (0.8m, 2)
    test_seeds: List[Tuple[int, int]] = []  # (0.2m, 2)
    linkEmbedding = []
    kg1E = []
    kg2E = []
    EA_results = {}

    def read_entity_align_list(self, entity_align_file_path):
        ret = []
        with open(entity_align_file_path, encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ret.append((int(th[0]), int(th[1])))
            self.seeds = ret
        # 80%训练集，20%测试集
        train_percent = 0.3
        train_max_idx = int(train_percent * len(self.seeds))
        self.train_seeds = self.seeds[:train_max_idx]
        self.test_seeds = self.seeds[train_max_idx:]
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


# endregion

# region 保存与加载模型，恢复训练状态
_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_EPOCH = "epoch"
_STEP = "step"
_BEST_SCORE = "best_score"
_LOSS = "loss"


def load_checkpoint(model: nn.Module, optim: optimizer.Optimizer,
                    checkpoint_path="./result/fr_en/checkpoint.tar") -> Tuple[int, int, float, float]:
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
    loss = checkpoint[_LOSS]
    return start_epoch_id, step, best_score, loss


def save_checkpoint(model: nn.Module, optim: optimizer.Optimizer,
                    epoch_id: int, step: int, best_score: float, loss: float,
                    save_path="./result/fr_en/checkpoint.tar"):
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optim.state_dict(),
        _EPOCH: epoch_id,
        _STEP: step,
        _BEST_SCORE: best_score,
        _LOSS: loss,
    }, save_path)


def save_entity_embedding_list(model, embedding_path="./result/fr_en/ATentsembed.txt"):
    with open(embedding_path, 'w') as f:
        d = model.entity_embedding.data.detach().cpu().numpy()
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


def read_triple(triple_path):
    with open(triple_path, 'r') as fr:
        triple = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[1])
            rel = int(line_split[2])
            triple.add((head, rel, tail))
    return list(triple)


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
        count += 1
        bar.update(count, [("step", count)])
    return triple + triple_replace_with_align


# endregion

class TransE:
    def __init__(self,
                 # input paths
                 entity_align_file="data/fr_en/ref_ent_ids",
                 all_entity_file="data/fr_en/ent_ids_all",
                 all_attr_file="data/fr_en/att2id_all",
                 all_value_file="data/fr_en/att_value2id_all",
                 all_triple_file="data/fr_en/att_triple_all",
                 # output paths
                 checkpoint_path="./result/TransE/fr_en/checkpoint.tar",
                 embedding_path="./result/TransE/fr_en/ATentsembed.txt",
                 tensorboard_log_dir="./result/TransE/fr_en/log/",

                 device="cuda",
                 learning_rate=0.001,
                 visualize=False
                 ):
        self.entity_align_file = entity_align_file
        self.all_entity_file = all_entity_file
        self.all_attr_file = all_attr_file
        self.all_value_file = all_value_file
        self.all_triple_file = all_triple_file
        self.device = device
        self.visualize = visualize
        self.tensorboard_log_dir = tensorboard_log_dir
        self.checkpoint_path = checkpoint_path
        self.embedding_path = embedding_path

        self.learning_rate = learning_rate

    def init_data(self):
        self.t = Tester()
        self.t.read_entity_align_list(self.entity_align_file)  # 得到已知对齐实体
        self.entity_list, self.entity_name_list = read_ids_and_names(self.all_entity_file)
        self.attr_list, _ = read_ids_and_names(self.all_attr_file)
        self.value_list, _ = read_ids_and_names(self.all_value_file)
        self.train_triples = read_triple(self.all_triple_file)

        self.entity_count = len(self.entity_list)
        self.attr_count = len(self.attr_list)
        self.value_count = len(self.value_list)

        logger.info("entity: " + str(self.entity_count)
                    + " attr: " + str(self.attr_count)
                    + " value: " + str(self.value_count))

    def append_align_triple(self):
        self.train_triples = append_align_triple(self.train_triples, self.t.train_seeds)

    def init_dataset(self):
        train_dataloader_head = DataLoader(
            TrainDataset(self.train_triples, self.entity_count, self.attr_count, self.value_count, 512, 'head-batch'),
            batch_size=1024,
            shuffle=False,
            num_workers=4,
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_tail = DataLoader(
            TrainDataset(self.train_triples, self.entity_count, self.attr_count, self.value_count, 512, 'tail-batch'),
            batch_size=1024,
            shuffle=False,
            num_workers=4,
            collate_fn=TrainDataset.collate_fn
        )
        self.train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

    def init_model(self):
        self.model = KGEModel(
            self.t.seeds,  # 所有seed
            nentity=self.entity_count,
            nrelation=self.attr_count,
            nvalue=self.value_count,
            hidden_dim=200,
            gamma=24.0,
        ).to(self.device)

    def init_optimizer(self):
        self.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )

    def init_soft_align(self):
        self.combination_threshold = 3  # 小于这个距离则模型认为已对齐
        self.combination_restriction = 5000  # 模型认为对齐的实体对的个数
        self.distance2entitiesPair: List[Tuple[int, Tuple[int, int]]] = []
        self.combinationProbability: List[float] = [0] * self.entity_count  # [0, 1)
        self.correspondingEntity = {}
        self.model_think_align_entities = []
        self.model_is_able_to_predict_align_entities = False

    def soft_align(self, positive_sample, mode='single'):
        batch_size = positive_sample.size()[0]
        # positive_sample (batch_size, 3)
        #     batch_size 个 (entity, attr, value) 的三元组
        # negative_sample (batch_size, negative_sample_size)
        #     batch_size 个长度为 negative_sample_size 的 (neg_id1, neg_id2, ...) 替换用的待使用id
        # 设 e 是正例实体，e' 是负例实体，e* 是模型认为的e的对齐实体
        # 1. head-batch
        # (e, a, v) + (e'1, e'2, ..., e'n) ->
        #   ((e, a, v), (e'1, a, v))
        #   ((e, a, v), (e'2, a, v))
        #   ...
        #   ((e, a, v), (e'n, a, v))
        # 2. tail-batch
        # (e, a, v) + (v'1, v'2, ..., v'n) ->
        #   ((e, a, v), (e, a, v'1))
        #   ((e, a, v), (e, a, v'2))
        #   ...
        #   ((e, a, v), (e, a, v'n))
        soft_positive_sample = positive_sample.clone()
        if mode == "head-batch":
            # 负例是随机替换头部
            # (neg_id1, neg_id2, ...) 是实体id
            # ((e, a, v), (e'1, a, v))
            # 已有 (e, a, v) + (e'1, e'2, ..., e'n)
            for i in range(batch_size):
                # 1. 模型认为头部是对齐的
                h1 = soft_positive_sample[i][0].item()
                if self.combinationProbability[h1] >= 0.5 and h1 in self.correspondingEntity:  # 如果可信
                    # 希望 (e, a, v) (e', a, v) -> (e*, a, v) (e', a, v)
                    h1_cor = self.correspondingEntity[h1]  # 获取模型认为的对齐实体
                    soft_positive_sample[i][0] = h1_cor  # 替换为模型认为的对齐实体
        elif mode == "tail-batch":
            # 负例是随机替换尾部
            # (neg_id1, neg_id2, ...) 是属性值id
            # ((e, a, v), (e, a, v'2))
            # 已有 (e, a, v) + (v'1, v'2, ..., v'n)
            for i in range(batch_size):
                # 1. 模型认为头部是对齐的
                h1 = soft_positive_sample[i][0].item()
                if self.combinationProbability[h1] >= 0.5 and h1 in self.correspondingEntity:  # 如果可信
                    # 希望 (e, a, v) (e', a, v) -> (e*, a, v) (e', a, v)
                    h1_cor = self.correspondingEntity[h1]  # 获取模型认为的对齐实体
                    soft_positive_sample[i][0] = h1_cor  # 替换为模型认为的对齐实体
        return soft_positive_sample

    def do_combine(self, thread_name, sim):
        # sim[i, j] 表示在 Lvec 的实体 i 到 Rvec 的实体 j 的距离
        logger.info(thread_name + " " + "模型对齐中")
        computing_time = time.time()
        # 1. 按距离排序
        self.distance2entitiesPair: List[Tuple[int, Tuple[int, int]]] = []
        filtered = np.where(sim == np.amin(sim, axis=1))
        for i, j in zip(filtered[0], filtered[1]):
            self.distance2entitiesPair.append((sim[i, j], (self.t.left_ids[i], self.t.right_ids[j])))
        filter_time = time.time()
        logger.info(thread_name + " " + "距离小于 "
                    + str(self.combination_threshold) + " 的实体对有 "
                    + str(len(self.distance2entitiesPair)) + " 个")
        logger.info(thread_name + " " + "扁平化，用时 " + str(int(filter_time - computing_time)) + " 秒")
        # 2.初始化"模型认为两实体是对齐的"这件事的可信概率
        combinationProbability: List[float] = [0] * self.entity_count  # [0, 1)
        # 3.模型认为的对齐实体
        correspondingEntity = {}
        self.model_think_align_entities = []

        occupied: Set[int] = set()
        combination_counter = 0
        sigmoid = lambda x: 1.0 / (1.0 + exp(-x))
        for dis, (ent1, ent2) in self.distance2entitiesPair:
            if dis > self.combination_threshold:
                # 超过可信范围，不可信
                continue
            # 距离在可信范围内
            if ent1 in occupied or ent2 in occupied:
                continue
            if combination_counter >= self.combination_restriction:
                break
            combination_counter += 1
            self.correspondingEntity[ent1] = ent2
            self.correspondingEntity[ent2] = ent1
            self.model_think_align_entities.append((ent1, ent2))
            occupied.add(ent1)
            occupied.add(ent2)
            combinationProbability[ent1] = sigmoid(self.combination_threshold - dis)  # 必有 p > 0.5
            combinationProbability[ent2] = sigmoid(self.combination_threshold - dis)
        logger.info(thread_name + " " + "对齐了 " + str(len(self.model_think_align_entities)) + " 个实体")
        self.combination_restriction += 1000

        self.model_is_able_to_predict_align_entities = False  # 上锁
        self.combinationProbability = combinationProbability
        self.correspondingEntity = correspondingEntity
        self.model_is_able_to_predict_align_entities = True  # 解锁
        align_time = time.time()
        logger.info(thread_name + " " + "模型对齐完成，用时 " + str(int(align_time - filter_time)) + " 秒")

    def run_train(self, need_to_load_checkpoint=True):
        logger.info("start training")
        init_step = 1
        total_steps = 200001
        test_steps = 5000
        last_loss = 100
        score = 0
        last_score = score

        if need_to_load_checkpoint:
            _, init_step, score, last_loss = load_checkpoint(self.model, self.optim, self.checkpoint_path)
            last_score = score

        summary_writer = tensorboard.SummaryWriter(log_dir=self.tensorboard_log_dir)
        progbar = Progbar(max_step=total_steps - init_step)
        start_time = time.time()
        for step in range(init_step, total_steps):
            positive_sample, negative_sample, subsampling_weight, mode = next(self.train_iterator)
            loss = self.model.train_step(self.model, self.optim,
                                         positive_sample, negative_sample,
                                         subsampling_weight, mode, self.device)
            # 软对齐
            # 根据模型认为的对齐实体，修改 positive_sample，negative_sample，再训练一轮
            if self.model_is_able_to_predict_align_entities:
                soft_positive_sample = self.soft_align(positive_sample, mode)
                loss2 = self.model.train_step(self.model, self.optim,
                                              soft_positive_sample, negative_sample,
                                              subsampling_weight, mode, self.device)
                loss = (loss + loss2) / 2

            progbar.update(step - init_step + 1, [
                ("loss", loss),
                ("cost", round((time.time() - start_time))),
                ("aligned", len(self.model_think_align_entities))
            ])
            if self.visualize:
                summary_writer.add_scalar(tag='Loss/train', scalar_value=loss, global_step=step)

            if step > init_step and step % test_steps == 0:
                logger.info("\n计算距离中")
                computing_time = time.time()
                left_vec = self.t.get_vec2(self.model.entity_embedding, self.t.left_ids)
                right_vec = self.t.get_vec2(self.model.entity_embedding, self.t.right_ids)
                sim = spatial.distance.cdist(left_vec, right_vec, metric='euclidean')
                logger.info("计算距离完成，用时 " + str(int(time.time() - computing_time)) + " 秒")
                self.do_combine("step-" + str(step), sim)
                # try:
                #     logger.info("启动线程，获取模型认为的对齐实体")
                #     _thread.start_new_thread(self.do_combine, ("Thread of step-" + str(step), sim,))
                # except SystemExit:
                #     logger.error("Error: 无法启动线程")
                logger.info("属性消融实验")
                hits = self.t.get_hits(left_vec, right_vec, sim)
                hits_left = hits["left"]
                hits_right = hits["right"]
                left_hits_10 = hits_left[2][1]
                right_hits_10 = hits_right[2][1]
                score = (left_hits_10 + right_hits_10) / 2
                logger.info("score = " + str(score))

                if self.visualize:
                    summary_writer.add_embedding(tag='Embedding',
                                                 mat=self.model.entity_embedding,
                                                 metadata=self.entity_name_list,
                                                 global_step=step)
                    summary_writer.add_scalar(tag='Hits@1/left', scalar_value=hits_left[0][1], global_step=step)
                    summary_writer.add_scalar(tag='Hits@10/left', scalar_value=hits_left[1][1], global_step=step)
                    summary_writer.add_scalar(tag='Hits@50/left', scalar_value=hits_left[2][1], global_step=step)
                    summary_writer.add_scalar(tag='Hits@100/left', scalar_value=hits_left[3][1], global_step=step)

                    summary_writer.add_scalar(tag='Hits@1/right', scalar_value=hits_right[0][1], global_step=step)
                    summary_writer.add_scalar(tag='Hits@10/right', scalar_value=hits_right[1][1], global_step=step)
                    summary_writer.add_scalar(tag='Hits@50/right', scalar_value=hits_right[2][1], global_step=step)
                    summary_writer.add_scalar(tag='Hits@100/right', scalar_value=hits_right[3][1], global_step=step)
                if score > last_score:
                    last_score = score
                    save_checkpoint(self.model, self.optim, 1, step, score, loss, self.checkpoint_path)
                    save_entity_embedding_list(self.model, self.embedding_path)

    def run_test(self):
        load_checkpoint(self.model, self.optim, self.checkpoint_path)
        logger.info("\n属性消融实验")
        left_vec = self.t.get_vec2(self.model.entity_embedding, self.t.left_ids)
        right_vec = self.t.get_vec2(self.model.entity_embedding, self.t.right_ids)
        hits = self.t.get_hits(left_vec, right_vec)
        hits_left = hits["left"]
        hits_right = hits["right"]
        left_hits_10 = hits_left[2][1]
        right_hits_10 = hits_right[2][1]
        score = (left_hits_10 + right_hits_10) / 2
        logger.info("score = " + str(score))


def train_model_for_fr_en(result_path="./result/TransE2/fr_en/"):
    m = TransE(entity_align_file="data/fr_en/ref_ent_ids",
               all_entity_file="data/fr_en/ent_ids_all",
               all_attr_file="data/fr_en/att2id_all",
               all_value_file="data/fr_en/att_value2id_all",
               all_triple_file="data/fr_en/att_triple_all",

               checkpoint_path=result_path + "checkpoint.tar",
               embedding_path=result_path + "ATentsembed.txt",
               tensorboard_log_dir=result_path + "log/"
               )
    m.init_data()
    # m.append_align_triple()
    m.init_soft_align()
    m.init_dataset()
    m.init_model()
    m.init_optimizer()
    m.run_train(need_to_load_checkpoint=False)


def train_model_for_ja_en(result_path="./result/TransE2/ja_en/"):
    m = TransE(entity_align_file="data/ja_en/ref_ent_ids",
               all_entity_file="data/ja_en/ent_ids_all",
               all_attr_file="data/ja_en/att2id_all",
               all_value_file="data/ja_en/att_value2id_all",
               all_triple_file="data/ja_en/att_triple_all",

               checkpoint_path=result_path + "checkpoint.tar",
               embedding_path=result_path + "ATentsembed.txt",
               tensorboard_log_dir=result_path + "log/")
    m.init_data()
    # m.append_align_triple()
    m.init_soft_align()
    m.init_dataset()
    m.init_model()
    m.init_optimizer()
    m.run_train(need_to_load_checkpoint=False)


def train_model_for_zh_en(result_path="./result/TransE2/zh_en/"):
    m = TransE(entity_align_file="data/zh_en/ref_ent_ids",
               all_entity_file="data/zh_en/ent_ids_all",
               all_attr_file="data/zh_en/att2id_all",
               all_value_file="data/zh_en/att_value2id_all",
               all_triple_file="data/zh_en/att_triple_all",

               checkpoint_path=result_path + "checkpoint.tar",
               embedding_path=result_path + "ATentsembed.txt",
               tensorboard_log_dir=result_path + "log/")
    m.init_data()
    # m.append_align_triple()
    m.init_soft_align()
    m.init_dataset()
    m.init_model()
    m.init_optimizer()
    m.run_train(need_to_load_checkpoint=False)


def test_model():
    m = TransE()
    m.init_data()
    m.init_model()
    m.init_optimizer()
    m.run_test()


train_model_for_fr_en()
# train_model_for_ja_en()
# train_model_for_zh_en()
