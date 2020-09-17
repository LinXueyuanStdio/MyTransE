import logging
from math import exp

import numpy as np
import torch
from torch.utils import data
from torch.utils import tensorboard
from torch import nn
from torch import optim
from torch.optim import optimizer
from typing import Tuple, List, Set, Dict
import numpy as np
from scipy import spatial
import random

import time
import sys

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator

torch.random.manual_seed(123456)


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


class DBP15kDataset(data.Dataset):
    """DBP15k"""

    def __init__(self, dir_triple: str, entity_ids: List[int], value_ids: List[int]):
        # 读取三元组
        self.triple = list()
        with open(dir_triple, 'r') as fr:
            for line in fr:
                line_split = line.split()
                entity = int(line_split[0])
                attr = int(line_split[1])
                value = int(line_split[2])
                self.triple.append((entity, attr, value))
        self.triple = list(set(self.triple))

        # 构造负例
        triple_size = len(self.triple)
        self.negative_entity_ids = []
        self.negative_value_ids = []
        entity_size = len(entity_ids) - 1
        value_size = len(value_ids) - 1
        for i in range(triple_size):
            entity, value, _ = self.triple[i]
            # 随机选entity
            random_index = random.randint(0, entity_size)
            negative_entity = entity_ids[random_index]
            while entity == negative_entity:
                random_index = random.randint(0, entity_size)
                negative_entity = entity_ids[random_index]

            # 随机选value
            random_index = random.randint(0, value_size)
            negative_value = value_ids[random_index]
            while value == negative_value:
                random_index = random.randint(0, value_size)
                negative_value = value_ids[random_index]

            self.negative_entity_ids.append(negative_entity)
            self.negative_value_ids.append(negative_value)
        # 看一看
        for i in range(5):
            print("triple:", self.triple[i],
                  "\tnegative_entity:", self.negative_entity_ids[i],
                  "\tnegative_value:", self.negative_value_ids[i])

    def __len__(self):
        return len(self.triple)

    def __getitem__(self, index):
        entity, value, attr = self.triple[index]
        negative_entity = self.negative_entity_ids[index]
        negative_value = self.negative_value_ids[index]
        return entity, attr, value, negative_entity, negative_value


class TransE(nn.Module):

    def __init__(self,
                 entity_count: int,
                 attr_count: int,
                 value_count: int,
                 device="cuda", norm=1, dim=200, margin=1.0):
        super(TransE, self).__init__()
        self.entity_count = entity_count
        self.attr_count = attr_count
        self.value_count = value_count
        self.device = device
        self.norm = norm
        self.dim = dim
        self.entities_embedding = self._init_entities_embedding()
        self.attrs_embedding = self._init_attrs_embedding()
        self.values_embedding = self._init_values_embedding()
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')

    def _init_entities_embedding(self):
        embedding = nn.Embedding(num_embeddings=self.entity_count + 1,
                                 embedding_dim=self.dim,
                                 padding_idx=self.entity_count)
        uniform_range = 6 / np.sqrt(self.dim)
        embedding.weight.data.uniform_(-uniform_range, uniform_range)
        return embedding

    def _init_attrs_embedding(self):
        embedding = nn.Embedding(num_embeddings=self.attr_count + 1,
                                 embedding_dim=self.dim,
                                 padding_idx=self.attr_count)
        uniform_range = 6 / np.sqrt(self.dim)
        embedding.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        embedding.weight.data[:-1, :].div_(embedding.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return embedding

    def _init_values_embedding(self):
        embedding = nn.Embedding(num_embeddings=self.value_count + 1,
                                 embedding_dim=self.dim,
                                 padding_idx=self.value_count)
        uniform_range = 6 / np.sqrt(self.dim)
        embedding.weight.data.uniform_(-uniform_range, uniform_range)
        return embedding

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        """Return model losses based on the input.

        :param positive_triplets: triplets of positives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :param negative_triplets: triplets of negatives in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: tuple of the model loss, positive triplets loss component, negative triples loss component
        """
        # -1 to avoid nan for OOV vector
        self.entities_embedding.weight.data[:-1, :] \
            .div_(self.entities_embedding.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))
        self.values_embedding.weight.data[:-1, :] \
            .div_(self.values_embedding.weight.data[:-1, :].norm(p=2, dim=1, keepdim=True))

        positive_distances = self._distance(positive_triplets)
        negative_distances = self._distance(negative_triplets)
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        loss = self.criterion(positive_distances, negative_distances, target)

        return loss, positive_distances, negative_distances

    def predict(self, triplets: torch.LongTensor):
        """Calculated dissimilarity score for given triplets.

        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        return self._distance(triplets)

    def _distance(self, triplets):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        assert triplets.size()[1] == 3
        entities = self.entities_embedding(triplets[:, 0])
        attrs = self.attrs_embedding(triplets[:, 1])
        values = self.values_embedding(triplets[:, 2])
        return (entities + attrs - values).norm(p=self.norm, dim=1)


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
        self.train_seeds = self.seeds[:]  # TODO 所有实体参与初始化
        self.test_seeds = self.seeds[:]  # TODO 所有实体参与测试
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

    def get_hits(self, Lvec, Rvec, top_k=(1, 10, 50, 100)):
        sim = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
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


_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_EPOCH = "epoch"
_STEP = "step"
_BEST_SCORE = "best_score"


def load_checkpoint(checkpoint_path: str, model: nn.Module, optim: optimizer.Optimizer) -> Tuple[int, int, float]:
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


def save_checkpoint(model: nn.Module, optim: optimizer.Optimizer, epoch_id: int, step: int, best_score: float):
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optim.state_dict(),
        _EPOCH: epoch_id,
        _STEP: step,
        _BEST_SCORE: best_score
    }, "./result/fr_en/checkpoint.tar")


def save_entity_list(model, embedding_path="./result/fr_en/ATentsembed.txt"):
    with open(embedding_path, 'w') as f:
        d = model.entities_embedding.weight.data.detach().cpu().numpy()
        for i in range(len(d)):
            f.write(" ".join([str(j) for j in d[i].tolist()]))
            f.write("\n")


def read_ids(dir_path, sp="\t"):
    ids = []
    with open(dir_path, encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            ids.append(int(line.strip().split(sp)[0]))
    return ids


def read_triple(triple_path):
    with open(triple_path, 'r') as fr:
        triples = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[1])
            rel = int(line_split[2])
            triples.add((head, rel, tail))
    return list(triples)


entity_list = read_ids("data/fr_en/ent_ids_all")
attr_list = read_ids("data/fr_en/att2id_all")
value_list = read_ids("data/fr_en/att_value2id_all")

entity_count = len(entity_list)
attr_count = len(attr_list)
value_count = len(value_list)
logger.info("entity: " + str(entity_count)
            + " attr: " + str(attr_count)
            + " value: " + str(value_count))
device = "cuda"
learning_rate = 0.001
tensorboard_log_dir = "./result/log/"
checkpoint_path = "./result/fr_en/checkpoint.tar"
batch_size = 1024
train_set = DBP15kDataset('data/fr_en/att_triple_all', entity_list, value_list)
train_generator = data.DataLoader(train_set, batch_size=batch_size)

train_triples = read_triple("data/fr_en/att_triple_all")
train_dataloader_head = data.DataLoader(
    TrainDataset(train_triples, entity_count, attr_count, value_count, 256, 'head-batch'),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=TrainDataset.collate_fn
)
train_dataloader_tail = data.DataLoader(
    TrainDataset(train_triples, entity_count, attr_count, value_count, 256, 'tail-batch'),
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=TrainDataset.collate_fn
)
train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

model = TransE(entity_count, attr_count, value_count, device).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

start_epoch_id = 1
step = 0
best_score = 0.0
epochs = 8000

# if checkpoint_path:
#     start_epoch_id, step, best_score = load_checkpoint(checkpoint_path, model, optimizer)

logger.info(model)

t = Tester()
t.read_entity_align_list('data/fr_en/ref_ent_ids')  # 得到已知对齐实体
entity_pair_count = len(t.left_ids)
for left_entity, right_entity in t.train_seeds:
    model.entities_embedding.weight.data[left_entity] = model.entities_embedding.weight.data[right_entity]

combination_restriction: int = 5000  # 模型认为对齐的实体对的个数
combination_threshold: int = 3  # 小于这个距离则模型认为已对齐
distance2entitiesPair: List[Tuple[float, Tuple[int, int]]] = []
combinationProbability: List[float] = [0] * entity_pair_count  # [0, 1)
correspondingEntity: Dict[int, int] = {}  # ent1 -> ent2


def sigmoid(x) -> float:
    return 1.0 / (1.0 + exp(-x))


def do_combine():
    global correspondingEntity, combinationProbability, distance2entitiesPair
    global combination_restriction, combination_threshold, entity_pair_count
    # 1. 按距离排序
    left_vec = t.get_vec(model.entities_embedding, t.left_ids)
    right_vec = t.get_vec(model.entities_embedding, t.right_ids)
    sim = spatial.distance.cdist(left_vec, right_vec, metric='euclidean')
    distance2entitiesPair = []
    for i in range(entity_pair_count):
        for j in range(entity_pair_count):
            distance2entitiesPair.append((sim[i, j], (t.left_ids[i], t.right_ids[j])))
    sorted(distance2entitiesPair, key=lambda it: it[0])
    # 初始化"模型认为两实体是对齐的"这件事的可信概率
    combinationProbability = [0] * entity_pair_count  # [0, 1)
    # 模型认为的对齐实体
    correspondingEntity = {}

    occupied: Set[int] = set()
    combination_counter = 0
    for dis, (ent1, ent2) in distance2entitiesPair:
        if dis > combination_threshold:
            break
        if ent1 in occupied or ent2 in occupied:
            continue
        correspondingEntity[ent1] = ent2
        correspondingEntity[ent2] = ent1
        occupied.add(ent1)
        occupied.add(ent2)
        combinationProbability[ent1] = sigmoid(combination_threshold - dis)
        combinationProbability[ent2] = sigmoid(combination_threshold - dis)
        if combination_counter == combination_restriction:
            break
        combination_counter += 1
    combination_restriction += 1000


# Training loop
for epoch_id in range(start_epoch_id, epochs + 1):
    logger.info("epoch: " + str(epoch_id))
    loss_impacting_samples_count = 0
    model.train()
    progbar = Progbar(max_step=len(train_generator))
    idx = 0
    if idx > 999 and idx % 500 == 0:
        do_combine()
    for entities, attrs, values, negative_entities, negative_values in train_generator:
        entities = entities.to(device)  # Bx1
        attrs = attrs.to(device)  # Bx1
        values = values.to(device)  # Bx1
        negative_entities = negative_entities.to(device)  # Bx1
        negative_values = negative_values.to(device)  # Bx1

        if random.random() < 0.5:
            # 替换头 [0, 0.5)
            positive_triples = torch.stack((entities, attrs, values), dim=1)  # B x 3
            negative_triples = torch.stack((negative_entities, attrs, values), dim=1)  # B x 3
            soft_h1 = entities
            soft_h2 = negative_entities
        else:
            # 替换尾 [0.5, 1)
            positive_triples = torch.stack((entities, attrs, values), dim=1)  # B x 3
            negative_triples = torch.stack((entities, attrs, negative_values), dim=1)  # B x 3
            soft_h1 = entities
            soft_h2 = entities

        optimizer.zero_grad()
        loss, pd, nd = model(positive_triples, negative_triples)
        loss.mean().backward()
        optimizer.step()

        # 软对齐
        # 换正例的头
        for i in range(batch_size):
            h1 = soft_h1[i]
            print(len(combinationProbability))
            print(h1)
            if random.random() < combinationProbability[h1]:
                soft_h1[i] = correspondingEntity[h1]
        soft_positive_triples = torch.stack((soft_h1, attrs, values), dim=1)  # B x 3
        optimizer.zero_grad()
        loss, pd, nd = model(soft_positive_triples, negative_triples)
        loss.mean().backward()
        optimizer.step()

        # 换负例的头
        for i in range(batch_size):
            h2 = soft_h2[i]
            if random.random() < combinationProbability[h2]:
                soft_h2[i] = correspondingEntity[h2]
        soft_negative_triples = torch.stack((soft_h2, attrs, values), dim=1)  # B x 3
        optimizer.zero_grad()
        loss, pd, nd = model(positive_triples, soft_negative_triples)
        loss.mean().backward()
        optimizer.step()

        step += 1
        idx += 1
        progbar.update(idx, [("loss", loss.mean().data.cpu().numpy()),
                             ("positive", pd.sum().data.cpu().numpy()),
                             ("negative", nd.sum().data.cpu().numpy())])

    if epoch_id % 50 == 0:
        logger.info("loss = " + str(loss.mean().data.cpu().numpy()))
        logger.info("属性消融实验")
        left_vec = t.get_vec(model.entities_embedding, t.left_ids)
        right_vec = t.get_vec(model.entities_embedding, t.right_ids)
        hits = t.get_hits(left_vec, right_vec)
        left_hits_10 = hits["left"][2][1]
        right_hits_10 = hits["right"][2][1]
        score = (left_hits_10 + right_hits_10) / 2
        logger.info("score = " + str(score))
        if score > best_score:
            best_score = score
            save_entity_list(model)
            save_checkpoint(model, optimizer, epoch_id, step, best_score)
