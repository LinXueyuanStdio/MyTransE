import math
import random
import pickle
from time import sleep

import numpy as np
import sys
from scipy import spatial

# lang = sys.argv[1]
# w = float(sys.argv[2])
lang = 'zh_en'
# w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  #


# w = [0.1, 0.2, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  #
w = [0.95, 0.96, 0.965, 0.97, 0.975, 0.98, 0.99, 1]  #


# w = [0.2, 0.5, 0.8]  #


class EAstrategy:
    seeds = []
    linkEmbedding = []
    kg1E = []
    kg2E = []
    EA_results = {}
    SE_embedding = []
    AE_embedding = []

    def read_EA_list(self, EAfile):
        """
        读取对齐实体，后70%为测试集
        """
        ret = []
        with open(EAfile, encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                x = []
                for i in range(2):
                    x.append(int(th[i]))
                ret.append(tuple(x))
            length = int(0.3 * len(ret))
            self.seeds = ret[length:]

    def read_KG1_and_KG2_list(self, kg1file, kg2file):
        """
        读取KG1和KG2的实体id
        """
        with open(kg1file, 'r', encoding='utf-8') as r:
            kg1lines = r.readlines()
        with open(kg2file, 'r', encoding='utf-8') as r:
            kg2lines = r.readlines()
        for line in kg1lines:
            line = line.strip()
            self.kg1E.append(line.split()[0])
        for line in kg2lines:
            line = line.strip()
            self.kg2E.append(line.split()[0])

    def XRR(self, filename="*.pkl"):
        self.linkEmbedding = []
        data = pickle.load(open(filename, 'rb'), encoding='utf-8')
        ent_length = len(data)
        for i in range(ent_length):
            line = data[i]
            line_list = line.tolist()
            self.linkEmbedding.append(line_list)

    def XRA(self, ATEembeddingfile):
        self.linkEmbedding = []
        with open(ATEembeddingfile, 'r', encoding='utf-8') as r:
            ATElines = r.readlines()
        entlength = len(ATElines)
        for i in range(entlength):
            aline = ATElines[i].strip()
            aline_list = aline.split()
            self.linkEmbedding.append(aline_list)

    def EAlinkstrategy(self, RTEembeddingfile, ATEembeddingfile):
        """
        拼接策略
        """
        self.linkEmbedding = []
        RTElines = pickle.load(open(RTEembeddingfile, 'rb'), encoding='utf-8')
        with open(ATEembeddingfile, 'r', encoding='utf-8') as r:
            ATElines = r.readlines()
        entlength = len(ATElines)
        for i in range(entlength):  # list连接操作
            rline = RTElines[i]
            rline_list = rline.tolist()
            aline = ATElines[i].strip()
            aline_list = aline.split()
            aline_list = [float(a) for a in aline_list]
            self.linkEmbedding.append(rline_list + aline_list)

    def read_SE_AE(self, RTEembeddingfile="*.pkl", ATEembeddingfile="*.txt"):
        """
        同时读取属性嵌入和关系嵌入
        """
        self.SE_embedding = []
        self.AE_embedding = []
        RTElines = pickle.load(open(RTEembeddingfile, 'rb'), encoding='utf-8')
        with open(ATEembeddingfile, 'r', encoding='utf-8') as r:
            ATElines = r.readlines()
        entlength = len(ATElines)
        for i in range(entlength):  # 分配权重操作
            rline = RTElines[i]
            rline_list = rline.tolist()
            rline_list_w = [float(j) for j in rline_list]
            self.SE_embedding.append(rline_list_w)
            aline = ATElines[i].strip()
            aline_list = aline.split()
            aline_list_w = [float(j) for j in aline_list]
            self.AE_embedding.append(aline_list_w)

    def EAlinkstrategy_weight_sim(self, w):
        embedding = []
        for i in range(len(self.SE_embedding)):
            SE_w = [j * w for j in self.SE_embedding[i]]
            AE_w = [j * (1 - w) for j in self.AE_embedding[i]]
            add_weight = list(map(lambda x: x[0] + x[1], zip(SE_w, AE_w)))
            embedding.append(add_weight)
        return self.get_sim(embedding)

    def EAlinkstrategy_weight(self, RTEembeddingfile, ATEembeddingfile, w):
        self.linkEmbedding = []
        RTElines = pickle.load(open(RTEembeddingfile, 'rb'), encoding='utf-8')
        with open(ATEembeddingfile, 'r', encoding='utf-8') as r:
            ATElines = r.readlines()
        entlength = len(ATElines)
        for i in range(entlength):  # 分配权重操作
            rline = RTElines[i]
            rline_list = rline.tolist()
            rline_list_w = [float(j) * float(w) for j in rline_list]
            aline = ATElines[i].strip()
            aline_list = aline.split()
            aline_list_w = [float(j) * float(1 - w) for j in aline_list]
            add_weight = list(map(lambda x: x[0] + x[1], zip(rline_list_w, aline_list_w)))
            self.linkEmbedding.append(add_weight)
        print('complete weighting')

    def EAlinkstrategy_iteration(self, RTEembeddingfile):
        self.linkEmbedding = []
        RTElines = pickle.load(open(RTEembeddingfile, 'rb'), encoding='utf-8')
        self.linkEmbedding = RTElines

    def EA_my_strategy(self, metric='euclidean'):
        """
        距离权重策略
        """
        sim1 = self.get_sim(self.SE_embedding, metric)
        sim2 = self.get_sim(self.AE_embedding, metric)
        sim_list = []
        print('距离权重策略', end=" ")
        for ww in w:
            print(str(ww), end=" ")
            sim_list.append(ww * sim1 + (1 - ww) * sim2)
        print("")
        self.result_batch(sim_list, [" %2.2f  " % ww for ww in w])

    def EA_distance_strategy(self, x, y, metric='euclidean'):
        """
        距离权重策略
        """
        sim1 = self.get_sim(self.SE_embedding, metric)
        sim2 = self.get_sim(self.AE_embedding, metric)
        print('距离权重策略 (', x, ", ", y, ")")
        self.result(x * sim1 + y * sim2)

    def get_sim(self, embedding_matrix, metric='cityblock'):
        """
        embedding_matrix是嵌入矩阵，根据嵌入矩阵和测试集seeds计算距离矩阵sim
        """
        Lvec = np.array([embedding_matrix[e1] for e1, e2 in self.seeds])
        Rvec = np.array([embedding_matrix[e2] for e1, e2 in self.seeds])
        sim = spatial.distance.cdist(Lvec, Rvec, metric=metric)
        return sim

    def get_hits(self, top_k=(1, 10, 50, 100), metric='cityblock'):
        self.result(self.get_sim(self.linkEmbedding, metric), top_k)

    def compute_batch_result(self, sim_list, top_k=(1, 10, 50, 100)):
        """
        sim_list是距离矩阵组成的列表，批量计算hits指标
        """
        top_lr_list = []
        top_rl_list = []
        for sim in sim_list:
            top_lr, top_rl = self.compute_result(sim, top_k)
            top_lr_list.append(top_lr)
            top_rl_list.append(top_rl)
        return top_lr_list, top_rl_list

    def compute_result(self, sim, top_k=(1, 10, 50, 100)):
        """
        sim是距离矩阵，根据距离矩阵计算hits指标
        """
        top_lr = [0] * len(top_k)
        for i in range(len(self.seeds)):  # 对于每个KG1实体
            rank = sim[i, :].argsort()
            rank_index = np.where(rank == i)[0][0]
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_lr[j] += 1
        top_rl = [0] * len(top_k)
        for i in range(len(self.seeds)):
            rank = sim[:, i].argsort()
            rank_index = np.where(rank == i)[0][0]
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    top_rl[j] += 1
        return top_lr, top_rl

    def result_batch(self, sim_list, titles=(), top_k=(1, 10, 50, 100)):
        top_lr_list, top_rl_list = self.compute_batch_result(sim_list, top_k)
        self.show_batch_result(top_lr_list, top_rl_list, titles)

    def result(self, sim, top_k=(1, 10, 50, 100)):
        top_lr, top_rl = self.compute_result(sim, top_k)
        self.show_result(top_lr, top_rl, top_k)

    def show_batch_result(self, top_lr_list, top_rl_list, titles=(), top_k=(1, 10, 50, 100)):
        """
        批量打印
        """
        print("          ", end="")
        for i in titles:
            print(i, end="")
        print()
        print('For each left:')
        for i in range(len(top_k)):
            print('Hits@%3d: ' % (top_k[i]), end="")
            for top_lr in top_lr_list:
                print('%.2f%% ' % (top_lr[i] / len(self.seeds) * 100), end="")
            print()
        print('For each right:')
        for i in range(len(top_k)):
            print('Hits@%3d: ' % (top_k[i]), end="")
            for top_rl in top_rl_list:
                print('%.2f%% ' % (top_rl[i] / len(self.seeds) * 100), end="")
            print()

    def show_result(self, top_lr, top_rl, top_k=(1, 10, 50, 100)):
        """
        打印一个
        """
        print('For each left:')
        for i in range(len(top_lr)):
            print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(self.seeds) * 100))
        print('For each right:')
        for i in range(len(top_rl)):
            print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(self.seeds) * 100))

        return ((top_lr[0] / len(self.seeds)) + (top_rl[0] / len(self.seeds))) / 2


test = EAstrategy()

test.read_EA_list('data/' + lang + '/ref_ent_ids')  # 得到已知对齐实体
test.read_KG1_and_KG2_list('data/' + lang + '/ent_ids_1', 'data/' + lang + '/ent_ids_2')  # 得到kg1和kg2中的实体

print('language:' + lang)

struct_embedding = './rdgcn_zh_en.pkl'
attribute_embedding = 'result/test4/' + lang + '.txt'
# attribute_embedding = 'result/test3/ATentsembed.txt_score_44'
test.read_SE_AE(struct_embedding, attribute_embedding)
# print('拼接策略')
# 拼接策略
# test.EAlinkstrategy(struct_embedding, attribute_embedding)  # 连接策略
# test.EAlinkstrategy('2.pkl', './result/ja_en/ATentsembed.txt')  # 连接策略
# test.get_hits(metric='cityblock')

print('距离权重策略')
# 距离权重策略
# test.EA_my_strategy("cityblock")
sim1 = test.get_sim(test.SE_embedding, "cityblock")
sim2 = test.get_sim(test.AE_embedding, "cityblock")
for x in range(10, 21):
    x /= 10
    print("x=", x)
    sim_list = []
    titles = ()
    for y in range(-19, 21):
        if y == -10 or y == 0 or y == 10 or y == 20:
            test.result_batch(sim_list, titles)
            sim_list = []
            titles = ()
            continue
        y /= 20
        print('距离权重策略 (', x, ", ", y, ")")
        titles += tuple(" %2.2f  " % y)
        sim = x * sim1 + y * sim2
        sim_list.append(sim)

# 权重策略
# ww = 0.8
# # test.EAlinkstrategy_weight('data/'+lang+'/RTentsembed.pkl','data/'+lang+'/ATentsembed.txt', ww) #连接策略
# sim_list = []
# for ww in w:
#     print('AE+SE 权重策略 w=' + str(ww))
#     sim_list.append(test.EAlinkstrategy_weight_sim(ww))
# top_lr_list, top_rl_list = test.compute_batch_result(sim_list)
# test.show_batch_result(top_lr_list, top_rl_list, [" %2.2f  " % ww for ww in w])

# 迭代策略
# test.EAlinkstrategy_iteration('results/'+'emb_it_'+lang+'.pkl')
# test.get_hits()
# with open('data/'+lang+'/EA_results_'+str(w)+'.txt','w',encoding='utf-8') as w:
#     w.write(str(test.EA_results))
#     w.write('\n')

# 消融实验
# print("关系消融实验")
# test.XRR(struct_embedding)
# test.get_hits()
#
# print("属性消融实验")
# test.XRA(attribute_embedding)
# test.get_hits()

# 迭代权重策略
# test.EAlinkstrategy_iteration('results/'+'emb_itwe_0.5_'+lang+'.pkl')
# test.get_hits()
# with open('data/'+lang+'/EA_results_'+str(w)+'.txt','w',encoding='utf-8') as w:
#     w.write(str(test.EA_results))
#     w.write('\n')
