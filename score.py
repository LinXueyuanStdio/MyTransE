import math
import random
import pickle
import numpy as np
import sys
from scipy import spatial


# lang = sys.argv[1]
# w = float(sys.argv[2])
lang = 'fr_en'
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #

class EAstrategy:
    seeds = []
    linkEmbedding=[]
    kg1E=[]
    kg2E=[]
    EA_results={}

    def read_EA_list(self,EAfile):
        # with open(EAfile,'r',encoding='utf-8') as r:
        #     lines=r.readlines()
        # for line in lines:
        #     line=line.strip()
        #     e1, e2=line.split()
        #     if e1 in self.seeds_map:
        #         print('error,',e1)
        #     else:
        #         self.seeds_map[e1]=e2

        ret = []
        with open(EAfile, encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                x = []
                for i in range(2):
                    x.append(int(th[i]))
                ret.append(tuple(x))
            self.seeds = ret

    def read_KG1_and_KG2_list(self,kg1file,kg2file):
        with open(kg1file,'r',encoding='utf-8') as r:
            kg1lines=r.readlines()
        with open(kg2file,'r',encoding='utf-8') as r:
            kg2lines=r.readlines()
        for line in kg1lines:
            line=line.strip()
            self.kg1E.append(line.split()[0])
        for line in kg2lines:
            line = line.strip()
            self.kg2E.append(line.split()[0])

    def XRR(self, RTEembeddingfile):
        RTElines = pickle.load(open(RTEembeddingfile, 'rb'), encoding='utf-8')
        entlength = len(RTElines)
        for i in range(entlength):
            rline = RTElines[i]
            rline_list = rline.tolist()
            self.linkEmbedding.append(rline_list)

    def XRA(self, ATEembeddingfile):
        with open(ATEembeddingfile, 'r', encoding='utf-8') as r:
            ATElines = r.readlines()
        entlength = len(ATElines)
        for i in range(entlength):
            aline = ATElines[i].strip()
            aline_list = aline.split()
            self.linkEmbedding.append(aline_list)

    def EAlinkstrategy(self,RTEembeddingfile,ATEembeddingfile):
        RTElines=pickle.load(open(RTEembeddingfile,'rb'),encoding='utf-8')
        with open(ATEembeddingfile,'r',encoding='utf-8') as r:
            ATElines=r.readlines()
        entlength=len(ATElines)
        for i in range(entlength): #list连接操作
            rline=RTElines[i]
            rline_list=rline.tolist()
            aline=ATElines[i].strip()
            aline_list=aline.split()
            self.linkEmbedding.append(rline_list+aline_list)

    def EAlinkstrategy_weight(self,RTEembeddingfile,ATEembeddingfile, w):
        RTElines=pickle.load(open(RTEembeddingfile,'rb'),encoding='utf-8')
        with open(ATEembeddingfile,'r',encoding='utf-8') as r:
            ATElines=r.readlines()
        entlength=len(ATElines)
        for i in range(entlength): #分配权重操作
            rline=RTElines[i]
            rline_list=rline.tolist()
            rline_list_w = [float(j) * float(w) for j in rline_list]
            aline=ATElines[i].strip()
            aline_list=aline.split()
            aline_list_w = [float(j) * float(1-w) for j in aline_list]
            add_weight = list(map(lambda x:x[0]+x[1], zip(rline_list_w, aline_list_w)))
            self.linkEmbedding.append(add_weight)
        print('complete weighting')

    def EAlinkstrategy_iteration(self, RTEembeddingfile):
        RTElines = pickle.load(open(RTEembeddingfile, 'rb'), encoding='utf-8')
        self.linkEmbedding = RTElines
    # def distance(self,yuzhi):
    #     count = 0
    #     for i in self.kg1E:
    #         count += 1
    #         align_id_list={} #id:juli
    #         for j in self.kg2E:
    #             dimension=len(self.linkEmbedding[int(j)])
    #             now_dis=0
    #             for k in range(dimension):
    #                 now_dis+=abs(float(self.linkEmbedding[int(i)][k])-float(self.linkEmbedding[int(j)][k])) #L1正则化计算
    #             if now_dis<yuzhi:
    #                 align_id_list[j]=now_dis
    #         #对align_id_list按距离进行排序
    #         sort_align_id_list=sorted(align_id_list.items(),key=lambda x:x[1]) #距离从小到大排序
    #         sort_align_id_list=sort_align_id_list[:100]
    #         self.EA_results[i]=sort_align_id_list
    #         if count % 10 == 0:
    #             print('w=' + str(w) + ' process:' + str(count) + '/' + str(len(self.kg1E)))
    #     print('complete distancing')

    def get_hits(self, top_k=(1, 10, 50, 100)):
        Lvec = np.array([self.linkEmbedding[e1] for e1,e2 in self.seeds])
        Rvec = np.array([self.linkEmbedding[e2] for e1,e2 in self.seeds])
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
        for i in range(len(top_lr)):
            print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(self.seeds) * 100))
        print('For each right:')
        for i in range(len(top_rl)):
            print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(self.seeds) * 100))

        return ((top_lr[0] / len(self.seeds)) + (top_rl[0] / len(self.seeds))) / 2


test=EAstrategy()


test.read_EA_list('data/' + lang + '/ref_ent_ids')  # 得到已知对齐实体
test.read_KG1_and_KG2_list('data/' + lang + '/ent_ids_1', 'data/' + lang + '/ent_ids_2')  # 得到kg1和kg2中的实体

print('language:' + lang)
# 拼接策略
# test.EAlinkstrategy('data/' + lang + '/RTentsembed.pkl', 'data/' + lang + '/ATentsembed.txt')  # 连接策略
# test.get_hits()

# 权重策略
# ww = 0.8
# print('w='+str(ww))
# test.EAlinkstrategy_weight('data/'+lang+'/RTentsembed.pkl','data/'+lang+'/ATentsembed.txt', ww) #连接策略
# test.get_hits()

# 迭代策略
# test.EAlinkstrategy_iteration('results/'+'emb_it_'+lang+'.pkl')
# test.get_hits()
# with open('data/'+lang+'/EA_results_'+str(w)+'.txt','w',encoding='utf-8') as w:
#     w.write(str(test.EA_results))
#     w.write('\n')

# 消融实验
# print("关系消融实验")
# test.XRR('data/' + lang + '/RTentsembed.pkl')
# test.get_hits()

# print("属性消融实验")
# test.XRA('data/'+lang+'/ATentsembed.txt')
test.XRA('res/entity2vec.bern')
test.get_hits()

# 迭代权重策略
# test.EAlinkstrategy_iteration('results/'+'emb_itwe_0.5_'+lang+'.pkl')
# test.get_hits()
# with open('data/'+lang+'/EA_results_'+str(w)+'.txt','w',encoding='utf-8') as w:
#     w.write(str(test.EA_results))
#     w.write('\n')