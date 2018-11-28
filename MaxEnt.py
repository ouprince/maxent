# -*- coding:utf-8 -*-
# 采用 GIS 算法实现最大熵模型
import sys,os
from collections import defaultdict
import math

reload(sys)
sys.setdefaultencoding("utf-8")

# 最大熵模型
class MaxEnt(object):
    def __init__(self):
        self.feats = defaultdict(int)
        self.trainset = [] # 训练集
        self.labels = set() # 标签集
    
    def load_data(self,file):
        f = open(file)
        for line in f:
            fields = line.strip().split()
            if len(fields) < 2:continue # 特征数要大于两列
            label = fields[0] # 默认第一列为标签
            self.labels.add(label)
            for s in set(fields[1:]):
                self.feats[(label,s)] += 1 # (标签,词)
            self.trainset.append(fields)
        f.close()
        
    def _initparams(self):# 初始化参数
        self.size = len(self.trainset)
        self.M = max([len(record)-1 for record in self.trainset]) # GIS 训练算法的 M 参数
        self.ep_ = [0.0]*len(self.feats)
        for i,f in enumerate(self.feats): # 计算经验分布的特征期望
            self.ep_[i] = float(self.feats[f]) / float(self.size)
            self.feats[f] = i # 为每个特征分配id
        self.w = [0.0]*len(self.feats)
        self.lastw = self.w
        
    def probwgt(self,features,label):# 计算 feature 下 label 的指数权重
        wgt = 0.0
        for f in features:
            if (label,f) in self.feats:
                wgt += self.w[self.feats[(label,f)]]
        return math.exp(wgt)
        
    def Ep(self): # 特征函数
        ep = [0.0]*len(self.feats)
        for record in self.trainset:# 从训练集中迭代输出特征
            features = record[1:]
            prob = self.calprob(features) # 计算 feature 下每个标签的概率 [(概率,标签)]
            for f in features:
                for w,l in prob:
                    if (l,f) in self.feats:
                        idx = self.feats[(l,f)]
                        ep[idx] += w*(1.0/self.size)
        return ep
    
    def _convergence(self,lastw,w): # 收敛唯一终止条件
        for w1,w2 in zip(lastw,w):
            if abs(w1-w2) >= 0.001:return False
        return True
            
    def calprob(self,features):
        wgts = [(self.probwgt(features,l),l) for l in self.labels] # 获得[(weight,label)]
        Z = sum([w for w,l in wgts]) # 归一化参数
        prob = [(w/Z,l) for w,l in wgts]
        return prob
        
    def train(self,max_iter = 10000): # 训练样本的主函数，默认迭代次数 1000
        self._initparams()
        for i in xrange(max_iter):
            print 'iter %d ...' %(i+1)
            self.ep = self.Ep() # 计算模型分布的特征期望
            self.lastw = self.w[:]
            for i,win in enumerate(self.w):
                delta = 1.0 / self.M * math.log(self.ep_[i]/self.ep[i])
                self.w[i] += delta # 更新 w
            print self.w
            
            if  self._convergence(self.lastw,self.w): # 判断是否收敛
                break
    
    def predict(self,input):
        features = input.strip().split()
        prob = self.calprob(features)
        prob.sort(reverse = True)
        return prob
    
if __name__ == "__main__":
    model = MaxEnt()
    model.load_data("data.txt")
    model.train()
    print model.predict("Rainy Happy Dry")
        
