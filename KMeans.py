import numpy as np
import random

class KMeans:
    def __init__(self,x_train,K):
        self.x_train = x_train
        self.K = K

    def euclidean(self,data,center):
        differ = data - center
        return np.linalg.norm(differ,axis=1)

    def process(self):
            center_pos = random.sample(range(0,len(self.x_train)),self.K)
            center = self.x_train[center_pos] # data[center_pos] have K 
            sigma = np.zeros(len(self.x_train[0]))

            for _ in range(100):
                arg = np.zeros(len(self.x_train))
                #算每個點的距離並給定是哪一群
                for i in range(len(self.x_train)):
                    arg[i] = np.argmin([self.euclidean(self.x_train[i],center)])

                #下一次的center更新
                for c_pos in range(self.K):
                    center[c_pos] =  np.mean(self.x_train[arg == c_pos],axis=0)

            for c_pos in range(self.K):
                sigma[c_pos] = np.sum(self.euclidean(self.x_train[arg == c_pos],center[c_pos])) / len(self.x_train[arg == c_pos])

            return center,sigma