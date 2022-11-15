import KMeans
import numpy as np

from KMeans import KMeans

class model():
    def __init__(self,epochs,learning_rate,m,sigma,K):
        self.theta = 1.0
        #學習率
        self.m , self.sigma = m,sigma
        self.epochs = epochs
        self.learning_rate = learning_rate
        #設定W初始值，先隨便給，目前K-means有想到怎麼改快一點，只是先pass做其他的
        self.w = np.random.rand(K)
        self.F = 0

    def euclidean(self,data,center):
        differ = data - center
        return np.linalg.norm(differ,axis=1)

    def activation(self,data,m,sigma):
        return np.exp(-1 / (2 * (sigma ** 2)) * (self.euclidean(m,data) ** 2) )

    def predict(self,data):
        active = self.activation(data,self.m,self.sigma)

        F = self.w.T.dot(active) + self.theta

        return active,F

    def train(self,x_train,y_train):
        for i in range(1,self.epochs + 1):
            loss_sum = 0
            for data,label in zip(x_train,y_train):
                #正向傳遞
                active,self.F = self.predict(data)

                loss_sum += ((label - self.F) ** 2) / 2
                #倒傳遞更新參數
                new_w = self.w + (self.learning_rate  * (label - self.F) * active)

                preprocess_m = self.learning_rate *(label - self.F) * self.w * active * (1 / (self.sigma ** 2))
                data_sub_m = data - self.m
                new_m = self.m + np.array([preprocess_m[i] * data_sub_m[i] for i in range(len(preprocess_m))])

                new_sigma = self.sigma + (self.learning_rate  * (label - self.F) * self.w * active * (1 / (self.sigma ** 3)) * (self.euclidean(data,self.m) ** 2))
                new_theta = self.theta + (self.learning_rate  * (label - self.F))

                self.w = new_w
                self.m = new_m
                self.sigma = new_sigma
                self.theta = new_theta

            if i % 10 == 0:
                print("epoch:",i)
                print(loss_sum)
                print("sum(loss_sum) / len(loss_sum)",loss_sum / len(x_train))   
                print("------") 