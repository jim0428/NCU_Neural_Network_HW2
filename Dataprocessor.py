import numpy as np

class Dataprocessor:
    def __init__(self):
        self.x_train = []
        self.y_train = []

    def splitFile(self,dataset_url):
        with open(dataset_url,'r',encoding='utf-8') as f:
            for data in f.readlines():
                data = data.split(' ')
                if(len(data) == 4):
                    self.x_train.append([float(data[0]),float(data[1]),float(data[2])])
                    self.y_train.append(float(data[3]))
                else:
                    self.x_train.append([float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4])])
                    self.y_train.append(float(data[5]))

            self.x_train = np.array(self.x_train)
            self.y_train = np.array(self.y_train)
            self.y_train = (self.y_train - np.amin(self.y_train)) / (np.amax(self.y_train) - np.amin(self.y_train))    
        f.close()
        
        return self.x_train,self.y_train

    def readfile(url):
        read = open(url)
        file = read.readlines()
        read.close()
        return file

    def text_to_numlist(dataset):
        """load text dataset to numeracial list dataset

        Args:
            dataset (string): txt or other file

        Returns:
            dataset: float_list
        """
        dataset = [list(map(float,data)) for data in dataset]
        return dataset