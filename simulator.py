import numpy as np
from sympy import Point,Segment
import math
import matplotlib.patches as patches

from RBFN import * 


class Simulator:
    def __init__(self,car_x,car_y,phi) -> None:
        self.car_x = car_x
        self.car_y = car_y
        self.phi = phi
        self.b = 3
        self.coordination = np.array([
            [-6,-3],[-6,22],[18,22],[18,50],[30,50],[30,10],[6,10],[6,-3],[-6,-3]
        ])
        
    def rotate(self,theta):
        theta = np.radians(theta)
        c, s = np.cos(theta), np.sin(theta)
        new_point_x = (c * 100) - (s * self.car_y)
        new_point_y = (s * 100) + (c * self.car_y)
        return np.array([new_point_x,new_point_y])

    def get_intersection(self,sensor):
        p1,p2 = Point(self.car_x,self.car_y), Point(sensor[0],sensor[1])        
        l1 = Segment(p1, p2)
        min_distance = float('inf')
        for i in range(1,len(self.coordination)):
            
            p3, p4 = Point(self.coordination[i - 1][0], self.coordination[i - 1][1]),\
                Point(self.coordination[i][0], self.coordination[i][1])
            
            l2 = Segment(p3, p4)
            
            # Point2D
            intersection = l1.intersection(l2)
            #如果有相交
            if intersection != []:
                intersection = intersection[0].evalf()
                distance = p1.distance(intersection)
                min_distance = min(min_distance,distance)

        return min_distance

    def update(self,theta):
        self.car_x = self.car_x + np.cos(np.radians(self.phi + theta)) + (np.sin(np.radians(theta)) * np.sin(np.radians(self.phi)))
        self.car_y = self.car_y + np.sin(np.radians(self.phi + theta)) - (np.sin(np.radians(theta)) * np.cos(np.radians(self.phi)))
        self.phi = self.phi - (np.arcsin(2 * np.sin(np.radians(theta)) / self.b) * 57.295779513)

    def start(self,window,canvas,f_plot,feature_len,rbfModel,front_distance,right_distance,left_distance):
        # [前,右,左] 
        four_dimension,six_dimension = [],[]
        while(self.car_y + 3 < 37):

            #前方感測器
            front_sensor_vec = self.rotate(self.phi)
            front_sensor_dis = self.get_intersection(front_sensor_vec)

            #右方感測器
            right_sensor_vec = self.rotate(self.phi - 45)
            right_sensor_dis = self.get_intersection(right_sensor_vec)

            #左方感測器
            left_sensor_vec = self.rotate(self.phi + 45)
            left_sensor_dis = self.get_intersection(left_sensor_vec)


            if(feature_len == 3):
                _,F = rbfModel.predict(np.array([float(front_sensor_dis),float(right_sensor_dis),float(left_sensor_dis)]))
            if(feature_len == 5):
                _,F = rbfModel.predict(np.array([float(self.car_x),float(self.car_y),float(front_sensor_dis),float(right_sensor_dis),float(left_sensor_dis)]))
            F = F * 80 - 40
            
            self.update(F)
            print(self.car_x,self.car_y,self.phi)

            #更新前右左距離
            front_distance.set(str(front_sensor_dis))
            right_distance.set(str(right_sensor_dis))
            left_distance.set(str(left_sensor_dis))
            window.update()

            if front_sensor_dis - 3 < 0 or right_sensor_dis - 3 < 0 or left_sensor_dis - 3 < 0:
                break

            #動畫畫圖
            self.plot(canvas,f_plot)

            four_dimension.append([front_sensor_dis,right_sensor_dis,left_sensor_dis,F])
            six_dimension.append([self.car_x,self.car_y,front_sensor_dis,right_sensor_dis,left_sensor_dis,F])

        return four_dimension,six_dimension

    def plot(self,canvas,f_plot):
        f_plot.scatter(self.car_x,self.car_y)
        circle = patches.Circle((self.car_x,self.car_y), radius=3,linewidth=0.5, fill=False, color="g")
        f_plot.add_patch(circle)
        canvas.draw()