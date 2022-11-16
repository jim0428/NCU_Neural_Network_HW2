import numpy as np
from sympy import Point,Segment
import math

from RBFN import * 

class simulator:
    def __init__(self,car_x,car_y,phi) -> None:
        self.car_x = car_x
        self.car_y = car_y
        self.phi = phi
        self.b = 6
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
        min_point,min_distance = Point(0,0),float('inf')
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

    def check_collision(self,y_train,rbfModel):
        # [前,右,左] 
        current_pos =[]
        while(self.car_y + 3 < 37):
            #print(the,theta)
            #前方感測器
            front_sensor_vec = self.rotate(self.phi)
            front_sensor_dis = self.get_intersection(front_sensor_vec)
            #print("front:",front_sensor_point)
            #右方感測器
            right_sensor_vec = self.rotate(self.phi - 45)
            right_sensor_dis = self.get_intersection(right_sensor_vec)
            #print("right",right_sensor_point)
            #左方感測器
            left_sensor_vec = self.rotate(self.phi + 45)
            left_sensor_dis = self.get_intersection(left_sensor_vec)
            #print(np.array([float(front_sensor_dis),float(right_sensor_dis),float(left_sensor_dis)]))

            _,F = rbfModel.predict(np.array([float(front_sensor_dis),float(right_sensor_dis),float(left_sensor_dis)]))
            F = F * 80 - 40
            #print("left",left_sensor_point)
            self.update(F)
            print(self.car_x,self.car_y,self.phi)
            current_pos.append([self.car_x - 1,self.car_y - 1])
            #print('----------------------------------------')
        return current_pos