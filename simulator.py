import numpy as np

class simulator:
    def __init__(self,car_x,car_y,phi) -> None:
        self.car_x = car_x
        self.car_y = car_y
        self.phi = phi

    def create_rotation_matrix(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array()
        