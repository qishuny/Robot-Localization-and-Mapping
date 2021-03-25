import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import expon
import pdb

from map_reader import MapReader
from sensor_model import SensorModel
from collections import defaultdict 

src_path_map = '../data/map/wean.dat'
map_obj = MapReader(src_path_map)
occupancy_map = map_obj.get_map()

_min_probability = 0.35
_max_range = 1000
bool_occ_map = (occupancy_map < _min_probability) & (occupancy_map >= 0)

walk_stride = 5


def ray_cast(laser_x_t1, laser_y_t1, laser_theta, walk_stride):
        
    x_end = laser_x_t1
    y_end = laser_y_t1
    x_idx = min(int(np.around(x_end/10)),799)
    y_idx = min(int(np.around(y_end/10)),799)

    temp_location = bool_occ_map[y_idx][x_idx]
    while x_idx >= 0 and  x_idx <= 799 and y_idx >= 0 and  y_idx <= 799 and temp_location==True:
        x_end += walk_stride * np.cos(laser_theta)
        y_end += walk_stride * np.sin(laser_theta)
        x_idx = int(np.around(x_end/10))
        y_idx = int(np.around(y_end/10))

    calc_distance = math.sqrt((laser_x_t1-x_end)**2+(laser_y_t1-y_end)**2)
    calc_distance = calc_distance-walk_stride/2
    return calc_distance 


arr = np.empty([800, 1600,360])
for i in range(800):
    x_val = 3000 +i*5
    for j in range(1600):
        y_val = j*5+0
        for k in range(360):
            theta_val = k*math.pi/180
            temp = int(ray_cast(x_val, y_val, theta_val, walk_stride))
            
            arr[i][j][k] = temp
    print("process",i/800*100)

np.save("lmap", arr) 