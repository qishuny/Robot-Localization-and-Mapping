import numpy as np
import sys, os
import math

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling
from readtable import ReadTable

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time

src_path_map = '../data/map/wean.dat'
src_path_log = '../data/log/robotdata1.log'
map_obj = MapReader(src_path_map)
occupancy_map = map_obj.get_map()
bool_occ_map = (occupancy_map < 0.35) & (occupancy_map >= 0)
print(bool_occ_map[500][400])
logfile = open(src_path_log, 'r')

motion_model = MotionModel()
sensor_model = SensorModel(occupancy_map)
resampler = Resampling()


def visualize_ray_cast(x_t1,bool_occ_map,tstep, output_path):
    walk_stride = 8
    angle_stride = 10
    x,y,theta = x_t1
    laser_x_t1 = x + 25 * math.cos(theta)
    laser_y_t1 = y + 25 * math.sin(theta)
    xout = laser_x_t1/10
    yout = laser_y_t1/10
    for deg in range (-90, 90, angle_stride):
        laser_theta = (deg + theta * 180 / math.pi ) * (math.pi/180)
        calc_distance, x_end, y_end =ray_cast_visual(laser_x_t1, laser_y_t1, laser_theta, walk_stride)
        
        
        xout = np.vstack((xout,x_end/10.0))
        yout = np.vstack((yout,y_end/10.0))
    
    ray = plt.scatter(xout, yout, c='b', marker='.')
    
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    ray.remove()



def ray_cast_visual(laser_x_t1, laser_y_t1, laser_theta, walk_stride):
        
        # calc_distance = 0
        # x_index = math.floor(laser_x_t1 / 10)
        # y_index = math.floor(laser_y_t1 / 10)

        # if self.bool_occ_map[y_index][x_index]==False:
        #     return 1

        # while x_index >= 0 and  x_index <= 799 and y_index >= 0 and  y_index <= 799:
        #     laser_x_t1 = laser_x_t1 + walk_stride * math.cos(laser_theta)
        #     laser_y_t1 = laser_y_t1 + walk_stride * math.sin(laser_theta)

        #     x_index = math.floor(laser_x_t1 / 10)
        #     y_index = math.floor(laser_y_t1 / 10)

        #     if self.bool_occ_map[y_index][x_index]==False:
        #         calc_distance += walk_stride / 2
        #         break
        #     else:
        #         calc_distance += walk_stride

        # if calc_distance > self._max_range:
        #     calc_distance = self._max_range # this is when the wall is further than max_range distance away
    x_end = laser_x_t1
    y_end = laser_y_t1
    x_idx = int(np.around(x_end/10))
    y_idx = int(np.around(y_end/10))

    temp_location = bool_occ_map[y_idx][x_idx]
    while x_idx >= 0 and  x_idx <= 799 and y_idx >= 0 and  y_idx <= 799 and temp_location==True:
        temp_location = bool_occ_map[y_idx][x_idx]
        x_end += walk_stride * np.cos(laser_theta)
        y_end += walk_stride * np.sin(laser_theta)
        x_idx = int(np.around(x_end/10))
        y_idx = int(np.around(y_end/10))

    calc_distance = math.sqrt((laser_x_t1-x_end)**2+(laser_y_t1-y_end)**2)
    print(calc_distance)
    return calc_distance, x_end, y_end

fig = plt.figure()
mng = plt.get_current_fig_manager()
plt.ion()
plt.imshow(occupancy_map, cmap='Greys')
plt.axis([0, 800, 0, 800])
plt.plot(600,150,'ro')
visualize_ray_cast([6000,1500,math.pi/2],bool_occ_map,0,'results')