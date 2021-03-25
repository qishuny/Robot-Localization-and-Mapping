'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
# from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 5
        self._z_short = 0.15
        self._z_max = 1
        self._z_rand = 1000

        z_sum = self._z_hit + self._z_short + self._z_max + self._z_rand 
        self._z_hit = self._z_hit / z_sum
        self._z_short = self._z_short / z_sum
        self._z_max = self._z_max / z_sum
        self._z_rand = self._z_rand / z_sum

        self._sigma_hit = 50
        self._lambda_short = 0.1

        self._max_range = 1000
        self._min_probability = 0.35
        self.bool_occ_map = (occupancy_map < self._min_probability) & (occupancy_map >= 0)
        

    def ray_cast(self, laser_x_t1, laser_y_t1, laser_theta, walk_stride):
        
        x_end = laser_x_t1
        y_end = laser_y_t1
        x_idx = int(np.round(x_end/10))
        y_idx = int(np.round(y_end/10))

        temp_location = self.bool_occ_map[y_idx][x_idx]
        while x_idx >= 0 and  x_idx <= 799 and y_idx >= 0 and  y_idx <= 799 and temp_location==True:
            temp_location = self.bool_occ_map[y_idx][x_idx]
            x_end += walk_stride * np.cos(laser_theta)
            y_end += walk_stride * np.sin(laser_theta)
            x_idx = int(np.around(x_end/10))
            y_idx = int(np.around(y_end/10))

        calc_distance = math.sqrt((laser_x_t1-x_end)**2+(laser_y_t1-y_end)**2)
        return calc_distance 

    def calc_prob(self, ray_cast_distance, measurement_distance):
        prob_sum = 0
        
        if measurement_distance >= 0 :

        #p_hit
            if measurement_distance >= 0 and measurement_distance <= self._max_range:
                coef_norm = 1/ (self._sigma_hit * math.sqrt(2*math.pi)) 
                exp_term = math.exp((-1/2)* ((ray_cast_distance - measurement_distance)**2/self._sigma_hit**2))
                p_hit = coef_norm * exp_term
            else:
                p_hit = 0
            prob_sum +=  self._z_hit * p_hit

        #p_short
        
            if measurement_distance >= 0 and measurement_distance <= ray_cast_distance:
                normalizer = 1/(1-math.exp(-self._lambda_short*ray_cast_distance))
                p_short = normalizer*self._lambda_short*math.exp(-self._lambda_short*measurement_distance)
            else:
                p_short = 0

            prob_sum += self._z_short * p_short

        #p_max

            if measurement_distance <= self._max_range and measurement_distance >= self._max_range - 5: #NEED TO TUNE THE WIDTH OF THIS
                p_max = 1.0
            else:
                p_max = 0.0

            prob_sum += self._z_max * p_max

        #p_rand

            if measurement_distance < self._max_range:
                p_rand = 1/(self._max_range)
            else:
                p_rand = 0.0

            prob_sum += self._z_rand * p_rand
        # if prob_sum == 0:
        #     print("PROB_SUM = 0")
        #     print("ray_cast_distance: ", ray_cast_distance)
        #     print("measurement_distance: ", measurement_distance)
        
        return prob_sum
    
    
    def beam_range_finder_model(self, z_t1_arr, x_t1,table):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        walk_stride = 8
        angle_stride = 10
        
        # Assume that we have distance metric for every k from ray casting
        x,y,theta = x_t1

        laser_x_t1 = x + 25 * math.cos(theta)
        laser_y_t1 = y + 25 * math.sin(theta)
        # print("laser_x_t1:",laser_x_t1)
        prob_zt1 = 1.0
        #CAN CHANGE ANGLE_STRIDE!!!
        for deg in range (-90, 90, angle_stride):
            
            
            laser_theta = (deg + theta * 180 / math.pi ) * (math.pi/180)
            measurement_distance = z_t1_arr[deg+90]

            # ray_cast_distance = self.ray_cast_read(laser_x_t1, laser_y_t1, laser_theta, table)
            ray_cast_distance= self.ray_cast(laser_x_t1, laser_y_t1, laser_theta, walk_stride)
            particle_prob = self.calc_prob(ray_cast_distance, measurement_distance)
            prob_zt1 *= particle_prob

        robot_x_index = math.floor(x / 10)
        robot_y_index = math.floor(y / 10)

        if self.bool_occ_map[robot_y_index][robot_x_index]==False:
            prob_zt1 = 0
            
        return prob_zt1

    def ray_cast_read(self,laser_x_t1, laser_y_t1, laser_theta, table):

        output = 1
        if(laser_x_t1 < 3000 or laser_x_t1 >= 7000 or laser_y_t1<0 or laser_y_t1 >= 8000):
            return output
            
        xlow = int(laser_x_t1/5)*5
        xhigh =int(xlow if xlow == 6995 else xlow + 5) 
        ylow = int(laser_y_t1/5)*5
        yhigh = int(ylow if xlow == 7995 else ylow + 5)
        laser_theta = laser_theta*180/math.pi if laser_theta>=0 else laser_theta*180/math.pi+180*2

        thetalow = min(int(laser_theta/5)*5,355)
        thetahigh = int(thetalow if thetalow == 355 else thetalow + 5)

        xd = (laser_x_t1-xlow)/(5)
        yd = (laser_y_t1-ylow)/(5)
        zd = (laser_theta-thetalow)/(5)

        xlow_idx = int((xlow-3000)/5)
        xhigh_idx = int((xhigh-3000)/5) 
        ylow_idx = int(ylow/5)
        yhigh_idx = int(yhigh/5) 
        thetalow_idx = int(thetalow/5)
        thetahigh_idx = int(thetahigh/5)

        c000 = table[xlow_idx][ylow_idx][thetalow_idx]
        c100 = table[xhigh_idx][ylow_idx][thetalow_idx]
        c010 = table[xlow_idx][yhigh_idx][thetalow_idx]
        c110 = table[xhigh_idx][yhigh_idx][thetalow_idx]
        c001 = table[xlow_idx][ylow_idx][thetahigh_idx]
        c101 = table[xhigh_idx][ylow_idx][thetahigh_idx]
        c011 = table[xlow_idx][yhigh_idx][thetahigh_idx]
        c111 = table[xhigh_idx][yhigh_idx][thetahigh_idx]

        c00 = c000*(1-xd) + c100*xd
        c01 = c001*(1-xd) + c101*xd
        c10 = c010*(1-xd) + c110*xd
        c11 = c011*(1-xd) + c111*xd

        c0 = c00*(1-yd) + c10*yd
        c1 = c01*(1-yd) + c11*yd

        output = c0*(1-zd)+ c1*zd
        return output
        

    def ray_cast_vec(self, particles, stride):
        particles = particles.copy()
        x_vals = particles[:,0]
        y_vals = particles[:,1]
        cos_thetas = np.cos(particles[:,2])
        sin_thetas = np.sin(particles[:,2])
        
        x_indeces = np.round(x_vals / 10).astype(int)
        y_indeces = np.round(y_vals / 10).astype(int)
        distance_acc = np.zeros(len(particles))
        
        hit_wall_or_bound = ~(self.bool_occ_map[y_indeces,x_indeces].astype(bool))| (x_vals < 0) | (x_vals > 7999) | (y_vals < 0) | (y_vals > 7999)
        while hit_wall_or_bound.sum() != len(particles):
            hit_wall_or_bound = ~(self.bool_occ_map[y_indeces,x_indeces]) | (x_vals < 0) | (x_vals > 7999) | (y_vals < 0) | (y_vals > 7999)
            x_vals = np.where(hit_wall_or_bound, x_vals, x_vals + stride * cos_thetas)
            y_vals = np.where(hit_wall_or_bound, y_vals, y_vals + stride * sin_thetas)

            desired_x_indeces = np.round(x_vals / 10).astype(int)
            desired_y_indeces = np.round(y_vals / 10).astype(int)

            x_indeces = np.where((desired_x_indeces >= 0) & (desired_x_indeces <= 799), desired_x_indeces, x_indeces) 
            y_indeces = np.where((desired_y_indeces >= 0) & (desired_y_indeces <= 799), desired_y_indeces, y_indeces)
            
            distance_acc = np.where(hit_wall_or_bound == True, distance_acc, distance_acc + stride)
        distance_acc = np.where(distance_acc > 0, distance_acc + stride, distance_acc)
        return distance_acc

    def calc_prob_vec(self, ray_cast_array, measurement_distance_array):
        n = len(ray_cast_array)
        prob_sum = np.zeros(n)

        coef_norm = 1/ (self._sigma_hit * math.sqrt(2*math.pi)) 
        
        # normal 
        exp_term = np.exp((-1/2)* ((ray_cast_array - measurement_distance_array)**2/self._sigma_hit**2))
        p_hit = exp_term * coef_norm
        prob_sum = prob_sum + p_hit * self._z_hit
        
        # print(p_hit * _z_hit)
        
        # exp
        
        normalizer = 1/(1-np.exp(-self._lambda_short*ray_cast_array))
        p_short = normalizer*self._lambda_short*np.exp(-self._lambda_short*measurement_distance_array)
        p_short = np.where((measurement_distance_array >= 0) & (measurement_distance_array <= ray_cast_array), p_short, 0)
        
        prob_sum = prob_sum + p_short * self._z_short
        
        
        # print(p_short * _z_short)
        # max

        p_max = np.where((measurement_distance_array <= self._max_range) & (measurement_distance_array >= self._max_range - 5), 1, 0)
        
        prob_sum = prob_sum + p_max * self._z_max

        # print(p_max * _z_max)
        # rand
        
        p_rand = np.where(measurement_distance_array < self._max_range, 1/self._max_range, 0)
        prob_sum = prob_sum + p_rand * self._z_rand
        
        # print(p_rand * _z_rand)
        
        prob_sum = np.where((measurement_distance_array <= self._max_range) & (measurement_distance_array >= 0), prob_sum, 0)
        
        return prob_sum

    def beam_range_finder_model_vec(self, z_t1_arr, x_t1,table):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        walk_stride = 8
        angle_stride = 10
        
        # Assume that we have distance metric for every k from ray casting
        x,y,theta = x_t1

        laser_x_t1 = x + 25 * math.cos(theta)
        laser_y_t1 = y + 25 * math.sin(theta)
        # print("laser_x_t1:",laser_x_t1)
        # prob_zt1 = 1.0
        #CAN CHANGE ANGLE_STRIDE!!!

        all_particles = np.array([[laser_x_t1, laser_y_t1,theta]] * int(180/angle_stride))

        all_particles[:,2] = all_particles[:,2] * 180 / math.pi

        all_particles[:,2] += np.arange(-90,90,angle_stride)

        all_particles[:,2] = all_particles[:,2] * (math.pi/180)

        measurement_distances = np.array(z_t1_arr)[::angle_stride]        

        ray_cast_distances = self.ray_cast_vec(all_particles, walk_stride)

        final_probs = self.calc_prob_vec(ray_cast_distances, measurement_distances)

        return np.prod(final_probs)