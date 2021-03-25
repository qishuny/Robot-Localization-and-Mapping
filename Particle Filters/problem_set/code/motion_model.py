'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.0005
        self._alpha2 = 0.0005
        self._alpha3 = 0.0005
        self._alpha4 = 0.0005


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """       
        xbar, ybar, thetabar = u_t0     

        xbarp, ybarp, thetabarp = u_t1

        x, y, theta= x_t0

        # calculate relative motion parameters
        # line 2-4
        delta_rot1 = math.atan2(ybarp-ybar,xbarp-xbar)-thetabar
        delta_trans = math.sqrt((ybarp-ybar)**2+(xbarp-xbar)**2)
        delta_rot2 = thetabarp-thetabar-delta_rot1
        
        # calculate true relative motion parameters
        # line 5-7
        s_rot1 = self._alpha1*(delta_rot1**2) + self._alpha2*(delta_trans**2)
        s_trans = self._alpha3*(delta_trans**2) + self._alpha4*(delta_rot1**2) + self._alpha4*(delta_rot2**2)
        s_rot2 = self._alpha1*(delta_rot2**2) + self._alpha2*(delta_trans**2)

        delta_rot1hat = delta_rot1- np.random.normal(0,math.sqrt(s_rot1))
        delta_transhat = delta_trans- np.random.normal(0,math.sqrt(s_trans))
        delta_rot2hat = delta_rot2 - np.random.normal(0,math.sqrt(s_rot2))

        # calculate true position
        # line 8-10
        
        xp = x + delta_transhat*math.cos(theta+delta_rot1hat)
        yp = y + delta_transhat*math.sin(theta+delta_rot1hat)
        thetap = theta + delta_rot1hat + delta_rot2hat

        return [xp, yp, thetap]
