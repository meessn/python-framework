#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:07:03 2021

@author: mees
"""

#kunnen jullie dit zien? Groetjes Olivier

import numpy as np



def crazyflie_dynamics(x, u):
    # Bebop 2 dynamics model
    #
    # state: x, y, z, vx, vy, vz, phi, theta, psi
    # control: phi_c, theta_c, vz_c, psi_rate_c
    #
    # f:
    #    \dot(x) == vx
    #   \dot(y) == vy
    #    \dot(z) == vz
    #    \dot(vx) == (cos(psi) * tan(theta) / cos(phi) + sin(psi) * tan(phi)) * g - kD_x * vx
    #    \dot(vy) == (sin(psi) * tan(theta) / cos(phi) - cos(psi) * tan(phi)) * g - kD_y * vy
    #    \dot(vz) == (k_vz * vz_c - vz) / tau_vz
    #    \dot(phi) == (k_phi * phi_c - phi) / tau_phi
    #    \dot(theta) == (k_theta * theta_c - theta) / tau_theta
    #    \dot(psi) == psi_rate_c
    #
    # g = 9.81
    # kD_x = 0.25
    # kD_y = 0.33
    # k_vz = 1.2270
    # tau_vz = 0.3367
    # k_phi = 1.1260
    # tau_phi = 0.2368
    # k_theta = 1.1075
    # tau_theta = 0.2318
    #
    # (c)Hai Zhu, TU Delft, 2019, h.zhu @ tudelft.nl
    #

    ## Model parameters
    g = 9.81
    m = 0.0335
    #kD_x = 0.25
    #kD_y = 0.33
    #k_vz = 1.2270
    #tau_vz = 0.3367
    k_phi = 1.1177
    tau_phi = 0.1766
    k_theta = 1.1177
    tau_theta = 0.1766

    ## control inputs
    pwm_c = u[2,0]  #new input #change mpc_generator_basic  (line 65) (MATLAB) , limits+constraints,  also modify the model that it has an effect on. 
    # line 78 and be careful where this parameter is used. 
    phi_c = u[0,0]
    theta_c = u[1,0]
    psi_rate_c = u[3,0] #is this omega_zc? 
#implement it in MATLAB as wel! watch out for ranges and inputs (start at 0 and 1) etc. , because the solver will use the matlab files.  generator_main mpc_generator_basic
    ## position dynamics
    vx = x[3,0]
    vy = x[4,0]
    vz = x[5,0]

    phi = x[6,0]
    theta = x[7,0]
    psi = x[8,0] #this was not in the bebop!!!!!!!!
    #psi = x[8] # in this way, should be more accurate?
    
    T_c = 4 * (2.130295*10**(-11)*pwm_c**2 + 1.032633*10**(-6)*pwm_c + 5.484560*10**(-4))
    k_dxy = 4*9.1785*10**(-7)*(0.04076521*pwm_c + 380.8359)
    k_dz = 4*10.311*10**(-7)*(0.04076521*pwm_c + 380.8359)
    #ax = (cos(psi)*tan(theta)/cos(phi) + sin(psi)*tan(phi))*g - kD_x*vx;
    #ay = (sin(psi)*tan(theta)/cos(phi) - cos(psi)*tan(phi))*g - kD_y*vy;
#small angles assumption?
    # in this way, might be computationally easier
    #ax = np.tan(theta) * g - kD_x * vx
    #ay = -np.tan(phi) * g - kD_y * vy

    #az = (k_vz * vz_c - vz) / tau_vz
    ax = (T_c - k_dxy* vx) /m * (np.cos(psi)*np.sin(theta)+ np.sin(psi)*np.sin(phi)*np.cos(theta))
    ay = (T_c - k_dxy* vy) /m * (np.sin(psi)*np.sin(theta)- np.cos(psi)*np.sin(phi)*np.cos(theta))
    az = -g + ((T_c-k_dz*vz )/m)*np.cos(phi)*np.cos(theta)
    #try both and see computational time difference (compare them) loss in precision and gain in iteration time. 
    #simplified, using small angles assumption
    #ax = (T_c - K_dxy* vx) /m  ....
    #ay = (T_c - K_dxy* vy) /m *....
    #az = -g ....
    
    ## attitude dynamics
    dphi = (k_phi * phi_c - phi) / tau_phi
    dtheta = (k_theta * theta_c - theta) / tau_theta
    dpsi = psi_rate_c

    ## output
    dx = np.array([[vx,vy,vz,ax,ay,az,dphi,dtheta,dpsi]]).T
    return dx
