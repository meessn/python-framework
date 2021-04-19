#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:50:42 2021

@author: matthijs
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy.random as rd
import re
   
def data_gen():
    gen_list=([x[12*i],y[12*i],x[12*i+1],y[12*i+1],x[12*i+2],y[12*i+2],x[12*i+3],y[12*i+3],x[12*i+4],y[12*i+4],x[12*i+5],y[12*i+5],x[12*i+6],y[12*i+6],x[12*i+7],y[12*i+7],x[12*i+8],y[12*i+8],x[12*i+9],y[12*i+9],x[12*i+10],y[12*i+10],x[12*i+11],y[12*i+11]] for i in range(1,200))
    return gen_list

def init():
    ax.set_ylim(-10, 10)
    ax.set_xlim(-10, 10)
    return point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, point11, point12

def run(data):

    x_1, y_1,x_2,y_2,x_3, y_3,x_4,y_4,x_5, y_5,x_6,y_6,x_7, y_7,x_8,y_8,x_9, y_9,x_10,y_10,x_11,y_11,x_12,y_12 = data
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()         
    point1.set_data(x_1, y_1)
    point2.set_data(x_2, y_2)
    point3.set_data(x_3, y_3)
    point4.set_data(x_4, y_4)
    point5.set_data(x_5, y_5)
    point6.set_data(x_6, y_6)
    point7.set_data(x_7, y_7)
    point8.set_data(x_8, y_8)
    point9.set_data(x_9, y_9)
    point10.set_data(x_10, y_10)
    point11.set_data(x_11, y_11)
    point12.set_data(x_12, y_12)
    return point1, point2, point3, point4, point5, point6, point7, point8, point9, point10, point11, point12,

    

if __name__=="__main__":
    fig, ax = plt.subplots()
    
    fd=open("dronesxy.txt", "r")
    x=[]
    y=[]
    for line in fd.readlines():
        r=re.split("[^0-9.e-]+", line)
        x.append(float(r[0]))
        y.append(float(r[1]))
    fd.close()
    for i in range(len(x)):
        if i%12==0:
            print(x[i])
    point1, = ax.plot([0], [0], 'go')
    point2, = ax.plot([0], [0], 'go')
    point3, = ax.plot([0], [0], 'go')
    point4, = ax.plot([0], [0], 'go')
    point5, = ax.plot([0], [0], 'go')
    point6, = ax.plot([0], [0], 'go')
    point7, = ax.plot([0], [0], 'go')
    point8, = ax.plot([0], [0], 'go')
    point9, = ax.plot([0], [0], 'go')
    point10, = ax.plot([0], [0], 'go')
    point11, = ax.plot([0], [0], 'go')
    point12, = ax.plot([0], [0], 'go')
    point1.set_data(x[0], y[0])
    point2.set_data(x[1], y[1])
    point3.set_data(x[2], y[2])
    point4.set_data(x[3], y[3])
    point5.set_data(x[4], y[4])
    point6.set_data(x[5], y[5])
    point7.set_data(x[6], y[6])
    point8.set_data(x[7], y[7])
    point9.set_data(x[8], y[8])
    point10.set_data(x[9], y[9])
    point11.set_data(x[10], y[10])
    point12.set_data(x[11], y[11])
    ax.grid()
    
    
     
    ani = animation.FuncAnimation(fig, run, data_gen, init_func=init,interval=10)