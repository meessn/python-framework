#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:12:12 2021

@author: matthijs
"""
import re

fd=open("dronesxy.txt", "r")
x=[]
y=[]
for line in fd.readlines():
    r=re.split("[^0-9.e-]+", line)
    x.append(float(r[0]))
    y.append(float(r[1]))
fd.close()
print(x)
print(y)