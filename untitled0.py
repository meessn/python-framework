#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:50:42 2021

@author: matthijs
"""
from numpy import pi
from sympy import symbols, solve
from math import sqrt
M1=1
M2=1
K=200
CC=0.1
G=(M2*(2*pi*1j)**2+CC*(2*pi*1j)+K)/((M1*(2*pi*1j)**2+CC+K)*(M2*(2*pi*1j)**2+CC*(2*pi*1j)+K)-(CC*(2*pi*1j)+K)**2)
kp=symbols('kp')
L=abs(G)*kp*abs((2*pi*1j+1)/((1/100)*2*pi*1j+1))-1
L=abs(G)*kp*sqrt(1+(2*pi)**2)/sqrt(1+(2*pi/100)**2)-1
sol=solve(L)
print(sol)

