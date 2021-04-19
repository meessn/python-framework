#!/usr/bin/env python3
from time import time
a=0
t1=time()
for i in range(10):
    a+=1
t2=time()
print(t2-t1)