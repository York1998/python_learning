
import os

with open("1.txt",'r')as f:
    dists = f.read()

print(type(dists))
_dis = eval(dists)
print(_dis)
print(type(_dis))