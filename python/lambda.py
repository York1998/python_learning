
import os

with open("1.txt",'r')as f:
    dists = f.read()

print(type(dists))
_dis = eval(dists) # 将str 转换成dict
print(_dis)
print(type(_dis))