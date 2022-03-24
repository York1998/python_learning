# 作者：York
# 时间：2022/3/24 15:33
"""
    对于多元输入
"""
# n,k,m = map(int,input().split())
# print(type(n),type(k),type(m)) # 全是int

# 方法一
# line = list(map(str,input().split())) # 输入的1 2 3 全部转化为列表
# print(line)

# 方法二
# L = []
# L.append(map(str,input().split()))
# print(L,type(L))

# 方法三
# arr = input("") #将每行输入读取成矩阵的形式 输入一个一维数组
# num = [int(n) for n in arr.split()]
# print(num,type(num))
"""
    处理多行输入
"""
# 读入二维矩阵，适用于 n * n矩阵
def n_n():
    n = int(input())
    m = [[0]*n]*n #初始化二维数组
    for i in range(n):
        m[i] = input().split(" ")
    print(m)

def n_m():
    n = int(input())
    m = []
    for i in range(n):
        m.append(list(map(int, input().split(" "))))
    print(m)

def n_m2():
    print("请输入数据的行数N：")
    n = int(input())
    print("n=",n)
    print("input 输入:")
    list1 = [[int(x) for x in input().split()]for y in range(n)]
    print(list1)

"""
    读取多行输入，不知道多少行，但肯定是以换行符结束，输出是一维列表形式
"""
import sys
def k_m():
    try:
        mx = []
        while True:
            m = sys.stdin.readline().strip()
            if m =='':
                break
            m = list(m.split())
            mx.append(m)
        print(mx)
        print(len(mx)) # 读取列
        print(len(mx[0])) # 读取行
    except:
        pass
def k_m1():
    try:
        mx = []
        while True:
            m = input().strip()
            if m == '':
                break
            m = list(m.split())
            mx.append(m)
        print(mx)
        print(len(mx)) # 读取列
        print(len(mx[0])) # 读取行

    except:
        pass
# k_m1()

"""
    不指定行数 但是每输入一行就处理一行 持续等待输入
"""
def dynamic_n_m():
    try:
        ssn = []
        while True:
            sn = input().strip()

            if sn == '':
                break
            sn = list(sn.split())
            ssn.append(sn)
        print(ssn)
    except: pass

dynamic_n_m()
