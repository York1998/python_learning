# 作者：York
# 时间：2022/4/4 12:21
import itertools

b = [[1, 2], [5, 6], [8, 9]]
b_update = itertools.chain(*b)
print(type(b_update))
print(list(b_update))

c = [{1, 2}, {3, 4}, {5, 6}]
c_update = itertools.chain(*c)
print(type(c_update))
print(list(c_update))

d = [1, 2, 3]
d1 = [4, 5, 6]
print(d+d1)
print(list(itertools.chain(*(d,d1))))

e = [1, 2, 3 ,4 ,5]
e1 = ['a', 'b', 'c', 'd', 'e']
e_ = zip(e, e1)
final = itertools.chain(*e_)
print(final)
print(list(final))

f = "abc"
f1 = "def"
print(f+f1)
print(list(itertools.chain(*(f,f1))))