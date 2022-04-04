# 作者：York
# 时间：2022/4/2 19:59
nums = [1,5,1,1,6,4]
nums.sort()
print(nums)
print(nums[:3][::-1])
nums[::2] = nums[:3][::-1]
print(nums)


" leetcode 867 "
class Solution:
    def transpose(self, matrix):
        n = len(matrix)
        m = len(matrix[0])
        ans = [[]] # 以后必须要先初始化才可以用 不然会索引越界 原因在于这生成的其实是一个一行一列的，即不是n行m列的！！！
        for i in range(m):
            for j in range(n):
                ans[i][j] = matrix[j][i]
        return ans
"例子"
a = [[0]*3]*3
b = [[0]*3 for _ in range(3)]
a[0][1] = 1
b[0][1] = 1
print(a)
print(b)
# 原因在于第一种方式创建的是第二行第三行是第一行的浅拷贝，所以其地址相同，给其中一行赋值则其他都改变。
# 第二个方式是深拷贝，产生了三个不同内存地址！