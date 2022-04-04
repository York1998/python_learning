# 作者：York
# 时间：2022/4/2 19:59
nums = [1,5,1,1,6,4]
nums.sort()
print(nums)
print(nums[:3][::-1])
nums[::2] = nums[:3][::-1]
print(nums)