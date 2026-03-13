import random
def quicksortHelper(nums, left, right):
    # [left, right]
    if left > right:
        return 
    pati = random.randint(left, right)
    baseline = nums[pati]
    nums[left], nums[pati] = nums[pati], nums[left]

    l, r = left+1, right

    while l <= r:
        while l <= r and nums[l] < baseline:
            l += 1
        while l <= r and nums[r] > baseline:
            r -= 1
        
        # 只有一个元素or两个元素
        if l >= r:
            break

        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1
    nums[left], nums[r] = nums[r], nums[left]
    
    quicksortHelper(nums, left, r-1)
    quicksortHelper(nums, r+1, right)


def quicksort(nums):
    quicksortHelper(nums, 0, len(nums)-1)


lis = [i for i in range(100, 0, -1)]
quicksort(lis)
print(lis)