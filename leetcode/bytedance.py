# 求数组内的方差

def std_max(nums):
    return




# 扔骰子问题:


def touzi(n, k):

    dfs = [0] * (k+1)

    dfs[0] = 1

    for i in range(n):
        temp_dfs = [0] * (k+1)
        for j in range(1, k+1):
            for kk in range(1, 7):
                if j - kk >= 0:
                    temp_dfs[j] += dfs[j-kk]
        dfs = temp_dfs
        print(dfs)
        
    return dfs[k]


a = touzi(3, 5)
print(a)



class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __str__(self,):
        return f"Node{self.val}"

root = TreeNode(5, TreeNode(3), TreeNode(3, TreeNode(1), None))

def midProcess(root):
    
    stack = []
    stack.append([root, 0])
    result = []
    while stack:
        popRoot, ifWrite = stack.pop(-1)
        
        if popRoot and ifWrite == 1:
            result.append(popRoot.val)
            continue
        if popRoot:
            stack.append([popRoot.right, 0])
            stack.append([popRoot, 1])
            stack.append([popRoot.left, 0])
    
    return result

p = TreeNode(100, TreeNode(6), TreeNode(2, TreeNode(7), TreeNode(4)))
q = TreeNode(1, TreeNode(0), TreeNode(8))
root = TreeNode(100, p, q)

def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root

    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    if left and right:
        return root
    else:
        return left if left else right

print(lowestCommonAncestor(root, p, q))



def findUniqueCount(nums):
    n = len(nums)
    left = 0
    right = n-1
    result = 0
    while left <= right:
        if nums[left] + nums[right] == 0:
            right -= 1
            left += 1
            while left < right and nums[left] == nums[left-1]:
                left += 1
            while left < right and nums[right] == nums[right+1]:
                right -= 1
        elif nums[left] + nums[right] > 0:
            right -= 1
            while left < right and nums[right] == nums[right+1]:
                right -= 1
        else:
            
            left += 1
            while left < right and nums[left] == nums[left-1]:
                left += 1
        
        result += 1

    return result
    

print(findUniqueCount([-7, -5, -3, -1, -1, -1, 0, 1, 1, 3, 5, 7, 8]))
print(findUniqueCount([2, 2, 2, 2]))
        
# word = 'abchello' -> [3, 2, 4]
vocab_map = {'a':0, 'b':1, 'c':2, 'ab':3, 'hello':4}
def tokenazation(word):
    tokens = []
    n = len(word)
    left = 0
    right = 1
    while left < n:
        isMatch = False

        for right in range(n, left, -1):
            if word[left:right] in vocab_map:
                tokens.append(vocab_map[word[left:right]])
                left = right
                isMatch = True
                break
        if not isMatch:
            left += 1
    return tokens

    
c = tokenazation('abchelloabbac')
print(c)
        


def cube(x):
    x0 = 1.0
    eposilon = 1e-5
    while (x0**3 - x) ** 2 > eposilon:
        x0 -= (x0**3 - x) / (3*x0**2)
    return x0
print(cube(8))
print(cube(64))
print(cube(27))


def findMaxSubSeq(s):
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = True
    max_length = 0
    res_left = 0
    res_right = 0
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            
            if s[i] == s[j]:
                if j-i < 2:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i+1][j-1]
            
                
            if dp[i][j] and j-i > max_length:
                max_length = dp[i][j]
                res_left = i
                res_right = j
    return s[res_left:res_right+1]

re = findMaxSubSeq('babad')
re2 = findMaxSubSeq('cbbd')

print(re)
print(re2)

