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
                max_length = j-i
                res_left = i
                res_right = j
    return s[res_left:res_right+1]

re = findMaxSubSeq('babad')
re2 = findMaxSubSeq('cbbd')

print(re)
print(re2)


def binarySearch(nums, target):
    n = len(nums)
    left, right = 0, n-1
    # [left, right]
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return -1

print(binarySearch([1, 2, 3, 4, 5, 6, 7], 5))

def isInterleave(s1: str, s2: str, s3: str) -> bool:
        m = len(s1)
        n = len(s2)
        if m+n != len(s3):
            return False
        dp = [[False] * (n+1) for _ in range(m+1)]
        dp[0][0] = True
        for i in range(1, m+1):
            if s1[i-1] == s3[i-1] and dp[i-1][0]:
                dp[i][0] = True
        for j in range(1, n+1):
            if s2[j-1] == s3[j-1] and dp[0][j-1]:
                dp[0][j] = True
        
        for i in range(m):
            for j in range(n):
                dp[i+1][j+1] = (s1[i] == s3[i+j+1] and dp[i][j+1]) or (s2[j] == s3[i+j+1] and dp[i+1][j])
        print("dp:")
        print(dp)
        return dp[-1][-1]
               
print(isInterleave('a', 'abaa', 'abaaa'))
print(isInterleave('a', 'b', 'ab'))
print(isInterleave('a', 'b', 'ba'))

def intToRoman(num: int) -> str:
        transMap = {1:'I', 4:'IV', 5:'V', 9:'IX', 10:'X', 40:'XL', 50:'L', 90:'XC', 100:'C', 400:'CD', 500:'D', 900:'CM', 1000:'M'}
        transList = list(transMap.keys())
        transList.reverse()
        res = ''
        print(transList)
        print(transMap)
        while num > 0:
            for x in transList:
                if num < x:
                    continue
                while num >= x:
                    res += transMap[x]
                    num -= x
                print(x, num)
                break
        return res
            
print(intToRoman(3749))


# hllm:


# ' xxxx. '

# sentence [B, seq]

# tokens [B, seq, 1], padding_mask [B, seq, 1]

# tokenzation.encoder(sentence)

# class ItemLLM:

#     def init(config:LlamaConfig):

#     self.model = model(config)


# def forward(tokens, padding_mask):

# # input [B, seq, 1]

# # hidden state [B, seq, dim]

#     hidden_state = self.model(tokens, padding_mask)

#     if parmas['pooling'] == 'avg':

#     # [B, dim]

#     avg_dim = torch.gather(hidden_state, padding_mask).mean(dim=-2)

#     elif parmas['pooling'] == 'last':

#     last_dim = torch.gather(hidden_state, padding_mask.sum(dim=-2, keepdim=True))

# # frozen : ItemLLM Inference -> redis

# # restrivel

# class UserLLM:

#     def init(self, config:LlamaConfig):

#         self.model = model(config)

# def infonceloss(self, X1, X2):
#     # X1 [B, dim]
#     norm1 = normlize(X1, dim=-1)
#     norm2 = normlize(X2, dim=-1)
#     sim = norm1 @ norm2.T
#     label = arange(0, batchsize)

#     return (cross_entropy(sim, label) + cross_entropy(sim.T, label)) / 2


# def forward(tokens, padding_mask):

#     # tokens [B, user_seq, dim]

#     embedding = None

#     # hidden state [B, user_seq, dim]

#     hidden_state = self.model(tokens, padding_mask)



# losses += self.infonceloss(hidden_state[:, 0, :], tokens[:, -1, :])



# return losses

# def infer(tokens, padding_mask):

# hidden_state = self.model(tokens, padding_mask)


# last_dim = torch.gather(hidden_state, padding_mask.sum(dim=-2, keepdim=True))

# return last_dim



# # ANN : userllm last hidden state as embedding

# # update: daily slow

# # update: second OOM

# # queue Kalfa：

# # user : last seq -> 10 items -> infer this user embedding task -> (UserLLM) -> update ANN

# # update: user embedding -> ANN restrivel



# # rank:

# class UserLLM:

#     def init(config:LlamaConfig):

#         self.model = model(config)

#         self.adapter1 = nn.Linear(dim, hidden_dim)

#         self.adapter2 = nn.Linear(hidden_dim, 1)

#         self.active = nn.sigmoid()

#     def forward(tokens, padding_mask):

#         # tokens [B, user_seq, dim]

#         embedding = None

#         # hidden state [B, seq, dim]

#         hidden_state = self.model(tokens, padding_mask)

#         if parmas['pooling'] == 'avg':

#         # [B, dim]
#          embedding = torch.gather(hidden_state, padding_mask).mean(dim=-2)
#         elif parmas['pooling'] == 'last':
#             embedding = torch.gather(hidden_state, padding_mask.sum(dim=-2, keepdim=True))
#         return self.avtive(self.adapter2(self.adapter1(embedding)))



#     itemlm = ItemLLM()



#     userllm = UserLLM()

x = [1, 2, 3, 4, 5]
for i in x:
    if i == 3:
        x.remove(i)
print(x)



def meetings(intervels:list):
    intervels.sort(key=lambda x : x[0])
    n = len(intervels)
    for i in range(1, n):
        if intervels[i][0] < intervels[i-1][1]:
            return False
    return True
print(meetings([[0,30],[5,10],[15,20]]))
print(meetings([[7,10],[2,4]]))

def meeting2(intervels:list):
    n = len(intervels)
    intervels.sort(key=lambda x : x[0])
    result = [intervels[0].copy()]
    for i in range(1, n):
        if intervels[i][0] >= result[-1][1]:
            result[-1][1] = intervels[i][1]
        else:
            result.append(intervels[i].copy())
    print(result)
    return len(result)

print(meeting2([[0,30],[5,10],[15,20]]))
print(meeting2([[7,10],[2,4]]))

import collections 
import heapq
def meeting3(k, intervels:list):
    n = len(intervels)
    counts = [0] * k
    intervels.sort(key=lambda x : x[0]) 
    queue = collections.

    for i in range(k):
        counts[i] += 1
        queue.append(intervels[i][1])
    for i in range(k, n):
