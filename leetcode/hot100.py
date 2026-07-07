

import collections

# 1.两数之和
def twoNums(nums, target):
	dicts = collections.defaultdict(int)

	n = len(nums)

	for i in range(n):
		if target - nums[i] in dicts:
			return [i, dicts[target - nums[i]] ]
		else:
			dicts[nums[i]] = i

	return -1


# nums = [2,7,11,15]
# target = 9
# print(twoNums(nums, target))
# nums = [3,2,4]
# target = 6
# print(twoNums(nums, target))
# nums = [3,3]
# target = 6
# print(twoNums(nums, target))


# 49. 字母异位词分组

import collections

def helper(s):
	letterCounts = [0] * 26
	for ss in s:
		letterCounts[ord(ss)-ord('a')] += 1
	return tuple(letterCounts)

def groupWords(strs):
	wordDict = collections.defaultdict(list)

	for ss in strs:
		wordDict[helper(ss)].append(ss)

	result = list(wordDict.values())
	# print('type', type(result))
	# assert type(result) == list
	# result.sort(key=lambda x: len(x))
	# result.reverse()
	return result

def groupWords2(strs):
	wordDict = collections.defaultdict(list)

	for ss in strs:
		wordDict[''.join(sorted(ss))].append(ss)

	return list(wordDict.values())

# strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
# print(groupWords2(strs))
# strs = [""]
# print(groupWords2(strs))
# strs = ["a"]
# print(groupWords2(strs))


# 128. 最长连续序列
def longestSeqLen(nums):
	numSet = set(nums)
	maxLength = 0

	for x in numSet:
		if x + 1 not in numSet:
			length = 1
			cur = x
			while cur - 1 in numSet:
				length += 1
				cur -= 1
			maxLength = max(maxLength, length)

	return maxLength


nums = [100,4,200,1,3,2]
# print(longestSeqLen(nums))

# 283. 移动零

def moveZero(nums):
	# 输入: nums = [0,1,0,3,12]
	# 输出: [1,3,12,0,0]
	zeroIndex = 0
	n = len(nums)
	for i in range(n):
		if nums[i] != 0:
			nums[i], nums[zeroIndex] = nums[zeroIndex], nums[i]
			zeroIndex += 1
	return 

# nums = [0,1,0,3,12]
# moveZero(nums)
# print(nums)


# 盛水最多的容器
def maxArea(nums):
	n = len(nums)
	left = 0
	right = n - 1
	result = 0
	while left < right:
		area = (right - left) * min(nums[left], nums[right])
		result = max(area, result)
		if nums[left] < nums[right]:
			left += 1
		else:
			right -= 1
	return result


# nums = [1,8,6,2,5,4,8,3,7]
# print(maxArea(nums))


# 三数之和
def threeSum(nums):
	nums.sort()
	n = len(nums)
	result = []
	for first in range(n):

		if first > 0 and nums[first] == nums[first-1]:
			continue

		second = first + 1
		third = n - 1
		target = -nums[first]
		while second < third:
			if nums[second] + nums[third] == target:
				result.append([nums[first], nums[second], nums[third]])
				second += 1
				third -= 1
				while second < third and nums[second] == nums[second-1]:
					second += 1
				while second < third and nums[third] == nums[third+1]:
					third -= 1

			elif nums[second] + nums[third] < target:
				second += 1
			else:
				third -= 1
	return result

# nums = [-1,0,1,2,-1,-4]
# print(threeSum(nums))
# nums = [-1, -1, -1, 0, 1, 1, 1, 1, 4]
# print(threeSum(nums))

# 3. 无重复字符的最长子串
def maxLengthofSubString(s):
	result = 0
	sett = set()
	n = len(s)
	right = 0
	for left in range(n):
		if left > 0:
			sett.remove(s[left-1])

		while right < n and s[right] not in sett:
			sett.add(s[right])
			right += 1
		result = max(result, right-left)

	return result

# s = "abcabcbb"
# print('3', maxLengthofSubString(s))




# # 42. 接雨水
def volumn(height):
	n = len(height)
	result = 0

	leftMax = [0] * n
	leftMax[0] = height[0]
	rightMax = [0] * n
	rightMax[-1] = height[-1]

	for i in range(n-1):
		leftMax[i+1] = max(leftMax[i], height[i+1])

	for i in range(n-2, -1, -1):
		rightMax[i] = max(rightMax[i+1], height[i])
	print(leftMax)
	print(rightMax)
	for i in range(n):
		result += (min(leftMax[i], rightMax[i]) - height[i])

	return result

def volumn2(height):
	n = len(height)
	result = 0
	leftMax = height[0]
	rightMax = height[-1]
	left, right = 0, n-1

	while left < right:
		leftMax = max(height[left], leftMax)
		rightMax = max(height[right], rightMax)

		if leftMax < rightMax:
			result += (leftMax - height[left])
			left += 1
		else:
			result += (rightMax - height[right])
			right -= 1

	return result


# height = [0,1,0,2,1,0,1,3,2,1,2,1]
# print(volumn2(height))




# # # 438. 找到字符串中所有字母异位词
def findDiff(s, p):
	ls = len(s)
	lp = len(p)
	result = []
	plist = [0] * 26
	slist = [0] * 26
	for pp in p:
		plist[ord(pp)-ord('a')] += 1
	print(plist)
	for i, ss in enumerate(s):
		slist[ord(ss)-ord('a')] += 1
		if i >= lp :
			slist[ord(s[i-lp]) - ord('a')] -= 1

		if slist == plist:
				result.append(i-lp+1)
	return result



s = "cbaebabacd"
p = "abc"
# [0,6]
# print('findDiff', findDiff(s, p))





# 560. 和为 K 的子数组

import collections

def subArraySum(nums, target):
	n = len(nums)
	prefix = [0] * (n+1)
	hashmap = collections.defaultdict(int)
	result = 0
	for i in range(n):
		prefix[i+1] = prefix[i] + nums[i]

	for i in range(n+1):
		if prefix[i] - target in hashmap:
			result += hashmap[prefix[i] - target]

		hashmap[prefix[i]] += 1
	return result

# print(subArraySum([1,1,1], 1))

# print(subArraySum([1,2,3], 3))


from math import *

# # 53. 最大子数组和 连续
def MaxSumOfSubArray(nums):
	curMax = -inf
	globalMax = -inf
	for i in range(len(nums)):
		curMax = max(curMax+nums[i], nums[i])
		globalMax = max(globalMax, curMax)

	return globalMax


# nums = [-2,1,-3,4,-1,2,1,-5,4]
# print(MaxSumOfSubArray(nums))



# 56. 合并区间
def merge(intervals):
	intervals.sort(key = lambda x : x[0])

	result = [intervals[0]]
	for interval in intervals:
		if result[-1][1] < interval[1]:
			if result[-1][1] < interval[0]:
				result.append(interval)
			else:
				result[-1][1] = interval[1]
	return result



# intervals = [[1,3],[2,6],[8,10],[15,18]]
# intervals = [[1,1.5],[2,6],[5.5,10],[15,18]]
# intervals = [[15,18], [1,3],[2,6],[8,10]]

# print(merge(intervals))



# 189. 轮转数组
def rotate(nums, k):
	# 输入: nums = [1,2,3,4,5,6,7], k = 3
	# 输出: [5,6,7,1,2,3,4]

	n = len(nums)
	for i in range(n // 2):
		nums[i], nums[n-i-1] = nums[n-i-1], nums[i]

	for i in range(k // 2):
		nums[i], nums[k-i-1] = nums[k-i-1], nums[i]

	for i in range((n-k) // 2):
		nums[k+i], nums[n-i-1] = nums[n-i-1], nums[k+i]



# nums = [1,2,3,4,5,6,7]
# k = 3
# rotate(nums, k)
# print(nums)



# 238. 除了自身以外数组的乘积
def productOfArray(nums):
	# 输入: nums = [1,2,3,4]
	# 输出: [24,12,8,6]
	n = len(nums)
	prefix = [nums[0]] * n
	subfix = [nums[-1]] * n

	for i in range(1, n):
		prefix[i] = prefix[i-1]*nums[i]

	for i in range(n-2, -1, -1):
		subfix[i] = subfix[i+1]*nums[i]
	print(prefix)
	print(subfix)

	result = [0] * n
	result[0] = subfix[1]
	result[-1] = prefix[-2]

	for i in range(1, n-1):
		result[i] = prefix[i-1]*subfix[i+1]

	return result

# print(productOfArray([1,2,3,4,4,4]))

# 41. 缺失的第一个正数
def firstMissingPositive(nums):
	# 输入：nums = [1,2,0]
	# 输出：3
	# 原地Hash, 利用ARRAY INDEX和其值的关系

	n = len(nums)
	for i in range(n):
		while 0 < nums[i] <= n and nums[nums[i]-1] != nums[i]:
			nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]

	for i in range(n):
		if nums[i] != i+1:
			return i+1
	return nums[-1]+1

# print(firstMissingPositive([1, 2, 0]))
# print(firstMissingPositive([3, 4, -1, 1]))
# print(firstMissingPositive([1]))


# 73. 矩阵置零
def setMatrixZero(matrix):
	# 输入：matrix = [[1,1,1],[1,0,1],[1,1,1]]
	# 输出：[[1,0,1],[0,0,0],[1,0,1]]
	rowIndex = set()
	columnIndex = set()
	rows = len(matrix)
	columns = len(matrix[0])

	for row in range(rows):
		for column in range(columns):
			if matrix[row][column] == 0:
				rowIndex.add(row)
				columnIndex.add(column)

	for row in range(rows):
		for column in columnIndex:
			matrix[row][column] = 0

	for row in rowIndex:
		for column in range(columns):
			matrix[row][column] = 0
	
	return 			

# matrix = [[1,1,1],[1,0,1],[1,1,1]]
# setMatrixZero(matrix)
# print(matrix)


# 54. 螺旋矩阵
def rotateMatrix(matrix):
	# 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
	# 输出：[1,2,3,6,9,8,7,4,5]

	directions = [[0,1], [1,0],[0,-1],[-1,0]]
	di = 0
	count = 0
	rows = len(matrix)
	columns = len(matrix[0])
	result = []
	i, j = 0, 0
	used = [[0]*columns for _ in range(rows)]
	while count < rows*columns:
		if used[i][j] == 0:
			result.append(matrix[i][j])
			used[i][j] = 1
			count += 1
			while count < rows*columns:
				x = i + directions[di][0]
				y = j + directions[di][1]
				if 0 <= x < rows and 0 <= y < columns and used[x][y] == 0:
					
					break
				else:
					di = (di + 1) % 4
			i += directions[di][0]
			j += directions[di][1]
		
			
	return result

# print('rotateMatrix', rotateMatrix([[1,2,3],[4,5,6],[7,8,9]]))



# 48. 旋转图像
def rotateMatrix2(matrix):
	# 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
	# 输出：[[7,4,1],[8,5,2],[9,6,3]]
	n = len(matrix)
	for i in range(n):
		for j in range(i, n):
			matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


	for i in range(n):
		for j in range(n // 2):
			matrix[i][j], matrix[i][n-j-1] = matrix[i][n-j-1], matrix[i][j]
	
	
# matrix = [[1,2,3],[4,5,6],[7,8,9]]
# rotateMatrix2(matrix)
# print(matrix)


# 240. 搜索二维矩阵 II
def searchMatrix(matrix, target):
	# 输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
	# 输出：true
	rows = len(matrix)
	columns = len(matrix[0])

	row, column = 0, columns-1

	while 0 <= row < rows and 0 <= column < columns:
		if matrix[row][column] == target:
			return True
		elif matrix[row][column] > target:
			column -= 1
		else:
			row += 1
	return False

# print(searchMatrix([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 5.5))

class ListNode:
	def __init__(self, val, nxt=None):
		self.val = val
		self.next = nxt

	def __str__(self):
		return 'Node{ ' + str(self.val) + " }"

	@staticmethod
	def print(head=None):
		lis = []
		p = head
		while p:
			lis.append(str(p.val))
			p = p.next
		print('->'.join(lis))

	@staticmethod
	def len(head):
		length = 0
		p = head
		while p:
			length += 1
			p = p.next
		return length

# # 160. 相交链表
def findTheLinkedListNode(headA, headB):

	lenA = ListNode.len(headA)
	lenB = ListNode.len(headB)
	if lenA > lenB:
		for _ in range(lenA-lenB):
			headA = headA.next
	else:
		for _ in range(lenB-lenA):
			headB = headB.next

	while headA and headB:
		if headA == headB:
			return headA
		headA = headA.next
		headB = headB.next

	return None
a = ListNode(3, ListNode(4))
h1 = ListNode(2, a)
h2 = ListNode(1, a)
# print(findTheLinkedListNode(h1, h2))



# 206. 反转链表
def reverseLinkedList(head):

	if not head or not head.next:
		return head

	pre = None
	cur = head
	curNext = None

	while cur:
		curNext = cur.next
		cur.next = pre
		pre = cur
		cur = curNext

	return pre

# head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
# ListNode.print(head)
# reversedHead = reverseLinkedList(head)
# ListNode.print(reversedHead)

# 
def reverseLinkedList2(head, left, right):
	

	times = right-left+1

	dummyHead = ListNode(-1, head)
	start = dummyHead

	for i in range(left-1):
		start = start.next

	pre = None
	cur = start.next
	curNext = None

	for t in range(times):
		curNext = cur.next
		cur.next = pre
		pre = cur
		cur = curNext
	tail = start.next
	start.next = pre
	tail.next = cur

	return dummyHead.next

# head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
# ListNode.print(head)
# reversedHead = reverseLinkedList2(head, 2, 4)
# ListNode.print(reversedHead)


# 234. 回文链表

def ifEqualList(head1, head2):
    while head1 and head2:
        if head1.val != head2.val:
            return False
        head1 = head1.next
        head2 = head2.next
    return True

def isRervesedList(head):
	if not head or not head.next:
		return True
	length = ListNode.len(head)
	tail = head
	for i in range(length // 2 - 1):
		tail = tail.next
	if length % 2 == 0:
		head2 = reverseLinkedList(tail.next)
	else:
		head2 = reverseLinkedList(tail.next.next)
	tail.next = None

	return ifEqualList(head2, head)
	

# head = ListNode(1, ListNode(2, ListNode(3, ListNode(3, ListNode(2, ListNode(1))))))
# print(isRervesedList(head))

# head = ListNode(1, ListNode(1, ListNode(2, ListNode(1))))

# print(isRervesedList(head))

# # 141. 环形链表
def hasCycle(head):
	fast = slow = head
	while fast and fast.next:
		slow = slow.next
		fast = fast.next.next

		if slow == fast:
			return True

	return False

# a = ListNode(1, ListNode(2, ListNode(3)))
# print(hasCycle(a))
# b = ListNode(0, a)
# a.next = b
# c = ListNode(-1, b)
# print(hasCycle(c))
# print(hasCycle(b))
# print(hasCycle(a))



# 142. 环形链表 II
def hasCycle2(head):
	fast = slow = head
	while fast and fast.next:
		slow = slow.next
		fast = fast.next.next

		if slow == fast:
			while True:
				if head == slow:
					return slow
				slow = slow.next
				head = head.next

	return None


# 21. 合并两个有序链表
def mergeTwoList(lis1, lis2):
	dummyHead = ListNode(-1, None)
	indexHead = dummyHead
	while lis1 and lis2:
		if lis1.val < lis2.val:
			indexHead.next = ListNode(lis1.val)
			lis1 = lis1.next
		else:
			indexHead.next = ListNode(lis2.val)
			lis2 = lis2.next
		indexHead = indexHead.next

	if lis1:
		indexHead.next = lis1
	if lis2:
		indexHead.next = lis2

	return dummyHead.next



lis1 = ListNode(1, ListNode(3, ListNode(5)))
lis2 = ListNode(2, ListNode(4, ListNode(6)))

# ListNode.print(mergeTwoList(lis1, lis2))




# 2. 两数相加
def addTwoNum(lis1, lis2):
	c = 0
	dummyHead = ListNode(-1)
	indexHead = dummyHead
	while lis1 and lis2:
		tail = (lis1.val + lis2.val + c) % 10
		indexHead.next = ListNode(tail)

		c = (lis1.val + lis2.val + c) // 10
		lis1 = lis1.next
		lis2 = lis2.next
		indexHead = indexHead.next


	while lis1:
		indexHead.next = ListNode((lis1.val + c) % 10)
		c = (lis1.val + c) // 10
		indexHead = indexHead.next

	while lis2:
		indexHead.next = ListNode((lis2.val + c) % 10)
		c = (lis2.val + c) // 10
		indexHead = indexHead.next

	if c > 0:
		indexHead.next = ListNode(c)
	return dummyHead.next





# 19. 删除链表的倒数第 N 个结点
def removeNthNode(head, n):
	length = ListNode.len(head)
	dummyHead = ListNode(-1, head)
	pHead = dummyHead
	assert n <= length
	for _ in range(length - n):
		pHead = pHead.next
	pHead.next = pHead.next.next
	return dummyHead.next

def removeNthNode2(head, n):
	stack = []
	dummyHead = ListNode(-1, head)
	pHead = dummyHead
	while pHead:
		stack.append(pHead)
		pHead = pHead.next 
	pre = None
	for _ in range(n+1):
		pre = stack.pop(-1)
	pre.next = pre.next.next

	return dummyHead.next


# lis1 = ListNode(1, ListNode(3, ListNode(5)))
# ListNode.print(lis1)
# lis1 = removeNthNode2(lis1, 3)
# ListNode.print(lis1)
# lis1 = removeNthNode2(lis1, 2)
# ListNode.print(lis1)



# 24. 两两交换链表中的节点
def swapTwoNode(head):

	if not head or not head.next:
		return head
	headNext = head.next
	headNextNext = swapTwoNode(head.next.next)

	head.next = headNextNext
	headNext.next = head

	return headNext



# lis1 = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5, ListNode(6))))))
# ListNode.print(lis1)
# lis1 = swapTwoNode(lis1)
# ListNode.print(lis1)


# 25. K 个一组翻转链表

def reverseKGroupList(head, k):
	length = ListNode.len(head)
	times = length // k
	dummyHead = ListNode(-1, head)
	pre = None
	cur = head
	tail = dummyHead
	curNext = None
	for i in range(times):

		for j in range(k):
			curNext = cur.next
			cur.next = pre
			pre = cur
			cur = curNext

		tailNext = tail.next
		tail.next = pre
		tailNext.next = cur
		tail = tailNext
		pre = None
	return dummyHead.next

# lis1 = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5, ListNode(6))))))
# ListNode.print(lis1)
# lis1 = reverseKGroupList(lis1,3)
# ListNode.print(lis1)



# 148. 排序链表

def mergeList(phead, qhead):
	dummyHead = ListNode(-1)
	indexNode = dummyHead

	while phead and qhead:
		if phead.val < qhead.val:
			indexNode.next = ListNode(phead.val)
			phead = phead.next
		else:
			indexNode.next = ListNode(qhead.val)
			qhead = qhead.next
		indexNode = indexNode.next
	if phead:
		indexNode.next = phead
	if qhead:
		indexNode.next = qhead
	return dummyHead.next


def sortList(head):
	if not head or not head.next:
		return head
	length = ListNode.len(head)
	tail = head
	for _ in range(length // 2 - 1):
		tail = tail.next
	qhead = tail.next
	tail.next = None


	phead = sortList(head)
	qhead = sortList(qhead)

	return mergeList(phead, qhead)




head = ListNode(1, ListNode(2, ListNode(3, ListNode(3, ListNode(2, ListNode(1))))))
ListNode.print(head)
head = sortList(head)
ListNode.print(head)

import collections
class RandomNode:
	def __init__(self, val, nxt=None, random=None):
		self.val = val
		self.next = nxt
		self.random = random

	def __str__(self):
		return 'Node{ ' + str(self.val) + " }"
	@staticmethod
	def print(head):
		lis = []
		lis2 = []
		p = head
		while p:
			lis.append(str(p.val))
			p = p.next
		p = head
		while p:
			lis.append(str(p.val))
			p = p.random
		print('next' + '->'.join(lis))
		print('random' + '->'.join(lis2))




def copyRandomList(head):
	hashmap = collections.defaultdict(RandomNode)
	dummyHead = RandomNode(-1)
	indexNode = dummyHead
	phead = head

	while phead:

		indexNode.next = RandomNode(phead.val)
		indexNode = indexNode.next
		hashmap[phead] = indexNode
		phead = phead.next


	phead = head

	while phead:
		if phead.random:
			hashmap[phead].random = hashmap[phead.random]
		phead = phead.next

	return dummyHead.next


# a = RandomNode(3)
# b = RandomNode(1, RandomNode(2, a), a)
# a.random = b
# c = copyRandomList(b)
# RandomNode.print(b)
# RandomNode.print(c)



def mergeTwoList(headList1, headList2):

	dummyHead = ListNode(-1)
	indexNode = dummyHead
	
	head1 = mergeKList(headList1)

	head2 = mergeKList(headList2)

	while head1 and head2:
		if head1.val < head2.val:
			indexNode.next = ListNode(head1.val)
			head1 = head1.next
		else:
			indexNode.next = ListNode(head2.val)
			head2 = head2.next

		indexNode = indexNode.next

	if head1:
		indexNode.next = head1
	if head2:
		indexNode.next = head2
	return dummyHead.next



def mergeKList(headList):
	k = len(headList)
	if k == 0: return None
	if k == 1: return headList[0]
	return mergeTwoList(headList[0:k//2], headList[k//2:])


# a = ListNode(1, ListNode(2, ListNode(2, ListNode(2, ListNode(2, ListNode(3))))))
# b = ListNode(1, ListNode(2, ListNode(2, ListNode(2, ListNode(2, ListNode(3))))))
# c = ListNode(1, ListNode(2, ListNode(2, ListNode(2, ListNode(2, ListNode(3))))))
# d = ListNode(1, ListNode(2, ListNode(2, ListNode(2, ListNode(2, ListNode(3))))))
# ListNode.print(mergeKList([a, b]))





'''
请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

示例：
输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
'''

import collections
class Node:
	def __init__(self, val, pre=None, next=None):
		self.val = val
		self.pre = pre
		self.next = next

	@staticmethod
	def print(head):
		print('Node', head.val)

	@staticmethod
	def printALL(head):
		if head.pre and head.next:
			print('Node ALL', head.val, head.pre.val, head.next.val)
		elif head.pre:
			print('Node ALL', head.val, head.pre.val)
		elif head.next:
			print('Node ALL', head.val, head.next.val)

class LRUCache:
	def __init__(self, capacity):
		self.hashmapNode = collections.defaultdict(Node)
		self.hashmapValue = collections.defaultdict(int)
		self.head = Node(-100)
		self.tail = Node(-200)
		self.size = 0
		self.capacity = capacity
		self.head.next = self.tail
		self.tail.pre = self.head

	def get(self, key):
		if key in self.hashmapValue:

			keyNode = self.hashmapNode[key]
			keyPre = keyNode.pre
			keyNext = keyNode.next
			tailPre = self.tail.pre

			keyPre.next = keyNext
			keyNext.pre = keyPre
			keyNode.pre = tailPre
			keyNode.next = self.tail
			tailPre.next = keyNode
			self.tail.pre = keyNode

			return self.hashmapValue[key]
		else:
			return '--1'

	def put(self, key, value):

		if key in self.hashmapValue:
			

		else:
			if self.size < self.capacity:

				self.hashmapValue[key] = value
				tailPre = self.tail.pre
				newNode = Node(key, tailPre, self.tail)
				tailPre.next = newNode
				self.tail.pre = newNode
				self.hashmapNode[key] = newNode
				Node.printALL(self.head)
				Node.printALL(newNode)
				Node.printALL(self.tail)
				self.size += 1
			else:
				headNext = self.head.next
				if headNext.val in self.hashmapValue:
					del self.hashmapValue[headNext.val]
					del self.hashmapNode[headNext.val]
				print('size', self.size)
				print('head.next', self.head.next.val)
				Node.print(headNext)
				headNextNext = headNext.next
				tailPre = self.tail.pre
				self.head.next = headNextNext

				headNextNext.pre = self.head

				self.hashmapValue[key] = value
				tailPre = self.tail.pre
				newNode = Node(key, tailPre, self.tail)
				tailPre.next = newNode
				self.tail.pre = newNode
				self.hashmapNode[key] = newNode


# lRUCache = LRUCache(2)
# lRUCache.put(1, 1)
# lRUCache.put(2, 2)
# print(lRUCache.get(1))
# lRUCache.put(3, 3)
# print(lRUCache.get(2))
# lRUCache.put(4, 4)
# print(lRUCache.get(1))
# print(lRUCache.get(3))
# print(lRUCache.get(4))

lRUCache = LRUCache(1)
lRUCache.put(2, 1)
print(lRUCache.get(2))
lRUCache.put(3, 2)
print(lRUCache.get(2))
print(lRUCache.get(3))



import collections

class TreeNode:
	def __init__(self, val, left=None, right=None):
		self.val = val
		self.left = left
		self.right = right

	@staticmethod
	def print(root):
		queue = collections.deque()
		# append / appendleft / pop / popleft
		queue.append(root)
		lis = []
		while queue:
			for i in range(len(queue)):
				queue.append()




# 94. 二叉树的中序遍历


# 104. 二叉树的最大深度


# 226. 翻转二叉树



# 101. 对称二叉树


# 543. 二叉树的直径


# 102. 二叉树的层序遍历


# 108. 将有序数组转换为二叉搜索树


# 98. 验证二叉搜索树


# 230. 二叉搜索树中第 K 小的元素





