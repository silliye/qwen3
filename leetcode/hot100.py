

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
			return 

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

# lRUCache = LRUCache(1)
# lRUCache.put(2, 1)
# print(lRUCache.get(2))
# lRUCache.put(3, 2)
# print(lRUCache.get(2))
# print(lRUCache.get(3))



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
			temp = []
			for i in range(len(queue)):
				popRoot = queue.popleft()
				temp.append(str(popRoot.val))

				if popRoot.left:
					queue.append(popRoot.left)
				if popRoot.right: 
					queue.append(popRoot.right)

			lis.append(temp)
		print(lis)
		return lis

# 94. 二叉树的中序遍历

def midTraceHelper(root, lis):
	if root:
		midTraceHelper(root.left, lis)
		lis.append(root.val)
		midTraceHelper(root.right, lis)

def midTrace(root):
	lis = []
	midTraceHelper(root, lis)
	return lis

root = TreeNode(1, TreeNode(2, TreeNode(3), TreeNode(4)), TreeNode(5))
print(midTrace(root))



def midTrace2(root):
	stack = []
	stack.append((root, 0))
	result = []
	while stack:
		popRoot = stack.pop(-1)
		if popRoot[1] == 1:
			result.append(popRoot[0].val)
		else:
			if popRoot[0].right:
				stack.append((popRoot[0].right, 0))
			stack.append((popRoot[0], 1))
			if popRoot[0].left:
				stack.append((popRoot[0].left, 0))
	return result

root = TreeNode(1, TreeNode(2, TreeNode(3), TreeNode(4)), TreeNode(5))
print(midTrace2(root))


def preTrace(root):
	stack = []
	stack.append((root, 0))
	result = []
	while stack:
		popRoot = stack.pop(-1)
		if popRoot[1] == 1:
			result.append(popRoot[0].val)
		else:
			if popRoot[0].right:
				stack.append((popRoot[0].right, 0))
			if popRoot[0].left:
				stack.append((popRoot[0].left, 0))
			stack.append((popRoot[0], 1))

	return result

root = TreeNode(1, TreeNode(2, TreeNode(3), TreeNode(4)), TreeNode(5))
print(preTrace(root))



def postTrace(root):
	stack = []
	stack.append((root, 0))
	result = []
	while stack:
		popRoot = stack.pop(-1)
		if popRoot[1] == 1:
			result.append(popRoot[0].val)
		else:
			stack.append((popRoot[0], 1))
			if popRoot[0].left:
				stack.append((popRoot[0].right, 0))
			if popRoot[0].left:
				stack.append((popRoot[0].left, 0))
			
	return result
root = TreeNode(1, TreeNode(2, TreeNode(3), TreeNode(4)), TreeNode(5))
print(postTrace(root))



# 104. 二叉树的最大深度
def maxDepth(root):
	if not root:
		return 0
	else:
		return 1 + max(maxDepth(root.left), maxDepth(root.right))



# 226. 翻转二叉树
def swapBinaryTree(root):
	if not root or (not root.left and not root.right):
		return root
	rightNode = swapBinaryTree(root.left)
	leftNode = swapBinaryTree(root.right)
	root.left = rightNode
	root.right = ListNode

	return root
	
# 101. 对称二叉树

def isSymmetricHelper(root1, root2):
	if not root1 and not root2:
		return True
	if not root1 or not root2:
		return False
	return root1.val == root2.val and isSymmetricHelper(root1.left, root2.right) and isSymmetricHelper(root2.left, root1.right)


def isSymmetric(root):
	if not root:
		return True
	return isSymmetricHelper(root.left, root.right)



# 543. 二叉树的直径
def depth(root):
	result = 0
	def dfs(root):
		nonlocal result
		if not root: return 0
		leftDepth = dfs(root.left)
		rightDepth = dfs(root.right)

		result = max(result, leftDepth+rightDepth)

		return max(leftDepth, rightDepth) + 1
	dfs(root)
	return result

# 102. 二叉树的层序遍历
import collections
def levelTrace(root):

	queue = collections.deque()
	queue.append(root)
	lis = []
	while queue:
		temp = []
		for _ in range(len(queue)):
			popRoot = queue.popleft()
			temp.append(popRoot.val)
			if popRoot.left:
				queue.append(popRoot.left)
			if popRoot.right:
				queue.append(popRoot.right)
		lis.append(temp)
	return lis

root = TreeNode(1, TreeNode(2, TreeNode(3), TreeNode(4)), TreeNode(5))

# print(levelTrace(root))
print(depth(root))


# 108. 将有序数组转换为二叉搜索树
def setArrayToBST(lis):
	def helper(lis, l, r):
		if l > r: return None
		if l == r:
			return TreeNode(lis[l])
		mid = (l + r) // 2
		return TreeNode(lis[mid], helper(lis, l, mid-1), helper(lis, mid+1, r))
	return helper(lis, 0, len(lis)-1)

from math import inf
# 98. 验证二叉搜索树
def isValidBST(root):
	def helper(root, maxx, minn):
		if not root:
			return True
		if root.val >= maxx or root.val <= minn:
			return False
		return helper(root.left, root.val, minn) and helper(root.right, maxx, root.val)
	return helper(root, inf, -inf)
root = TreeNode(2, TreeNode(1), TreeNode(3))
print(isValidBST(root))
root = TreeNode(2, TreeNode(2), TreeNode(2))
print(isValidBST(root))
root = TreeNode(5, TreeNode(1), TreeNode(4, TreeNode(3), TreeNode(6)))
print(isValidBST(root))


# 230. 二叉搜索树中第 K 小的元素

def kthSmallest(root, k):
	if not root: return root
	stack = []
	stack.append((root, 0))
	result = []
	while stack:
		popRoot, flag = stack.pop(-1)
		if flag:
			result.append(popRoot.val)
		else:
			if popRoot.right:
				stack.append((popRoot.right, 0))
			stack.append((popRoot, 1))
			if popRoot.left:
				stack.append((popRoot.left, 0))
			
	return result[k-1]


def kthSmallest2(root, k):
	# ** 重点review
	ans = 0
	def dfs(root):
		nonlocal k, ans
		if not root: return

		dfs(root.left)
		k = k - 1
		if k == 0:
			ans = root.val
		dfs(root.right)

		return 
	dfs(root, k)
	return ans


# 199. 二叉树的右视图
import collections
def rightViewofTree(root):
	queue = collections.deque()
	queue.append(root)
	result = []

	while queue:
		for i in range(len(queue)):
			popRoot = queue.popleft()
			if i == 0:
				result.append(popRoot.val)
			if popRoot.right:
				queue.append(popRoot.right)
			if popRoot.left:
				queue.append(popRoot.left)
	
	return result

root = TreeNode(1, TreeNode(2, None, TreeNode(5)), TreeNode(3, None, TreeNode(4)))
# print(rightViewofTree(root))




# 114. 二叉树展开为链表
def flatten(root):
	def helper(root):
		if not root or (not root.left and not root.right):
			return root
		rightFlatten = helper(root.right)
		root.right = rightFlatten

		leftFlatten = helper(root.left)
		if leftFlatten:
			proot = leftFlatten
			while proot.right:
				proot = proot.right
			proot.right = root.right
			root.right = leftFlatten
			root.left = None
		return root
	helper(root)
	return 

root = TreeNode(1, TreeNode(2, TreeNode(3), TreeNode(4)), TreeNode(5, None, TreeNode(6)))
# TreeNode.print(root)
# flatten(root)
# TreeNode.print(root)




# # 105. 从前序与中序遍历序列构造二叉树
def buildTree(preorder, inorder):
	if not preorder: return None

	rootValue = preorder[0]
	rootIndex = inorder.index(rootValue)
	return TreeNode(rootValue, buildTree(preorder[1:1+rootIndex], inorder[0:rootIndex]), buildTree(preorder[1+rootIndex:], inorder[1+rootIndex:]))


preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]

# root = buildTree(preorder, inorder)
# TreeNode.print(root)


# # 437. 路径总和 III
import collections
def pathSumIII(root, targetSum):
	if not root: return 0
	result = 0
	hashmap = collections.defaultdict(int)
	hashmap[0] = 1
	def dfs(root, pathSum):
		nonlocal result
		if pathSum - targetSum in hashmap:
			result += hashmap[pathSum - targetSum]
			
		hashmap[pathSum] += 1
		if root.left:
			dfs(root.left, pathSum+root.left.val)
		if root.right:
			dfs(root.right, pathSum+root.right.val)

		hashmap[pathSum] -= 1

	dfs(root, root.val)
	return result

print('路径总和 III')
root = TreeNode(10, TreeNode(5, TreeNode(3, TreeNode(3), TreeNode(-2)), TreeNode(2, None, TreeNode(1))), TreeNode(-3, None, TreeNode(11))) 
TreeNode.print(root)
print(pathSumIII(root, 8) == 3)
print(pathSumIII(root, 21) == 1)

# 112. 路径总和
def pathSumI(root, targetSum):
		
	if not root: return False

	if (not root.left and not root.right) and root.val == targetSum:
		return True

	return pathSumI(root.left, targetSum-root.val) or pathSumI(root.right, targetSum-root.val)	


print('路径总和 I')
root = TreeNode(5, TreeNode(4, TreeNode(11, TreeNode(7), TreeNode(2))), TreeNode(8, TreeNode(13), TreeNode(4, None, TreeNode(1))))
print(pathSumI(root, 22))
root = TreeNode(1, TreeNode(2), TreeNode(3))
print(pathSumI(root, 5))


# 113. 路径总和 II
def pathSumII(root, targetSum):
	if not root: return []
	result = []
	path = [root.val]	
	def dfs(root, prefix):
		if not root: return 
		if (not root.left and not root.right) and prefix == targetSum:
			result.append(path.copy())
			return 
		if root.left:
			path.append(root.left.val)
			dfs(root.left, prefix+root.left.val)
			path.pop(-1)
		if root.right:
			path.append(root.right.val)
			dfs(root.right, prefix+root.right.val)
			path.pop(-1)
	dfs(root, root.val)
	return result

def pathSumII2(root, targetSum):
	if not root: return []
	result = []
	path = []	
	def dfs(root, prefix):
		if not root and prefix == targetSum:
			result.append(path.copy())
			return 
		if not root: return 
		path.append(root.val)

		dfs(root.left, prefix+root.val)

		dfs(root.right, prefix+root.val)

		path.pop(-1)
		
	dfs(root, 0)
	return result


print('路径总和 II')
root = TreeNode(5, TreeNode(4, TreeNode(11, TreeNode(7), TreeNode(2))), TreeNode(8, TreeNode(13), TreeNode(4, TreeNode(5), TreeNode(1))))
print(pathSumII(root, 22))

# # 236. 二叉树的最近公共祖先
def nearestCommon(root, node1, node2):
	if root == node1 or root == node2:
		return root

	ifLeftExist = nearestCommon(root.left, node1, node2)
	ifRightExist = nearestCommon(root.right, node1, node2)

	if ifLeftExist and ifRightExist:
		return root

	return ifLeftExist if ifLeftExist else ifRightExist


# def nearestCommon2(root, node1, node2):
# 	if root == node1 or root == node2:
# 		return root

# 	ifLeftExist = nearestCommon(root.left, node1, node2)
# 	ifRightExist = nearestCommon(root.right, node1, node2)

# 	if ifLeftExist and ifRightExist:
# 		return root

# 	return False if (not ifLeftExist and not ifRightExist) else True

	
a = TreeNode(5, TreeNode(6), TreeNode(2, TreeNode(7), TreeNode(4)))
b = TreeNode(1, TreeNode(0), TreeNode(8))
root = TreeNode(3, a, b)

print(nearestCommon(root, a, b).val)


# 124. 二叉树中的最大路径和
from math import inf
def maxPathSum(root):
	result = -inf

	def dfs(root):
		nonlocal result
		if not root: return 0
		if not root.left and not root.right:
			result = max(result, root.val)
			return root.val
		leftPathSum = dfs(root.left)

		rightPathSum = dfs(root.right)
		result = max(result, leftPathSum+rightPathSum+root.val)

		return max(leftPathSum+root.val, rightPathSum+root.val, 0)

	dfsResult = dfs(root)
	return max(result, dfsResult)
	


print('二叉树中的最大路径和')
root = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
# print(maxPathSum(root))
root = TreeNode(1, TreeNode(2), TreeNode(3))
# print(maxPathSum(root))
root = TreeNode(-2, TreeNode(1))
print(maxPathSum(root))



# 200. 岛屿数量
def numsIslands(grid):
	rows = len(grid)
	columns = len(grid[0])
	def dfs(r, c):
		nonlocal rows, columns
		if grid[r][c] == '1':
			grid[r][c] = '0'
			for i, j in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
				x = r+i
				y = c+j
				if 0 <= x < rows and 0 <= y < columns and grid[x][y] == '1':
					dfs(x, y)
	# dfs
	count = 0
	for r in range(rows):
		for c in range(columns):
			if grid[r][c] == '1':
				dfs(r, c)
				count += 1
	return count

print('200. 岛屿数量')
grid = [
  ['1','1','1','1','0'],
  ['1','1','0','1','0'],
  ['1','1','0','0','0'],
  ['0','0','0','0','0']
]
print(numsIslands(grid) == 1)
grid = [
  ['1','1','0','0','0'],
  ['1','1','0','0','0'],
  ['0','0','1','0','0'],
  ['0','0','0','1','1']
]
print(numsIslands(grid) == 3)


# 695. 岛屿的最大面积
def maxAreaOfIslands(grid):
	rows = len(grid)
	columns = len(grid[0])
	def dfs(r, c, count):
		nonlocal rows, columns
		if grid[r][c] == 1:
			grid[r][c] = 0
			count[0] += 1
			for i, j in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
				x = r+i
				y = c+j
				if 0 <= x < rows and 0 <= y < columns and grid[x][y] == 1:
					dfs(x, y, count)
	# dfs
	result = 0
	for r in range(rows):
		for c in range(columns):
			if grid[r][c] == 1:
				count = [0]
				dfs(r, c, count)
				result = max(result, count[0])

	return result

grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
# print(maxAreaOfIslands(grid))

import collections
def BadOranges(grid):
	# dfs + bfs
	queue = collections.deque()
	rows = len(grid)
	columns = len(grid[0])
	orangeNum = 0
	badOrangeNum = 0
	if orangeNum == 0: return 0
	for r in range(rows):
		for c in range(columns):
			if grid[r][c] != 0:
				orangeNum += 1
			if grid[r][c] == 2:
				queue.append((r, c))
				badOrangeNum += 1
	times = 0
	while queue:
		for _ in range(len(queue)):
			popR, popC = queue.popleft()
			
			for i, j in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
				x = popR + i
				y = popC + j

				if 0 <= x < rows and 0 <= y < columns and grid[x][y] == 1:
					grid[x][y] = 2
					queue.append((x, y))
					badOrangeNum += 1

		times += 1
	return times - 1 if badOrangeNum == orangeNum else -1


# grid = [[2,1,1],[1,1,0],[0,1,1]]
# print(BadOranges(grid))


# 207. 课程表
import collections
def ifFinish(classNums, orders):
	courses = collections.defaultdict(list)
	degrees = [0] * classNums
	listCount = 0
	for nxt, pre in orders:
		courses[pre].append(nxt)
		degrees[nxt] += 1


	queue = collections.deque()
	
	for i in range(classNums):
		if degrees[i] == 0:
			queue.append(i)
			listCount += 1


	while queue:
		node = queue.popleft()
		for nxt in courses[node]:
			degrees[nxt] -= 1
			if degrees[nxt] == 0:
				queue.append(nxt)
				listCount += 1

	return listCount == classNums

print('# 207. 课程表')

print(ifFinish(2, [[1,0]]) == True)



# classNums = 5
# orders = [[1, 0], [2, 1], [3, 1], [4, 2], [4, 3]]
# print(ifFinish(classNums, orders) == True)
# classNums = 5
# orders = [[1, 0], [2, 1], [3, 1], [3, 0], [4, 2], [4, 3]]
# print(ifFinish(classNums, orders) == True)
# classNums = 5
# orders = [[1, 0], [2, 1], [3, 1], [3, 0], [4, 2], [4, 3], [1, 4]]
# print(ifFinish(classNums, orders) == False)


# print(ifFinish(2, [[1,0],[0,1]]) == False)


# 208. 实现 Trie (前缀树)

class Node:
	def __init__(self):
		self.son = [None]*26
		self.ifEnd = False

class Trie:
	def __init__(self):
		self.root = Node()

	def insert(self, word):
		proot = self.root
		for i, x in enumerate(word):
			if not proot.son[ord(x)-ord('a')]:
				proot.son[ord(x)-ord('a')] = Node()
			proot = proot.son[ord(x)-ord('a')]
			if i == len(word)-1:
				proot.ifEnd = True

	def search(self, word):
		proot = self.root
		for i, x in enumerate(word):
			if proot.son[ord(x)-ord('a')]:
				proot = proot.son[ord(x)-ord('a')]
			else:
				return False
		return proot.ifEnd

	def startWith(self, word):
		proot = self.root
		for i, x in enumerate(word):
			if proot.son[ord(x)-ord('a')]:
				proot = proot.son[ord(x)-ord('a')]
			else:
				return False
		return True

print('208. 实现 Trie (前缀树)')

trie = Trie()
trie.insert("apple")
print(trie.search("apple") == True)
print(trie.search('app') == False)
print(trie.startWith('app') == True)
trie.insert("app")
print(trie.search('app') == True)



# 46. 全排列
'''
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
'''
def rankToList(nums):

	path = []
	result = []
	n = len(nums)
	flag = [False] * n
	def dfs(i):
		if len(path) == n:
			result.append(path.copy())

		for j in range(n):
			if not flag[j]:
				path.append(nums[j])
				flag[j] = True
				dfs(j+1)
				path.pop(-1)
				flag[j] = False
	dfs(0)
	return result
nums = [1,2,3]
print(rankToList(nums))


# 78. 子集
def subsets(nums):
	# 2^n
	path = []
	result = []
	n = len(nums)
	def dfs(i):
		if i >= n:
			result.append(path.copy())
			return
		path.append(nums[i])
		dfs(i+1)
		path.pop(-1)
		dfs(i+1)

	dfs(0)
	return result
nums = [1,2,3]
result = subsets(nums)
result.sort(key=lambda x : len(x))
# print(result)

# 77. 组合
def combine(n, k):
	path = []
	result = []
	def dfs(i):
		if i > n:
			return
		if len(path) == k:
			result.append(path.copy())
			return

		path.append(i+1)
		dfs(i+1)
		path.pop(-1)
		dfs(i+1)
	dfs(0)
	return result
print('# 77. 组合')
# print(combine(4, 2))


# 17. 电话号码的字母组合



# 39. 组合总和
def combinationSum(candidates, target):
	'''
	输入：candidates = [2,3,6,7], target = 7
	输出：[[2,2,3],[7]] 
	可重复利用 	      
	'''
	path = []
	result = []
	n = len(candidates)
	def dfs(i, summ):

		if summ == target:
			result.append(path.copy())
			return 
		if n == i or summ > target:
			return 
		path.append(candidates[i])
		dfs(i, summ+candidates[i])
		path.pop(-1)
		dfs(i+1, summ)

	dfs(0, 0)
	return result

candidates = [2,3,6,7]
target = 7
# print(combinationSum(candidates, target))


# 22. 括号生成
def generate(n):
	'''
	输入：n = 3
	输出：["((()))","(()())","(())()","()(())","()()()"] '''
	path = []
	result = []
	def dfs(leftRest, rightRest):
		if leftRest > rightRest:
			return 
		if leftRest == 0 and rightRest == 0:
			result.append(''.join(path.copy()))
			return 

		if leftRest > 0:
			path.append('(')
			dfs(leftRest-1, rightRest)
			path.pop(-1)

		if rightRest > leftRest:
			path.append(')')
			dfs(leftRest, rightRest-1)
			path.pop(-1)

	dfs(n, n)
	return result

print(generate(3))


# 79. 单词搜索

def searchWord(board, word):
	rows = len(board)
	columns = len(board[0])
	used = set()

	def dfs(r, c, i):
		if i == len(word)-1 and board[r][c] == word[-1]:
			return True

		if board[r][c] == word[i]:
			used.add((r, c))
			flagResult = False
			for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
				x = r + dx
				y = c + dy
				if 0 <= x < rows and 0 <= y < columns and (x,y) not in used:
					flagResult = dfs(x, y, i+1)
					if flagResult:
						return True

			used.remove((r, c))
		else:
			return False

	
	flag = False
	for r in range(rows):
		for c in range(columns):
			if dfs(r, c, 0):
				return True

	return flag

print('# 79. 单词搜索')
print(searchWord(board = [['A','B','C','E'],['S','F','C','S'],['M','D','E','E']], word = "ABCCED"))
# print(searchWord(board = [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']], word = "SEE"))
# print(searchWord(board = [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']], word = "ABCB"))
print('# 79. 单词搜索')


# 35. 搜索插入位置
def searchIndex(nums, target):
	'''
	nums = [1,3,5,6], target = 5
	'''
	n = len(nums)
	left, right = 0, n-1

	while left <= right:
		mid = (left + right) // 2
		if nums[mid] == target:
			return mid
		elif nums[mid] > target:
			right = mid - 1
		else:
			left = mid + 1

	return left

print(searchIndex([1,3,5,6], 5) == 2)
print(searchIndex([1,3,5,6], 2) == 1)
print(searchIndex([1,3,5,6], 7) == 4)




# 74. 搜索二维矩阵
def searchMatrix(matrix, target):
	rows = len(matrix)
	columns = len(matrix[0])
	left = 0
	right = rows*columns-1
	while left <= right:
		mid = (left+right) // 2
		print(mid, mid//columns, mid%columns)
		if matrix[mid//columns][mid%columns] == target:
			return True
		elif matrix[mid//columns][mid%columns] > target:
			right = mid - 1 
		else:
			left = mid + 1

	return False


print(searchMatrix([[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3))


# 34. 在排序数组中查找元素的第一个和最后一个位置

def searchLeft(nums, target):
	n = len(nums)
	left, right = 0, n-1
	while left <= right:
		mid = (left+right) // 2
		if nums[mid] >= target:
			right = mid - 1
		else:
			left = mid + 1

	return left

def searchRight(nums, target):
	n = len(nums)
	left, right = 0, n-1
	while left <= right:
		mid = (left+right) // 2
		if nums[mid] <= target:
			left = mid + 1
		else:
			right = mid - 1

	return right


def searchDoubleEnd(nums, target):
	leftResult = searchLeft(nums, target)
	rightResult = searchRight(nums, target)
	if nums[leftResult] == target and nums[rightResult] == target:
		return leftResult, rightResult
	else:
		return [-1, -1]

print(searchDoubleEnd([5,7,7,8,8,10], target = 8))



# 33. 搜索旋转排序数组
def searchRotateArray(nums, target):
	# 对mid分类讨论;

	n = len(nums)
	left, right = 0, n-1

	while left <= right:

		mid = (left + right) // 2
		if nums[mid] == target:
			return mid

		if nums[mid] <= nums[-1]:
			if nums[mid] < target <= nums[-1]:
				left = mid + 1
			else:
				right = mid - 1
		if nums[mid] >= nums[0]:
			if nums[0] <= target < nums[mid]:
				right = mid - 1
			else:
				left = mid + 1
	return -1


# print(searchRotateArray(nums = [4,5,6,7,0,1,2], target = 0))
# print(searchRotateArray(nums = [4,5,6,7,0,1,2], target = 3))
# print(searchRotateArray(nums = [1], target = 0))


def findMinOfRotateArray(nums):
	n = len(nums)
	left, right = 0, n-1
	while left < right:

		mid = (left + right) // 2

		if nums[mid] > nums[-1]:
			left = mid + 1
		else:
			right = mid 

	return nums[right]

# print(findMinOfRotateArray(nums = [3,4,5,1,2]))
# print(findMinOfRotateArray(nums = [4,5,6,7,0,1,2]))
# print(findMinOfRotateArray(nums = [11,13,15,17]))



# 295. 数据流的中位数
def MedianFinder():
	return 




# 20. 有效的括号
def validBrackets(s):
	stack = []
	hashmap = {')':'(', ']':'[', '}':'{'}

	for c in s:
		if c in hashmap:
			popItem = stack.pop(-1)
			if popItem != hashmap[c]:
				return False
		else:
			stack.append(c)
	return True
# print(validBrackets("()"))
# print(validBrackets("()[]{}"))
# print(validBrackets("(]"))
# print(validBrackets("([])"))
# print(validBrackets("([)]"))


# 155. 最小栈
class minStack:

	def __init__(self):
		self.stack = []
		self.minstack = [inf]
		self.size = 0



	def push(self, value):
		
		if value < self.getMin():
			self.minstack.append(value)
		else:
			self.minstack.append(self.getMin())

		self.stack.append(value)
		self.size += 1


	def pop(self):
		if self.size > 0:
			popItem = self.stack[self.size-1]
			self.size -= 1
			return popItem
		else:
			return None

		


	def top(self):
		if self.size > 0:
			return self.stack[self.size-1]
		else:
			return None


	def getMin(self):
		
		return self.minstack[self.size]


# class Stack:
# 	def __init__(self, capacity):
# 		self.stack = [0] * capacity
# 		self.capacity = capacity
# 		self.size = 0	

# 	def expand(self):
# 		self.newStack = [0] * (2 * self.capacity)
# 		for i in range(self.size):
# 			self.newStack[i] = self.stack[i]
# 		self.stack = self.newStack

# 	def push(self, val):
# 		if self.size > self.capacity:
# 			self.expand()
# 		self.stack[self.size] = val
# 			self.size += 1

# 	def pop(self):
# 		return 

# 	def top(self):
# 		return

# stack = Stack()
# stack.push(1)
# stack.push(2)
# stack.push(3)
# print(stack.top())
# stack.push(4)
# print(stack.pop())
# print(stack.pop())
# print(stack.pop())
# print(stack.pop())


# print(' 155. 最小栈')

# stack = minStack()
# stack.push(-2)
# stack.push(0)
# stack.push(-3)
# print(stack.getMin())
# print(stack.pop())
# print(stack.size)
# print(stack.top())
# print(stack.getMin())



# 394. 字符串解码
def decodeString(s):
	curNum = 0
	curStr = ''
	stack = []
	for c in s:
		if c.isdigit():
			curNum = curNum*10 + int(c)

		elif c == '[':
			stack.append((curNum, curStr))
			curNum = 0
			curStr = ''

		elif c == ']':
			preNum, preStr = stack.pop(-1)
			curStr = preStr + curStr * preNum

		else:
			curStr += c
	return curStr

print(decodeString("3[a]2[bc]"))
print(decodeString("3[a2[c]]"))
print(decodeString("3[a2[c5[r4[e]]]]"))
print(decodeString("3[a]2[bc]") == "aaabcbc")
print(decodeString("3[a2[c]]") == "accaccacc")


# 739. 每日温度
def dailyTemperuature(temperatures):
	# 递减的栈，一直加小的, 栈顶是一直放元素的那个方向，应该要随着栈顶越来越小； 并且不能重复，重复的需要去除，因为还要考虑 2 2 4，第一个2的next应该是4，而不是第二个2
	# 如果不去除重复的, 第一个2的答案就是1了，而不是2
	# 存下标
	# 

	n = len(temperatures)
	result = [0] * n
	stack = []
	for i in range(n-1, -1, -1):
		while stack and temperatures[i] >= temperatures[stack[-1]]:
			stack.pop(-1)
		if stack:
			result[i] = stack[-1] - i
		stack.append(i)
	return result

# print('739. 每日温度')
# print(dailyTemperuature([73,74,75,71,69,72,76,73]))
# print(dailyTemperuature([73,74,75,71,69,72,76,73]) == [1,1,4,2,1,1,0,0])


# 数组中的第K个最大元素
import random
def quickSortHelper(nums, left, right):
	# [left, right]
	if left >= right:
		return

	pati = random.randint(left, right) # randomInt 闭区间
	baseline = nums[pati]
	nums[left], nums[pati] = nums[pati], nums[left]

	l, r = left+1, right

	while l <= r:

		while l <= r and nums[l] <= baseline:
			l += 1
		while l <= r and nums[r] >= baseline:
			r -= 1

		if l <= r:
			nums[l], nums[r] = nums[r], nums[l]
			l += 1
			r -= 1

	nums[left], nums[r] = nums[r], nums[left]

	quickSortHelper(nums, left, r-1)
	quickSortHelper(nums, r+1, right)


def quickSort(nums):
	quickSortHelper(nums, 0, len(nums)-1)


import random
def quickSortHelper2(nums, left, right):
	# [left, right]
	if left >= right:
		return

	pati = random.randint(left, right) # randomInt 闭区间
	baseline = nums[pati]
	nums[left], nums[pati] = nums[pati], nums[left]

	l, r = left+1, right

	while l <= r:

		while l <= r and nums[l] < baseline:
			l += 1
		while l <= r and nums[r] > baseline:
			r -= 1

		if l <= r:
			nums[l], nums[r] = nums[r], nums[l]
			l += 1
			r -= 1

	nums[left], nums[r] = nums[r], nums[left]

	quickSortHelper2(nums, left, r-1)
	quickSortHelper2(nums, r+1, right)


def quickSort2(nums):
	quickSortHelper2(nums, 0, len(nums)-1)


lis = [i for i in range(1000, 0, -1)]
# print(lis)
# quickSort2(lis)
# print(lis)



lis = [1 for i in range(1000, 0, -1)]
# print(lis)
# quickSort2(lis)
# print(lis)

import random

def quickselect(nums, left, right):
	# [left. right]
	if left > right:
		return -1 
	pati = random.randint(left, right)
	baseline = nums[pati]
	nums[left], nums[pati] = nums[pati], nums[left]

	l, r = left+1, right

	while l <= r:
		while l <= r and nums[l] > baseline:
			l += 1
		while l <= r and nums[r] < baseline:
			r -= 1
		if l <= r:
			nums[l], nums[r] = nums[r], nums[l]
			l += 1
			r -= 1
	nums[left], nums[r] = nums[r], nums[left]
	return r

def biggestKItem(nums, k):
	# quickselect 顺便排序
	k = k - 1 
	left = 0
	right = len(nums) - 1

	while left <= right:

		p = quickselect(nums, left, right)

		if p == k:
			return nums[p]
		elif p > k:
			right = p - 1
		else:
			left = p + 1

print('biggestKItem : 第K大的数')
print(biggestKItem([3,2,1,5,6,4], k = 2))
print(biggestKItem([3,2,3,1,2,4,5,5,6], k = 4))

import collections
def topKElement(nums, k):
	counts = collections.defaultdict(int)
	result = []

	for n in nums:
		counts[n] += 1
	reverse = collections.defaultdict(list)

	for item, count in counts.items():
		reverse[count].append(item)
	maxCount = max(reverse.keys())
	print(reverse)
	print(maxCount)

	index = maxCount
	while index > 0 and k > 0:
		if index in reverse.keys() and reverse[index]:
			result.append(reverse[index][-1])
			reverse[index].pop(-1)
			k -= 1
		else:
			index -= 1
	return result
print(topKElement(nums = [1,2,1,2,1,2,3,1,3,2], k=2))

print(topKElement(nums = [1,1,1,2,2,3], k = 2))
print(topKElement(nums = [1], k = 1))



from math import inf
# 121. 买卖股票的最佳时机
def maxProfit(prices):
	curMin = inf
	maxProfit = -inf
	n = len(prices)
	for i in range(n):
		curMin = min(curMin, prices[i])
		maxProfit = max(maxProfit, prices[i] - curMin)

	return maxProfit

# print(maxProfit([7,1,5,3,6,4]))
# print(maxProfit([7,1,5,3,6,4]) == 5)
# print(maxProfit([7,6,4,3,1]))
# print(maxProfit([7,6,4,3,1]) == 0)




def maxProfit2(prices):
	'''
	给你一个整数数组 prices ，其中 prices[i] 表示某支股票第 i 天的价格。

	在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。然而，你可以在 同一天 多次买卖该股票，但要确保你持有的股票不超过一股。

	返回 你能获得的 最大 利润 。
	'''
	n = len(prices)
	result = 0
	for i in range(1, n):
		if prices[i] > prices[i-1]:
			result += (prices[i] - prices[i-1])

	return result


# print(maxProfit2(prices = [7,1,5,3,6,4]))
# print(maxProfit2(prices = [7,1,5,3,6,4]) == 7)

# 55. 跳跃游戏
def canJump(nums):
	lastIndex = 0
	for i in range(len(nums)):
		if i > lastIndex:
			return False
		lastIndex = max(lastIndex, i + nums[i])

	return True

print(canJump(nums = [2,3,1,1,4]))
print(canJump(nums = [3,2,1,0,4]))

print(canJump([2,0,0]))



# 45. 跳跃游戏 II
def jump(nums):
	result = 0
	curLastIndex = 0
	maxLastIndex = 0
	n = len(nums)
	for i in range(n-1):
		maxLastIndex = max(maxLastIndex, i+nums[i])
		if i == curLastIndex:
			curLastIndex = maxLastIndex
			result += 1

	return result



print(jump([2,3,1,1,4]))
print(jump([2,3,1,1,4]) == 2)


print(jump([2,3,0,1,4]))
print(jump([2,3,0,1,4]) == 2)



# 763. 划分字母区间
import collections
def splitLetter(letter):
	# 找到这个字母的最后一个index
	lastIndexs = collections.defaultdict(int)
	for i, x in enumerate(letter):
		lastIndexs[x] = i
	# print(lastIndexs)

	count = 0
	result = []
	maxLastIndex = 0
	for i, x in enumerate(letter):
		count += 1
		maxLastIndex = max(maxLastIndex, lastIndexs[x])
		if i == maxLastIndex:
			result.append(count)
			count = 0
	return result


print(splitLetter("ababcbacadefegdehijhklij"))
print(splitLetter("ababcbacadefegdehijhklij") == [9,7,8])



# 70. 爬楼梯
def climb(n):
	F = [1] * (n+1)

	for i in range(1, n):
		F[i+1] = F[i] + F[i-1]

	return F[n]
# print(climb(4))	
# print(climb(3))
# print(climb(2))


# 118. 杨辉三角
def triangle(numRows):
	result = [[1] * (i+1) for i in range(numRows)]
	for i in range(2, numRows):
		for j in range(1, i):
			result[i][j] = result[i-1][j-1] + result[i-1][j]
	return result


print(triangle(5))

# 198. 打家劫舍
'''
	示例 1：

	输入：[1,2,3,1]
	输出：4
	解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
	     偷窃到的最高金额 = 1 + 3 = 4 。
	示例 2：

	输入：[2,7,9,3,1]
	输出：12
	解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
	     偷窃到的最高金额 = 2 + 9 + 1 = 12 。

'''
def rob(nums):
	n = len(nums)
	
	f0 = f1 = 0
	f = 0
	for i in range(n):
		f = max(nums[i]+f0, f1)
		f0 = f1
		f1 = f
	return f


print(rob([2,7,9,3,1]))
print(rob([2,7,9,3,1]) == 12)


'''
你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。

 
示例 1：

输入：nums = [2,3,2]
输出：3
解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
示例 2：

输入：nums = [1,2,3,1]
输出：4
解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
     偷窃到的最高金额 = 1 + 3 = 4 。
     '''
def rob2(nums):
	def robb(nums):
		f0 = f1 = f = 0
		n = len(nums)
		for i in range(n):
			f = max(f0+nums[i], f1)
			f0 = f1 
			f1 = f
		return f
	return max(nums[0] + robb(nums[2:-1]), robb(nums[1:]))

print(rob2([2,3,2]) == 3)
print(rob2([1,2,3,1]) == 4)


# 322. 零钱兑换
def coinChange(coins, amount):
	# 创建好数组:
	# 注意边界: n+1这样就可以防止i越界
	# a也需要多创建一个，相当于它的全周期或者可能出现的条件吧，所以遍历的时候也是
	# 像这里i = 0的就一般都是初始化条件，可以自己直接把值落下来
	# 注意初始化，初始化一些base condition
	# 找到递推公式，去进行递推
	# 这里就是dp[i][a]上面的值 with 左边的值进行比较

	n = len(coins)

	dp = [[inf] * (amount+1) for _ in range(n + 1)]
	for i in range(n):
		dp[i][0] = 0
		for a in range(amount+1):

			if a < coins[i]:
				dp[i+1][a] = dp[i][a]
			else:
				dp[i+1][a] = min(dp[i][a], dp[i+1][a-coins[i]]+1)
	print(dp)
	F = dp[n][-1]
	return F if F < inf else -1

print(coinChange(coins = [1, 2, 5], amount = 11) == 3)

def coinChange2(coins, amount):
	# 这里就是dp[i][a]上面的值 with 左边的值进行比较
	# 单数组就是可以覆盖，上面的值变成了自己上次循环本身的值，左边的值进行比较依旧是左边的值
	# 初始化依旧不变
	n = len(coins)

	dp = [inf] * (amount+1)
	dp[0] = 0
	for i in range(n):
		for a in range(coins[i], amount+1):
			dp[a] = min(dp[a], dp[a-coins[i]]+1)
	print(dp)
	F = dp[-1]
	return F if F < inf else -1

print(coinChange2(coins = [1, 2, 5], amount = 11) == 3)


# 279. 完全平方数

def numSquares(n):
	# [1, 4, 9]
	squares = [(i+1)**2 for i in range(int(sqrt(n)))]
	counts = len(squares)
	dp = [[inf] * (n+1) for _ in range(counts+1)]
	print(squares)
	for i in range(counts):
		dp[i][0] = 0
		for a in range(n+1):
			if a < squares[i]:
				dp[i+1][a] = dp[i][a]
			else:
				dp[i+1][a] = min(dp[i][a], dp[i+1][a-squares[i]]+1)
	F = dp[-1][-1]
	return F if F < inf else -1
print(numSquares(12))
print(numSquares(13))



# 139. 单词拆分
def splitWord(s, wordDict):
	maxlen = len(max(wordDict, key = lambda x : len(x))) # max返回的不是len的值, 而是原始值
	n = len(s)
	dp = [False] * (n+1)
	dp[0] = True

	for i in range(n):
		for j in range(i, max(-1, i-maxlen), -1):
			if s[j:i+1] in wordDict:
				if dp[j] == True:
					dp[i+1] = True
	return dp[-1]

print('139. 单词拆分')
print(splitWord(s = "leetcode", wordDict = ["leet", "code"]) == True)
print(splitWord(s = "applepenapple", wordDict = ["apple", "pen"]) == True)
print('139. 单词拆分')



# 300. 最长递增子序列
def LIS(nums):
	n = len(nums)
	F = [1] * n
	for i in range(n):
		for j in range(i):
			if nums[j] < nums[i]:
				F[i] = max(F[i], F[j]+1)
	return max(F)

def LIS2(nums):
	def searchInsert(nums, target):
		left, right = 0, len(nums)-1
		while left <= right:
			mid = (left + right) // 2
			if nums[mid] == target:
				return mid
			elif nums[mid] < target:
				left = mid + 1
			else:
				right = mid - 1
		return left


	n = len(nums)
	result = [nums[0]]

	for i in range(1, n):
		insertIndex = searchInsert(result, nums[i])
		if insertIndex == len(result):
			result.append(nums[i])
		else:
			result[insertIndex] = nums[i]
	return len(result)

print(LIS([10,9,2,5,3,7,101,18]) == 4)
print(LIS([0,1,0,3,2,3]) == 4)
print(LIS([7,7,7,7,7,7,7]) == 1)

print(LIS2([10,9,2,5,3,7,101,18]) == 4)
print(LIS2([0,1,0,3,2,3]) == 4)
print(LIS2([7,7,7,7,7,7,7]) == 1)


# 152. 乘积最大子数组
def maxProductOfSubArray(nums):
	globalMax = nums[0]
	curMax = nums[0]
	curMin = nums[0]
	n = len(nums)
	for i in range(1, n):
		tempMin = min(curMin * nums[i], curMax * nums[i], nums[i])
		curMax = max(curMax * nums[i], curMin * nums[i], nums[i])
		curMin = tempMin
		globalMax = max(globalMax, curMin, curMax)

	return globalMax

print('乘积最大子数组')
print(maxProductOfSubArray(nums = [2,3,-2,4]))
print(maxProductOfSubArray(nums = [-2,0,-1]))

# 416. 分割等和子集

def splitLetterWithSameSummary(nums):
	summary = sum(nums)
	if summary % 2 == 1:
		return False
	n = len(nums)
	target = summary // 2
	# 0-1背包

	dp = [[False] * (target + 1) for _ in range(n+1)]
	dp[0][0] = True # dp[0][1-11]这些相当于 有target,但是没有元素,所以肯定找不到和为target的items

	for i in range(n):
		for t in range(target+1):
			if t < nums[i]:
				dp[i+1][t] = dp[i][t] 
			else:
				# 看上和左上:
				dp[i+1][t] = dp[i][t] or dp[i][t-nums[i]]
	print(dp)
	return dp[-1][-1]
print(splitLetterWithSameSummary([1,5,11,5]))
# print(splitLetterWithSameSummary([1,2,3,5]))

# 
def longestPam(s):
	stack = []
	n = len(s)
	result = [0] * n
	for i in range(n):
		if s[i] == '(':
			stack.append(i)
		else:
			j = stack.pop(-1)
			result[i] = 1
			result[j] = 1

	print(result)
	count = 0
	maxCount = 0
	for i in range(n):
		if result[i] == 1:
			count += 1
		else:
			maxCount = max(maxCount, count)
			count = 0
	maxCount = max(maxCount, count)
	return maxCount

print(longestPam(s = "(()"))


def uniquePaths(m, n):
	'''
	一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

	机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

	问总共有多少条不同的路径？'''

	# dp = [[0] * (n) for _ in range(m)]
	# for i in range(m):
	# 	dp[i][0] = 1
	# for j in range(n):
	# 	dp[0][j] = 1
	# for i in range(1, m):
	# 	for j in range(1, n):
	# 		dp[i][j] = dp[i][j-1] + dp[i-1][j]
	# return dp[m-1][n-1]

	# dp = [[0] * (n+1) for _ in range(m+1)]
	# dp[1][1] = 1
	# for i in range(1, m+1):
	# 	for j in range(1, n+1):
	# 		if i != 1 or j != 1:
	# 			dp[i][j] = dp[i-1][j] + dp[i][j-1]
	# return dp[-1][-1]

	dp = [0] * (n+1) 
	dp[1] = 1
	for i in range(1, m+1):
		for j in range(1, n+1):
			if i != 1 or j != 1:
				dp[j] = dp[j] + dp[j-1]

	return dp[-1]


print(uniquePaths(m = 3, n = 7))
print(uniquePaths(m = 3, n = 2))


#  64. 最小路径和
def minPathSum(grid):
	m = len(grid)
	n = len(grid[0])
	# dp = [[inf] * (n+1) for _ in range(m+1)]
	# dp[1][1] = grid[0][0]
	# for i in range(m):
	# 	for j in range(n):
	# 		if not (i == 0 and j == 0):
	# 			dp[i+1][j+1] = min(dp[i+1][j], dp[i][j+1]) + grid[i][j]
	# return dp[-1][-1]

	dp = [inf] * (n+1) 
	dp[1] = grid[0][0]
	for i in range(m):
		for j in range(n):
			if not (i == 0 and j == 0):
				# dp[i+1][j+1] = min(dp[i+1][j], dp[i][j+1]) + grid[i][j]
				dp[j+1] = min(dp[j], dp[j+1]) + grid[i][j]

	return dp[-1]

print(minPathSum(grid = [[1,3,1],[1,5,1],[4,2,1]]))
print(minPathSum(grid = [[1,2,3],[4,5,6]]))




# 1143. 最长公共子序列 
def LCS(text1, text2):
	m = len(text1)
	n = len(text2)
	dp = [[0] * (n+1) for _ in range(m+1)]

	for i in range(m):
		for j in range(n):
			if text1[i] == text2[j]:
				# 左上角
				dp[i+1][j+1] = dp[i][j] + 1
			else:
				# max 左 and 上 -> 
				dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
	print(dp)
	return dp[m][n]

	'''
	[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 3]]
	'''

	# 当我们做一维矩阵的时候：注意我们的推来自三个: 左上(when equal) 左 & 上; 当我们在用左上的时候,它和左会重合; 想象一下: 本来j+1要用到左上的, 那么下标是j,但是j已经在上一次loop被覆盖了; 所以左上(上一轮i外循环)要单独保存

def LCS2(text1, text2):
	m = len(text1)
	n = len(text2)
	dp = [0] * (n+1)

	for i in range(m):
		leftup = dp[0]
		print(dp)
		for j in range(n):
			temp = dp[j+1]
			if text1[i] == text2[j]:
				# 左上角
				dp[j+1] = leftup + 1
			else:
				# max 左 and 上 -> 
				dp[j+1] = max(dp[j+1], dp[j])
			leftup = temp
	print(dp)
	return dp[n]

	'''
	[[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 2, 2], [0, 1, 2, 2], [0, 1, 2, 3]]
	'''

print(LCS2(text1 = "abcde", text2 = "ace" ))
# print(LCS(text1 = "abc", text2 = "abc"))
# print(LCS(text1 = "abc", text2 = "def"))

# 72. 编辑距离
def minDistance(word1, word2):
	'''
	给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数。

	你可以对一个单词进行如下三种操作：

	插入一个字符
	删除一个字符
	替换一个字符
	
	'''
	m = len(word1)
	n = len(word2)
	dp = [[0] * (n+1) for _ in range(m+1)]
	for j in range(n+1):
		dp[0][j] = j
	for i in range(m):
		dp[i+1][0] = i+1
		for j in range(n):
			if word1[i] == word2[j]:
				dp[i+1][j+1] = dp[i][j]
			else:
				dp[i+1][j+1] = min(dp[i+1][j], dp[i][j+1], dp[i][j]) + 1
	print(dp)
	# return dp[m][n]


	m = len(word1)
	n = len(word2)
	dp = [0] * (n+1)
	for j in range(n+1):
		dp[j] = j
	for i in range(m):
		leftup = dp[0]
		dp[0] = i+1
		for j in range(n):
			temp = dp[j+1]
			if word1[i] == word2[j]:
				dp[j+1] = leftup
			else:
				dp[j+1] = min(dp[j], dp[j+1], leftup) + 1
			leftup = temp
	return dp[n]

print(minDistance(word1 = "horse", word2 = "ros"))


# 5. 最长回文子串
def longestPalidram(s):
	n = len(s)
	maxLength = 0
	if n % 2 == 1:
		for i in range(n):
			l, r = i, i
			while l >= 0 and r <= n-1:
				if s[l] != s[r]:
					break
				l -= 1
				r += 1
			maxLength = max(maxLength, r-l-1)

	else:
		for i in range(n-1):
			l, r = i, i+1
			while l >= 0 and r <= n-1:
				if s[l] != s[r]:
					break
				l -= 1
				r += 1
			maxLength = max(maxLength, r-l-1)

	return maxLength

def longestPalidram2(s):
	return 


print(longestPalidram("babad"))
print(longestPalidram("cbbd"))

# 5. 最长回文子串 dp做法

def NumPalidram(s):
	# 有多少个回文子串
	n = len(s)
	count = 0
	dp = [[False] * n for _ in range(n)] 
	for i in range(n):
		dp[i][i] = True
	for i in range(n):
		for j in range(i, n):
			if j - i >= 2:
				dp[i][j] = (s[i] == s[j]) and dp[i+1][j-1]

			else:
				dp[i][j] = (s[i] == s[j])
			if dp[i][j]:
				count += 1

	print(dp)
	return count

print('# 5. 最长回文子串 dp做法')
print(NumPalidram('babad'))
print('# 5. 最长回文子串 dp做法')


def longestPalidram2(s):
	# 有多少个回文子串
	n = len(s)
	maxLength = 0
	dp = [[False] * n for _ in range(n)] 
	for i in range(n):
		dp[i][i] = True
	for i in range(n):
		for j in range(i, n):
			if j - i >= 2:
				dp[i][j] = (s[i] == s[j]) and dp[i+1][j-1]

			else:
				dp[i][j] = (s[i] == s[j])
			if dp[i][j]:
				maxLength = max(maxLength, j-i+1)

	return maxLength

print(longestPalidram2("babad"))
print(longestPalidram2("cbbd"))



# 131. 分割回文串
def splitPalidram(s):
	result = []
	path = []
	n = len(s)
	def dfs(i):
		if i == n:
			result.append(path.copy())
			return 

		for j in range(i, n):
			t = s[i:j+1]
			if t == t[::-1]:
				path.append(t)
				dfs(j+1)
				path.pop(-1)
	dfs(0)
	return result

print(splitPalidram(s = "aab"))
print(splitPalidram(s = "a"))



# 136. 只出现一次的数字
def oneTimeNumber(nums):
	# a ^ 0 = a
	# a ^ a = 0
	# a ^ b = b ^ a
	result = nums[0]
	for n in nums[1:]:
		result = result ^ n
	return result

print(oneTimeNumber([1, 1, 2, 2, 3, 6, 6, 7, 7]))
print(oneTimeNumber([1, 2, 1, 2, 6, 6, 0, 7, 7]))


# 169. 多数元素
def MostElement(nums):
	most = nums[0]
	count = 1

	for x in nums[1:]:
		if x == most:
			count += 1
		else:
			count -= 1
			if count == 0:
				most = x
				count = 1
	return most

print(MostElement([3,2,3]))
print(MostElement([2,2,1,1,1,2,2]))


# 75. 颜色分类
def sortColors(nums):
	n = len(nums)
	zeroIndex = 0
	for i in range(n):
		if nums[i] == 0:
			nums[zeroIndex], nums[i] = nums[i], nums[zeroIndex]
			zeroIndex += 1

	twoIndex = n-1
	for j in range(n-1, -1, -1):
		if nums[j] == 2:
			nums[twoIndex], nums[j] = nums[j], nums[twoIndex]
			twoIndex -= 1


def sortColors2(nums):
	zeroIndex = oneIndex = 0

	for i, x in enumerate(nums):
		nums[i] = 2
		if x == 1:
			nums[oneIndex] = 1
			oneIndex += 1
		if x == 0:
			nums[oneIndex] = 1
			oneIndex += 1
			nums[zeroIndex] = 0
			zeroIndex += 1


nums = [2,0,2,1,1,0,0,1,0,0,1,2,0]
print(nums)
sortColors2(nums)
print(nums)


# 下一个排列
def nextList(nums):
	# [1, 2, 3] -> [1, 3, 2] -> [2, 1, 3] -> [2, 3, 1] -> [1, 2, 3]
	candi = -1
	n = len(nums)
	for i in range(n-2, -1, -1):
		if nums[i] < nums[i+1]:
			candi = i
			break
	print(candi)
	
	for i in range(n-1, candi, -1):
		if nums[i] > nums[candi]:
			nums[i], nums[candi] = nums[candi], nums[i]
			break
	
	l, r = candi+1, n-1
	while l < r:
		nums[l], nums[r] = nums[r], nums[l]
		l += 1
		r -= 1

print(' 下一个排列')
nums = [1, 2, 3]
print(nums)
nextList(nums)
print(nums)
nextList(nums)
print(nums)
nextList(nums)
print(nums)
print(' 下一个排列')



def findUnUniqueNumber(nums):
	# 想象成一个从0开始的环
	fast = slow = 0
	while True:
		fast = nums[nums[fast]]
		slow = nums[slow]
		if slow == fast:
			head = 0
			while slow != head:
				slow = nums[slow]
				head = nums[head]
			break
	return slow


print(findUnUniqueNumber(nums = [1,3,4,2,2]))

print(findUnUniqueNumber(nums = [3,1,3,4,2]))

print(findUnUniqueNumber(nums = [3,3,3,3,3]))




