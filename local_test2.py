class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    @staticmethod
    def mergeTwoLists(list1, list2 ):
        dummy = ListNode(-1, None)
        index = dummy
        l1, l2 = list1, list2
        while l1 and l2:
            if l1.val < l2.val:
                index.next = ListNode(l1.val)
                l1 = l1.next
            else:
                index.next = ListNode(l2.val)
                l2 = l2.next
            index = index.next
        print(l1, l2)
        while l1:
            index.next = l1
            
        while l2:
            index.next = l2
            
        
        return dummy.next

a = ListNode(3, None)
b = ListNode(6, None)

c = Solution.mergeTwoLists(a, b)
print(c)
