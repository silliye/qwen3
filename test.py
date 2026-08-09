'''
给你两个字符串 s 和 t ，统计并返回在 s 的 子序列 中 t 出现的个数。

测试用例保证结果在 32 位有符号整数范围内。
'''


def count(s, t):
    n1 = len(s)
    n2 = len(t)
    dp = [[0] * (n2+1) for _ in range(n1+1)]
    dp[0][0] = 1
    for i in range(n1):
        dp[i][0] = 1
    for i in range(n1):
        for j in range(n2):
            if s[i] == t[j]:
                dp[i+1][j+1] = dp[i][j] + dp[i][j+1]
            else:
                dp[i+1][j+1] = dp[i][j+1]
    return dp[-1][-1]


print(count("rabbbit", t = "rabbit"))
print(count("babgbag", t = "bag"))

