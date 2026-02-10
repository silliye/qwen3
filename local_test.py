import torch
from torch import Tensor

def gen_causal_mask(seq_mask:Tensor):
        # seq_mask [B, seq_decoder]  [1 1 1 0 0 ]

        # MAKE 下三角矩阵 [B, 8, seq, seq]  8 = nums_of_query_heads
        BatchSize, seq = seq_mask.shape[0], seq_mask.shape[1]
        # [seq, seq]
        mask = torch.tril(torch.ones(seq, seq))  # 下面是用的mask_fill_(bool_tensor)

        return torch.reshape(mask, [1, 1, seq, seq]).expand(BatchSize, 8, -1, -1)


t = [[1, 1, 1, 1, 0], [1, 1, 0, 0 ,0]]


x = Tensor(t)

# mask = gen_causal_mask(x)

# # print(mask)
# import numpy as np
# c = np.cos(12)
# l = []
# l.append(c)
# z = []
# z.append(l)
# print(Tensor(z))


a = Tensor([True, True, True, False])
print(a.mean())