import torch

from math import *
import torch.nn as nn

from torch import Tensor

class MultiHeadAttention(nn.Module):
	def __init__(self, head_dim, head_num, dropout_rate):
		super().__init__()
		self.head_dim = head_dim
		self.head_num = head_num

		self.proj = nn.Linear(head_dim*head_num, head_dim*head_num*3, bias=False)
		self.o_proj = nn.Linear(head_dim*head_num, head_dim*head_num, bias=False)

		self.dropout = nn.Dropout(dropout_rate)


	def forward(self, X:Tensor, mask=None):
		batchsize, seqlen, dim = X.shape

		qkv = self.proj(X).view(batchsize, seqlen, 3, self.head_num, self.head_dim).permute(2, 0, 3, 1, 4)
		# [B, H, seq, dim]
		querys, keys, values = qkv.unbind(0)

		#

		attention_score = torch.matmul(querys, keys.transpose(-1, -2)) / sqrt(self.head_dim)
		if mask is not None:
			
			# mask [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]
			# [batch, seq] -> [batch, 1, 1, seq]
			mask = mask.unsqueeze(1).unsqueeze(1)
			attention_score = attention_score.masked_fill(mask == 0, -1e9)


		attention_score = torch.softmax(attention_score, dim=-1)
		attention_score = self.dropout(attention_score)

		O = torch.matmul(attention_score, values).transpose(1, 2).contiguous().view(batchsize, seqlen, self.head_dim*self.head_num)
		

		return self.o_proj(O)


class MultiQueryAttention(nn.Module):
	def __init__(self, head_dim, head_num, dropout_rate):
		super().__init__()
		self.head_dim = head_dim
		self.head_num = head_num

		self.q_proj = nn.Linear(head_dim*head_num, head_dim*head_num, bias=False)
		self.kv_proj = nn.Linear(head_dim*head_num, head_dim*2, bias=False)
		self.o_proj = nn.Linear(head_dim*head_num, head_dim*head_num, bias=False)

		self.dropout = nn.Dropout(dropout_rate)


	def forward(self, X:Tensor, mask=None):
		batchsize, seqlen, dim = X.shape

		querys = self.q_proj(X).view(batchsize, seqlen, self.head_num, self.head_dim).transpose(1, 2)

		kv = self.kv_proj(X).view(batchsize, seqlen, 2, 1, self.head_dim).permute(2, 0, 3, 1, 4)
		# [B, H, seq, dim]
		keys, values = kv.unbind(0)

		# querys [B, head_num, seq, dim]
		# keys/values [B, 1, seq, dim]
		attention_score = torch.matmul(querys, keys.transpose(-1, -2)) / sqrt(self.head_dim)
		if mask is not None:
			
			# mask [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]
			# [batch, seq] -> [batch, 1, 1, seq]
			mask = mask.unsqueeze(1).unsqueeze(1)
			S = S.masked_fill(mask == 0, -1e9)


		attention_score = torch.softmax(S, dim=-1)
		# [B, head_num, seq, seq]
		attention_score = self.dropout(A)

		O = torch.matmul(attention_score, values).transpose(1, 2).contiguous().view(batchsize, seqlen, self.head_dim*self.head_num)
		

		return self.o_proj(O)

class GroupQueryAttention(nn.Module):
	def __init__(self, head_dim, query_head_num, kv_head_num, dropout_rate):
		super().__init__()
		self.head_dim = head_dim
		self.query_head_num = query_head_num
		self.kv_head_num = kv_head_num


		self.q_proj = nn.Linear(head_dim*query_head_num, head_dim*query_head_num, bias=False)

		self.kv_proj = nn.Linear(head_dim*query_head_num, head_dim*kv_head_num*2, bias=False)

		self.o_proj = nn.Linear(head_dim*query_head_num, head_dim*query_head_num, bias=False)

		self.dropout = nn.Dropout(dropout_rate)


	def forward(self, X:Tensor, mask=None):
		batchsize, seqlen, dim = X.shape

		querys = self.q_proj(X).view(batchsize, seqlen, self.query_head_num, self.head_dim).transpose(1, 2)

		kv = self.kv_proj(X).view(batchsize, seqlen, 2, self.kv_head_num, self.head_dim).permute(2, 0, 3, 1, 4)
		# [B, query_head_num, seq, dim]

		# [B, kv_head_num, seq, dim]
		keys, values = kv.unbind(0)

		# keys.repeat_interleave(self.query_head_num // self.kv_head_num, dim=1)
		# values.repeat_interleave(self.query_head_num // self.kv_head_num, dim=1)

		keys = keys.unsqueeze(2).expand(batchsize, self.kv_head_num, self.query_head_num // self.kv_head_num, seqlen, self.head_dim).reshape(batchsize, self.query_head_num, seqlen, self.head_dim)
		values = values.unsqueeze(2).expand(batchsize, self.kv_head_num, self.query_head_num // self.kv_head_num, seqlen, self.head_dim).reshape(batchsize, self.query_head_num, seqlen, self.head_dim)

		# querys [B, head_num, seq, dim]
		# keys/values [B, 1, seq, dim]
		attention_score = torch.matmul(querys, keys.transpose(-1, -2)) / sqrt(self.head_dim)
		if mask is not None:
			
			# mask [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]
			# [batch, seq] -> [batch, 1, 1, seq]
			mask = mask.unsqueeze(1).unsqueeze(1)
			attention_score = attention_score.masked_fill(mask == 0, -1e9)


		attention_score = torch.softmax(attention_score, dim=-1)
		# [B, head_num, seq, seq]
		attention_score = self.dropout(attention_score)

		O = torch.matmul(attention_score, values).transpose(1, 2).contiguous().view(batchsize, seqlen, self.head_dim*self.query_head_num)
		
		return self.o_proj(O)


'''
SDPA scale dot product attention : Pytorch有自己的backend, 能高效的实现GQA
FlashAttn GQA: 高效兼容, 不同query用相同的shape,设置一下索引，共用对应的kv即可
GQA主要省推理时KV cache的显存占用; 训练有收益, 参数少了一些， kv projection是等价的
'''

class MultiHeadTargetAttention(nn.Module):
	def __init__(self, kv_dim, query_dim, head_dim, head_num, dropout_rate):
		super().__init__()
		self.head_dim = head_dim
		self.head_num = head_num

		self.kv_proj = nn.Linear(kv_dim, head_dim*head_num*2, bias=False)
		self.q_proj = nn.Linear(query_dim, head_dim*head_num, bias=False)

		self.o_proj = nn.Linear(head_dim*head_num, head_dim*head_num, bias=False)

		self.dropout = nn.Dropout(dropout_rate)


	def forward(self, query_input:Tensor, kv_input:Tensor, mask=None):
		batchsize, q_seqlen, query_dim = query_input.shape
		batchsize, kv_seqlen, kv_dim = kv_input.shape

		querys = self.q_proj(query_input).view(batchsize, q_seqlen, self.head_num, self.head_dim).transpose(1, 2)
		kv = self.kv_proj(kv_input).view(batchsize, kv_seqlen, 2, self.head_num, self.head_dim).permute(2, 0, 3, 1, 4)
		# [B, H, seq, dim]
		keys, values = kv.unbind(0)


		attention_score = torch.matmul(querys, keys.transpose(-1, -2)) / sqrt(self.head_dim)
		if mask is not None:
			
			# mask [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]
			# [batch, seq] -> [batch, 1, 1, seq]
			mask = mask.unsqueeze(1).unsqueeze(1)
			attention_score = attention_score.masked_fill(mask == 0, -1e9)


		attention_score = torch.softmax(attention_score, dim=-1)
		attention_score = self.dropout(attention_score)

		O = torch.matmul(attention_score, values).transpose(1, 2).contiguous().view(batchsize, q_seqlen, self.head_dim*self.head_num)
		

		return self.o_proj(O)



class MultiHeadTargetAttentionWithAbosrb(nn.Module):
	def __init__(self, kv_dim, query_dim, head_dim, head_num, dropout_rate):
		super().__init__()
		self.head_dim = head_dim
		self.head_num = head_num

		
		self.q_proj = nn.Parameter(torch.rand(head_num, query_dim, head_dim))
		self.k_proj = nn.Parameter(torch.rand(head_num, kv_dim, head_dim))

		self.v_proj = nn.Parameter(torch.rand(head_num, kv_dim, head_dim))

		self.o_proj = nn.Linear(head_dim*head_num, head_dim*head_num, bias=False)

		self.dropout = nn.Dropout(dropout_rate)

		self.param_init()

	def param_init(self):
		# 均匀分布
		# nn.init.uniform_(self.v_proj)

		# 正态分布
		nn.init.normal_(self.q_proj, mean=0, std=0.02)
		nn.init.normal_(self.k_proj, mean=0, std=0.02)
		nn.init.normal_(self.v_proj, mean=0, std=0.02)
		nn.init.normal_(self.o_proj.weight, mean=0, std=0.02)

		if self.o_proj.bias is not None:
			nn.init.zeros_(self.o_proj.bias)

		# nn.init.zeros_(self)
		# # He初始化 & xavier初始化:
		# # uniformer : 均匀分布
		# nn.init.xavier_uniform_(self.q_proj)

		# nn.init.kaiming_uniform_(self.q_proj)

		# # normal_ : 正态分布:
		# nn.init.xavier_normal_(self.q_proj)

		# nn.init.kaiming_normal_(self.k_proj, a=sqrt(5))


	def forward(self, query_input:Tensor, kv_input:Tensor, mask=None):
		# [B, qs, d]
		batchsize, q_seqlen, query_dim = query_input.shape
		batchsize, kv_seqlen, kv_dim = kv_input.shape

		# [B, 1, qs, d] @ [1, H, d, head_dim] @ [1, H, head_dim, kv_dim] -> [B, H, qs, kv_dim]
		u = query_input.unsqueeze(1) @ self.q_proj.unsqueeze(0) @ self.k_proj.transpose(-1, -2).unsqueeze(0)

		# u : [B, H, qs, kv_dim]
		# X : [B, 1, kvs, kv_dim]
		# -> [B, H, qs, kvs]
		attention_score = torch.matmul(u, kv_input.unsqueeze(1).transpose(-1, -2)) / sqrt(self.head_dim) # 注意这里还是除self.head_dim, 保持一致性, 但是方差就不是1了
		if mask is not None:
			
			# mask [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]
			# [batch, seq] -> [batch, 1, 1, seq]
			mask = mask.unsqueeze(1).unsqueeze(1)
			attention_score = attention_score.masked_fill(mask == 0, -1e9)

		attention_score = torch.softmax(attention_score, dim=-1)
		attention_score = self.dropout(attention_score)

		# [B, H, qs, kvs] @ [B, 1, kvs, kv_input] @ [1, H, kv_input, head_dim] -> [B, H, qs, head_dim]
		O = (attention_score @ kv_input.unsqueeze(1) @ self.v_proj.unsqueeze(0).transpose(1, 2) ).contiguous().view(batchsize, q_seqlen, -1)

		return self.o_proj(O)
