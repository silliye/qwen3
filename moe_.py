# 简单把面试中可能会考到的MOE复习一下

# 负载均衡的bias应该注册成buffer, 这样才会和的torch框架解耦
import torch
import torch.nn as nn 

class Router_with_Balance(nn.Module):
    def __init__(self, k, hidden_dim, expert_nums, lambda_bias=0.001):
        super().__init__()
        self.expert_nums = expert_nums
        self.topk = k
        self.lambda_bias = lambda_bias
        
        # 路由线性层
        self.router_linear = nn.Linear(hidden_dim, expert_nums, bias=False)
        # 偏置项：用于动态调整负载平衡
        self.register_buffer("router_bias", torch.zeros(expert_nums))

    def forward(self, x: torch.Tensor):
        """
        x: [Total_Tokens, Hidden_Dim]
        returns: weights [Total_Tokens, K], indices [Total_Tokens, K]
        """
        # 1. 计算原始路由 logits
        logits = self.router_linear(x) # [N, expert_nums]
        scores = torch.sigmoid(logits) 

        # 2. 加入偏置进行 Top-K 选择 (负载均衡策略)
        # 在训练时，如果某个专家被选得太多，其 bias 会减小，从而降低下次被选中的概率
        balancing_scores = scores + self.router_bias
        topk_scores, topk_indices = balancing_scores.topk(self.topk, dim=-1)

        # 3. 这里的关键：最终权重必须基于原始 scores，而不是带偏置的 balancing_scores
        # 这样可以保证权重反映了模型真实的信心，而 bias 只影响选择逻辑
        final_weights = torch.gather(scores, dim=-1, index=topk_indices)
        
        # 4. 归一化权重 (通常使用 Softmax 或简单的 L1 归一化)
        final_weights = F.normalize(final_weights, p=1, dim=-1)

        # 5. 在训练阶段更新 Bias (简单的频率控制)
        if self.training:
            selected_counts = torch.bincount(topk_indices.view(-1), minlength=self.expert_nums)
            avg_count = x.size(0) * self.topk / self.expert_nums
            # 如果选中次数 > 平均值，diff 为正，bias 减小
            diff = selected_counts - avg_count
            self.router_bias.data -= self.lambda_bias * diff

        return final_weights, topk_indices
    


def moe_v1_mask_loop(self, moe_input: torch.Tensor):
    """
    moe_input: [Batch, Seq, Dim]
    """
    B, S, D = moe_input.shape
    tokens = moe_input.view(-1, D)
    total_tokens = B * S
    
    # 初始化输出矩阵
    final_output = torch.zeros_like(tokens)
    
    # 1. 获取路由权重和索引
    # weights: [N, K], indices: [N, K]
    weights, indices = self.Router(tokens)
    
    # 2. 共享专家计算 (Shared Experts)
    # 所有 Token 都会经过共享专家，这有助于保留通用知识
    for expert in self.shared_experts:
        final_output += expert(tokens)
    
    # 3. 路由专家计算 (Routed Experts) - 循环遍历法
    for i in range(self.num_routed_experts):
        # 找出哪些 Token 的 Top-K 中包含当前专家 i
        # hit_mask: [N, K]
        hit_mask = (indices == i)
        
        # 只要 Top-K 里有一个命中了专家 i，就提取出来
        # token_idx: 命中专家的 token 序号
        # k_idx: 命中专家在 Top-K 中的位置（用于取对应的权重）
        token_idx, k_idx = torch.where(hit_mask)
        
        if token_idx.numel() > 0:
            # 提取 token 并计算
            selected_tokens = tokens[token_idx]
            expert_out = self.routed_experts[i](selected_tokens)
            
            # 乘上对应的路由权重并累加
            weighted_out = expert_out * weights[token_idx, k_idx].unsqueeze(-1)
            final_output.index_add_(0, token_idx, weighted_out)
            
    return final_output.view(B, S, D)

router = Router_with_Balance()

def moe_v2_sorting(self, moe_input: torch.Tensor):
    """
    高性能重排版本
    """
    B, S, D = moe_input.shape
    tokens = moe_input.view(-1, D)
    num_tokens = B * S
    
    # 1. 路由
    weights, indices = router(tokens)
    
    # 2. 准备重排数据
    # 将 [N, K] 展平为 [N*K]
    flat_weights = weights.view(-1)
    flat_indices = indices.view(-1)
    
    # 生成原始 Token 索引，对应展平后的索引
    # 例如 K=2, 则索引序列为 [0, 0, 1, 1, 2, 2...]
    token_map = torch.arange(num_tokens, device=tokens.device).repeat_interleave(self.topk)
    
    # 3. 关键步骤：按专家 ID 排序
    # sorted_expert_ids: [N*K] 排序后的专家序列，如 [0, 0, 0, 1, 1, 2...]
    # sort_idx: 记录了原来的位置，用于重排 token_map 和 weights
    sorted_expert_ids, sort_idx = flat_indices.sort()
    
    # 重排 Token 指向和权重
    sorted_token_map = token_map[sort_idx]
    sorted_weights = flat_weights[sort_idx]
    
    # 构造专家连续输入
    # 此时内存是连续的：[Expert0_tokens, Expert1_tokens, ...]
    sorted_tokens = tokens[sorted_token_map]
    
    # 4. 批量计算
    # 使用 bincount 统计每个专家分配到的 token 数量
    expert_counts = torch.bincount(flat_indices, minlength=self.num_routed_experts)
    
    final_output = torch.zeros_like(tokens)
    
    # 预先计算共享专家
    shared_out = sum(e(tokens) for e in self.shared_experts)
    final_output += shared_out
    
    curr_ptr = 0
    for i, count in enumerate(expert_counts.tolist()):
        if count == 0: continue
        
        # 连续切片提取
        expert_input = sorted_tokens[curr_ptr : curr_ptr + count]
        expert_out = self.routed_experts[i](expert_input)
        
        # 应用权重
        weighted_out = expert_out * sorted_weights[curr_ptr : curr_ptr + count].unsqueeze(-1)
        
        # 写回原始位置
        # index_add_ 是原位累加，处理一个 Token 同时被分给多个专家的情况
        final_output.index_add_(0, sorted_token_map[curr_ptr : curr_ptr + count], weighted_out)
        
        curr_ptr += count
        
    return final_output.view(B, S, D)