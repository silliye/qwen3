"""
PyTorch 面试复习 —— 从 Tensor 到训练循环
按面试常考顺序组织，每个 API 都能独立跑，直接 python torchreview.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Tensor 创建 —— 面试高频：这些工厂函数的区别
# ============================================================

# 1.1 从数据/形状直接建
a = torch.tensor([1, 2, 3])              # 从 python list/numpy 建，会推断 dtype
b = torch.tensor([1.0, 2, 3])            # 有浮点 → float32
c = torch.Tensor([1, 2, 3])              # 类构造器（大写 T），dtype 恒为 float32；不推荐
d = torch.as_tensor([1, 2, 3])           # 共享内存（如果可能），比 tensor() 少一次拷贝

# 1.2 按形状建（考点：形状参数是 *sizes，不是 tuple）
z = torch.zeros(2, 3)                    # 全 0
o = torch.ones(2, 3)                     # 全 1
e = torch.empty(2, 3)                    # 未初始化，值是内存里的垃圾数据
f = torch.full((2, 3), 7.0)              # 填充指定值
i = torch.eye(3)                         # 单位矩阵

# 1.3 按范围/随机建
r1 = torch.arange(0, 10, 2)              # [0,2,4,6,8]，左闭右开，同 numpy
r2 = torch.linspace(0, 1, 5)             # [0, 0.25, 0.5, 0.75, 1]，左闭右闭，n 个点
rand1 = torch.rand(2, 3)                 # [0,1) 均匀分布
rand2 = torch.randn(2, 3)                # 标准正态 N(0,1) —— 神经网络初始化常用
rand3 = torch.randint(0, 10, (2, 3))     # 整数均匀
rand4 = torch.randperm(10)               # 0~9 的随机排列，DataLoader shuffle 用

# 1.4 xxx_like —— 继承另一个 tensor 的 shape/dtype/device
like = torch.zeros_like(a)               # 同 shape、同 dtype、同 device 的全 0

# 1.5 指定 dtype 和 device（面试考点：GPU 转移方式）
x = torch.zeros(3, dtype=torch.float32, device="cpu")
# x_gpu = x.cuda()          # 老写法
# x_gpu = x.to("cuda:0")    # 新写法，更灵活（也能转 dtype）

# 1.6 固定随机种子 —— 复现实验必备
torch.manual_seed(42)                    # CPU
torch.cuda.manual_seed_all(42)           # 所有 GPU


# ============================================================
# 2. Tensor 属性 —— 面试常问：shape/dtype/device/requires_grad
# ============================================================
t = torch.randn(2, 3, 4)
_ = t.shape                              # torch.Size([2,3,4])，等价 t.size()
_ = t.size(0)                            # 2，指定维度
_ = t.dtype                              # torch.float32
_ = t.device                             # cpu
_ = t.ndim                               # 3，维度数（同 len(t.shape)）
_ = t.numel()                            # 24，元素总数
_ = t.requires_grad                      # False，是否追踪梯度
_ = t.is_leaf                            # True，叶子节点（用户直接创建的都是）


# ============================================================
# 3. 形状变换 —— 面试重灾区：view / reshape / transpose / permute
# ============================================================
x = torch.arange(12)

# 3.1 view vs reshape（面试高频）
v = x.view(3, 4)                         # 要求内存连续（contiguous），否则报错
r = x.reshape(3, 4)                      # 自动处理不连续情况（必要时拷贝）
# 面试标准答案：view 要求连续；reshape 更安全但可能拷贝。生产建议 reshape。

# 3.2 -1 表示自动推断
_ = x.view(-1, 4)                        # 剩下那个维度自动算 → (3,4)

# 3.3 transpose vs permute
m = torch.randn(2, 3, 4)
_ = m.transpose(0, 1)                    # 只交换两个维度 → (3,2,4)
_ = m.permute(2, 0, 1)                   # 任意重排 → (4,2,3)
# 注意：transpose/permute 后内存不连续，接 view 前要 .contiguous()

# 3.4 squeeze / unsqueeze —— 增删长度为 1 的维度
s = torch.zeros(1, 3, 1, 4)
_ = s.squeeze()                          # → (3,4)，去掉所有长度 1 的维
_ = s.squeeze(0)                         # → (3,1,4)，只去掉第 0 维
_ = torch.zeros(3, 4).unsqueeze(0)       # → (1,3,4)，加一个 batch 维（常见操作）

# 3.5 展平
_ = m.flatten()                          # → (24,) 全部展平
_ = m.flatten(start_dim=1)               # → (2, 12) 保留 batch 维（分类头常用）

# 3.6 拼接与堆叠（面试考点：区别）
a1 = torch.zeros(2, 3)
a2 = torch.ones(2, 3)
_ = torch.cat([a1, a2], dim=0)           # (4,3) 沿已有维度拼接
_ = torch.stack([a1, a2], dim=0)         # (2,2,3) 新建一个维度


# ============================================================
# 4. 索引与切片 —— 与 numpy 几乎一致，考高级索引
# ============================================================
x = torch.arange(20).view(4, 5)
_ = x[0]                                 # 第一行
_ = x[:, 1]                              # 第 1 列
_ = x[1:3, 2:4]                          # 切片
_ = x[[0, 2], [1, 3]]                    # 花式索引：取 (0,1) 和 (2,3) → shape (2,)
_ = x[x > 10]                            # 布尔索引：所有 >10 的元素展平


# ============================================================
# 5. 数学运算 —— 逐元素 / 归约 / 矩阵乘（面试高频）
# ============================================================
a = torch.tensor([1.0, 2, 3])
b = torch.tensor([4.0, 5, 6])

# 5.1 逐元素
_ = a + b; _ = a - b; _ = a * b; _ = a / b
_ = torch.add(a, b)                      # 函数式，等价 a+b
_ = a.add_(b)                            # 下划线结尾 = in-place（省内存，但对 autograd 不友好）

# 5.2 归约（考点：dim 参数和 keepdim）
m = torch.randn(3, 4)
_ = m.sum()                              # 标量
_ = m.sum(dim=0)                         # 沿第 0 维求和 → (4,)
_ = m.sum(dim=1, keepdim=True)           # (3,1)，keepdim 保留维度方便广播
_ = m.mean(); _ = m.max(); _ = m.min()
_ = m.argmax(dim=1)                      # 每行最大值的索引 —— 分类预测常用
_ = m.std(); _ = m.var()

# 5.3 矩阵运算（面试必考：@ / matmul / mm / bmm 的区别）
A = torch.randn(2, 3)
B = torch.randn(3, 4)
_ = A @ B                                # 推荐写法，(2,4)
_ = torch.matmul(A, B)                   # 同上，支持广播批量
_ = torch.mm(A, B)                       # 只支持 2D
BA = torch.randn(10, 2, 3)               # batch=10
BB = torch.randn(10, 3, 4)
_ = torch.bmm(BA, BB)                    # batched matmul → (10,2,4)
_ = A.T                                  # 转置（2D）；高维用 .transpose 或 .mT

# 5.4 常用函数
_ = torch.exp(a); _ = torch.log(a); _ = torch.sqrt(a); _ = torch.pow(a, 2)
_ = torch.abs(a); _ = torch.sigmoid(a); _ = torch.tanh(a); _ = torch.relu(a)


# ============================================================
# 6. 广播机制（Broadcasting）—— 面试必问
# ============================================================
# 规则：从右往左对齐维度，每一维要么相等，要么其中一个是 1，要么其中一个不存在。
x = torch.zeros(3, 1, 5)
y = torch.zeros(   4, 1)                 # → 补齐成 (1,4,1)
_ = (x + y).shape                        # → (3,4,5)


# ============================================================
# 7. 自动求导 Autograd —— 面试核心考点
# ============================================================
# 7.1 基础：requires_grad + backward()
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1                   # y = x^2 + 3x + 1
y.backward()                             # 反向传播，计算 dy/dx
_ = x.grad                               # tensor(7.)  =  2x+3 = 7

# 7.2 中间变量默认不保留 grad，非叶子节点需要 retain_grad()
x = torch.tensor(2.0, requires_grad=True)
z = x * 3
z.retain_grad()                          # 否则 z.grad 是 None
loss = z ** 2
loss.backward()

# 7.3 梯度累加 —— 面试常考陷阱！
# .backward() 会把梯度累加到 .grad 上，而不是覆盖。
# 所以训练循环里每一步都必须 optimizer.zero_grad() 或 param.grad = None
w = torch.tensor(1.0, requires_grad=True)
(w * 2).backward()                       # w.grad = 2
(w * 2).backward()                       # w.grad = 4，累加了！
w.grad.zero_()                           # 手动清零

# 7.4 阻断梯度的三种方式（面试高频）
with torch.no_grad():                    # 上下文管理：推理时用，省显存
    y = x * 2
y = x.detach()                           # 从计算图剥离，共享数据但不追踪
# @torch.no_grad() 装饰器同 with no_grad()

# 7.5 手动求梯度（不建议用于训练）
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
grads = torch.autograd.grad(y, x)        # 返回 tuple，(dy/dx,) = (12,)


# ============================================================
# 8. GPU / device —— 面试常问：怎么把模型和数据搬 GPU
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.zeros(3).to(device)            # 数据搬 GPU
# model = MyModel().to(device)           # 模型搬 GPU
# 注意：模型和输入必须在同一 device，否则报错


# ============================================================
# 9. nn 模块 —— 定义模型（面试必考）
# ============================================================
class MLP(nn.Module):
    """标准三层 MLP，面试白板题常见"""

    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        super().__init__()               # 必须调用，否则参数注册失败
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)              # 训练时随机置零；eval() 模式下自动关闭
        return self.fc2(x)               # logits，不接 softmax（交给 CrossEntropyLoss）


# 9.1 常见层
_ = nn.Linear(10, 5)                     # 全连接 y = xW^T + b
_ = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 卷积 (N,C,H,W)
_ = nn.BatchNorm2d(16)                   # BN，train/eval 行为不同
_ = nn.LayerNorm(128)                    # LN，Transformer 用
_ = nn.Embedding(1000, 64)               # 词嵌入，vocab_size=1000, dim=64
_ = nn.LSTM(64, 128, batch_first=True)   # LSTM，注意 batch_first
_ = nn.MultiheadAttention(64, num_heads=8, batch_first=True)  # 多头注意力

# 9.2 容器
seq = nn.Sequential(                     # 顺序容器，前一层输出接后一层输入
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1),
)
mlist = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])  # 可迭代，forward 里手动调
# 注意：不能用 python list 存 Module，否则参数不会被 .parameters() 收集

# 9.3 nn.Module 常用方法（面试考点）
model = MLP()
_ = list(model.parameters())             # 所有可训练参数，传给 optimizer
_ = list(model.named_parameters())       # (name, param) 对，调试打印用
model.train()                            # 训练模式：BN 用 batch 统计，Dropout 开
model.eval()                             # 评估模式：BN 用 running 统计，Dropout 关
# state_dict = model.state_dict()        # 保存权重字典
# model.load_state_dict(state_dict)      # 加载权重


# ============================================================
# 10. 损失函数与优化器
# ============================================================
# 10.1 常见损失（面试考点：CrossEntropyLoss 内含 softmax）
ce = nn.CrossEntropyLoss()               # 分类，target 是类别索引，不是 one-hot
bce = nn.BCEWithLogitsLoss()             # 二分类，内含 sigmoid，比 BCELoss 更稳
mse = nn.MSELoss()                       # 回归
l1 = nn.L1Loss()

# 10.2 常见优化器
# optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optim = torch.optim.Adam(model.parameters(), lr=1e-3)
# optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)  # 现代默认

# 10.3 学习率调度器
# sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
# sched = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)


# ============================================================
# 11. 标准训练循环 —— 面试白板题必背
# ============================================================
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()                                    # 1. 切训练模式
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)            # 2. 数据搬 device
        optimizer.zero_grad()                        # 3. 清零梯度（关键！）
        logits = model(x)                            # 4. 前向
        loss = loss_fn(logits, y)                    # 5. 算损失
        loss.backward()                              # 6. 反向传播
        optimizer.step()                             # 7. 更新参数
        total_loss += loss.item()                    # .item() 取标量，避免累积计算图
    return total_loss / len(loader)


@torch.no_grad()                                     # 推理不建图，省显存
def evaluate(model, loader, device):
    model.eval()                                     # 关 dropout / 用 BN running stats
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


# ============================================================
# 12. 保存与加载（面试常问：state_dict vs 整个模型）
# ============================================================
# 推荐：只存 state_dict（跨版本更稳）
# torch.save(model.state_dict(), "model.pth")
# model.load_state_dict(torch.load("model.pth", map_location=device))

# 恢复训练时要一起存 optimizer 和 epoch
# torch.save({
#     "model": model.state_dict(),
#     "optim": optimizer.state_dict(),
#     "epoch": epoch,
# }, "ckpt.pth")


# ============================================================
# 13. DataLoader —— 面试知道基本用法即可
# ============================================================
# from torch.utils.data import Dataset, DataLoader
# class MyDataset(Dataset):
#     def __init__(self): ...
#     def __len__(self): return N               # 必须实现
#     def __getitem__(self, idx): return x, y   # 必须实现
# loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)


# ============================================================
# 14. 面试高频陷阱 —— 面试官爱问的细节
# ============================================================
# Q1: view 和 reshape 的区别？
#   A: view 要求内存连续；reshape 内部会判断，不连续时自动拷贝。
#      transpose/permute 后必须 .contiguous().view() 或直接用 reshape。

# Q2: 为什么每次 backward 前要 zero_grad？
#   A: PyTorch 的梯度是累加的（为支持 RNN、gradient accumulation 等场景）。
#      不清零会导致上一步的梯度混进来。

# Q3: detach() 和 with no_grad() 区别？
#   A: detach() 作用于单个 tensor，返回一个不追踪梯度的新 tensor（共享内存）。
#      with no_grad() 是上下文管理器，作用域内所有操作都不建图，适合整段推理代码。

# Q4: model.eval() 和 with torch.no_grad() 区别？（面试超高频）
#   A: 两者独立，通常一起用。
#      - eval() 只是切换 Dropout / BatchNorm 的行为（用 running stats、关 dropout）
#      - no_grad() 是关掉 autograd 建图，省显存加速
#      推理时两个都要开。

# Q5: nn.Module 的 forward 为什么不直接调用而是 model(x)？
#   A: model(x) 会走 __call__，里面会触发 hooks（forward pre/post hooks）
#      并做一些内部记账。直接调 forward 会绕过这些机制。

# Q6: nn.CrossEntropyLoss 输入要不要过 softmax？
#   A: 不要！它内部是 log_softmax + NLLLoss，传 raw logits 即可。
#      target 是类别索引 (LongTensor)，不是 one-hot。

# Q7: BatchNorm 和 LayerNorm 的区别？
#   A: BN 在 batch 维度归一化（每个 channel 一组 μ/σ），依赖 batch size，训练/推理行为不同。
#      LN 在特征维度归一化（每个样本独立），与 batch 无关，训练/推理一致。Transformer 用 LN。

# Q8: register_buffer 是什么？
#   A: 注册一个不参与梯度更新、但会被 state_dict 保存、会跟着 .to(device) 走的 tensor。
#      典型用例：BatchNorm 的 running_mean / running_var，位置编码 sin/cos。

# Q9: 参数初始化怎么做？
#   A: nn.init.kaiming_normal_(layer.weight)  # ReLU 家族
#      nn.init.xavier_uniform_(layer.weight)  # tanh/sigmoid
#      通过 model.apply(init_fn) 递归应用到所有子模块。

# Q10: contiguous() 什么时候要调？
#   A: 一个 tensor 经过 transpose/permute/narrow 后内存不再连续，
#      view/.flatten(*, start_dim=k>0) 会报错，此时先 .contiguous()。


if __name__ == "__main__":
    # 冒烟测试：跑一次 MLP 的 forward + backward
    torch.manual_seed(0)
    model = MLP(in_dim=20, hidden=32, out_dim=3)
    x = torch.randn(8, 20)
    y = torch.randint(0, 3, (8,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    print(f"loss = {loss.item():.4f}")
    print(f"fc1.weight.grad shape = {model.fc1.weight.grad.shape}")
