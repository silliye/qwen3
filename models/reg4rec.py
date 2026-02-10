import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from base import BaseModel
from utils import MLP, Qwen2DecoderCrossLayer, Qwen2EncoderLayer

class REG4RecModel(BaseModel):
    def __init__(self, config:dict):
        super(REG4RecModel, self).__init__()

        self.codebook_size = config.get('codebook_size', 301)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.tau = config.get('temperature', 0.1)
        self.token_level = config.get('token_level', 8)
        self.token_dim = config.get('token_dim', 128)
        self.decoder_pos_disable = config.get('decoder_pos_disable', False)
        self.encoder_pos_disable = float(config.get("encoder_pos_disable", False))

        self.encoder_layer_num = int(config.get("encoder_layer_num", 3))

        self.beamsize = int(config.get('beam_size', 4))

        # Qwen Decoder Params
        self.qwen_params = {
            'hidden_size': config.get('hidden_size', 128),
            'num_attention_heads': config.get('qwen02_num_attention_heads', 8),
            'num_key_value_heads': config.get('qwen02_num_kv_heads', 2),
            'intermediate_size': config.get('qwen02_intermediate_size', 640),
            'token_size': config.get('token_size', 0),
            'rope_theta': config.get('qwen02_rope_theta', 1000000)
        }

        # --- Layer Definitions ---

        # 1. Embeddings (VQ Codebooks)
        # 对应 'vq_codebook_level_{i}'
        # 创建 token_level 个码本 (例如 level_0 到 level_7)
        self.vq_codebooks:list[nn.Embedding] = nn.ModuleList([
            nn.Embedding(self.codebook_size, self.token_dim) 
            for _ in range(self.token_level)
        ])
        
        # 2. Position Embeddings
        # 对应 encoder/decoder_position_emb_table
        self.encoder_pos_emb = nn.Embedding(self.token_level, self.token_dim)
        self.decoder_pos_emb = nn.Embedding(self.token_level, self.token_dim)
        
        # 3. BOS Vector (Learnable parameter)
        # 对应 'bos_vector', shape [1, 1, 128]
        self.bos_vector = nn.Parameter(torch.randn(1, 1, self.token_dim) * 0.036)

        # 4. Cross Decoder
        self.cross_decoder = Qwen2DecoderCrossLayer(
            name='token_cross_decoder', **self.qwen_params
        )

        self.qwen_encoder_layers = nn.ModuleList([
            Qwen2EncoderLayer(
                name=f'token_cross_decoder_{i}', **self.qwen_params ) for i in range(self.encoder_layer_num) 
        ])



        # 5. Prediction Towers (MLP Heads)
        # 对应 'expert_predict_tower_nn_level_{i}'
        self.seq_predictors = nn.ModuleList([
            MLP(input_dim=self.token_dim,         # 注意：输入是 Decoder 输出 (dim)，
                hidden_dims=[self.token_dim * 2], # 隐层扩大
                output_dim=self.token_dim, 
                name=f'seq_expert_predict_level_{i}')
            for i in range(self.token_level)
        ])

        self.encoder_mlp_vec =  MLP(input_dim=self.token_dim,        
                hidden_dims=[self.token_dim * self.token_level], 
                output_dim=self.token_dim, 
                name='encoder_mlp_vec')

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 使用 named_modules() 可以同时获取 名字(name) 和 模块对象(m)
        for name, m in self.named_modules():
            
            # --- 1. 处理 Embedding 层 ---
            if isinstance(m, nn.Embedding):
                # A. 如果名字里包含 'vq_codebooks' -> 大方差初始化
                if 'vq_codebooks' in name:
                    nn.init.trunc_normal_(m.weight, std=0.36)
                    print(f"Initialized {name} with std=0.36 (VQ)")
                
                # B. 其他 Embedding (如 Pos, Token) -> 标准小方差
                else:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    # print(f"Initialized {name} with std=0.02 (Standard)")

            
            # --- 2. 处理 Linear 层 ---
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # --- 3. 处理 LayerNorm ---
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            
            # --- 4. 特殊处理 BOS ---
            # 因为 BOS 是 nn.Parameter，不在 named_modules 的 Embedding/Linear 里
            # 需要单独在 __init__ 里最后处理，或者在这里通过名字判断
            
        # 显式处理 nn.Parameter (如 self.bos_vector)
        # 因为它不是 layer，不会被上面的 isinstance(m, nn.Embedding) 捕获
        if hasattr(self, 'bos_vector'):
            nn.init.trunc_normal_(self.bos_vector, std=0.036) 
    

    def get_token_embedding(self, level_idx, indices:Tensor, decoder_pos_disable=False, encoder_pos_disable=False) -> Tensor:
        """
        level_idx: int (0, 1, 2...)
        indices: [B, 1] or [B]
        """
        # [B]
        indices = indices.view(-1)
        
        # 1. Codebook Lookup
        # 对应 tf.gather(vq_code_book, indices)
        codebook = self.vq_codebooks[level_idx] # Module
        code_vectors:Tensor = codebook(indices) # [B, dim]

        # 2. Position Embedding
        # level_idx 直接作为 position index

        # [1] 
        pos_idx_tensor = torch.tensor([level_idx], device=indices.device)
        
        if not encoder_pos_disable:
            # Encoder Pos  [token_level, dim] -> 相当于tf.gather -> [1, dim]
            pos_vec = self.encoder_pos_emb(pos_idx_tensor) # [1, dim]

            # [B, dim] + [1, dim] = [B, dim]
            code_vectors = code_vectors + pos_vec
        elif not decoder_pos_disable:
            # Decoder Pos
            pos_vec = self.decoder_pos_emb(pos_idx_tensor)

            # [B, dim] + [1, dim] = [B, dim]
            code_vectors = code_vectors + pos_vec
            
        return code_vectors.view(-1)

    def forward(self, lastN, target):
        # lastN [B, seq, 8(token_level)] -> [B, 8]
        return 

    def seq_encoder(self, item_seq:Tensor, max_seq_len=128):
        '''这里不一样的是, item_seq是[B, 128, T], T = token_level
        相当于需要把item_seq的8个sid id取出来, 并通过一个MLP组成一个item token
        之后送进下游的qwen2Encoder
        
        item_seq : [B, 128, 8]
        '''

        # [B, 128, dim]
        token_input_emb:Tensor = self.get_token_embedding(0, item_seq[:, : ,0], encoder_pos_disable=True).reshape([-1, max_seq_len, self.token_dim])
        BatchSize = token_input_emb.shape(0)

        for i in range(self.token_level-1):
            token_input_emb_i = self.get_token_embedding(i+1, item_seq[:, : ,i+1], encoder_pos_disable=True).reshape([-1, max_seq_len, 1])
            token_input_emb = torch.cat([token_input_emb, token_input_emb_i], dim=-1)

        # [B, 128, dim*8] -> [B, 128, dim]
        token_encoder_emb = self.encoder_mlp_vec(token_input_emb)

        # [B, 128, 1]
        encoder_seq_mask = torch.ones(BatchSize, max_seq_len, 1, device=token_input_emb.device)
        for i in range(self.encoder_layer_num):
            token_encoder_emb = self.qwen_encoder_layers[i](token_encoder_emb, encoder_seq_mask)
        
        return token_encoder_emb


    def seq_decoder_train(self, encoder_fusion:Tensor, encoder_seq_mask:Tensor, token_level_target:Tensor, is_train=True):
        """
        encoder_fusion: [B, seq, dim]
        encoder_seq_mask: [B, seq, 1]
        token_level_target: List of [B, 1], length = token_level
        """
        if not is_train: return
        
        batch_size = encoder_fusion.size(0)
        device = encoder_fusion.device

        
        # BOS: [B, 1, dim]          expand = tf.tile 
        bos_input = self.bos_vector.expand(batch_size, -1, -1)
        
        # 收集每一层的 GT Embedding
        # list of [B, 1, dim]
        gt_emb_list = []
        for i in range(self.token_level):
            # 获取第 i 层的真实 label 对应的 embedding
            emb = self.get_token_embedding(
                level_idx=i, 
                indices=token_level_target[i], 
                decoder_pos_disable=self.decoder_pos_disable
            )
            gt_emb_list.append(emb.unsqueeze(1))
            
        # 确定 Padding 长度
        # 逻辑：decoder_length 必须固定以适应硬件
        decoder_length = 16
        current_len = 1 + self.token_level # BOS + levels
        pad_size = 0
        
        # 简单的 padding 策略逻辑 (参考原代码的 if-elif)
        if current_len <= 4: decoder_length = 4
        elif current_len <= 8: decoder_length = 8
        elif current_len <= 16: decoder_length = 16
        
        pad_size = decoder_length - current_len
        pad_size = max(0, pad_size) # 防止负数

        # 拼接 Input: [BOS, GT_Level_0, ..., GT_Level_N]

        # [B, 1+token_level, dim]
        seq_input = torch.cat([bos_input] + gt_emb_list, dim=1)
        
        # 添加 Zero Pad
        if pad_size > 0:
            # [B, pad_size, dim]
            zero_pad = torch.zeros(batch_size, pad_size, self.token_dim, device=device)

            seq_input = torch.cat([seq_input, zero_pad], dim=1)
            
        # 最终 seq_input: [B, decoder_length, dim]

        # --- 2. 准备 Mask ---
        # 简单逻辑：BOS 和 GT 部分是 1 (valid)，Pad 部分是 0
        # [B, 1+token_level, 1]
        valid_mask = torch.ones(batch_size, current_len, 1, device=device)
        # [B, decoder_length, 1]

        if pad_size > 0:
            pad_mask = torch.zeros(batch_size, pad_size, 1, device=device)
            decoder_casual_mask = torch.cat([valid_mask, pad_mask], dim=1)
        else:
            decoder_casual_mask = valid_mask

        # [B, decoder_length, dim]
        token_output = self.cross_decoder(
            tgt=seq_input, 
            memory=encoder_fusion, 
            seq_mask=decoder_casual_mask, 
            kv_mask=encoder_seq_mask
        )

        # --- 4. 计算 Loss (Loop over levels) ---
        # 注意：Decoder 是 Auto regressive 的，第 i 个位置的输出预测第 i+1 个位置
        # input:  [BOS, L0, L1, ...]
        # output: [Out0, Out1, Out2, ...]
        # Out0 应该预测 L0
        # Out1 应该预测 L1
        
        # 我们只计算前 token_level 个有效位置的 loss
        for i in range(self.token_level):
            # 取出第 i 个时间步的输出 [B, dim]
            step_output = token_output[:, i, :]
            
            # 经过 MLP 投影 [B, dim]
            logits = self.seq_predictors[i](step_output)
            
            # Ground Truth Embedding (无位置编码纯净版，用于算相似度)
            # 注意：原代码的 token_sftmx_loss 里面重新获取了一次 embedding，
            # 这里的 idx=i，对应的 target 是 token_level_target[i]

            # [B, 1]
            target_indices = token_level_target[i]

            # 我们直接从 codebook 取，不加 Position
            # [B, dim]
            target_clean_emb = self.get_token_embedding(i, target_indices).reshape([-1, self.token_dim])
            
            # 计算 Loss
            loss, logits = self.token_sftmx_loss(logits, target_clean_emb, level_idx=i)

            # [B, 1]
            hit_1 = (logits.topk(1).indices == target_indices).mean().item()
            hit_2 = (logits.topk(2).indices == target_indices).mean().item()
            hit_4 = (logits.topk(4).indices == target_indices).mean().item()
            hit_5 = (logits.topk(5).indices == target_indices).mean().item()
            hit_10 = (logits.topk(10).indices == target_indices).mean().item()
            print(hit_1, hit_2, hit_4, hit_5, hit_10)
            target_indices


    def seq_decoder_eval(self,  encoder_fusion:Tensor, encoder_seq_mask:Tensor, is_eval=True):

        if not is_eval:
            return False
        
        batch_size, encoder_seq_length, dim = encoder_fusion.shape

        decoder_length = 16
        expand_flag = True

        # [B, 1, dim]
        bos_input = self.bos_vector.expand(batch_size, 1, -1)
        # [B, 1, dim]
        zero_pad = torch.zeros(batch_size, 1, self.token_dim)
        # [B, 1, 1]
        one_mask = torch.ones(batch_size, -1, -1)
        zero_mask = torch.zeros(batch_size, 1, 1)

        idxs_comb:Tensor = None
        probs_comb:Tensor = None

        seq_input_no_padding = bos_input  # 一直更新seq_input_no_padding
        for i in range(self.token_level):
            
            # seq_input_no_padding
                # i == 0: [B, 1, dim] -> add [B, K, 1, dim] = [B, K, 2, dim]
                # i == 1: [B, K, 2, dim] -> [B, K, 1, dim] = [B, K, 3, dim]
                # i == 2: [B, K, 3, dim] -> [B, K, 1, dim] = [B, K, 4, dim]
            # 

            # i == 0 : [B, 1, dim]
            seq_input:Tensor = torch.cat(seq_input_no_padding, zero_pad.expand(-1, decoder_length-i+1, -1), dim=-2)
            # i == 1 : [B, K, 2, dim]


            if i > 0:
                seq_input = torch.cat(seq_input_no_padding, zero_pad.unsqueeze(1).expand(-1, self.beamsize, decoder_length-i+1, -1), dim=-2)


            # [B, 16, 1]
            decoder_casual_mask:Tensor = torch.cat(one_mask.expand(-1, i+1, -1), zero_mask.expand(-1, decoder_length-i-1, -1))

            if i > 0 and expand_flag:
                #[Batchsize * beamsize, seq, dim]
                # 只需要扩充一次;
                encoder_fusion = encoder_fusion.unsqueeze(1).expand(-1, self.beamsize, -1, -1).reshape(-1, encoder_seq_length, self.token_dim)
                encoder_seq_mask = encoder_seq_mask.unsqueeze(1).expand(-1, self.beamsize, -1, -1).reshape(-1, encoder_seq_length, 1)
                expand_flag = False
            
            # 这个是一直要变的
            if i > 0:
                decoder_casual_mask = decoder_casual_mask.unsqueeze(1).expand(-1, self.beamsize, -1, -1).reshape(-1, decoder_length, 1)

            # [B, 16, dim]
            token_output = self.cross_decoder(
                tgt=seq_input, 
                memory=encoder_fusion, 
                seq_mask=decoder_casual_mask, 
                kv_mask=encoder_seq_mask
            )

            if i == 0:
                # [B, 1, dim]
                logits = self.seq_predictors[i](token_output[:, i, :])

                # [B, K, 1]
                probs, idxs = self.beam_search(logits, i)

                # [B, K, i+1]
                idxs_comb = idxs.reshape(batch_size, self.beamsize, 1)
                probs_comb = probs.reshape(batch_size, self.beamsize, 1)

                # [B, K, dim]
                token_embedding = self.get_token_embedding(i, idxs_comb).reshape(batch_size, self.beamsize, 1, self.token_dim)
            else:

                # [B, K, 1, dim]
                logits = self.seq_predictors[i](token_output[:, i, :])

                # [B*K,  K, 1]
                probs, idxs = self.beam_search(logits, i)
                
                probs = probs.reshape(batch_size, self.beamsize, self.beamsize, 1)
                idxs = idxs.reshape(batch_size, self.beamsize, self.beamsize, 1)
                
                # 扩充idx_combs [B, K, 1] -> [B, K, K, 1] concat  [B, K, K, 2]
                # [[1], [2]] -> [[[1], [1]], [[2], [2]]]  concat   [[[3], [4]], [[3], [4]]]
                # == [[[1, 3], [1, 4]], [[2, 3], [2, 4]]]

                # [B, K, K, i+1]
                idxs_comb_temp:Tensor = torch.concat(idxs_comb.unsqueeze(-2).expand(-1, -1, self.beamsize, -1), idxs, dim=-1)

                prob_comb_temp:Tensor = torch.concat(probs_comb.unsqueeze(-2).expand(-1, -1, self.beamsize, -1), probs, dim=-1)

                # [B, K*K, i+1]
                idxs_comb_temp = idxs_comb_temp.reshape([batch_size, -1, i+1])
                prob_comb_temp = prob_comb_temp.reshape([batch_size, -1, i+1])

                # [B, K*K, 1] -> [B, K, 1]
                # _, probs_comb_temp_prod_topk_indices = torch.prod(prob_comb_temp, dim=-1, keepdim=True).topk(self.beamsize, dim=-2)

                _, probs_comb_temp_prod_topk_indices = torch.sum(torch.log(prob_comb_temp), dim=-1, keepdim=True).topk(self.beamsize, dim=-2)


                # [B, K, i+1]
                idxs_comb = torch.gather(idxs_comb_temp, -2, probs_comb_temp_prod_topk_indices)
                probs_comb = torch.gather(prob_comb_temp, -2, probs_comb_temp_prod_topk_indices)

                
                token_embedding = self.get_token_embedding(i, idxs_comb[:, :, i:]).reshape(batch_size, self.beamsize, 1, self.token_dim)


            if i == 0:
                seq_input_no_padding = torch.concat(seq_input_no_padding.unsqueeze(1).expand(-1, self.beamsize, -1, -1), token_embedding, dim=-2)
            else:
                seq_input_no_padding = torch.concat(seq_input_no_padding, token_embedding, dim=-2)


        # [B, K, 8]
        return idxs_comb, probs_comb    
                

    def beam_search(self, logits:Tensor, i) -> Tensor:
        B = logits.shape[0]
        # logits = [B, 1, dim]
        # [code_book_size, dim]
        vq_book:Tensor = self.vq_codebooks[i].weight

        logits = F.normalize(logits, dim=-1)
        vq_book = F.normalize(vq_book, dim=-1)

        # [B, code_size]
        logits = torch.matmul(logits, vq_book.T).reshape(B, -1) / self.tau

        # 
        return torch.softmax(logits, dim=-1).topk(self.beamsize, dim=-1)



    def token_sftmx_loss(self, token_logits, real_token_emb, level_idx):
        """
        token_logits: [B, dim] 预测值
        real_token_emb: [B, dim] 真实值的向量
        level_idx: int 用于获取对应的整个 codebook
        """
        # 1. 归一化 (L2 Norm)
        # [B, dim]
        token_logits_norm = F.normalize(token_logits, p=2, dim=-1)
        # [B, dim]
        real_token_emb_norm = F.normalize(real_token_emb, p=2, dim=-1)

        # 2. 计算正样本得分 (Positive Score)
        # [B]
        # Sim(pred, gt) / tau
        pos_score = torch.sum(token_logits_norm * real_token_emb_norm, dim=-1) / self.tau

        # 3. 计算所有样本得分 (Logits for Softmax)
        # 获取第 i 层的完整 Codebook: [Codebook_Size, Dim]

        '''Linear 存的weight矩阵是和声明的反着的:
            比如 声明 Linear(input_dim, output_dim), 存储的weightTensor矩阵实际上是[output_dim, input_dim], forward相当于  X_input @ weight^T
           nn.Embedding 存储是正着的, 它是通过gather的, (1, code_size) (code_size, dim)  得到 (1, dim), 其中(1, code_size)是从(1)的一个indexs构造的one-hot '''
        
        
        codebook_weight = self.vq_codebooks[level_idx].weight
        codebook_norm = F.normalize(codebook_weight, p=2, dim=-1)

        # 矩阵乘法: [B, Dim] @ [Dim, Codebook_Size] -> [B, Codebook_Size]
        # Sim(pred, all_codes) / tau

        # [B, code_size]
        all_logits:Tensor = torch.matmul(token_logits_norm, codebook_norm.T) / self.tau

        # 4. 计算 Cross Entropy (Softmax) Loss
        # = softmax(logits) vs one-hot  做交叉熵
        # Formula: - log( exp(pos) / sum(exp(all)) )
        #        = - ( pos - logsumexp(all) )
        #        = logsumexp(all) - pos

        # [B]
        log_sum_exp = torch.logsumexp(all_logits, dim=-1)
        loss = log_sum_exp - pos_score
        
        loss_mean = loss.mean()
        self.losses.append(loss_mean.item())



        return loss_mean, all_logits