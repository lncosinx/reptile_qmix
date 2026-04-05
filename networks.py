import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedDRQN(nn.Module):
    def __init__(self, obs_channels, num_actions, hidden_dim=128):
        super().__init__()
        # Input: (B*N, C, 11, 11)
        # 2-layer CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_channels, 32, kernel_size=3, stride=1, padding=0), # 11 -> 9
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), # 9 -> 7
            nn.ReLU()
        )
        # Flattened size: 64 * 7 * 7 = 3136
        self.lstm = nn.LSTMCell(3136, hidden_dim)

        self.q_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        self.h_out = nn.Linear(hidden_dim, hidden_dim)

    def init_hidden(self, batch_size):
        """Initialize LSTM hidden states with zeros"""
        # 动态获取当前模型张量所在的设备（CPU/CUDA）
        device = next(self.parameters()).device
        
        # 动态获取 LSTM 的隐藏层维度 (自动对应你初始化时的 128)
        hidden_dim = self.lstm.hidden_size
        
        # 生成初始隐藏状态
        h = torch.zeros(batch_size, hidden_dim, device=device)
        c = torch.zeros(batch_size, hidden_dim, device=device)
        
        return (h, c)

    def forward(self, obs, hidden_state):
        """
        obs: (B*N, C, 11, 11)
        hidden_state: tuple of (h, c), each (B*N, hidden_dim)
        """
        x = self.cnn(obs)
        x = x.view(x.size(0), -1) # Flatten
        h, c = self.lstm(x, hidden_state)
        q = self.q_mlp(h)
        h_i = self.h_out(h)
        return q, h_i, (h, c)


class StaticMapEncoder(nn.Module):
    def __init__(self, map_channels, hidden_dim=128):
        super().__init__()
        # Input: (B, C, H, W) dynamic map sizes 15~25
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(128, hidden_dim)

    def forward(self, global_map):
        """
        global_map: (B, C, H, W)
        """
        x = self.cnn(global_map) # (B, 128, H, W)
        x = self.pool(x)         # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)# Flatten -> (B, 128)
        map_token = self.fc(x)   # (B, hidden_dim)
        return map_token.unsqueeze(1) # (B, 1, hidden_dim)


class TransformerMixer(nn.Module):
    def __init__(self, num_agents, hidden_dim=128, embed_dim=128, num_heads=4, mix_hidden_dim=32):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.mix_hidden_dim = mix_hidden_dim

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Hypernetwork for QMIX-like mixing
        # W1: s_global -> (N, mix_hidden_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mix_hidden_dim)
        )
        # B1: s_global -> (1, mix_hidden_dim)
        self.hyper_b1 = nn.Linear(hidden_dim, mix_hidden_dim)

        # W2: s_global -> (mix_hidden_dim, 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, mix_hidden_dim)
        )
        # B2: s_global -> (1, 1)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, map_token, h_i, q_i, dones):
        """
        map_token: (B, 1, hidden_dim)
        h_i: (B, N, hidden_dim)
        q_i: (B, N)
        dones: (B, N)
        """
        B = h_i.size(0)
        N = h_i.size(1)

        # 1. Sequence Concatenation
        # Sequence length becomes N+1
        seq = torch.cat([map_token, h_i], dim=1) # (B, N+1, hidden_dim)

        # 2. Key Padding Mask
        # PyTorch expects True for positions that are masked out (ignored).
        # Map_token is always False (not masked).
        # Dones are 1 if agent reached target -> masked out.
        map_mask = torch.zeros((B, 1), dtype=torch.bool, device=dones.device)
        agent_mask = dones.bool() # (B, N)
        padding_mask = torch.cat([map_mask, agent_mask], dim=1) # (B, N+1)

        # 3. Transformer Encoder
        out_seq = self.transformer(seq, src_key_padding_mask=padding_mask) # (B, N+1, hidden_dim)

        # Extract 0th position feature (map_token) as high-level global state
        s_global = out_seq[:, 0, :] # (B, hidden_dim)

        agent_features = out_seq[:, 1:, :] # (B, N, hidden_dim)

        # s_global 是 (B, 128) -> 扩展成 (B, N, 128)
        s_global_expanded = s_global.unsqueeze(1).expand(-1, N, -1)

        # 把全局宏观情报发给每个人，拼在一起变成 256 维度
        combined_features = torch.cat([agent_features, s_global_expanded], dim=-1)

        # 4. QMIX-like Non-linear Mixing
        
        # dones 值为 1 表示已到达终点，因此 (1 - dones) 为有效掩码
        q_i_masked = q_i * (1.0 - dones)

        # Ensure q_i is reshaped for batched matrix multiplication: (B, 1, N)
        q_i_reshaped = q_i_masked.view(B, 1, N)

        # W1: Absolute values to ensure non-negative weights
        w1 = torch.abs(self.hyper_w1(combined_features))
        # w1 = w1.view(B, N, self.mix_hidden_dim)
        b1 = self.hyper_b1(s_global).view(B, 1, self.mix_hidden_dim)

        # Hidden layer: (B, 1, N) @ (B, N, mix_hidden_dim) -> (B, 1, mix_hidden_dim)
        hidden = F.elu(torch.bmm(q_i_reshaped, w1) + b1)

        # W2: Absolute values
        w2 = torch.abs(self.hyper_w2(s_global))
        w2 = w2.view(B, self.mix_hidden_dim, 1)
        b2 = self.hyper_b2(s_global).view(B, 1, 1)

        # Q_tot: (B, 1, mix_hidden_dim) @ (B, mix_hidden_dim, 1) -> (B, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        q_tot = q_tot.view(B) # (B,)

        return q_tot
    

class StandardQMIXMixer(nn.Module):
    """
    经典 QMIX 的 MLP 超网络混合器 (Ablation Baseline)
    用于替换 TransformerMixer，验证注意力机制的空间推理优势。
    """
    def __init__(self, num_agents, state_dim=128, embed_dim=32, hypernet_embed=64):
        super(StandardQMIXMixer, self).__init__()
        self.n_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # -------------------------------------------------------------
        # Hypernetwork 1: 生成 W1 和 b1
        # W1 必须非负，因此输出维度为 n_agents * embed_dim
        # -------------------------------------------------------------
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim * self.n_agents)
        )
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # -------------------------------------------------------------
        # Hypernetwork 2: 生成 W2 和 b2 (用 V(s) 代替 b2)
        # W2 必须非负，输出维度为 embed_dim * 1
        # -------------------------------------------------------------
        self.hyper_w_2 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim)
        )
        # V(s) 作为最终的全局偏移量 (不需要非负约束)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states, dones=None):
        """
        agent_qs: (Batch_Size, N_agents) - 每个智能体局部的 Q 值
        states: (Batch_Size, State_Dim) - StaticMapEncoder 提取的全局地图特征向量
        dones: (Batch_Size, N_agents) - 可选掩码，抹除已到达终点的 agent 的 Q 值
        """
        bs = agent_qs.size(0)

        # 1. 展平全局状态为 1D 向量，丢弃拓扑结构 (这就是它的劣势所在)
        states = states.view(bs, -1) 

        # 2. 处理局部 Q 值掩码
        if dones is not None:
            agent_qs = agent_qs * (1.0 - dones)
        
        # 调整形状以进行批次矩阵乘法 (B, 1, N)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # -------------------------------------------------------------
        # 第一层混合
        # -------------------------------------------------------------
        w1 = torch.abs(self.hyper_w_1(states)) # 绝对值激活，保证单调性
        w1 = w1.view(-1, self.n_agents, self.embed_dim) # (B, N, Embed)
        b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim) # (B, 1, Embed)

        # F.elu 激活
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1) # (B, 1, Embed)

        # -------------------------------------------------------------
        # 第二层混合
        # -------------------------------------------------------------
        w2 = torch.abs(self.hyper_w_2(states)) # 绝对值激活，保证单调性
        w2 = w2.view(-1, self.embed_dim, 1) # (B, Embed, 1)
        v = self.V(states).view(-1, 1, 1) # (B, 1, 1)

        # 计算总 Q 值 Q_tot
        y = torch.bmm(hidden, w2) + v # (B, 1, 1)
        q_tot = y.view(bs, -1) # (B, 1)

        return q_tot

class VDNMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_i, dones):
        """
        q_i: (B, N)
        dones: (B, N)
        """
        # 如果 agent 到达终点，其 Q 值不计入全局 Q
        q_i_masked = q_i * (1.0 - dones)
        
        # VDN 的核心：直接求和
        q_tot = torch.sum(q_i_masked, dim=1) # 输出 shape: (B,)
        return q_tot
    
class ViTMapEncoder(nn.Module):
    def __init__(self, map_channels, hidden_dim=128, num_heads=4, num_layers=1):
        super().__init__()
        # 1. CNN 提取局部高级语义 (Patch Embedding 的前置特征)
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # 2. 将任意大小的地图自适应切分为 5x5 的网格 (总共 25 个 Patch)
        self.grid_size = 5
        self.num_patches = self.grid_size * self.grid_size
        self.patch_pool = nn.AdaptiveAvgPool2d((self.grid_size, self.grid_size))
        
        # 3. ViT 核心组件
        # 类别 Token (类似 ViT 的 [CLS] token)，用于汇聚全局信息
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 位置编码 (Positional Encoding)：25 个 Patch + 1 个 CLS Token
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))
        
        # Transformer Encoder 处理 Patch 之间的全局空间关系
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, global_map):
        """
        global_map: (B, C, H, W)
        """
        B, C, H, W = global_map.shape
        
        # --- 步骤 A: CNN 提取局部特征并生成 Patches ---
        x = self.cnn(global_map)          # (B, hidden_dim, H, W)
        x = self.patch_pool(x)            # (B, hidden_dim, 5, 5)
        
        # 展平空间维度，转换为 Sequence: (B, 25, hidden_dim)
        x = x.flatten(2).transpose(1, 2)  
        
        # --- 步骤 B: 拼接 [CLS] Token 并加入位置编码 ---
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, hidden_dim)
        x = torch.cat((cls_tokens, x), dim=1)           # (B, 26, hidden_dim)
        x = x + self.pos_embed                          # 加入空间位置信息
        
        # --- 步骤 C: Transformer 建立全局注意力机制 ---
        x = self.transformer(x)                         # (B, 26, hidden_dim)
        
        # --- 步骤 D: 提取 [CLS] Token 作为最终的全局地图表征 ---
        map_token = x[:, 0, :]                          # (B, hidden_dim)
        
        # 扩增维度以匹配后续 TransformerMixer 的输入要求: (B, 1, hidden_dim)
        return map_token.unsqueeze(1)