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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * mix_hidden_dim)
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

        # 4. QMIX-like Non-linear Mixing
        
        # dones 值为 1 表示已到达终点，因此 (1 - dones) 为有效掩码
        q_i_masked = q_i * (1.0 - dones)

        # Ensure q_i is reshaped for batched matrix multiplication: (B, 1, N)
        q_i_reshaped = q_i_masked.view(B, 1, N)

        # W1: Absolute values to ensure non-negative weights
        w1 = torch.abs(self.hyper_w1(s_global))
        w1 = w1.view(B, N, self.mix_hidden_dim)
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
