import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import SharedDRQN

class AgentTrainer:
    def __init__(self, obs_channels, num_actions, map_channels, num_agents, device='cuda', lr=1e-4):
        self.device = device
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.use_scaler = torch.cuda.is_available()
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler) #由于cuda版本问题，cuda版本较新时，请注释此行，恢复下行
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_scaler) 
        # -------------------------------------------------------------
        # 1. Initialize Networks (Eval and Target for Double Q-learning)
        # -------------------------------------------------------------

        # Eval Networks
        self.eval_drqn = SharedDRQN(obs_channels, num_actions).to(self.device)
        # self.eval_map_encoder = StaticMapEncoder(map_channels).to(self.device)
        # self.eval_mixer = TransformerMixer(num_agents).to(self.device)

        # Target Networks
        self.target_drqn = SharedDRQN(obs_channels, num_actions).to(self.device)
        # self.target_map_encoder = StaticMapEncoder(map_channels).to(self.device)
        # self.target_mixer = TransformerMixer(num_agents).to(self.device)

        # Load Eval weights into Target networks initially
        self.target_drqn.load_state_dict(self.eval_drqn.state_dict())
        # self.target_map_encoder.load_state_dict(self.eval_map_encoder.state_dict())
        # self.target_mixer.load_state_dict(self.eval_mixer.state_dict())

        # We only optimize the Evaluation networks
        self.optimizer = torch.optim.Adam(
            list(self.eval_drqn.parameters()) ,
            # list(self.eval_map_encoder.parameters()) +
            # list(self.eval_mixer.parameters()),
            lr=lr
        )

    def update_target_networks(self, tau=0.005):
        """Soft update target networks: θ_target = τ * θ_eval + (1 - τ) * θ_target"""
        for target_param, eval_param in zip(self.target_drqn.parameters(), self.eval_drqn.parameters()):
            target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

        # for target_param, eval_param in zip(self.target_map_encoder.parameters(), self.eval_map_encoder.parameters()):
        #     target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

        # for target_param, eval_param in zip(self.target_mixer.parameters(), self.eval_mixer.parameters()):
        #     target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

    def select_actions(self, obs, hidden_state, epsilon=0.0):
        """
        Decentralized Execution (CTDE):
        Agents select actions based ONLY on local observations using SharedDRQN.
        No TransformerMixer or MapEncoder is used here.

        obs: (N, C, H, W) numpy array or tensor
        hidden_state: tuple of (h, c), each (N, hidden_dim)
        """
        self.eval_drqn.eval()

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            current_n = obs.shape[0] # 智能体数量
        with torch.no_grad():
            # Get Q-values from DRQN
            q_values, _, new_hidden_state = self.eval_drqn(obs, hidden_state) # q_values: (N, num_actions)

            # Epsilon-greedy exploration
            if torch.rand(1).item() < epsilon:
                actions = torch.randint(0, self.num_actions, (current_n,), device=self.device)
            else:
                actions = torch.argmax(q_values, dim=1) # (N,)

        return actions.cpu().numpy(), new_hidden_state

    def init_hidden(self, actual_batch_size):
        """Initialize LSTM hidden states with zeros"""
        # Batch size * Number of agents
        h = torch.zeros(actual_batch_size, 128, device=self.device)
        c = torch.zeros(actual_batch_size, 128, device=self.device)
        return (h, c)

    def train_step(self, batch, gamma=0.99):
        """
        Centralized Training with Truncated BPTT (TBPTT).

        batch: Dict of tensors from Rust replay buffer.
               Expected shapes:
               - states: (B, T, N, C, H, W)
               - actions: (B, T, N)
               - rewards: (B, T, N)
               - next_states: (B, T, N, C, H, W)
               - dones: (B, T, N)
               - global_maps: (B, C_g, H_g, W_g)  # Notice: Only 1 global map per episode
        """
        self.eval_drqn.train()
        # self.eval_map_encoder.train()
        # self.eval_mixer.train()

        # 1. Extract Batch Data and Move to Device
        states = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32, device=self.device)
        global_maps = torch.tensor(batch['global_maps'], dtype=torch.float32, device=self.device)

        # 获取时间序列的有效位 Mask: 形状 (B, T)
        masks = torch.tensor(batch['masks'], dtype=torch.float32, device=self.device)

        # 在 agent_trainer.py 的 train_step 方法中：
        # 假设 batch['masks'] 的 shape 是 (B, Seq_len, N)
        # 获取这个 batch 中真实的最大有效长度 (非 0 元素的数量)
        valid_steps = masks.sum(dim=1).max().int().item()
        
        B, T, N, C, H, W = states.shape

        # 动态设定 burn_in，例如只取有效长度的前 1/3 作为 burn-in，如果太短则不 burn-in
        burn_in = min(8, valid_steps // 3)  

        # Initialize hidden states
        # Shape: (B * N, hidden_dim)
        eval_hidden = self.init_hidden(actual_batch_size=B * N)
        target_hidden = self.init_hidden(actual_batch_size=B * N)

        # -------------------------------------------------------------
        # 2. Burn-in Phase (Pre-warming LSTM)
        # -------------------------------------------------------------
        # STRICT REQUIREMENT: Prevent VRAM OOM on RTX 4090 by using torch.no_grad().
        # We only pass data through DRQN to update the LSTM hidden states without saving the graph.
        # with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_scaler):   #由于cuda版本问题，cuda版本较新时，请注释此行，恢复下行
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_scaler):
            for t in range(burn_in):
                # Reshape states to (B*N, C, H, W)
                obs_t = states[:, t].reshape(B * N, C, H, W)
                next_obs_t = next_states[:, t].reshape(B * N, C, H, W)

                # Forward pass to update hidden states
                _, _, eval_hidden = self.eval_drqn(obs_t, eval_hidden)
                _, _, target_hidden = self.target_drqn(next_obs_t, target_hidden)

        # STRICT REQUIREMENT: Detach hidden states after burn-in before entering the learning phase.
        # This severs the connection to the burn-in graph (even though we used no_grad, it's safe practice).
        eval_hidden = (eval_hidden[0].detach(), eval_hidden[1].detach())
        target_hidden = (target_hidden[0].detach(), target_hidden[1].detach())

        # -------------------------------------------------------------
        # 3. Learn Phase (TBPTT over the remaining sequence)
        # -------------------------------------------------------------
        # Get Map Token from Eval MapEncoder
        # global_maps shape: (B, C_g, H_g, W_g)
        # with torch.cuda.amp.autocast(enabled=self.use_scaler):   #由于cuda版本问题，cuda版本较新时，请注释此行，恢复下行
        with torch.amp.autocast('cuda', enabled=self.use_scaler):
            eval_map_token = self.eval_map_encoder(global_maps) # (B, 1, hidden_dim)

            with torch.no_grad():
                target_map_token = self.target_map_encoder(global_maps) # (B, 1, hidden_dim)

            # Lists to store Q-values and hidden states for the learning phase
            q_evals = []
            q_targets = []

            for t in range(burn_in, T):
                # Reshape observations
                obs_t = states[:, t].reshape(B * N, C, H, W)
                next_obs_t = next_states[:, t].reshape(B * N, C, H, W)

                # Forward DRQN (Eval)
                # Output: q_vals (B*N, num_actions), h_i (B*N, hidden_dim)
                q_eval_t, h_i_eval, eval_hidden = self.eval_drqn(obs_t, eval_hidden)

                # Select the Q-value corresponding to the action actually taken
                action_t = actions[:, t].reshape(B * N, 1) # (B*N, 1)
                chosen_q_eval = q_eval_t.gather(1, action_t).squeeze(-1) # (B*N,)

                # Reshape outputs to (B, N, ...) for TransformerMixer
                chosen_q_eval = chosen_q_eval.view(B, N) # (B, N)
                h_i_eval = h_i_eval.view(B, N, -1)       # (B, N, hidden_dim)

                # Forward TransformerMixer (Eval) to get Q_tot
                # Pass map_token, h_i, q_i, and dones
                dones_t = dones[:, t] # (B, N)
                # q_tot_eval = self.eval_mixer(eval_map_token, h_i_eval, chosen_q_eval, dones_t) # (B,)
                q_evals.append(chosen_q_eval)

                # --- Double Q-Learning Target Calculation ---
                with torch.no_grad():
                    # 1. Use Eval network to select the BEST next action
                    next_q_eval_t, _, _ = self.eval_drqn(next_obs_t, eval_hidden) # (B*N, num_actions)
                    best_next_actions = torch.argmax(next_q_eval_t, dim=1).unsqueeze(-1) # (B*N, 1)

                    # 2. Use Target network to evaluate that action
                    q_target_t, h_i_target, target_hidden = self.target_drqn(next_obs_t, target_hidden)
                    chosen_q_target = q_target_t.gather(1, best_next_actions).squeeze(-1) # (B*N,)

                    # Reshape for Mixer
                    chosen_q_target = chosen_q_target.view(B, N)
                    h_i_target = h_i_target.view(B, N, -1)

                    # Mixer (Target)
                    # Next state dones (for target masking).
                    # Note: if the state was already done at t, it remains done.
                    q_tot_target = self.target_mixer(target_map_token, h_i_target, chosen_q_target, dones_t) # (B,)

                    # Calculate TD Target: R_tot + gamma * Q_tot_target (if not fully done)
                    # Here we sum the rewards across agents for the centralized Q_tot
                    reward_t = rewards[:, t]

                    # If all agents are done, no future reward
                    all_done_t = torch.all(dones_t == 1, dim=1).float() # (B,)

                    td_target = reward_t + gamma * (1 - dones_t) * chosen_q_target
                    q_targets.append(td_target)

            # -------------------------------------------------------------
            # 4. Compute Loss and Backpropagate
            # -------------------------------------------------------------
            # Stack sequences along time dimension: (B, learn_len)
            q_evals = torch.stack(q_evals, dim=1)
            q_targets = torch.stack(q_targets, dim=1)

            # 截取 Learn 阶段的 masks
            learn_masks = masks[:, burn_in:]

            # 使用 Masked MSE Loss
            # F.mse_loss(reduction='none') 会返回每个元素的独立 loss，形状为 (B, learn_len)
            element_wise_loss = F.mse_loss(q_evals, q_targets.detach(), reduction='none') # (B, learn_len, N)

            # learn_masks 原本是 (B, learn_len)，需要扩展一维来匹配 N
            learn_masks_expanded = learn_masks[:, burn_in:].unsqueeze(-1) # (B, learn_len, 1)
            
            # 掩码相乘
            masked_loss = element_wise_loss * learn_masks_expanded # (B, learn_len, N)
            
            # 先在时间维度求和，除以有效步数，得到每个 episode 中每个 agent 的平均 loss
            episode_valid_steps = learn_masks[:, burn_in:].sum(dim=1).clamp(min=1.0).unsqueeze(-1) # (B, 1)
            episode_agent_losses = masked_loss.sum(dim=1) / episode_valid_steps # (B, N)
            # 最后对 Batch 和 Agent 维度求平均
            loss = episode_agent_losses.mean()

        # Optimize
        self.optimizer.zero_grad()
        # 使用 Scaler 接管反向传播
        if self.use_scaler and self.scaler is not None:
            # 1. 放缩 Loss 并反向传播
            self.scaler.scale(loss).backward()
            
            # 2. 关键！在裁剪梯度前必须先 unscale，否则裁剪阈值就错了
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.eval_drqn.parameters()) ,
                # list(self.eval_map_encoder.parameters()) +
                # list(self.eval_mixer.parameters()),
                max_norm=10.0
            )
            
            # 3. 步进优化器并更新 Scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 兼容不使用 AMP 的情况
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.eval_drqn.parameters()) ,
                # list(self.eval_map_encoder.parameters()) 
                # list(self.eval_mixer.parameters()),
                max_norm=10.0
            )
            self.optimizer.step()

        # Soft update target networks
        self.update_target_networks()

        return loss.item()
