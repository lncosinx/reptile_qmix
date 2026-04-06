import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import SharedDRQN, FusedCrossAttentionMixer

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
        self.eval_mixer = FusedCrossAttentionMixer(num_agents, map_channels).to(self.device)

        # Target Networks
        self.target_drqn = SharedDRQN(obs_channels, num_actions).to(self.device)
        self.target_mixer = FusedCrossAttentionMixer(num_agents, map_channels).to(self.device)

        # Load Eval weights into Target networks initially
        self.target_drqn.load_state_dict(self.eval_drqn.state_dict())
        self.target_mixer.load_state_dict(self.eval_mixer.state_dict())

        # We only optimize the Evaluation networks
        self.optimizer = torch.optim.Adam(
            list(self.eval_drqn.parameters()) +
            list(self.eval_mixer.parameters()),
            lr=lr
        )

    def update_target_networks(self, tau=0.005):
        """Soft update target networks: θ_target = τ * θ_eval + (1 - τ) * θ_target"""
        for target_param, eval_param in zip(self.target_drqn.parameters(), self.eval_drqn.parameters()):
            target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

        for target_param, eval_param in zip(self.target_mixer.parameters(), self.eval_mixer.parameters()):
            target_param.data.copy_(tau * eval_param.data + (1.0 - tau) * target_param.data)

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

    def train_step(self, batch, gamma=0.99, current_step=0, max_anneal_steps=2000):
        # 🌟 Alpha 退火计算保持不变
        start_alpha = 0.9
        end_alpha = 0.0
        progress = min(1.0, current_step / max_anneal_steps) 
        alpha = start_alpha - progress * (start_alpha - end_alpha)

        self.eval_drqn.train()
        self.eval_mixer.train()

        # 1. 解析 Batch
        states = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32, device=self.device)
        global_maps = torch.tensor(batch['global_maps'], dtype=torch.float32, device=self.device)
        masks = torch.tensor(batch['masks'], dtype=torch.float32, device=self.device)
        
        # 🌟 修复 2：提取智能体坐标！
        agent_coords = torch.tensor(batch['agent_coords'], dtype=torch.float32, device=self.device)

        valid_steps = masks.sum(dim=1).max().int().item()
        B, T, N, C, H, W = states.shape
        burn_in = min(8, valid_steps // 3)  

        eval_hidden = self.init_hidden(actual_batch_size=B * N)
        target_hidden = self.init_hidden(actual_batch_size=B * N)

        # 2. Burn-in Phase
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_scaler):
            for t in range(burn_in):
                obs_t = states[:, t].reshape(B * N, C, H, W)
                next_obs_t = next_states[:, t].reshape(B * N, C, H, W)
                _, _, eval_hidden = self.eval_drqn(obs_t, eval_hidden)
                _, _, target_hidden = self.target_drqn(next_obs_t, target_hidden)

        eval_hidden = (eval_hidden[0].detach(), eval_hidden[1].detach())
        target_hidden = (target_hidden[0].detach(), target_hidden[1].detach())

        # 3. Learn Phase
        with torch.amp.autocast('cuda', enabled=self.use_scaler):
            q_evals = []      # 存全局 Q_tot (CTDE Loss 用)
            q_targets = []    # 存全局 Target_Q_tot (CTDE Loss 用)
            
            # 🌟 修复 3.1：新增列表，存储局部的 Q_i (IQL Loss 用)
            iql_q_evals = []  
            iql_q_targets = [] 

            for t in range(burn_in, T):
                obs_t = states[:, t].reshape(B * N, C, H, W)
                next_obs_t = next_states[:, t].reshape(B * N, C, H, W)

                # --- Eval DRQN ---
                q_eval_t, h_i_eval, eval_hidden = self.eval_drqn(obs_t, eval_hidden)
                action_t = actions[:, t].reshape(B * N, 1) 
                chosen_q_eval = q_eval_t.gather(1, action_t).squeeze(-1) 
                
                chosen_q_eval = chosen_q_eval.view(B, N) 
                h_i_eval = h_i_eval.view(B, N, -1)       
                
                # 记录局部 Q_i
                iql_q_evals.append(chosen_q_eval)

                # --- 调用新的 Mixer ---
                # 注意参数顺序：agent_qs, hidden_states, global_map, agent_coords
                coords_t = agent_coords[:, t] # (B, N, 2)
                q_tot_eval = self.eval_mixer(
                    chosen_q_eval.unsqueeze(1),  # (B, 1, N)
                    h_i_eval.unsqueeze(1),       # (B, 1, N, hidden_dim)
                    global_maps, 
                    coords_t.unsqueeze(1)        # (B, 1, N, 2)
                )
                q_evals.append(q_tot_eval.view(B)) # squeeze 成 (B,)

                # --- Target DRQN ---
                with torch.no_grad():
                    next_q_eval_t, _, _ = self.eval_drqn(next_obs_t, eval_hidden) 
                    best_next_actions = torch.argmax(next_q_eval_t, dim=1).unsqueeze(-1) 

                    q_target_t, h_i_target, target_hidden = self.target_drqn(next_obs_t, target_hidden)
                    chosen_q_target = q_target_t.gather(1, best_next_actions).squeeze(-1) 

                    chosen_q_target = chosen_q_target.view(B, N)
                    h_i_target = h_i_target.view(B, N, -1)
                    
                    # 记录局部 Target_Q_i
                    iql_q_targets.append(chosen_q_target)

                    # Mixer Target
                    dones_t = dones[:, t] 
                    q_tot_target = self.target_mixer(
                        chosen_q_target.unsqueeze(1), 
                        h_i_target.unsqueeze(1), 
                        global_maps, 
                        coords_t.unsqueeze(1)
                    )
                    # 展平成 (B,) 用来算 TD Error
                    q_tot_target = q_tot_target.view(B)

                    reward_tot_t = rewards[:, t].sum(dim=1) 
                    all_done_t = torch.all(dones_t == 1, dim=1).float() 
                    td_target = reward_tot_t + gamma * (1 - all_done_t) * q_tot_target
                    q_targets.append(td_target)

            # 4. 计算融合 Loss
            q_evals = torch.stack(q_evals, dim=1)           # (B, learn_len)
            q_targets = torch.stack(q_targets, dim=1)       # (B, learn_len)
            iql_q_evals = torch.stack(iql_q_evals, dim=1)   # (B, learn_len, N)
            iql_q_targets = torch.stack(iql_q_targets, dim=1) # (B, learn_len, N)

            learn_masks = masks[:, burn_in:] # (B, learn_len)

            # =================================================================
            # 🌟 修复 3.2：绝对正确的 IQL Loss 计算方式
            # =================================================================
            # 取出 Learn 阶段的 reward 和 done，形状 (B, learn_len, N)
            rewards_learn = rewards[:, burn_in:]
            dones_learn = dones[:, burn_in:]
            
            # 现在的运算是纯正的个体级别: (B, learn_len, N)
            iql_target = rewards_learn + gamma * (1 - dones_learn) * iql_q_targets
            
            # 计算 element-wise Huber Loss
            loss_iql_element = F.smooth_l1_loss(iql_q_evals, iql_target.detach(), reduction='none')
            
            # 对 N 个智能体求平均，得到 (B, learn_len)，以便和全局 Mask 对齐
            loss_iql_mean_agents = loss_iql_element.mean(dim=-1) 
            loss_iql = loss_iql_mean_agents * learn_masks

            # =================================================================
            # CTDE Loss (QMIX) 计算
            # =================================================================
            loss_ctde_element = F.smooth_l1_loss(q_evals, q_targets.detach(), reduction='none')
            loss_ctde = loss_ctde_element * learn_masks

            # =================================================================
            # 动态融合！
            # =================================================================
            total_loss = (1.0 - alpha) * loss_ctde + alpha * loss_iql
            
            episode_valid_steps = learn_masks.sum(dim=1).clamp(min=1.0)
            episode_losses = total_loss.sum(dim=1) / episode_valid_steps
            loss = episode_losses.mean()

        # Optimize (去掉了 eval_map_encoder)
        self.optimizer.zero_grad()
        if self.use_scaler and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.eval_drqn.parameters()) + list(self.eval_mixer.parameters()),
                max_norm=10.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.eval_drqn.parameters()) + list(self.eval_mixer.parameters()),
                max_norm=10.0
            )
            self.optimizer.step()

        self.update_target_networks()
        return loss.item()