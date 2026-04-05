import torch
import numpy as np
import gymnasium as gym
import random
from pogema import pogema_v0, GridConfig  
from pogema_toolbox.generators.maze_generator import MazeGenerator
from pogema_toolbox.generators.random_generator import generate_map as generate_random_map
from pogema_toolbox.generators.house_generator import HouseGenerator
from pogema_toolbox.generators.warehouse_generator import generate_wfi_warehouse, WarehouseConfig
from env_wrapper import NativePogemaWrapper
from agent_trainer import AgentTrainer
from rust_buffer import RustReplayBuffer, RustRewardCalculator

def get_generated_map_grid(difficulty_ratio, map_type_config=None, size=None,seed=None):
    if seed is None:
        seed = random.randint(0, 10_000)
    
    # ================= 1. 尺寸与智能体动态控制 (课程学习) =================
    min_size = 15
    max_size = 25
    
    # 地图尺寸线性增长
    current_size = int(min_size + (max_size - min_size) * difficulty_ratio)
    width = random.randint(max(6, current_size - 2), current_size + 2) if size is None else size
    height = random.randint(max(6, current_size - 2), current_size + 2) if size is None else size

    # 智能体数量线性增长：从早期的 8 个平滑增加到后期的 32 个
    # min_agents = 8
    # max_agents = 32
    # current_agents = int(min_agents + (max_agents - min_agents) * difficulty_ratio) #到12个智能体的时候，显存就炸了
    current_agents = 8
    # ================= 2. 步数控制 =================
    # 同步修改为 256，匹配你 yaml 测试文件中的长远规划需求
    max_episode_steps = 256 
    
    # ================= 3. 地图类型控制 =================    
    weights = [0.27, 0.05, 0.33, 0.35] 
    map_type = random.choices(['random', 'warehouse', 'house', 'maze'], weights=weights, k=1)[0]
    map_type = map_type_config if map_type_config is not None else map_type
    
    map_str = ""
    if map_type == 'maze':
        map_str = MazeGenerator.generate_maze(width=width, height=height, obstacle_density=random.uniform(0.2, 0.4), wall_components=random.randint(2, 8), go_straight=random.uniform(0.7, 0.9), seed=seed)
    elif map_type == 'random':
        map_str = generate_random_map({"width": width, "height": height, "obstacle_density": 0.2 + (0.2 * difficulty_ratio), "seed": seed})
    elif map_type == 'house':
         map_str = HouseGenerator.generate(width=width, height=height, obstacle_ratio=random.randint(4, 6), remove_edge_ratio=random.randint(4, 8), seed=seed)
    elif map_type == 'warehouse':
        cfg = WarehouseConfig(wall_width=random.randint(3, 6), wall_height=random.randint(2, 3), walls_in_row=random.randint(2, 4), walls_rows=random.randint(2, 4), bottom_gap=random.randint(1, 3), horizontal_gap=random.randint(1, 2), vertical_gap=random.randint(2, 3))
        map_str = generate_wfi_warehouse(cfg)

    return map_str, map_type, seed, max_episode_steps, current_agents


def persistent_worker_process(worker_id, global_models, task_queue, result_queue, config):
    """
    常驻 Worker 进程：接收信号 -> 同步全局权重 -> 内部训练 -> 返回 Deltas
    """
    # ---------------------------------------------------------
    # 1. 进程级常驻初始化 (只执行一次，避免每轮重复分配 GPU 显存)
    # ---------------------------------------------------------
    device = config.get('device', 'cpu')
    num_agents = config['num_agents']
    obs_channels = config['obs_channels']
    map_channels = config['map_channels']
    num_actions = config['num_actions']
    inner_epochs = config['inner_epochs']
    batch_size = config['batch_size']
    seq_len = config['seq_len']

    # Initialize Local Trainer
    trainer = AgentTrainer(
        obs_channels=obs_channels,
        num_actions=num_actions,
        map_channels=map_channels,
        num_agents=num_agents,
        device=device,
        lr=config.get('inner_lr', 1e-4)
    )

    # 用于记录当前 Reward Calculator 绑定的智能体数量
    current_calculator_agents = None
    reward_calculator = None

    # ---------------------------------------------------------
    # 2. 常驻循环 (Inner-loop 执行区)
    # ---------------------------------------------------------
    while True:
        # 阻塞等待主进程发来任务信号
        task = task_queue.get()
        
        # 接收到毒丸信号，安全退出循环
        if task == "TERMINATE":
            break
            
        progress = task['curr_progress']

        # 从共享内存中的全局模型拉取最新权重
        # load_state_dict 会自动将 CPU 上的 shared weights 转移至当前 Trainer 所在的 GPU
        trainer.eval_drqn.load_state_dict(global_models['drqn'].state_dict())
        trainer.eval_map_encoder.load_state_dict(global_models['map_encoder'].state_dict())
        trainer.eval_mixer.load_state_dict(global_models['mixer'].state_dict())

        # Sync Target networks initially for this meta-iteration
        trainer.target_drqn.load_state_dict(global_models['drqn'].state_dict())
        trainer.target_map_encoder.load_state_dict(global_models['map_encoder'].state_dict())
        trainer.target_mixer.load_state_dict(global_models['mixer'].state_dict())

        # ---------------------------------------------------------
        # 环境与 Buffer 初始化 (每轮 Meta-Iter 需要重置，因为地图大小可能变化)
        # ---------------------------------------------------------
        map_str, map_type, seed, max_episode_steps, current_num_agents = get_generated_map_grid(difficulty_ratio=progress)

        if current_calculator_agents != current_num_agents:
            reward_calculator = RustRewardCalculator(current_num_agents)
            current_calculator_agents = current_num_agents

        grid_config = GridConfig(
            map=map_str,
            num_agents=current_num_agents,                  
            observation_radius=5,          
            on_target='finish',            
            max_episode_steps=max_episode_steps,
            seed=seed
        )

        raw_env = pogema_v0(grid_config=grid_config)
        env = NativePogemaWrapper(raw_env, current_num_agents)

        # Extract map shapes by doing a dummy reset
        obs, info = env.reset()
        dummy_global_map = env.current_global_map
        global_map_shape = dummy_global_map.shape # (C_g, H_g, W_g)
        obs_shape = obs[0].shape # (C, H, W)

        # Initialize Rust Replay Buffer for current map shape
        buffer = RustReplayBuffer(
            capacity=config.get('buffer_capacity', 1000),
            seq_len=seq_len,
            num_agents=current_num_agents,
            obs_shape=obs_shape,
            global_map_shape=global_map_shape
        )

        # ---------------------------------------------------------
        # 执行内层训练循环
        # ---------------------------------------------------------
        total_loss = 0.0
        total_reward = 0.0
        success_count = 0.0
        episodes_run = 0

        for epoch in range(inner_epochs):
            obs, info = env.reset()
            obs = np.array(obs, dtype=np.float32)
            hidden_state = trainer.init_hidden(actual_batch_size=env.num_agents)

            episode_reward = 0.0
            done = False
            step = 0
            max_steps = config.get('max_steps', 200)

            dones = [False] * env.num_agents
            native_arrivals = 0.0

            while not done and step < max_steps:
                actions, next_hidden_state = trainer.select_actions(obs, hidden_state, epsilon=config.get('epsilon', 0.1))
                next_obs, rewards, dones, truncated, infos = env.step(actions)

                native_arrivals += sum(rewards)
                next_obs = np.array(next_obs, dtype=np.float32)

                rewards = reward_calculator.calculate(
                    rewards=np.array(rewards, dtype=np.float32),
                    obs=np.array(obs, dtype=np.float32),
                    next_obs=next_obs,
                    actions=np.array(actions, dtype=np.int64),
                    alignment_config={"use_alignment": False, "value": 0.1},
                    stop_penalty_config={"use_stop_penalty": True, "value": 0.01},
                    step_penalty_config={"use_step_penalty": True, "value": 0.01},
                    goal_reward_multiple=100.0
                )

                env.cache_step(obs, actions, rewards, next_obs, dones)
                episode_reward += sum(rewards)
                obs = next_obs
                hidden_state = next_hidden_state

                if isinstance(dones, (list, tuple, np.ndarray)):
                    if all(dones):
                        done = True
                else:
                    if dones:
                        done = True

                step += 1

            episode_data = env.get_episode_data()
            buffer.push(episode_data)

            total_reward += episode_reward
            episodes_run += 1
            success_count += (native_arrivals / env.num_agents)

            # Perform Local Update using TBPTT
            if buffer.len() >= batch_size:
                for _ in range(4): 
                    batch = buffer.sample(batch_size)
                    loss = trainer.train_step(batch, gamma=config.get('gamma', 0.99))
                    total_loss += loss
                torch.cuda.empty_cache()

        # ---------------------------------------------------------
        # 计算 Parameter Differences (Deltas)
        # \Delta = \theta_{task} - \theta_{global}
        # ---------------------------------------------------------
        deltas = {'drqn': {}, 'map_encoder': {}, 'mixer': {}}
        
        # 提取全局模型状态用于比对
        global_drqn_state = global_models['drqn'].state_dict()
        global_map_encoder_state = global_models['map_encoder'].state_dict()
        global_mixer_state = global_models['mixer'].state_dict()

        with torch.no_grad():
            for name, param in trainer.eval_drqn.state_dict().items():
                deltas['drqn'][name] = (param.cpu() - global_drqn_state[name].cpu()).clone()

            for name, param in trainer.eval_map_encoder.state_dict().items():
                deltas['map_encoder'][name] = (param.cpu() - global_map_encoder_state[name].cpu()).clone()

            for name, param in trainer.eval_mixer.state_dict().items():
                deltas['mixer'][name] = (param.cpu() - global_mixer_state[name].cpu()).clone()

        # Compile Metrics
        metrics = {
            'loss': total_loss / max(1, (inner_epochs - batch_size + 1)) if inner_epochs >= batch_size else 0.0,
            'reward': total_reward / max(1, episodes_run),
            'success_rate': success_count / max(1, episodes_run)
        }

        # 通过队列将结果传回主进程
        result_queue.put((worker_id, deltas, metrics))