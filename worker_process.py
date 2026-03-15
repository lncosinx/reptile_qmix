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

def get_generated_map_grid(difficulty_ratio, map_type_config=None, seed=None):
    if seed is None:
        seed = random.randint(0, 10_000)
    
    # ================= 1. 尺寸控制：改为线性 =================
    min_size = 15
    max_size = 25
    
    # 这样模型有充足的时间适应中等大小的地图
    adjusted_ratio = difficulty_ratio 
    
    current_size = int(min_size + (max_size - min_size) * adjusted_ratio)
    
    width = random.randint(max(6, current_size - 2), current_size + 2)
    height = random.randint(max(6, current_size - 2), current_size + 2)

    # ================= 2. 步数控制 =================
    max_episode_steps = 128 
    # ================= 3. 地图类型控制 =================    
    weights = [0.27, 0.05, 0.33, 0.35] 
    map_type = random.choices(['random', 'warehouse', 'house', 'maze'], weights=weights, k=1)[0]
        
    #测试指定地图类型
    map_type = map_type_config if map_type_config is not None else map_type
    # 这里的代码需要补全之前的逻辑
    map_str = ""
    if map_type == 'maze':
        map_str = MazeGenerator.generate_maze(width=width, 
                                              height=height, 
                                              obstacle_density=random.uniform(0.2, 0.4), 
                                              wall_components=random.randint(2, 8), 
                                              go_straight=random.uniform(0.7, 0.9), 
                                              seed=seed)
    elif map_type == 'random':
        map_str = generate_random_map({"width": width, 
                                       "height": height, 
                                       "obstacle_density": 0.2 + (0.2 * difficulty_ratio), 
                                       "seed": seed})
    elif map_type == 'house':
         map_str = HouseGenerator.generate(width=width, 
                                           height=height, 
                                           obstacle_ratio=random.randint(4, 6), 
                                           remove_edge_ratio=random.randint(4, 8), 
                                           seed=seed)
    elif map_type == 'warehouse':
        cfg = WarehouseConfig(wall_width=random.randint(3, 6), 
                              wall_height=random.randint(2, 3), 
                              walls_in_row=random.randint(2, 4), 
                              walls_rows=random.randint(2, 4), 
                              bottom_gap=random.randint(1, 3), 
                              horizontal_gap=random.randint(1, 2), 
                              vertical_gap=random.randint(2, 3))
        map_str = generate_wfi_warehouse(cfg)

    return map_str, map_type, seed, max_episode_steps

def run_worker_task(worker_id, global_state_dict, config):
    """
    Executes the inner-loop (Inner-loop) training for the Reptile Meta-learning framework.

    Args:
        worker_id (int): ID of the worker.
        global_state_dict (dict): Dictionary containing the state_dicts of the global Eval networks.
        config (dict): Hyperparameters and configurations.

    Returns:
        dict: Contains the parameter differences (Deltas) and training metrics.
    """
    # 1. Initialization
    device = config.get('device', 'cpu')
    num_agents = config['num_agents']
    obs_channels = config['obs_channels']
    map_channels = config['map_channels']
    num_actions = config['num_actions']
    inner_epochs = config['inner_epochs']
    batch_size = config['batch_size']
    seq_len = config['seq_len']
    env_name = config.get('env_name', 'Pogema-8x8-normal-v0')

    # Initialize Local Trainer
    trainer = AgentTrainer(
        obs_channels=obs_channels,
        num_actions=num_actions,
        map_channels=map_channels,
        num_agents=num_agents,
        device=device,
        lr=config.get('inner_lr', 1e-4)
    )

    # Load Global Weights (\theta_{task} = \theta_{global})
    trainer.eval_drqn.load_state_dict(global_state_dict['drqn'])
    trainer.eval_map_encoder.load_state_dict(global_state_dict['map_encoder'])
    trainer.eval_mixer.load_state_dict(global_state_dict['mixer'])

    # Also sync Target networks initially
    trainer.target_drqn.load_state_dict(global_state_dict['drqn'])
    trainer.target_map_encoder.load_state_dict(global_state_dict['map_encoder'])
    trainer.target_mixer.load_state_dict(global_state_dict['mixer'])
    progress = config.get('curr_progress', 0.0)

    # Initialize Environment
    # 1. 动态生成一张地图的字符串（你可以根据课程学习的进度动态调整 difficulty_ratio）
    map_str, map_type, seed, max_episode_steps = get_generated_map_grid(difficulty_ratio=progress)

    # 2. 将地图字符串注入到 Pogema 的 GridConfig 中
    grid_config = GridConfig(
        map=map_str,
        num_agents=num_agents,                  # 必须与你设定的智能体数量一致
        observation_radius=5,          # 视野范围
        on_target='finish',            # 到达目标后的行为
        max_episode_steps=max_episode_steps,
        seed=seed
    )

    # 3. 使用核心基础名称 "Pogema-v0" 结合 grid_config 创建环境，彻底摆脱静态名称限制
    raw_env = pogema_v0(grid_config=grid_config)
    env = NativePogemaWrapper(raw_env, num_agents)

    reward_calculator = RustRewardCalculator(num_agents)

    # Extract map shapes by doing a dummy reset
    obs, info = env.reset()
    dummy_global_map = env.current_global_map
    global_map_shape = dummy_global_map.shape # (C_g, H_g, W_g)

    obs_shape = obs[0].shape # (C, H, W)

    # Initialize Rust Replay Buffer
    buffer = RustReplayBuffer(
        capacity=config.get('buffer_capacity', 1000),
        seq_len=seq_len,
        num_agents=num_agents,
        obs_shape=obs_shape,
        global_map_shape=global_map_shape
    )

    # 2. Collect Trajectories and Train
    total_loss = 0.0
    total_reward = 0.0
    success_count = 0
    episodes_run = 0

    for epoch in range(inner_epochs):
        # Reset Environment
        obs, info = env.reset()

        # Format obs to (N, C, H, W)
        obs = np.array(obs, dtype=np.float32)

        # Initialize hidden states for execution
        hidden_state = trainer.init_hidden(batch_size=1)

        episode_reward = 0.0
        done = False
        step = 0
        max_steps = config.get('max_steps', 200)

        # 在 while 循环上方初始化
        dones = [False] * env.num_agents

        while not done and step < max_steps:
            # Select actions (Decentralized Execution)
            actions, next_hidden_state = trainer.select_actions(obs, hidden_state, epsilon=config.get('epsilon', 0.1))

            # 强制接管已完成智能体的动作
            for i in range(env.num_agents):
                if dones[i]:
                    actions[i] = 0  # 强制设为 0 (Stay) 操作，避免触发 Rust 端的撞墙惩罚

            # Step environment
            next_obs, rewards, dones, truncated, infos = env.step(actions)

            # Format outputs
            next_obs = np.array(next_obs, dtype=np.float32)

            rewards = reward_calculator.calculate(
            rewards=np.array(rewards, dtype=np.float32),
            obs=np.array(obs, dtype=np.float32),
            next_obs=next_obs,
            actions=np.array(actions, dtype=np.int64),
            # 传入在 Rust 中定义的 Config
            alignment_config={"use_alignment": True, "value": 0.1},
            stop_penalty_config={"use_stop_penalty": True, "value": 0.01},
            step_penalty_config={"use_step_penalty": True, "value": 0.01},
            goal_reward_multiple=100.0
            )

            # Cache step data locally in Python wrapper
            env.cache_step(obs, actions, rewards, next_obs, dones)

            episode_reward += sum(rewards)
            obs = next_obs
            hidden_state = next_hidden_state

            # In POGEMA, done is usually a list of booleans per agent.
            # We consider episode done if all agents are done.
            if isinstance(dones, list) or isinstance(dones, tuple) or isinstance(dones, np.ndarray):
                if all(dones):
                    done = True
            else:
                if dones:
                    done = True

            step += 1

        # Push the full episode to Rust Replay Buffer
        episode_data = env.get_episode_data()
        buffer.push(episode_data)

        total_reward += episode_reward
        episodes_run += 1

        # Calculate success (Did all agents reach target?)
        # Simple heuristic based on dones array at the last step
        if all(episode_data['dones'][-1]):
            success_count += 1

        # Perform Local Update using TBPTT
        if buffer.len() >= batch_size:
            # 内部进行 4 次抽样更新，极大提升样本利用率和 4090 GPU 负载
            for _ in range(4): 
                batch = buffer.sample(batch_size)
                loss = trainer.train_step(batch, gamma=config.get('gamma', 0.99))
                total_loss += loss
            # 释放由 TBPTT 动态图产生的显存碎片
            torch.cuda.empty_cache()

    # 3. Compute Parameter Differences: \Delta = \theta_{task} - \theta_{global}
    # This prevents sending huge absolute weights back and saves IPC overhead.
    deltas = {
        'drqn': {},
        'map_encoder': {},
        'mixer': {}
    }

    # Compute differences for DRQN
    for name, param in trainer.eval_drqn.state_dict().items():
        deltas['drqn'][name] = (param.cpu() - global_state_dict['drqn'][name].cpu()).clone()

    # Compute differences for MapEncoder
    for name, param in trainer.eval_map_encoder.state_dict().items():
        deltas['map_encoder'][name] = (param.cpu() - global_state_dict['map_encoder'][name].cpu()).clone()

    # Compute differences for Mixer
    for name, param in trainer.eval_mixer.state_dict().items():
        deltas['mixer'][name] = (param.cpu() - global_state_dict['mixer'][name].cpu()).clone()

    # Compile Metrics
    metrics = {
        'loss': total_loss / max(1, (inner_epochs - batch_size + 1)) if inner_epochs >= batch_size else 0.0,
        'reward': total_reward / max(1, episodes_run),
        'success_rate': success_count / max(1, episodes_run)
    }

    return worker_id, deltas, metrics
