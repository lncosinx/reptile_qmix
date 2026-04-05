import os
import yaml
import torch
import numpy as np
from pogema import pogema_v0, GridConfig
from env_wrapper import NativePogemaWrapper
from agent_trainer import AgentTrainer
from rust_buffer import RustReplayBuffer, RustRewardCalculator

config_path = {
    'random': {
        'maps': 'random/maps.yaml',
        'config': 'random/01-random.yaml'
    },
    'warehouse': {
        'maps': 'warehouse/maps.yaml',
        'config': 'warehouse/03-warehouse.yaml'
    },
    'maze': {
        'maps': 'maze/maps.yaml',
        'config': 'maze/02-maze.yaml'
    }
}

def load_yaml_configs(config_name):
    with open(config_path[config_name]['maps'], 'r') as f:
        maps = yaml.safe_load(f)
    with open(config_path[config_name]['config'], 'r') as f:
        config = yaml.safe_load(f)
    
    map_names = config['environment']['map_name']['grid_search']
    num_agents_list = config['environment']['num_agents']['grid_search']
    max_episode_steps = config['environment']['max_episode_steps']
    
    return maps, map_names, num_agents_list, max_episode_steps

def fine_tune():
    # --- 微调超参数配置 ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    META_MODEL_PATH_DRQN = './models/global_drqn_iter_4800.pth' # 请替换为实际的元模型路径
    META_MODEL_PATH_ENCODER = './models/global_map_encoder_iter_4800.pth'
    META_MODEL_PATH_MIXER = './models/global_mixer_iter_4800.pth'
    
    INNER_EPOCHS = 512     # 微调回合数
    UPDATE_ITERS = 4       # 每个 epoch 后从 Buffer 采样的更新次数
    BATCH_SIZE = 32
    SEQ_LEN = 120
    BUFFER_CAPACITY = 1000
    ALGORITHM_NAME = "Reptile-Finetuned"
    
    os.makedirs('./models', exist_ok=True)
    maps_dict, map_names, num_agents_list, max_episode_steps = load_yaml_configs('random')

    for num_agents in num_agents_list:
        # 初始化 Trainer (支持多智能体数量变化)
        trainer = AgentTrainer(
            obs_channels=3, map_channels=1, num_actions=5, 
            num_agents=num_agents, device=DEVICE, lr=1e-4
        )
        reward_calculator = RustRewardCalculator(num_agents)

        for map_name in map_names:
            print(f"========== 开始微调: {map_name} | Agents: {num_agents} ==========")
            map_str = maps_dict[map_name]
            
            # 每次微调前，重置为 Meta 模型的全局权重
            trainer.eval_drqn.load_state_dict(torch.load(META_MODEL_PATH_DRQN, map_location=DEVICE))
            trainer.eval_map_encoder.load_state_dict(torch.load(META_MODEL_PATH_ENCODER, map_location=DEVICE))
            trainer.eval_mixer.load_state_dict(torch.load(META_MODEL_PATH_MIXER, map_location=DEVICE))
            
            trainer.target_drqn.load_state_dict(trainer.eval_drqn.state_dict())
            trainer.target_map_encoder.load_state_dict(trainer.eval_map_encoder.state_dict())
            trainer.target_mixer.load_state_dict(trainer.eval_mixer.state_dict())

            # 初始化环境
            grid_config = GridConfig(map=map_str, num_agents=num_agents, observation_radius=5, 
                                     on_target='finish', max_episode_steps=max_episode_steps, seed=0)
            raw_env = pogema_v0(grid_config=grid_config)
            env = NativePogemaWrapper(raw_env, num_agents)

            # 初始化 Buffer
            obs, _ = env.reset()
            global_map_shape = env.current_global_map.shape
            obs_shape = obs[0].shape
            buffer = RustReplayBuffer(capacity=BUFFER_CAPACITY, seq_len=SEQ_LEN, 
                                      num_agents=num_agents, obs_shape=obs_shape, 
                                      global_map_shape=global_map_shape)

            # 内层微调循环
            for epoch in range(INNER_EPOCHS):
                obs, _ = env.reset()
                obs = np.array(obs, dtype=np.float32)
                hidden_state = trainer.init_hidden(actual_batch_size=num_agents)
                done = False
                
                while not done:
                    actions, next_hidden_state = trainer.select_actions(obs, hidden_state, epsilon=0.1)
                    next_obs, rewards, dones, _, _ = env.step(actions)
                    next_obs = np.array(next_obs, dtype=np.float32)
                    
                    rewards = reward_calculator.calculate(
                        rewards=np.array(rewards, dtype=np.float32), obs=np.array(obs, dtype=np.float32),
                        next_obs=next_obs, actions=np.array(actions, dtype=np.int64),
                        alignment_config={"use_alignment": False, "value": 0.1},
                        stop_penalty_config={"use_stop_penalty": True, "value": 0.01},
                        step_penalty_config={"use_step_penalty": True, "value": 0.01},
                        goal_reward_multiple=100.0
                    )
                    
                    env.cache_step(obs, actions, rewards, next_obs, dones)
                    obs = next_obs
                    hidden_state = next_hidden_state
                    
                    done = all(dones) if isinstance(dones, (list, tuple, np.ndarray)) else dones

                # Push to buffer and train
                buffer.push(env.get_episode_data())
                if buffer.len() >= BATCH_SIZE:
                    for _ in range(UPDATE_ITERS):
                        batch = buffer.sample(BATCH_SIZE)
                        loss = trainer.train_step(batch, gamma=0.99)
            
            # 保存微调后的 DRQN 模型
            save_path = f'./models/{ALGORITHM_NAME}_drqn_{map_name}_agents_{num_agents}.pth'
            torch.save(trainer.eval_drqn.state_dict(), save_path)
            print(f"已保存微调模型至: {save_path}")

if __name__ == '__main__':
    fine_tune()