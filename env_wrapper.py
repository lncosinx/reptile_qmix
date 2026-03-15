import numpy as np

class NativePogemaWrapper:
    """
    处理原生 POGEMA 环境，强制将所有输入输出维度转换为 Rust ReplayBuffer 所需的严格格式。
    采用标准的 List/Array 接口，不使用字典解析。
    """
    def __init__(self, env, num_agents):
        self.env = env
        self.num_agents = num_agents
        self.unwrapped = self._get_unwrapped_env()
        
        self.current_global_map = None
        self.episode_history = []

    def _get_unwrapped_env(self):
        """层层剥开 Gym 的皮，找到真正包含 map/grid 的原生对象"""
        env_ptr = self.env
        while hasattr(env_ptr, 'env') or hasattr(env_ptr, 'unwrapped'):
            if hasattr(env_ptr, 'grid'):
                break
            # 防止死循环
            if hasattr(env_ptr, 'unwrapped') and env_ptr.unwrapped == env_ptr:
                break
            env_ptr = getattr(env_ptr, 'env', getattr(env_ptr, 'unwrapped', env_ptr))
        return env_ptr

    def _extract_global_map(self):
        """提取全局地图，并严格固定尺寸为 40x40，补充通道维度"""
        if hasattr(self.unwrapped, 'grid') and hasattr(self.unwrapped.grid, 'get_obstacles'):
            obstacles = self.unwrapped.grid.get_obstacles()
        elif hasattr(self.unwrapped, 'grid') and hasattr(self.unwrapped.grid, 'obstacles'):
            obstacles = self.unwrapped.grid.obstacles
        else:
            raise AttributeError("无法在底层 POGEMA 环境中找到 obstacles 矩阵。")

        obs_map = np.array(obstacles, dtype=np.float32)

        # 锁定最大尺寸，包容 observation_radius 带来的边界扩展
        MAX_SIZE = 40
        padded_map = np.zeros((MAX_SIZE, MAX_SIZE), dtype=np.float32)
        h, w = obs_map.shape
        
        h_max = min(h, MAX_SIZE)
        w_max = min(w, MAX_SIZE)
        padded_map[:h_max, :w_max] = obs_map[:h_max, :w_max]

        # 核心：增加通道维度，从 (40, 40) 变成 (1, 40, 40)
        padded_map = np.expand_dims(padded_map, axis=0)

        return padded_map

    def _format_obs(self, obs):
        """将包含 N 个观测值的列表转为严谨的 NumPy 数组，并补齐通道维度"""
        obs_array = np.array(obs, dtype=np.float32)
        
        # 如果数组形状是 (N, H, W) 缺少通道数，强制补齐为 (N, 1, H, W)
        if len(obs_array.shape) == 3:
            obs_array = np.expand_dims(obs_array, axis=1)
            
        return obs_array

    def reset(self, **kwargs):
        """重置环境并初始化"""
        reset_result = self.env.reset(**kwargs)
        
        # 兼容不同的 Gym 版本返回值 (obs) 或 (obs, info)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}

        # 1. 提取静态地图
        self.current_global_map = self._extract_global_map()

        # 2. 格式化局部观测
        obs_array = self._format_obs(obs)

        # 3. 清空历史记录
        self.episode_history = {
            'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []
        }

        return obs_array, info

    def step(self, actions):
        """单步执行，处理列表类型的输入输出"""
        # 确保传入的是标准的 Python list
        if isinstance(actions, np.ndarray):
            env_actions = actions.tolist()
        else:
            env_actions = actions

        # 执行环境 step
        next_obs, rewards, terminated, truncated, infos = self.env.step(env_actions)

        # 格式化返回值
        next_obs_array = self._format_obs(next_obs)
        rewards_array = np.array(rewards, dtype=np.float32)

        # 统一处理 done 信号 (合并 terminated 和 truncated)
        if isinstance(terminated, (list, tuple, np.ndarray)):
            dones = [t or tr for t, tr in zip(terminated, truncated)]
        else:
            # 兼容个别环境全局只返回一个 bool 的情况
            dones = [terminated or truncated] * self.num_agents
            
        dones_array = np.array(dones, dtype=bool)

        return next_obs_array, rewards_array, dones_array, truncated, infos

    def cache_step(self, obs, actions, rewards, next_obs, dones):
        self.episode_history['states'].append(np.array(obs, dtype=np.float32))
        self.episode_history['actions'].append(np.array(actions, dtype=np.int64))
        self.episode_history['rewards'].append(np.array(rewards, dtype=np.float32))
        self.episode_history['next_states'].append(np.array(next_obs, dtype=np.float32))
        self.episode_history['dones'].append(np.array([float(d) for d in dones], dtype=np.float32))

    def get_episode_data(self):
        return {
            'states': self.episode_history['states'],
            'actions': self.episode_history['actions'],
            'rewards': self.episode_history['rewards'],
            'next_states': self.episode_history['next_states'],
            'dones': self.episode_history['dones'],
            'global_map': self.current_global_map
        }