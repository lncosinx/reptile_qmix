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
        self.episode_history = {}

    def _get_unwrapped_env(self):
        """层层剥开 Gym 的皮，找到真正包含 map/grid 的原生对象"""
        env_ptr = self.env
        while hasattr(env_ptr, 'env') or hasattr(env_ptr, 'unwrapped'):
            if hasattr(env_ptr, 'grid'):
                break
            if hasattr(env_ptr, 'unwrapped') and env_ptr.unwrapped == env_ptr:
                break
            env_ptr = getattr(env_ptr, 'env', getattr(env_ptr, 'unwrapped', env_ptr))
        return env_ptr

    def _extract_global_map(self):
        """
        提取全局地图，并严格固定尺寸为 MAX_SIZE，补充通道维度
        MAX_SIZE计算公式：MAX_SIZE = Max_Map_Size + 2 * observation_radius
        """
        if hasattr(self.unwrapped, 'grid') and hasattr(self.unwrapped.grid, 'get_obstacles'):
            obstacles = self.unwrapped.grid.get_obstacles()
        elif hasattr(self.unwrapped, 'grid') and hasattr(self.unwrapped.grid, 'obstacles'):
            obstacles = self.unwrapped.grid.obstacles
        else:
            raise AttributeError("无法在底层 POGEMA 环境中找到 obstacles 矩阵。")

        obs_map = np.array(obstacles, dtype=np.float32)

        # MAX_SIZE计算公式：MAX_SIZE = Max_Map_Size + 2 * observation_radius
        MAX_SIZE = 40
        padded_map = np.zeros((MAX_SIZE, MAX_SIZE), dtype=np.float32)
        h, w = obs_map.shape
        
        h_max = min(h, MAX_SIZE)
        w_max = min(w, MAX_SIZE)
        padded_map[:h_max, :w_max] = obs_map[:h_max, :w_max]

        padded_map = np.expand_dims(padded_map, axis=0)
        return padded_map

    def _format_obs(self, obs):
        """将包含 N 个观测值的列表转为严谨的 NumPy 数组，并补齐通道维度"""
        obs_array = np.array(obs, dtype=np.float32)
        if len(obs_array.shape) == 3:
            obs_array = np.expand_dims(obs_array, axis=1)
        return obs_array

    def reset(self, **kwargs):
        """重置环境并初始化"""
        reset_result = self.env.reset(**kwargs)
        
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}

        self.current_global_map = self._extract_global_map()
        obs_array = self._format_obs(obs)

        # 🌟 修复 2.1: 在 history 中增加 agent_coords 列表
        self.episode_history = {
            'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': [],
            'agent_coords': [] 
        }

        return obs_array, info

    def step(self, actions):
        """单步执行，处理列表类型的输入输出"""
        if isinstance(actions, np.ndarray):
            env_actions = actions.tolist()
        else:
            env_actions = actions

        next_obs, rewards, terminated, truncated, infos = self.env.step(env_actions)

        next_obs_array = self._format_obs(next_obs)
        rewards_array = np.array(rewards, dtype=np.float32)

        if isinstance(terminated, (list, tuple, np.ndarray)):
            dones = [t or tr for t, tr in zip(terminated, truncated)]
        else:
            dones = [terminated or truncated] * self.num_agents
            
        dones_array = np.array(dones, dtype=bool)

        return next_obs_array, rewards_array, dones_array, truncated, infos

    # 🌟 修复 2.2: 方法签名增加 agent_coords 参数
    def cache_step(self, obs, actions, rewards, next_obs, dones, agent_coords=None):
        self.episode_history['states'].append(np.array(obs, dtype=np.float32))
        self.episode_history['actions'].append(np.array(actions, dtype=np.int64))
        self.episode_history['rewards'].append(np.array(rewards, dtype=np.float32))
        self.episode_history['next_states'].append(np.array(next_obs, dtype=np.float32))
        self.episode_history['dones'].append(np.array([float(d) for d in dones], dtype=np.float32))
        
        if agent_coords is not None:
            self.episode_history['agent_coords'].append(np.array(agent_coords, dtype=np.float32))
        else:
            dummy_coords = np.zeros((self.num_agents, 2), dtype=np.float32)
            self.episode_history['agent_coords'].append(dummy_coords)

    def get_episode_data(self):
        return {
            'states': self.episode_history['states'],
            'actions': self.episode_history['actions'],
            'rewards': self.episode_history['rewards'],
            'next_states': self.episode_history['next_states'],
            'dones': self.episode_history['dones'],
            # 🌟 修复 2.4: 传递给 Rust Buffer
            'agent_coords': self.episode_history['agent_coords'], 
            'global_map': self.current_global_map
        }
    
    def get_normalized_coords(self):
        """
        获取当前所有智能体的归一化全局坐标 (x, y)
        专供 CTDE 架构下的 Mixer (上帝视角) 使用
        """
        unwrapped_env = self.env.unwrapped
        
        # 删掉动态获取 width 和 height 的逻辑
        # 统一使用和 _extract_global_map 一致的固定最大画布尺寸！
        MAX_SIZE = 40.0 

        positions = unwrapped_env.grid.positions_xy if hasattr(unwrapped_env.grid, 'positions_xy') else unwrapped_env.grid.positions
        
        coords = []
        for pos in positions:
            # 兼容可能的旧版格式
            x, y = pos[0], pos[1] if isinstance(pos, (list, tuple)) else pos
            
            # 🌟 核心修复：绝对物理空间向统一归一化空间的无损映射
            # 这样保证了 1 个物理格子的距离永远是 0.025，彻底解决 sigma 的跨地图缩放畸变！
            norm_x = x / MAX_SIZE
            norm_y = y / MAX_SIZE
            coords.append([norm_x, norm_y])
            
        return np.array(coords, dtype=np.float32)