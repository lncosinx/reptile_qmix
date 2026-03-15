import gymnasium as gym
import numpy as np

class PogemaWrapper(gym.Wrapper):
    """
    Wrapper for POGEMA environment.
    Responsible for handling dynamic grid sizes (15~25), fixed agent count (N=8),
    and correctly extracting the static global map upon reset.
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_agents = 8
        self.current_global_map = None

        # We need to maintain a history of episode steps to push to the Rust buffer
        self.episode_history = []

    def _extract_global_map(self):
        """
        Penetrates the underlying POGEMA environment to extract the static obstacle matrix.
        Returns it as a numpy array with shape (C_g, H_g, W_g), where C_g=1 for obstacles.
        """
        # POGEMA grid is stored in env.grid.obstacles
        # Handling different POGEMA versions/configurations if necessary,
        # but typically it's an attribute like env.unwrapped.grid.obstacles
        unwrapped = self.env.unwrapped

        if hasattr(unwrapped, 'grid') and hasattr(unwrapped.grid, 'get_obstacles'):
            # Some versions might have a method
            obstacles = unwrapped.grid.get_obstacles()
        elif hasattr(unwrapped, 'grid') and hasattr(unwrapped.grid, 'obstacles'):
            # Most common: access attributes directly
            obstacles = unwrapped.grid.obstacles
        else:
            raise AttributeError("Could not find the obstacles matrix in the underlying POGEMA environment.")

        # Convert to numpy and add Channel dimension: (1, H, W)
        obs_map = np.array(obstacles, dtype=np.float32)

        # Assuming obs_map is 2D (H, W). Add channel dimension
        if len(obs_map.shape) == 2:
            obs_map = np.expand_dims(obs_map, axis=0) # (1, H, W)

        return obs_map

    def reset(self, **kwargs):
        """
        Resets the environment, clears the episode history, and caches the global map.
        Handles Gymnasium >= 0.26 API.
        """
        reset_result = self.env.reset(**kwargs)
        # Handle cases where reset returns (obs, info) vs just obs
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs = reset_result

        # 1. Extract and cache the static global obstacle map
        # Dynamic sizes 15x15 ~ 25x25 will be inherently supported here
        # since we just take the matrix as is. The StaticMapEncoder will
        # handle the dynamic size via AdaptiveAvgPool2d.
        self.current_global_map = self._extract_global_map()

        # 2. Reset episode history
        self.episode_history = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
            # We don't store global_map for every step to save memory,
            # we bundle it at the end of the episode.
        }

        # Preserve original return signature of the env (usually tuple obs, info in Gym >= 0.26)
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            return obs, info
        return obs

    def step(self, actions):
        """
        Steps the environment and stores the transition locally.
        Handles Gymnasium >= 0.26 API.
        """
        next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

        # Combine terminated and truncated into dones for legacy compatibility or easier handling
        dones = [t or tr for t, tr in zip(terminated, truncated)] if isinstance(terminated, list) else (terminated or truncated)

        return next_obs, rewards, dones, truncated, infos

    def cache_step(self, obs, actions, rewards, next_obs, dones):
        """
        Explicitly caches the step data. Call this from the training loop
        after interacting with the environment.
        """
        self.episode_history['states'].append(np.array(obs, dtype=np.float32))
        self.episode_history['actions'].append(np.array(actions, dtype=np.int64))
        self.episode_history['rewards'].append(np.array(rewards, dtype=np.float32))
        self.episode_history['next_states'].append(np.array(next_obs, dtype=np.float32))

        # Assuming dones is a list of booleans, convert to floats for easier tensor ops
        dones_float = np.array([float(d) for d in dones], dtype=np.float32)
        self.episode_history['dones'].append(dones_float)

    def get_episode_data(self):
        """
        Bundles the local step data with the cached current_global_map
        for pushing into the Rust replay buffer.
        """
        episode_data = {
            'states': self.episode_history['states'],
            'actions': self.episode_history['actions'],
            'rewards': self.episode_history['rewards'],
            'next_states': self.episode_history['next_states'],
            'dones': self.episode_history['dones'],
            'global_map': self.current_global_map
        }
        return episode_data
