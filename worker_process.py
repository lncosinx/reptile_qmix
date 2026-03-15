import torch
import numpy as np
from env_wrapper import PogemaWrapper
from agent_trainer import AgentTrainer
from rust_buffer import RustReplayBuffer

# Note: POGEMA needs to be imported to register the environment, but we'll mock it if not present
try:
    import pogema
    import gymnasium as gym
except ImportError:
    pass

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

    # Initialize Environment
    raw_env = gym.make(env_name)
    env = PogemaWrapper(raw_env)

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

        while not done and step < max_steps:
            # Select actions (Decentralized Execution)
            actions, next_hidden_state = trainer.select_actions(obs, hidden_state, epsilon=config.get('epsilon', 0.1))

            # Step environment
            next_obs, rewards, dones, truncated, infos = env.step(actions)

            # Format outputs
            next_obs = np.array(next_obs, dtype=np.float32)

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
            # Sample batch from Rust Buffer
            batch = buffer.sample(batch_size)

            # Centralized Training Step
            loss = trainer.train_step(batch, gamma=config.get('gamma', 0.99))
            total_loss += loss

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
