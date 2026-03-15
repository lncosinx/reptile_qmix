import os
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from networks import SharedDRQN, StaticMapEncoder, TransformerMixer
from worker_process import run_worker_task

def meta_update(global_model, deltas, alpha_meta):
    """
    Performs the Reptile meta-update: \\theta_{global} <- \\theta_{global} + \\alpha_{meta} * \\frac{1}{Batch} \\sum \\Delta_i
    """
    for name, param in global_model.state_dict().items():
        # Compute mean Delta across all workers
        mean_delta = torch.mean(torch.stack([d[name] for d in deltas]), dim=0)

        # Apply Meta-Update
        new_param = param.cpu() + alpha_meta * mean_delta

        # Load back to global model
        param.copy_(new_param.to(param.device))

if __name__ == '__main__':
    # Required for PyTorch multiprocessing to avoid deadlocks
    mp.set_start_method('spawn')

    # -------------------------------------------------------------
    # 1. Configuration and Hyperparameters
    # -------------------------------------------------------------
    config = {
        'num_workers': 8,
        'meta_iterations': 6400,
        'alpha_meta': 0.1,          # Meta-learning rate
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # Network & Env Hyperparameters
        'num_agents': 8,
        'obs_channels': 3,          # e.g., Agent, Target, Obstacles, Other Agents
        'map_channels': 1,          # Static Map usually has 1 channel (Obstacles)
        'num_actions': 5,           # Up, Down, Left, Right, Stay
        'hidden_dim': 128,

        # Inner-loop Hyperparameters
        'inner_epochs': 128,
        'batch_size': 32,
        'seq_len': 120,             # For TBPTT Burn-in and Learn phases
        'buffer_capacity': 1000,
        'inner_lr': 1e-4,
        'gamma': 0.99,
        'epsilon': 0.1,
        'max_steps': 128,
        'env_name': 'Pogema-8x8-normal-v0' # Will be dynamically masked/changed in workers ideally
    }

    # -------------------------------------------------------------
    # 2. Setup File Management (Directories and Logging)
    # -------------------------------------------------------------
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir='./logs')
    print(f"[{config['device'].upper()}] Initialized TensorBoard logging to ./logs/")

    # -------------------------------------------------------------
    # 3. Initialize Global \theta_{global} Networks
    # -------------------------------------------------------------
    global_drqn = SharedDRQN(config['obs_channels'], config['num_actions']).to(config['device'])
    global_map_encoder = StaticMapEncoder(config['map_channels']).to(config['device'])
    global_mixer = TransformerMixer(config['num_agents']).to(config['device'])

    print("Global Meta-Models Initialized.")

    # -------------------------------------------------------------
    # 4. Meta-Learning Loop (Reptile)
    # -------------------------------------------------------------
    print(f"Starting Reptile Meta-Training with {config['num_workers']} workers...")

    # Setup Pool outside the training loop to avoid huge overhead of recreating processes
    with mp.Pool(processes=config['num_workers']) as pool:
        for meta_iter in range(config['meta_iterations']):
            
            # 计算当前课程学习的进度 (0.0 到 1.0 之间)
            if config['meta_iterations'] > 1:
                config['curr_progress'] = meta_iter / (config['meta_iterations'] - 1)
            else:
                config['curr_progress'] = 1.0

            # Snapshot the current global weights to send to workers
            # We move them to CPU before pickling to save GPU VRAM and IPC overhead
            global_state_dict = {
                'drqn': {k: v.cpu().clone() for k, v in global_drqn.state_dict().items()},
                'map_encoder': {k: v.cpu().clone() for k, v in global_map_encoder.state_dict().items()},
                'mixer': {k: v.cpu().clone() for k, v in global_mixer.state_dict().items()}
            }

            # Dispatch workers
            results = []
            for worker_id in range(config['num_workers']):
                # In a real scenario, we might pass different `env_name` configurations
                # to different workers to enforce task heterogeneity (e.g., different map sizes)

                result = pool.apply_async(run_worker_task, args=(worker_id, global_state_dict, config))
                results.append(result)

            # Collect \Delta_i from all workers
            worker_outputs = [res.get() for res in results]
            # Parse results
            deltas_drqn = []
            deltas_map_encoder = []
            deltas_mixer = []

            total_loss = 0.0
            total_reward = 0.0
            total_success = 0.0

            for w_id, deltas, metrics in worker_outputs:
                deltas_drqn.append(deltas['drqn'])
                deltas_map_encoder.append(deltas['map_encoder'])
                deltas_mixer.append(deltas['mixer'])

                total_loss += metrics['loss']
                total_reward += metrics['reward']
                total_success += metrics['success_rate']

            # -------------------------------------------------------------
            # 5. Reptile Meta-Update (\\theta_{global} <- \\theta_{global} + \\alpha_{meta} \* \\frac{1}{Batch} \\sum \\Delta_i)
            # -------------------------------------------------------------
            meta_update(global_drqn, deltas_drqn, config['alpha_meta'])
            meta_update(global_map_encoder, deltas_map_encoder, config['alpha_meta'])
            meta_update(global_mixer, deltas_mixer, config['alpha_meta'])

            # -------------------------------------------------------------
            # 6. Logging & Checkpointing
            # -------------------------------------------------------------
            avg_loss = total_loss / config['num_workers']
            avg_reward = total_reward / config['num_workers']
            avg_success = total_success / config['num_workers']

            writer.add_scalar('Meta/Loss', avg_loss, meta_iter)
            writer.add_scalar('Meta/Reward', avg_reward, meta_iter)
            writer.add_scalar('Meta/Success_Rate', avg_success, meta_iter)

            print(f"Meta-Iter {meta_iter+1}/{config['meta_iterations']} | "
                  f"Loss: {avg_loss:.4f} | Reward: {avg_reward:.4f} | Success: {avg_success:.2%}")

            # Save checkpoints periodically
            if (meta_iter + 1) % 50 == 0:
                torch.save(global_drqn.state_dict(), f'./models/global_drqn_iter_{meta_iter+1}.pth')
                torch.save(global_map_encoder.state_dict(), f'./models/global_map_encoder_iter_{meta_iter+1}.pth')
                torch.save(global_mixer.state_dict(), f'./models/global_mixer_iter_{meta_iter+1}.pth')
                print(f"Saved Checkpoint to ./models/ at Meta-Iteration {meta_iter+1}")

    # Final Save
    torch.save(global_drqn.state_dict(), './models/global_drqn_final.pth')
    torch.save(global_map_encoder.state_dict(), './models/global_map_encoder_final.pth')
    torch.save(global_mixer.state_dict(), './models/global_mixer_final.pth')

    writer.close()
    print("Meta-Training Complete. Final models saved to ./models/")
