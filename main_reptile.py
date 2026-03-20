import os
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from networks import SharedDRQN, StaticMapEncoder, TransformerMixer
# 注意：你需要将原来的 run_worker_task 改为常驻 worker 的逻辑
from worker_process import persistent_worker_process

def meta_update(global_model, deltas, alpha_meta):
    """
    Performs the Reptile meta-update.
    注意：为了防止多进程下梯度/权重的竞态条件，建议在无梯度上下文中更新。
    """
    with torch.no_grad():
        for name, param in global_model.state_dict().items():
            # Compute mean Delta across all workers
            mean_delta = torch.mean(torch.stack([d[name].to(param.device) for d in deltas]), dim=0)
            # Apply Meta-Update
            param.add_(alpha_meta * mean_delta)

if __name__ == '__main__':
    # 必须使用 spawn，特别是在使用 CUDA 时
    mp.set_start_method('spawn', force=True)

    # -------------------------------------------------------------
    # 1. Configuration and Hyperparameters
    # -------------------------------------------------------------
    config = {
        'num_workers': 8,
        'meta_iterations': 8000,
        'alpha_meta': 0.001,          
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # ... [保留你原有的其他配置] ...
        'num_agents': 8,
        'obs_channels': 3,          
        'map_channels': 1,          
        'num_actions': 5,           
        'hidden_dim': 128,
        # Inner-loop Hyperparameters
        'inner_epochs': 48,
        'batch_size': 32,
        'seq_len': 120,             # For TBPTT Burn-in and Learn phases
        'buffer_capacity': 1000,
        'inner_lr': 1e-4,
        'gamma': 0.99,
        'epsilon': 0.1,
        'max_steps': 128,
        'env_name': 'Pogema-8x8-normal-v0' # Will be dynamically masked/changed in workers ideally
    }

    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    writer = SummaryWriter(log_dir='./logs')
    print(f"[{config['device'].upper()}] Initialized TensorBoard logging to ./logs/")

    # -------------------------------------------------------------
    # 2. 初始化全局网络并将其放入共享内存 (Shared Memory)
    # -------------------------------------------------------------
    # 建议将全局模型保留在 CPU 上进行 share_memory，Worker 拉取时再放到 GPU，
    # 这样可以最大程度避免多进程抢占同一块 GPU 显存导致的碎片化和 OOM。
    global_drqn = SharedDRQN(config['obs_channels'], config['num_actions']).cpu()
    global_map_encoder = StaticMapEncoder(config['map_channels']).cpu()
    global_mixer = TransformerMixer(config['num_agents']).cpu()

    global_drqn.share_memory()
    global_map_encoder.share_memory()
    global_mixer.share_memory()

    global_models = {
        'drqn': global_drqn,
        'map_encoder': global_map_encoder,
        'mixer': global_mixer
    }

    print("Global Meta-Models Initialized and shared in memory.")

    # -------------------------------------------------------------
    # 3. 建立通信队列与启动常驻 Worker 进程
    # -------------------------------------------------------------
    # 主进程向 Worker 发送任务信号的队列
    task_queues = [mp.Queue() for _ in range(config['num_workers'])]
    # Worker 向主进程返回 Deltas 和 Metrics 的队列
    result_queue = mp.Queue()

    workers = []
    print(f"Starting {config['num_workers']} persistent workers...")
    for worker_id in range(config['num_workers']):
        p = mp.Process(
            target=persistent_worker_process, 
            args=(worker_id, global_models, task_queues[worker_id], result_queue, config)
        )
        p.start()
        workers.append(p)

    # -------------------------------------------------------------
    # 4. Meta-Learning Loop (Reptile)
    # -------------------------------------------------------------
    try:
        for meta_iter in range(config['meta_iterations']):
            
            # 课程学习进度
            config['curr_progress'] = meta_iter / max(1, (config['meta_iterations'] - 1))

            # 步骤 A：下发任务信号 (只传非常轻量的数据)
            for w_id in range(config['num_workers']):
                task_msg = {
                    'meta_iter': meta_iter,
                    'curr_progress': config['curr_progress']
                }
                task_queues[w_id].put(task_msg)

            # 步骤 B：收集结果 (由于 global_models 已通过共享内存读取，此处只回收 Deltas)
            deltas_drqn, deltas_map_encoder, deltas_mixer = [], [], []
            total_loss, total_reward, total_success = 0.0, 0.0, 0.0

            for _ in range(config['num_workers']):
                # 阻塞等待任何一个 Worker 完成
                w_id, worker_deltas, metrics = result_queue.get()
                
                deltas_drqn.append(worker_deltas['drqn'])
                deltas_map_encoder.append(worker_deltas['map_encoder'])
                deltas_mixer.append(worker_deltas['mixer'])

                total_loss += metrics['loss']
                total_reward += metrics['reward']
                total_success += metrics['success_rate']

            # 步骤 C：执行 Reptile 元更新
            meta_update(global_drqn, deltas_drqn, config['alpha_meta'])
            meta_update(global_map_encoder, deltas_map_encoder, config['alpha_meta'])
            meta_update(global_mixer, deltas_mixer, config['alpha_meta'])

            # 步骤 D：日志与保存
            avg_loss = total_loss / config['num_workers']
            avg_reward = total_reward / config['num_workers']
            avg_success = total_success / config['num_workers']

            writer.add_scalar('Meta/Loss', avg_loss, meta_iter)
            writer.add_scalar('Meta/Reward', avg_reward, meta_iter)
            writer.add_scalar('Meta/Success_Rate', avg_success, meta_iter)

            print(f"Meta-Iter {meta_iter+1}/{config['meta_iterations']} | "
                  f"Loss: {avg_loss:.4f} | Reward: {avg_reward:.4f} | Success: {avg_success:.2%}")

            if (meta_iter + 1) % 50 == 0:
                torch.save(global_drqn.state_dict(), f'./models/global_drqn_iter_{meta_iter+1}.pth')
                torch.save(global_map_encoder.state_dict(), f'./models/global_map_encoder_iter_{meta_iter+1}.pth')
                torch.save(global_mixer.state_dict(), f'./models/global_mixer_iter_{meta_iter+1}.pth')
                print(f"Saved Checkpoint to ./models/ at Meta-Iteration {meta_iter+1}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current models...")
    except Exception as e:
        print(f"\nException occurred: {e}")
    finally:
        # -------------------------------------------------------------
        # 5. 安全清理与退出
        # -------------------------------------------------------------
        print("Shutting down workers...")
        for q in task_queues:
            q.put("TERMINATE") # 发送毒丸 (Poison Pill) 终止 Worker
        
        for p in workers:
            p.join() # 等待进程安全退出

        torch.save(global_drqn.state_dict(), f'./models/global_drqn_iter_{meta_iter+1}.pth')
        torch.save(global_map_encoder.state_dict(), f'./models/global_map_encoder_iter_{meta_iter+1}.pth')
        torch.save(global_mixer.state_dict(), f'./models/global_mixer_iter_{meta_iter+1}.pth')
        print(f"Saved Checkpoint to ./models/ at Meta-Iteration {meta_iter+1}")
        writer.close()
        print("Meta-Training Complete and resources cleaned up.")