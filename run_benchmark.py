import os
import time
import yaml
import json
import torch
import numpy as np
from pathlib import Path
from pogema import pogema_v0, GridConfig, AnimationMonitor

# 导入你的模型和 Wrapper
from networks import SharedDRQN 
from env_wrapper import NativePogemaWrapper

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

def run_manual_benchmark():
    # ==========================================
    # 0. 配置与初始化
    # ==========================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    algorithm_name = "RQTMIX"  # 输出到 JSON 的算法名
    model_prefix = "Reptile-Finetuned"
    
    # 读取配置和地图
    try:
        with open(config_path['random']['config'], 'r', encoding='utf-8') as f:
            eval_config = yaml.safe_load(f)
        with open(config_path['random']['maps'], 'r', encoding='utf-8') as f:
            maps_dict = yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"❌ 找不到配置文件: {e}")
        return

    map_names = eval_config['environment']['map_name']['grid_search']
    num_agents_list = eval_config['environment']['num_agents']['grid_search']
    max_steps = eval_config['environment'].get('max_episode_steps', 256)
    
    # 初始化独立的 DRQN (遵守 CTDE)
    drqn = SharedDRQN(obs_channels=3, num_actions=5).to(device)
    drqn.eval()

    results_json = []
    print(f"🚀 开始执行纯净版手动测试管线 (脱离 pogema_toolbox 限制)...")

    # ==========================================
    # 1. 主测试循环
    # ==========================================
    for num_agents in num_agents_list:
        for map_name in map_names:
            map_str = maps_dict.get(map_name)
            if not map_str:
                print(f"⚠️ 在 maps.yaml 中未找到地图 {map_name}，跳过。")
                continue
                
            # 动态加载对应地图和智能体数量的微调模型
            model_path = f"./models/{model_prefix}_drqn_{map_name}_agents_{num_agents}.pth"
            if os.path.exists(model_path):
                drqn.load_state_dict(torch.load(model_path, map_location=device))
            else:
                print(f"⚠️ 找不到对应的微调模型 {model_path}，跳过该项测试。")
                continue

            print(f"Testing: {map_name} | Agents: {num_agents}")

            # 初始化底层环境与 Wrapper (与 fine_tune.py 完全一致)
            grid_config = GridConfig(
                map=map_str, 
                num_agents=num_agents, 
                max_episode_steps=max_steps,
                observation_radius=5,
                on_target='finish',
                seed=0
            )
            raw_env = pogema_v0(grid_config=grid_config)
            env_with_monitor = AnimationMonitor(raw_env)
            env = NativePogemaWrapper(env_with_monitor, num_agents)

            obs, infos = env.reset()
            hidden_state = drqn.init_hidden(batch_size=num_agents)
            
            done = False
            start_time = time.time()
            total_rewards = np.zeros(num_agents)
            
            # ==========================================
            # 2. 推理循环
            # ==========================================
            with torch.no_grad():
                while not done:
                    # 转换观测 (num_agents, 3, H, W)
                    obs_tensor = torch.tensor(np.array(obs, dtype=np.float32), device=device)
                    
                    q_values, _, hidden_state = drqn(obs_tensor, hidden_state)
                    actions = q_values.argmax(dim=-1).cpu().numpy().tolist()
                    
                    # 步进环境
                    next_obs, rewards, dones, truncated, step_infos = env.step(actions)
                    
                    obs = next_obs
                    total_rewards += np.array(rewards)
                    
                    # 检查是否全部结束 (兼容列表或标量)
                    if isinstance(dones, (list, tuple, np.ndarray)):
                        done = all(dones) or (all(truncated) if isinstance(truncated, (list, tuple, np.ndarray)) else truncated)
                    else:
                        done = dones or truncated

            runtime = time.time() - start_time

            env_with_monitor.save_animation(f"{model_path}_{map_name}_{num_agents}.svg")
            # ==========================================
            # 3. 提取与计算指标 (严格对齐 QMIX.json)
            # ==========================================
            metrics = {
                "avg_throughput": 0.0,
                "a_collisions": 0,
                "o_collisions": 0,
                "runtime": float(runtime),
                "avg_agents_density": num_agents / (grid_config.width * grid_config.height) if grid_config.width else 0.0
            }
            
            # 尝试从 POGEMA 原生 info 中提取官方指标
            if step_infos and isinstance(step_infos, list) and len(step_infos) > 0 and 'metrics' in step_infos[0]:
                m = step_infos[0]['metrics']
                metrics["avg_throughput"] = float(m.get('avg_throughput', sum(total_rewards)/num_agents))
                metrics["a_collisions"] = int(m.get('a_collisions', 0))
                metrics["o_collisions"] = int(m.get('o_collisions', 0))
            else:
                # Fallback：如果 Wrapper 吃掉了 metrics，则做简单估算
                metrics["avg_throughput"] = float(sum(total_rewards) / max_steps)

            # 组装 JSON 条目
            results_json.append({
                "metrics": metrics,
                "env_grid_search": {
                    "num_agents": num_agents,
                    "map_name": map_name
                },
                "algorithm": algorithm_name
            })

    # ==========================================
    # 4. 导出 JSON
    # ==========================================
    save_dir = Path('./my_rqtmix_results')
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"{algorithm_name}_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2)
        
    print(f"\n✅ 测试完全结束！共有 {len(results_json)} 条结果已保存至 {output_file.absolute()}")

if __name__ == '__main__':
    run_manual_benchmark()