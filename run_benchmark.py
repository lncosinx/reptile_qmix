import os
import yaml
import torch
import numpy as np
from pathlib import Path

from pogema_toolbox.create_env import Environment, create_env_base
from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.evaluator import evaluation

# 导入你的 DRQN 网络（请确保路径正确）
from networks import SharedDRQN 

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

class MyQMIXPolicy:
    def __init__(self, **kwargs):
        # 自动识别环境：笔记本测试优先用 CPU/轻量级 CUDA，4090 服务器上自动满血调用 CUDA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 严格遵守 CTDE 原则，只实例化局部 Q 网络
        self.drqn = SharedDRQN(obs_channels=3, num_actions=5).to(self.device)
        self.drqn.eval() # 锁定评估模式
        
        self.hidden_state = None
        self.current_model_path = ""
        self.algorithm_name = "Reptile-Finetuned"
        
        print(f"RQTMIX 策略初始化完成。当前推理设备: {self.device}")

    def reset_states(self):
        """
        官方工具箱必需的接口。
        在每个新的 Episode 开始前，工具箱会自动调用这里，清空 RNN 的记忆。
        """
        self.hidden_state = None

    def act(self, observations, rewards, dones, info, **kwargs):
        num_agents = len(observations)
        
        # Pogema Toolbox 在评测时，会将当前环境的配置信息传入 kwargs 或 info 中
        # 这里提取 map_name 以便加载对应的微调模型
        map_name = kwargs.get('map_name', 'unknown_map')
        if info and isinstance(info, list) and 'map_name' in info[0]:
            map_name = info[0]['map_name']

        expected_model_path = f"./models/{self.algorithm_name}_drqn_{map_name}_agents_{num_agents}.pth"

        # 判断是否需要重置隐藏状态或切换权重
        # 条件：1. 刚初始化; 2. 所有智能体回合结束(dones); 3. 切图或切换智能体数量了
        if self.hidden_state is None or all(dones) or self.current_model_path != expected_model_path:
            
            # 动态加载对应地图和数量的微调模型
            if self.current_model_path != expected_model_path:
                if os.path.exists(expected_model_path):
                    self.drqn.load_state_dict(torch.load(expected_model_path, map_location=self.device))
                    self.current_model_path = expected_model_path
                else:
                    print(f"警告: 未找到微调模型 {expected_model_path}，当前可能在使用旧权重运行。")
            
            # 重置隐藏状态 (batch_size = num_agents)
            h = torch.zeros(num_agents, 128, device=self.device)
            c = torch.zeros(num_agents, 128, device=self.device)
            self.hidden_state = (h, c)

        # 进行纯贪婪推理 (无探索，epsilon=0)
        with torch.no_grad():
            # 将观测转换为张量，形状通常为 (num_agents, channels, height, width)
            obs_tensor = torch.tensor(np.array(observations, dtype=np.float32), device=self.device)
            
            # DRQN 只需要局部观测和前一时刻的隐藏状态，不传入全局地图(Map Encoder)
            q_values, _, self.hidden_state = self.drqn(obs_tensor, self.hidden_state)
            
            # 取 Q 值最大的动作作为输出
            actions = q_values.argmax(dim=-1).cpu().numpy().tolist()

        return actions

def main():
    # 1. 注册基础环境
    ToolboxRegistry.register_env('Environment', create_env_base, Environment)
    
    # 2. 注册你的算法，名字 "RQTMIX" 必须与 yaml 中的键名一致
    ToolboxRegistry.register_algorithm('RQTMIX', MyQMIXPolicy)
    
    # 3. 读取配置文件
    config_path = config_path['random']['config']
    with open(config_path, 'r', encoding='utf-8') as f:
        eval_config = yaml.safe_load(f)
        
    # 4. 设置结果保存目录并启动评测
    save_dir = Path('./my_rqtmix_results')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("开始执行评测任务...")
    
    # evaluation 自动接管 grid_search 的遍历，并收集 metrics 生成 JSON
    evaluation(eval_config, eval_dir=save_dir)
    
    print(f"评测完成！结果包含 avg_throughput, a_collisions 等指标，已保存至 {save_dir.absolute()}")

if __name__ == '__main__':
    main()