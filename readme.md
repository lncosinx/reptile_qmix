下载pytorch:https://pytorch.org/get-started/locally/

安装pogema-toolbox:
'''bash
cd reptile_qmix
git clone https://github.com/Cognitive-AI-Systems/pogema-toolbox.git
cd pogema-toolbox
pip install -e .
'''

安装rust编译工具：
'''bash
pip install maturin
'''

编译rust_buffer:
'''bash
maturin develop
'''

日志查看：
'''bash
tensorboard --logdir=./logs
'''

开启NVIDIA MPS：
'''bash
nvidia-cuda-mps-control -d
'''

关闭AMP（自动混合精度）：
'''txt
将agent_trainer.py中的AgentTrainer类的__init__方法中的
self.use_scaler = torch.cuda.is_available()
改为
self.use_scaler = False
'''
