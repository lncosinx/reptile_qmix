下载pytorch:https://pytorch.org/get-started/locally/

安装pogema-toolbox:
```bash
cd reptile_qmix
git clone https://github.com/Cognitive-AI-Systems/pogema-toolbox.git
cd pogema-toolbox
pip install -e .
```

安装rust编译工具：
```bash
pip install maturin
```

编译rust_buffer:
```bash
maturin develop
```

日志查看：
```bash
tensorboard --logdir=./logs
```

开启NVIDIA MPS：
```bash
nvidia-cuda-mps-control -d
```

关闭AMP（自动混合精度）：
```txt
将agent_trainer.py中的AgentTrainer类的__init__方法中的
self.use_scaler = torch.cuda.is_available()
改为
self.use_scaler = False
```

下载screen:
```bash
sudo apt update
sudo apt install screen
```
查看当前有哪些会话：screen -ls。

重新连接之前的会话：screen -r <会话ID或名称>。

彻底退出并关闭会话：在会话内输入 exit 或按下 Ctrl + d

端口转发：
VS Code 的 Remote 系列插件（包括 Tunnels）内置了一个端口转发功能。

操作步骤：

在 VS Code 底部状态栏找到 端口 (Ports) 面板。

点击 添加端口 (Forward a Port)。

输入你本地代理软件的端口号。

VS Code 会将服务器上的该端口映射到你本地。

如何使用：
在服务器终端执行：

```bash
export http_proxy=http://127.0.0.1:10808
export https_proxy=http://127.0.0.1:10808
```
微调与测试
微调：
```bash
python -u fine_tune.py
```
测试：
```bash
python -u run_benchmark.py
```  
注意，由于pogema官方硬编码了map的路径，所以请把需要测试的map.yaml放在reptile_qmix/pogema-toolbox/pogema_toolbox/maps文件夹下

