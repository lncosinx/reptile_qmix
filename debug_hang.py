import time
import sys

print(f"[{time.ctime()}] 1. 准备加载 PyTorch...")
import torch
print(f"[{time.ctime()}] 2. PyTorch 加载成功！")

print(f"[{time.ctime()}] 3. 准备加载 Multiprocessing...")
import torch.multiprocessing as mp
print(f"[{time.ctime()}] 4. MP 加载成功！")

print(f"[{time.ctime()}] 5. 准备加载 TensorBoard...")
from torch.utils.tensorboard import SummaryWriter
print(f"[{time.ctime()}] 6. TensorBoard 加载成功！")

print(f"[{time.ctime()}] 7. 准备加载自定义 Rust 库与环境...")
from pogema import pogema_v0
from rust_buffer import RustReplayBuffer
print(f"[{time.ctime()}] 8. 所有库 Import 完毕！")

print(f"\n--- 开始测试核心组件 ---")

print("测试 A: TensorBoard 初始化 (如果卡在这里，说明残留代理拦截了本机的 DNS 查询)")
writer = SummaryWriter(log_dir='./logs_debug')
print("测试 A 通过！")

print("测试 B: 多进程队列 (如果卡在这里，说明残留代理拦截了 127.0.0.1 端口)")
q = mp.Queue()
q.put("Hello_Queue")
res = q.get()
print(f"测试 B 通过！获取到数据: {res}")

print("\n🎉 如果你能看到这句话，说明基础环境完全没问题！")