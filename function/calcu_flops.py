'''
计算模型的吞吐量
'''

import torch
from tqdm import tqdm
import time
def calcu_flops(model, args):


    # 假设model是你的PyTorch模型，input_tensor是一个批次的输入数据
    model = model  # 你的模型
    input_tensor = torch.randn(args.batch_size, 3, 224, 224) # 你的输入数据，例如一个批次的图像

    # 确保模型处于评估模式
    model.eval()

    total_time = 0

    repetitions = 10

    # 启用torch.no_grad()来减少计算资源的消耗
    with torch.no_grad():
        # 进行预热运行，以消除第一次运行时的初始化开销
        print('___________预热_________________')
        for _ in range(10):
            _ = model(input_tensor)

        start_time = time.time()
        # 进行一定次数的迭代以获得更稳定的测量结果
        print('___________计算_________________')
        for _ in tqdm(range(repetitions)):
            # 这里更加专业，使用GPU计时，而不是CPU计时
            #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # 开始计时
            #starter.record()
            _ = model(input_tensor)
            #结束计时
            #ender.record()
            torch.cuda.synchronize()
            #curr_time = starter.elapsed_time(ender) / 1000
            #total_time += curr_time

        end_time = time.time()
    print(f'CPUtime:{end_time-start_time}')
    total_number = repetitions * args.batch_size
    Throughput = total_number/ (end_time-start_time)

    return Throughput

    #print(f"模型吞吐量: {Throughput:.2f} 样本/秒")
