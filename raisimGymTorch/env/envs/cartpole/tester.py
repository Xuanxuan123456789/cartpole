from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import cartpole
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymTorch.algo.ppo.module as ppo_module
import os
import math
import time
import torch
import argparse

# 配置解析
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='训练好的模型路径', type=str, default='')
args = parser.parse_args()

# 目录配置
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = os.path.join(task_path, "../../../../..")

# 加载配置文件
cfg_path = os.path.join(task_path, "cfg.yaml")
cfg = YAML().load(open(cfg_path, 'r'))

# 设置单环境测试
cfg['environment']['num_envs'] = 1

# 创建环境
env = VecEnv(
    cartpole.RaisimGymEnv(
        resourceDir=task_path,
        cfg=dump(cfg['environment'], Dumper=RoundTripDumper)
    ),
    cfg['environment']
)

# 获取观测和动作维度
ob_dim = env.num_obs
act_dim = env.num_acts

# 加载模型
weight_path = args.weight
weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'
if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], torch.nn.LeakyReLU, ob_dim, act_dim)
    loaded_graph.load_state_dict(torch.load(weight_path,weights_only=True)['actor_architecture_state_dict'])
    env.load_scaling(weight_dir, int(iteration_number))
    #env.turn_on_visualization()

    # max_steps = 1000000
    max_steps = 1000 ## 10 secs

    for step in range(max_steps):
        time.sleep(0.01)
        obs = env.observe(False)
        action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())
        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
        reward_ll_sum = reward_ll_sum + reward_ll[0]
        if dones or step == max_steps - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * 0.01)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0
