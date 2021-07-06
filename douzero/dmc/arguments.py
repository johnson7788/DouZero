import argparse

parser = argparse.ArgumentParser(description='DouZero: PyTorch DouDizhu AI')

# General Settings
parser.add_argument('--xpid', default='douzero',
                    help='实验的id (default: douzero)')
parser.add_argument('--save_interval', default=30, type=int,
                    help='保存模型的时间间隔')
parser.add_argument('--objective', default='adp', type=str, choices=['adp', 'wp'],
                    help='使用ADP的目标还是WP的目标，ADP表示加倍的玩法，WP是只是胜率，没有加倍的那种分数(default: ADP)')

# Training settings
parser.add_argument('--gpu_devices', default='0', type=str,
                    help='哪个GPU用于训练')
parser.add_argument('--num_actor_devices', default=1, type=int,
                    help='默认用于模拟的的设备数量，默认一个actor')
parser.add_argument('--num_actors', default=5, type=int,
                    help='每个模拟的设备的actor的数量')
parser.add_argument('--training_device', default=0, type=int,
                    help='哪个GPU用于训练模型，就一个learner')
parser.add_argument('--load_model', action='store_true',
                    help='是否加载一个已经存在的模型，继续训练，加载checkpoint的路径根据你给定的xpid确定')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='是否disable checkpoint')
parser.add_argument('--savedir', default='douzero_checkpoints',
                    help='实验数据的保存的root目录')

# Hyperparameters
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='用于训练的总的环境的帧数')
parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='随机探索的概率')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Learner的batch size大小')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=50, type=int,
                    help='缓存的数量，记录每个操作和状态等，Number of shared-memory buffers')
parser.add_argument('--num_threads', default=4, type=int,
                    help='Number learner threads')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    help='最大的梯度裁正则')

# Optimizer settings
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon')
