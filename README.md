# [ICML 2021] DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning
<img width="500" src="https://raw.githubusercontent.com/kwai/DouZero/main/imgs/douzero_logo.jpg" alt="Logo" />

[![Downloads](https://pepy.tech/badge/douzero)](https://pepy.tech/project/douzero)
[![Downloads](https://pepy.tech/badge/douzero/month)](https://pepy.tech/project/douzero)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daochenzha/douzero-colab/blob/main/douzero-colab.ipynb)

DouZero是一个针对[斗地主](https://en.wikipedia.org/wiki/Dou_dizhu)的强化学习框架，斗地主是中国最流行的纸牌游戏。这是一个甩牌游戏，玩家的目标是在其他玩家之前清空自己手中的所有牌。斗地主是一个非常具有挑战性的领域，有竞争、协作、不完美的信息、大的状态空间，特别是有大量可能的行动，其中合法的行动在不同的回合有很大的不同。斗地主是由人工智能平台(快手)开发。

*   Online Demo: [https://www.douzero.org/](https://www.douzero.org/)
*   Run the Demo Locally: [https://github.com/datamllab/rlcard-showdown](https://github.com/datamllab/rlcard-showdown)
*   Paper: [https://arxiv.org/abs/2106.06135](https://arxiv.org/abs/2106.06135) 
*   Related Project: [RLCard Project](https://github.com/datamllab/rlcard)
*   Related Resources: [Awesome-Game-AI](https://github.com/datamllab/awesome-game-ai)
*   Google Colab: [jupyter notebook](https://github.com/daochenzha/douzero-colab/blob/main/douzero-colab.ipynb)

**Community:**
*  **Slack**: Discuss in [DouZero](https://join.slack.com/t/douzero/shared_invite/zt-rg3rygcw-ouxxDk5o4O0bPZ23vpdwxA) channel.
*  **QQ Group**: Join our QQ group 819204202. Password: douzeroqqgroup

<img width="500" src="https://daochenzha.github.io/files/douzero-gif.gif" alt="Demo" />

## Cite this Work
For now, please cite our Arxiv version:

Zha, Daochen, et al. "DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning." arXiv preprint arXiv:2106.06135 (2021).

```bibtex
@article{zha2021douzero,
  title={DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning},
  author={Zha, Daochen and Xie, Jingru and Ma, Wenye and Zhang, Sheng and Lian, Xiangru and Hu, Xia and Liu, Ji},
  journal={arXiv preprint arXiv:2106.06135},
  year={2021}
}
```

## What Makes DouDizhu Challenging?
除了信息不完善的挑战外，DouDizhu有巨大的状态和行动空间。特别是，DouDizhu的行动空间是10^4（见[此表](https://github.com/datamllab/rlcard#available-environments)）。不幸的是，大多数强化学习算法只能处理非常小的行动空间。此外，斗地主中的玩家需要在一个部分可观察的环境中与他人竞争和合作，并且沟通有限，例如，两个农民玩家将作为一个团队与地主玩家进行对抗。对竞争和合作进行建模是一个开放的研究挑战。
在这项工作中，我们提出了带有动作编码和平行actor的深度蒙特卡洛（DMC）算法。这导致了一个非常简单但令人惊讶的斗地主的有效解决方案。请阅读[我们的论文]（https://arxiv.org/abs/2106.06135）以了解更多细节。

## Installation
Clone the repo with
```
git clone https://github.com/kwai/DouZero.git
```
Make sure you have python 3.6+ installed. Install dependencies:
```
cd douzero
pip3 install -r requirements.txt
```
我们建议使用安装稳定版本的Douzero 
```
pip3 install douzero
```
或安装最新版本（它可能不稳定） 
```
pip3 install -e .
```

## Training
我们假设您至少有一个可用的GPU。 Run
```
python3 train.py
```
这将在一个GPU上训练DouZero。要在多个GPU上训练DouZero。请使用以下参数。
*   `--gpu_devices`: GPU设备是可见的 
*   `--num_actors_devices`: 有多少个GPU节点将被用于模拟，即自我对战
*   `--num_actors`:每个设备将使用多少个actor进程 
*   `--training_device`: 哪个设备将用于训练Douzero 

例如，如果我们有4个GPU，其中我们想用前3个GPU分别有15个actor进行模拟，第4个GPU用于训练，我们可以运行以下命令。
```
python3 train.py --gpu_devices 0,1,2,3 --num_actors_devices 3 --num_actors 15 --training_device 3
```
有关更多自定义的训练配置，请参阅以下可选参数：
```
--xpid XPID           Experiment id (default: douzero)
--save_interval SAVE_INTERVAL
                      Time interval (in minutes) at which to save the model
--objective {adp,wp}  Use ADP or WP as reward (default: ADP)
--gpu_devices GPU_DEVICES
                      Which GPUs to be used for training
--num_actor_devices NUM_ACTOR_DEVICES
                      The number of devices used for simulation
--num_actors NUM_ACTORS
                      The number of actors for each simulation device
--training_device TRAINING_DEVICE
                      The index of the GPU used for training models
--load_model          Load an existing model
--disable_checkpoint  Disable saving checkpoint
--savedir SAVEDIR     Root dir where experiment data will be saved
--total_frames TOTAL_FRAMES
                      Total environment frames to train for
--exp_epsilon EXP_EPSILON
                      The probability for exploration
--batch_size BATCH_SIZE
                      Learner batch size
--unroll_length UNROLL_LENGTH
                      The unroll length (time dimension)
--num_buffers NUM_BUFFERS
                      Number of shared-memory buffers
--num_threads NUM_THREADS
                      Number learner threads
--max_grad_norm MAX_GRAD_NORM
                      Max norm of gradients
--learning_rate LEARNING_RATE
                      Learning rate
--alpha ALPHA         RMSProp smoothing constant
--momentum MOMENTUM   RMSProp momentum
--epsilon EPSILON     RMSProp epsilon
```

## Evaluation
评估可以用GPU或CPU进行（GPU会更快）。预训练模型可在[Google Drive](https://drive.google.com/drive/folders/1NmM2cXnI5CIWHaLJeoDZMiwt6lOTV_UB?usp=sharing)或[百度网盘](https://pan.baidu.com/s/18g-JUKad6D8rmBONXUDuOQ)下载，提取码。4624. 把预训练的权重放在`baselines/`中。通过自我对战来评估性能。我们已经提供了预训练的模型和一些启发式方法作为基线。
*   [random](douzero/evaluation/random_agent.py): agents that play randomly (uniformly)
*   [rlcard](douzero/evaluation/rlcard_agent.py): the rule-based agent in [RLCard](https://github.com/datamllab/rlcard)
*   SL (`baselines/sl/`): 预训练的人类数据的深度agent 
*   DouZero-ADP (`baselines/douzero_ADP/`): 预训练的Douzero agent，Average Difference Points（ADP）为目标
*   DouZero-WP (`baselines/douzero_WP/`): 预训练的Douzero agent，Winning Percentage（WP）作为目标

### Step 1: Generate evaluation data
```
python3 generate_eval_data.py
```
一些重要的超参数如下。 
*   `--output`: picked数据将保存
*   `--num_games`: 将生成多少个随机游戏，默认为10000

### Step 2: Self-Play
```
python3 evaluate.py
```
一些重要的超参数如下。 
*   `--landlord`: 哪个agent将扮演地主，可以是随机的，也可以是rlcard，或者是预训练过的模型的路径。 
*   `--landlord_up`: 哪个agent将作为地主上家（在地主之前上场的agent），可以是随机的，也可以是rlcard，或者是预训练好的模型的路径。 
*   `--landlord_down`: 哪一个agent将作为Landlord下家（在地主之后进行游戏），可以是随机的、rlcard或预训练好的模型的路径。
*   `--eval_data`: 包含评估数据的pickle文件 

例如，下面的命令在地主位置对随机agent进行DouZero-ADP评估
```
python3 evaluate.py --landlord baselines/douzero_ADP/landlord.ckpt --landlord_up random --landlord_down random
```
以下命令评估Douzero-ADP在对RLCard agent的农民职位
```
python3 evaluate.py --landlord rlcard --landlord_up baselines/douzero_ADP/landlord_up.ckpt --landlord_down baselines/douzero_ADP/landlord_down.ckpt
```
默认情况下，我们的模型将每半小时保存在`douzero_checkpoints/douzero`中。我们提供一个脚本来帮助你识别最近的checkpoint。运行
```
sh get_most_recent.sh douzero_checkpoints/douzero/
```
最近的模型将在“most_recent_model”目录中。

## Core Team
*   Algorithm: [Daochen Zha](https://github.com/daochenzha), [Jingru Xie](https://github.com/karoka), Wenye Ma, Sheng Zhang, [Xiangru Lian](https://xrlian.com/), Xia Hu, [Ji Liu](http://jiliu-ml.org/)
*   GUI Demo: [Songyi Huang](https://github.com/hsywhu)

## Acknowlegements
*   The demo is largely based on [RLCard-Showdown](https://github.com/datamllab/rlcard-showdown)
*   Code implementation is inspired by [TorchBeast](https://github.com/facebookresearch/torchbeast)














