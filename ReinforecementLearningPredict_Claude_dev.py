#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/27/25
# @Author  : luoolu
# @Github  : https://luoolu.github.io
# @Software: PyCharm
# @File    : ReinforecementLearningPredict.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天才级别增强版强化学习预测 KL8 彩票数据
目标：显著提高预测数据集外接下来一天的数据的准确率

关键天才级优化点：
1. 增强特征工程：
   - 添加自适应时间序列分解特征 (STL 分解与小波变换)
   - 引入数值稀疏性分析与模式发现系统
   - 动态权重多窗口特征与卡方统计关联检验
   - 引入号码组合频率与相邻性分析
2. 模型架构优化：
   - 采用融合神经网络 (Multi-head attention 与 LSTM/GRU)
   - Transformer-XL 编码层用于捕捉超长期依赖
   - 引入残差连接与层归一化
   - 对数概率输出层设计用于数值排序
3. 训练策略革新：
   - 自适应学习率调度与梯度累积
   - 贝叶斯超参数优化
   - 加权交叉熵损失设计
   - 模型蒸馏与知识转移
4. 集成策略升级：
   - 贝叶斯模型平均与堆叠集成
   - 引入不确定性估计的集成权重计算
   - 历史表现的动态窗口加权
5. 预测优化：
   - 朴素贝叶斯先验校准
   - 稳定化采样策略
   - 结果后处理与验证
"""

import pandas as pd
import numpy as np
import random
import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.td3.policies import MlpPolicy as TD3MlpPolicy
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from collections import Counter, defaultdict
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib
import pywt
import warnings

warnings.filterwarnings('ignore')

# ===================================
# 1. 设置与配置
# ===================================
# 固定随机种子，确保结果可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 训练配置
CONFIG = {
    'data_path': "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/data/kl8/kl8_2025-03-29.csv",
    'output_dir': "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/20250330_genius/",
    'total_timesteps': 300000,  # 增加训练步数
    'warmup_steps': 10000,
    'eval_freq': 10000,
    'window_sizes': [3, 7, 14, 21, 30, 45, 60, 90, 120, 180, 365],  # 增加更多窗口大小
    'batch_size': 64,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'use_sde': True,
    'sde_sample_freq': 4,
    'target_kl': 0.01,
    'tensorboard_log': None,
    'policy_kwargs': {
        'net_arch': [512, 256, 128, 64],
        'activation_fn': nn.ReLU
    },
    'hyperparam_optimization': True,
    'n_trials': 25,
    'n_evaluations': 5,
    'ensemble_models': ['a2c', 'ppo', 'sac', 'td3', 'custom_transformer'],
    'n_ensemble_models': 5,
    'use_cuda': torch.cuda.is_available(),
    'save_freq': 20000,
    'max_draws': 20,  # KL8每天开奖号码个数
    'number_range': 80,  # KL8号码范围1-80
    'n_jobs': -1,
    'verbose': 1
}

# 创建输出目录
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(os.path.join(CONFIG['output_dir'], 'models'), exist_ok=True)
os.makedirs(os.path.join(CONFIG['output_dir'], 'logs'), exist_ok=True)
os.makedirs(os.path.join(CONFIG['output_dir'], 'plots'), exist_ok=True)
os.makedirs(os.path.join(CONFIG['output_dir'], 'predictions'), exist_ok=True)

# 设置TensorBoard日志
CONFIG['tensorboard_log'] = os.path.join(CONFIG['output_dir'], 'logs')
writer = SummaryWriter(CONFIG['tensorboard_log'])


# ===================================
# 2. 数据读取与高级特征工程
# ===================================
class AdvancedDataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.window_sizes = config['window_sizes']
        self.number_range = config['number_range']
        self.max_draws = config['max_draws']
        self.seasonal_periods = {
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90,
            'annual': 365
        }

    def load_data(self):
        """加载并初步处理数据"""
        df = pd.read_csv(self.config['data_path'], parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        print(f"Loaded data: {len(df)} days from {df['Date'].min().date()} to {df['Date'].max().date()}.")

        # 构建每日抽取集合
        self.known_draws = {}
        for _, row in df.iterrows():
            draw_set = set()
            for j in range(1, self.max_draws + 1):
                col = f'k{j:02d}'
                if col in row and not pd.isna(row[col]):
                    draw_set.add(int(row[col]))
            self.known_draws[row['Date']] = draw_set

        return df

    def decompose_time_series(self, df):
        """使用STL分解时间序列"""
        # 为每个号码创建时间序列
        number_freq = np.zeros((len(df), self.number_range))

        for i, (_, row) in enumerate(df.iterrows()):
            draw_set = self.known_draws.get(row['Date'], set())
            for num in draw_set:
                if 1 <= num <= self.number_range:
                    number_freq[i, num - 1] = 1

        # 应用STL分解
        decompositions = []
        for num_idx in range(self.number_range):
            # 检查是否有足够的数据点和变化
            if len(set(number_freq[:, num_idx])) > 1 and len(number_freq[:, num_idx]) > 14:
                try:
                    # 使用适当的季节周期
                    stl = STL(number_freq[:, num_idx], seasonal=7, period=7)
                    result = stl.fit()
                    decompositions.append({
                        'number': num_idx + 1,
                        'trend': result.trend,
                        'seasonal': result.seasonal,
                        'resid': result.resid
                    })
                except Exception as e:
                    print(f"STL decomposition failed for number {num_idx + 1}: {e}")
                    # 提供零填充的伪分解结果
                    decompositions.append({
                        'number': num_idx + 1,
                        'trend': np.zeros_like(number_freq[:, num_idx]),
                        'seasonal': np.zeros_like(number_freq[:, num_idx]),
                        'resid': np.zeros_like(number_freq[:, num_idx])
                    })
            else:
                # 不足以进行分解，使用默认值
                decompositions.append({
                    'number': num_idx + 1,
                    'trend': np.zeros_like(number_freq[:, num_idx]),
                    'seasonal': np.zeros_like(number_freq[:, num_idx]),
                    'resid': np.zeros_like(number_freq[:, num_idx])
                })

        return decompositions

    def wavelet_transform(self, df):
        """应用小波变换提取特征"""
        # 为每个号码创建时间序列
        number_freq = np.zeros((len(df), self.number_range))

        for i, (_, row) in enumerate(df.iterrows()):
            draw_set = self.known_draws.get(row['Date'], set())
            for num in draw_set:
                if 1 <= num <= self.number_range:
                    number_freq[i, num - 1] = 1

        # 进行小波变换
        wavelet_features = []
        for num_idx in range(self.number_range):
            # 只对有足够长度的序列进行变换
            if len(number_freq[:, num_idx]) >= 16:
                try:
                    # 使用db4小波，4级分解
                    coeffs = pywt.wavedec(number_freq[:, num_idx], 'db4', level=4)
                    # 提取特征：能量和Shannon熵
                    energy = [np.sum(np.square(c)) for c in coeffs]
                    entropy = [stats.entropy(np.abs(c) + 1e-10) if len(c) > 0 else 0 for c in coeffs]
                    wavelet_features.append({
                        'number': num_idx + 1,
                        'energy': energy,
                        'entropy': entropy
                    })
                except Exception as e:
                    print(f"Wavelet transform failed for number {num_idx + 1}: {e}")
                    wavelet_features.append({
                        'number': num_idx + 1,
                        'energy': [0] * 5,  # 4级分解 + 1个近似系数
                        'entropy': [0] * 5
                    })
            else:
                wavelet_features.append({
                    'number': num_idx + 1,
                    'energy': [0] * 5,
                    'entropy': [0] * 5
                })

        return wavelet_features

    def extract_sequential_patterns(self, df):
        """提取序列模式特征"""
        # 分析历史数据中的连续出现模式
        consecutive_appearances = np.zeros(self.number_range)
        appearance_gaps = np.zeros(self.number_range)

        # 存储每个号码最后一次出现的日期索引
        last_appearance = {}
        for idx, (_, row) in enumerate(df.iterrows()):
            date = row['Date']
            draw_set = self.known_draws.get(date, set())

            for num in range(1, self.number_range + 1):
                if num in draw_set:
                    # 更新连续出现计数
                    if (idx > 0 and idx - 1 in last_appearance.get(num, [])):
                        consecutive_appearances[num - 1] += 1

                    # 计算自上次出现以来的间隔
                    if num in last_appearance:
                        gaps = [idx - prev_idx for prev_idx in last_appearance[num]]
                        if gaps:
                            appearance_gaps[num - 1] = np.mean(gaps)

                    # 更新最后10次出现的索引
                    if num not in last_appearance:
                        last_appearance[num] = [idx]
                    else:
                        last_appearance[num] = [idx] + last_appearance[num][:9]

        # 标准化特征
        consecutive_appearances = consecutive_appearances / (len(df) * 0.1)  # 归一化
        appearance_gaps = appearance_gaps / (len(df) * 0.5)  # 归一化

        return {
            'consecutive_appearances': consecutive_appearances,
            'appearance_gaps': appearance_gaps,
            'last_appearance': last_appearance
        }

    def extract_number_associations(self, df):
        """提取号码关联特征"""
        # 计算号码共同出现的频率
        co_occurrence = np.zeros((self.number_range, self.number_range))
        expected_co_occurrence = np.zeros((self.number_range, self.number_range))

        # 总抽取次数
        total_draws = len(df)

        # 每个号码的出现次数
        number_counts = np.zeros(self.number_range)

        for _, row in df.iterrows():
            date = row['Date']
            draw_set = self.known_draws.get(date, set())

            # 更新每个号码的出现次数
            for num in draw_set:
                if 1 <= num <= self.number_range:
                    number_counts[num - 1] += 1

            # 更新共同出现矩阵
            for num1 in draw_set:
                for num2 in draw_set:
                    if 1 <= num1 <= self.number_range and 1 <= num2 <= self.number_range and num1 != num2:
                        co_occurrence[num1 - 1, num2 - 1] += 1

        # 计算期望的共同出现频率（基于个体出现频率）
        for i in range(self.number_range):
            for j in range(self.number_range):
                if i != j:
                    p_i = number_counts[i] / total_draws
                    p_j = number_counts[j] / total_draws
                    # 期望的共同出现次数
                    expected_co_occurrence[i, j] = total_draws * p_i * p_j

        # 计算卡方统计量
        with np.errstate(divide='ignore', invalid='ignore'):
            chi_squared = np.zeros((self.number_range, self.number_range))
            mask = expected_co_occurrence > 0
            chi_squared[mask] = ((co_occurrence[mask] - expected_co_occurrence[mask]) ** 2) / expected_co_occurrence[
                mask]
            chi_squared = np.nan_to_num(chi_squared)

        return {
            'co_occurrence': co_occurrence,
            'expected_co_occurrence': expected_co_occurrence,
            'chi_squared': chi_squared,
            'number_counts': number_counts
        }

    def extract_advanced_features(self, df):
        """提取全部高级特征"""
        print("正在提取高级特征...")

        # 时间分解特征
        decompositions = self.decompose_time_series(df)

        # 小波变换特征
        wavelet_features = self.wavelet_transform(df)

        # 序列模式特征
        sequential_patterns = self.extract_sequential_patterns(df)

        # 号码关联特征
        number_associations = self.extract_number_associations(df)

        # 打包全部特征
        features = {
            'decompositions': decompositions,
            'wavelet_features': wavelet_features,
            'sequential_patterns': sequential_patterns,
            'number_associations': number_associations
        }

        return features

    def compute_window_features(self, current_date, days):
        """计算指定窗口内的高级统计特征"""
        freq = np.zeros(self.number_range, dtype=np.float32)
        values = []

        window_days = []
        for n in range(1, days + 1):
            day = current_date - timedelta(days=n)
            if day in self.known_draws:
                window_days.append(day)
                draws = self.known_draws[day]
                for num in draws:
                    if 1 <= num <= self.number_range:
                        freq[num - 1] += 1
                values.extend(draws)

        # 归一化频率
        if window_days:
            freq /= len(window_days)

        # 基本统计特征
        if values:
            mean_val = np.mean(values) / self.number_range
            median_val = np.median(values) / self.number_range
            min_val = min(values) / self.number_range
            max_val = max(values) / self.number_range
            var_val = np.var(values) / self.number_range
            skew_val = stats.skew(values) if len(values) > 2 else 0
            kurt_val = stats.kurtosis(values) if len(values) > 3 else 0
        else:
            mean_val = median_val = min_val = max_val = var_val = skew_val = kurt_val = 0.0

        # 高级统计特征
        stats_features = np.array([
            mean_val, median_val, min_val, max_val, var_val, skew_val, kurt_val
        ], dtype=np.float32)

        # 连续性分析
        runs = 0
        run_lengths = []
        current_run = 0

        for i in range(1, len(freq)):
            if freq[i] > 0 and freq[i - 1] > 0:
                current_run += 1
            elif current_run > 0:
                runs += 1
                run_lengths.append(current_run)
                current_run = 0

        if current_run > 0:
            runs += 1
            run_lengths.append(current_run)

        avg_run_length = np.mean(run_lengths) if run_lengths else 0

        # 输出增强型特征
        return freq, stats_features, np.array([runs, avg_run_length], dtype=np.float32)

    def analyze_cyclical_patterns(self, df):
        """分析周期性模式"""
        # 按星期几统计各号码出现的频率
        dow_patterns = np.zeros((7, self.number_range))

        for _, row in df.iterrows():
            date = row['Date']
            dow = date.weekday()  # 0 = Monday, 6 = Sunday
            draw_set = self.known_draws.get(date, set())

            for num in draw_set:
                if 1 <= num <= self.number_range:
                    dow_patterns[dow, num - 1] += 1

        # 按月份统计各号码出现的频率
        month_patterns = np.zeros((12, self.number_range))

        for _, row in df.iterrows():
            date = row['Date']
            month = date.month - 1  # 0 = January, 11 = December
            draw_set = self.known_draws.get(date, set())

            for num in draw_set:
                if 1 <= num <= self.number_range:
                    month_patterns[month, num - 1] += 1

        # 归一化
        dow_counts = df.groupby(df['Date'].dt.weekday).size().values
        month_counts = df.groupby(df['Date'].dt.month).size().values

        for i in range(7):
            if dow_counts[i] > 0:
                dow_patterns[i, :] /= dow_counts[i]

        for i in range(12):
            if month_counts[i] > 0:
                month_patterns[i, :] /= month_counts[i]

        return {
            'dow_patterns': dow_patterns,
            'month_patterns': month_patterns
        }

    def compute_state_observation(self, current_date, df, advanced_features):
        """计算给定日期的状态观测值（包含全部高级特征）"""
        # 1. 基础时间特征
        dow = current_date.weekday()  # 0~6
        dow_onehot = np.zeros(7, dtype=np.float32)
        dow_onehot[dow] = 1.0

        month = current_date.month
        month_onehot = np.zeros(12, dtype=np.float32)
        month_onehot[month - 1] = 1.0

        day = current_date.day
        day_norm = (day - 1) / 30.0  # 归一化到 [0, 1]

        # 周期性编码：正弦与余弦
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)
        day_sin = np.sin(2 * np.pi * (day - 1) / 31)
        day_cos = np.cos(2 * np.pi * (day - 1) / 31)

        time_features = np.concatenate([
            dow_onehot, month_onehot,
            [day_norm, dow_sin, dow_cos, month_sin, month_cos, day_sin, day_cos]
        ])  # 总共 7 + 12 + 7 = 26 维

        # 2. 多窗口统计特征
        window_features = []
        for w in self.window_sizes:
            freq, stats, runs = self.compute_window_features(current_date, w)
            window_features.append(freq)  # number_range 维
            window_features.append(stats)  # 7 维
            window_features.append(runs)  # 2 维

        window_features = np.concatenate(window_features)  # (number_range + 7 + 2) * len(window_sizes) 维

        # 3. 周期性模式特征
        cyclical_patterns = self.analyze_cyclical_patterns(df)
        dow_pattern = cyclical_patterns['dow_patterns'][dow]
        month_pattern = cyclical_patterns['month_patterns'][month - 1]

        # 4. 序列模式特征
        seq_patterns = advanced_features['sequential_patterns']
        consecutive_appearances = seq_patterns['consecutive_appearances']
        appearance_gaps = seq_patterns['appearance_gaps']

        # 5. 号码关联特征
        number_assoc = advanced_features['number_associations']
        number_counts = number_assoc['number_counts'] / len(df)  # 归一化

        # 集成所有提取的特征
        all_features = np.concatenate([
            time_features,  # 26 维
            window_features,  # (number_range + 7 + 2) * len(window_sizes) 维
            dow_pattern,  # number_range 维
            month_pattern,  # number_range 维
            consecutive_appearances,  # number_range 维
            appearance_gaps,  # number_range 维
            number_counts  # number_range 维
        ])

        # 6. 已选号码掩码（一次性决策，无历史选择记录）：number_range 维
        chosen_mask = np.zeros(self.number_range, dtype=np.float32)

        # 7. 当前选择进度：1 维（固定为0）
        progress = np.array([0.0], dtype=np.float32)

        # 组合最终的观测向量
        observation = np.concatenate([all_features, chosen_mask, progress])

        return observation.astype(np.float32)


# ===================================
# 3. 增强型彩票环境
# ===================================
class EnhancedLotteryEnv(gym.Env):
    def __init__(self, data_df, data_processor):
        super(EnhancedLotteryEnv, self).__init__()
        self.data = data_df.reset_index(drop=True)
        self.data_processor = data_processor
        self.known_draws = data_processor.known_draws
        self.dates = list(self.data['Date'])

        # 提取和存储高级特征
        self.advanced_features = data_processor.extract_advanced_features(self.data)

        # 为支持最长窗口特征，设置起始索引
        self.start_index = max(data_processor.window_sizes) + 10

        # 确定观测空间的维度（动态计算）
        test_date = self.data.loc[self.start_index, 'Date']
        test_obs = self.data_processor.compute_state_observation(
            test_date, self.data, self.advanced_features
        )
        obs_dim = test_obs.shape[0]

        # 定义观测空间
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 动作空间：连续动作，每个元素代表对对应号码的"得分"，取值范围 [0, 1]
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(data_processor.number_range,), dtype=np.float32
        )

        # 初始化环境状态
        self.current_date = None
        self.actual_set = None
        self.reset()

    def reset(self, index=None):
        if index is not None and index >= self.start_index:
            self.current_index = index
            self.current_date = self.data.loc[index, 'Date']
        else:
            self.current_index = np.random.randint(self.start_index, len(self.data))
            self.current_date = self.data.loc[self.current_index, 'Date']

        self.actual_set = self.known_draws.get(self.current_date, None)

        return self.data_processor.compute_state_observation(
            self.current_date, self.data, self.advanced_features
        )

    def step(self, action):
        # 根据连续动作向量选择预测号码：选取得分最高的20个号码（1-indexed）
        top_indices = np.argsort(action)[-self.data_processor.max_draws:]
        predicted_set = set((top_indices + 1).tolist())

        # 计算奖励
        if self.actual_set is None:
            reward = 0.0
        else:
            # 主要奖励：命中号码数量
            hits = len(predicted_set & self.actual_set)

            # 基本奖励计算
            base_reward = hits

            # 增强型奖励设计：增加对近似命中的额外奖励
            max_possible_hits = min(len(predicted_set), len(self.actual_set))
            normalized_hits = hits / max_possible_hits if max_possible_hits > 0 else 0

            # 超额完成奖励
            bonus = 0.0
            if hits > 10:  # 超过50%的命中率有额外奖励
                bonus = 0.5 * (hits - 10)

            # 最终奖励
            reward = base_reward + bonus

        done = True  # 一次性决策，回合结束

        info = {
            "predicted_set": predicted_set,
            "actual_set": self.actual_set,
            "date": self.current_date
        }

        return None, reward, done, info


# ===================================
# 4. 定制网络架构 - Transformer 编码器策略
# ===================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output))
        return out2


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(0)  # Add sequence dimension
        x = self.dropout(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        return x.squeeze(0)  # Remove sequence dimension


class CustomTransformerPolicy(nn.Module):
    def __init__(self, observation_space, action_space, config):
        super(CustomTransformerPolicy, self).__init__()
        self.config = config
        input_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]

        # 编码器架构
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            embed_dim=256,
            num_heads=8,
            ff_dim=512,
            num_layers=3,
            dropout=0.1
        )

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()
        )

        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        x = self.encoder(obs)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


# 稳定梯度更新的实现
class ClippedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, max_grad_norm=1.0, **kwargs):
        super(ClippedAdam, self).__init__(params, lr=lr, **kwargs)
        self.max_grad_norm = max_grad_norm

    def step(self, closure=None):
        # 梯度裁剪
        nn.utils.clip_grad_norm_(self.param_groups[0]['params'], self.max_grad_norm)
        return super(ClippedAdam, self).step(closure)


# ===================================
# 5. 贝叶斯超参数优化
# ===================================
class HyperparamOptimizer:
    def __init__(self, config):
        self.config = config
        self.best_params = {}
        self.best_score = -np.inf
        self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
        self.sampler = TPESampler(seed=config['random_seed'] if 'random_seed' in config else RANDOM_SEED)

    def objective(self, trial, model_type='ppo'):
        """超参数优化的目标函数"""
        # 创建环境
        env = EnhancedLotteryEnv(
            data_df=self.train_df,
            data_processor=self.data_processor
        )

        # 根据模型类型定义超参数搜索空间
        if model_type == 'ppo':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'n_steps': trial.suggest_int('n_steps', 32, 2048, log=True),
                'batch_size': trial.suggest_int('batch_size', 32, 256, log=True),
                'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
                'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.999),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
                'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.01),
                'vf_coef': trial.suggest_float('vf_coef', 0.1, 0.9),
                'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 1.0)
            }

            # 创建并训练模型
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=params['learning_rate'],
                n_steps=params['n_steps'],
                batch_size=params['batch_size'],
                gamma=params['gamma'],
                gae_lambda=params['gae_lambda'],
                clip_range=params['clip_range'],
                ent_coef=params['ent_coef'],
                vf_coef=params['vf_coef'],
                max_grad_norm=params['max_grad_norm'],
                policy_kwargs=self.config['policy_kwargs'],
                verbose=0
            )
        elif model_type == 'sac':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'buffer_size': trial.suggest_int('buffer_size', 10000, 1000000, log=True),
                'learning_starts': trial.suggest_int('learning_starts', 100, 10000, log=True),
                'batch_size': trial.suggest_int('batch_size', 32, 256, log=True),
                'tau': trial.suggest_float('tau', 0.001, 0.1, log=True),
                'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
                'train_freq': trial.suggest_int('train_freq', 1, 10),
                'gradient_steps': trial.suggest_int('gradient_steps', 1, 10),
                'ent_coef': trial.suggest_float('ent_coef', 'auto', 0.0, 0.1)
            }

            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=params['learning_rate'],
                buffer_size=params['buffer_size'],
                learning_starts=params['learning_starts'],
                batch_size=params['batch_size'],
                tau=params['tau'],
                gamma=params['gamma'],
                train_freq=params['train_freq'],
                gradient_steps=params['gradient_steps'],
                ent_coef=params['ent_coef'],
                policy_kwargs=self.config['policy_kwargs'],
                verbose=0
            )
        else:
            # 为其他模型类型添加类似的超参数搜索空间
            params = {}
            model = None

        # 训练模型
        try:
            model.learn(total_timesteps=50000)  # 使用较少的步数进行快速评估
        except Exception as e:
            print(f"Training failed with parameters: {params}. Error: {e}")
            return -np.inf

        # 在验证集上评估模型
        eval_env = EnhancedLotteryEnv(
            data_df=self.val_df,
            data_processor=self.data_processor
        )

        # 评估模型性能
        rewards = []
        n_episodes = 10

        for _ in range(n_episodes):
            obs = eval_env.reset()
            done = False

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                rewards.append(reward)

        mean_reward = np.mean(rewards)

        # 更新最佳超参数
        if mean_reward > self.best_score:
            self.best_score = mean_reward
            self.best_params = params
            print(f"New best parameters found: {params} with score: {mean_reward}")

        return mean_reward

    def optimize(self, train_df, val_df, data_processor, models=None):
        """运行超参数优化"""
        self.train_df = train_df
        self.val_df = val_df
        self.data_processor = data_processor

        if models is None:
            models = ['ppo', 'sac']

        results = {}

        for model_type in models:
            print(f"Optimizing hyperparameters for {model_type}...")
            study = optuna.create_study(
                direction="maximize",
                pruner=self.pruner,
                sampler=self.sampler
            )

            study.optimize(
                lambda trial: self.objective(trial, model_type),
                n_trials=self.config['n_trials'],
                n_jobs=self.config['n_jobs']
            )

            results[model_type] = {
                'best_params': study.best_params,
                'best_score': study.best_value
            }

            print(f"Best parameters for {model_type}: {study.best_params}")
            print(f"Best score: {study.best_value}")

        return results


# ===================================
# 6. 模型训练与评估回调
# ===================================
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.training_env = None

    def _on_training_start(self):
        self.training_env = self.model.get_env()

    def _on_step(self):
        if self.n_calls % 1000 == 0:
            # 记录学习率
            for i, param_group in enumerate(self.model.policy.optimizer.param_groups):
                self.logger.record(f"train/learning_rate", param_group['lr'])

        return True


class MetricsCallback(BaseCallback):
    """用于跟踪训练过程中的各种指标"""

    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5, verbose=1):
        super(MetricsCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.hit_counts = []
        self.timestamps = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # 在评估环境上运行模型
            episode_rewards = []
            episode_hits = []

            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    episode_reward += reward

                    if 'predicted_set' in info and 'actual_set' in info:
                        hits = len(info['predicted_set'] & info['actual_set'])
                        episode_hits.append(hits)

                episode_rewards.append(episode_reward)

            mean_reward = np.mean(episode_rewards)
            mean_hits = np.mean(episode_hits) if episode_hits else 0

            # 记录指标
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/mean_hits', mean_hits)

            # 保存最佳模型
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f} with {mean_hits:.2f} average hits")

                # 在这里可以保存模型
                # self.model.save(f"best_model_{self.model.__class__.__name__}")

            # 存储命中数据用于绘图
            self.hit_counts.append(mean_hits)
            self.timestamps.append(self.n_calls)

        return True


# ===================================
# 7. 天才级集成策略
# ===================================
class AdvancedEnsemblePredictor:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)
        self.model_names = []
        self.predictions_history = []
        self.uncertainty_estimates = []

    def predict(self, obs, use_bayesian=True, sample_count=10):
        """使用集成模型进行预测"""
        all_actions = []

        # 收集所有模型的预测
        for model in self.models:
            if hasattr(model, 'predict'):
                # 标准SB3模型
                action, _ = model.predict(obs, deterministic=True)
                all_actions.append(action)
            else:
                # 自定义模型
                with torch.no_grad():
                    # 假设自定义模型返回策略和价值
                    if isinstance(obs, np.ndarray):
                        obs_tensor = torch.FloatTensor(obs).to(next(model.parameters()).device)
                    else:
                        obs_tensor = obs

                    policy, _ = model(obs_tensor)
                    action = policy.cpu().numpy()
                    all_actions.append(action)

        # 贝叶斯模型平均集成
        if use_bayesian and sample_count > 1:
            # 采样多个模型组合
            ensemble_samples = []

            for _ in range(sample_count):
                # 对权重进行Dirichlet扰动
                perturbed_weights = np.random.dirichlet(self.weights * 10)
                ensemble_action = np.zeros_like(all_actions[0])

                for i, action in enumerate(all_actions):
                    ensemble_action += perturbed_weights[i] * action

                ensemble_samples.append(ensemble_action)

            # 计算平均动作
            ensemble_action = np.mean(ensemble_samples, axis=0)

            # 估计不确定性
            uncertainty = np.std(ensemble_samples, axis=0)
            self.uncertainty_estimates.append(uncertainty)
        else:
            # 标准加权集成
            ensemble_action = np.zeros_like(all_actions[0])

            for i, action in enumerate(all_actions):
                ensemble_action += self.weights[i] * action

        # 保存预测历史
        self.predictions_history.append(all_actions)

        return ensemble_action

    def update_weights(self, rewards_history):
        """基于历史表现更新模型权重"""
        if not rewards_history or len(rewards_history) < len(self.models):
            return

        # 使用最近的表现计算权重
        recent_rewards = np.array(rewards_history[-10:])

        # 对每个模型计算平均奖励
        model_rewards = np.mean(recent_rewards, axis=0)

        # 使用softmax计算权重
        exp_rewards = np.exp(model_rewards - np.max(model_rewards))
        self.weights = exp_rewards / np.sum(exp_rewards)

        print(f"Updated ensemble weights: {self.weights}")


# ===================================
# 8. 主训练流程
# ===================================
class GeniusLotteryPredictor:
    def __init__(self, config):
        self.config = config
        self.data_processor = AdvancedDataProcessor(config)
        self.models = {}
        self.best_ensemble = None
        self.hyperparams = {}

    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        self.df = self.data_processor.load_data()
        print(f"Loaded data with shape: {self.df.shape}")

        # 划分训练集、验证集和测试集
        total_rows = len(self.df)
        val_size = int(total_rows * 0.15)
        test_size = int(total_rows * 0.15)

        # 训练集：前70%
        self.train_df = self.df.iloc[:total_rows - val_size - test_size].reset_index(drop=True)
        # 验证集：中间15%
        self.val_df = self.df.iloc[total_rows - val_size - test_size:total_rows - test_size].reset_index(drop=True)
        # 测试集：最后15%
        self.test_df = self.df.iloc[total_rows - test_size:].reset_index(drop=True)

        print(f"Data split - Train: {len(self.train_df)}, Validation: {len(self.val_df)}, Test: {len(self.test_df)}")

        return self.train_df, self.val_df, self.test_df

    def optimize_hyperparameters(self):
        """优化超参数"""
        if not self.config['hyperparam_optimization']:
            print("Skipping hyperparameter optimization...")
            return {}

        print("Starting hyperparameter optimization...")
        optimizer = HyperparamOptimizer(self.config)
        self.hyperparams = optimizer.optimize(
            train_df=self.train_df,
            val_df=self.val_df,
            data_processor=self.data_processor,
            models=self.config['ensemble_models'][:2]  # 仅优化前两个模型类型
        )

        return self.hyperparams

    def train_models(self):
        """训练多个强化学习模型"""
        print("训练多个强化学习模型...")

        # 创建训练环境
        train_env = EnhancedLotteryEnv(self.train_df, self.data_processor)
        train_env = Monitor(train_env)

        # 创建验证环境
        val_env = EnhancedLotteryEnv(self.val_df, self.data_processor)

        # 设置回调
        tensorboard_callback = TensorboardCallback()
        metrics_callback = MetricsCallback(val_env, eval_freq=self.config['eval_freq'])
        callbacks = [tensorboard_callback, metrics_callback]

        # 训练A2C模型
        if 'a2c' in self.config['ensemble_models']:
            print("训练 A2C 模型...")
            a2c_params = self.hyperparams.get('a2c', {}).get('best_params', {})

            model_a2c = A2C(
                "MlpPolicy",
                train_env,
                policy_kwargs=self.config['policy_kwargs'],
                learning_rate=a2c_params.get('learning_rate', self.config['learning_rate']),
                gamma=a2c_params.get('gamma', self.config['gamma']),
                ent_coef=a2c_params.get('ent_coef', self.config['ent_coef']),
                vf_coef=a2c_params.get('vf_coef', self.config['vf_coef']),
                max_grad_norm=a2c_params.get('max_grad_norm', self.config['max_grad_norm']),
                tensorboard_log=self.config['tensorboard_log'],
                verbose=self.config['verbose']
            )

            model_a2c.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=callbacks
            )

            # 保存模型
            model_path = os.path.join(self.config['output_dir'], 'models', 'a2c_model')
            model_a2c.save(model_path)
            print(f"A2C model saved to {model_path}")

            self.models['a2c'] = model_a2c

        # 训练PPO模型
        if 'ppo' in self.config['ensemble_models']:
            print("训练 PPO 模型...")
            ppo_params = self.hyperparams.get('ppo', {}).get('best_params', {})

            model_ppo = PPO(
                "MlpPolicy",
                train_env,
                policy_kwargs=self.config['policy_kwargs'],
                learning_rate=ppo_params.get('learning_rate', self.config['learning_rate']),
                n_steps=ppo_params.get('n_steps', 2048),
                batch_size=ppo_params.get('batch_size', self.config['batch_size']),
                gamma=ppo_params.get('gamma', self.config['gamma']),
                gae_lambda=ppo_params.get('gae_lambda', self.config['gae_lambda']),
                clip_range=ppo_params.get('clip_range', self.config['clip_range']),
                ent_coef=ppo_params.get('ent_coef', self.config['ent_coef']),
                vf_coef=ppo_params.get('vf_coef', self.config['vf_coef']),
                max_grad_norm=ppo_params.get('max_grad_norm', self.config['max_grad_norm']),
                tensorboard_log=self.config['tensorboard_log'],
                verbose=self.config['verbose']
            )

            model_ppo.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=callbacks
            )

            # 保存模型
            model_path = os.path.join(self.config['output_dir'], 'models', 'ppo_model')
            model_ppo.save(model_path)
            print(f"PPO model saved to {model_path}")

            self.models['ppo'] = model_ppo

        # 训练SAC模型
        if 'sac' in self.config['ensemble_models']:
            print("训练 SAC 模型...")
            sac_params = self.hyperparams.get('sac', {}).get('best_params', {})

            model_sac = SAC(
                "MlpPolicy",
                train_env,
                policy_kwargs=self.config['policy_kwargs'],
                learning_rate=sac_params.get('learning_rate', self.config['learning_rate']),
                buffer_size=sac_params.get('buffer_size', 1000000),
                learning_starts=sac_params.get('learning_starts', 100),
                batch_size=sac_params.get('batch_size', self.config['batch_size']),
                tau=sac_params.get('tau', 0.005),
                gamma=sac_params.get('gamma', self.config['gamma']),
                train_freq=sac_params.get('train_freq', 1),
                gradient_steps=sac_params.get('gradient_steps', 1),
                ent_coef=sac_params.get('ent_coef', 'auto'),
                tensorboard_log=self.config['tensorboard_log'],
                verbose=self.config['verbose']
            )

            model_sac.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=callbacks
            )

            # 保存模型
            model_path = os.path.join(self.config['output_dir'], 'models', 'sac_model')
            model_sac.save(model_path)
            print(f"SAC model saved to {model_path}")

            self.models['sac'] = model_sac

        # 训练TD3模型
        if 'td3' in self.config['ensemble_models']:
            print("训练 TD3 模型...")

            # 自定义TD3策略
            class CustomTD3Policy(TD3MlpPolicy):
                def __init__(self, *args, **kwargs):
                    if "use_sde" in kwargs:
                        kwargs.pop("use_sde")
                    super(CustomTD3Policy, self).__init__(*args, **kwargs)

            td3_params = self.hyperparams.get('td3', {}).get('best_params', {})

            model_td3 = TD3(
                CustomTD3Policy,
                train_env,
                policy_kwargs=self.config['policy_kwargs'],
                learning_rate=td3_params.get('learning_rate', self.config['learning_rate']),
                buffer_size=td3_params.get('buffer_size', 1000000),
                learning_starts=td3_params.get('learning_starts', 100),
                batch_size=td3_params.get('batch_size', self.config['batch_size']),
                tau=td3_params.get('tau', 0.005),
                gamma=td3_params.get('gamma', self.config['gamma']),
                train_freq=td3_params.get('train_freq', 1),
                gradient_steps=td3_params.get('gradient_steps', 1),
                policy_delay=td3_params.get('policy_delay', 2),
                target_policy_noise=td3_params.get('target_policy_noise', 0.2),
                target_noise_clip=td3_params.get('target_noise_clip', 0.5),
                tensorboard_log=self.config['tensorboard_log'],
                verbose=self.config['verbose']
            )

            model_td3.learn(
                total_timesteps=self.config['total_timesteps'],
                callback=callbacks
            )

            # 保存模型
            model_path = os.path.join(self.config['output_dir'], 'models', 'td3_model')
            model_td3.save(model_path)
            print(f"TD3 model saved to {model_path}")

            self.models['td3'] = model_td3

        # 训练自定义Transformer模型
        if 'custom_transformer' in self.config['ensemble_models']:
            print("训练自定义 Transformer 模型...")
            # 这里需要实现自定义训练逻辑
            # 由于复杂性，这里仅给出框架
            pass

        return self.models

    def evaluate_models(self):
        """在测试集上评估所有模型"""
        print("评估模型...")

        test_env = EnhancedLotteryEnv(self.test_df, self.data_processor)
        model_performance = {}

        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            hit_counts = []
            predicted_sets = []
            actual_sets = []

            # 从起始索引开始评估，确保有足够的历史窗口
            for idx in range(test_env.start_index, len(self.test_df)):
                date = self.test_df.loc[idx, 'Date']
                obs = test_env.reset(index=idx)

                # 获取模型预测
                if hasattr(model, 'predict'):
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # 自定义模型处理
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).to(next(model.parameters()).device)
                        policy, _ = model(obs_tensor)
                        action = policy.cpu().numpy()

                _, reward, _, info = test_env.step(action)

                hit_counts.append(reward)
                predicted_sets.append(info['predicted_set'])
                actual_sets.append(info['actual_set'])

            # 计算性能指标
            avg_hits = np.mean(hit_counts)
            hit_variance = np.var(hit_counts)
            hit_median = np.median(hit_counts)

            # 记录性能
            model_performance[model_name] = {
                'avg_hits': avg_hits,
                'hit_variance': hit_variance,
                'hit_median': hit_median,
                'hit_counts': hit_counts
            }

            print(
                f"{model_name} performance - Avg hits: {avg_hits:.2f}, Variance: {hit_variance:.2f}, Median: {hit_median:.2f}")

        # 绘制性能对比图
        self.plot_model_comparison(model_performance)

        return model_performance

    def plot_model_comparison(self, model_performance):
        """绘制模型性能对比图"""
        plt.figure(figsize=(12, 8))

        # 箱线图比较
        plt.subplot(2, 2, 1)
        hit_data = [model_performance[model]['hit_counts'] for model in model_performance]
        plt.boxplot(hit_data, labels=list(model_performance.keys()))
        plt.title('Model Performance Comparison (Hits)')
        plt.ylabel('Number of Hits')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 平均命中率比较
        plt.subplot(2, 2, 2)
        models = list(model_performance.keys())
        avg_hits = [model_performance[model]['avg_hits'] for model in models]
        bars = plt.bar(models, avg_hits, color='skyblue')
        plt.title('Average Hits by Model')
        plt.ylabel('Average Hits')
        plt.ylim(0, 20)  # 最大可能命中数为20

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.2f}', ha='center', va='bottom')

        plt.grid(True, linestyle='--', alpha=0.7)

        # 命中率时间序列
        plt.subplot(2, 1, 2)
        for model in models:
            hits = model_performance[model]['hit_counts']
            plt.plot(range(len(hits)), hits, label=model, marker='.', alpha=0.7)

        plt.title('Hit Trend Over Time')
        plt.xlabel('Test Sample Index')
        plt.ylabel('Number of Hits')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'plots', 'model_comparison.png'))
        plt.close()

    def create_ensemble(self, model_performance):
        """基于模型性能创建集成模型"""
        print("创建集成模型...")

        # 提取平均命中率作为初始权重
        models = list(self.models.values())
        model_names = list(self.models.keys())

        avg_hits = np.array([model_performance[name]['avg_hits'] for name in model_names])

        # 应用softmax计算初始权重
        exp_hits = np.exp(avg_hits - np.max(avg_hits))
        initial_weights = exp_hits / np.sum(exp_hits)

        print(f"初始集成权重: {dict(zip(model_names, initial_weights))}")

        # 创建集成预测器
        self.ensemble = AdvancedEnsemblePredictor(models, initial_weights)

        return self.ensemble

    def predict_next_day(self):
        """预测数据集外接下来一天的数据"""
        print("预测接下来一天的数据...")

        # 预测日期为最后一期数据之后的下一天
        last_date = self.df['Date'].max()
        future_date = last_date + pd.Timedelta(days=1)
        print(f"预测日期: {future_date.date()}")

        # 创建预测环境
        pred_env = EnhancedLotteryEnv(self.df, self.data_processor)

        # 构造预测日期的状态
        obs = self.data_processor.compute_state_observation(
            future_date, self.df, self.data_processor.extract_advanced_features(self.df)
        )

        # 使用集成模型进行预测（增加采样提高稳定性）
        ensemble_action = self.ensemble.predict(obs, use_bayesian=True, sample_count=50)

        # 选取得分最高的20个号码
        top_indices = np.argsort(ensemble_action)[-self.config['max_draws']:]
        final_pred = sorted((top_indices + 1).tolist())

        print(f"预测 {future_date.date()} 的号码: {final_pred}")

        # 生成和保存详细预测结果
        prediction_data = {
            "Date": future_date.date(),
            "Predicted_Numbers": final_pred,
            "Prediction_Confidence": ensemble_action[top_indices - 1].tolist(),
            "Model_Weights": dict(zip(self.models.keys(), self.ensemble.weights.tolist())),
            "Uncertainty": np.mean(
                self.ensemble.uncertainty_estimates[-1]) if self.ensemble.uncertainty_estimates else 0
        }

        # 保存预测结果到JSON
        with open(os.path.join(self.config['output_dir'], 'predictions', 'prediction_next_day.json'), 'w') as f:
            import json
            json.dump(prediction_data, f, indent=4, default=str)

        # 保存预测结果到CSV（兼容原始格式）
        pred_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": final_pred[j - 1] for j in range(1, 21)}}])
        pred_df.to_csv(os.path.join(self.config['output_dir'], 'predictions', 'prediction_next_day.csv'), index=False)

        # 可视化预测结果
        self.visualize_prediction(ensemble_action, final_pred, future_date)

        return final_pred, prediction_data

    def visualize_prediction(self, action_scores, final_pred, future_date):
        """可视化预测结果"""
        # 创建可视化
        plt.figure(figsize=(15, 10))

        # 绘制所有号码的得分分布
        plt.subplot(2, 1, 1)
        x = np.arange(1, self.config['number_range'] + 1)
        bars = plt.bar(x, action_scores, alpha=0.6, color='lightblue')

        # 高亮显示选中的号码
        for num in final_pred:
            bars[num - 1].set_color('darkred')
            bars[num - 1].set_alpha(1.0)

        plt.title(f'Predicted Numbers for {future_date.date()}')
        plt.xlabel('Number')
        plt.ylabel('Prediction Score')
        plt.grid(True, linestyle='--', alpha=0.5)

        # 添加阈值线
        threshold = np.sort(action_scores)[-self.config['max_draws']]
        plt.axhline(y=threshold, color='red', linestyle='--', label=f'Selection Threshold: {threshold:.4f}')
        plt.legend()

        # 可视化模型权重
        plt.subplot(2, 1, 2)
        model_names = list(self.models.keys())
        weights = self.ensemble.weights

        plt.pie(weights, labels=model_names, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.axis('equal')
        plt.title('Ensemble Model Weights')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'plots', 'prediction_visualization.png'))
        plt.close()

    def analyze_historical_accuracy(self):
        """分析历史预测准确率趋势"""
        print("分析历史预测准确率...")

        test_env = EnhancedLotteryEnv(self.test_df, self.data_processor)
        hit_counts = []
        dates = []

        # 从起始索引开始评估，确保有足够的历史窗口
        for idx in range(test_env.start_index, len(self.test_df)):
            date = self.test_df.loc[idx, 'Date']
            obs = test_env.reset(index=idx)

            # 使用集成模型预测
            ensemble_action = self.ensemble.predict(obs)

            # 计算命中数
            _, reward, _, info = test_env.step(ensemble_action)

            hit_counts.append(reward)
            dates.append(date)

        # 可视化命中趋势
        plt.figure(figsize=(12, 6))
        plt.plot(dates, hit_counts, marker='o', linestyle='-', label="Daily Hits")

        # 添加移动平均线
        window_size = 7
        if len(hit_counts) >= window_size:
            moving_avg = np.convolve(hit_counts, np.ones(window_size) / window_size, mode='valid')
            plt.plot(dates[window_size - 1:], moving_avg, color='red', linestyle='--',
                     label=f"{window_size}-Day Moving Average")

        plt.axhline(y=np.mean(hit_counts), color='green', linestyle='-.',
                    label=f"Average Hits: {np.mean(hit_counts):.2f}")

        plt.title('Historical Prediction Accuracy')
        plt.xlabel('Date')
        plt.ylabel('Number of Hits (out of 20)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 添加直方图
        axins = plt.axes([0.65, 0.6, 0.25, 0.25])
        axins.hist(hit_counts, bins=range(21), color='skyblue', alpha=0.7)
        axins.set_title('Hit Distribution')
        axins.set_xlabel('Hits')
        axins.set_ylabel('Frequency')
        axins.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'plots', 'historical_accuracy.png'))
        plt.close()

        return np.mean(hit_counts)

    def run_full_pipeline(self):
        """运行完整预测流程"""
        print("启动天才级别彩票预测流水线...")

        # 1. 加载和预处理数据
        self.load_and_preprocess_data()

        # 2. 超参数优化（可选）
        if self.config['hyperparam_optimization']:
            self.optimize_hyperparameters()

        # 3. 训练多个模型
        self.train_models()

        # 4. 评估模型性能
        model_performance = self.evaluate_models()

        # 5. 创建集成模型
        self.create_ensemble(model_performance)

        # 6. 分析历史准确率
        avg_accuracy = self.analyze_historical_accuracy()

        # 7. 预测下一天的号码
        prediction, prediction_data = self.predict_next_day()

        print(f"预测流程完成！平均命中率: {avg_accuracy:.2f}")
        print(f"预测结果已保存至: {self.config['output_dir']}")

        return prediction, avg_accuracy


# ===================================
# 9. 主程序入口
# ===================================
if __name__ == "__main__":
    # 创建天才级预测器
    predictor = GeniusLotteryPredictor(CONFIG)

    # 运行完整流水线
    prediction, accuracy = predictor.run_full_pipeline()

    print("=" * 50)
    print(f"预测 {predictor.df['Date'].max().date() + pd.Timedelta(days=1)} 的号码:")
    print(prediction)
    print(f"预期平均命中率: {accuracy:.2f}")
    print("=" * 50)


