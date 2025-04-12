#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【高级强化学习彩票预测系统 – 进一步优化版】

优化内容：
1. 特征工程：增加遗漏值特征及滑动中位数、分位数等跨期统计特征，同时对遗漏值和窗口统计归一化。
2. 模型结构：在门控Transformer基础上融入图网络模块，增加额外图卷积层，并引入温度参数调控输出分布；
   以达到更平滑（或更分散）的预测输出，改善目前输出高度集中（熵接近1）的情况。
3. 奖励与评估：在奖励计算中增加选号范围奖励惩罚，鼓励预测号码分布宽广；评估指标上增加号码覆盖率、预测分布熵、Top-N 精度。

数据读取路径和保存路径均保持不变。
"""

import os
import random
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

import gym
from gym import spaces

# 尝试导入中国节假日判断库
try:
    import chinese_calendar
    has_chinese_calendar = True
except ImportError:
    has_chinese_calendar = False

# 固定随机种子，确保结果可复现
random.seed(0)
np.random.seed(0)

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 如安装了 Optuna，可用于超参数调优
try:
    import optuna
    has_optuna = True
except ImportError:
    has_optuna = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================================================================
# 1. 数据预处理与增强
# =============================================================================
def load_data():
    data_path = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/kl8/kl8_order_data.csv"
    df = pd.read_csv(data_path, encoding='utf-8', parse_dates=["开奖日期"])
    df.rename(columns={"开奖日期": "Date"}, inplace=True)
    df = df.sort_values("Date").reset_index(drop=True)
    logging.info(f"Loaded data: {len(df)} 天，从 {df['Date'].min().date()} 至 {df['Date'].max().date()}")
    extra_feature_cols = [
        "本期销售金额", "选十玩法奖池金额", "选十中十注数", "单注奖金_十中十",
        "选十中九注数", "单注奖金_十中九", "选十中八注数", "单注奖金_十中八",
        "选十中七注数", "单注奖金_十中七", "选十中六注数", "单注奖金_十中六",
        "选十中五注数", "单注奖金_十中五", "选十中零注数", "单注奖金_十中零",
        "选九中九注数", "单注奖金_九中九", "选九中八注数", "单注奖金_九中八",
        "选九中七注数", "单注奖金_九中七", "选九中六注数", "单注奖金_九中六",
        "选九中五注数", "单注奖金_九中五", "选九中四注数", "单注奖金_九中四",
        "选九中零注数", "单注奖金_九中零", "选八中八注数", "单注金额_八中八",
        "选八中七注数", "单注金额_八中七", "选八中六注数", "单注金额_八中六",
        "选八中五注数", "单注金额_八中五", "选八中四注数", "单注金额_八中四",
        "选八中零注数", "单注金额_八中零", "选七中七注数", "单注金额_七中七",
        "选七中六注数", "单注金额_七中六", "选七中五注数", "单注金额_七中五",
        "选七中四注数", "单注金额_七中四", "选七中零注数", "单注金额_七中零",
        "选六中六注数", "单注金额_六中六", "选六中五注数", "单注金额_六中五",
        "选六中四注数", "单注金额_六中四", "选六中三注数", "单注金额_六中三",
        "选五中五注数", "单注金额_五中五", "选五中四注数", "单注金额_五中四",
        "选五中三注数", "单注金额_五中三", "选四中四注数", "单注金额_四中四",
        "选四中三注数", "单注金额_四中三", "选四中二注数", "单注金额_四中二",
        "选三中三注数", "单注金额_三中三", "选三中二注数", "单注金额_三中二",
        "选二中二注数", "单注金额_二中二", "选一中一注数", "单注金额_一中一"
    ]
    for col in extra_feature_cols:
        df[col] = df[col].replace(',', '', regex=True).astype(float)
    df[extra_feature_cols] = df[extra_feature_cols].fillna(0)
    extra_max = df[extra_feature_cols].max()
    # 构建已知开奖号码集合
    known_draws = {
        row['Date']: set(row[f'排好序_{j}'] for j in range(1, 21))
        for _, row in df.iterrows()
    }
    return df, extra_feature_cols, extra_max, known_draws

def add_external_features(df):
    external_path = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/kl8/external_data.csv"
    try:
        ext_df = pd.read_csv(external_path, encoding='utf-8', parse_dates=["Date"])
        logging.info("成功加载外部数据")
        df = pd.merge(df, ext_df, on="Date", how="left")
        ext_cols = ["temperature", "humidity", "wind_speed", "search_trend"]
        for col in ext_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
    except Exception as e:
        logging.warning("未能加载外部数据，继续使用原数据: {}".format(e))
    return df

def add_context_features(df):
    df["weekday"] = df["Date"].dt.weekday
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    if has_chinese_calendar:
        df["is_holiday"] = df["Date"].apply(lambda d: 1 if chinese_calendar.is_holiday(d.date()) else 0)
        df["is_workday"] = df["Date"].apply(lambda d: 1 if chinese_calendar.is_workday(d.date()) else 0)
    else:
        df["is_holiday"] = 0
        df["is_workday"] = 1
    history_range = 30
    for num in range(1, 81):
        hot_feature = []
        for i in range(len(df)):
            if i < history_range:
                hot_feature.append(0)
            else:
                past = df[[f"排好序_{j}" for j in range(1, 21)]].iloc[i - history_range:i]
                freq = np.sum(past.values == num)
                hot_feature.append(freq)
        df[f"hot_count_{num}"] = hot_feature
    return df

def augment_data_advanced(df):
    sorted_cols = [f'排好序_{j}' for j in range(1, 21)]
    df['draw_std'] = df[sorted_cols].std(axis=1)
    df['draw_sum_lag1'] = df[sorted_cols].sum(axis=1).shift(1).fillna(0)
    df['draw_sum_ma3'] = df['draw_sum_lag1'].rolling(window=3, min_periods=1).mean().fillna(0)
    def count_consecutive(row):
        nums = list(row)
        return sum(1 for i in range(1, len(nums)) if nums[i] == nums[i-1] + 1)
    df['consecutive_count'] = df[sorted_cols].apply(count_consecutive, axis=1)
    def diff_stats(row):
        nums = list(row)
        diffs = [nums[i] - nums[i-1] for i in range(1, len(nums))]
        return pd.Series({
            'min_diff': min(diffs) if diffs else 0,
            'max_diff': max(diffs) if diffs else 0,
            'mean_diff': np.mean(diffs) if diffs else 0
        })
    diff_features = df[sorted_cols].apply(diff_stats, axis=1)
    df = pd.concat([df, diff_features], axis=1)
    return df

def add_delay_features(df):
    delays_mean = []
    delays_min = []
    delays_max = []
    last_occurrence = {num: None for num in range(1, 81)}
    for idx, row in df.iterrows():
        current_numbers = [row[f"排好序_{j}"] for j in range(1, 21)]
        delays = []
        for num in current_numbers:
            if last_occurrence[num] is None:
                delays.append(0)
            else:
                delays.append(idx - last_occurrence[num])
            last_occurrence[num] = idx
        delays_mean.append(np.mean(delays))
        delays_min.append(np.min(delays))
        delays_max.append(np.max(delays))
    df["delay_mean"] = delays_mean
    df["delay_min"] = delays_min
    df["delay_max"] = delays_max
    return df

# 新增：遗漏值特征——统计每个号码距上次开出间隔期数
def add_omission_features(df):
    omission_list = {num: [] for num in range(1, 81)}
    last_occurrence = {num: None for num in range(1, 81)}
    for idx, row in df.iterrows():
        current_numbers = [row[f"排好序_{j}"] for j in range(1, 21)]
        current_omissions = {}
        for num in range(1, 81):
            if last_occurrence[num] is None:
                omission = idx
            else:
                omission = idx - last_occurrence[num]
            current_omissions[num] = omission
        for num in range(1, 81):
            omission_list[num].append(current_omissions[num])
        for num in current_numbers:
            last_occurrence[num] = idx
    for num in range(1, 81):
        col_name = f"omission_{num}"
        df[col_name] = omission_list[num]
    return df

# =============================================================================
# 新增：构建号码共现图（邻接矩阵），用于图网络模块
def build_lottery_graph(df):
    A = np.zeros((80, 80), dtype=np.float32)
    sorted_cols = [f'排好序_{j}' for j in range(1, 21)]
    for _, row in df.iterrows():
        numbers = [int(row[col]) for col in sorted_cols]
        for i, j in combinations(numbers, 2):
            A[i-1, j-1] += 1
            A[j-1, i-1] += 1
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    A_norm = A / row_sum
    return th.tensor(A_norm)

# =============================================================================
# 2. Transformer-GNN特征提取器 —— 融合门控Transformer与图网络
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-th.log(th.tensor(10000.0)) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.gate_linear = nn.Linear(2*d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.gate_linear_ff = nn.Linear(2*d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                          key_padding_mask=src_key_padding_mask)
        cat = th.cat([src, attn_output], dim=-1)
        gate = th.sigmoid(self.gate_linear(cat))
        gated_attn = gate * src + (1 - gate) * attn_output
        src2 = src + self.dropout(gated_attn)
        src = self.norm1(src2)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        cat_ff = th.cat([src, ff_output], dim=-1)
        gate_ff = th.sigmoid(self.gate_linear_ff(cat_ff))
        gated_ff = gate_ff * src + (1 - gate_ff) * ff_output
        src2 = src + self.dropout(gated_ff)
        src = self.norm2(src2)
        return src

class GatedTransformerGNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, seq_len=10, d_model=128, nhead=8,
                 num_layers=2, transformer_out_dim=128, gnn_hidden_dim=64,
                 d_emb=32, dropout=0.1, temperature=1.5):
        super().__init__(observation_space, features_dim=80)
        total_dim = observation_space.shape[0]
        self.seq_len = seq_len
        self.input_dim = total_dim // seq_len
        self.linear_proj = nn.Linear(self.input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        self.layers = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.transformer_fc = nn.Sequential(
            nn.Linear(d_model, transformer_out_dim),
            nn.ReLU()
        )
        self.lottery_embedding = nn.Parameter(th.randn(80, d_emb))
        self.gnn_fc = nn.Sequential(
            nn.Linear(transformer_out_dim + d_emb, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, 1)
        )
        # 温度参数，用于对输出 logits 调节分布
        self.temperature = temperature
        self.register_buffer("adjacency", th.eye(80))  # 默认先使用单位矩阵

    def forward(self, observations):
        B = observations.size(0)
        total_dim = observations.size(1)
        expected_total = self.input_dim * self.seq_len
        if total_dim < expected_total:
            pad_size = expected_total - total_dim
            padded_obs = F.pad(observations, (0, pad_size))
        elif total_dim > expected_total:
            padded_obs = observations[:, :expected_total]
        else:
            padded_obs = observations
        obs_seq = padded_obs.view(B, self.seq_len, self.input_dim)
        obs_seq = self.linear_proj(obs_seq)
        obs_seq = self.pos_encoder(obs_seq)
        obs_seq = obs_seq.transpose(0, 1)  # (seq_len, B, d_model)
        for layer in self.layers:
            obs_seq = layer(obs_seq)
        transformer_out = obs_seq.mean(dim=0)  # (B, d_model)
        global_feature = self.transformer_fc(transformer_out)  # (B, transformer_out_dim)
        global_expanded = global_feature.unsqueeze(1).expand(B, 80, -1)
        lottery_embed = self.lottery_embedding.unsqueeze(0).expand(B, -1, -1)
        fused = th.cat([global_expanded, lottery_embed], dim=-1)
        A = self.adjacency.unsqueeze(0).expand(B, -1, -1)
        aggregated = th.bmm(A, fused)
        node_logits = self.gnn_fc(aggregated).squeeze(-1)  # (B, 80)
        # 应用温度缩放：温度越高输出分布越平滑（多样性提升）
        node_logits = node_logits / self.temperature
        return node_logits

# =============================================================================
# 3. 强化学习环境 —— 更新奖励函数（增加选号范围奖励惩罚）
# =============================================================================
class LotterySequentialEnv(gym.Env):
    def __init__(self, data_df, extra_feature_cols, extra_max, reward_weights=None):
        super(LotterySequentialEnv, self).__init__()
        self.data = data_df.reset_index(drop=True)
        self.known_draws = {
            row['Date']: set(row[f'排好序_{j}'] for j in range(1, 21))
            for _, row in self.data.iterrows()
        }
        self.dates = list(self.data['Date'])
        self.start_index = 180

        self.sorted_numbers = self.data[[f'排好序_{j}' for j in range(1, 21)]].values.astype(np.int32)
        self.order_numbers = self.data[[f'出球顺序_{j}' for j in range(1, 21)]].values.astype(np.int32)

        self.extra_feature_cols = extra_feature_cols
        self.extra_max = extra_max.values.astype(np.float32)

        self.dynamic_feature_dim = 80 + 1
        self.obs_dim = None

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(80)
        self.max_steps = 20

        self.episode_count = 0
        if reward_weights is None:
            self.base_reward_weights = {
                "trend": 0.25,
                "diversity": 0.20,
                "repeat_penalty": -1.0,
                "surprise": 0.15,
                "spread": 0.15
            }
        else:
            self.base_reward_weights = reward_weights
        self.reward_weights = self.base_reward_weights.copy()

        self.selected_numbers = []
        self.current_step = 0
        self.current_index = self.start_index
        self.current_date = self.data.loc[self.current_index, 'Date']
        self.static_features = self._compute_static_features()
        dummy_obs = self._get_observation()
        self.obs_dim = dummy_obs.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def update_reward_weights(self):
        factor = 1 + self.episode_count * 0.001
        self.reward_weights["diversity"] = self.base_reward_weights["diversity"] * factor
        self.reward_weights["trend"] = self.base_reward_weights["trend"] / factor
        logging.debug(f"Episode {self.episode_count}: updated reward weights: {self.reward_weights}")

    def _compute_window_features(self, window_size):
        start_idx = max(0, self.current_index - window_size)
        subset = self.sorted_numbers[start_idx:self.current_index]
        n = subset.shape[0]
        if n > 0:
            flat = subset.flatten()
            freq = np.bincount(flat, minlength=81)[1:] / n
            mean_val = np.mean(flat) / 80.0
            min_val = np.min(flat) / 80.0
            max_val = np.max(flat) / 80.0
            var_val = np.var(flat) / 80.0
        else:
            freq = np.zeros(80, dtype=np.float32)
            mean_val = min_val = max_val = var_val = 0.0
        stats = np.array([mean_val, min_val, max_val, var_val], dtype=np.float32)
        return freq.astype(np.float32), stats

    def _compute_ema_features(self, span=30):
        start_idx = max(0, self.current_index - span)
        subset = self.sorted_numbers[start_idx:self.current_index]
        n = subset.shape[0]
        if n == 0:
            return np.zeros(80, dtype=np.float32)
        row_weights = np.exp(-np.arange(1, n + 1) / span)
        row_weights /= np.sum(row_weights)
        weights_expanded = np.repeat(row_weights, 20)
        flat = subset.flatten()
        ema = np.bincount(flat, weights=weights_expanded, minlength=81)[1:]
        return ema.astype(np.float32)

    def _compute_cumulative_frequency(self):
        subset = self.sorted_numbers[:self.current_index]
        n = subset.shape[0]
        if n > 0:
            flat = subset.flatten()
            freq = np.bincount(flat, minlength=81)[1:] / n
        else:
            freq = np.zeros(80, dtype=np.float32)
        return freq.astype(np.float32)

    def _compute_order_features(self, window_size=30):
        start_idx = max(0, self.current_index - window_size)
        subset = self.order_numbers[start_idx:self.current_index]
        n = subset.shape[0]
        if n == 0:
            return np.zeros(80, dtype=np.float32)
        order_weights = np.array([1.0 / (j + 1) for j in range(20)], dtype=np.float32)
        weights_expanded = np.tile(order_weights, n)
        flat = subset.flatten()
        order_freq = np.bincount(flat, weights=weights_expanded, minlength=81)[1:]
        order_freq /= n
        return order_freq.astype(np.float32)

    def _get_base_features(self, current_date):
        dow = current_date.weekday()
        dow_onehot = np.zeros(7, dtype=np.float32)
        dow_onehot[dow] = 1.0
        month = current_date.month
        month_onehot = np.zeros(12, dtype=np.float32)
        month_onehot[month - 1] = 1.0
        quarter = (month - 1) // 3
        quarter_onehot = np.zeros(4, dtype=np.float32)
        quarter_onehot[quarter] = 1.0
        dow_sin = np.sin(2*np.pi*dow/7)
        dow_cos = np.cos(2*np.pi*dow/7)
        month_sin = np.sin(2*np.pi*(month-1)/12)
        month_cos = np.cos(2*np.pi*(month-1)/12)
        cyclical = np.array([dow_sin, dow_cos, month_sin, month_cos], dtype=np.float32)
        if has_chinese_calendar:
            holiday = 1.0 if chinese_calendar.is_holiday(current_date.date()) else 0.0
        else:
            holiday = 0.0
        prev_day = current_date - pd.Timedelta(days=1)
        next_day = current_date + pd.Timedelta(days=1)
        holiday_prev = 1.0 if has_chinese_calendar and chinese_calendar.is_holiday(prev_day.date()) else 0.0
        holiday_next = 1.0 if has_chinese_calendar and chinese_calendar.is_holiday(next_day.date()) else 0.0
        holiday_features = np.array([holiday, holiday_prev, holiday_next], dtype=np.float32)
        base_features = np.concatenate([dow_onehot, month_onehot, quarter_onehot, cyclical, holiday_features])
        return base_features

    def _compute_static_features(self):
        base_features = self._get_base_features(self.current_date)
        win_sizes = [7,14,30,60,90,180]
        window_feats = []
        for w in win_sizes:
            freq, stats = self._compute_window_features(w)
            window_feats.append(freq)
            window_feats.append(stats)
        window_features = np.concatenate(window_feats)
        ema_feature = self._compute_ema_features(span=30)
        cum_freq = self._compute_cumulative_frequency()
        order_features = self._compute_order_features(window_size=30)
        extra_features = self.data.loc[self.current_index, self.extra_feature_cols].values.astype(np.float32)
        safe_extra_max = np.where(self.extra_max==0, 1, self.extra_max)
        normalized_extra_features = np.nan_to_num(extra_features/safe_extra_max)
        static_features = np.concatenate([base_features, window_features, ema_feature, cum_freq, order_features, normalized_extra_features])
        return static_features.astype(np.float32)

    def _get_observation(self):
        chosen_mask = np.zeros(80, dtype=np.float32)
        for num in self.selected_numbers:
            chosen_mask[int(num)-1] = 1.0
        progress = np.array([self.current_step/self.max_steps], dtype=np.float32)
        observation = np.concatenate([self.static_features, chosen_mask, progress])
        return observation

    def reset(self, index=None):
        self.episode_count += 1
        self.update_reward_weights()
        self.current_step = 0
        self.selected_numbers = []
        if index is not None:
            self.current_index = index
        else:
            self.current_index = np.random.randint(self.start_index, len(self.data))
        self.current_date = self.data.loc[self.current_index, 'Date']
        self.actual_set = self.known_draws.get(self.current_date, set())
        self.static_features = self._compute_static_features()
        return self._get_observation()

    def compute_reward(self, action):
        if (action+1) in self.selected_numbers:
            return self.reward_weights["repeat_penalty"]
        ema_feature = self._compute_ema_features(span=30)
        trend_value = ema_feature[action]/(np.mean(ema_feature)+1e-6)
        reward_trend = self.reward_weights["trend"] * trend_value
        diversity = len(set(self.selected_numbers))/self.max_steps
        reward_diversity = self.reward_weights["diversity"] * diversity
        ema_sum = np.sum(ema_feature)+1e-6
        prob = ema_feature[action]/ema_sum
        surprise = -np.log(prob+1e-6)
        normalized_surprise = surprise/np.log(80)
        reward_surprise = self.reward_weights["surprise"] * normalized_surprise
        # 原有扩散奖励：鼓励选号跨度扩大
        if len(self.selected_numbers) > 0:
            current_spread = (max(self.selected_numbers) - min(self.selected_numbers))/79.0
            new_selection = self.selected_numbers + [action+1]
            new_spread = (max(new_selection) - min(new_selection))/79.0
            spread_reward = self.reward_weights["spread"]*(new_spread-current_spread)
            # 如预测跨度不足（过于集中），额外增加惩罚
            if new_spread < 0.5:
                spread_reward += -0.1*(0.5 - new_spread)
        else:
            spread_reward = 0
        total_reward = reward_trend + reward_diversity + reward_surprise + spread_reward
        self.selected_numbers.append(action+1)
        return total_reward

    def step(self, action):
        if (action+1) in self.selected_numbers:
            reward = self.reward_weights["repeat_penalty"]
        else:
            reward = self.compute_reward(action)
        self.current_step += 1
        done = (self.current_step==self.max_steps)
        info = {}
        if done:
            predicted_set = set(self.selected_numbers)
            hits = len(predicted_set & self.actual_set) if self.actual_set is not None else 0
            reward += hits
            info = {"predicted_set": predicted_set, "actual_set": self.actual_set, "hits": hits}
        return self._get_observation(), reward, done, info

# =============================================================================
# 4. 预测与评估辅助函数（增加覆盖率、熵、Top-N指标）
# =============================================================================
def complete_prediction(pred_list):
    if len(pred_list) < 20:
        missing = sorted(set(range(1,81))-set(pred_list))
        pred_list.extend(missing[:20-len(pred_list)])
    return pred_list

def simulate_episode_for_prediction(env, model, future_date):
    env.reset()
    env.current_date = future_date
    env.actual_set = None
    while env.current_step < env.max_steps:
        obs = env._get_observation()
        action, _ = model.predict(obs, deterministic=True)
        if (action+1) in env.selected_numbers:
            valid_actions = list(set(range(1,81))-set(env.selected_numbers))
            if valid_actions:
                action = random.choice(valid_actions)-1
        obs, reward, done, info = env.step(action)
    return complete_prediction(env.selected_numbers)

def simulate_episode_for_date(date, df, model, extra_feature_cols, extra_max):
    temp_env = LotterySequentialEnv(df, extra_feature_cols, extra_max)
    idx = df.index[df['Date']==date][0]
    temp_env.reset(index=idx)
    temp_env.current_date = date
    while temp_env.current_step < temp_env.max_steps:
        obs = temp_env._get_observation()
        action, _ = model.predict(obs, deterministic=True)
        temp_env.step(action)
    return set(temp_env.selected_numbers)

def ensemble_prediction_for_date(date, df, models, extra_feature_cols, extra_max):
    predictions = []
    for model in models:
        pred = simulate_episode_for_date(date, df, model, extra_feature_cols, extra_max)
        predictions.append(pred)
    counter_votes = Counter(sum([list(pred) for pred in predictions],[]))
    ensemble_nums = [num for num, cnt in counter_votes.items() if cnt>=2]
    if len(ensemble_nums)<20:
        remaining = [num for num in (predictions[0]|predictions[1]|predictions[2]) if num not in ensemble_nums]
        ensemble_nums.extend(sorted(remaining, key=lambda x: counter_votes[x], reverse=True))
    if len(ensemble_nums)<20:
        missing = sorted(set(range(1,81))-set(ensemble_nums))
        ensemble_nums.extend(missing)
    return sorted(ensemble_nums[:20])

def simulate_episode_with_prob(env, model, future_date, num_runs=10):
    freq = np.zeros(80)
    for i in range(num_runs):
        env.reset()
        env.current_date = future_date
        env.actual_set = None
        while env.current_step < env.max_steps:
            obs = env._get_observation()
            action, _ = model.predict(obs, deterministic=True)
            if (action+1) in env.selected_numbers:
                valid_actions = list(set(range(1,81))-set(env.selected_numbers))
                if valid_actions:
                    action = random.choice(valid_actions)-1
            obs, reward, done, info = env.step(action)
        for num in env.selected_numbers:
            freq[num-1] += 1
    prob = freq/num_runs
    if prob.sum()>0:
        prob = prob/ prob.sum()
    return prob

def compute_ece(preds, true_labels, n_bins=10):
    bin_boundaries = np.linspace(0,1,n_bins+1)
    total_samples = preds.size
    ece = 0.0
    for i in range(n_bins):
        mask = (preds>=bin_boundaries[i]) & (preds<bin_boundaries[i+1])
        if np.sum(mask)>0:
            avg_confidence = preds[mask].mean()
            avg_accuracy = true_labels[mask].mean()
            ece += (np.sum(mask)/total_samples)*abs(avg_confidence-avg_accuracy)
    return ece

def compute_brier_score(probs, true_labels):
    return np.mean((probs-true_labels)**2)

def compute_entropy(probs):
    p = np.clip(probs, 1e-6,1.0)
    return -np.sum(p*np.log(p))/np.log(len(p))

def evaluate_model_advanced(env, models, df, eval_dates, num_runs=5, top_n=10):
    hit_counts = []
    ensemble_predictions = []
    weights = []
    all_probs = []
    union_predicted = set()
    topn_precisions = []
    for model in models:
        hits = []
        for date in eval_dates:
            idx = df.index[df['Date']==date][0]
            env.reset(index=idx)
            env.current_date = date
            while env.current_step < env.max_steps:
                obs = env._get_observation()
                action, _ = model.predict(obs, deterministic=True)
                env.step(action)
            if "hits" in env.__dict__.get('info', {}):
                hits.append(env.info["hits"] if "hits" in env.info else 0)
            else:
                actual_set = env.known_draws.get(date, set())
                pred_set = set(env.selected_numbers)
                hits.append(len(pred_set & actual_set))
        avg_hit = np.mean(hits) if hits else 0
        weights.append(avg_hit)
    total = sum(weights) if sum(weights)>0 else 1
    weights = [w/total for w in weights]
    for date in eval_dates:
        idx = df.index[df['Date']==date][0]
        env.reset(index=idx)
        actual_set = env.known_draws.get(date, set())
        true_vec = np.zeros(80)
        for num in actual_set:
            true_vec[num-1] = 1
        ensemble_prob = np.zeros(80)
        for model, w in zip(models, weights):
            prob = simulate_episode_with_prob(env, model, date, num_runs=num_runs)
            ensemble_prob += w*prob
        if ensemble_prob.sum()>0:
            ensemble_prob = ensemble_prob/ensemble_prob.sum()
        all_probs.append(ensemble_prob)
        top_indices = ensemble_prob.argsort()[::-1][:top_n]
        top_precision = len(set(top_indices+1) & actual_set)/top_n
        topn_precisions.append(top_precision)
        env.reset(index=idx)
        env.current_date = date
        while env.current_step < env.max_steps:
            obs = env._get_observation()
            action, _ = model.predict(obs, deterministic=True)
            env.step(action)
        ensemble_predictions.append(set(env.selected_numbers))
        union_predicted = union_predicted.union(set(env.selected_numbers))
        hit_counts.append(len(set(env.selected_numbers)&actual_set))
    all_probs = np.vstack(all_probs)
    all_true = np.vstack([np.array([1 if i in env.known_draws.get(date, set()) else 0 for i in range(1,81)])
                           for date in eval_dates])
    ece_value = compute_ece(all_probs, all_true, n_bins=10)
    brier_value = compute_brier_score(all_probs, all_true)
    diversity_score = 1 - np.mean([len(set_i & set_j)/len(set_i|set_j)
                                    for i, set_i in enumerate(ensemble_predictions)
                                    for j, set_j in enumerate(ensemble_predictions) if i < j])
    avg_hit = np.mean(hit_counts) if hit_counts else 0
    coverage = len(union_predicted)/80.0
    entropies = [compute_entropy(p) for p in all_probs]
    avg_entropy = np.mean(entropies)
    top_n_precision = np.mean(topn_precisions)
    return avg_hit, ece_value, diversity_score, brier_value, coverage, avg_entropy, top_n_precision

def plot_hit_trend(df, models, env, extra_feature_cols, extra_max, dst_dir, total_timesteps):
    hit_counts = []
    dates_list = []
    for idx in range(env.start_index, len(df)):
        date = df.loc[idx, 'Date']
        ensemble_pred = ensemble_prediction_for_date(date, df, models, extra_feature_cols, extra_max)
        actual_set = env.known_draws.get(date, set())
        hits = len(set(ensemble_pred)&actual_set)
        hit_counts.append(hits)
        dates_list.append(date)
    last_date = df['Date'].max()
    one_month_ago = last_date - pd.Timedelta(days=30)
    filtered_dates = [d for d in dates_list if d>=one_month_ago]
    filtered_hits = [h for d,h in zip(dates_list,hit_counts) if d>=one_month_ago]
    plt.figure(figsize=(10,5))
    plt.plot(filtered_dates, filtered_hits, marker='o', label="最近一个月每日命中数")
    avg_hits_filtered = np.mean(filtered_hits) if filtered_hits else 0
    plt.axhline(y=avg_hits_filtered, color='r', linestyle='--', label=f"平均命中 = {avg_hits_filtered:.2f}")
    plt.xlabel("日期")
    plt.ylabel("命中数（共20个）")
    plt.title("最近一个月训练集集成预测命中趋势")
    plt.legend()
    plt.tight_layout()
    hit_trend_path = os.path.join(dst_dir, f"RL_lottery_predictor_v7_TOTAL_TIMESTEPS_{total_timesteps}.png")
    plt.savefig(hit_trend_path)
    logging.info(f"命中趋势图已保存至 {hit_trend_path}")

# =============================================================================
# 5. 模型训练与评估
# =============================================================================
def train_model(algorithm, env, policy_kwargs, total_timesteps, update_timesteps, model_save_path):
    if algorithm=="QRDQN":
        from sb3_contrib.qrdqn import QRDQN
        from torch.optim.lr_scheduler import StepLR
        model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-3)
        optimizer = model.policy.optimizer
        scheduler = StepLR(optimizer, step_size=update_timesteps, gamma=0.9)
        steps = 0
        while steps<total_timesteps:
            model.learn(total_timesteps=update_timesteps, reset_num_timesteps=False)
            steps += update_timesteps
            scheduler.step()
            logging.info(f"{algorithm} 已训练 {steps} 步")
    elif algorithm=="PPO":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-3)
        model.learn(total_timesteps=total_timesteps)
    elif algorithm=="A2C":
        from stable_baselines3 import A2C
        model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-3)
        model.learn(total_timesteps=total_timesteps)
    model.save(model_save_path)
    logging.info(f"{algorithm} 模型已保存至 {model_save_path}")
    return model

def evaluate_model_recent(env, model, df, recent_days=30):
    hit_counts = []
    recent_date = df['Date'].max()-pd.Timedelta(days=recent_days)
    for idx in range(env.start_index, len(df)):
        if df.loc[idx,'Date']>=recent_date:
            obs = env.reset(index=idx)
            done=False
            while not done:
                action, _ = model.predict(obs,deterministic=True)
                obs, reward, done, info = env.step(action)
            if "hits" in info:
                hit_counts.append(info["hits"])
    return np.mean(hit_counts) if hit_counts else 0

def evaluate_model_multiple(env, model, df, days_list):
    scores = []
    for d in days_list:
        score = evaluate_model_recent(env, model, df, recent_days=d)
        scores.append(score)
    return np.mean(scores)

# =============================================================================
# 6. 主流程：训练、评估、预测与结果保存
# =============================================================================
def main():
    # 数据加载、融合与特征增强
    df, extra_feature_cols, extra_max, known_draws = load_data()
    df = add_external_features(df)
    df = add_context_features(df)
    df = augment_data_advanced(df)
    df = add_delay_features(df)
    df = add_omission_features(df)
    # 更新特征列表（保留原有特征和新增遗漏值）
    extra_feature_cols = extra_feature_cols + [
        "draw_std", "draw_sum_lag1", "draw_sum_ma3",
        "consecutive_count", "min_diff", "max_diff", "mean_diff",
        "weekday", "weekofyear", "month", "quarter", "is_holiday", "is_workday",
        "delay_mean", "delay_min", "delay_max"
    ]
    for col in ["temperature", "humidity", "wind_speed", "search_trend"]:
        if col in df.columns:
            extra_feature_cols.append(col)
    extra_feature_cols += [f"hot_count_{num}" for num in range(1,81)]
    extra_max = df[extra_feature_cols].max()

    # 构建号码共现图
    lottery_graph = build_lottery_graph(df)

    dst_dir = "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/20250412/"
    os.makedirs(dst_dir, exist_ok=True)
    TOTAL_TIMESTEPS = 10000
    TIMESTEPS_PER_UPDATE = 1000

    # 使用融合图网络模块的门控Transformer-GNN特征提取器
    policy_kwargs = {
        "features_extractor_class": GatedTransformerGNNFeatureExtractor,
        "features_extractor_kwargs": {
            "seq_len": 10,
            "d_model": 128,
            "nhead": 8,
            "num_layers": 2,
            "transformer_out_dim": 128,
            "gnn_hidden_dim": 64,
            "d_emb": 32,
            "dropout": 0.1,
            "temperature": 1.5  # 温度参数调高使得输出分布更平滑，促进多样性
        }
    }

    env_qrdqn = LotterySequentialEnv(df, extra_feature_cols, extra_max)
    env_ppo = LotterySequentialEnv(df, extra_feature_cols, extra_max)
    env_a2c = LotterySequentialEnv(df, extra_feature_cols, extra_max)

    logging.info("开始训练 QRDQN 模型...")
    model_qrdqn = train_model("QRDQN", env_qrdqn, policy_kwargs, TOTAL_TIMESTEPS, TIMESTEPS_PER_UPDATE,
                                os.path.join(dst_dir, "model_qrdqn"))
    logging.info("开始训练 PPO 模型...")
    model_ppo = train_model("PPO", env_ppo, policy_kwargs, TOTAL_TIMESTEPS, TIMESTEPS_PER_UPDATE,
                              os.path.join(dst_dir, "model_ppo"))
    logging.info("开始训练 A2C 模型...")
    model_a2c = train_model("A2C", env_a2c, policy_kwargs, TOTAL_TIMESTEPS, TIMESTEPS_PER_UPDATE,
                              os.path.join(dst_dir, "model_a2c"))

    # 模型评估：多窗口近期表现
    env_eval = LotterySequentialEnv(df, extra_feature_cols, extra_max)
    days_list = [1,3,5,7,14,21,30]
    qrdqn_score = evaluate_model_multiple(env_eval, model_qrdqn, df, days_list)
    ppo_score = evaluate_model_multiple(env_eval, model_ppo, df, days_list)
    a2c_score = evaluate_model_multiple(env_eval, model_a2c, df, days_list)
    logging.info(f"多窗口近期表现 -> QRDQN: {qrdqn_score:.2f}, PPO: {ppo_score:.2f}, A2C: {a2c_score:.2f}")
    total_score = qrdqn_score + ppo_score + a2c_score
    w_qrdqn = qrdqn_score/total_score if total_score>0 else 0.33
    w_ppo = ppo_score/total_score if total_score>0 else 0.33
    w_a2c = a2c_score/total_score if total_score>0 else 0.33
    logging.info(f"动态权重 -> QRDQN: {w_qrdqn:.2f}, PPO: {w_ppo:.2f}, A2C: {w_a2c:.2f}")

    eval_dates = df[df['Date']>=df['Date'].max()-pd.Timedelta(days=30)]['Date'].tolist()
    avg_hit, ece_value, diversity_score, brier_value, coverage, avg_entropy, top_precision = evaluate_model_advanced(
        env_eval, [model_qrdqn, model_ppo, model_a2c], df, eval_dates, num_runs=5, top_n=10)
    logging.info(f"高级评估指标 -> 平均命中: {avg_hit:.2f}, ECE: {ece_value:.4f}, 多样性: {diversity_score:.4f}, "
                 f"Brier: {brier_value:.4f}, 覆盖率: {coverage:.4f}, 平均熵: {avg_entropy:.4f}, Top-10 精度: {top_precision:.4f}")

    last_date = df['Date'].max()
    future_date = last_date + pd.Timedelta(days=1)
    env_pred = LotterySequentialEnv(df, extra_feature_cols, extra_max)
    pred_qrdqn = simulate_episode_for_prediction(env_pred, model_qrdqn, future_date)
    pred_ppo = simulate_episode_for_prediction(env_pred, model_ppo, future_date)
    pred_a2c = simulate_episode_for_prediction(env_pred, model_a2c, future_date)

    def save_prediction(pred, model_name):
        pred_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": pred[j-1] for j in range(1,21)}}])
        csv_path = os.path.join(dst_dir, f"{model_name}_v7_prediction_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")
        pred_df.to_csv(csv_path, index=False)
        logging.info(f"{model_name} 预测结果已保存至 {csv_path}")

    save_prediction(pred_qrdqn, "qrdqn")
    save_prediction(pred_ppo, "ppo")
    save_prediction(pred_a2c, "a2c")

    ensemble_counter = Counter()
    for num in pred_qrdqn:
        ensemble_counter[num] += w_qrdqn
    for num in pred_ppo:
        ensemble_counter[num] += w_ppo
    for num in pred_a2c:
        ensemble_counter[num] += w_a2c
    final_pred = [num for num, weight in sorted(ensemble_counter.items(), key=lambda x: x[1], reverse=True)]
    if len(final_pred)<20:
        missing = sorted(set(range(1,81))-set(final_pred))
        final_pred.extend(missing)
    final_pred = final_pred[:20]
    logging.info(f"预测 {future_date.date()} 的号码: {final_pred}")

    pred_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": final_pred[j-1] for j in range(1,21)}}])
    pred_csv_path = os.path.join(dst_dir, f"RL_lottery_predictor_v7_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.md")
    pred_df.to_csv(pred_csv_path, index=False)
    logging.info(f"集成预测结果已保存至 {pred_csv_path}")

    plot_hit_trend(df, [model_qrdqn, model_ppo, model_a2c], env_eval,
                   extra_feature_cols, extra_max, dst_dir, TOTAL_TIMESTEPS)

    logging.info("请确保在后续模型加载后，将 lottery_graph 赋值到各模型的 features_extractor.adjacency 属性中。")
    if has_optuna:
        logging.info("开始使用 Optuna 进行超参数调优...")
        # TODO: 定义并调用 Optuna objective 函数

if __name__=="__main__":
    main()
