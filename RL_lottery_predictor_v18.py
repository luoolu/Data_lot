#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【高级强化学习彩票预测系统 – 综合优化版 v18】

修订内容：
1. 引入 Cosine Beta 调度策略用于 Diffusion 模型（可选线性或余弦调度）。
2. 调整命中奖励权重（bonus_multiplier 从 4.0 提升至 5.0）。
3. 在奖励函数、集成策略等模块中做细节调整，以期提升预测命中数量。
4. 增加最近一个月每天命中数量的图的绘制与保存。
"""

import os
import random
import logging
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import gym
from gym import spaces

try:
    import chinese_calendar
    has_chinese_calendar = True
except ImportError:
    has_chinese_calendar = False

random.seed(0)
np.random.seed(0)

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    import optuna
    has_optuna = True
except ImportError:
    has_optuna = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =============================================================================
# 辅助函数：Beta 调度策略实现（线性和余弦）
# =============================================================================
def cosine_beta_schedule(T, s=0.008):
    """
    基于余弦调度策略生成 beta 序列。
    根据 https://arxiv.org/abs/2102.09672 的思想，构造余弦调度下的 alpha_cum 值，
    再由 alpha_cum 计算 beta_t = 1 - alpha_bar[t+1] / alpha_bar[t]。
    """
    steps = T + 1
    x = np.linspace(0, T, steps)
    # 使用余弦函数计算累计产品
    alpha_bar = np.cos((x / T + s) / (1 + s) * (np.pi / 2)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = []
    for t in range(T):
        beta = 1 - alpha_bar[t + 1] / alpha_bar[t]
        betas.append(min(beta, 0.999))
    return np.array(betas, dtype=np.float32)


# =============================================================================
# 1. 数据预处理与特征工程
# =============================================================================
def load_data():
    data_path = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/kl8/kl8_order_data.csv"
    df = pd.read_csv(data_path, encoding='utf-8', parse_dates=["开奖日期"])
    df.rename(columns={"开奖日期": "Date"}, inplace=True)
    df = df.sort_values("Date").reset_index(drop=True)
    logging.info(f"Loaded data: {len(df)} 天，从 {df['Date'].min().date()} 至 {df['Date'].max().date()}")
    preset_extra_feature_cols = [
        "本期销售金额", "选十玩法奖池金额", "选十中十注数", "单注奖金_十中十",
        "选十中九注数", "单注奖金_十中九", "选十中八注数", "单注奖金_十中八",
        "选十中七注数", "单注奖金_十中七", "选十中六注数", "单注奖金_十中六",
        "选十中五注数", "单注奖金_十中五", "选十中零注数", "单注奖金_十中零",
        "选九中九注数", "单注奖金_九中九", "选九中八注数", "单注奖金_九中八",
        "选九中七注数", "单注奖金_九中七", "选九中六注数", "单注奖金_九中六",
        "选九中五注数", "单注奖金_九中五", "选九中四注数", "单注奖金_九中四"
    ]
    valid_extra_feature_cols = [col for col in preset_extra_feature_cols if col in df.columns]
    # 去除逗号并转换为数值型
    for col in valid_extra_feature_cols:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[valid_extra_feature_cols] = df[valid_extra_feature_cols].fillna(0)
    extra_mean = df[valid_extra_feature_cols].mean()
    extra_std = df[valid_extra_feature_cols].std().replace(0, 1)
    extra_max = df[valid_extra_feature_cols].max()
    known_draws = {
        row["Date"]: set(int(row[f"排好序_{j}"]) for j in range(1, 21))
        for _, row in df.iterrows()
    }
    return df, valid_extra_feature_cols, extra_max, extra_mean, extra_std, known_draws


def add_external_features(df):
    external_path = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/kl8/external_data.csv"
    try:
        ext_df = pd.read_csv(external_path, encoding="utf-8", parse_dates=["Date"])
        logging.info("成功加载外部数据")
        df = pd.merge(df, ext_df, on="Date", how="left")
        ext_cols = ["temperature", "humidity", "wind_speed", "search_trend"]
        for col in ext_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std() if df[col].std() != 0 else 1
                df[col] = (df[col].fillna(mean_val) - mean_val) / std_val
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
        hot_counts = []
        for i in range(len(df)):
            if i < history_range:
                hot_counts.append(0)
            else:
                past = df[[f"排好序_{j}" for j in range(1, 21)]].iloc[i - history_range:i]
                freq = np.sum(past.values == num)
                hot_counts.append(int(freq))
        df[f"hot_count_{num}"] = hot_counts
    return df


def augment_data_advanced(df):
    sorted_cols = [f"排好序_{j}" for j in range(1, 21)]
    df["draw_std"] = df[sorted_cols].std(axis=1)
    df["draw_sum_lag1"] = df[sorted_cols].sum(axis=1).shift(1).fillna(0)
    df["draw_sum_ma3"] = df["draw_sum_lag1"].rolling(window=3, min_periods=1).mean().fillna(0)

    def count_consecutive(row):
        nums = list(row)
        return sum(1 for i in range(1, len(nums)) if nums[i] == nums[i - 1] + 1)

    df["consecutive_count"] = df[sorted_cols].apply(count_consecutive, axis=1)

    def diff_stats(row):
        nums = list(row)
        diffs = [nums[i] - nums[i - 1] for i in range(1, len(nums))]
        return pd.Series({
            "min_diff": min(diffs) if diffs else 0,
            "max_diff": max(diffs) if diffs else 0,
            "mean_diff": np.mean(diffs) if diffs else 0
        })

    diff_features = df[sorted_cols].apply(diff_stats, axis=1)
    df = pd.concat([df, diff_features], axis=1)
    df["rolling_mean"] = df[sorted_cols].sum(axis=1).rolling(window=7, min_periods=1).mean()
    df["rolling_std"] = df[sorted_cols].sum(axis=1).rolling(window=7, min_periods=1).std().fillna(0)
    return df


def add_delay_features(df):
    delays_mean = []
    delays_min = []
    delays_max = []
    last_occurrence = {num: None for num in range(1, 81)}
    sorted_cols = [f"排好序_{j}" for j in range(1, 21)]
    for idx, row in df.iterrows():
        current_numbers = [row[col] for col in sorted_cols]
        current_delays = []
        for num in current_numbers:
            if last_occurrence[num] is None:
                current_delays.append(0)
            else:
                current_delays.append(idx - last_occurrence[num])
            last_occurrence[num] = idx
        delays_mean.append(np.mean(current_delays))
        delays_min.append(np.min(current_delays))
        delays_max.append(np.max(current_delays))
    df["delay_mean"] = delays_mean
    df["delay_min"] = delays_min
    df["delay_max"] = delays_max
    return df


def add_omission_features(df):
    omission_list = {num: [] for num in range(1, 81)}
    last_occurrence = {num: None for num in range(1, 81)}
    sorted_cols = [f"排好序_{j}" for j in range(1, 21)]
    for idx, row in df.iterrows():
        current_numbers = [row[col] for col in sorted_cols]
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
        df[f"omission_{num}"] = omission_list[num]
    return df


def build_lottery_graph(df):
    A = np.zeros((80, 80), dtype=np.float32)
    sorted_cols = [f"排好序_{j}" for j in range(1, 21)]
    for _, row in df.iterrows():
        numbers = [int(row[col]) for col in sorted_cols]
        for a, b in combinations(numbers, 2):
            A[a - 1, b - 1] += 1
            A[b - 1, a - 1] += 1
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    A_norm = A / row_sum
    return th.tensor(A_norm)  # 归一化后的邻接矩阵


# =============================================================================
# 2. Transformer-GNN 特征提取器（整合图网络信息）
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, learnable=False):
        super().__init__()
        self.learnable = learnable
        if not learnable:
            pe = th.zeros(max_len, d_model)
            position = th.arange(0, max_len, dtype=th.float32).unsqueeze(1)
            div_term = th.exp(th.arange(0, d_model, 2, dtype=th.float32) * -(th.log(th.tensor(10000.0)) / d_model))
            pe[:, 0::2] = th.sin(position * div_term)
            pe[:, 1::2] = th.cos(position * div_term)
            pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
            self.register_buffer("pe", pe)
        else:
            self.pe = nn.Parameter(th.zeros(1, max_len, d_model))

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:, :seq_len, :]


class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + attn_output
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        gate = th.sigmoid(self.gate(src))
        src = src + self.dropout2(src2) * gate
        src = self.norm2(src)
        return src


class GatedTransformerGNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, seq_len=10, d_model=128, nhead=8,
                 num_layers=2, transformer_out_dim=128, gnn_hidden_dim=64,
                 d_emb=32, dropout=0.1, temperature=0.7, learnable_pe=False, adjacency=None):
        super(GatedTransformerGNNFeatureExtractor, self).__init__(
            observation_space, features_dim=transformer_out_dim
        )
        self.d_model = d_model
        self.temperature = temperature
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, learnable=learnable_pe)

        encoder_layer = GatedTransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 2, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.gnn_layer = nn.Linear(80, gnn_hidden_dim)
        self.output_layer = nn.Linear(d_model + gnn_hidden_dim, transformer_out_dim)

        if adjacency is not None:
            self.register_buffer("adjacency", adjacency)
        else:
            self.adjacency = None

        obs_dim = observation_space.shape[0]
        static_dim = obs_dim - 82  # [static_features, chosen_mask(80), progress(1), avg_reward(1)]
        if static_dim != d_model:
            self.proj = nn.Linear(static_dim, d_model)
        else:
            self.proj = nn.Identity()

        self.seq_len = 1

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)

        batch_size = observations.shape[0]
        obs_dim = observations.shape[1]
        static_len = obs_dim - 82

        static_feats = observations[:, :static_len]  # (B, static_dim)
        chosen_mask = observations[:, static_len:static_len + 80]  # (B, 80)

        seq = static_feats.unsqueeze(1)
        seq = self.proj(seq)
        seq = seq.transpose(0, 1)
        seq = self.pos_encoder(seq)
        transformer_out = self.transformer_encoder(seq)
        transformer_out = transformer_out.squeeze(0)

        if self.adjacency is not None:
            gnn_input = th.matmul(chosen_mask, self.adjacency)
        else:
            gnn_input = chosen_mask

        gnn_out = F.relu(self.gnn_layer(gnn_input))
        combined = th.cat([transformer_out, gnn_out], dim=1)
        features = F.relu(self.output_layer(combined))
        return features


# =============================================================================
# 3. 强化学习环境 —— 彩票选号环境
# =============================================================================
class LotterySequentialEnv(gym.Env):
    def __init__(self, data_df, extra_feature_cols, extra_max, extra_mean, extra_std, reward_weights=None,
                 bonus_multiplier=5.0):  # 提升奖励倍数
        super(LotterySequentialEnv, self).__init__()
        self.data = data_df.reset_index(drop=True)
        self.extra_feature_cols = extra_feature_cols
        self.extra_max = extra_max.values.astype(np.float32)
        self.extra_mean = extra_mean.values.astype(np.float32)
        self.extra_std = extra_std.values.astype(np.float32)

        self.sorted_numbers = self.data[[f"排好序_{j}" for j in range(1, 21)]].values.astype(np.int32)
        self.order_numbers = self.data[[f"出球顺序_{j}" for j in range(1, 21)]].values.astype(np.int32)

        if reward_weights is None:
            self.base_reward_weights = {
                "trend": 0.25,
                "diversity": 0.20,
                "repeat_penalty": -1.0,
                "surprise": 0.15,
                "spread": 0.15,
                "confidence": 0.05,
                "coverage": 0.05
            }
        else:
            self.base_reward_weights = reward_weights
        self.reward_weights = self.base_reward_weights.copy()

        self.start_index = 180
        self.selected_numbers = []
        self.current_step = 0
        self.current_index = self.start_index
        self.current_date = self.data.loc[self.current_index, "Date"]
        self.accumulated_reward = 0.0
        self.known_draws = {
            row["Date"]: set(int(row[f"排好序_{j}"]) for j in range(1, 21))
            for _, row in self.data.iterrows()
        }
        self.static_features = self._compute_static_features()
        dummy_obs = self._get_observation()
        self.obs_dim = dummy_obs.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(80)
        self.bonus_multiplier = bonus_multiplier
        self.episode_count = 0

    def update_reward_weights(self):
        # 可在此处引入更复杂的策略，目前仍采用线性更新
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

        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)
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
        win_sizes = [7, 14, 30, 60, 90, 180]
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
        safe_extra_max = np.where(self.extra_max == 0, 1, self.extra_max)
        normalized_extra_features = np.nan_to_num(extra_features / safe_extra_max)

        static_features = np.concatenate(
            [base_features, window_features, ema_feature, cum_freq, order_features, normalized_extra_features]
        )
        return static_features.astype(np.float32)

    def _get_observation(self):
        chosen_mask = np.zeros(80, dtype=np.float32)
        for num in self.selected_numbers:
            chosen_mask[int(num) - 1] = 1.0
        progress = np.array([self.current_step / self.max_steps], dtype=np.float32)
        if self.current_step > 0:
            avg_reward = np.array([self.accumulated_reward / self.current_step])
        else:
            avg_reward = np.array([0.0])
        observation = np.concatenate([self.static_features, chosen_mask, progress, avg_reward])
        return observation

    @property
    def max_steps(self):
        return 20

    def reset(self, index=None):
        self.episode_count += 1
        self.update_reward_weights()
        self.current_step = 0
        self.selected_numbers = []
        self.accumulated_reward = 0.0
        if index is not None:
            self.current_index = index
        else:
            self.current_index = np.random.randint(self.start_index, len(self.data))
        self.current_date = self.data.loc[self.current_index, "Date"]
        self.actual_set = self.known_draws.get(self.current_date, set())
        self.static_features = self._compute_static_features()
        return self._get_observation()

    def compute_reward(self, action):
        ema_feature = self._compute_ema_features(span=30)

        trend_value = ema_feature[action] / (np.mean(ema_feature) + 1e-6)
        reward_trend = self.reward_weights["trend"] * trend_value

        diversity = len(set(self.selected_numbers)) / self.max_steps
        reward_diversity = self.reward_weights["diversity"] * diversity

        ema_sum = np.sum(ema_feature) + 1e-6
        prob = ema_feature[action] / ema_sum
        surprise = -np.log(prob + 1e-6)
        normalized_surprise = surprise / np.log(80)
        reward_surprise = self.reward_weights["surprise"] * normalized_surprise

        if len(self.selected_numbers) > 0:
            current_spread = (max(self.selected_numbers) - min(self.selected_numbers)) / 79.0
            new_selection = self.selected_numbers + [action + 1]
            new_spread = (max(new_selection) - min(new_selection)) / 79.0
            spread_reward = self.reward_weights["spread"] * (new_spread - current_spread)
            if new_spread < 0.6:
                spread_reward += -0.6 * (0.6 - new_spread)
        else:
            spread_reward = 0

        if len(self.selected_numbers) >= 2:
            std_val = np.std(self.selected_numbers) / 80.0
            diversity_bonus = 0.1 * std_val
        else:
            diversity_bonus = 0

        reward_confidence = self.reward_weights["confidence"] * (prob - 1 / 80)
        reward_coverage = self.reward_weights["coverage"] * (len(set(self.selected_numbers)) / 80)

        hot_bonus = 0
        avg_ema = np.mean(ema_feature)
        max_ema = np.max(ema_feature)
        if ema_feature[action] > avg_ema and ema_feature[action] >= 0.9 * max_ema:
            hot_bonus = 0.1 * (ema_feature[action] - avg_ema)

        total_reward = (
            reward_trend
            + reward_diversity
            + reward_surprise
            + spread_reward
            + diversity_bonus
            + reward_confidence
            + reward_coverage
            + hot_bonus
        )

        self.selected_numbers.append(action + 1)
        return total_reward

    def step(self, action):
        if (action + 1) in self.selected_numbers:
            logging.debug("重复选择，自动修正动作")
            valid_actions = list(set(range(1, 81)) - set(self.selected_numbers))
            if valid_actions:
                action = random.choice(valid_actions) - 1
            reward = self.reward_weights.get("repeat_penalty", -1.0)
        else:
            reward = self.compute_reward(action)

        self.accumulated_reward += reward
        self.current_step += 1

        done = (self.current_step == self.max_steps)
        info = {}
        if done:
            predicted_set = set(self.selected_numbers)
            hits = len(predicted_set & self.actual_set) if self.actual_set is not None else 0
            bonus_reward = self.bonus_multiplier * hits
            reward += bonus_reward
            info = {
                "predicted_set": predicted_set,
                "actual_set": self.actual_set,
                "hits": hits,
                "bonus_reward": bonus_reward
            }
            self.info = info

        return self._get_observation(), reward, done, info


# =============================================================================
# 4. 预测与评估辅助函数
# =============================================================================
def simulate_episode_with_prob(env, model, future_date, num_runs=10):
    prob_accum = np.zeros(80)
    for i in range(num_runs):
        env.reset()
        env.current_date = future_date
        env.actual_set = None
        while env.current_step < env.max_steps:
            obs = env._get_observation()
            action, _ = model.predict(obs, deterministic=True)
            if (action + 1) in env.selected_numbers:
                valid_actions = list(set(range(1, 81)) - set(env.selected_numbers))
                if valid_actions:
                    action = random.choice(valid_actions) - 1
            env.step(action)
        noise = np.random.normal(0, 0.02, size=prob_accum.shape)
        prob_accum += noise
        for num in env.selected_numbers:
            prob_accum[num - 1] += 1
    noise = np.random.normal(0, 0.02, size=prob_accum.shape)
    prob_accum = prob_accum + noise
    prob = prob_accum / num_runs
    prob = np.clip(prob, 1e-6, None)
    if prob.sum() > 0:
        prob = prob / prob.sum()
    return prob


def compute_ece(preds, true_labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    total_samples = preds.size
    ece = 0.0
    for i in range(n_bins):
        mask = (preds >= bin_boundaries[i]) & (preds < bin_boundaries[i + 1])
        if np.sum(mask) > 0:
            avg_confidence = preds[mask].mean()
            avg_accuracy = true_labels[mask].mean()
            ece += (np.sum(mask) / total_samples) * abs(avg_confidence - avg_accuracy)
    return ece


def compute_brier_score(probs, true_labels):
    return np.mean((probs - true_labels) ** 2)


def compute_entropy(probs):
    p = np.clip(probs, 1e-6, 1.0)
    return -np.sum(p * np.log(p)) / np.log(len(p))


# =============================================================================
# 5. 模型训练与评估
# =============================================================================
def train_model(algorithm, env, policy_kwargs, total_timesteps, update_timesteps, model_save_path):
    model = None
    if algorithm == "QRDQN":
        from stable_baselines3 import DQN
        model = DQN(
            "MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-3,
            buffer_size=10000, exploration_fraction=0.2, train_freq=1, target_update_interval=1000,
            max_grad_norm=10, learning_starts=1000
        )
        steps = 0
        epoch = 0
        while steps < total_timesteps:
            model.learn(total_timesteps=update_timesteps, reset_num_timesteps=False)
            steps += update_timesteps
            epoch += 1
            logging.info(f"{algorithm} 已训练 {steps} 步")
    elif algorithm == "PPO":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-3)
        model.learn(total_timesteps=total_timesteps)
    elif algorithm == "A2C":
        from stable_baselines3 import A2C
        model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-3)
        model.learn(total_timesteps=total_timesteps)

    model.save(model_save_path)
    logging.info(f"{algorithm} 模型已保存至 {model_save_path}")
    return model


def evaluate_model_recent(env, model, df, recent_days=30):
    hit_counts = []
    recent_date = df["Date"].max() - pd.Timedelta(days=recent_days)
    for idx in range(env.start_index, len(df)):
        if df.loc[idx, "Date"] >= recent_date:
            obs = env.reset(index=idx)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            if "hits" in info:
                hit_counts.append(info["hits"])
            else:
                actual_set = env.known_draws.get(df.loc[idx, "Date"], set())
                pred_set = set(env.selected_numbers)
                hit_counts.append(len(pred_set & actual_set))
    return np.mean(hit_counts) if hit_counts else 0


def evaluate_model_multiple(env, model, df, days_list):
    scores = []
    for d in days_list:
        score = evaluate_model_recent(env, model, df, recent_days=d)
        scores.append(score)
    return np.mean(scores)


# =============================================================================
# 6. 扩散模型（Diffusion Model）集成与训练
# =============================================================================
class DiffusionModel:
    def __init__(self, static_feat_dim, T=50, time_embed_dim=32, hidden_dim=256, beta_schedule_type="cosine"):
        self.T = T
        self.time_embed_dim = time_embed_dim
        # 选择 beta 调度策略：余弦或线性
        if beta_schedule_type == "cosine":
            self.beta_schedule = cosine_beta_schedule(T, s=0.008)
        else:
            self.beta_schedule = np.linspace(1e-4, 0.02, T, dtype=np.float32)
        self.alpha = 1.0 - self.beta_schedule
        self.alpha_cum = np.cumprod(self.alpha)

        input_dim = 80 + static_feat_dim + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 80)
        )
        self.optimizer = th.optim.Adam(self.net.parameters(), lr=1e-3)
        self.device = th.device("cpu")
        self.net.to(self.device)
        self.net.train()

    def _time_embedding(self, t):
        if isinstance(t, th.Tensor):
            t_float = t.float()
        else:
            t_float = th.tensor(float(t), dtype=th.float32)
        t_flat = t_float.view(-1)
        batch_size = t_flat.shape[0]

        half_dim = self.time_embed_dim // 2
        exponent = th.exp(th.arange(half_dim, dtype=th.float32) * -(th.log(th.tensor(10000.0)) / half_dim))
        exponent = exponent.to(self.device)

        angles = t_flat.unsqueeze(1) * exponent.unsqueeze(0)
        emb = th.zeros(batch_size, self.time_embed_dim, device=self.device)
        emb[:, 0::2] = th.sin(angles)
        emb[:, 1::2] = th.cos(angles)
        if emb.shape[0] == 1:
            emb = emb.squeeze(0)
        return emb

    def predict_noise(self, x_t, t, context):
        if not th.is_tensor(x_t):
            x_t = th.tensor(x_t, dtype=th.float32)
        if not th.is_tensor(context):
            context = th.tensor(context, dtype=th.float32)

        x_t = x_t.to(self.device)
        context = context.to(self.device)

        t_tensor = t if th.is_tensor(t) else th.tensor([t], dtype=th.long)
        t_embed = self._time_embedding(t_tensor)
        if t_embed.dim() == 1:
            t_embed = t_embed.unsqueeze(0)
        if t_embed.shape[0] != x_t.shape[0]:
            t_embed = t_embed.repeat(x_t.shape[0], 1)

        combined_input = th.cat([x_t, context, t_embed], dim=1)
        noise_pred = self.net(combined_input)
        return noise_pred

    def train_model(self, env, known_draws, epochs=50, batch_size=64):
        X_context = []
        Y_target = []
        for idx in range(env.start_index, len(env.data)):
            env.reset(index=idx)
            static_feat = env.static_features
            actual_set = known_draws.get(env.current_date, set())
            true_vec = np.zeros(80, dtype=np.float32)
            for num in actual_set:
                true_vec[num - 1] = 1.0
            X_context.append(static_feat)
            Y_target.append(true_vec)

        X_context = np.array(X_context, dtype=np.float32)
        Y_target = np.array(Y_target, dtype=np.float32)
        data_size = X_context.shape[0]

        X_context_t = th.tensor(X_context, dtype=th.float32).to(self.device)
        Y_target_t = th.tensor(Y_target, dtype=th.float32).to(self.device)

        for epoch in range(1, epochs + 1):
            perm = th.randperm(data_size)
            total_loss = 0.0
            for i in range(0, data_size, batch_size):
                batch_idx = perm[i:i + batch_size]
                x_context_batch = X_context_t[batch_idx]
                y_batch = Y_target_t[batch_idx]

                t_batch = th.randint(1, self.T + 1, (len(batch_idx),), device=self.device)
                noise = th.randn(y_batch.size(), device=self.device)

                alpha_cum_t = th.from_numpy(self.alpha_cum).to(self.device)[t_batch - 1]
                alpha_cum_t = alpha_cum_t.view(-1, 1)

                x_t = (alpha_cum_t.sqrt() * y_batch) + ((1 - alpha_cum_t).sqrt() * noise)

                noise_pred = self.predict_noise(x_t, t_batch, x_context_batch)
                loss = F.mse_loss(noise_pred, noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(batch_idx)

            avg_loss = total_loss / data_size
            logging.info(f"Diffusion Model 训练 Epoch {epoch}/{epochs}, 平均损失: {avg_loss:.6f}")

        self.net.eval()

    def generate_sample(self, context):
        if not th.is_tensor(context):
            context_t = th.tensor(context, dtype=th.float32).unsqueeze(0)
        else:
            context_t = context.clone().detach().float().unsqueeze(0)
        context_t = context_t.to(self.device)

        x_t = th.randn((1, 80), device=self.device)
        for t in range(self.T, 0, -1):
            t_tensor = th.tensor([t], dtype=th.long, device=self.device)
            noise_pred = self.predict_noise(x_t, t_tensor, context_t)
            alpha_t = self.alpha[t - 1]
            alpha_cum_t = self.alpha_cum[t - 1]

            coef = (1 - alpha_t) / (np.sqrt(1 - alpha_cum_t) + 1e-8)
            x_pred = (1.0 / np.sqrt(alpha_t)) * (x_t - coef * noise_pred)

            if t > 1:
                beta_t = self.beta_schedule[t - 1]
                noise = th.randn_like(x_t)
                sigma_t = np.sqrt(beta_t)
                x_t = x_pred + sigma_t * noise
            else:
                x_t = x_pred

        x0 = x_t.detach().cpu().numpy().reshape(-1)
        top20_indices = np.argsort(x0)[::-1][:20]
        predicted_set = [int(i + 1) for i in top20_indices]
        return predicted_set


def simulate_diffusion_with_prob(env, diff_model, target_date, num_runs=10):
    prob_accum = np.zeros(80)
    if target_date in env.known_draws:
        idx = env.data.index[env.data["Date"] == target_date][0]
        env.reset(index=idx)
        context_vec = env.static_features
    else:
        env.reset(index=len(env.data) - 1)
        env.current_date = target_date
        context_vec = env.static_features

    for i in range(num_runs):
        predicted_set = diff_model.generate_sample(context_vec)
        prob_accum += np.random.normal(0, 0.02, size=prob_accum.shape)
        for num in predicted_set:
            prob_accum[num - 1] += 1

    prob_accum += np.random.normal(0, 0.02, size=prob_accum.shape)
    prob = prob_accum / num_runs
    prob = np.clip(prob, 1e-6, None)
    if prob.sum() > 0:
        prob = prob / prob.sum()
    return prob


def evaluate_model_recent_diffusion(env, diff_model, df, recent_days=30):
    hit_counts = []
    recent_date = df["Date"].max() - pd.Timedelta(days=recent_days)
    for idx in range(env.start_index, len(df)):
        if df.loc[idx, "Date"] >= recent_date:
            date = df.loc[idx, "Date"]
            actual_set = env.known_draws.get(date, set())
            prob = simulate_diffusion_with_prob(env, diff_model, date, num_runs=5)
            pred_set = set(np.argsort(prob)[::-1][:20] + 1)
            hits = len(pred_set & actual_set)
            hit_counts.append(hits)
    return np.mean(hit_counts) if hit_counts else 0


def evaluate_model_multiple_diffusion(env, diff_model, df, days_list):
    scores = []
    for d in days_list:
        score = evaluate_model_recent_diffusion(env, diff_model, df, recent_days=d)
        scores.append(score)
    return np.mean(scores)


# =============================================================================
# 7. 元学习集成策略与超参数调优
# =============================================================================
def tune_ensemble_weights(env, models, diff_model, df, eval_dates, n_trials=50):
    def objective(trial):
        weights = []
        for j in range(len(models)):
            w = trial.suggest_float(f"w_model{j + 1}", 0.0, 1.0)
            weights.append(w)
        w_diff = trial.suggest_float("w_diff", 0.0, 1.0)
        weights.append(w_diff)
        weights = np.array(weights)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        hit_counts = []
        for date in eval_dates:
            ensemble_prob = np.zeros(80)
            for j, model in enumerate(models):
                prob_model = simulate_episode_with_prob(env, model, date, num_runs=3)
                ensemble_prob += weights[j] * prob_model
            prob_diff = simulate_diffusion_with_prob(env, diff_model, date, num_runs=3)
            ensemble_prob += weights[-1] * prob_diff

            if ensemble_prob.sum() > 0:
                ensemble_prob = ensemble_prob / ensemble_prob.sum()
            pred_set = set(np.argsort(ensemble_prob)[::-1][:20] + 1)
            actual_set = env.known_draws.get(date, set())
            hits = len(pred_set & actual_set)
            hit_counts.append(hits)

        return np.mean(hit_counts)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_weights = [best_params.get(f"w_model{j + 1}", 0.0) for j in range(len(models))]
    best_weights.append(best_params.get("w_diff", 0.0))
    best_weights = np.array(best_weights)

    if best_weights.sum() == 0:
        best_weights = np.ones_like(best_weights)
    best_weights = best_weights / best_weights.sum()
    logging.info(f"Optuna 最优权重: {best_weights}")
    return best_weights


# =============================================================================
# 8. 结果绘图辅助函数
# =============================================================================
def plot_daily_hits(dates, hit_counts, save_path):
    plt.figure(figsize=(10, 6))
    plt.bar(dates, hit_counts, color="skyblue")
    plt.xlabel("日期")
    plt.ylabel("命中数量")
    plt.title("最近一个月每天的命中数量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"每日命中图已保存至 {save_path}")


# =============================================================================
# 9. 主流程：训练、评估、预测、结果保存与绘图
# =============================================================================
def main():
    df, extra_feature_cols, extra_max, extra_mean, extra_std, known_draws = load_data()
    df = add_external_features(df)
    df = add_context_features(df)
    df = augment_data_advanced(df)
    df = add_delay_features(df)
    df = add_omission_features(df)

    extra_feature_cols = extra_feature_cols + [
        "draw_std", "draw_sum_lag1", "draw_sum_ma3",
        "consecutive_count", "min_diff", "max_diff", "mean_diff",
        "weekday", "weekofyear", "month", "quarter", "is_holiday", "is_workday",
        "delay_mean", "delay_min", "delay_max", "rolling_mean", "rolling_std"
    ]
    for col in ["temperature", "humidity", "wind_speed", "search_trend"]:
        if col in df.columns:
            extra_feature_cols.append(col)
    extra_feature_cols += [f"hot_count_{num}" for num in range(1, 81)]
    extra_max = df[extra_feature_cols].max()
    extra_mean = df[extra_feature_cols].mean()
    extra_std = df[extra_feature_cols].std().replace(0, 1)

    lottery_graph = build_lottery_graph(df)

    dst_dir = "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/" + pd.Timestamp.now().strftime("%Y%m%d") + "/"
    os.makedirs(dst_dir, exist_ok=True)

    TOTAL_TIMESTEPS = 100000
    TIMESTEPS_PER_UPDATE = 1000

    policy_kwargs = {
        "features_extractor_class": GatedTransformerGNNFeatureExtractor,
        "features_extractor_kwargs": {
            "seq_len": 1,
            "d_model": 128,
            "nhead": 8,
            "num_layers": 2,
            "transformer_out_dim": 128,
            "gnn_hidden_dim": 64,
            "d_emb": 32,
            "dropout": 0.1,
            "temperature": 0.7,
            "learnable_pe": True,
            "adjacency": lottery_graph
        }
    }

    env_qrdqn = LotterySequentialEnv(df, extra_feature_cols, extra_max, extra_mean, extra_std, bonus_multiplier=5.0)
    env_ppo = LotterySequentialEnv(df, extra_feature_cols, extra_max, extra_mean, extra_std, bonus_multiplier=5.0)
    env_a2c = LotterySequentialEnv(df, extra_feature_cols, extra_max, extra_mean, extra_std, bonus_multiplier=5.0)

    logging.info("开始训练 QRDQN 模型...")
    model_qrdqn = train_model(
        "QRDQN", env_qrdqn, policy_kwargs, TOTAL_TIMESTEPS, TIMESTEPS_PER_UPDATE,
        os.path.join(dst_dir, "model_qrdqn")
    )

    logging.info("开始训练 PPO 模型...")
    model_ppo = train_model(
        "PPO", env_ppo, policy_kwargs, TOTAL_TIMESTEPS, TIMESTEPS_PER_UPDATE,
        os.path.join(dst_dir, "model_ppo")
    )

    logging.info("开始训练 A2C 模型...")
    model_a2c = train_model(
        "A2C", env_a2c, policy_kwargs, TOTAL_TIMESTEPS, TIMESTEPS_PER_UPDATE,
        os.path.join(dst_dir, "model_a2c")
    )

    env_eval = LotterySequentialEnv(df, extra_feature_cols, extra_max, extra_mean, extra_std, bonus_multiplier=5.0)

    days_list = [1, 3, 5, 7, 14, 21, 30]
    qrdqn_score = evaluate_model_multiple(env_eval, model_qrdqn, df, days_list)
    ppo_score = evaluate_model_multiple(env_eval, model_ppo, df, days_list)
    a2c_score = evaluate_model_multiple(env_eval, model_a2c, df, days_list)
    logging.info(f"多窗口近期表现 -> QRDQN: {qrdqn_score:.2f}, PPO: {ppo_score:.2f}, A2C: {a2c_score:.2f}")

    total_score = qrdqn_score + ppo_score + a2c_score
    w_qrdqn = qrdqn_score / total_score if total_score > 0 else 1 / 3
    w_ppo = ppo_score / total_score if total_score > 0 else 1 / 3
    w_a2c = a2c_score / total_score if total_score > 0 else 1 / 3

    logging.info(f"动态权重 -> QRDQN: {w_qrdqn:.2f}, PPO: {w_ppo:.2f}, A2C: {w_a2c:.2f}")

    logging.info("开始训练 Diffusion 扩散模型...")
    static_feat_dim = env_eval.static_features.shape[0]
    diff_model = DiffusionModel(static_feat_dim=static_feat_dim, T=50, time_embed_dim=32, hidden_dim=256, beta_schedule_type="cosine")
    diff_model.train_model(env_eval, known_draws, epochs=100000, batch_size=128)

    diff_score = evaluate_model_multiple_diffusion(env_eval, diff_model, df, days_list)
    logging.info(f"多窗口近期表现 -> Diffusion: {diff_score:.2f}")

    total_score_all = qrdqn_score + ppo_score + a2c_score + diff_score
    if total_score_all > 0:
        w_qrdqn = qrdqn_score / total_score_all
        w_ppo = ppo_score / total_score_all
        w_a2c = a2c_score / total_score_all
        w_diff = diff_score / total_score_all
    else:
        w_qrdqn = w_ppo = w_a2c = w_diff = 0.25

    logging.info(f"初始集成权重 -> QRDQN: {w_qrdqn:.2f}, PPO: {w_ppo:.2f}, A2C: {w_a2c:.2f}, Diffusion: {w_diff:.2f}")

    eval_dates = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=30)]["Date"].tolist()

    all_probs = []
    topn = 20
    topn_hits = []
    union_pred = set()
    union_actual = set()
    # 用于统计每天的命中数量（方便后续绘图）
    daily_hit_counts = {}

    for date in eval_dates:
        prob_qrdqn = simulate_episode_with_prob(env_eval, model_qrdqn, date, num_runs=5)
        prob_ppo = simulate_episode_with_prob(env_eval, model_ppo, date, num_runs=5)
        prob_a2c = simulate_episode_with_prob(env_eval, model_a2c, date, num_runs=5)
        prob_diff = simulate_diffusion_with_prob(env_eval, diff_model, date, num_runs=5)

        ensemble_prob = w_qrdqn * prob_qrdqn + w_ppo * prob_ppo + w_a2c * prob_a2c + w_diff * prob_diff
        if ensemble_prob.sum() > 0:
            ensemble_prob = ensemble_prob / ensemble_prob.sum()

        true_vec = np.zeros(80)
        actual_set = env_eval.known_draws.get(date, set())
        for num in actual_set:
            true_vec[num - 1] = 1

        all_probs.append(ensemble_prob)
        top_pred = set(np.argsort(ensemble_prob)[::-1][:topn] + 1)
        hits = len(top_pred & actual_set)
        topn_hits.append(hits / topn)

        union_pred.update(top_pred)
        union_actual.update(actual_set)
        daily_hit_counts[date] = hits

    all_probs = np.vstack(all_probs)
    ece_value = compute_ece(
        all_probs,
        np.vstack([
            np.array([
                1 if i in env_eval.known_draws.get(date, set()) else 0
                for i in range(1, 81)
            ]) for date in eval_dates
        ])
    )
    brier_value = compute_brier_score(
        all_probs,
        np.vstack([
            np.array([
                1 if i in env_eval.known_draws.get(date, set()) else 0
                for i in range(1, 81)
            ]) for date in eval_dates
        ])
    )
    avg_entropy = np.mean([compute_entropy(p) for p in all_probs])
    top_precision = np.mean(topn_hits)
    diversity_score = len(union_pred) / 80.0
    coverage = len(union_pred & union_actual) / (len(union_actual) if len(union_actual) > 0 else 1)

    logging.info(
        f"高级评估指标 -> 平均命中: {(np.mean(topn_hits) * topn):.2f}, ECE: {ece_value:.4f}, "
        f"多样性: {diversity_score:.4f}, Brier: {brier_value:.4f}, 覆盖率: {coverage:.4f}, "
        f"平均熵: {avg_entropy:.4f}, Top-{topn} 精度: {top_precision:.4f}"
    )

    if has_optuna:
        logging.info("开始使用 Optuna 进行集成权重调优...")
        best_weights = tune_ensemble_weights(
            env_eval, [model_qrdqn, model_ppo, model_a2c], diff_model, df, eval_dates,
            n_trials=30
        )
        w_qrdqn, w_ppo, w_a2c, w_diff = best_weights
        logging.info(
            f"Optuna调优后集成权重 -> QRDQN: {w_qrdqn:.2f}, PPO: {w_ppo:.2f}, A2C: {w_a2c:.2f}, Diffusion: {w_diff:.2f}"
        )

    future_date = df["Date"].max() + pd.Timedelta(days=1)
    env_pred = LotterySequentialEnv(df, extra_feature_cols, extra_max, extra_mean, extra_std, bonus_multiplier=5.0)
    prob_qrdqn = simulate_episode_with_prob(env_pred, model_qrdqn, future_date, num_runs=10)
    prob_ppo = simulate_episode_with_prob(env_pred, model_ppo, future_date, num_runs=10)
    prob_a2c = simulate_episode_with_prob(env_pred, model_a2c, future_date, num_runs=10)
    prob_diff = simulate_diffusion_with_prob(env_pred, diff_model, future_date, num_runs=10)

    ensemble_prob = w_qrdqn * prob_qrdqn + w_ppo * prob_ppo + w_a2c * prob_a2c + w_diff * prob_diff
    if ensemble_prob.sum() > 0:
        ensemble_prob = ensemble_prob / ensemble_prob.sum()

    calib_temp = 0.7
    calibrated_prob = np.exp(np.log(np.clip(ensemble_prob, 1e-9, None)) / calib_temp)
    calibrated_prob = calibrated_prob / np.sum(calibrated_prob)
    final_pred = np.argsort(calibrated_prob)[::-1][:20] + 1
    final_pred = final_pred.tolist()
    logging.info(f"预测 {future_date.date()} 的号码: {final_pred}")

    def save_prediction(pred_list, model_name):
        pred_df = pd.DataFrame(
            [{"Date": future_date.date(), **{f"Pred{j}": int(pred_list[j - 1]) for j in range(1, 21)}}]
        )
        csv_path = os.path.join(dst_dir, f"{model_name}_prediction_v18_FIXED_TOTALTIMESTEPS_{TOTAL_TIMESTEPS}.csv")
        pred_df.to_csv(csv_path, index=False)
        logging.info(f"{model_name} 预测结果已保存至 {csv_path}")

    save_prediction(list(np.argsort(np.array(prob_qrdqn))[::-1][:20] + 1), "qrdqn")
    save_prediction(list(np.argsort(np.array(prob_ppo))[::-1][:20] + 1), "ppo")
    save_prediction(list(np.argsort(np.array(prob_a2c))[::-1][:20] + 1), "a2c")
    save_prediction(final_pred, "ensemble")

    final_df = pd.DataFrame(
        [{"Date": future_date.date(), **{f"Pred{j}": int(final_pred[j - 1]) for j in range(1, 21)}}]
    )
    final_md_path = os.path.join(dst_dir, f"RL_lottery_predictor_v18_FIXED_{TOTAL_TIMESTEPS}.md")
    final_df.to_csv(final_md_path, index=False)
    logging.info(f"集成预测结果已保存至 {final_md_path}")

    # 绘制最近一个月每天的命中数量图
    # 将日期转换为字符串便于绘图
    plot_dates = [date.strftime("%Y-%m-%d") for date in daily_hit_counts.keys()]
    plot_hits = list(daily_hit_counts.values())
    plot_save_path = os.path.join(dst_dir, "daily_hits_RL_lottery_predictor_v18_.png")
    plot_daily_hits(plot_dates, plot_hits, plot_save_path)


if __name__ == "__main__":
    main()
