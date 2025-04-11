#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合提升强化学习彩票预测准确率的策略：
1. 丰富的特征工程：加入更多历史开奖数据、组合特征、节假日上下文等
2. 引入 LSTM 模块作为序列特征提取器（含补零处理确保尺寸整除）
3. 精细的趋势奖励设计及多目标奖励（趋势 + 多样性）
4. 模型集成策略优化（动态加权融合）

数据读取和保存路径沿用之前的设置。
"""

import os
import random
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
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

# 导入 torch 和 stable-baselines3 相关模块
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 可选：如果安装了 Optuna，可用于超参数调优
try:
    import optuna
    has_optuna = True
except ImportError:
    has_optuna = False

# 配置日志格式
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =============================================================================
# 1. 数据读取与增强：加载数据、丰富上下文特征与历史统计特征
# =============================================================================
def load_data():
    # 数据路径（沿用之前设置）
    data_path = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/kl8/kl8_order_data.csv"
    df = pd.read_csv(data_path, encoding='utf-8', parse_dates=["开奖日期"])
    df.rename(columns={"开奖日期": "Date"}, inplace=True)
    df = df.sort_values("Date").reset_index(drop=True)
    logging.info(f"Loaded data: {len(df)} 天，从 {df['Date'].min().date()} 至 {df['Date'].max().date()}")
    # 原有业务特征列（共 80 个），假设已有号码特征列名为 "排好序_1" ~ "排好序_20"
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
    # 数字列处理：去掉逗号，转换为 float
    for col in extra_feature_cols:
        df[col] = df[col].replace(',', '', regex=True).astype(float)
    df[extra_feature_cols] = df[extra_feature_cols].fillna(0)
    extra_max = df[extra_feature_cols].max()

    # 构造每期开奖号码集合（假设列 "排好序_1" 至 "排好序_20"）
    known_draws = {
        row['Date']: set(row[f'排好序_{j}'] for j in range(1, 21))
        for _, row in df.iterrows()
    }
    return df, extra_feature_cols, extra_max, known_draws


def add_context_features(df):
    """增加日期上下文特征和历史冷热统计特征"""
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

    # 示例：统计过去30天内每个号码出现次数（冷热特征）
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
    """原有特征增强：统计标准差、上一期总和与滑动平均；连号计数；相邻差值统计"""
    sorted_cols = [f'排好序_{j}' for j in range(1, 21)]
    df['draw_std'] = df[sorted_cols].std(axis=1)
    df['draw_sum_lag1'] = df[sorted_cols].sum(axis=1).shift(1).fillna(0)
    df['draw_sum_ma3'] = df['draw_sum_lag1'].rolling(window=3, min_periods=1).mean().fillna(0)

    # 连号计数
    def count_consecutive(row):
        nums = list(row)
        return sum(1 for i in range(1, len(nums)) if nums[i] == nums[i-1] + 1)
    df['consecutive_count'] = df[sorted_cols].apply(count_consecutive, axis=1)

    # 相邻差值统计：最小、最大、平均
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


# =============================================================================
# 2. 模型特征提取器：使用 LSTM 提取序列特征（含补零处理确保尺寸整除）
# =============================================================================
class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, seq_len=10, hidden_dim=64, features_dim=128):
        # observation_space.shape[0] 预期为总特征维度
        super(LSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        # 此处暂时不确定 input_dim，将在 forward 中根据补零后确定
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        B = observations.size(0)
        total_dim = observations.size(1)
        # 计算需要补充的维度，使得总长度能被 seq_len 整除
        pad_size = (self.seq_len - total_dim % self.seq_len) % self.seq_len
        if pad_size > 0:
            padded_obs = th.nn.functional.pad(observations, (0, pad_size))
        else:
            padded_obs = observations
        new_total = padded_obs.size(1)
        input_dim = new_total // self.seq_len
        # reshape 成 (B, seq_len, input_dim)
        obs_seq = padded_obs.view(B, self.seq_len, input_dim)
        # 对每个时间步取均值（简化处理），并扩展为 (B, seq_len, 1)
        obs_seq = obs_seq.mean(dim=2, keepdim=True)
        self.lstm.flatten_parameters()
        _, (hn, _) = self.lstm(obs_seq)
        features = self.linear(hn[-1])
        return features


# =============================================================================
# 3. 强化学习环境：结合扩展特征与奖励函数多目标设计
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
        self.obs_dim = None  # 待初始化

        # 先初始化 observation_space 为占位（稍后更新）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(80)
        self.max_steps = 20

        self.episode_count = 0
        if reward_weights is None:
            self.base_reward_weights = {
                "trend": 0.25,
                "diversity": 0.20,
                "repeat_penalty": -1.0
            }
        else:
            self.base_reward_weights = reward_weights
        self.reward_weights = self.base_reward_weights.copy()

        # —— 在构造函数中进行一次虚拟 reset，以计算正确的观测维度并更新 observation_space  ——
        self.selected_numbers = []  # 初始化 selected_numbers
        self.current_step = 0       # 初始化 current_step
        self.current_index = self.start_index
        self.current_date = self.data.loc[self.current_index, 'Date']
        self.static_features = self._compute_static_features()
        dummy_obs = self._get_observation()  # 获取观测向量
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
            [base_features, window_features, ema_feature, cum_freq, order_features, normalized_extra_features])
        return static_features.astype(np.float32)

    def _get_observation(self):
        # 动态部分：已选号码掩码（80维） + 当前步长比例
        chosen_mask = np.zeros(80, dtype=np.float32)
        for num in self.selected_numbers:
            chosen_mask[int(num) - 1] = 1.0
        progress = np.array([self.current_step / self.max_steps], dtype=np.float32)
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
        if (action + 1) in self.selected_numbers:
            return self.reward_weights["repeat_penalty"]
        ema_feature = self._compute_ema_features(span=30)
        trend_value = ema_feature[action] / (np.mean(ema_feature) + 1e-6)
        diversity = len(set(self.selected_numbers)) / self.max_steps
        reward_trend = self.reward_weights["trend"] * trend_value
        reward_diversity = self.reward_weights["diversity"] * diversity
        total_reward = reward_trend + reward_diversity
        self.selected_numbers.append(action + 1)
        return total_reward

    def step(self, action):
        if (action + 1) in self.selected_numbers:
            reward = self.reward_weights["repeat_penalty"]
        else:
            reward = self.compute_reward(action)
        self.current_step += 1
        done = (self.current_step == self.max_steps)
        info = {}
        if done:
            predicted_set = set(self.selected_numbers)
            hits = len(predicted_set & self.actual_set) if self.actual_set is not None else 0
            reward += hits
            info = {"predicted_set": predicted_set, "actual_set": self.actual_set, "hits": hits}
        return self._get_observation(), reward, done, info


# =============================================================================
# 4. 模型预测、集成与评估辅助函数
# =============================================================================
def complete_prediction(pred_list):
    if len(pred_list) < 20:
        missing = sorted(set(range(1, 81)) - set(pred_list))
        pred_list.extend(missing[:20 - len(pred_list)])
    return pred_list


def simulate_episode_for_prediction(env, model, future_date):
    env.reset()
    env.current_date = future_date
    env.actual_set = None  # 未来无真实开奖数据
    while env.current_step < env.max_steps:
        obs = env._get_observation()
        action, _ = model.predict(obs, deterministic=True)
        if (action + 1) in env.selected_numbers:
            valid_actions = list(set(range(1, 81)) - set(env.selected_numbers))
            if valid_actions:
                action = random.choice(valid_actions) - 1
        obs, reward, done, info = env.step(action)
    return complete_prediction(env.selected_numbers)


def simulate_episode_for_date(date, df, model, extra_feature_cols, extra_max):
    temp_env = LotterySequentialEnv(df, extra_feature_cols, extra_max)
    idx = df.index[df['Date'] == date][0]
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
    counter_votes = Counter(sum([list(pred) for pred in predictions], []))
    ensemble_nums = [num for num, cnt in counter_votes.items() if cnt >= 2]
    if len(ensemble_nums) < 20:
        remaining = [num for num in (predictions[0] | predictions[1] | predictions[2]) if num not in ensemble_nums]
        ensemble_nums.extend(sorted(remaining, key=lambda x: counter_votes[x], reverse=True))
    if len(ensemble_nums) < 20:
        missing = sorted(set(range(1, 81)) - set(ensemble_nums))
        ensemble_nums.extend(missing)
    return sorted(ensemble_nums[:20])


def dynamic_ensemble_prediction(preds, weights, top_k=20):
    from collections import defaultdict
    score_map = defaultdict(float)
    for pred, w in zip(preds, weights):
        for num in pred:
            score_map[num] += w
    sorted_pred = sorted(score_map.items(), key=lambda x: -x[1])
    return [x[0] for x in sorted_pred[:top_k]]


def plot_hit_trend(df, models, env, extra_feature_cols, extra_max, dst_dir, total_timesteps):
    hit_counts = []
    dates_list = []
    for idx in range(env.start_index, len(df)):
        date = df.loc[idx, 'Date']
        ensemble_pred = ensemble_prediction_for_date(date, df, models, extra_feature_cols, extra_max)
        actual_set = env.known_draws.get(date, set())
        hits = len(set(ensemble_pred) & actual_set)
        hit_counts.append(hits)
        dates_list.append(date)
    last_date = df['Date'].max()
    one_month_ago = last_date - pd.Timedelta(days=30)
    filtered_dates = [d for d in dates_list if d >= one_month_ago]
    filtered_hits = [h for d, h in zip(dates_list, hit_counts) if d >= one_month_ago]
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_dates, filtered_hits, marker='o', label="最近一个月每日命中数")
    avg_hits_filtered = np.mean(filtered_hits) if filtered_hits else 0
    plt.axhline(y=avg_hits_filtered, color='r', linestyle='--', label=f"平均命中 = {avg_hits_filtered:.2f}")
    plt.xlabel("日期")
    plt.ylabel("命中数（共20个）")
    plt.title("最近一个月训练集集成预测命中趋势")
    plt.legend()
    plt.tight_layout()
    hit_trend_path = os.path.join(dst_dir, f"RL_v20250411_TOTAL_TIMESTEPS_{total_timesteps}.png")
    plt.savefig(hit_trend_path)
    logging.info(f"命中趋势图已保存至 {hit_trend_path}")


# =============================================================================
# 5. 模型训练与评估函数
# =============================================================================
def train_model(algorithm, env, policy_kwargs, total_timesteps, update_timesteps, model_save_path):
    if algorithm == "QRDQN":
        from sb3_contrib.qrdqn import QRDQN
        from torch.optim.lr_scheduler import StepLR
        model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-3)
        optimizer = model.policy.optimizer
        scheduler = StepLR(optimizer, step_size=update_timesteps, gamma=0.9)
        steps = 0
        while steps < total_timesteps:
            model.learn(total_timesteps=update_timesteps, reset_num_timesteps=False)
            steps += update_timesteps
            scheduler.step()
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
    recent_date = df['Date'].max() - pd.Timedelta(days=recent_days)
    for idx in range(env.start_index, len(df)):
        if df.loc[idx, 'Date'] >= recent_date:
            obs = env.reset(index=idx)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
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
    # 数据加载与增强
    df, extra_feature_cols, extra_max, known_draws = load_data()
    df = add_context_features(df)
    df = augment_data_advanced(df)
    # 将新增特征加入业务特征中
    extra_feature_cols = extra_feature_cols + [
        "draw_std", "draw_sum_lag1", "draw_sum_ma3",
        "consecutive_count", "min_diff", "max_diff", "mean_diff",
        "weekday", "weekofyear", "month", "quarter", "is_holiday", "is_workday"
    ]
    extra_feature_cols += [f"hot_count_{num}" for num in range(1, 81)]
    extra_max = df[extra_feature_cols].max()

    # 输出目录（沿用之前的路径）
    dst_dir = "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/20250411/"
    os.makedirs(dst_dir, exist_ok=True)
    TOTAL_TIMESTEPS = 1000000
    TIMESTEPS_PER_UPDATE = 1000

    # 使用 LSTM 特征提取器（也可替换为 Transformer）
    policy_kwargs = {
        "features_extractor_class": LSTMFeatureExtractor,
        "features_extractor_kwargs": {
            "seq_len": 10,
            "hidden_dim": 64,
            "features_dim": 128
        }
    }

    # 构建三个环境用于 QRDQN、PPO、A2C 模型训练
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

    # 模型评估：基于多窗口近期表现评估平均命中数
    env_eval = LotterySequentialEnv(df, extra_feature_cols, extra_max)
    days_list = [1, 3, 5, 7, 14, 21, 30]
    qrdqn_score = evaluate_model_multiple(env_eval, model_qrdqn, df, days_list)
    ppo_score = evaluate_model_multiple(env_eval, model_ppo, df, days_list)
    a2c_score = evaluate_model_multiple(env_eval, model_a2c, df, days_list)
    logging.info(f"多窗口近期表现 -> QRDQN: {qrdqn_score:.2f}, PPO: {ppo_score:.2f}, A2C: {a2c_score:.2f}")
    total_score = qrdqn_score + ppo_score + a2c_score
    w_qrdqn = qrdqn_score / total_score if total_score > 0 else 0.33
    w_ppo = ppo_score / total_score if total_score > 0 else 0.33
    w_a2c = a2c_score / total_score if total_score > 0 else 0.33
    logging.info(f"动态权重 -> QRDQN: {w_qrdqn:.2f}, PPO: {w_ppo:.2f}, A2C: {w_a2c:.2f}")

    # 预测未来日期（最新日期的下一天）的开奖号码
    last_date = df['Date'].max()
    future_date = last_date + pd.Timedelta(days=1)
    env_pred = LotterySequentialEnv(df, extra_feature_cols, extra_max)

    pred_qrdqn = simulate_episode_for_prediction(env_pred, model_qrdqn, future_date)
    pred_ppo = simulate_episode_for_prediction(env_pred, model_ppo, future_date)
    pred_a2c = simulate_episode_for_prediction(env_pred, model_a2c, future_date)

    def save_prediction(pred, model_name):
        pred_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": pred[j - 1] for j in range(1, 21)}}])
        csv_path = os.path.join(dst_dir, f"{model_name}_v20250410_prediction_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")
        pred_df.to_csv(csv_path, index=False)
        logging.info(f"{model_name} 预测结果已保存至 {csv_path}")

    save_prediction(pred_qrdqn, "qrdqn")
    save_prediction(pred_ppo, "ppo")
    save_prediction(pred_a2c, "a2c")

    # 模型集成：动态加权融合三个模型的预测结果
    ensemble_counter = Counter()
    for num in pred_qrdqn:
        ensemble_counter[num] += w_qrdqn
    for num in pred_ppo:
        ensemble_counter[num] += w_ppo
    for num in pred_a2c:
        ensemble_counter[num] += w_a2c
    final_pred = [num for num, weight in sorted(ensemble_counter.items(), key=lambda x: x[1], reverse=True)]
    if len(final_pred) < 20:
        missing = sorted(set(range(1, 81)) - set(final_pred))
        final_pred.extend(missing)
    final_pred = final_pred[:20]
    logging.info(f"预测 {future_date.date()} 的号码: {final_pred}")

    pred_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": final_pred[j - 1] for j in range(1, 21)}}])
    pred_csv_path = os.path.join(dst_dir, f"RL_v20250411_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.md")
    pred_df.to_csv(pred_csv_path, index=False)
    logging.info(f"集成预测结果已保存至 {pred_csv_path}")

    # 绘制最近一个月每日预测命中趋势图
    plot_hit_trend(df, [model_qrdqn, model_ppo, model_a2c], env_eval,
                   extra_feature_cols, extra_max, dst_dir, TOTAL_TIMESTEPS)

    if has_optuna:
        logging.info("开始使用 Optuna 进行超参数调优...")
        # TODO: 定义并调用 Optuna objective 函数进行调优

if __name__ == "__main__":
    main()
