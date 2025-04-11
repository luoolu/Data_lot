#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# 自定义 MLP 特征提取器，加入 dropout 防止过拟合
class CustomMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, net_arch=[512, 256, 128], dropout=0.2):
        super(CustomMLP, self).__init__(observation_space, features_dim)
        layers = []
        input_dim = observation_space.shape[0]
        for hidden_size in net_arch:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, features_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, observations):
        return self.mlp(observations)


# ----------------------------
# 1. 数据加载与预处理
# ----------------------------
def load_data():
    data_path = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/kl8/kl8_order_data.csv"
    df = pd.read_csv(data_path, encoding='utf-8', parse_dates=["开奖日期"])
    df.rename(columns={"开奖日期": "Date"}, inplace=True)
    df = df.sort_values("Date").reset_index(drop=True)
    logging.info(f"Loaded data: {len(df)} days from {df['Date'].min().date()} to {df['Date'].max().date()}.")

    # 定义80个业务特征列
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
    # 去除逗号并转换为 float
    for col in extra_feature_cols:
        df[col] = df[col].replace(',', '', regex=True).astype(float)
    df[extra_feature_cols] = df[extra_feature_cols].fillna(0)
    extra_max = df[extra_feature_cols].max()
    known_draws = {
        row['Date']: set(row[f'排好序_{j}'] for j in range(1, 21))
        for _, row in df.iterrows()
    }
    return df, extra_feature_cols, extra_max, known_draws


# 数据增强（此处可以加入更多数据增强策略）
def augment_data(df):
    # 例如，可以考虑添加移动平均、波动率等统计指标作为新特征
    # 这里简单返回原数据
    return df


# ----------------------------
# 2. 定义强化学习环境（含高级奖励函数）
# ----------------------------
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

        # 观察值维度：静态特征 + 动态特征
        static_feature_dim = 854
        dynamic_feature_dim = 81
        self.obs_dim = static_feature_dim + dynamic_feature_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(80)
        self.max_steps = 20

        self.current_step = 0
        self.selected_numbers = []
        self.current_date = None
        self.current_index = None
        self.actual_set = None
        self.static_features = None

        # 设置奖励函数各项权重（经过调优后的参数）
        if reward_weights is None:
            self.reward_weights = {
                "ema": 0.25,
                "cum": 0.20,
                "order": 0.15,
                "shape": 0.15,
                "diversity": 0.10,
                "repeat_penalty": -1.0
            }
        else:
            self.reward_weights = reward_weights

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
        holiday = 1.0 if has_chinese_calendar and chinese_calendar.is_holiday(current_date.date()) else 0.0
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
        chosen_mask = np.zeros(80, dtype=np.float32)
        for num in self.selected_numbers:
            chosen_mask[int(num) - 1] = 1.0
        progress = np.array([self.current_step / self.max_steps], dtype=np.float32)
        observation = np.concatenate([self.static_features, chosen_mask, progress])
        return observation

    def reset(self, index=None):
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

    # —— 优化后的奖励函数设计 ——
    def compute_reward(self, action):
        # 若重复选择，则直接返回重复处罚
        if (action + 1) in self.selected_numbers:
            return self.reward_weights["repeat_penalty"]

        # 计算各项特征（全部归一化处理）
        ema_feature = self._compute_ema_features(span=30)
        cum_freq = self._compute_cumulative_frequency()
        order_feature = self._compute_order_features(window_size=30)
        max_ema = np.max(ema_feature) if np.max(ema_feature) > 0 else 1.0
        norm_ema = ema_feature[action] / max_ema
        norm_cum = cum_freq[action]
        norm_order = order_feature[action]
        # 优化前向奖励：鼓励选择出现频率较低的数字
        reward_ema = self.reward_weights["ema"] * (1 - norm_ema)
        reward_cum = self.reward_weights["cum"] * (1 - norm_cum)
        reward_order = self.reward_weights["order"] * (1 - norm_order)
        # 形状奖励：当累计频率低于均值时给予额外奖励
        expected = 1 / 80
        reward_shape = self.reward_weights["shape"] * (expected - norm_cum)
        # 多样性奖励：随着已选数字增多，适当加权鼓励多样化预测
        diversity_bonus = self.reward_weights["diversity"] * len(set(self.selected_numbers))
        total_reward = reward_ema + reward_cum + reward_order + reward_shape + diversity_bonus

        # 记录选中的数字
        self.selected_numbers.append(action + 1)
        return total_reward

    def step(self, action):
        # 防止重复选择的额外检查
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
            reward += hits  # 额外奖励命中数量
            info = {"predicted_set": predicted_set, "actual_set": self.actual_set, "hits": hits}
        return self._get_observation(), reward, done, info


# 辅助函数：若选号码不足20个，则补全
def complete_prediction(pred_list):
    if len(pred_list) < 20:
        missing = sorted(set(range(1, 81)) - set(pred_list))
        pred_list.extend(missing[:20 - len(pred_list)])
    return pred_list


# ----------------------------
# 3. 模型训练与评估函数（异构模型）
# ----------------------------
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
        # 设置动态学习率参数（可在回调中进一步调整）
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
                obs = env._get_observation()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            if "hits" in info:
                hit_counts.append(info["hits"])
    return np.mean(hit_counts) if hit_counts else 0


# 新增：对多个时间窗口的评估，返回均值
def evaluate_model_multiple(env, model, df, days_list):
    scores = []
    for d in days_list:
        score = evaluate_model_recent(env, model, df, recent_days=d)
        scores.append(score)
    return np.mean(scores)


def simulate_episode_for_prediction(env, model, future_date):
    env.reset()
    env.current_date = future_date
    env.actual_set = None  # 未来没有真实开奖数据
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
    # 多模型投票：至少2票进入集成列表
    ensemble_nums = [num for num, cnt in counter_votes.items() if cnt >= 2]
    if len(ensemble_nums) < 20:
        remaining = [num for num in (predictions[0] | predictions[1] | predictions[2]) if num not in ensemble_nums]
        ensemble_nums.extend(sorted(remaining, key=lambda x: counter_votes[x], reverse=True))
    if len(ensemble_nums) < 20:
        missing = sorted(set(range(1, 81)) - set(ensemble_nums))
        ensemble_nums.extend(missing)
    return sorted(ensemble_nums[:20])


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
    hit_trend_path = os.path.join(dst_dir, f"RL_advanced20250408_hit_trend_TOTAL_TIMESTEPS_{total_timesteps}.png")
    plt.savefig(hit_trend_path)
    logging.info(f"命中趋势图已保存至 {hit_trend_path}")


# ----------------------------
# 4. 主流程：训练、评估、预测与结果保存
# ----------------------------
def main():
    df, extra_feature_cols, extra_max, known_draws = load_data()
    df = augment_data(df)

    dst_dir = "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/20250409/"
    os.makedirs(dst_dir, exist_ok=True)
    TOTAL_TIMESTEPS = 10000000
    TIMESTEPS_PER_UPDATE = 1000
    policy_kwargs = {
        "features_extractor_class": CustomMLP,
        "features_extractor_kwargs": dict(features_dim=128, net_arch=[512, 256, 128], dropout=0.2)
    }

    # 分别创建三个环境实例用于不同模型训练
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

    # 多窗口动态评估，获取各模型在多个时间尺度上的平均表现
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

    # 对未来一天进行预测
    last_date = df['Date'].max()
    future_date = last_date + pd.Timedelta(days=1)
    env_pred = LotterySequentialEnv(df, extra_feature_cols, extra_max)

    pred_qrdqn = simulate_episode_for_prediction(env_pred, model_qrdqn, future_date)
    pred_ppo = simulate_episode_for_prediction(env_pred, model_ppo, future_date)
    pred_a2c = simulate_episode_for_prediction(env_pred, model_a2c, future_date)

    def save_prediction(pred, model_name):
        pred_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": pred[j - 1] for j in range(1, 21)}}])
        csv_path = os.path.join(dst_dir, f"{model_name}_prediction_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")
        pred_df.to_csv(csv_path, index=False)
        logging.info(f"{model_name} 预测结果已保存至 {csv_path}")

    save_prediction(pred_qrdqn, "qrdqn")
    save_prediction(pred_ppo, "ppo")
    save_prediction(pred_a2c, "a2c")

    # 多模型集成加权投票，利用各模型动态权重
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
    pred_csv_path = os.path.join(dst_dir, f"RL_advanced_20250408_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.md")
    pred_df.to_csv(pred_csv_path, index=False)
    logging.info(f"集成预测结果已保存至 {pred_csv_path}")

    # 绘制最近一个月的命中趋势图
    plot_hit_trend(df, [model_qrdqn, model_ppo, model_a2c], env_eval,
                   extra_feature_cols, extra_max, dst_dir, TOTAL_TIMESTEPS)


if __name__ == "__main__":
    main()
