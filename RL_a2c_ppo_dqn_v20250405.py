#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/27/25
# @Author  : luoolu
# @Github  : https://luoolu.github.io
# @Software: PyCharm
# @File    : ReinforecementLearningPredict.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import gym

# 固定随机种子，确保结果可复现
random.seed(0)
np.random.seed(0)

# ----------------------------
# 1. 数据读取与初步处理（确保数据文件路径正确）
# ----------------------------
data_path = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/kl8/kl8_order_data.csv"
df = pd.read_csv(data_path, encoding='utf-8', parse_dates=["开奖日期"])
df.rename(columns={"开奖日期": "Date"}, inplace=True)
df = df.sort_values("Date").reset_index(drop=True)
print(f"Loaded data: {len(df)} days from {df['Date'].min().date()} to {df['Date'].max().date()}.")

# 构建历史开奖数据字典：日期 -> 当天开奖号码集合（排好序的号码）
known_draws = {
    row['Date']: set(row[f'排好序_{j}'] for j in range(1, 21))
    for _, row in df.iterrows()
}

# 预测结果保存目录
dst_dir = "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/20250405/"
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)


# ----------------------------
# 2. 定义加速版强化学习环境（预计算静态特征 + 向量化运算）
# ----------------------------
class LotterySequentialEnv(gym.Env):
    def __init__(self, data_df):
        super(LotterySequentialEnv, self).__init__()
        self.data = data_df.reset_index(drop=True)
        self.known_draws = {
            row['Date']: set(row[f'排好序_{j}'] for j in range(1, 21))
            for _, row in self.data.iterrows()
        }
        self.dates = list(self.data['Date'])
        self.start_index = 180  # 确保有足够历史数据

        # 预先构造“排好序”和“出球顺序”列名列表
        self.sorted_cols = [f'排好序_{j}' for j in range(1, 21)]
        self.order_cols = [f'出球顺序_{j}' for j in range(1, 21)]

        # 观察空间维度（静态特征 + 已选号码掩码 + 选择进度）
        self.static_feat_dim = 23 + 504 + 80 + 80 + 80  # 767 维
        self.obs_dim = self.static_feat_dim + 80 + 1  # 848 维
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        # 动作空间：80 个离散动作，对应号码 1~80
        self.action_space = gym.spaces.Discrete(80)
        self.max_steps = 20
        self.current_step = 0
        self.selected_numbers = []  # 已选号码（1-indexed）
        self.current_date = None
        self.current_index = None
        self.actual_set = None

        # 静态特征缓存（在 reset 时计算）
        self.static_features = None

    def _get_base_features(self, current_date):
        dow = current_date.weekday()
        dow_onehot = np.zeros(7, dtype=np.float32)
        dow_onehot[dow] = 1.0
        month = current_date.month
        month_onehot = np.zeros(12, dtype=np.float32)
        month_onehot[month - 1] = 1.0
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)
        cyclical = np.array([dow_sin, dow_cos, month_sin, month_cos], dtype=np.float32)
        base_features = np.concatenate([dow_onehot, month_onehot, cyclical])
        return base_features  # 23 维

    def _compute_window_features(self, window_size):
        start_idx = max(0, self.current_index - window_size)
        subset = self.data.iloc[start_idx:self.current_index]
        n_rows = len(subset)
        if n_rows == 0:
            freq = np.zeros(80, dtype=np.float32)
            stats = np.zeros(4, dtype=np.float32)
        else:
            nums_array = subset[self.sorted_cols].values.astype(np.int32)  # shape: (n_rows, 20)
            # 频率统计（利用 np.bincount）
            freq = np.bincount(nums_array.flatten()-1, minlength=80).astype(np.float32) / n_rows
            values = nums_array.flatten().astype(np.float32)
            stats = np.array([np.mean(values), np.min(values), np.max(values), np.var(values)], dtype=np.float32) / 80.0
        return freq, stats  # (80, ) 和 (4, )

    def _compute_ema_features(self, span=30):
        start_idx = max(0, self.current_index - span)
        subset = self.data.iloc[start_idx:self.current_index]
        actual_span = len(subset)
        if actual_span == 0:
            return np.zeros(80, dtype=np.float32)
        # 权重计算
        weights = np.exp(-np.arange(1, actual_span + 1) / span)
        weights = weights / np.sum(weights)
        nums_array = subset[self.sorted_cols].values.astype(np.int32)  # shape: (actual_span, 20)
        # 利用 np.add.at 实现加权累积（避免 Python 循环）
        ema = np.zeros(80, dtype=np.float32)
        np.add.at(ema, nums_array.flatten()-1, np.repeat(weights, 20))
        return ema

    def _compute_cumulative_frequency(self):
        if self.current_index == 0:
            return np.zeros(80, dtype=np.float32)
        subset = self.data.iloc[:self.current_index]
        n_rows = len(subset)
        nums_array = subset[self.sorted_cols].values.astype(np.int32)
        freq = np.bincount(nums_array.flatten()-1, minlength=80).astype(np.float32) / n_rows
        return freq

    def _compute_order_features(self, window_size=30):
        start_idx = max(0, self.current_index - window_size)
        subset = self.data.iloc[start_idx:self.current_index]
        n_rows = len(subset)
        if n_rows == 0:
            return np.zeros(80, dtype=np.float32)
        nums_array = subset[self.order_cols].values.astype(np.int32)  # shape: (n_rows, 20)
        # 每一期的权重：越早出球权重越大，1/j
        col_weights = 1.0 / np.arange(1, 21, dtype=np.float32)  # shape: (20,)
        order_sum = np.zeros(80, dtype=np.float32)
        np.add.at(order_sum, nums_array.flatten()-1, np.tile(col_weights, n_rows))
        order_freq = order_sum / n_rows
        return order_freq

    def _precompute_static_features(self):
        # 基础日期特征
        base_features = self._get_base_features(self.current_date)  # 23 维

        # 多窗口统计特征（6个窗口，每个窗口 84 维，共504 维）
        win_sizes = [7, 14, 30, 60, 90, 180]
        window_feats = []
        for w in win_sizes:
            freq, stats = self._compute_window_features(w)
            window_feats.append(freq)  # 80 维
            window_feats.append(stats)  # 4 维
        window_features = np.concatenate(window_feats)  # 504 维

        # 额外 EMA 特征（80维）
        ema_feature = self._compute_ema_features(span=30)
        # 累计频率特征（80维）
        cum_freq = self._compute_cumulative_frequency()
        # 出球顺序特征（80维）
        order_features = self._compute_order_features(window_size=30)

        # 合并所有静态特征，共 23 + 504 + 80 + 80 + 80 = 767 维
        static_features = np.concatenate([base_features, window_features, ema_feature, cum_freq, order_features])
        return static_features.astype(np.float32)

    def _get_observation(self, current_date):
        # 若已缓存静态特征，则直接调用
        if self.static_features is None:
            self.static_features = self._precompute_static_features()
        # 已选号码掩码（80维）
        chosen_mask = np.zeros(80, dtype=np.float32)
        for num in self.selected_numbers:
            chosen_mask[int(num)-1] = 1.0
        # 当前选择进度（1维，归一化）
        progress = np.array([self.current_step / self.max_steps], dtype=np.float32)
        # 组合最终观测向量：静态特征 + 掩码 + 进度，共848维
        observation = np.concatenate([self.static_features, chosen_mask, progress])
        return observation.astype(np.float32)

    def reset(self, index=None):
        self.current_step = 0
        self.selected_numbers = []
        if index is not None:
            self.current_index = index
        else:
            self.current_index = np.random.randint(self.start_index, len(self.data))
        self.current_date = self.data.loc[self.current_index, 'Date']
        self.actual_set = self.known_draws.get(self.current_date, set())
        # 预计算当前局不变的静态特征
        self.static_features = self._precompute_static_features()
        return self._get_observation(self.current_date)

    def step(self, action):
        if (action + 1) in self.selected_numbers:
            reward = -1.0  # 重复选择惩罚
        else:
            self.selected_numbers.append(action + 1)
            # 奖励与所选号码对应的 EMA 值相关
            ema_feature = self._compute_ema_features(span=30)
            reward = 0.2 * ema_feature[action]
        self.current_step += 1
        done = (self.current_step == self.max_steps)
        if done:
            predicted_set = set(self.selected_numbers)
            hits = len(predicted_set & self.actual_set) if self.actual_set is not None else 0
            reward += hits
            info = {"predicted_set": predicted_set, "actual_set": self.actual_set, "hits": hits}
        else:
            info = {}
        return self._get_observation(self.current_date), reward, done, info


# ----------------------------
# 辅助函数：补全预测号码（不足20个时补全）
# ----------------------------
def complete_prediction(pred_list):
    if len(pred_list) < 20:
        missing = sorted(set(range(1, 81)) - set(pred_list))
        pred_list.extend(missing[:20 - len(pred_list)])
    return pred_list


# ----------------------------
# 3. 训练强化学习模型（使用 stable-baselines3）
# ----------------------------
TOTAL_TIMESTEPS = 10000  # 根据需要调整训练步数

policy_kwargs = {"net_arch": [256, 128, 64]}

from stable_baselines3 import A2C, PPO, DQN

# 分别创建环境实例（加速版）
env_a2c = LotterySequentialEnv(df)
env_ppo = LotterySequentialEnv(df)
env_dqn = LotterySequentialEnv(df)

print("训练 A2C 模型...")
model_a2c = A2C("MlpPolicy", env_a2c, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-3)
model_a2c.learn(total_timesteps=TOTAL_TIMESTEPS)
model_a2c.save(os.path.join(dst_dir, "a2c_kl8_model"))

print("训练 PPO 模型...")
model_ppo = PPO("MlpPolicy", env_ppo, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-3)
model_ppo.learn(total_timesteps=TOTAL_TIMESTEPS)
model_ppo.save(os.path.join(dst_dir, "ppo_kl8_model"))

print("训练 DQN 模型...")
model_dqn = DQN("MlpPolicy", env_dqn, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-3)
model_dqn.learn(total_timesteps=TOTAL_TIMESTEPS)
model_dqn.save(os.path.join(dst_dir, "dqn_kl8_model"))

# ----------------------------
# 4. 在验证集上评估各模型表现（平均命中数）
# ----------------------------
def evaluate_model(env, model, df_eval):
    hit_counts = []
    for idx in range(env.start_index, len(df_eval)):
        obs = env.reset(index=idx)
        done = False
        while not done:
            obs = env._get_observation(env.current_date)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        if "hits" in info:
            hit_counts.append(info["hits"])
    return np.mean(hit_counts)

env_eval = LotterySequentialEnv(df)
a2c_avg_hits = evaluate_model(env_eval, model_a2c, df)
ppo_avg_hits = evaluate_model(env_eval, model_ppo, df)
dqn_avg_hits = evaluate_model(env_eval, model_dqn, df)
print(f"A2C avg hits: {a2c_avg_hits:.2f}, PPO avg hits: {ppo_avg_hits:.2f}, DQN avg hits: {dqn_avg_hits:.2f}")

total_hits = a2c_avg_hits + ppo_avg_hits + dqn_avg_hits
w_a2c = a2c_avg_hits / total_hits if total_hits > 0 else 0.33
w_ppo = ppo_avg_hits / total_hits if total_hits > 0 else 0.33
w_dqn = dqn_avg_hits / total_hits if total_hits > 0 else 0.33
print(f"Dynamic weights -> A2C: {w_a2c:.2f}, PPO: {w_ppo:.2f}, DQN: {w_dqn:.2f}")

# ----------------------------
# 5. 多模型集成预测——预测下一期数据（未知开奖）
# ----------------------------
def simulate_episode_for_prediction(env, model, future_date):
    env.reset()
    env.current_date = future_date
    env.actual_set = None
    while env.current_step < env.max_steps:
        obs = env._get_observation(env.current_date)
        action, _ = model.predict(obs, deterministic=True)
        if (action + 1) in env.selected_numbers:
            valid_actions = list(set(range(1, 81)) - set(env.selected_numbers))
            if valid_actions:
                action = random.choice(valid_actions) - 1
        obs, reward, done, info = env.step(action)
    return complete_prediction(env.selected_numbers)

last_date = df['Date'].max()
future_date = last_date + pd.Timedelta(days=1)
env_pred = LotterySequentialEnv(df)

pred_a2c = simulate_episode_for_prediction(env_pred, model_a2c, future_date)
pred_ppo = simulate_episode_for_prediction(env_pred, model_ppo, future_date)
pred_dqn = simulate_episode_for_prediction(env_pred, model_dqn, future_date)

# 保存各模型预测结果
pred_a2c_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": pred_a2c[j - 1] for j in range(1, 21)}}])
pred_a2c_csv_path = os.path.join(dst_dir, f"a2c_prediction_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")
pred_a2c_df.to_csv(pred_a2c_csv_path, index=False)
print(f"A2C prediction saved to {pred_a2c_csv_path}")

pred_ppo_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": pred_ppo[j - 1] for j in range(1, 21)}}])
pred_ppo_csv_path = os.path.join(dst_dir, f"ppo_prediction_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")
pred_ppo_df.to_csv(pred_ppo_csv_path, index=False)
print(f"PPO prediction saved to {pred_ppo_csv_path}")

pred_dqn_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": pred_dqn[j - 1] for j in range(1, 21)}}])
pred_dqn_csv_path = os.path.join(dst_dir, f"dqn_prediction_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")
pred_dqn_df.to_csv(pred_dqn_csv_path, index=False)
print(f"DQN prediction saved to {pred_dqn_csv_path}")

# 多模型集成：利用各模型验证集上的表现动态加权
ensemble_counter = Counter()
for num in pred_a2c:
    ensemble_counter[num] += w_a2c
for num in pred_ppo:
    ensemble_counter[num] += w_ppo
for num in pred_dqn:
    ensemble_counter[num] += w_dqn

final_pred = [num for num, weight in sorted(ensemble_counter.items(), key=lambda x: x[1], reverse=True)]
if len(final_pred) < 20:
    missing = sorted(set(range(1, 81)) - set(final_pred))
    final_pred.extend(missing)
final_pred = final_pred[:20]
print(f"预测 {future_date.date()} 的号码: {final_pred}")

pred_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": final_pred[j - 1] for j in range(1, 21)}}])
pred_csv_path = os.path.join(dst_dir, f"ReinforcementLearningPredict_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")
pred_df.to_csv(pred_csv_path, index=False)
print(f"集成预测结果已保存至 {pred_csv_path}")

# ----------------------------
# 6. 绘制最近一个月训练集上集成预测命中趋势图
# ----------------------------
def simulate_episode_for_date(date, df, model):
    temp_env = LotterySequentialEnv(df)
    idx = df.index[df['Date'] == date][0]
    temp_env.reset(index=idx)
    temp_env.current_date = date
    while temp_env.current_step < temp_env.max_steps:
        obs = temp_env._get_observation(date)
        action, _ = model.predict(obs, deterministic=True)
        temp_env.step(action)
    return set(temp_env.selected_numbers)

def ensemble_prediction_for_date(date, df, models):
    predictions = []
    for model in models:
        pred = simulate_episode_for_date(date, df, model)
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

hit_counts = []
dates_list = []
for idx in range(env_eval.start_index, len(df)):
    date = df.loc[idx, 'Date']
    ensemble_pred = ensemble_prediction_for_date(date, df, [model_a2c, model_ppo, model_dqn])
    actual_set = env_eval.known_draws.get(date, set())
    hits = len(set(ensemble_pred) & actual_set)
    hit_counts.append(hits)
    dates_list.append(date)

last_date_in_data = df['Date'].max()
one_month_ago = last_date_in_data - pd.Timedelta(days=30)
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
hit_trend_path = os.path.join(dst_dir, "hit_trend_last_month.png")
plt.savefig(hit_trend_path)
print(f"命中趋势图已保存至 {hit_trend_path}")

"""
说明
预计算静态特征
在 reset() 中调用 self._precompute_static_features() 后，将返回的 767 维静态特征存入 self.static_features，后续每次 step() 调用时，仅更新 80 维的已选号码掩码和 1 维进度。

向量化操作
例如，在 _compute_window_features() 中，直接利用 DataFrame 切片和 np.bincount 快速统计所有号码；在 _compute_ema_features() 和 _compute_order_features() 中，则利用 np.add.at 和 np.tile 处理加权累积问题。

这种改造不仅减少了重复计算，而且充分发挥了 numpy 向量化计算的优势，在你的硬件平台上能够显著加速整个流程。
"""







