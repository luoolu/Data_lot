#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/27/25
# @Author  : luoolu
# @Github  : https://luoolu.github.io
# @Software: PyCharm
# @File    : ReinforecementLearningPredict.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
顶级科学家级优化版强化学习预测 KL8 彩票数据（逐步决策版本）
目标：通过改进特征、奖励设计、延长训练步数和多模型集成，
     尽可能提高模型在验证集上的预测（命中）表现

主要改进：
1. 将一次性决策改为逐步决策（顺序选择20个号码），使奖励更细化
2. 引入额外 EMA 特征，丰富历史信息表示
3. 延长训练步数（示例中采用 50,000 步，可根据实际情况调整）
4. 采用适用于离散动作的多种算法：A2C、PPO、DQN
5. 多模型集成采用多数投票策略，对各模型独立预测结果进行集成，
   并在号码数量不足时自动补全，避免索引越界
"""

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
data_path = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/data/kl8/kl8_2025-03-31.csv"
df = pd.read_csv(data_path, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print(f"Loaded data: {len(df)} days from {df['Date'].min().date()} to {df['Date'].max().date()}.")

# 整理历史开奖数据：日期 -> 当天开奖号码集合
known_draws = {
    row['Date']: set(row[f'k{j:02d}'] for j in range(1, 21))
    for _, row in df.iterrows()
}

# 预测结果保存目录
dst_dir = "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/20250401/"
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# ----------------------------
# 2. 定义逐步决策的强化学习环境（离散动作版）
# ----------------------------
class LotterySequentialEnv(gym.Env):
    def __init__(self, data_df):
        super(LotterySequentialEnv, self).__init__()
        self.data = data_df.reset_index(drop=True)
        # 利用数据中的开奖信息构造已知开奖字典
        self.known_draws = {
            row['Date']: set(row[f'k{j:02d}'] for j in range(1, 21))
            for _, row in self.data.iterrows()
        }
        self.dates = list(self.data['Date'])
        # 为支持最长 180 天窗口特征，起始索引设为 180
        self.start_index = 180
        # 状态构成：
        #   - 基础特征：星期几 one-hot (7) + 月份 one-hot (12) + 周期性编码 (4) = 23 维
        #   - 多窗口统计特征：6个窗口，每个窗口 80 (频率) + 4 (统计量) = 84，每个窗口共 84 维，共 504 维
        #   - 额外 EMA 特征：80 维
        #   - 已选号码掩码：80 维
        #   - 当前选择进度：1 维（当前步数/20）
        # 总计：23 + 504 + 80 + 80 + 1 = 688 维
        self.obs_dim = 23 + 504 + 80 + 80 + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        # 动作空间：离散动作，共80个选项（代表号码1~80）
        self.action_space = gym.spaces.Discrete(80)
        self.max_steps = 20  # 一局选择20个号码
        self.current_step = 0
        self.selected_numbers = []  # 存储已选号码（1-indexed）
        self.current_date = None
        self.actual_set = None

    def _compute_window_features(self, current_date, days):
        freq = np.zeros(80, dtype=np.float32)
        values = []
        for n in range(1, days + 1):
            day = current_date - timedelta(days=n)
            if day in self.known_draws:
                draws = self.known_draws[day]
                for num in draws:
                    freq[num - 1] += 1
                values.extend(draws)
        freq /= days
        if values:
            mean_val = np.mean(values) / 80.0
            min_val = min(values) / 80.0
            max_val = max(values) / 80.0
            var_val = np.var(values) / 80.0
        else:
            mean_val = min_val = max_val = var_val = 0.0
        stats = np.array([mean_val, min_val, max_val, var_val], dtype=np.float32)
        return freq, stats

    def _compute_ema_features(self, current_date, span=30):
        # 指数加权平均特征：对最近 span 天中每一天的开奖号码给予衰减权重
        weights = np.exp(-np.arange(1, span + 1) / span)
        weights = weights / np.sum(weights)
        ema = np.zeros(80, dtype=np.float32)
        for n in range(1, span + 1):
            day = current_date - timedelta(days=n)
            if day in self.known_draws:
                draws = self.known_draws[day]
                for num in draws:
                    ema[num - 1] += weights[n - 1]
        return ema

    def _get_observation(self, current_date):
        # 基础特征：星期几和月份 one-hot 以及周期性编码
        dow = current_date.weekday()  # 0~6
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
        base_features = np.concatenate([dow_onehot, month_onehot, cyclical])  # 23 维

        # 多窗口统计特征
        win_sizes = [7, 14, 30, 60, 90, 180]
        window_feats = []
        for w in win_sizes:
            freq, stats = self._compute_window_features(current_date, w)
            window_feats.append(freq)    # 80 维
            window_feats.append(stats)   # 4 维
        window_features = np.concatenate(window_feats)  # 84 * 6 = 504 维

        # 额外 EMA 特征（30天）
        ema_feature = self._compute_ema_features(current_date, span=30)  # 80 维

        # 合并特征
        features = np.concatenate([base_features, window_features, ema_feature])  # 23 + 504 + 80 = 607 维

        # 已选号码掩码：80 维，1 表示已选
        chosen_mask = np.zeros(80, dtype=np.float32)
        for num in self.selected_numbers:
            chosen_mask[num - 1] = 1.0

        # 当前选择进度：1 维（归一化至 [0,1]）
        progress = np.array([self.current_step / self.max_steps], dtype=np.float32)

        observation = np.concatenate([features, chosen_mask, progress])  # 总共 607 + 80 + 1 = 688 维
        return observation.astype(np.float32)

    def reset(self, index=None):
        self.current_step = 0
        self.selected_numbers = []
        if index is not None:
            self.current_date = self.data.loc[index, 'Date']
        else:
            idx = np.random.randint(self.start_index, len(self.data))
            self.current_date = self.data.loc[idx, 'Date']
        self.actual_set = self.known_draws.get(self.current_date, set())
        return self._get_observation(self.current_date)

    def step(self, action):
        # 检查动作是否有效（未重复选择）
        if (action + 1) in self.selected_numbers:
            # 重复选择则给予负奖励
            reward = -0.5
        else:
            self.selected_numbers.append(action + 1)
            reward = 0.0
        self.current_step += 1
        done = (self.current_step == self.max_steps)
        if done:
            predicted_set = set(self.selected_numbers)
            # 最终奖励：命中个数（最大20）
            hits = len(predicted_set & self.actual_set) if self.actual_set is not None else 0
            reward += hits
            info = {"predicted_set": predicted_set, "actual_set": self.actual_set, "hits": hits}
        else:
            info = {}
        return self._get_observation(self.current_date), reward, done, info

# ----------------------------
# 3. 训练多种强化学习模型
# ----------------------------
# 训练步数（根据实际情况调大）
TOTAL_TIMESTEPS = 10000

# 策略网络参数
policy_kwargs = {
    "net_arch": [256, 128, 64]
}

# 导入稳定基线库中的模型（DQN 适用于离散动作空间）
from stable_baselines3 import A2C, PPO, DQN

# 分别创建环境实例
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
    # 从 start_index 开始评估，确保有足够的历史数据
    for idx in range(env.start_index, len(df_eval)):
        obs = env.reset(index=idx)
        done = False
        # 执行一局，累计奖励（最终奖励中包含命中数）
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
# 5. 多模型集成预测——预测数据集外接下来一天的数据
# ----------------------------
def simulate_episode_for_prediction(env, model, future_date):
    env.reset()  # 随机 reset 后覆盖日期
    env.current_date = future_date  # 强制使用预测日期
    # 实际开奖数据未知时，此处 actual_set 保持为 None
    env.actual_set = None
    done = False
    while not done:
        obs = env._get_observation(env.current_date)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    return set(env.selected_numbers)

# 预测日期为最后一期数据之后的下一天
last_date = df['Date'].max()
future_date = last_date + pd.Timedelta(days=1)
env_pred = LotterySequentialEnv(df)

# 分别用各模型模拟一局预测
pred_a2c = simulate_episode_for_prediction(env_pred, model_a2c, future_date)
pred_ppo = simulate_episode_for_prediction(env_pred, model_ppo, future_date)
pred_dqn = simulate_episode_for_prediction(env_pred, model_dqn, future_date)

# ----------------------------
# 保存各模型预测结果
# ----------------------------
pred_a2c_sorted = sorted(list(pred_a2c))
pred_ppo_sorted = sorted(list(pred_ppo))
pred_dqn_sorted = sorted(list(pred_dqn))

pred_a2c_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": pred_a2c_sorted[j-1] for j in range(1, 21)}}])
pred_a2c_csv_path = os.path.join(dst_dir, f"a2c_prediction_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")
pred_a2c_df.to_csv(pred_a2c_csv_path, index=False)
print(f"A2C prediction saved to {pred_a2c_csv_path}")

pred_ppo_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": pred_ppo_sorted[j-1] for j in range(1, 21)}}])
pred_ppo_csv_path = os.path.join(dst_dir, f"ppo_prediction_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")
pred_ppo_df.to_csv(pred_ppo_csv_path, index=False)
print(f"PPO prediction saved to {pred_ppo_csv_path}")

pred_dqn_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": pred_dqn_sorted[j-1] for j in range(1, 21)}}])
pred_dqn_csv_path = os.path.join(dst_dir, f"dqn_prediction_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")
pred_dqn_df.to_csv(pred_dqn_csv_path, index=False)
print(f"DQN prediction saved to {pred_dqn_csv_path}")

# 多模型集成：采用多数投票策略
counter = Counter(list(pred_a2c) + list(pred_ppo) + list(pred_dqn))
ensemble_numbers = [num for num, cnt in counter.items() if cnt >= 2]
if len(ensemble_numbers) < 20:
    # 尝试补充：先将未入多数投票的号码补上
    remaining = [num for num in (pred_a2c | pred_ppo | pred_dqn) if num not in ensemble_numbers]
    remaining_sorted = sorted(remaining, key=lambda x: counter[x], reverse=True)
    ensemble_numbers.extend(remaining_sorted)
if len(ensemble_numbers) < 20:
    # 若仍不足，补充全范围内未选号码
    all_numbers = set(range(1, 81))
    missing = list(all_numbers - set(ensemble_numbers))
    missing_sorted = sorted(missing)
    ensemble_numbers.extend(missing_sorted)
final_pred = sorted(ensemble_numbers[:20])
print(f"预测 {future_date.date()} 的号码: {final_pred}")

# 保存集成预测结果到 CSV
pred_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": final_pred[j-1] for j in range(1, 21)}}])
pred_csv_path = os.path.join(dst_dir, f"ReinforecementLearningPredict_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")
pred_df.to_csv(pred_csv_path, index=False)
print(f"集成预测结果已保存至 {pred_csv_path}")

# ----------------------------
# 6. 绘制训练集上集成预测的命中趋势图
# ----------------------------
def simulate_episode_for_date(date, df, model):
    temp_env = LotterySequentialEnv(df)
    temp_env.reset(index=df.index[df['Date'] == date][0])
    temp_env.current_date = date
    done = False
    while not done:
        obs = temp_env._get_observation(date)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = temp_env.step(action)
    return set(temp_env.selected_numbers)

def ensemble_prediction_for_date(date, df, models):
    predictions = []
    for model in models:
        pred = simulate_episode_for_date(date, df, model)
        predictions.append(pred)
    counter = Counter(sum([list(pred) for pred in predictions], []))
    ensemble_nums = [num for num, cnt in counter.items() if cnt >= 2]
    if len(ensemble_nums) < 20:
        remaining = [num for num in (predictions[0] | predictions[1] | predictions[2]) if num not in ensemble_nums]
        remaining_sorted = sorted(remaining, key=lambda x: counter[x], reverse=True)
        ensemble_nums.extend(remaining_sorted)
    if len(ensemble_nums) < 20:
        all_numbers = set(range(1, 81))
        missing = list(all_numbers - set(ensemble_nums))
        missing_sorted = sorted(missing)
        ensemble_nums.extend(missing_sorted)
    ensemble_nums = sorted(ensemble_nums[:20])
    return ensemble_nums

hit_counts = []
dates_list = []
for idx in range(env_eval.start_index, len(df)):
    date = df.loc[idx, 'Date']
    ensemble_pred = ensemble_prediction_for_date(date, df, [model_a2c, model_ppo, model_dqn])
    actual_set = env_eval.known_draws.get(date, set())
    hits = len(set(ensemble_pred) & actual_set)
    hit_counts.append(hits)
    dates_list.append(date)

plt.figure(figsize=(10, 5))
plt.plot(dates_list, hit_counts, marker='o', label="Hits per day")
avg_hits = np.mean(hit_counts)
plt.axhline(y=avg_hits, color='r', linestyle='--', label=f"Avg hits = {avg_hits:.2f}")
plt.xlabel("Date")
plt.ylabel("Hits (out of 20)")
plt.title("Training Set Ensemble Prediction Hit Trend")
plt.legend()
plt.tight_layout()
hit_trend_path = os.path.join(dst_dir, "hit_trend.png")
plt.savefig(hit_trend_path)
print(f"命中趋势图已保存至 {hit_trend_path}")





