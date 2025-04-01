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
改进版强化学习彩票预测系统
新增内容：
1. 每个模型保存预测下一天的结果
2. 集成预测保存为独立文件
3. 保持数据读取与前一致
"""

import os
import random
from datetime import timedelta
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
from stable_baselines3 import A2C, PPO, DQN

# 固定随机种子
random.seed(0)
np.random.seed(0)

# 数据读取
DATA_PATH = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/data/kl8/kl8_2025-03-31.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

known_draws = {
    row['Date']: set(row[f'k{j:02d}'] for j in range(1, 21))
    for _, row in df.iterrows()
}

# 输出路径
OUT_DIR = "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/20250401v2/"
os.makedirs(OUT_DIR, exist_ok=True)

class LotterySequentialEnv(gym.Env):
    def __init__(self, data_df):
        super(LotterySequentialEnv, self).__init__()
        self.data = data_df.reset_index(drop=True)
        self.known_draws = {
            row['Date']: set(row[f'k{j:02d}'] for j in range(1, 21))
            for _, row in self.data.iterrows()
        }
        self.dates = list(self.data['Date'])
        self.start_index = 180
        self.obs_dim = 23 + 504 + 80 + 80 + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(80)
        self.max_steps = 20
        self.current_step = 0
        self.selected_numbers = []
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

        win_sizes = [7, 14, 30, 60, 90, 180]
        window_feats = []
        for w in win_sizes:
            freq, stats = self._compute_window_features(current_date, w)
            window_feats.append(freq)
            window_feats.append(stats)
        window_features = np.concatenate(window_feats)
        ema_feature = self._compute_ema_features(current_date, span=30)
        features = np.concatenate([base_features, window_features, ema_feature])
        chosen_mask = np.zeros(80, dtype=np.float32)
        for num in self.selected_numbers:
            chosen_mask[num - 1] = 1.0
        progress = np.array([self.current_step / self.max_steps], dtype=np.float32)
        observation = np.concatenate([features, chosen_mask, progress])
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
        if (action + 1) in self.selected_numbers:
            reward = -0.5
        else:
            self.selected_numbers.append(action + 1)
            reward = 0.0
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

# 环境和模型配置
TOTAL_TIMESTEPS = 10000
policy_kwargs = {"net_arch": [256, 128, 64]}

env_a2c = LotterySequentialEnv(df)
env_ppo = LotterySequentialEnv(df)
env_dqn = LotterySequentialEnv(df)

model_a2c = A2C("MlpPolicy", env_a2c, policy_kwargs=policy_kwargs, verbose=0, learning_rate=1e-3)
model_a2c.learn(total_timesteps=TOTAL_TIMESTEPS)
model_a2c.save(os.path.join(OUT_DIR, "a2c_model"))

model_ppo = PPO("MlpPolicy", env_ppo, policy_kwargs=policy_kwargs, verbose=0, learning_rate=1e-3)
model_ppo.learn(total_timesteps=TOTAL_TIMESTEPS)
model_ppo.save(os.path.join(OUT_DIR, "ppo_model"))

model_dqn = DQN("MlpPolicy", env_dqn, policy_kwargs=policy_kwargs, verbose=0, learning_rate=1e-3)
model_dqn.learn(total_timesteps=TOTAL_TIMESTEPS)
model_dqn.save(os.path.join(OUT_DIR, "dqn_model"))

# 预测下一天
def simulate_prediction(model, env, future_date):
    env.reset()
    env.current_date = future_date
    env.actual_set = None  # 预测时不提供真实号码
    done = False
    while not done:
        obs = env._get_observation(env.current_date)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    selected = sorted(env.selected_numbers)
    # 如果选取的号码不足20个，则从剩余号码中补充
    if len(selected) < 20:
        missing = sorted(set(range(1, 81)) - set(selected))
        selected.extend(missing[:20 - len(selected)])
    return sorted(selected)

future_date = df['Date'].max() + pd.Timedelta(days=1)
env_pred = LotterySequentialEnv(df)

preds = {
    "a2c": simulate_prediction(model_a2c, env_pred, future_date),
    "ppo": simulate_prediction(model_ppo, env_pred, future_date),
    "dqn": simulate_prediction(model_dqn, env_pred, future_date),
}

# 保存单模型预测结果
for name, numbers in preds.items():
    pred_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": numbers[j-1] for j in range(1, 21)}}])
    pred_df.to_csv(os.path.join(OUT_DIR, f"{name}_predict_{future_date.date()}_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv"), index=False)

# 多模型集成
all_votes = preds["a2c"] + preds["ppo"] + preds["dqn"]
counter = Counter(all_votes)
ensemble = [num for num, cnt in counter.items() if cnt >= 2]

if len(ensemble) < 20:
    remaining = sorted(set(all_votes) - set(ensemble), key=lambda x: counter[x], reverse=True)
    ensemble += remaining
if len(ensemble) < 20:
    missing = sorted(set(range(1, 81)) - set(ensemble))
    ensemble += missing
final_ensemble = sorted(ensemble[:20])

# 保存集成预测结果
ensemble_df = pd.DataFrame([{"Date": future_date.date(), **{f"Pred{j}": final_ensemble[j-1] for j in range(1, 21)}}])
ensemble_df.to_csv(os.path.join(OUT_DIR, f"ensemble_predict_{future_date.date()}_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv"), index=False)

print("预测完成，结果已保存：")
for name in preds:
    print(f"- {name} 模型预测结果保存在 {name}_predict_{future_date.date()}_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")
print(f"- 集成预测结果保存在 ensemble_predict_{future_date.date()}_TOTAL_TIMESTEPS_{TOTAL_TIMESTEPS}.csv")





