#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/28/25
# @Author  : luoolu
# @Github  : https://luoolu.github.io
# @Software: PyCharm
# @File    : kl8_RL_deepseekv30324.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
顶级科学家级优化版强化学习预测 KL8 彩票数据 (V2)
目标：尽可能准确预测数据集外接下来一天的数据

主要改进 (V2):
1. 引入多种强化学习算法：A2C、PPO、SAC、TD3（采用连续动作空间设计）
2. 利用更多历史数据特征：
    - 多窗口统计特征（7, 14, 30, 60, 90, 180 天）：频率, 均值, 最小值, 最大值, 方差, 标准差
    - 周期性特征 (日/月)
    - 热/冷号特征：每个号码最后一次出现距离现在的天数
    - 差分频率特征：不同窗口频率的差异
3. 使用深层网络（本例采用默认 MLP，可扩展为 Transformer/LSTM 等） - 略微加深
4. 特征标准化：使用 StandardScaler 对特征进行标准化处理
5. Train/Validation Split: 使用训练集训练模型，使用验证集评估和计算集成权重
6. 多模型集成：基于验证集表现计算动态加权集成多个模型预测
7. 增加训练步数
8. **优化加速**：利用 SubprocVecEnv 进行向量化环境训练，以及 GPU 加速 (device='cuda')
"""

import os
import time
import random
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import StandardScaler

# 设置 PyTorch 加速（需确保 torch 已安装且支持 GPU）
import torch
torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动优化

# Stable Baselines3 相关导入（确保安装了支持 GPU 的版本）
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.td3.policies import MlpPolicy as TD3MlpPolicy

# ----------------------------
# 配置参数
# ----------------------------
DATA_FILE = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/data/kl8/kl8_2025-03-28.csv"
RESULTS_DIR = "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/20250329/"
RANDOM_SEED = 42
VALIDATION_SPLIT_DATE = "2024-01-01"
TRAINING_TIMESTEPS = 5000  # 根据需要调大训练步数以充分利用资源
WINDOW_SIZES = [7, 14, 30, 60, 90, 180]
MAX_COLD_STREAK = 90
NUM_LOTTERY_BALLS = 80
NUM_SELECTED_BALLS = 20

# 使用 GPU 设备（若可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs(RESULTS_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ----------------------------
# 1. 数据读取与初步处理
# ----------------------------
print("--- 1. Loading and Preprocessing Data ---")
df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print(f"Loaded data: {len(df)} days from {df['Date'].min().date()} to {df['Date'].max().date()}.")

# 预计算历史开奖号码字典以提高效率
known_draws = {
    row['Date']: set(int(row[f'k{j:02d}']) for j in range(1, NUM_SELECTED_BALLS + 1))
    for _, row in df.iterrows()
}
print("Pre-calculated known draws.")

# 划分训练/验证集
df_train = df[df['Date'] < pd.to_datetime(VALIDATION_SPLIT_DATE)].copy()
df_val = df[df['Date'] >= pd.to_datetime(VALIDATION_SPLIT_DATE)].copy()

if df_val.empty or len(df_train) < max(WINDOW_SIZES):
    raise ValueError(f"Validation split date {VALIDATION_SPLIT_DATE} results in insufficient data.")

print(f"Training data: {len(df_train)} days (until {df_train['Date'].max().date()})")
print(f"Validation data: {len(df_val)} days (from {df_val['Date'].min().date()})")

# ----------------------------
# 2. 定义强化学习环境（增强版特征）
# ----------------------------
print("--- 2. Defining Enhanced RL Environment ---")

class EnhancedLotteryEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, historical_df, data_subset_indices, known_draws_dict, scaler=None, is_training=True):
        super(EnhancedLotteryEnv, self).__init__()
        self.historical_df = historical_df
        self.data_subset_indices = data_subset_indices
        self.known_draws = known_draws_dict
        self.scaler = scaler
        self.is_training = is_training

        self.global_min_feature_index = max(WINDOW_SIZES)
        self.valid_subset_indices = [idx for idx in self.data_subset_indices if idx >= self.global_min_feature_index]
        if not self.valid_subset_indices:
            raise ValueError("No valid indices for required history windows.")
        self.current_df_index = -1

        # 计算观察空间维度
        # 基础特征 23, 每个窗口 80+5, 冷热特征 80, 差分特征 80*(窗口数-1)=400; 总计 23+510+80+400 = 1013
        # 加上占位符 80+1 = 81, 整体维度 1013+81 = 1094
        self.raw_feature_dim = 23 + (NUM_LOTTERY_BALLS + 5) * len(WINDOW_SIZES) + NUM_LOTTERY_BALLS + NUM_LOTTERY_BALLS * (len(WINDOW_SIZES) - 1)
        obs_dim = self.raw_feature_dim + NUM_LOTTERY_BALLS + 1
        print(f"Calculated Observation Dimension: {obs_dim}")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(NUM_LOTTERY_BALLS,), dtype=np.float32)

        self.current_date = None
        self.actual_set = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_window_features(self, current_date):
        all_window_freqs = {}
        all_window_stats = {}
        last_seen_days = np.full(NUM_LOTTERY_BALLS, MAX_COLD_STREAK, dtype=np.float32)
        number_values_by_window = {w: [] for w in WINDOW_SIZES}

        max_window = max(WINDOW_SIZES)
        temp_last_seen = {num: MAX_COLD_STREAK for num in range(1, NUM_LOTTERY_BALLS + 1)}
        days_elapsed = 0
        for i in range(1, max_window + MAX_COLD_STREAK + 1):
            day = current_date - timedelta(days=i)
            if day < self.historical_df['Date'].min():
                break
            days_elapsed = i
            draw = self.known_draws.get(day)
            if draw:
                if i <= max_window:
                    for num in draw:
                        for w in WINDOW_SIZES:
                            if i <= w:
                                number_values_by_window[w].append(num)
                for num in draw:
                    if temp_last_seen[num] == MAX_COLD_STREAK:
                        temp_last_seen[num] = days_elapsed

        for num in range(1, NUM_LOTTERY_BALLS + 1):
            last_seen_days[num - 1] = min(temp_last_seen[num], MAX_COLD_STREAK) / MAX_COLD_STREAK

        for w in WINDOW_SIZES:
            freq = np.zeros(NUM_LOTTERY_BALLS, dtype=np.float32)
            values = number_values_by_window[w]
            if values:
                unique, counts = np.unique(values, return_counts=True)
                freq[unique - 1] = counts
                freq /= len(values)
                mean_val = np.mean(values) / NUM_LOTTERY_BALLS
                min_val = np.min(values) / NUM_LOTTERY_BALLS
                max_val = np.max(values) / NUM_LOTTERY_BALLS
                var_val = np.var(values) / (NUM_LOTTERY_BALLS ** 2)
                std_val = np.sqrt(var_val)
            else:
                mean_val = min_val = max_val = var_val = std_val = 0.0
            stats = np.array([mean_val, min_val, max_val, var_val, std_val], dtype=np.float32)
            all_window_freqs[w] = freq
            all_window_stats[w] = stats

        diff_freqs = []
        sorted_windows = sorted(WINDOW_SIZES)
        for i in range(len(sorted_windows) - 1):
            w_curr = sorted_windows[i + 1]
            w_prev = sorted_windows[i]
            diff = all_window_freqs.get(w_curr, np.zeros(NUM_LOTTERY_BALLS)) - \
                   all_window_freqs.get(w_prev, np.zeros(NUM_LOTTERY_BALLS))
            diff_freqs.append(diff.astype(np.float32))

        window_features_flat = []
        for w in sorted_windows:
            window_features_flat.append(all_window_freqs.get(w, np.zeros(NUM_LOTTERY_BALLS, dtype=np.float32)))
            window_features_flat.append(all_window_stats.get(w, np.zeros(5, dtype=np.float32)))

        return np.concatenate(window_features_flat), last_seen_days, np.concatenate(diff_freqs) if diff_freqs else np.array([], dtype=np.float32)

    def _get_observation(self, current_date):
        # 基础时间特征
        dow = current_date.weekday()
        month = current_date.month
        dow_onehot = np.eye(7, dtype=np.float32)[dow]
        month_onehot = np.eye(12, dtype=np.float32)[month - 1]
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)
        cyclical = np.array([dow_sin, dow_cos, month_sin, month_cos], dtype=np.float32)
        base_features = np.concatenate([dow_onehot, month_onehot, cyclical])  # 23 dims

        window_feats, cold_streaks, diff_freqs = self._compute_window_features(current_date)
        raw_features = np.concatenate([base_features, window_feats, cold_streaks, diff_freqs])

        if self.scaler:
            try:
                scaled_features = self.scaler.transform(raw_features.reshape(1, -1)).flatten()
            except Exception as e:
                print(f"Error during scaling: {e}")
                scaled_features = raw_features
        else:
            scaled_features = raw_features

        chosen_mask = np.zeros(NUM_LOTTERY_BALLS, dtype=np.float32)
        progress = np.array([0.0], dtype=np.float32)

        observation = np.concatenate([scaled_features, chosen_mask, progress]).astype(np.float32)
        return observation

    def reset(self, index=None):
        if index is not None:
            self.current_df_index = index
        elif self.is_training:
            self.current_df_index = random.choice(self.valid_subset_indices)
        else:
            if self.current_df_index == -1:
                self.current_df_index = self.valid_subset_indices[0]
            else:
                current_subset_pos = self.valid_subset_indices.index(self.current_df_index)
                next_subset_pos = current_subset_pos + 1
                if next_subset_pos < len(self.valid_subset_indices):
                    self.current_df_index = self.valid_subset_indices[next_subset_pos]
                else:
                    print("Validation sequence ended, looping back to start.")
                    self.current_df_index = self.valid_subset_indices[0]

        self.current_date = self.historical_df.loc[self.current_df_index, 'Date']
        self.actual_set = self.known_draws.get(self.current_date, set())

        if self.current_df_index < self.global_min_feature_index:
            print(f"Warning: Index {self.current_df_index} insufficient, moving to next valid.")
            return self.reset(index=self.global_min_feature_index if index is None else index + 1)

        return self._get_observation(self.current_date)

    def step(self, action):
        if action.shape != (NUM_LOTTERY_BALLS,):
            raise ValueError(f"Action shape mismatch. Expected ({NUM_LOTTERY_BALLS},), got {action.shape}")
        noise = np.random.uniform(low=-1e-6, high=1e-6, size=action.shape)
        noisy_action = action + noise
        top_indices = np.argsort(noisy_action)[-NUM_SELECTED_BALLS:]
        predicted_set = set((top_indices + 1).tolist())

        reward = 0.0
        if self.actual_set:
            hits = len(predicted_set.intersection(self.actual_set))
            reward = float(hits)
        else:
            reward = 0.0

        done = True
        info = {"predicted_set": predicted_set, "actual_set": self.actual_set, "date": self.current_date}

        return self._get_observation(self.current_date), reward, done, info

    def close(self):
        pass

    def render(self, mode='human', close=False):
        if close:
            return
        pred_list = sorted(list(self.info['predicted_set'])) if 'predicted_set' in self.info else []
        actual_list = sorted(list(self.info['actual_set'])) if self.info.get('actual_set') else "N/A"
        print(f"Date: {self.info.get('date', 'N/A')}, Predicted: {pred_list}, Actual: {actual_list}")

# ----------------------------
# 3. 特征缩放设置
# ----------------------------
print("--- 3. Setting up Feature Scaling ---")
temp_env = EnhancedLotteryEnv(df, df_train.index, known_draws, scaler=None)
initial_obs_unscaled = temp_env._get_observation(df.loc[temp_env.global_min_feature_index, 'Date'])
raw_feature_part = initial_obs_unscaled[:-NUM_LOTTERY_BALLS - 1]

print("Fitting StandardScaler on training data...")
scaler = StandardScaler()
num_scaler_samples = min(5000, len(temp_env.valid_subset_indices))
fitting_indices = random.sample(temp_env.valid_subset_indices, num_scaler_samples)
feature_samples = []
for idx in fitting_indices:
    obs = temp_env._get_observation(df.loc[idx, 'Date'])
    feature_samples.append(obs[:-NUM_LOTTERY_BALLS - 1])
scaler.fit(np.array(feature_samples))
print("StandardScaler fitted.")

del temp_env, initial_obs_unscaled, raw_feature_part, feature_samples

# ----------------------------
# 4. 环境创建与模型训练（使用向量化环境）
# ----------------------------
print("--- 4. Creating Environments and Training Models ---")
train_indices = df_train.index.tolist()

def make_env(rank, seed=0):
    def _init():
        env_seed = seed + rank
        random.seed(env_seed)
        np.random.seed(env_seed)
        return EnhancedLotteryEnv(df, train_indices, known_draws, scaler=scaler, is_training=True)
    return _init

# 设置并行环境数，充分利用 CPU 核心（这里上限设为 8，可根据实际情况调整）
num_cpu = min(os.cpu_count(), 8)
vec_env_a2c = SubprocVecEnv([make_env(i, seed=RANDOM_SEED) for i in range(num_cpu)])
vec_env_ppo = SubprocVecEnv([make_env(i, seed=RANDOM_SEED) for i in range(num_cpu)])
vec_env_sac = SubprocVecEnv([make_env(i, seed=RANDOM_SEED) for i in range(num_cpu)])
vec_env_td3 = SubprocVecEnv([make_env(i, seed=RANDOM_SEED) for i in range(num_cpu)])

policy_kwargs = {"net_arch": [512, 256, 128]}

# --- 模型训练 ---
tb_log_dir = os.path.join(RESULTS_DIR, "tb_logs/")
start_time = time.time()
print(f"\nTraining A2C for {TRAINING_TIMESTEPS} timesteps...")
model_a2c = A2C("MlpPolicy", vec_env_a2c, policy_kwargs=policy_kwargs, verbose=1,
                learning_rate=1e-4, seed=RANDOM_SEED, device=device, tensorboard_log=tb_log_dir)
model_a2c.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)
model_a2c.save(os.path.join(RESULTS_DIR, "a2c_kl8_model"))
print(f"A2C Training Time: {(time.time() - start_time) / 60:.2f} minutes")

start_time = time.time()
print(f"\nTraining PPO for {TRAINING_TIMESTEPS} timesteps...")
model_ppo = PPO("MlpPolicy", vec_env_ppo, policy_kwargs=policy_kwargs, verbose=1,
                learning_rate=1e-4, seed=RANDOM_SEED, device=device, tensorboard_log=tb_log_dir)
model_ppo.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)
model_ppo.save(os.path.join(RESULTS_DIR, "ppo_kl8_model"))
print(f"PPO Training Time: {(time.time() - start_time) / 60:.2f} minutes")

start_time = time.time()
print(f"\nTraining SAC for {TRAINING_TIMESTEPS} timesteps...")
model_sac = SAC("MlpPolicy", vec_env_sac, policy_kwargs=policy_kwargs, verbose=1,
                learning_rate=1e-4, seed=RANDOM_SEED, device=device, tensorboard_log=tb_log_dir,
                buffer_size=int(1e5))
model_sac.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)
model_sac.save(os.path.join(RESULTS_DIR, "sac_kl8_model"))
print(f"SAC Training Time: {(time.time() - start_time) / 60:.2f} minutes")

class CustomTD3Policy(TD3MlpPolicy):
    def __init__(self, *args, **kwargs):
        if "use_sde" in kwargs:
            kwargs.pop("use_sde")
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

start_time = time.time()
print(f"\nTraining TD3 for {TRAINING_TIMESTEPS} timesteps...")
model_td3 = TD3(CustomTD3Policy, vec_env_td3, policy_kwargs=policy_kwargs, verbose=1,
                learning_rate=1e-4, seed=RANDOM_SEED, device=device, tensorboard_log=tb_log_dir,
                buffer_size=int(1e5))
model_td3.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)
model_td3.save(os.path.join(RESULTS_DIR, "td3_kl8_model"))
print(f"TD3 Training Time: {(time.time() - start_time) / 60:.2f} minutes")

print("\nLoading trained models...")
model_a2c = A2C.load(os.path.join(RESULTS_DIR, "a2c_kl8_model"), device=device)
model_ppo = PPO.load(os.path.join(RESULTS_DIR, "ppo_kl8_model"), device=device)
model_sac = SAC.load(os.path.join(RESULTS_DIR, "sac_kl8_model"), device=device)
model_td3 = TD3.load(os.path.join(RESULTS_DIR, "td3_kl8_model"), device=device)

# ----------------------------
# 5. 在验证集上评估并计算权重
# ----------------------------
print("--- 5. Evaluating Models on Validation Set ---")
val_indices = df_val.index.tolist()
valid_val_indices = [idx for idx in val_indices if idx >= max(WINDOW_SIZES)]
if not valid_val_indices:
    raise ValueError("Validation set has no dates with enough historical data.")

def evaluate_model_on_validation(env, model, n_episodes):
    all_rewards = []
    obs = env.reset()
    for i in range(n_episodes):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        all_rewards.append(reward)
        if done:
            if i < len(valid_val_indices) - 1:
                obs = env.reset()
            else:
                break
    return np.mean(all_rewards) if all_rewards else 0

# 使用非向量化环境进行验证
env_eval = EnhancedLotteryEnv(df, valid_val_indices, known_draws, scaler=scaler, is_training=False)
num_val_episodes = len(valid_val_indices)
print(f"Evaluating on {num_val_episodes} validation days...")

a2c_avg_hits = evaluate_model_on_validation(env_eval, model_a2c, num_val_episodes)
env_eval = EnhancedLotteryEnv(df, valid_val_indices, known_draws, scaler=scaler, is_training=False)
ppo_avg_hits = evaluate_model_on_validation(env_eval, model_ppo, num_val_episodes)
env_eval = EnhancedLotteryEnv(df, valid_val_indices, known_draws, scaler=scaler, is_training=False)
sac_avg_hits = evaluate_model_on_validation(env_eval, model_sac, num_val_episodes)
env_eval = EnhancedLotteryEnv(df, valid_val_indices, known_draws, scaler=scaler, is_training=False)
td3_avg_hits = evaluate_model_on_validation(env_eval, model_td3, num_val_episodes)

print(f"\nValidation Avg Hits -> A2C: {a2c_avg_hits:.3f}, PPO: {ppo_avg_hits:.3f}, SAC: {sac_avg_hits:.3f}, TD3: {td3_avg_hits:.3f}")

epsilon = 1e-6
total_hits = a2c_avg_hits + ppo_avg_hits + sac_avg_hits + td3_avg_hits + 4 * epsilon
w_a2c = (a2c_avg_hits + epsilon) / total_hits if total_hits > 0 else 0.25
w_ppo = (ppo_avg_hits + epsilon) / total_hits if total_hits > 0 else 0.25
w_sac = (sac_avg_hits + epsilon) / total_hits if total_hits > 0 else 0.25
w_td3 = (td3_avg_hits + epsilon) / total_hits if total_hits > 0 else 0.25
w_sum = w_a2c + w_ppo + w_sac + w_td3
w_a2c /= w_sum
w_ppo /= w_sum
w_sac /= w_sum
w_td3 /= w_sum

print(f"Calculated Ensemble Weights -> A2C: {w_a2c:.3f}, PPO: {w_ppo:.3f}, SAC: {w_sac:.3f}, TD3: {w_td3:.3f}")

# ----------------------------
# 6. 集成预测下一天
# ----------------------------
print("--- 6. Predicting Next Day with Ensemble ---")
last_historical_date = df['Date'].max()
prediction_date = last_historical_date + pd.Timedelta(days=1)
print(f"Predicting for date: {prediction_date.date()}")

env_pred = EnhancedLotteryEnv(df, df.index, known_draws, scaler=scaler, is_training=False)
try:
    obs_pred = env_pred._get_observation(prediction_date)
except Exception as e:
    print(f"Error getting observation for prediction date: {e}")
    obs_pred = None

if obs_pred is not None:
    action_a2c, _ = model_a2c.predict(obs_pred, deterministic=True)
    action_ppo, _ = model_ppo.predict(obs_pred, deterministic=True)
    action_sac, _ = model_sac.predict(obs_pred, deterministic=True)
    action_td3, _ = model_td3.predict(obs_pred, deterministic=True)
    ensemble_action = (w_a2c * action_a2c + w_ppo * action_ppo +
                       w_sac * action_sac + w_td3 * action_td3)
    noise = np.random.uniform(low=-1e-6, high=1e-6, size=ensemble_action.shape)
    top_indices = np.argsort(ensemble_action + noise)[-NUM_SELECTED_BALLS:]
    final_pred_numbers = sorted((top_indices + 1).tolist())
    print(f"Ensemble Prediction for {prediction_date.date()}: {final_pred_numbers}")
    pred_df = pd.DataFrame([{
        "Date": prediction_date.date(),
        **{f"Pred{j:02d}": final_pred_numbers[j - 1] for j in range(1, NUM_SELECTED_BALLS + 1)}
    }])
    pred_path = os.path.join(RESULTS_DIR, "prediction_next_day_ensemble.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Prediction saved to {pred_path}")
else:
    print("Could not generate prediction due to observation error.")

# ----------------------------
# 7. 绘制验证集表现趋势图
# ----------------------------
print("--- 7. Plotting Validation Set Performance ---")
ensemble_hits = []
dates_list = []

env_eval = EnhancedLotteryEnv(df, valid_val_indices, known_draws, scaler=scaler, is_training=False)
obs = env_eval.reset()

for i in range(num_val_episodes):
    action_a2c, _ = model_a2c.predict(obs, deterministic=True)
    action_ppo, _ = model_ppo.predict(obs, deterministic=True)
    action_sac, _ = model_sac.predict(obs, deterministic=True)
    action_td3, _ = model_td3.predict(obs, deterministic=True)
    ensemble_action = (w_a2c * action_a2c + w_ppo * action_ppo +
                       w_sac * action_sac + w_td3 * action_td3)
    noise = np.random.uniform(low=-1e-6, high=1e-6, size=ensemble_action.shape)
    top_indices = np.argsort(ensemble_action + noise)[-NUM_SELECTED_BALLS:]
    predicted_set = set((top_indices + 1).tolist())
    current_date = env_eval.current_date
    actual_set = env_eval.actual_set
    if actual_set:
        hits = len(predicted_set.intersection(actual_set))
        ensemble_hits.append(hits)
        dates_list.append(current_date)
    obs, _, done, _ = env_eval.step(ensemble_action)
    if i >= len(valid_val_indices) - 1:
        break

if ensemble_hits:
    plt.figure(figsize=(12, 6))
    plt.plot(dates_list, ensemble_hits, marker='.', linestyle='-', markersize=4, label="Ensemble Hits per day")
    avg_ensemble_hits = np.mean(ensemble_hits)
    plt.axhline(y=avg_ensemble_hits, color='r', linestyle='--', label=f"Avg Hits = {avg_ensemble_hits:.3f}")
    plt.xlabel("Date")
    plt.ylabel(f"Hits (out of {NUM_SELECTED_BALLS})")
    plt.title("Validation Set - Ensemble Prediction Hit Trend")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "validation_hit_trend_ensemble.png")
    plt.savefig(plot_path)
    print(f"Validation hit trend plot saved to {plot_path}")
    plt.close()
else:
    print("No hits recorded for validation set plot.")

print("--- Script Finished ---")

