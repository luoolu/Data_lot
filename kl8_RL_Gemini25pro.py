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
"""

import pandas as pd
import numpy as np
import random
from datetime import timedelta, datetime
import gym
from gym import spaces
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # For potential parallelization
from stable_baselines3.td3.policies import MlpPolicy as TD3MlpPolicy
import time
import warnings

# --- Configuration ---
DATA_FILE = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/data/kl8/kl8_2025-03-28.csv"
RESULTS_DIR = "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/20250329/"
RANDOM_SEED = 42  # Changed seed for potentially different initialization
VALIDATION_SPLIT_DATE = "2025-03-01"  # Use data before this for training, after for validation
TRAINING_TIMESTEPS = 200000  # Increased training time
WINDOW_SIZES = [7, 14, 30, 60, 90, 180]
MAX_COLD_STREAK = 90  # Maximum days back to check for last appearance
NUM_LOTTERY_BALLS = 80
NUM_SELECTED_BALLS = 20

# --- Setup ---
warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs(RESULTS_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# Note: Stable Baselines 3 might need its own seeding for full reproducibility if using PyTorch backend directly

# ----------------------------
# 1. 数据读取与初步处理
# ----------------------------
print("--- 1. Loading and Preprocessing Data ---")
df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print(f"Loaded data: {len(df)} days from {df['Date'].min().date()} to {df['Date'].max().date()}.")

# Pre-calculate known draws dictionary for efficiency
known_draws = {
    row['Date']: set(int(row[f'k{j:02d}']) for j in range(1, NUM_SELECTED_BALLS + 1))
    for _, row in df.iterrows()
}
print("Pre-calculated known draws.")

# Train/Validation Split
df_train = df[df['Date'] < pd.to_datetime(VALIDATION_SPLIT_DATE)].copy()
df_val = df[df['Date'] >= pd.to_datetime(VALIDATION_SPLIT_DATE)].copy()

# Ensure validation set isn't empty
if df_val.empty or len(df_train) < max(WINDOW_SIZES):
    raise ValueError(
        f"Validation split date {VALIDATION_SPLIT_DATE} results in insufficient training or validation data.")

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

        self.historical_df = historical_df  # Full history for feature calculation
        self.data_subset_indices = data_subset_indices  # Indices (in historical_df) this env instance will sample from
        self.known_draws = known_draws_dict
        self.scaler = scaler  # Fitted StandardScaler
        self.is_training = is_training  # Controls random sampling vs sequential iteration

        # Find the earliest valid index in the full dataset for feature calculation
        self.global_min_feature_index = max(WINDOW_SIZES)

        # Ensure subset indices are valid
        self.valid_subset_indices = [idx for idx in self.data_subset_indices if idx >= self.global_min_feature_index]
        if not self.valid_subset_indices:
            raise ValueError("No valid indices available in the provided data subset for the required history windows.")
        self.current_df_index = -1  # Index within historical_df

        # --- Feature Calculation ---
        # Basic: day of week (7), month (12), cyclical (4) = 23
        # Windows (6 sizes): freq (80), mean, min, max, var, std (5) = 85 * 6 = 510
        # Hot/Cold Streaks: days since last seen (80) = 80
        # Diff Freq: (win[i] - win[i-1]) freq (80) * 5 pairs = 400
        # Total Raw Features: 23 + 510 + 80 + 400 = 1013
        # Placeholders: chosen mask (80, always 0 for one-shot), progress (1, always 0) = 81
        # Final Observation Dim: 1013 + 81 = 1094
        self.raw_feature_dim = 23 + (NUM_LOTTERY_BALLS + 5) * len(WINDOW_SIZES) + NUM_LOTTERY_BALLS + NUM_LOTTERY_BALLS * (len(WINDOW_SIZES) - 1)
        obs_dim = self.raw_feature_dim + NUM_LOTTERY_BALLS + 1
        print(f"Calculated Observation Dimension: {obs_dim}")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(NUM_LOTTERY_BALLS,), dtype=np.float32)

        self.current_date = None
        self.actual_set = None

    def seed(self, seed=None):
        from gym.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _compute_window_features(self, current_date):
        """Computes frequency, stats, cold streaks, and diff freqs up to current_date."""
        all_window_freqs = {}
        all_window_stats = {}
        last_seen_days = np.full(NUM_LOTTERY_BALLS, MAX_COLD_STREAK, dtype=np.float32)
        number_values_by_window = {w: [] for w in WINDOW_SIZES}

        # Efficiently gather historical draws within the largest window
        max_window = max(WINDOW_SIZES)
        relevant_draws = {}
        min_date = current_date - timedelta(days=max_window)

        # Iterate backwards once to collect relevant draws and last seen info
        temp_last_seen = {num: MAX_COLD_STREAK for num in range(1, NUM_LOTTERY_BALLS + 1)}
        days_elapsed = 0
        for i in range(1, max_window + MAX_COLD_STREAK + 1):  # Go back further for cold streaks
            day = current_date - timedelta(days=i)
            if day < self.historical_df['Date'].min():
                break  # Stop if we go beyond known history

            days_elapsed = i
            draw = self.known_draws.get(day)

            if draw:
                if i <= max_window:  # Collect draws within the largest window for stats
                    relevant_draws[day] = draw
                    for num in draw:
                        # Accumulate values for each window they fall into
                        for w in WINDOW_SIZES:
                            if i <= w:
                                number_values_by_window[w].append(num)

                # Update last seen day (relative to current_date)
                for num in draw:
                    if temp_last_seen[num] == MAX_COLD_STREAK:  # Update only first time seen going back
                        temp_last_seen[num] = days_elapsed

        # Finalize last_seen_days array (scaled)
        for num in range(1, NUM_LOTTERY_BALLS + 1):
            last_seen_days[num - 1] = min(temp_last_seen[num], MAX_COLD_STREAK) / MAX_COLD_STREAK  # Normalize

        # Calculate stats per window using collected values
        for w in WINDOW_SIZES:
            freq = np.zeros(NUM_LOTTERY_BALLS, dtype=np.float32)
            values = number_values_by_window[w]

            if values:
                unique, counts = np.unique(values, return_counts=True)
                freq[unique - 1] = counts
                freq /= len(values)  # Normalize freq within the window's actual draws

                mean_val = np.mean(values) / NUM_LOTTERY_BALLS
                min_val = np.min(values) / NUM_LOTTERY_BALLS
                max_val = np.max(values) / NUM_LOTTERY_BALLS
                var_val = np.var(values) / (NUM_LOTTERY_BALLS ** 2)  # Variance scales quadratically
                std_val = np.sqrt(var_val)  # Std Dev is sqrt of variance
            else:
                mean_val = min_val = max_val = var_val = std_val = 0.0

            stats = np.array([mean_val, min_val, max_val, var_val, std_val], dtype=np.float32)
            all_window_freqs[w] = freq
            all_window_stats[w] = stats

        # Calculate differenced frequencies
        diff_freqs = []
        sorted_windows = sorted(WINDOW_SIZES)
        for i in range(len(sorted_windows) - 1):
            w_curr = sorted_windows[i + 1]
            w_prev = sorted_windows[i]
            diff = all_window_freqs.get(w_curr, np.zeros(NUM_LOTTERY_BALLS)) - \
                   all_window_freqs.get(w_prev, np.zeros(NUM_LOTTERY_BALLS))
            diff_freqs.append(diff.astype(np.float32))

        # Combine all features
        window_features_flat = []
        for w in sorted_windows:
            window_features_flat.append(all_window_freqs.get(w, np.zeros(NUM_LOTTERY_BALLS, dtype=np.float32)))
            window_features_flat.append(all_window_stats.get(w, np.zeros(5, dtype=np.float32)))

        return np.concatenate(window_features_flat), last_seen_days, np.concatenate(diff_freqs) if diff_freqs else np.array([], dtype=np.float32)

    def _get_observation(self, current_date):
        # Basic Time Features
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

        # Advanced Historical Features
        window_feats, cold_streaks, diff_freqs = self._compute_window_features(current_date)  # 510, 80, 400 dims

        # Combine raw features
        raw_features = np.concatenate([base_features, window_feats, cold_streaks, diff_freqs])  # 1013 dims

        # --- Apply Scaling ---
        if self.scaler:
            try:
                # Reshape for scaler (expects 2D array)
                scaled_features = self.scaler.transform(raw_features.reshape(1, -1)).flatten()
            except Exception as e:
                print(f"Error during scaling: {e}")
                print(f"Raw features shape: {raw_features.shape}")
                # Handle potential shape mismatches or NaN values if they occur
                # Fallback: Use unscaled features or zeros, depending on strategy
                scaled_features = raw_features  # Fallback to unscaled
        else:
            scaled_features = raw_features  # No scaler available (e.g., before fitting)

        # Add placeholders
        chosen_mask = np.zeros(NUM_LOTTERY_BALLS, dtype=np.float32)
        progress = np.array([0.0], dtype=np.float32)

        observation = np.concatenate([scaled_features, chosen_mask, progress]).astype(np.float32)
        return observation

    def reset(self, index=None):
        if index is not None:  # Used for specific date evaluation
            self.current_df_index = index
        elif self.is_training:  # Random sampling for training
            self.current_df_index = random.choice(self.valid_subset_indices)
        else:  # Sequential iteration for validation/testing
            # Find the next valid index in the subset
            if self.current_df_index == -1:  # First call
                self.current_df_index = self.valid_subset_indices[0]
            else:
                current_subset_pos = self.valid_subset_indices.index(self.current_df_index)
                next_subset_pos = current_subset_pos + 1
                if next_subset_pos < len(self.valid_subset_indices):
                    self.current_df_index = self.valid_subset_indices[next_subset_pos]
                else:
                    # This indicates end of validation sequence - handle appropriately
                    # For gym env, maybe loop back or raise Done? Loop back for continuous eval.
                    print("Validation sequence ended, looping back to start.")
                    self.current_df_index = self.valid_subset_indices[0]

        self.current_date = self.historical_df.loc[self.current_df_index, 'Date']
        self.actual_set = self.known_draws.get(self.current_date, set())  # Use actual future data if available

        # Ensure the date is valid (has enough history) before getting observation
        if self.current_df_index < self.global_min_feature_index:
            print(f"Warning: Reset to index {self.current_df_index} which has insufficient history. Trying next valid.")
            # This case should ideally be prevented by valid_subset_indices, but as a safeguard:
            return self.reset(index=self.global_min_feature_index if index is None else index + 1)  # Recursive call to find valid start

        return self._get_observation(self.current_date)

    def step(self, action):
        # Action is vector of scores [0, 1] for each number
        # Select top N based on scores
        # Ensure action has the expected shape
        if action.shape != (NUM_LOTTERY_BALLS,):
            raise ValueError(f"Action shape mismatch. Expected ({NUM_LOTTERY_BALLS},), got {action.shape}")

        # Adding small noise to break ties randomly but consistently if seeds are set
        noise = np.random.uniform(low=-1e-6, high=1e-6, size=action.shape)
        noisy_action = action + noise

        top_indices = np.argsort(noisy_action)[-NUM_SELECTED_BALLS:]  # Indices 0 to 79
        predicted_set = set((top_indices + 1).tolist())  # Convert to numbers 1 to 80

        reward = 0.0
        if self.actual_set:  # If we have ground truth for this day
            hits = len(predicted_set.intersection(self.actual_set))
            # --- Advanced Reward Shaping (Optional) ---
            # Example: reward = hits**2 (emphasize higher hits)
            # Example: reward = hits - (NUM_SELECTED_BALLS - hits) * 0.1 # Penalize misses slightly
            reward = float(hits)  # Keep simple reward = hits for now
        else:
            # This happens if predicting future date where outcome is unknown
            reward = 0.0

        done = True  # One-shot decision per day
        info = {"predicted_set": predicted_set, "actual_set": self.actual_set, "date": self.current_date}

        # Return current observation for SB3 compatibility
        return self._get_observation(self.current_date), reward, done, info

    def close(self):
        pass

    def render(self, mode='human', close=False):
        if close:
            return
        pred_list = sorted(list(self.info['predicted_set'])) if 'predicted_set' in self.info else []
        actual_list = sorted(list(self.info['actual_set'])) if self.info.get('actual_set') else "N/A"
        print(f"Date: {self.info.get('date', 'N/A')}, Predicted: {pred_list}, Actual: {actual_list}, Hits: {self.reward if hasattr(self, 'reward') else 'N/A'}")


# ----------------------------
# 3. Feature Scaling Setup
# ----------------------------
print("--- 3. Setting up Feature Scaling ---")
# Create a temporary env instance just to get one raw feature vector for scaler fitting
temp_env = EnhancedLotteryEnv(df, df_train.index, known_draws, scaler=None)
initial_obs_unscaled = temp_env._get_observation(df.loc[temp_env.global_min_feature_index, 'Date'])
raw_feature_part = initial_obs_unscaled[:-NUM_LOTTERY_BALLS - 1]  # Extract the feature part

# Fit the scaler ONLY on training data features
print("Fitting StandardScaler on training data...")
scaler = StandardScaler()
# Collect features from a sample of the training set for fitting the scaler
num_scaler_samples = min(5000, len(temp_env.valid_subset_indices))  # Use up to 5000 samples
fitting_indices = random.sample(temp_env.valid_subset_indices, num_scaler_samples)
feature_samples = []
for idx in fitting_indices:
    obs = temp_env._get_observation(df.loc[idx, 'Date'])
    feature_samples.append(obs[:-NUM_LOTTERY_BALLS - 1])  # Append only the feature part

scaler.fit(np.array(feature_samples))
print("StandardScaler fitted.")

# Clean up temporary env
del temp_env, initial_obs_unscaled, raw_feature_part, feature_samples

# ----------------------------
# 4. Environment Creation & Model Training
# ----------------------------
print("--- 4. Creating Environments and Training Models ---")

# Create separate environments for each algorithm, using the FITTED scaler
# Pass only TRAINING indices to the training environments
train_indices = df_train.index.tolist()

# Use DummyVecEnv for simplicity, SubprocVecEnv for potential speedup with multiple cores
# vec_env_class = SubprocVecEnv if os.cpu_count() > 1 else DummyVecEnv
vec_env_class = DummyVecEnv  # Stick to Dummy for stability in complex envs first

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed envs.
    """
    def _init():
        # Important: Each process needs its own random seed if using SubprocVecEnv
        env_seed = seed + rank
        random.seed(env_seed)
        np.random.seed(env_seed)
        # Pass only training indices
        env = EnhancedLotteryEnv(df, train_indices, known_draws, scaler=scaler, is_training=True)
        # env.seed(env_seed) # Deprecated, seeding done via random/np
        return env
    return _init

# For SB3, it's often better to create a VecEnv even for n_envs=1
# num_cpu = 1 # Set to desired number of parallel environments (e.g., os.cpu_count())
# env_a2c = vec_env_class([make_env(i) for i in range(num_cpu)])
# env_ppo = vec_env_class([make_env(i) for i in range(num_cpu)])
# env_sac = vec_env_class([make_env(i) for i in range(num_cpu)])
# env_td3 = vec_env_class([make_env(i) for i in range(num_cpu)])

# Simpler: Create non-vectorized envs first for debugging, wrap later if needed
env_a2c = EnhancedLotteryEnv(df, train_indices, known_draws, scaler=scaler, is_training=True)
env_ppo = EnhancedLotteryEnv(df, train_indices, known_draws, scaler=scaler, is_training=True)
env_sac = EnhancedLotteryEnv(df, train_indices, known_draws, scaler=scaler, is_training=True)
env_td3 = EnhancedLotteryEnv(df, train_indices, known_draws, scaler=scaler, is_training=True)

# Define policy kwargs (slightly deeper network)
policy_kwargs = {
    "net_arch": [512, 256, 128]  # Increased complexity
}

# --- Model Training ---
# Consider using callbacks for evaluation during training, early stopping etc.
# Use Tensorboard for logging: tensorboard_log=os.path.join(RESULTS_DIR, "tb_logs/")

start_time = time.time()
print(f"\nTraining A2C for {TRAINING_TIMESTEPS} timesteps...")
model_a2c = A2C("MlpPolicy", env_a2c, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-4, seed=RANDOM_SEED,
                tensorboard_log=os.path.join(RESULTS_DIR, "tb_logs/"))  # Lower LR often better for longer training
model_a2c.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)  # Log more often
model_a2c.save(os.path.join(RESULTS_DIR, "a2c_kl8_model"))
print(f"A2C Training Time: {(time.time() - start_time) / 60:.2f} minutes")

start_time = time.time()
print(f"\nTraining PPO for {TRAINING_TIMESTEPS} timesteps...")
model_ppo = PPO("MlpPolicy", env_ppo, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-4, seed=RANDOM_SEED,
                tensorboard_log=os.path.join(RESULTS_DIR, "tb_logs/"))
model_ppo.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)
model_ppo.save(os.path.join(RESULTS_DIR, "ppo_kl8_model"))
print(f"PPO Training Time: {(time.time() - start_time) / 60:.2f} minutes")

start_time = time.time()
print(f"\nTraining SAC for {TRAINING_TIMESTEPS} timesteps...")
# SAC might benefit from buffer size adjustments, gradient steps per timestep etc.
model_sac = SAC("MlpPolicy", env_sac, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-4, seed=RANDOM_SEED,
                tensorboard_log=os.path.join(RESULTS_DIR, "tb_logs/"),
                buffer_size=int(1e5))  # Smaller buffer might adapt faster to recent data
model_sac.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)
model_sac.save(os.path.join(RESULTS_DIR, "sac_kl8_model"))
print(f"SAC Training Time: {(time.time() - start_time) / 60:.2f} minutes")

# TD3 Policy definition (as before)
class CustomTD3Policy(TD3MlpPolicy):
    def __init__(self, *args, **kwargs):
        if "use_sde" in kwargs:
            kwargs.pop("use_sde")
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

start_time = time.time()
print(f"\nTraining TD3 for {TRAINING_TIMESTEPS} timesteps...")
model_td3 = TD3(CustomTD3Policy, env_td3, policy_kwargs=policy_kwargs, verbose=1, learning_rate=1e-4, seed=RANDOM_SEED,
                tensorboard_log=os.path.join(RESULTS_DIR, "tb_logs/"),
                buffer_size=int(1e5))
model_td3.learn(total_timesteps=TRAINING_TIMESTEPS, log_interval=100)
model_td3.save(os.path.join(RESULTS_DIR, "td3_kl8_model"))
print(f"TD3 Training Time: {(time.time() - start_time) / 60:.2f} minutes")

# --- Load models back (optional, good practice) ---
print("\nLoading trained models...")
model_a2c = A2C.load(os.path.join(RESULTS_DIR, "a2c_kl8_model"))
model_ppo = PPO.load(os.path.join(RESULTS_DIR, "ppo_kl8_model"))
model_sac = SAC.load(os.path.join(RESULTS_DIR, "sac_kl8_model"))
model_td3 = TD3.load(os.path.join(RESULTS_DIR, "td3_kl8_model"))

# ----------------------------
# 5. Evaluate on VALIDATION Set & Calculate Weights
# ----------------------------
print("--- 5. Evaluating Models on Validation Set ---")

# Create a single environment instance for validation (iterates sequentially)
val_indices = df_val.index.tolist()
# Ensure validation indices are valid w.r.t history requirements
valid_val_indices = [idx for idx in val_indices if idx >= max(WINDOW_SIZES)]
if not valid_val_indices:
    raise ValueError("Validation set has no dates with enough historical data.")

env_eval = EnhancedLotteryEnv(df, valid_val_indices, known_draws, scaler=scaler, is_training=False)

def evaluate_model_on_validation(env, model, n_episodes):
    """Evaluates the model sequentially on the validation set."""
    all_rewards = []
    obs = env.reset()  # Start from the beginning of the validation sequence
    for i in range(n_episodes):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        all_rewards.append(reward)
        if done:  # Should always be true for one-shot env
            if i < len(valid_val_indices) - 1:
                obs = env.reset()  # This actually just prepares the *next* state in seq
            else:
                break  # Stop after evaluating all validation points once
    return np.mean(all_rewards) if all_rewards else 0

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

# Calculate Dynamic Weights based on validation performance
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
# 6. Ensemble Prediction for Next Day
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
# 7. Plot Validation Set Hit Trend
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
