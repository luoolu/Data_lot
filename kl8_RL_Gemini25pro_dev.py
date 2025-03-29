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
顶级科学家级优化版强化学习预测 KL8 彩票数据 (V3)
目标：尽可能准确预测数据集外接下来一天的数据

主要改进 (V3):
1.  **Hyperparameter Tuning Framework**: Added structure/comments for integrating Optuna.
2.  **Increased Training Time**: Significantly longer default training steps (adjust based on convergence).
3.  **Parallel Training**: Utilized SubprocVecEnv for faster training on multi-core systems.
4.  **Refined Network Architecture**: Slightly deeper default MLP, emphasized tuning.
5.  **Enhanced Ensemble Weighting**: Introduced temperature scaling for validation-based weights.
6.  **Robustness**: Added basic error handling during validation.
7.  **Configuration Clarity**: Centralized more parameters.
8.  **Code Structure**: Improved comments, added basic docstrings, minor refactoring.
9.  **Potential Numba Optimization**: Added notes for optimizing feature calculation.
10. **Clearer Seeding**: Ensured proper seeding for parallel environments.

Previous Features Retained (from V2):
- Multiple RL Algorithms (A2C, PPO, SAC, TD3)
- Rich Feature Set (Multi-window stats, Periodicity, Hot/Cold, Diff Freq)
- Feature Standardization (StandardScaler)
- Train/Validation Split
- Ensemble Prediction
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor # For logging episode stats
from stable_baselines3.td3.policies import MlpPolicy as TD3MlpPolicy
# Optional: For advanced network architectures
# from stable_baselines3.common.policies import ActorCriticPolicy
# from torch import nn
import time
import warnings
# Optional: For feature calculation optimization
# import numba

# --- Configuration ---
DATA_FILE = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/data/kl8/kl8_2025-03-28.csv"
RESULTS_DIR = "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/20250329/"
TENSORBOARD_LOG_DIR = os.path.join(RESULTS_DIR, "tb_logs/")
RANDOM_SEED = 42
VALIDATION_SPLIT_DATE = "2025-03-01"
# >>> INCREASED TRAINING TIME <<< Need monitoring via TensorBoard
TRAINING_TIMESTEPS = 1_000_000 # Increased significantly, adjust based on convergence/resources
WINDOW_SIZES = [7, 14, 30, 60, 90, 180]
MAX_COLD_STREAK = 90  # Max days back for last appearance check (relative normalization)
NUM_LOTTERY_BALLS = 80
NUM_SELECTED_BALLS = 20
NUM_CPU = max(1, os.cpu_count() // 2) if os.cpu_count() else 1 # Use half available cores, minimum 1
USE_SUBPROC_ENV = True # Set to False to use DummyVecEnv (slower but sometimes easier to debug)

# --- Hyperparameters (Critical to Tune!) ---
# These are starting points. Use Optuna/Ray Tune for optimization.
LEARNING_RATE = 1e-4 # Often needs tuning (e.g., 3e-4, 1e-5)
# >>> Slightly Deeper Network <<< Also critical to tune
NET_ARCH = [512, 512, 256, 128] # Example: Deeper MLP
# NET_ARCH = dict(pi=[256, 256], vf=[256, 256]) # Alternative for ActorCritic
# NET_ARCH = [256, 128] # Shallower Example
GAMMA = 0.99 # Discount factor (less relevant for episodic env, but still used)
BUFFER_SIZE_OFF_POLICY = int(1e5) # For SAC/TD3, adjust based on memory/data correlation needs
BATCH_SIZE_OFF_POLICY = 256 # For SAC/TD3
# PPO Specific: n_steps, batch_size, n_epochs, clip_range, gae_lambda
# A2C Specific: n_steps, vf_coef, ent_coef
ENSEMBLE_TEMP = 2.0 # Temperature for weighting. 1.0 = proportional, >1.0 emphasizes winners more

# --- Setup ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning) # Ignore some SB3 warnings if needed
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# Note: PyTorch seeding needed for full reproducibility if using GPU
# import torch
# torch.manual_seed(RANDOM_SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(RANDOM_SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# ----------------------------
# 1. Data Loading & Preparation
# ----------------------------
print("--- 1. Loading and Preparing Data ---")
try:
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    exit()

df = df.sort_values("Date").reset_index(drop=True)
print(f"Loaded data: {len(df)} days from {df['Date'].min().date()} to {df['Date'].max().date()}.")

# Pre-calculate known draws dictionary for faster lookups
known_draws = {
    row['Date']: frozenset(int(row[f'k{j:02d}']) for j in range(1, NUM_SELECTED_BALLS + 1) if pd.notna(row[f'k{j:02d}']))
    for _, row in df.iterrows()
}
print("Pre-calculated known draws.")

# Train/Validation Split
df_train = df[df['Date'] < pd.to_datetime(VALIDATION_SPLIT_DATE)].copy()
df_val = df[df['Date'] >= pd.to_datetime(VALIDATION_SPLIT_DATE)].copy()

# Validate split
if df_val.empty or len(df_train) < max(WINDOW_SIZES):
    raise ValueError(
        f"Validation split date {VALIDATION_SPLIT_DATE} results in insufficient training "
        f"({len(df_train)} days, need >{max(WINDOW_SIZES)}) or empty validation data ({len(df_val)} days)."
    )

print(f"Training data: {len(df_train)} days (until {df_train['Date'].max().date()})")
print(f"Validation data: {len(df_val)} days (from {df_val['Date'].min().date()})")


# ----------------------------
# 2. Enhanced RL Environment
# ----------------------------
print("--- 2. Defining Enhanced RL Environment ---")

class EnhancedLotteryEnv(gym.Env):
    """
    Enhanced Gym environment for KL8 prediction.

    Args:
        historical_df (pd.DataFrame): Full historical data.
        data_subset_indices (list): Indices within historical_df for this env instance.
        known_draws_dict (dict): Pre-calculated {date: frozenset(numbers)}.
        scaler (StandardScaler): Fitted StandardScaler instance.
        is_training (bool): If True, sample randomly from subset; otherwise, iterate sequentially.
        seed (int): Random seed for the environment.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, historical_df, data_subset_indices, known_draws_dict, scaler, is_training=True, seed=RANDOM_SEED):
        super(EnhancedLotteryEnv, self).__init__()

        self.historical_df = historical_df
        self.data_subset_indices = data_subset_indices
        self.known_draws = known_draws_dict
        self.scaler = scaler
        self.is_training = is_training
        self.np_random = None # Will be initialized by seed()

        # Find the earliest valid index globally (for feature calculation)
        self.global_min_feature_index = max(WINDOW_SIZES)

        # Filter subset indices to ensure enough history
        self.valid_subset_indices = [idx for idx in self.data_subset_indices if idx >= self.global_min_feature_index]
        if not self.valid_subset_indices:
            raise ValueError("No valid indices in the provided data subset have enough history for feature calculation.")

        self._internal_subset_iterator = 0 # For sequential validation iteration
        self.current_df_index = -1 # Index within historical_df

        # --- Calculate Feature Dimensions (Do this once) ---
        # Base: day of week (7), month (12), cyclical (4) = 23
        base_dim = 7 + 12 + 4
        # Window Stats: freq (80), mean, min, max, var, std (5) = 85 per window
        window_stat_dim = (NUM_LOTTERY_BALLS + 5) * len(WINDOW_SIZES)
        # Hot/Cold Streaks: days since last seen (80)
        cold_streak_dim = NUM_LOTTERY_BALLS
        # Diff Freq: (win[i] - win[i-1]) freq (80) * (num_windows - 1) pairs
        diff_freq_dim = NUM_LOTTERY_BALLS * (len(WINDOW_SIZES) - 1)
        self.raw_feature_dim = base_dim + window_stat_dim + cold_streak_dim + diff_freq_dim

        # Placeholders: chosen mask (80, currently unused but kept for potential future state), progress (1)
        placeholder_dim = NUM_LOTTERY_BALLS + 1
        obs_dim = self.raw_feature_dim + placeholder_dim
        self._obs_dim_calculated = obs_dim # Store calculated dimension

        print(f"Env Instance: Calculated Observation Dimension: {self._obs_dim_calculated}")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim_calculated,), dtype=np.float32)
        # Continuous action space: Assign a score [0, 1] to each ball
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(NUM_LOTTERY_BALLS,), dtype=np.float32)

        self.current_date = None
        self.actual_set = None
        self.info = {} # Store info for rendering/debugging
        self.reward = 0.0

        self.seed(seed) # Initialize random generator

    def seed(self, seed=None):
        from gym.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        # Also seed python's random for choices if necessary
        random.seed(seed)
        np.random.seed(seed)
        # print(f"Env seeded with {seed}") # Debugging
        return [seed]

    # Optional: Decorate with numba if feature calculation is a bottleneck
    # Requires careful handling of data types and supported operations (e.g., dicts might be tricky)
    # @numba.njit(cache=True) # Example decorator (would need significant refactoring)
    def _compute_window_features(self, current_date_ordinal):
        """Computes historical features up to the day *before* current_date."""
        # Convert ordinal back to datetime for calculations if needed, or pass date object directly
        # current_date = pd.Timestamp.fromordinal(current_date_ordinal) # If passing ordinal
        current_date = current_date_ordinal # Assuming date object is passed

        all_window_freqs = {}
        all_window_stats = {}
        # Initialize last seen days - normalized later
        last_seen_days_ago = np.full(NUM_LOTTERY_BALLS, MAX_COLD_STREAK, dtype=np.int32)
        number_values_by_window = {w: [] for w in WINDOW_SIZES}

        max_window = max(WINDOW_SIZES)
        max_check_days = max(max_window, MAX_COLD_STREAK) # Go back enough for stats and cold streaks

        # Iterate backwards once to collect all necessary info efficiently
        temp_last_seen_relative = {num: max_check_days + 1 for num in range(1, NUM_LOTTERY_BALLS + 1)}

        for days_back in range(1, max_check_days + 1):
            day_to_check = current_date - timedelta(days=days_back)
            # Optimization: Check if date exists in known_draws quickly
            draw = self.known_draws.get(day_to_check)

            if draw: # If there was a draw on this day
                # Update last seen (only the first time we see it going back)
                for num in draw:
                    if temp_last_seen_relative[num] > max_check_days: # If not yet seen
                        temp_last_seen_relative[num] = days_back

                # Collect numbers within the largest window for statistics
                if days_back <= max_window:
                    for num in draw:
                        # Append to all relevant window lists
                        for w in WINDOW_SIZES:
                            if days_back <= w:
                                number_values_by_window[w].append(num)

            # Optimization: Stop checking for cold streaks if all numbers found within MAX_COLD_STREAK? Maybe not worth complexity.
            # Optimization: Stop checking stats if day_to_check is before the start of data? Handled implicitly by known_draws.get

        # Finalize last_seen_days array (normalized 0-1, where 1 means >= MAX_COLD_STREAK days ago)
        for num in range(1, NUM_LOTTERY_BALLS + 1):
            last_seen_days_ago[num - 1] = min(temp_last_seen_relative[num], MAX_COLD_STREAK)
        # Normalize relative to MAX_COLD_STREAK
        norm_last_seen = last_seen_days_ago.astype(np.float32) / MAX_COLD_STREAK

        # Calculate stats per window
        sorted_windows = sorted(WINDOW_SIZES)
        raw_freq_collector = {} # Store raw frequencies for differencing

        for w in sorted_windows:
            freq = np.zeros(NUM_LOTTERY_BALLS, dtype=np.float32)
            values = number_values_by_window[w]

            if values:
                # Calculate frequency within the window
                unique_nums, counts = np.unique(values, return_counts=True)
                # Normalize freq by number of draws in the window (can estimate or count exactly if needed)
                # Simple approach: divide by number of values found * (NUM_SELECTED_BALLS / NUM_LOTTERY_BALLS) as rough estimate?
                # Safer approach: divide by number of *days* in window with draws?
                # Current approach: Normalize by count of numbers drawn
                # Let's normalize by window size for consistency:
                freq[unique_nums - 1] = counts / w # Frequency per day in window

                raw_freq_collector[w] = freq.copy() # Store for diff calc

                # Calculate statistics (normalized 0-1 where appropriate)
                mean_val = np.mean(values) / NUM_LOTTERY_BALLS
                min_val = np.min(values) / NUM_LOTTERY_BALLS
                max_val = np.max(values) / NUM_LOTTERY_BALLS
                var_val = np.var(values) / (NUM_LOTTERY_BALLS ** 2) # Variance scales quadratically
                std_val = np.sqrt(var_val) # Std Dev is sqrt of variance
            else:
                raw_freq_collector[w] = freq # Store zeros
                mean_val = min_val = max_val = var_val = std_val = 0.0

            stats = np.array([mean_val, min_val, max_val, var_val, std_val], dtype=np.float32)
            # Use raw_freq_collector[w] for the frequency feature for this window
            all_window_freqs[w] = raw_freq_collector[w]
            all_window_stats[w] = stats

        # Calculate differenced frequencies
        diff_freqs_list = []
        for i in range(len(sorted_windows) - 1):
            w_curr = sorted_windows[i + 1]
            w_prev = sorted_windows[i]
            # Use the stored raw frequencies
            diff = raw_freq_collector[w_curr] - raw_freq_collector[w_prev]
            diff_freqs_list.append(diff.astype(np.float32))

        # Combine all features into flat arrays
        window_features_flat = []
        for w in sorted_windows:
            window_features_flat.append(all_window_freqs[w]) # Already normalized freq
            window_features_flat.append(all_window_stats[w]) # Normalized stats

        window_features_concat = np.concatenate(window_features_flat) if window_features_flat else np.array([], dtype=np.float32)
        diff_freqs_concat = np.concatenate(diff_freqs_list) if diff_freqs_list else np.array([], dtype=np.float32)

        return window_features_concat, norm_last_seen, diff_freqs_concat

    def _get_observation(self, current_date):
        """Calculates the observation vector for the given date."""
        # 1. Basic Time Features
        day_of_week = current_date.weekday()
        month = current_date.month
        dow_onehot = np.eye(7, dtype=np.float32)[day_of_week]
        month_onehot = np.eye(12, dtype=np.float32)[month - 1]
        # Cyclical features (normalized)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)
        cyclical_features = np.array([dow_sin, dow_cos, month_sin, month_cos], dtype=np.float32)
        base_features = np.concatenate([dow_onehot, month_onehot, cyclical_features])

        # 2. Advanced Historical Features (calculated based on data *before* current_date)
        # Pass current_date directly, could convert to ordinal if numba requires simple types
        window_feats, cold_streaks, diff_freqs = self._compute_window_features(current_date)

        # 3. Combine Raw Features
        raw_features_list = [base_features, window_feats, cold_streaks, diff_freqs]
        raw_features = np.concatenate([f for f in raw_features_list if f.size > 0])

        if raw_features.shape[0] != self.raw_feature_dim:
             print(f"Warning: Raw feature dimension mismatch! Expected {self.raw_feature_dim}, Got {raw_features.shape[0]}. Date: {current_date}")
             # Attempt to pad or truncate, but indicates an issue in feature calculation logic
             # This should ideally not happen if calculation is correct.
             # For now, let's pad with zeros if too short, or truncate if too long, and raise warning.
             if raw_features.shape[0] < self.raw_feature_dim:
                 padding = np.zeros(self.raw_feature_dim - raw_features.shape[0], dtype=np.float32)
                 raw_features = np.concatenate([raw_features, padding])
             else:
                 raw_features = raw_features[:self.raw_feature_dim]

        # 4. Apply Scaling
        scaled_features = raw_features # Default if no scaler
        if self.scaler:
            try:
                # Scaler expects 2D array: (n_samples, n_features)
                scaled_features = self.scaler.transform(raw_features.reshape(1, -1)).flatten()
            except ValueError as e:
                 print(f"Error during scaling: {e}. Raw features shape: {raw_features.shape}. Expected features: {self.scaler.n_features_in_}")
                 # Fallback: Use unscaled features or zeros, depending on strategy. Using unscaled here.
                 scaled_features = raw_features
            except Exception as e:
                 print(f"Unexpected error during scaling: {e}")
                 scaled_features = raw_features


        # 5. Add Placeholders
        # chosen_mask indicates which numbers were chosen in previous steps (if multi-step). Here, always 0.
        chosen_mask = np.zeros(NUM_LOTTERY_BALLS, dtype=np.float32)
        # progress indicates how far through an episode we are (if multi-step). Here, always 0.
        progress = np.array([0.0], dtype=np.float32)

        observation = np.concatenate([scaled_features, chosen_mask, progress]).astype(np.float32)

        # Final dimension check
        if observation.shape[0] != self._obs_dim_calculated:
            raise ValueError(f"Final observation dimension mismatch! Expected {self._obs_dim_calculated}, Got {observation.shape[0]}. Date: {current_date}")

        return observation

    def reset(self, index=None):
        """Resets the environment to a new state (day)."""
        if index is not None: # Specific index requested (e.g., for prediction)
             if index < self.global_min_feature_index:
                 raise ValueError(f"Cannot reset to index {index}, insufficient history. Min required: {self.global_min_feature_index}")
             self.current_df_index = index
        elif self.is_training: # Training mode: sample randomly
            self.current_df_index = self.np_random.choice(self.valid_subset_indices)
        else: # Validation/Testing mode: iterate sequentially
            if self._internal_subset_iterator >= len(self.valid_subset_indices):
                 print("Validation sequence finished. Resetting iterator.")
                 self._internal_subset_iterator = 0 # Loop back for continuous evaluation if needed
                 # Or could raise an error/signal done if only one pass is desired
            self.current_df_index = self.valid_subset_indices[self._internal_subset_iterator]
            self._internal_subset_iterator += 1

        # Set current date and actual result (if known)
        self.current_date = self.historical_df.loc[self.current_df_index, 'Date']
        # Look ahead one day for the actual result to predict against
        target_date = self.current_date # In this setup, we predict FOR the current_date based on past
        self.actual_set = self.known_draws.get(target_date, frozenset()) # Ground truth for this date

        # print(f"Resetting env to index {self.current_df_index}, date {self.current_date.date()}") # Debugging
        return self._get_observation(self.current_date)

    def step(self, action):
        """Takes an action, returns observation, reward, done, info."""
        # Action is expected to be a vector of scores [0, 1] for each number.
        if action.shape != (NUM_LOTTERY_BALLS,):
            action = np.squeeze(action) # Handle potential extra dimension
            if action.shape != (NUM_LOTTERY_BALLS,):
                 raise ValueError(f"Action shape mismatch. Expected ({NUM_LOTTERY_BALLS},), got {action.shape}")

        # Select top N based on scores. Add small noise for tie-breaking.
        # Use the environment's np_random for reproducibility within the env instance
        noise = self.np_random.uniform(low=-1e-6, high=1e-6, size=action.shape)
        noisy_action = action + noise

        # Indices of the top N scores (0 to NUM_LOTTERY_BALLS-1)
        top_indices = np.argsort(noisy_action)[-NUM_SELECTED_BALLS:]
        # Convert indices to lottery numbers (1 to NUM_LOTTERY_BALLS)
        predicted_set = frozenset((top_indices + 1).tolist())

        # Calculate reward based on hits
        reward = 0.0
        if self.actual_set: # Check if ground truth is available
            hits = len(predicted_set.intersection(self.actual_set))
            # Reward shaping: Simple hits count is robust. Could experiment with:
            # reward = float(hits**2) # Emphasize more hits
            # reward = float(hits) / NUM_SELECTED_BALLS # Normalized reward
            reward = float(hits)
        else:
            # This occurs if predicting a future date beyond known data
            reward = 0.0 # No ground truth to calculate reward

        self.reward = reward # Store reward for rendering

        # Episode is done after one prediction step
        done = True
        self.info = {
            "predicted_set": predicted_set,
            "actual_set": self.actual_set,
            "hits": reward, # Using reward directly as hits here
            "date": self.current_date
        }

        # The observation returned should be for the *next* state.
        # Since this is an episodic task (one prediction per day),
        # returning the *current* observation is standard practice in SB3 for episodic tasks.
        # The environment will be reset externally before the next prediction.
        current_observation = self._get_observation(self.current_date)

        return current_observation, reward, done, self.info

    def render(self, mode='human', close=False):
        """Prints the prediction results for the current step."""
        if close:
            return
        pred_list = sorted(list(self.info.get('predicted_set', [])))
        actual_list = sorted(list(self.info.get('actual_set', []))) if self.info.get('actual_set') else "N/A"
        hits = self.info.get('hits', 'N/A')
        date_str = self.info.get('date', pd.Timestamp('NaT')).strftime('%Y-%m-%d')

        print(f"Date: {date_str}, Hits: {hits}, "
              f"Predicted: {pred_list}, Actual: {actual_list}")

    def close(self):
        """Clean up environment resources."""
        pass


# ----------------------------
# 3. Feature Scaling Setup
# ----------------------------
print("--- 3. Setting up Feature Scaling ---")
# Create a temporary env instance to get one raw feature vector's shape and fit scaler
# Use *training* data indices for fitting the scaler
train_indices = df_train.index.tolist()
# Ensure we use indices that have enough history
valid_train_indices_for_scaling = [idx for idx in train_indices if idx >= max(WINDOW_SIZES)]

if not valid_train_indices_for_scaling:
     raise ValueError("Insufficient training data points with enough history to fit scaler.")

# Use a temporary env *without* a scaler to generate features for fitting
temp_env_for_scaling = EnhancedLotteryEnv(df, valid_train_indices_for_scaling, known_draws, scaler=None, is_training=True, seed=RANDOM_SEED)
_ = temp_env_for_scaling.reset() # Call reset once to initialize properly

# Extract raw features from a sample of the training set
print(f"Collecting feature samples for scaling (up to 5000 points)...")
num_scaler_samples = min(5000, len(valid_train_indices_for_scaling))
fitting_indices = random.sample(valid_train_indices_for_scaling, num_scaler_samples)
feature_samples = []
start_time_scaling = time.time()
for i, idx in enumerate(fitting_indices):
    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{num_scaler_samples} samples for scaler...")
    # Get observation for the date *at* index idx
    obs_date = df.loc[idx, 'Date']
    obs = temp_env_for_scaling._get_observation(obs_date)
    # Extract only the raw feature part (before placeholders)
    raw_feature_part = obs[:-NUM_LOTTERY_BALLS - 1]
    if raw_feature_part.shape[0] == temp_env_for_scaling.raw_feature_dim:
        feature_samples.append(raw_feature_part)
    else:
        print(f"Warning: Skipping index {idx} for scaler fitting due to feature dimension mismatch ({raw_feature_part.shape[0]} vs {temp_env_for_scaling.raw_feature_dim}).")

print(f"Feature sample collection took {time.time() - start_time_scaling:.2f} seconds.")

if not feature_samples:
    raise ValueError("No valid feature samples collected for scaler fitting. Check environment logic.")

# Fit the scaler
scaler = StandardScaler()
print("Fitting StandardScaler...")
start_time_fitting = time.time()
scaler.fit(np.array(feature_samples))
print(f"StandardScaler fitted on {len(feature_samples)} samples in {time.time() - start_time_fitting:.2f} seconds.")
print(f"Scaler expects input features: {scaler.n_features_in_}")

# Clean up temporary env
del temp_env_for_scaling, feature_samples, raw_feature_part

# ----------------------------
# 4. Environment Creation & Model Training (Parallelized)
# ----------------------------
# Requires the `if __name__ == "__main__":` block for SubprocVecEnv
def main_training_and_evaluation():
    print("--- 4. Creating Environments and Training Models ---")

    # --- Environment Factory Function ---
    def make_env(rank, seed=RANDOM_SEED, env_indices=None, is_training=True):
        """
        Utility function for multiprocessed envs or single env creation.
        """
        if env_indices is None:
            env_indices = train_indices # Default to training indices
        def _init():
            env_seed = seed + rank # Ensure different seed for each parallel env
            env = EnhancedLotteryEnv(
                historical_df=df,
                data_subset_indices=env_indices,
                known_draws_dict=known_draws,
                scaler=scaler, # Use the globally fitted scaler
                is_training=is_training,
                seed=env_seed
            )
            # Wrap with Monitor for episode logging (rewards, lengths) -> useful for TensorBoard
            log_dir = os.path.join(TENSORBOARD_LOG_DIR, f'monitor_{rank}')
            os.makedirs(log_dir, exist_ok=True)
            env = Monitor(env, log_dir)
            return env
        return _init

    # --- Choose VecEnv implementation ---
    vec_env_class = SubprocVecEnv if USE_SUBPROC_ENV and NUM_CPU > 1 else DummyVecEnv
    print(f"Using VecEnv Class: {vec_env_class.__name__} with {NUM_CPU} environments.")

    # --- Create Vectorized Environments for Training ---
    # Pass only TRAINING indices to the training environments
    valid_train_indices = [idx for idx in train_indices if idx >= max(WINDOW_SIZES)]
    if not valid_train_indices:
        raise ValueError("No valid training indices with enough history found.")

    print("Creating vectorized environments for training...")
    train_env_a2c = vec_env_class([make_env(i, RANDOM_SEED, valid_train_indices, is_training=True) for i in range(NUM_CPU)])
    train_env_ppo = vec_env_class([make_env(i, RANDOM_SEED + NUM_CPU, valid_train_indices, is_training=True) for i in range(NUM_CPU)]) # Offset seed
    train_env_sac = vec_env_class([make_env(i, RANDOM_SEED + 2*NUM_CPU, valid_train_indices, is_training=True) for i in range(NUM_CPU)])
    train_env_td3 = vec_env_class([make_env(i, RANDOM_SEED + 3*NUM_CPU, valid_train_indices, is_training=True) for i in range(NUM_CPU)])

    # --- Define Policy Kwargs ---
    # This is a crucial hyperparameter to tune!
    policy_kwargs = {"net_arch": NET_ARCH}
    print(f"Using Network Architecture: {NET_ARCH}")
    # Example for custom Actor/Critic architecture (more complex tuning)
    # policy_kwargs_ac = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    # Example for using LSTM/Transformer (requires sb3-contrib & careful setup)
    # policy_kwargs_lstm = dict(net_arch=[...], features_extractor_class=...) # Placeholder

    # --- Model Training ---
    # Launch TensorBoard in your terminal: tensorboard --logdir={TENSORBOARD_LOG_DIR}
    print(f"\n--- Starting Model Training ({TRAINING_TIMESTEPS} timesteps each) ---")
    print(f"Monitor training progress using TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}")

    models = {}
    model_configs = {
        "A2C": {"class": A2C, "env": train_env_a2c, "policy": "MlpPolicy",
                "params": {"learning_rate": LEARNING_RATE, "gamma": GAMMA, "seed": RANDOM_SEED, "policy_kwargs": policy_kwargs}},
        "PPO": {"class": PPO, "env": train_env_ppo, "policy": "MlpPolicy",
                "params": {"learning_rate": LEARNING_RATE, "gamma": GAMMA, "seed": RANDOM_SEED, "policy_kwargs": policy_kwargs}},
        "SAC": {"class": SAC, "env": train_env_sac, "policy": "MlpPolicy",
                "params": {"learning_rate": LEARNING_RATE, "gamma": GAMMA, "seed": RANDOM_SEED, "policy_kwargs": policy_kwargs,
                           "buffer_size": BUFFER_SIZE_OFF_POLICY, "batch_size": BATCH_SIZE_OFF_POLICY}},
        "TD3": {"class": TD3, "env": train_env_td3, "policy": TD3MlpPolicy, # Use imported MlpPolicy directly for TD3
                "params": {"learning_rate": LEARNING_RATE, "gamma": GAMMA, "seed": RANDOM_SEED, "policy_kwargs": policy_kwargs,
                           "buffer_size": BUFFER_SIZE_OFF_POLICY, "batch_size": BATCH_SIZE_OFF_POLICY}},
    }

    # --- OPTUNA HYPERPARAMETER TUNING (Example Structure) ---
    # To implement this, you would wrap the training loop in an Optuna study:
    #
    # import optuna
    #
    # def objective(trial):
    #     # 1. Suggest hyperparameters
    #     lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    #     net_arch_type = trial.suggest_categorical("net_arch_type", ["shallow", "deep"])
    #     if net_arch_type == "shallow":
    #         net_arch = [trial.suggest_int("n_units_l1", 64, 256), trial.suggest_int("n_units_l2", 32, 128)]
    #     else:
    #         net_arch = [trial.suggest_int("n_units_l1", 256, 1024), trial.suggest_int("n_units_l2", 128, 512), trial.suggest_int("n_units_l3", 64, 256)]
    #     policy_kwargs = {"net_arch": net_arch}
    #     # ... suggest other relevant params (gamma, batch_size, etc.) ...
    #
    #     # 2. Create model with suggested params (choose one algorithm or tune algorithm choice too)
    #     model = PPO("MlpPolicy", train_env_ppo, learning_rate=lr, policy_kwargs=policy_kwargs, verbose=0, ...)
    #
    #     # 3. Train the model (maybe for fewer steps during tuning)
    #     model.learn(total_timesteps=100000) # Shorter training for HPO
    #
    #     # 4. Evaluate the model on the validation set
    #     eval_env = make_env(0, seed=RANDOM_SEED+100, env_indices=valid_val_indices, is_training=False)() # Create single eval env
    #     avg_hits = evaluate_model_on_validation(eval_env, model, num_val_episodes)
    #     eval_env.close()
    #
    #     # 5. Return the metric to optimize (e.g., average hits)
    #     return avg_hits
    #
    # study = optuna.create_study(direction="maximize") # Maximize hits
    # study.optimize(objective, n_trials=100) # Run 100 trials
    #
    # print("Best trial:", study.best_trial.params)
    # # After finding best params, train the final model with those params for full timesteps
    # --- END OPTUNA EXAMPLE ---


    # --- Regular Training Loop ---
    total_start_time = time.time()
    for name, config in model_configs.items():
        print(f"\nTraining {name} for {TRAINING_TIMESTEPS} timesteps...")
        start_time = time.time()
        model = config["class"](
            config["policy"],
            config["env"],
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            **config["params"]
        )
        # Consider adding callbacks for evaluation during training or early stopping
        # from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
        # eval_callback = EvalCallback(eval_env, ...)
        try:
            model.learn(
                total_timesteps=TRAINING_TIMESTEPS,
                log_interval=max(1, TRAINING_TIMESTEPS // 1000), # Log ~1000 times
                tb_log_name=name # Separate logs for each model
            )
            model_path = os.path.join(RESULTS_DIR, f"{name.lower()}_kl8_model")
            model.save(model_path)
            models[name] = model # Store the trained model instance
            print(f"{name} Training Time: {(time.time() - start_time) / 60:.2f} minutes. Model saved to {model_path}")
        except Exception as e:
            print(f"!!! ERROR training {name}: {e}")
            models[name] = None # Mark model as failed

    print(f"\nTotal Training Time: {(time.time() - total_start_time) / 3600:.2f} hours")

    # Close training environments
    for name, config in model_configs.items():
        try:
             if hasattr(config["env"], 'close'):
                 config["env"].close()
        except Exception as e:
            print(f"Warning: Error closing environment for {name}: {e}")

    # --- Load models back (optional, ensures saving/loading works) ---
    print("\nReloading trained models...")
    loaded_models = {}
    for name in model_configs.keys():
        if models.get(name): # Only load if training succeeded
            model_path = os.path.join(RESULTS_DIR, f"{name.lower()}_kl8_model")
            try:
                loaded_models[name] = model_configs[name]["class"].load(model_path)
                print(f"Successfully reloaded {name}.")
            except Exception as e:
                print(f"Error reloading {name} from {model_path}: {e}")
                loaded_models[name] = None # Mark as failed to load
        else:
            loaded_models[name] = None

    # ----------------------------
    # 5. Evaluate on VALIDATION Set & Calculate Weights
    # ----------------------------
    print("\n--- 5. Evaluating Models on Validation Set ---")

    # Ensure validation indices are valid w.r.t history requirements
    val_indices = df_val.index.tolist()
    valid_val_indices = [idx for idx in val_indices if idx >= max(WINDOW_SIZES)]
    if not valid_val_indices:
        raise ValueError("Validation set has no dates with enough historical data for feature calculation.")

    num_val_episodes = len(valid_val_indices)
    print(f"Evaluating on {num_val_episodes} validation days...")

    # --- Evaluation Function ---
    def evaluate_model_on_validation(eval_env, model, n_episodes):
        """Evaluates the model sequentially on the validation environment."""
        all_rewards = []
        all_hits_details = [] # Store detailed info
        try:
            obs = eval_env.reset() # Start from the beginning of the validation sequence
            for i in range(n_episodes):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                all_rewards.append(reward)
                all_hits_details.append(info)
                if done:
                    if i >= n_episodes - 1: # Check if we reached the end
                        break
                    # Reset is handled implicitly by is_training=False logic in env.reset()
                    # obs = eval_env.reset() # This would restart the sequence if called incorrectly
                else: # Should not happen in this setup
                    print(f"Warning: Episode not 'done' after step {i} during validation.")

        except Exception as e:
            print(f"Error during validation evaluation: {e}")
            # import traceback
            # traceback.print_exc()
            return 0, [] # Return 0 hits if evaluation fails

        avg_reward = np.mean(all_rewards) if all_rewards else 0
        return avg_reward, all_hits_details

    # --- Run Evaluation for each loaded model ---
    model_avg_hits = {}
    model_validation_details = {}

    # Create a single environment instance for sequential validation
    # Use a different seed for evaluation environment
    eval_env = make_env(0, seed=RANDOM_SEED + 5*NUM_CPU, env_indices=valid_val_indices, is_training=False)()

    for name, model in loaded_models.items():
        if model:
            print(f"Evaluating {name}...")
            # IMPORTANT: Reset the single eval_env before evaluating each model
            # to ensure they all start from the same point in the validation sequence.
            eval_env._internal_subset_iterator = 0 # Manually reset sequential iterator

            avg_hits, details = evaluate_model_on_validation(eval_env, model, num_val_episodes)
            model_avg_hits[name] = avg_hits
            model_validation_details[name] = details
            print(f"  {name} Avg Hits: {avg_hits:.4f}")
        else:
            print(f"Skipping evaluation for {name} (not loaded).")
            model_avg_hits[name] = 0.0
            model_validation_details[name] = []

    eval_env.close() # Close the single evaluation environment

    # --- Calculate Dynamic Ensemble Weights ---
    print("\nCalculating Ensemble Weights...")
    epsilon = 1e-7 # Small value to prevent division by zero and give tiny weight to zero-hit models
    total_weighted_score = 0
    model_weights = {}

    # Use temperature scaling: score = (hits + epsilon) ** temp
    scores = {name: (hits + epsilon)**ENSEMBLE_TEMP for name, hits in model_avg_hits.items() if hits >= 0} # Ensure non-negative hits
    total_score = sum(scores.values())

    if total_score > 0:
        model_weights = {name: score / total_score for name, score in scores.items()}
    else: # Fallback to equal weights if all scores are effectively zero
        num_valid_models = len([m for m in loaded_models.values() if m is not None])
        equal_weight = 1.0 / num_valid_models if num_valid_models > 0 else 0
        model_weights = {name: equal_weight for name in loaded_models if loaded_models[name] is not None}

    print("Validation Avg Hits & Calculated Ensemble Weights:")
    weights_sum = 0
    for name in sorted(model_avg_hits.keys()):
        hits = model_avg_hits.get(name, 0.0)
        weight = model_weights.get(name, 0.0)
        print(f"  - {name}: Avg Hits = {hits:.4f}, Weight = {weight:.4f}")
        weights_sum += weight
    print(f"Sum of weights: {weights_sum:.4f}") # Should be close to 1.0

    # ----------------------------
    # 6. Ensemble Prediction for Next Day
    # ----------------------------
    print("\n--- 6. Predicting Next Day with Ensemble ---")
    last_historical_date = df['Date'].max()
    prediction_date = last_historical_date + pd.Timedelta(days=1)
    print(f"Predicting for date: {prediction_date.date()}")

    # Create a single environment instance configured for prediction
    # Need all historical data indices to potentially calculate features
    pred_env = EnhancedLotteryEnv(df, df.index.tolist(), known_draws, scaler=scaler, is_training=False, seed=RANDOM_SEED + 6*NUM_CPU)

    try:
        # Find the index corresponding to the last historical date
        last_hist_index = df[df['Date'] == last_historical_date].index[0]
        # Reset the environment to the *last known date* to get the observation
        # that predicts the *next* day. The observation is based on data *before* prediction_date.
        # The internal feature calculation looks back from the date provided.
        obs_pred = pred_env._get_observation(prediction_date)
        print(f"Observation created for prediction date {prediction_date.date()}.")

        # Get actions (scores) from all valid models
        model_actions = {}
        for name, model in loaded_models.items():
            if model and name in model_weights and model_weights[name] > 0:
                action, _ = model.predict(obs_pred, deterministic=True)
                model_actions[name] = action
            else:
                print(f"Skipping prediction for {name} (model invalid or zero weight).")


        if not model_actions:
            print("Error: No valid models available for ensemble prediction.")
            final_pred_numbers = []
        else:
            # Calculate weighted ensemble action
            ensemble_action = np.zeros_like(next(iter(model_actions.values())))
            for name, action in model_actions.items():
                ensemble_action += model_weights[name] * action

            # Select top N numbers from ensemble scores
            noise = np.random.uniform(low=-1e-6, high=1e-6, size=ensemble_action.shape)
            top_indices = np.argsort(ensemble_action + noise)[-NUM_SELECTED_BALLS:]
            final_pred_numbers = sorted((top_indices + 1).tolist())

            print(f"Ensemble Prediction for {prediction_date.date()}: {final_pred_numbers}")

            # Save prediction
            pred_df = pd.DataFrame([{
                "Date": prediction_date.date(),
                **{f"Pred{j:02d}": final_pred_numbers[j - 1] for j in range(1, NUM_SELECTED_BALLS + 1)}
            }])
            pred_path = os.path.join(RESULTS_DIR, "prediction_next_day_ensemble.csv")
            pred_df.to_csv(pred_path, index=False)
            print(f"Prediction saved to {pred_path}")

    except IndexError:
         print(f"Error: Could not find last historical date {last_historical_date} in the dataframe.")
         final_pred_numbers = []
    except Exception as e:
        print(f"Error during prediction for {prediction_date.date()}: {e}")
        # import traceback
        # traceback.print_exc()
        final_pred_numbers = []

    pred_env.close()

    # ----------------------------
    # 7. Plot Validation Set Hit Trend (Ensemble)
    # ----------------------------
    print("\n--- 7. Plotting Validation Set Performance (Ensemble) ---")
    ensemble_hits_over_time = []
    dates_list = []

    # Re-create the eval env for plotting if needed, or reuse details if stored correctly
    plot_env = make_env(0, seed=RANDOM_SEED + 7*NUM_CPU, env_indices=valid_val_indices, is_training=False)()
    obs = plot_env.reset()

    if not loaded_models or not any(loaded_models.values()):
        print("Skipping validation plot: No models loaded.")
    else:
        for i in range(num_val_episodes):
            try:
                current_date = plot_env.current_date # Get date before stepping
                actual_set = plot_env.actual_set     # Get actual set before stepping

                # Get actions from all valid models for this observation
                step_model_actions = {}
                for name, model in loaded_models.items():
                     if model and name in model_weights and model_weights[name] > 0:
                         action, _ = model.predict(obs, deterministic=True)
                         step_model_actions[name] = action

                if not step_model_actions:
                    print(f"Warning: No valid model actions for date {current_date.date()}. Skipping hit calculation.")
                    # Need to step the environment anyway to proceed
                    # Use a dummy action (e.g., zeros) or action from first available model?
                    # Using zeros might be safer if no model is reliable here.
                    dummy_action = np.zeros(plot_env.action_space.shape)
                    obs, _, done, _ = plot_env.step(dummy_action)
                    continue # Skip hit calculation for this day

                # Calculate ensemble action for this step
                step_ensemble_action = np.zeros_like(next(iter(step_model_actions.values())))
                for name, action in step_model_actions.items():
                     step_ensemble_action += model_weights[name] * action

                # Get predicted set
                noise = plot_env.np_random.uniform(low=-1e-6, high=1e-6, size=step_ensemble_action.shape)
                top_indices = np.argsort(step_ensemble_action + noise)[-NUM_SELECTED_BALLS:]
                predicted_set = frozenset((top_indices + 1).tolist())

                # Calculate hits if actual set exists
                if actual_set:
                    hits = len(predicted_set.intersection(actual_set))
                    ensemble_hits_over_time.append(hits)
                    dates_list.append(current_date)
                else:
                    ensemble_hits_over_time.append(np.nan) # Mark missing actual data
                    dates_list.append(current_date)

                # Step the environment using the ensemble action (or a dummy action if needed)
                # The reward/info from this step isn't strictly needed for the plot, just the state transition
                obs, _, done, _ = plot_env.step(step_ensemble_action)

                if done and i >= num_val_episodes - 1:
                    break

            except Exception as e:
                print(f"Error during validation plot generation step {i}: {e}")
                # import traceback
                # traceback.print_exc()
                break # Stop plotting if error occurs

        plot_env.close()

        # --- Create Plot ---
        if ensemble_hits_over_time and dates_list:
            plt.figure(figsize=(15, 7))
            plt.plot(dates_list, ensemble_hits_over_time, marker='.', linestyle='-', markersize=4, label="Ensemble Hits per day")

            # Calculate average ignoring potential NaNs
            valid_hits = [h for h in ensemble_hits_over_time if not np.isnan(h)]
            if valid_hits:
                 avg_ensemble_hits = np.mean(valid_hits)
                 plt.axhline(y=avg_ensemble_hits, color='r', linestyle='--', label=f"Avg Hits = {avg_ensemble_hits:.3f}")

            plt.xlabel("Date")
            plt.ylabel(f"Hits (out of {NUM_SELECTED_BALLS})")
            plt.title(f"Validation Set ({df_val['Date'].min().date()} to {df_val['Date'].max().date()}) - Ensemble Prediction Hit Trend")
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.ylim(bottom=0) # Ensure y-axis starts at 0
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = os.path.join(RESULTS_DIR, "validation_hit_trend_ensemble.png")
            plt.savefig(plot_path)
            print(f"Validation hit trend plot saved to {plot_path}")
            plt.close()
        else:
            print("No hits recorded or dates available for validation set plot.")

    print("\n--- Script Finished ---")

# --- Main Execution Guard for Multiprocessing ---
if __name__ == "__main__":
    # Check if data file exists before starting
    if not os.path.exists(DATA_FILE):
        print(f"FATAL ERROR: Data file not found at {DATA_FILE}. Please check the path.")
    else:
        # Set TF logging level (optional, before importing TF indirectly via SB3)
        # 0 = all messages are logged (default)
        # 1 = INFO messages are filtered out
        # 2 = WARNING messages are filtered out
        # 3 = ERROR messages are filtered out
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Filter out INFO and WARNING

        main_training_and_evaluation()
