#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/30/25
# @Author  : luoolu
# @Github  : https://luoolu.github.io
# @Software: PyCharm
# @File    : Keno.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(csv_filename):
    """加载 CSV 数据文件，并返回 DataFrame."""
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"Error: The file '{csv_filename}' was not found.")
    df = pd.read_csv(csv_filename)
    return df

def validate_dataframe(df, k_cols):
    """验证 DataFrame 中是否存在所有必需的 Keno 列."""
    missing_cols = [col for col in k_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

def update_values_vectorized(number_values, drawn_numbers, learning_rate, decay_factor):
    """
    矢量化更新函数：
    - 对于本期出现的数字：按照公式 value += learning_rate*(1 - value) 增加权重。
    - 对于未出现的数字：乘以衰减因子 decay_factor。
    """
    # 对所有数字（1到NUM_KENO_BALLS）生成布尔 mask：若在 drawn_numbers 中则为 True
    mask = np.isin(np.arange(1, len(number_values)), drawn_numbers)
    current_values = number_values[1:]  # 忽略索引0
    updated_values = np.where(mask, current_values + learning_rate * (1 - current_values),
                                       current_values * decay_factor)
    number_values[1:] = updated_values

def process_data(df, k_cols, number_values, learning_rate, decay_factor):
    """
    对 DataFrame 中每一行数据执行 RL 更新：
    如果该行数据存在问题（如转换错误），则跳过该行。
    """
    for idx, row in df.iterrows():
        try:
            drawn_numbers = row[k_cols].astype(int).values
            update_values_vectorized(number_values, drawn_numbers, learning_rate, decay_factor)
        except Exception as e:
            print(f"Row {idx} skipped due to error: {e}")
            continue

def predict_numbers(number_values, num_predictions):
    """根据最终学习到的值，选取数值最大的 num_predictions 个号码."""
    # 对数字 1~NUM_KENO_BALLS 的学习值进行排序（忽略索引 0）
    sorted_indices = np.argsort(number_values[1:])
    top_indices = sorted_indices[-num_predictions:]
    predicted_numbers = sorted(list(top_indices + 1))  # 恢复到 1~NUM_KENO_BALLS 的编号
    return predicted_numbers

def plot_learned_values(number_values):
    """绘制所有 Keno 数字的学习值分布图."""
    plt.figure(figsize=(10, 6))
    x = np.arange(1, len(number_values))
    plt.bar(x, number_values[1:])
    plt.xlabel("Keno Numbers")
    plt.ylabel("Learned Value")
    plt.title("Learned Values Distribution for Keno Numbers")
    plt.show()

# --- 参数设置 ---
NUM_KENO_BALLS = 80
NUM_PREDICTIONS = 20
LEARNING_RATE = 0.05      # Alpha: 出现时的学习速率
DECAY_FACTOR = 0.995      # Gamma: 未出现时的衰减因子
CSV_FILENAME = '/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/data/kl8/kl8_2025-03-31.csv'
TARGET_DATE = '2025-04-01'  # 目标预测日期

# 定义 Keno 列（例如 k01, k02, ... , k20）
k_cols = [f'k{i:02d}' for i in range(1, NUM_PREDICTIONS + 1)]

# --- 主程序 ---
try:
    df = load_data(CSV_FILENAME)
    print(f"成功加载 '{CSV_FILENAME}'，共 {len(df)} 条记录。")
    validate_dataframe(df, k_cols)
except Exception as e:
    print(f"数据加载/验证错误: {e}")
    exit()

# 初始化数字的学习值（1-indexed，索引0未使用）
number_values = np.full(NUM_KENO_BALLS + 1, 0.5)

print("开始应用增强版的 RL 更新机制（矢量化操作）...")
process_data(df, k_cols, number_values, LEARNING_RATE, DECAY_FACTOR)
print("RL 更新过程完成。")

# --- 预测 ---
predicted_numbers = predict_numbers(number_values, NUM_PREDICTIONS)

print("\n--- Keno 预测 (增强版 RL 方法) ---")
print(f"预测日期: {TARGET_DATE}")
print("\n预测数字（基于学习到的权重）:")
print(predicted_numbers)
print(f"\n预测数字总数: {len(predicted_numbers)}")

print("\n--- 免责声明 ---")
print("Keno 是一种纯随机游戏，历史数据不能影响未来结果。")
print("本预测基于增强的 RL 模型与历史数据，仅供学习/演示用途；")
print("绝不能保证预测准确，也不应用于实际投注。")

# 可选：展示学习值分布图
plot_learned_values(number_values)
