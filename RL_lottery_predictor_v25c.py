#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级强化学习彩票预测系统 – 全方位增强版 v25c (Gymnasium + VecNormalize 兼容)
--------------------------------------------------------------
• 双智能体：PPO + QR‑DQN，均使用 Transformer 特征提取器
• 观测归一化：使用 VecNormalize 对 obs 做标准化并裁剪
• 扩散模型：T=100 + Swish/LayerNorm + KL 退火
• GateNet 软门控融合 (RL×Diffusion×不确定度×历史频率)
• 丰富特征工程、全面评估、自动可视化
--------------------------------------------------------------
运行：
    python RL_lottery_predictor_v25c.py
"""
import os
import re
import random
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tensorboardX import SummaryWriter

try:
    import chinese_calendar
    HAS_CN_CAL = True
except ImportError:
    HAS_CN_CAL = False

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from sb3_contrib.qrdqn import QRDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sklearn.metrics import roc_auc_score, f1_score

# ------------------- 全局随机种子 -------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)

# ------------------- 路径 --------------------------
DATA_CSV    = "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/kl8/kl8_order_data.csv"
RESULT_ROOT = "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/"

# ===================================================
# Diffusion β‑schedule
# ===================================================
def cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    steps = T + 1
    x = np.linspace(0, T, steps, dtype=np.float32)
    alpha_bar = np.cos((x / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar /= alpha_bar[0]
    return np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 1e-5, 0.999).astype(np.float32)

# ===================================================
# 数据加载 & 特征工程
# ===================================================
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, parse_dates=["开奖日期"], encoding="utf-8")
    df.rename(columns={"开奖日期": "Date"}, inplace=True)
    df.sort_values("Date", inplace=True, ignore_index=True)
    return df

def build_known(df: pd.DataFrame) -> dict:
    return {
        row.Date: {int(row[f"排好序_{i}"]) for i in range(1, 21)}
        for _, row in df.iterrows()
    }

def add_features(df: pd.DataFrame):
    # 货币相关列基础
    monetary_base = [
        c for c in df.columns
        if re.search(r"(奖金|金额|注数)$", c) and not re.search(r"(_diff|_pct)$", c)
    ]

    # 时间周期特征
    df["weekday"] = df["Date"].dt.weekday
    df["month"]   = df["Date"].dt.month
    df["day"]     = df["Date"].dt.day
    for col, p in [("weekday", 7), ("month", 12), ("day", 31)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / p)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / p)

    # 节假日特征
    df["is_holiday"] = df["Date"].apply(
        lambda d: 1 if HAS_CN_CAL and chinese_calendar.is_holiday(d.date()) else 0
    )

    # 货币列差分及百分比
    for c in monetary_base:
        df[c] = df[c].astype(str).str.replace(",", "").astype(float).fillna(0.0)
        df[f"{c}_diff"] = df[c].diff().fillna(0.0)
        df[f"{c}_pct"]  = df[c].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # 奇偶、质数、大数计数
    PRIME = {
        2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79
    }
    def _stats(row):
        nums = [int(row[f"排好序_{i}"]) for i in range(1, 21)]
        return pd.Series({
            "cnt_odd":   sum(n % 2 for n in nums),
            "cnt_prime": sum(n in PRIME for n in nums),
            "cnt_big":   sum(n > 40 for n in nums),
        })
    df[["cnt_odd","cnt_prime","cnt_big"]] = df.apply(_stats, axis=1)

    # 滚动热度特征
    for W in [30, 90, 180]:
        for n in range(1, 81):
            mask = df[[f"排好序_{i}" for i in range(1,21)]].eq(n).any(axis=1).astype(int)
            df[f"hot{W}_{n}"] = mask.rolling(W).sum().fillna(0)

    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df.fillna(0.0, inplace=True)
    return df, monetary_base

# ===================================================
# Transformer 特征提取器
# ===================================================
class TransExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, emb=128, heads=4, layers=2):
        super().__init__(observation_space, features_dim=emb)
        in_dim = int(np.prod(observation_space.shape))
        self.proj = nn.Linear(in_dim, emb)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=heads, dim_feedforward=emb*2,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.ln = nn.LayerNorm(emb)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = self.proj(obs.view(obs.size(0), -1)).unsqueeze(1)
        return self.ln(self.encoder(x).squeeze(1))

# ===================================================
# 彩票环境（Gymnasium API）
# ===================================================
class LotteryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, df: pd.DataFrame, known: dict, monetary_cols: list,
        bonus: float = 7.0, ent_coef: float = 0.15,
        risk_coef: float = 0.05, w_freq: int = 50
    ):
        super().__init__()
        self.df        = df.reset_index(drop=True)
        self.known     = known
        self.m_cols    = monetary_cols
        self.bonus     = bonus
        self.ent_coef  = ent_coef
        self.risk_coef = risk_coef
        self.w_freq    = w_freq
        self.start     = 200

        # 计算静态特征维度
        self.static_dim = self._static(self.start).size
        obs_dim = self.static_dim + 80 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(80)
        self.seed(SEED)

        # 初始化
        self.idx      = self.start
        self.cur_step = 0
        self.selected = []
        self.actual   = self.known[self.df.loc[self.idx, "Date"]]
        self.static   = self._static(self.idx)

    def seed(self, seed: int = None):
        if seed is None:
            seed = random.randint(0, 2**32-1)
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def _static(self, idx: int) -> np.ndarray:
        row = self.df.loc[idx]
        feats = []
        feats.extend(row[[
            "weekday_sin","weekday_cos",
            "month_sin","month_cos",
            "day_sin","day_cos",
            "is_holiday",
            "cnt_odd","cnt_prime","cnt_big"
        ]].to_list())
        feats.extend(row[self.m_cols].to_list())
        feats.extend(row[[f"{c}_diff" for c in self.m_cols]].to_list())
        feats.extend(row[[f"{c}_pct"  for c in self.m_cols]].to_list())
        # 频率特征
        freq = np.zeros(80, np.float32)
        s = max(0, idx - self.w_freq)
        for j in range(s, idx):
            freq[[n-1 for n in self.known[self.df.loc[j,"Date"]]]] += 1
        feats.extend((freq / max(1, (idx-s)*20)).tolist())
        # 滚动热度
        for W in [30, 90, 180]:
            feats.extend(row[[f"hot{W}_{n}" for n in range(1,81)]].to_list())
        return np.asarray(feats, np.float32)

    def _obs(self) -> np.ndarray:
        mask = np.zeros(80, np.float32)
        if self.selected:
            mask[[n-1 for n in self.selected]] = 1.0
        return np.concatenate([
            self.static, mask,
            [self.cur_step/20.0, len(self.selected)/20.0]
        ], dtype=np.float32)

    def reset(self, *, seed: int=None, options: dict=None, index: int=None):
        if seed is not None:
            self.seed(seed)
        self.cur_step = 0
        self.selected = []
        self.idx = (
            self.rng.integers(self.start, len(self.df))
            if index is None else index
        )
        # 更新 actual（确保 known 包含对应日期）
        self.actual = self.known.get(self.df.loc[self.idx, "Date"], set())
        self.static = self._static(self.idx)
        return self._obs(), {}

    def step(self, action: int):
        number = action + 1
        if number in self.selected:
            reward = -1.0
        else:
            self.selected.append(number)
            pv = np.bincount(self.selected, minlength=81)[1:] / len(self.selected)
            entropy = -np.sum(pv * np.log(pv + 1e-9))
            reward = pv[action] + self.ent_coef*entropy - self.risk_coef*np.var(pv)

        self.cur_step += 1
        terminated = self.cur_step >= 20
        truncated = False
        info = {}
        if terminated:
            hits = len(set(self.selected) & self.actual)
            reward += self.bonus * hits
            info["hits"] = hits
        return self._obs(), reward, terminated, truncated, info

# ===================================================
# RL 训练函数
# ===================================================
def train_ppo(env, total_timesteps: int = 300_000) -> PPO:
    pk = dict(
        features_extractor_class=TransExtractor,
        features_extractor_kwargs=dict(emb=128, heads=4, layers=2),
        net_arch=[256, 256],
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=0,
        tensorboard_log="./runs/ppo",
        learning_rate=1e-4,
        gamma=0.99,
        n_steps=256,
        batch_size=256,
        gae_lambda=0.95,
        ent_coef=1e-3,
        max_grad_norm=0.5,
        target_kl=0.02,
        policy_kwargs=pk,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=EvalCallback(env, n_eval_episodes=12, eval_freq=40_000),
    )
    return model

def train_qrdqn(env, steps: int = 300_000) -> QRDQN:
    pk = dict(
        features_extractor_class=TransExtractor,
        features_extractor_kwargs=dict(emb=128, heads=4, layers=2),
        net_arch=[256,256],
    )
    model = QRDQN(
        "MlpPolicy", env,
        seed=SEED, verbose=0, tensorboard_log="./runs/qrdqn",
        learning_rate=5e-4, buffer_size=200_000, learning_starts=5_000,
        batch_size=512, gamma=0.99, tau=0.005,
        train_freq=(4, "step"), target_update_interval=10_000,
        exploration_fraction=0.08, exploration_final_eps=0.05,
        policy_kwargs=pk
    )
    model.learn(steps)
    return model

# ===================================================
# 扩散模型
# ===================================================
class Diffusion(nn.Module):
    def __init__(self, ctx_dim: int, T: int=100, hidden: int=512, drop: float=0.2):
        super().__init__()
        self.T = T
        self.betas     = th.tensor(cosine_beta_schedule(T)).to("cuda" if th.cuda.is_available() else "cpu")
        self.alphas    = 1 - self.betas
        self.alpha_cum = th.cumprod(self.alphas, 0)
        self.net = nn.Sequential(
            nn.Linear(ctx_dim+80, hidden),
            nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(drop),
            nn.Linear(hidden, 80),
        )
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.to(self.device)
        self.opt = th.optim.AdamW(self.parameters(), lr=1e-3)

    @th.no_grad()
    def sample(self, ctx: np.ndarray, runs: int=10) -> np.ndarray:
        ctx_t = th.tensor(ctx, dtype=th.float32, device=self.device).unsqueeze(0)
        acc = np.zeros(80)
        self.eval()
        for _ in range(runs):
            xt = th.randn(1, 80, device=self.device)
            for t in reversed(range(self.T)):
                a = self.alpha_cum[t]
                xt = (xt - (1-a).sqrt() * self.net(th.cat([xt, ctx_t], 1))) / a.sqrt()
                if t:
                    xt += self.betas[t-1].sqrt() * th.randn_like(xt)
            acc += F.softmax(xt, 1).cpu().numpy().ravel()
        prob = acc / runs
        return prob / prob.sum()

    def fit(self, ctxs, tgts, epochs: int=300, bs: int=128, warm: int=50):
        data = list(zip(ctxs, tgts))
        best, patience = float("inf"), 0
        for ep in range(1, epochs+1):
            random.shuffle(data)
            tot = 0.0
            for i in range(0, len(data), bs):
                batch = data[i:i+bs]
                ctx = th.tensor([c for c,_ in batch], dtype=th.float32, device=self.device)
                y   = th.tensor([t for _,t in batch], dtype=th.float32, device=self.device)
                t   = th.randint(1, self.T, (len(ctx),), device=self.device)
                a   = self.alpha_cum[t-1].unsqueeze(1)
                noise = th.randn_like(y)
                xt = a.sqrt()*y + (1-a).sqrt()*noise
                pred = self.net(th.cat([xt, ctx],1))
                mse  = F.mse_loss(pred, noise)
                loss = mse * min(1.0, ep/warm)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                tot += loss.item()*len(ctx)
            avg = tot/len(data)
            logging.info(f"[Diff] Ep{ep}/{epochs} loss={avg:.6f}")
            if avg < best - 1e-4:
                best, patience, best_state = avg, 0, self.state_dict()
            else:
                patience += 1
                if patience >= 15:
                    break
        self.load_state_dict(best_state)

# ===================================================
# 模拟概率 & GateNet
# ===================================================
def simulate_probs(env: LotteryEnv, model, date, runs: int=15) -> np.ndarray:
    acc = np.zeros(80)
    idx = env.df.index[env.df["Date"] == date][0]
    for _ in range(runs):
        obs, _ = env.reset(index=idx)
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            while act+1 in env.selected:
                act = env.rng.integers(0, 80)
            obs, _, done, _, _ = env.step(act)
        acc[[n-1 for n in env.selected]] += 1
    prob = np.clip(acc / runs, 1e-6, None)
    return prob / prob.sum()

class GateNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 80)
    def forward(self, x):
        return F.softmax(self.fc(x), 1)

# ===================================================
# 可视化工具
# ===================================================
def bar(dates, hits, path, title):
    plt.figure(figsize=(12,5))
    plt.bar(dates, hits)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def cum(dates, hits, path):
    plt.figure(figsize=(12,5))
    plt.plot(dates, np.cumsum(hits), marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Cumulative Hits")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ===================================================
# 主流程
# ===================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 1. 加载原始数据 & 特征
    df_raw = load_raw()
    known  = build_known(df_raw)
    df, mon_cols = add_features(df_raw.copy())

    # 2. 准备结果目录
    out_dir = os.path.join(RESULT_ROOT, pd.Timestamp.now().strftime("%Y%m%d"))
    os.makedirs(out_dir, exist_ok=True)

    # 3. 构建环境
    env_raw   = LotteryEnv(df, known, mon_cols)                  # 用于模拟 & 验证
    train_env = DummyVecEnv([lambda: LotteryEnv(df, known, mon_cols)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 4. 训练 PPO & QRDQN
    logging.info("== PPO Training ==")
    ppo   = train_ppo(train_env)
    logging.info("== QRDQN Training ==")
    qrdqn = train_qrdqn(train_env)

    # 5. 训练 Diffusion
    logging.info("== Diffusion Training ==")
    ctxs, tgts = [], []
    for i in range(env_raw.start, len(df)):
        _, _ = env_raw.reset(index=i)
        ctxs.append(env_raw.static.copy())
        v = np.zeros(80, np.float32)
        v[[n - 1 for n in env_raw.actual]] = 1.0
        tgts.append(v)
    diff = Diffusion(len(ctxs[0]))
    diff.fit(ctxs, tgts)

    # 6. 准备元模型训练数据
    dates = df["Date"].tolist()[env_raw.start:]
    X_meta, Y_meta = [], []
    for d in dates[:-60]:
        p1  = simulate_probs(env_raw, ppo,   d)
        p2  = simulate_probs(env_raw, qrdqn, d)
        p3  = diff.sample(env_raw._static(df.index[df["Date"] == d][0]), runs=15)
        unc = np.std(
            [diff.sample(env_raw._static(df.index[df["Date"] == d][0]), 1) for _ in range(8)],
            axis=0
        )
        hist = env_raw._static(df.index[df["Date"] == d][0])[-80:]
        X_meta.append(np.concatenate([p1, p2, p3, unc, hist]))
        y_vec = np.zeros(80)
        y_vec[[n - 1 for n in known[d]]] = 1 / 20.0
        Y_meta.append(y_vec)
    X_meta = th.tensor(X_meta, dtype=th.float32)
    Y_meta = th.tensor(Y_meta, dtype=th.float32)

    # 7. 训练 GateNet
    gate = GateNet(X_meta.shape[1])
    opt_g = th.optim.Adam(gate.parameters(), lr=2e-3)
    best_loss, patience = float("inf"), 0
    for ep in range(200):
        gate.train()
        pred = gate(X_meta)
        loss = F.mse_loss(pred, Y_meta)
        opt_g.zero_grad()
        loss.backward()
        opt_g.step()
        if loss.item() < best_loss - 1e-5:
            best_loss, patience, best_state = loss.item(), 0, gate.state_dict()
        else:
            patience += 1
            if patience >= 20:
                break
    gate.load_state_dict(best_state)
    logging.info(f"[GateNet] best MSE = {best_loss:.6f}")

    # 8. 验证 & 评估
    test_dates = dates[-60:]
    hit_dict, all_pred, all_true = {n: [] for n in [1, 3, 5, 10, 20]}, [], []
    for d in test_dates:
        p1  = simulate_probs(env_raw, ppo,   d)
        p2  = simulate_probs(env_raw, qrdqn, d)
        p3  = diff.sample(env_raw._static(df.index[df["Date"] == d][0]), runs=20)
        unc = np.std(
            [diff.sample(env_raw._static(df.index[df["Date"] == d][0]), 1) for _ in range(10)],
            axis=0
        )
        hist  = env_raw._static(df.index[df["Date"] == d][0])[-80:]
        inp   = th.tensor(np.concatenate([p1, p2, p3, unc, hist]), dtype=th.float32).unsqueeze(0)
        prob  = gate(inp).detach().numpy().ravel()
        prob /= prob.sum()

        order    = np.argsort(prob)
        true_set = known[d]
        for N in hit_dict:
            hit_dict[N].append(len(set(order[-N:] + 1) & true_set))

        y_true = np.zeros(80)
        y_true[[n - 1 for n in true_set]] = 1
        all_true.extend(y_true)
        all_pred.extend(prob)

    # 9. 指标计算
    metrics = {f"Top{N}_mean": np.mean(hit_dict[N]) for N in hit_dict}
    metrics.update({f"Top{N}_std":  np.std(hit_dict[N]) for N in hit_dict})
    arr_t, arr_p = np.array(all_true), np.array(all_pred)
    metrics["Brier"] = np.mean((arr_t - arr_p) ** 2)
    ece, bins = 0.0, np.linspace(0, 1, 11)
    for i in range(10):
        m = (arr_p >= bins[i]) & (arr_p < bins[i + 1])
        if m.sum():
            ece += (m.sum() / len(arr_p)) * abs(arr_p[m].mean() - arr_t[m].mean())
    metrics["ECE"] = ece
    try:
        metrics["ROC_AUC"] = roc_auc_score(arr_t, arr_p)
        metrics["F1"]      = f1_score(arr_t, (arr_p > 0.025).astype(int))
    except ValueError:
        metrics["ROC_AUC"] = metrics["F1"] = np.nan
    logging.info("=== Validation (Last 60) ===")
    for k, v in metrics.items():
        logging.info(f"{k}: {v:.4f}")

    # 10. 可视化保存
    bar(
        [d.strftime("%Y-%m-%d") for d in test_dates],
        hit_dict[5],
        os.path.join(out_dir, "daily_hits.png"),
        "Daily Top‑5 Hits"
    )
    cum(
        [d.strftime("%Y-%m-%d") for d in test_dates],
        hit_dict[5],
        os.path.join(out_dir, "cum_hits.png")
    )

    # 11. 未来一天预测
    future_date = df["Date"].max() + pd.Timedelta(days=1)
    known[future_date] = set()                           # 确保 known 包含 future_date
    stub = {c: 0 for c in df.columns}                    # 构造 stub 行并重新生成特征
    stub["Date"] = future_date
    df_future = pd.concat([df_raw, pd.DataFrame([stub])], ignore_index=True)
    df_future, _ = add_features(df_future)
    env_raw.df = df_future.reset_index(drop=True)

    ctx_fut  = env_raw._static(len(env_raw.df) - 1)
    p1f      = simulate_probs(env_raw, ppo,   future_date, runs=30)
    p2f      = simulate_probs(env_raw, qrdqn, future_date, runs=30)
    p3f      = diff.sample(ctx_fut, runs=30)
    uncf     = np.std([diff.sample(ctx_fut, 1) for _ in range(15)], axis=0)
    histf    = ctx_fut[-80:]
    prob_fut = gate(
        th.tensor(np.concatenate([p1f, p2f, p3f, uncf, histf]), dtype=th.float32).unsqueeze(0)
    ).detach().numpy().ravel()
    prob_fut /= prob_fut.sum()
    order = np.argsort(prob_fut)

    # —— 新增：生成并保存横向、逗号分隔的 Top‑列表 ——
    top5  = (order[-5:]  + 1)[::-1]
    top7  = (order[-7:]  + 1)[::-1]
    top10 = (order[-10:] + 1)[::-1]
    top20 = (order[-20:] + 1)[::-1]

    with open(os.path.join(out_dir, "final_probs.csv"), "w", encoding="utf-8") as f:
        f.write("Top5,"  + ",".join(map(str, top5))  + "\n")
        f.write("Top7,"  + ",".join(map(str, top7))  + "\n")
        f.write("Top10," + ",".join(map(str, top10)) + "\n")
        f.write("Top20," + ",".join(map(str, top20)) + "\n")

    logging.info(f"预测 {future_date.date()} Top‑20:")
    for n in top20:
        logging.info(f"{n:2d}: {prob_fut[n - 1]:.4f}")

    logging.info(f"结果已保存至 {out_dir}")


if __name__ == "__main__":
    main()
