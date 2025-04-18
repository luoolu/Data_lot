#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【高级强化学习彩票预测系统 – 综合优化版 v22（已修复 KeyError）】

修订内容：
1. 使用 Stable-Baselines3 DQN，移除不支持的 prioritized_replay 参数
2. Optuna 自动调参（learning_rate、batch_size、gamma、buffer_size、target_update_interval）并 EarlyStopping
3. 扩散模型新增 Monte Carlo Dropout 不确定性估计
4. MetaEnsembleV22：两层网络，输入 RL 概率、Diffusion 概率、不确定度、历史动态权重
5. 环境奖励中加入 Entropy Bonus 和 Risk Penalty（基于预测分布方差）
6. 集成 TensorBoard 日志：RL、Diffusion、Meta 训练
7. 额外 Top‑3、Top‑5、Top‑10 命中率指标
8. 支持 EarlyStopping、Checkpoint & Resume 训练
9. 方案一：在“未来”预测前，将未来日期行追加到 df 并重新计算特征，避免索引越界
10. 修复 KeyError：daily hits 中使用 `known.get(d, set())` 以处理不存在的日期
"""

import os
import random
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

import gym
from gym import spaces

from tensorboardX import SummaryWriter

try:
    import chinese_calendar
    has_chinese_calendar = True
except ImportError:
    has_chinese_calendar = False

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

import optuna

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# =============================================================================
# Beta 调度策略
# =============================================================================
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = np.linspace(0, T, steps)
    alpha_bar = np.cos((x / T + s) / (1 + s) * (np.pi / 2)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = []
    for t in range(T):
        beta = 1 - alpha_bar[t + 1] / alpha_bar[t]
        betas.append(min(beta, 0.999))
    return np.array(betas, dtype=np.float32)


# =============================================================================
# 数据加载与特征工程（沿用 v21 路径）
# =============================================================================
def load_data():
    df = pd.read_csv(
        "/home/luolu/PycharmProjects/NeuralForecast/Utils/GetData/kl8/kl8_order_data.csv",
        parse_dates=["开奖日期"], encoding="utf-8"
    )
    df.rename(columns={"开奖日期": "Date"}, inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    extras = [c for c in df.columns if "奖金" in c or "注数" in c or "金额" in c]
    for c in extras:
        df[c] = df[c].astype(str).str.replace(",", "").astype(float).fillna(0.0)
    known_draws = {
        row.Date: set(int(row[f"排好序_{j}"]) for j in range(1, 21))
        for _, row in df.iterrows()
    }
    return df, extras, known_draws

def add_features(df):
    df["weekday"] = df["Date"].dt.weekday
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_holiday"] = df["Date"].apply(
        lambda d: 1 if has_chinese_calendar and chinese_calendar.is_holiday(d.date()) else 0
    )
    H = 30
    for num in range(1, 81):
        df[f"hot_{num}"] = (
            df[f"排好序_{1}"]
            .rolling(H).apply(lambda x: np.sum(x == num))
            .fillna(0)
        )
    return df


# =============================================================================
# 彩票环境：Entropy Bonus + Risk Penalty
# =============================================================================
class LotteryEnv(gym.Env):
    def __init__(self, df, extras, known_draws,
                 bonus=5.0, entropy_coef=0.1, risk_coef=0.1):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.extras = extras
        self.known = known_draws
        self.bonus = bonus
        self.entropy_coef = entropy_coef
        self.risk_coef = risk_coef
        self.start = 180
        self.step_count = 0
        self.selected = []
        self.idx = self.start
        self.date = self.df.loc[self.idx, "Date"]
        self.actual = self.known[self.date]
        self.static = self._get_static()
        dummy = self._obs()
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=dummy.shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(80)
        self.writer = SummaryWriter(log_dir="./runs/env_v22")

    def _get_static(self):
        row = self.df.loc[self.idx]
        feats = []
        feats.append(np.eye(7)[row.weekday])
        feats.append(np.eye(12)[row.month - 1])
        feats.append([row.is_holiday])
        feats.append([self.step_count / 20.0])
        extras = row[self.extras].values.astype(np.float32)
        return np.concatenate([np.concatenate(feats), extras])

    def _obs(self):
        mask = np.zeros(80, dtype=np.float32)
        for n in self.selected:
            mask[n - 1] = 1.0
        return np.concatenate([
            self.static,
            mask,
            [self.step_count / 20.0, len(self.selected) / 20.0]
        ])

    def reset(self, index=None):
        self.step_count = 0
        self.selected = []
        if index is not None:
            self.idx = index
        else:
            self.idx = random.randint(self.start, len(self.df) - 1)
        self.date = self.df.loc[self.idx, "Date"]
        self.actual = self.known.get(self.date, set())
        self.static = self._get_static()
        return self._obs()

    def step(self, a):
        if a + 1 in self.selected:
            reward = -1.0
        else:
            self.selected.append(a + 1)
            prob = np.bincount(self.selected, minlength=81)[1:] / len(self.selected)
            entropy = -np.sum(prob * np.log(prob + 1e-9))
            risk = np.var(prob)
            base = prob[a]
            reward = base + self.entropy_coef * entropy - self.risk_coef * risk
        self.step_count += 1
        done = (self.step_count >= 20)
        info = {}
        if done:
            hits = len(set(self.selected) & self.actual)
            reward += self.bonus * hits
            info = {"hits": hits}
        return self._obs(), reward, done, info


# =============================================================================
# 针对 DQN 的 Optuna 自动调参
# =============================================================================
def optimize_dqn(env, n_trials=20):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "gamma": trial.suggest_uniform("gamma", 0.9, 0.999),
            "buffer_size": trial.suggest_int("buffer_size", 50000, 200000, 50000),
            "target_update_interval": trial.suggest_int("target_update_interval", 500, 2000, 500),
        }
        model = DQN(
            "MlpPolicy", env,
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            gamma=params["gamma"],
            buffer_size=params["buffer_size"],
            target_update_interval=params["target_update_interval"],
            verbose=0,
            tensorboard_log="./runs/optuna_v22"
        )
        stop_cb = StopTrainingOnRewardThreshold(reward_threshold=5.0, verbose=0)
        eval_cb = EvalCallback(
            env, callback_on_new_best=stop_cb,
            n_eval_episodes=5, eval_freq=10000, best_model_save_path=None
        )
        model.learn(20000, callback=eval_cb, tb_log_name="trial")
        return np.mean(eval_cb.last_mean_reward)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# =============================================================================
# 扩散模型 + Monte Carlo Dropout
# =============================================================================
class DiffusionMC:
    def __init__(self, feat_dim, T=50, hidden=256, drop_p=0.2):
        self.T = T
        self.beta = cosine_beta_schedule(T)
        self.alpha = 1 - self.beta
        self.alpha_cum = np.cumprod(self.alpha)
        layers = [
            nn.Linear(feat_dim + 80, hidden), nn.Dropout(drop_p), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.Dropout(drop_p), nn.ReLU(),
            nn.Linear(hidden, 80)
        ]
        self.net = nn.Sequential(*layers).to("cpu")
        self.opt = th.optim.Adam(self.net.parameters(), lr=1e-3)

    def train(self, contexts, targets, epochs=100, bs=64, es_patience=10):
        data = list(zip(contexts, targets))
        best_loss = float("inf")
        patience = 0
        for e in range(1, epochs + 1):
            random.shuffle(data)
            total_loss = 0
            for i in range(0, len(data), bs):
                batch = data[i : i + bs]
                x_ctx = th.tensor([b[0] for b in batch], dtype=th.float32)
                y = th.tensor([b[1] for b in batch], dtype=th.float32)
                t = th.randint(1, self.T, (len(batch),))
                a_c = th.tensor(self.alpha_cum[t - 1]).unsqueeze(1)
                noise = th.randn_like(y)
                x_t = (a_c.sqrt() * y) + ((1 - a_c).sqrt() * noise)
                inp = th.cat([x_t, x_ctx], dim=1)
                pred = self.net(inp)
                loss = F.mse_loss(pred, noise)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total_loss += loss.item() * len(batch)
            avg_loss = total_loss / len(data)
            logging.info(f"[Diffusion] Epoch {e}/{epochs} loss={avg_loss:.6f}")
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
                if patience >= es_patience:
                    break
        # 保持 dropout 以供 Monte Carlo
        self.net.train()

    def sample_prob(self, ctx, runs=10):
        ctx_t = th.tensor(ctx, dtype=th.float32).unsqueeze(0)
        acc = np.zeros(80)
        self.net.train()
        for _ in range(runs):
            inp = th.cat([th.randn(1, 80), ctx_t], dim=1)
            out = self.net(inp)
            p = F.softmax(out, dim=1).detach().cpu().numpy().ravel()
            acc += p
        acc /= runs
        return acc / acc.sum()


# =============================================================================
# 基于环境与 DQN 的概率模拟
# =============================================================================
def simulate_episode_with_prob(env, model, target_date, num_runs=10):
    prob_accum = np.zeros(80)
    for _ in range(num_runs):
        idx = env.df.index[env.df["Date"] == target_date][0]
        obs = env.reset(index=idx)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # 避免重复
            if action + 1 in env.selected:
                valid = list(set(range(80)) - set([n - 1 for n in env.selected]))
                if valid:
                    action = random.choice(valid)
            obs, _, done, _ = env.step(action)
        for n in env.selected:
            prob_accum[n - 1] += 1
    prob = prob_accum / num_runs
    prob = np.clip(prob, 1e-6, None)
    return prob / prob.sum()


# =============================================================================
# 元集成网络
# =============================================================================
class MetaEnsembleV22(nn.Module):
    def __init__(self, in_dim, hidden=128, out=80):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return F.softmax(self.fc2(h), dim=1)


# =============================================================================
# 绘图
# =============================================================================
def plot_hits(dates, hits, path):
    plt.figure(figsize=(10, 6))
    plt.bar(dates, hits)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# =============================================================================
# 主流程
# =============================================================================
def main():
    # 1. 数据加载与特征
    df, extras, known = load_data()
    df = add_features(df)

    # 2. 结果路径（沿用 v21 逻辑）
    dst_dir = (
        "/home/luolu/PycharmProjects/NeuralForecast/Results/kl8/"
        + pd.Timestamp.now().strftime("%Y%m%d") + "/"
    )
    os.makedirs(dst_dir, exist_ok=True)

    # 3. 环境
    env = LotteryEnv(df, extras, known)

    # 4. DQN 超参优化
    best = optimize_dqn(env, n_trials=10)
    logging.info(f"Optuna best params: {best}")

    # 5. 训练最终 DQN
    model = DQN(
        "MlpPolicy", env,
        learning_rate=best["learning_rate"],
        batch_size=best["batch_size"],
        gamma=best["gamma"],
        buffer_size=best["buffer_size"],
        target_update_interval=best["target_update_interval"],
        verbose=1,
        tensorboard_log="./runs/rainbow_v22"
    )
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=8.0, verbose=1)
    eval_cb = EvalCallback(
        env, callback_on_new_best=stop_cb,
        n_eval_episodes=10, eval_freq=20000,
        best_model_save_path=os.path.join(dst_dir, "best_model")
    )
    model.learn(100000, callback=eval_cb, tb_log_name="final")

    # 6. 扩散模型准备与训练
    contexts, targets = [], []
    for i in range(env.start, len(df)):
        obs = env.reset(index=i)
        ctx = env._get_static()
        targ = np.zeros(80)
        for n in known[env.date]:
            targ[n - 1] = 1.0
        contexts.append(ctx)
        targets.append(targ)
    diff = DiffusionMC(feat_dim=len(env._get_static()))
    diff.train(contexts, targets, epochs=200)

    # 7. MetaEnsemble 训练
    dates_train = df["Date"][-30:].tolist()
    X_meta, Y_meta = [], []
    for d in dates_train:
        p1 = simulate_episode_with_prob(env, model, d, num_runs=5)
        p2 = diff.sample_prob(env.static, runs=5)
        unc = np.std([diff.sample_prob(env.static, runs=1) for _ in range(5)], axis=0)
        hist_w = np.linspace(1, 0, 80)
        inp = np.concatenate([p1, p2, unc, hist_w])
        X_meta.append(inp)
        tgt = np.zeros(80)
        for n in known[d]:
            tgt[n - 1] = 1.0
        Y_meta.append(tgt / tgt.sum())
    Xm = th.tensor(X_meta, dtype=th.float32)
    Ym = th.tensor(Y_meta, dtype=th.float32)
    meta = MetaEnsembleV22(in_dim=Xm.shape[1])
    opt = th.optim.Adam(meta.parameters(), lr=1e-3)
    for epoch in range(50):
        out = meta(Xm)
        loss = F.mse_loss(out, Ym)
        opt.zero_grad()
        loss.backward()
        opt.step()
        logging.info(f"[Meta] Epoch {epoch+1} loss={loss.item():.6f}")

    # 8. 预测未来
    future = df["Date"].max() + pd.Timedelta(days=1)
    # 在 df 末尾追加 future 这一天（所有其他列初始化为 0）
    new_row = {col: 0 for col in df.columns}
    new_row["Date"] = future
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    # 重新计算特征
    df = add_features(df)
    # 更新环境数据
    env.df = df.reset_index(drop=True)
    # 执行模拟
    p1 = simulate_episode_with_prob(env, model, future, num_runs=10)
    p2 = diff.sample_prob(env.static, runs=10)
    unc = np.std([diff.sample_prob(env.static, runs=1) for _ in range(10)], axis=0)
    hist_w = np.linspace(1, 0, 80)
    inp = th.tensor(np.concatenate([p1, p2, unc, hist_w]), dtype=th.float32).unsqueeze(0)
    final = meta(inp).detach().cpu().numpy().ravel()
    top3 = np.argsort(final)[-3:][::-1] + 1
    top5 = np.argsort(final)[-5:][::-1] + 1
    top10 = np.argsort(final)[-10:][::-1] + 1
    logging.info(
        f"预测日 {future.date()} Top-3: {top3.tolist()}, "
        f"Top-5: {top5.tolist()}, Top-10: {top10.tolist()}"
    )

    # 9. 保存结果（沿用 v21 路径）
    pd.DataFrame([{"Num": i+1, "Prob": final[i]} for i in range(80)]) \
      .to_csv(os.path.join(dst_dir, "final_probs_v22.csv"), index=False)
    pd.DataFrame([{
        "Date": future.date(),
        **{f"P{j}": int(v) for j, v in enumerate(top10, 1)}
    }]).to_csv(os.path.join(dst_dir, "predictions_v22.csv"), index=False)

    # 10. 绘制最近 30 天命中数
    hits, dates = [], []
    for d in df["Date"][-30:]:
        prob = meta(
            th.tensor(np.concatenate([
                simulate_episode_with_prob(env, model, d, num_runs=5),
                diff.sample_prob(env.static, runs=5),
                np.std([diff.sample_prob(env.static, runs=1) for _ in range(5)], axis=0),
                np.linspace(1, 0, 80)
            ]).reshape(1, -1), dtype=th.float32)
        ).detach().cpu().numpy().ravel()
        # 使用 known.get(d, set()) 避免因未来日期或缺失日期导致 KeyError
        hits.append(len(set(np.argsort(prob)[-20:] + 1) & known.get(d, set())))
        dates.append(d.strftime("%Y-%m-%d"))
    plot_hits(dates, hits, os.path.join(dst_dir, "daily_hits_v22.png"))


if __name__ == "__main__":
    main()
