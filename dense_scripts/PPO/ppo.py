## <dense_scripts/PPO/ppo.py>
import os
import time
import warnings
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Type, Union
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

try:
    # Assuming these are in your utils/tools.py file, as in grpo.py
    from dense_scripts.utils.tools import (
        record_videos,
        evaluate_model,
        make_env_from_spec,
        env_to_spec,
    )
except Exception:
    record_videos = None
    evaluate_model = None
    make_env_from_spec = None
    env_to_spec = None

torch.set_num_threads(1)
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")

# ======================================================
# Utility
# ======================================================

# Copied from grpo.py
EnvSpec = Union[str, Tuple[Type[gym.Env], Dict[str, Any]]]


def _infer_value_net_kwargs(policy_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    for k in ("hidden", "hidden_size", "hid", "width"):
        if k in policy_kwargs:
            return {"hidden": int(policy_kwargs[k])}
    return {"hidden": 128}


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ======================================================
# Worker: collect exactly T steps (SB3 semantics)
# ======================================================


def _rollout_worker(args):
    (
        env_spec,  # <-- MODIFIED
        policy_class,
        policy_kwargs,
        policy_state,
        value_state,
        T,
        gamma,
        lam,
        seed,
    ) = args

    # ✅ MODIFIED: Use make_env_from_spec
    if make_env_from_spec is None:
        raise ImportError("Could not import make_env_from_spec from utils.tools")
    env = make_env_from_spec(env_spec)

    if seed is not None:
        try:
            env.reset(seed=seed)
        except TypeError:
            pass

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    pi = policy_class(obs_dim, act_dim, **policy_kwargs)
    pi.load_state_dict({k: v.cpu() for k, v in policy_state.items()})
    pi.eval()

    vnet = ValueNet(obs_dim, **_infer_value_net_kwargs(policy_kwargs))
    vnet.load_state_dict({k: v.cpu() for k, v in value_state.items()})
    vnet.eval()

    obs, _ = env.reset()
    S, A, LOGP, R, V, DONE_TERM, TRUNC, NEXT_OBS = [], [], [], [], [], [], [], []
    ep_returns, ep_ret = [], 0.0
    steps = 0

    while steps < T:
        s_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = pi.dist(s_t)
            a = dist.sample()
            logp = dist.log_prob(a).item()
            v = vnet(s_t).item()
        obs2, r, term, trunc, _ = env.step(int(a.item()))
        S.append(obs)
        A.append(int(a))
        LOGP.append(float(logp))
        R.append(float(r))
        V.append(float(v))
        DONE_TERM.append(bool(term))
        TRUNC.append(bool(trunc))
        NEXT_OBS.append(obs2)
        ep_ret += r
        steps += 1
        if term or trunc:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
        else:
            obs = obs2

    # TimeLimit bootstrap
    with torch.no_grad():
        if any(TRUNC):
            next_values = (
                vnet(torch.tensor(np.array(NEXT_OBS), dtype=torch.float32))
                .cpu()
                .numpy()
            )
        else:
            next_values = np.zeros_like(R)
    for i, trunc in enumerate(TRUNC):
        if trunc:
            R[i] += gamma * next_values[i]

    # GAE (done mask uses terminated only)
    rewards, values = np.asarray(R), np.asarray(V)
    dones_term = np.asarray(DONE_TERM)
    adv = np.zeros_like(rewards)
    with torch.no_grad():
        last_v = vnet(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
    lastgaelam = 0.0
    for t in reversed(range(T)):
        next_value = last_v if t == T - 1 else values[t + 1]
        nonterminal = 0.0 if dones_term[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = values + adv

    batch = dict(
        s=np.array(S, np.float32),
        a=np.array(A, np.int64),
        logp=np.array(LOGP, np.float32),
        adv=adv.astype(np.float32),
        ret=ret.astype(np.float32),
        val_old=values.astype(np.float32),
    )
    stats = dict(episodes=len(ep_returns), returns=ep_returns)
    env.close()
    return batch, stats


# ======================================================
# Config
# ======================================================


@dataclass
class PPOConfig:
    # ✅ MODIFIED: Accept env ID string or env object
    env: Union[str, gym.Env] = "LunarLander-v3"
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    clip_eps: float = 0.2
    clip_range_vf: Optional[float] = None
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    G: int = 32
    T: int = 1024
    epochs: int = 10
    minibatches: int = 32
    n_workers: Optional[int] = None
    target_kl: Optional[float] = 0.03
    normalize_advantage: bool = True
    log_dir: str = "./runs/PPO"
    seed: Optional[int] = None
    verbose: int = 1


# ======================================================
# PPO Trainer
# ======================================================


class PPOTrainer:
    def __init__(
        self,
        policy: nn.Module,
        config: Optional[PPOConfig] = None,
        device: Optional[str] = None,
    ):
        self.cfg = config or PPOConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        base_seed = self.cfg.seed or int(time.time())
        np.random.seed(base_seed)
        torch.manual_seed(base_seed)

        now = datetime.now()
        self.timestamp = (
            f"{now.hour:02d}h{now.minute:02d}_{now.day:02d}{now.month:02d}{now.year}"
        )
        self.run_dir = Path(self.cfg.log_dir) / f"ppo_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # ✅ MODIFIED: Use env_spec logic from GRPO
        if env_to_spec is None or make_env_from_spec is None:
            raise ImportError("Could not import env_spec functions from utils.tools")
        self.env_spec, self.env_id_str = env_to_spec(self.cfg.env)
        probe_env = make_env_from_spec(self.env_spec)
        self.obs_dim = probe_env.observation_space.shape[0]
        self.act_dim = probe_env.action_space.n
        probe_env.close()

        self.policy_class = policy.__class__
        self.policy_kwargs = (
            getattr(policy, "_init_kwargs", {})
            if hasattr(policy, "_init_kwargs")
            else {}
        )
        self.pi = self.policy_class(
            self.obs_dim, self.act_dim, **self.policy_kwargs
        ).to(self.device)
        self.pi.load_state_dict(policy.state_dict())
        self.v = ValueNet(
            self.obs_dim, **_infer_value_net_kwargs(self.policy_kwargs)
        ).to(self.device)
        self.opt = optim.Adam(
            list(self.pi.parameters()) + list(self.v.parameters()), lr=self.cfg.lr
        )

        self.writer = SummaryWriter(self.run_dir)
        self._save_hparams()

        self.global_steps = 0
        self.start_time = time.time()

        if self.cfg.n_workers is None:
            self.cfg.n_workers = min(self.cfg.G, os.cpu_count() or 1)

        # ✅ ADDED: Set fork method like GRPO for clean startup
        if mp.get_start_method(allow_none=True) != "fork":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass  # Can't set start method more than once

        self._pool = mp.Pool(processes=self.cfg.n_workers)

    def close(self):
        if hasattr(self, "_pool"):
            self._pool.close()
            self._pool.join()

    def train(
        self,
        iters: int,
        # ✅ ADDED: Evaluation args from GRPO
        eval_env: Optional[gym.Env] = None,
        eval_interval: int = 10,
        eval_episodes: int = 100,
        # ---
        video_dir: Optional[str] = None,
        video_fps: int = 30,
        video_format: str = "mp4",
        video_episodes: int = 6,
    ):
        pbar = tqdm(range(iters), desc=f"PPO ({self.timestamp})", leave=True)
        total_per_iter = self.cfg.G * self.cfg.T
        batch_size = max(1, total_per_iter // self.cfg.minibatches)

        for it in pbar:
            t0 = time.time()
            batch, stats = self._collect_batch()
            self.global_steps += total_per_iter
            logs = self._update(batch, batch_size)
            it_time = time.time() - t0

            if self.cfg.verbose:
                print(
                    f"Iter {it:03d} | avgR {stats['avgR']:.1f} ±{stats['stdR']:.1f} "
                    f"| KL~ {logs['approx_kl']:.4f} | clipfrac {logs['clipfrac']:.3f} "
                    f"| Lπ {logs['pg_loss']:.4f} Lv {logs['v_loss']:.2f} "
                    f"H {-logs['ent_loss']:.3f} | steps {self.global_steps} | {it_time:.2f}s"
                )

            pbar.set_postfix(
                avgR=f"{stats['avgR']:.0f}",
                KL=f"{logs['approx_kl']:.3f}",
                it_s=f"{it_time:.2f}",
            )
            self._log_tb(stats, logs, it_time)

            # ✅ ADDED: Periodic evaluation logic from GRPO
            if (it + 1) % eval_interval == 0 and eval_env is not None:
                if evaluate_model is None:
                    print("Skipping evaluation, 'evaluate_model' not imported.")
                    continue

                results = evaluate_model(
                    self.pi,  # PPO evaluates the policy (pi)
                    eval_env,
                    n_episodes=eval_episodes,
                    device=self.device,
                    num_workers=self.cfg.n_workers,
                    T=self.cfg.T,
                    disable_tqdm=True,
                )
                self.writer.add_scalar(
                    "eval/mean_reward", results["mean_reward"], self.global_steps
                )
                if "success_rate" in results:
                    self.writer.add_scalar(
                        "eval/success_rate",
                        results["success_rate"],
                        self.global_steps,
                    )
                if "mean_velocity" in results:
                    self.writer.add_scalar(
                        "eval/mean_velocity",
                        results["mean_velocity"],
                        self.global_steps,
                    )
                if "mean_distance" in results:
                    self.writer.add_scalar(
                        "eval/mean_distance",
                        results["mean_distance"],
                        self.global_steps,
                    )
                if "mean_legs" in results:
                    self.writer.add_scalar(
                        "eval/mean_legs", results["mean_legs"], self.global_steps
                    )

        print(f"✅ PPO training complete — logs at {self.run_dir}")
        self.writer.flush()

        # ✅ ADDED: Final evaluation logic from GRPO
        if (eval_env is not None) and (iters % eval_interval != 0):
            if evaluate_model is None:
                print("Skipping final evaluation, 'evaluate_model' not imported.")
            else:
                print("\n🏁 Final evaluation after training:")
                evaluate_model(
                    self.pi,
                    eval_env,
                    n_episodes=eval_episodes,
                    device=self.device,
                    num_workers=self.cfg.n_workers,
                    T=self.cfg.T,
                    disable_tqdm=True,
                )

        if video_dir and record_videos:
            self._record_videos(video_dir, video_episodes, video_fps, video_format)

        self.close()
        return self.pi, self.v

    # ======================================================
    # Internals
    # ======================================================

    def _collect_batch(self):
        state_cpu = {k: v.cpu() for k, v in self.pi.state_dict().items()}
        value_cpu = {k: v.cpu() for k, v in self.v.state_dict().items()}
        
        # ✅ MODIFIED: Pass self.env_spec instead of self.cfg.env_id
        args = [
            (
                self.env_spec,
                self.policy_class,
                self.policy_kwargs,
                state_cpu,
                value_cpu,
                self.cfg.T,
                self.cfg.gamma,
                self.cfg.lam,
                None,
            )
            for _ in range(self.cfg.G)
        ]
        results = self._pool.map(_rollout_worker, args)
        batches, stat_list = zip(*results)

        seg = {
            k: np.concatenate([b[k] for b in batches], axis=0)
            for k in ["s", "a", "logp", "adv", "ret", "val_old"]
        }

        all_returns = [r for st in stat_list for r in st["returns"]]
        avgR = float(np.mean(all_returns)) if all_returns else 0.0
        stdR = float(np.std(all_returns)) if all_returns else 0.0
        stats = {"avgR": avgR, "stdR": stdR}
        return seg, stats

    def _update(self, seg, batch_size):
        device = self.device
        S = torch.tensor(seg["s"], dtype=torch.float32, device=device)
        A = torch.tensor(seg["a"], dtype=torch.long, device=device)
        LOGP_OLD = torch.tensor(seg["logp"], dtype=torch.float32, device=device)
        ADV = torch.tensor(seg["adv"], dtype=torch.float32, device=device)
        RET = torch.tensor(seg["ret"], dtype=torch.float32, device=device)
        VAL_OLD = torch.tensor(seg["val_old"], dtype=torch.float32, device=device)
        n = S.size(0)
        ent_losses, pg_losses, v_losses, clipfracs, approx_kls = [], [], [], [], []

        for _ in range(self.cfg.epochs):
            idx = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                mb = idx[start : start + batch_size]
                s, a = S[mb], A[mb]
                logp_old, adv, ret, v_old = (
                    LOGP_OLD[mb],
                    ADV[mb],
                    RET[mb],
                    VAL_OLD[mb],
                )

                if self.cfg.normalize_advantage and adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                dist = self.pi.dist(s)
                logp = dist.log_prob(a)
                ratio = (logp - logp_old).exp()

                # Policy loss
                clip_eps = self.cfg.clip_eps
                policy_loss = -torch.min(
                    adv * ratio,
                    adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps),
                ).mean()

                # Entropy
                ent = dist.entropy().mean()
                entropy_loss = -ent

                # Value loss
                v_pred = self.v(s)
                if self.cfg.clip_range_vf:
                    v_pred_clipped = v_old + (v_pred - v_old).clamp(
                        -self.cfg.clip_range_vf, self.cfg.clip_range_vf
                    )
                    v_loss = 0.5 * torch.mean((ret - v_pred_clipped) ** 2)
                else:
                    v_loss = 0.5 * torch.mean((ret - v_pred) ** 2)

                loss = (
                    policy_loss
                    + self.cfg.ent_coef * entropy_loss
                    + self.cfg.vf_coef * v_loss
                )
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.pi.parameters()) + list(self.v.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.opt.step()

                with torch.no_grad():
                    log_ratio = logp - logp_old
                    approx_kl = torch.mean((log_ratio.exp() - 1) - log_ratio).item()
                    clipfrac = torch.mean(
                        (torch.abs(ratio - 1.0) > clip_eps).float()
                    ).item()
                ent_losses.append(entropy_loss.item())
                pg_losses.append(policy_loss.item())
                v_losses.append(v_loss.item())
                clipfracs.append(clipfrac)
                approx_kls.append(approx_kl)

                if self.cfg.target_kl and approx_kl > 1.5 * self.cfg.target_kl:
                    break
            if (
                self.cfg.target_kl
                and approx_kls
                and approx_kls[-1] > 1.5 * self.cfg.target_kl
            ):
                break

        return dict(
            ent_loss=np.mean(ent_losses),
            pg_loss=np.mean(pg_losses),
            v_loss=np.mean(v_losses),
            clipfrac=np.mean(clipfracs),
            approx_kl=np.mean(approx_kls),
        )

    # ======================================================
    # Logging
    # ======================================================

    def _log_tb(self, stats, logs, it_time):
        s = self.global_steps
        self.writer.add_scalar("reward/avg", stats["avgR"], s)
        self.writer.add_scalar("reward/std", stats["stdR"], s)
        self.writer.add_scalar("train/approx_kl", logs["approx_kl"], s)
        self.writer.add_scalar("loss/policy", logs["pg_loss"], s)
        self.writer.add_scalar("loss/value", logs["v_loss"], s)
        self.writer.add_scalar("loss/entropy", logs["ent_loss"], s)
        self.writer.add_scalar("train/clip_fraction", logs["clipfrac"], s)
        self.writer.add_scalar("time/iter_sec", it_time, s)

    def _record_videos(self, video_dir, episodes, fps, fmt):
        out_dir = Path(video_dir) / f"ppo_G{self.cfg.G}_γ{self.cfg.gamma}_{self.timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"🎥 Recording PPO evaluation videos to {out_dir}")

        if record_videos is None:
            print("⚠️ 'record_videos' utility not available; skipping video recording.")
            return

        record_videos(
            self.pi,
            self.env_spec,
            video_dir=out_dir,
            episodes=episodes,
            fps=fps,
            device=self.device,
            out_format=fmt,
            T=self.cfg.T,
        )

        # Save run metadata
        txt_path = out_dir / f"ppo_{self.timestamp}.txt"
        with open(txt_path, "w") as f:
            f.write("=== PPO Run Parameters (Video Export) ===\n")
            for k, v in vars(self.cfg).items():
                if k != "env":
                    f.write(f"{k:20s}: {v}\n")
            f.write(f"{'env_id':20s}: {self.env_id_str}\n")
            f.write(f"{'device':20s}: {self.device}\n")
            f.write(f"{'video_dir':20s}: {out_dir}\n")
        print(f"📝 Saved video parameters to {txt_path}")


    def _save_hparams(self):
        txt_path = self.run_dir / f"ppo_{self.timestamp}.txt"
        with open(txt_path, "w") as f:
            f.write("=== PPO Hyperparameters (GRPO-style log format) ===\n")
            for k, v in vars(self.cfg).items():
                if k != "env":  # env may be object
                    f.write(f"{k:20s}: {v}\n")
            f.write(f"{'env_id':20s}: {self.env_id_str}\n")
            f.write(f"{'device':20s}: {self.device}\n")
            f.write(f"{'timestamp':20s}: {self.timestamp}\n")
            f.write(f"{'run_dir':20s}: {self.run_dir}\n")
        print(f"📝 Saved hyperparameters to {txt_path}")
