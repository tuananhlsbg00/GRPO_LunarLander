# <dense_scripts/GRPO/grpo.py>
import os, time, multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Type, Union
from pathlib import Path
import importlib

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dense_scripts.utils.tools import record_videos, evaluate_model, make_env_from_spec, env_to_spec
from tqdm.auto import tqdm

torch.set_num_threads(1)  # keep workers lean


# ===========================================
# Utilities to carry env specs across processes
# ===========================================

EnvSpec = Union[str, Tuple[Type[gym.Env], Dict[str, Any]]]


# ===========================================
# Rollout worker
# ===========================================

def _rollout_worker(env_spec: EnvSpec,
                    policy_class: Type[nn.Module],
                    policy_kwargs: Dict[str, Any],
                    state_dict: Dict[str, torch.Tensor],
                    max_steps: int,
                    gamma: float,
                    seed: Optional[int]) -> Tuple[Dict[str, list], float]:
    """Run a single rollout with a CPU policy copy built from given class + state_dict."""
    env = make_env_from_spec(env_spec)

    # ✅ --- REMOVE THIS ENTIRE BLOCK ---
    # if seed is not None:
    #     try:
    #         env.reset(seed=seed)  # <-- This is the bad, first call
    #     except TypeError:
    #         pass
    # --- END REMOVE ---

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = policy_class(obs_dim, act_dim, **policy_kwargs)
    policy.load_state_dict({k: v.cpu() for k, v in state_dict.items()})
    policy.eval()

    # ✅ --- MODIFY THIS LINE ---
    # s, _ = env.reset()
    
    # ✅ --- TO THIS ---
    # Call reset() ONCE and pass the seed to it
    s, _ = env.reset(seed=seed)
    # --- END MODIFY ---
    ep = {"s": [], "a": [], "logp": [], "r": [], "done": [], "s_next": []}

    for _ in range(max_steps):
        s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = policy.dist(s_t)
            a = dist.sample()
            logp = dist.log_prob(a).item()
        s2, r, terminated, truncated, _ = env.step(a.item())
        ep["s"].append(s)
        ep["a"].append(a.item())
        ep["logp"].append(logp)
        ep["r"].append(float(r))
        ep["done"].append(bool(terminated or truncated))
        ep["s_next"].append(s2)
        s = s2
        if terminated or truncated:
            break

    env.close()
    return ep, float(sum(ep["r"]))


# ===========================================
# Advantage computation
# ===========================================

def compute_process_advantages(episodes: List[Dict[str, list]], gamma: float) -> np.ndarray:
    """Per-step group normalization of rewards (DeepSeek-style) + discounted sum of normalized future rewards."""
    G = len(episodes)
    max_len = max(len(ep["r"]) for ep in episodes)
    R = np.zeros((G, max_len), dtype=np.float32)
    M = np.zeros((G, max_len), dtype=np.float32)

    for i, ep in enumerate(episodes):
        L = len(ep["r"])
        R[i, :L] = np.asarray(ep["r"], dtype=np.float32)
        M[i, :L] = 1.0

    sums = (R * M).sum(axis=0)
    counts = M.sum(axis=0)
    means = np.divide(sums, np.maximum(counts, 1.0), where=counts > 0)
    var = ((R - means) * M) ** 2
    stds = np.sqrt(np.divide(var.sum(axis=0), np.maximum(counts, 1.0), where=counts > 0))
    stds[stds < 1e-8] = 1e-8

    normR = (R - means) / stds

    A = np.zeros_like(normR)
    for i in range(G):
        running = 0.0
        for t in reversed(range(max_len)):
            if M[i, t] == 0:
                continue
            running = normR[i, t] + gamma * running
            A[i, t] = running

    mask_vals = M > 0
    meanA = A[mask_vals].mean()
    stdA = A[mask_vals].std()
    stdA = max(stdA, 1e-8)
    A = (A - meanA) / stdA
    return A


def flatten_batch(episodes: List[Dict[str, list]], A: np.ndarray):
    """Flatten (S, A, LOGP_OLD, ADV) aligned to per-step advantages."""
    S, ACT, LOGP, ADV = [], [], [], []
    for i, ep in enumerate(episodes):
        n = len(ep["s"])
        S.append(torch.tensor(ep["s"], dtype=torch.float32))
        ACT.append(torch.tensor(ep["a"], dtype=torch.int64))
        LOGP.append(torch.tensor(ep["logp"], dtype=torch.float32))
        ADV.append(torch.tensor(A[i, :n], dtype=torch.float32))
    return torch.cat(S), torch.cat(ACT), torch.cat(LOGP), torch.cat(ADV)


# ===========================================
# Config
# ===========================================

@dataclass
class GRPOConfig:
    env: Union[str, gym.Env] = "LunarLander-v3"   # ← single unified env param
    gamma: float = 0.99
    lr: float = 3e-4
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    G: int = 32
    T: int = 1024
    epochs: int = 8
    minibatches: int = 16
    n_workers: Optional[int] = None
    target_kl: float = 0.015
    beta_kl: float = 0.02
    kl_adjust_up: float = 1.5
    kl_adjust_down: float = 1 / 1.5
    log_dir: str = "./runs/GRPO"
    seed: Optional[int] = None
    verbose: int = 1
    ref_update_freq: int = 1


# ===========================================
# GRPO Trainer
# ===========================================

class PerStepAdvGRPOTrainer:
    """GRPO trainer using per-step advantages and PPO-style adaptive KL penalty."""

    def __init__(self, policy: nn.Module,
                 config: Optional[GRPOConfig] = None,
                 device: Optional[str] = None):
        self.cfg = config or GRPOConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- Build env spec (picklable) and friendly id ---
        self.env_spec, self.env_id_str = env_to_spec(self.cfg.env)

        # Create a master random number generator for the trainer
        base_seed = self.cfg.seed or int(time.time())
        self.np_random = np.random.default_rng(base_seed)

        # --- Probe spaces using a real instance (main process only) ---
        probe_env = make_env_from_spec(self.env_spec)
        self.obs_dim = probe_env.observation_space.shape[0]
        self.act_dim = probe_env.action_space.n
        probe_env.close()

        # --- Policy setup ---
        self.policy_class = policy.__class__
        self.policy_kwargs = getattr(policy, "_init_kwargs", {}) if hasattr(policy, "_init_kwargs") else {}
        self.pi = self.policy_class(self.obs_dim, self.act_dim, **self.policy_kwargs).to(self.device)
        self.pi.load_state_dict(policy.state_dict())
        self.pi_ref = self.policy_class(self.obs_dim, self.act_dim, **self.policy_kwargs).to(self.device)
        self.pi_ref.load_state_dict(policy.state_dict())
        self.opt = optim.Adam(self.pi.parameters(), lr=self.cfg.lr)

        # --- Logging ---
        now = datetime.now()
        self.timestamp = f"{now.hour:02d}h{now.minute:02d}_{now.day:02d}{now.month:02d}{now.year}"
        self.run_dir = Path(self.cfg.log_dir) / f"grpo_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.run_dir)
        self._save_hparams()

        self.global_steps = 0
        self.global_episodes = 0
        self.start_time = time.time()

        if self.cfg.n_workers is None:
            self.cfg.n_workers = min(self.cfg.G, os.cpu_count() or 1)
        if mp.get_start_method(allow_none=True) != "fork":
            try:
                mp.set_start_method("fork", force=True)
            except RuntimeError:
                pass

    # ------------------------------------
    def _save_hparams(self):
        txt_path = self.run_dir / f"grpo_{self.timestamp}.txt"
        with open(txt_path, "w") as f:
            f.write("=== GRPO Hyperparameters ===\n")
            for k, v in vars(self.cfg).items():
                if k != "env":
                    f.write(f"{k:20s}: {v}\n")
            f.write(f"{'env_id':20s}: {self.env_id_str}\n")
            f.write(f"{'device':20s}: {self.device}\n")
            f.write(f"{'run_dir':20s}: {self.run_dir}\n")
        print(f"📝 Saved hyperparameters to {txt_path}")

    # ------------------------------------
    def train(self, iters: int,
              eval_env: Optional[gym.Env] = None,
              eval_interval: int = 10,
              eval_episodes: int = 100,
              video_dir: Optional[str] = None,
              video_fps: int = 30,
              video_format: str = "mp4",
              video_episodes: int = 6):
        """
        Train GRPO with optional periodic evaluation and video recording.
        """
        pbar = tqdm(range(iters), desc=f"GRPO ({self.timestamp})", leave=True)
        for it in pbar:
            t0 = time.time()
            episodes, ep_returns = self._collect_group(self.pi)
            avgR, stdR = float(np.mean(ep_returns)), float(np.std(ep_returns))
            self.global_episodes += len(episodes)
            self.global_steps += sum(len(ep["r"]) for ep in episodes)

            A = compute_process_advantages(episodes, gamma=self.cfg.gamma)
            L_clip, L_kl, L_ent, total_loss, last_kl = self._update(episodes, A)
            self._adapt_beta(last_kl)

            if (it + 1) % self.cfg.ref_update_freq == 0:
                self.pi_ref.load_state_dict(self.pi.state_dict())

            it_time = time.time() - t0
            if self.cfg.verbose:
                msg = (f"Iter {it:04d} | avgR {avgR:7.2f} ±{stdR:6.2f} "
                       f"| KL {last_kl:.4f} (β={self.cfg.beta_kl:.4g}) "
                       f"| Lclip {L_clip:.4f} Lkl {L_kl:.4f} Lent {L_ent:.4f} "
                       f"| steps {self.global_steps} eps {self.global_episodes} "
                       f"| time {it_time:.2f}s")
                print(msg)
            pbar.set_postfix(avgR=f"{avgR:.1f}", KL=f"{last_kl:.3f}", beta=self.cfg.beta_kl, it_s=f"{it_time:.2f}")
            self._log_tb(avgR, stdR, last_kl, L_clip, L_kl, L_ent, total_loss, it_time)

            # --- periodic evaluation ---
            if (it + 1) % eval_interval == 0 and eval_env is not None:
                results = evaluate_model(self.pi,
                                        eval_env,
                                        n_episodes=eval_episodes,
                                        device=self.device,
                                        num_workers=24,         # or 24 if your CPU can handle it
                                        T=self.cfg.T,
                                        disable_tqdm=True)
                self.writer.add_scalar("eval/mean_reward", results["mean_reward"], self.global_steps)
                self.writer.add_scalar("eval/success_rate", results["success_rate"], self.global_steps)
                self.writer.add_scalar("eval/mean_velocity", results["mean_velocity"], self.global_steps)
                self.writer.add_scalar("eval/mean_distance", results["mean_distance"], self.global_steps)
                self.writer.add_scalar("eval/mean_legs", results["mean_legs"], self.global_steps)

        self.writer.flush()
        print(f"✅ Training finished. Logs saved to {self.run_dir}")

        # --- final evaluation ---
        if (eval_env is not None) and (iters % eval_interval != 0):
            print("\n🏁 Final evaluation after training:")
            evaluate_model(self.pi,
                            eval_env,
                            n_episodes=eval_episodes,
                            device=self.device,
                            num_workers=24,         # or 24 if your CPU can handle it
                            T=self.cfg.T,
                            disable_tqdm=True)

        # --- optional video recording ---
        if video_dir is not None:
            video_path = (Path(video_dir) / f"G{self.cfg.G}_gamma{self.cfg.gamma}_{self.timestamp}")
            print(f"🎥 Saving videos to {video_path}")
            record_videos(
                policy=self.pi,
                env=self.env_spec,  # record_videos handles str or spec via make_env_from_spec internally if needed
                video_dir=video_path,
                episodes=video_episodes,
                fps=video_fps,
                device=self.device,
                out_format=video_format,
                T=self.cfg.T
            )
            txt_path = video_path / f"grpo_{self.timestamp}.txt"
            video_path.mkdir(parents=True, exist_ok=True)
            with open(txt_path, "w") as f:
                f.write("=== GRPO Hyperparameters ===\n")
                for k, v in vars(self.cfg).items():
                    if k != "env":
                        f.write(f"{k:20s}: {v}\n")
                f.write(f"{'env_id':20s}: {self.env_id_str}\n")
                f.write(f"{'device':20s}: {self.device}\n")
                f.write(f"{'video_dir':20s}: {video_path}\n")
            print(f"📝 Saved video run params to {txt_path}")

        return self.pi

    # ------------------------------------
    def _collect_group(self, policy: nn.Module):
        state_dict_cpu = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        
        # Generate ONE seed for the entire batch
        batch_seed = int(self.np_random.integers(2**31 - 1))
        # Give all G workers the EXACT SAME seed
        seeds = [batch_seed] * self.cfg.G
        
        with mp.Pool(processes=self.cfg.n_workers) as pool:
            results = pool.starmap(
                _rollout_worker,
                [(self.env_spec, self.policy_class, self.policy_kwargs,
                  state_dict_cpu, self.cfg.T, self.cfg.gamma, seeds[i]) for i in range(self.cfg.G)]
            )
        episodes, ep_returns = zip(*results)
        return list(episodes), np.asarray(ep_returns, dtype=np.float32)

    def _update(self, episodes: List[Dict[str, list]], A: np.ndarray):
        S, ACT, LOGP_OLD, ADV = flatten_batch(episodes, A)
        device = self.device
        S, ACT, LOGP_OLD, ADV = S.to(device), ACT.to(device), LOGP_OLD.to(device), ADV.to(device)
        idx = torch.randperm(len(S), device=device)
        S, ACT, LOGP_OLD, ADV = S[idx], ACT[idx], LOGP_OLD[idx], ADV[idx]

        total_clip = total_ent = total_kl = total_loss = 0.0
        batches = list(torch.chunk(torch.arange(len(S), device=device), self.cfg.minibatches))

        for _ in range(self.cfg.epochs):
            for mb in batches:
                s, a, logp_old, adv = S[mb], ACT[mb], LOGP_OLD[mb], ADV[mb]
                dist = self.pi.dist(s)
                logp = dist.log_prob(a)
                ratio = torch.exp(logp - logp_old)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv
                L_clip = torch.min(unclipped, clipped).mean()
                ent = dist.entropy().mean()

                with torch.no_grad():
                    dist_ref = self.pi_ref.dist(s)
                dkl = kl_divergence(dist, dist_ref).mean()

                loss = -(L_clip - self.cfg.beta_kl * dkl + self.cfg.ent_coef * ent)

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                total_clip += float(L_clip.item())
                total_ent += float(ent.item())
                total_kl += float(dkl.item())
                total_loss += float(loss.item())

        k = max(self.cfg.epochs * len(batches), 1)
        return total_clip / k, total_kl / k, total_ent / k, total_loss / k, total_kl / k

    def _adapt_beta(self, measured_kl: float):
        if measured_kl > self.cfg.target_kl * 1.5:
            self.cfg.beta_kl *= self.cfg.kl_adjust_up
        elif measured_kl < self.cfg.target_kl / 1.5:
            self.cfg.beta_kl *= self.cfg.kl_adjust_down
        self.cfg.beta_kl = float(np.clip(self.cfg.beta_kl, 1e-4, 1.0))

    def _log_tb(self, avgR, stdR, kl, L_clip, L_kl, L_ent, total_loss, it_time):
        self.writer.add_scalar("reward/avg", avgR, self.global_steps)
        self.writer.add_scalar("reward/std", stdR, self.global_steps)
        self.writer.add_scalar("kl/value", kl, self.global_steps)
        self.writer.add_scalar("kl/beta", self.cfg.beta_kl, self.global_steps)
        self.writer.add_scalar("loss/clip", L_clip, self.global_steps)
        self.writer.add_scalar("loss/entropy", L_ent, self.global_steps)
        self.writer.add_scalar("loss/kl", L_kl, self.global_steps)
        self.writer.add_scalar("loss/total", total_loss, self.global_steps)
        self.writer.add_scalar("time/iter_sec", it_time, self.global_steps)
        self.writer.add_scalar("progress/episodes", self.global_episodes, self.global_steps)




# ==== add near your other imports / helpers ====
import numpy as np
import torch

# ------------------------------------------------
# Hybrid per-step × episodic advantages
# ------------------------------------------------
def compute_hybrid_advantages(
    episodes: List[Dict[str, list]],
    gamma: float,
    alpha: float = 1.0,   # weight/exponent on episodic term
    beta: float = 1.0,    # weight/exponent on per-step term
    clip_mag: Optional[float] = None,  # e.g. 10.0 to tame outliers
) -> np.ndarray:
    """
    Build A_hybrid[i,t] that is positive ONLY when BOTH
      (1) the episode-level advantage for trajectory i is positive, and
      (2) the future per-step return from t is positive.
    Otherwise it is negative. (Solves the 'double negative becomes positive' issue.)

    Steps:
      1) Pack rewards to (G, T_max) with a mask M.
      2) Z-normalize rewards across the group at each step (DeepSeek-style).
      3) Compute per-step future returns R_future[i,t] = sum_{k>=t} normR[i,k].
      4) Compute episodic advantage A_ep[i] by z-normalizing episode returns.
      5) Combine magnitudes multiplicatively (|A_ep|^alpha * |R_future|^beta)
         and set the sign = +1 iff A_ep>0 AND R_future>0 else -1.
      6) Optional clipping and final z-norm over all valid (i,t).

    Returns:
      A_hybrid: float32 array shape (G, T_max) with zeros where M==0.
    """
    G = len(episodes)
    T_max = max(len(ep["r"]) for ep in episodes)
    R = np.zeros((G, T_max), dtype=np.float32)
    M = np.zeros((G, T_max), dtype=np.float32)

    # 1) pack + mask
    ep_returns = np.zeros(G, dtype=np.float32)
    for i, ep in enumerate(episodes):
        L = len(ep["r"])
        R[i, :L] = np.asarray(ep["r"], dtype=np.float32)
        M[i, :L] = 1.0
        ep_returns[i] = float(np.sum(ep["r"]))

    # 2) per-time z-norm (DeepSeek-style)
    sums = (R * M).sum(axis=0)
    counts = M.sum(axis=0)
    means = np.divide(sums, np.maximum(counts, 1.0), where=counts > 0)
    var = ((R - means) * M) ** 2
    stds = np.sqrt(np.divide(var.sum(axis=0), np.maximum(counts, 1.0), where=counts > 0))
    stds[stds < 1e-8] = 1e-8
    normR = (R - means) / stds

    # 3) discounted sum of normalized future rewards (you can switch to discounted if desired)
    #    Here we keep the original process-supervision sum without extra gamma inside the sum,
    #    but you can uncomment the discounted version below if you prefer.
    R_future = np.zeros_like(normR, dtype=np.float32)
    for i in range(G):
        run = 0.0
        # (option A: plain cumulative sum)
        for t in range(T_max - 1, -1, -1):
            if M[i, t] == 0:
                continue
            run = normR[i, t] + gamma * run  # discounted future sum works well in control settings
            R_future[i, t] = run

        # (option B: no discount)
        # run = 0.0
        # for t in range(T_max - 1, -1, -1):
        #     if M[i, t] == 0:
        #         continue
        #     run = normR[i, t] + run
        #     R_future[i, t] = run

    # 4) episodic advantage (group z-norm of episode returns)
    ep_mean = ep_returns.mean()
    ep_std  = ep_returns.std() if ep_returns.std() > 1e-8 else 1e-8
    A_ep = (ep_returns - ep_mean) / ep_std  # shape (G,)

    # 5) combine with 'only-positive-if-both-positive' sign rule
    # sign = +1  iff (A_ep>0 and R_future>0) else -1
    sign_mat = np.where(
        (A_ep[:, None] > 0.0) & (R_future > 0.0),
        1.0,
        -1.0
    ).astype(np.float32)

    # Magnitude as product of absolute values with optional exponents
    mag = (np.abs(A_ep)[:, None] ** alpha) * (np.abs(R_future) ** beta)

    A_hybrid = sign_mat * mag

    # Zero out invalid steps
    A_hybrid *= M

    # Optional clipping (helps when rewards can explode)
    if clip_mag is not None:
        A_hybrid = np.clip(A_hybrid, -clip_mag, clip_mag)

    # Final z-normalization over valid entries (stabilizes updates)
    valid = M > 0
    meanA = A_hybrid[valid].mean() if np.any(valid) else 0.0
    stdA  = A_hybrid[valid].std()  if np.any(valid) else 1.0
    stdA  = max(stdA, 1e-8)
    A_hybrid = (A_hybrid - meanA) / stdA

    return A_hybrid.astype(np.float32)


# ------------------------------------------------
# Trainer that uses the hybrid advantages
# ------------------------------------------------
class HybridAdvGRPOTrainer(PerStepAdvGRPOTrainer):
    """
    GRPO variant that mixes per-episode and per-step signals:
      A_hybrid[i,t] is positive only when BOTH the episode-level advantage is positive
      AND the step-level future return is positive; otherwise it's negative.

    Use when you want process supervision to agree with the episode outcome and
    avoid the 'double negative => positive' pitfall.
    """

    def __init__(self, policy: nn.Module, config: Optional[GRPOConfig] = None,
                 device: Optional[str] = None,
                 alpha: float = 1.0, beta: float = 1.0, clip_mag: Optional[float] = None):
        super().__init__(policy, config, device)
        self.hybrid_alpha = alpha
        self.hybrid_beta  = beta
        self.hybrid_clip  = clip_mag

    def train(self,
          iters: int,
          eval_env: Optional[gym.Env] = None,
          eval_interval: int = 10,
          eval_episodes: int = 100,
          video_dir: Optional[str] = None,
          video_fps: int = 30,
          video_format: str = "mp4",
          video_episodes: int = 6):

        pbar = tqdm(range(iters), desc=f"GRPO-Hybrid ({self.timestamp})", leave=True)
        for it in pbar:
            t0 = time.time()
            episodes, ep_returns = self._collect_group(self.pi)
            avgR, stdR = float(np.mean(ep_returns)), float(np.std(ep_returns))
            self.global_episodes += len(episodes)
            self.global_steps += sum(len(ep["r"]) for ep in episodes)

            # <<< swap in hybrid >>>
            A = compute_hybrid_advantages(
                episodes, gamma=self.cfg.gamma,
                alpha=self.hybrid_alpha, beta=self.hybrid_beta, clip_mag=self.hybrid_clip
            )

            L_clip, L_kl, L_ent, total_loss, last_kl = self._update(episodes, A)
            self._adapt_beta(last_kl)

            if (it + 1) % self.cfg.ref_update_freq == 0:
                self.pi_ref.load_state_dict(self.pi.state_dict())

            it_time = time.time() - t0
            if self.cfg.verbose:
                print(
                    f"Iter {it:04d} | avgR {avgR:7.2f} ±{stdR:6.2f} "
                    f"| KL {last_kl:.4f} (β={self.cfg.beta_kl:.4g}) "
                    f"| Lclip {L_clip:.4f} Lkl {L_kl:.4f} Lent {L_ent:.4f} "
                    f"| steps {self.global_steps} eps {self.global_episodes} "
                    f"| time {it_time:.2f}s"
                )
            pbar.set_postfix(avgR=f"{avgR:.1f}", KL=f"{last_kl:.3f}",
                             beta=self.cfg.beta_kl, it_s=f"{it_time:.2f}")
            self._log_tb(avgR, stdR, last_kl, L_clip, L_kl, L_ent, total_loss, it_time)

            if (it + 1) % eval_interval == 0 and eval_env is not None:
                results = evaluate_model(
                    self.pi, eval_env, n_episodes=eval_episodes,
                    device=self.device, num_workers=24, T=self.cfg.T, disable_tqdm=True
                )
                self.writer.add_scalar("eval/mean_reward", results["mean_reward"], self.global_steps)
                self.writer.add_scalar("eval/success_rate", results["success_rate"], self.global_steps)
                self.writer.add_scalar("eval/mean_velocity", results["mean_velocity"], self.global_steps)
                self.writer.add_scalar("eval/mean_distance", results["mean_distance"], self.global_steps)
                self.writer.add_scalar("eval/mean_legs", results["mean_legs"], self.global_steps)

        self.writer.flush()
        print(f"✅ Training finished. Logs saved to {self.run_dir}")

        if (eval_env is not None) and (iters % eval_interval != 0):
            print("\n🏁 Final evaluation after training:")
            evaluate_model(self.pi, eval_env, n_episodes=eval_episodes,
                           device=self.device, num_workers=24, T=self.cfg.T, disable_tqdm=True)

        if video_dir is not None:
            video_path = (Path(video_dir) / f"G{self.cfg.G}_gamma{self.cfg.gamma}_{self.timestamp}")
            print(f"🎥 Saving videos to {video_path}")
            record_videos(self.pi, self.env_spec, video_path, video_episodes,
                          video_fps, self.device, video_format, self.cfg.T)
            txt_path = video_path / f"grpo_{self.timestamp}.txt"
            video_path.mkdir(parents=True, exist_ok=True)
            with open(txt_path, "w") as f:
                f.write("=== GRPO-Hybrid Hyperparameters ===\n")
                for k, v in vars(self.cfg).items():
                    if k != "env":
                        f.write(f"{k:20s}: {v}\n")
                f.write(f"{'env_id':20s}: {self.env_id_str}\n")
                f.write(f"{'device':20s}: {self.device}\n")
                f.write(f"{'alpha':20s}: {self.hybrid_alpha}\n")
                f.write(f"{'beta':20s}: {self.hybrid_beta}\n")
                f.write(f"{'clip_mag':20s}: {self.hybrid_clip}\n")
                f.write(f"{'video_dir':20s}: {video_path}\n")
            print(f"📝 Saved video run params to {txt_path}")

        return self.pi


# ===========================================
# NEW: Outcome Advantage computation
# ===========================================

def compute_outcome_advantages(episodes: List[Dict[str, list]], gamma: float) -> np.ndarray:
    """
    Computes advantage based on "Outcome Supervision" (Image 4.1.2).
    1. Calculates the total return for each of the G episodes.
    2. Standard-normalizes this (G,) vector of returns.
    3. Broadcasts the normalized episodic advantage to all timesteps
       in that episode.
    4. Applies a final batch normalization for PPO stability.
    """
    G = len(episodes)
    max_len = max(len(ep["r"]) for ep in episodes)
    
    # --- 1. Calculate total return for each episode ---
    total_rewards = np.zeros(G, dtype=np.float32)
    for i, ep in enumerate(episodes):
        total_rewards[i] = float(sum(ep["r"]))

    # --- 2. Standard-normalize the episodic returns ---
    # This is A_ep, the advantage for each episode
    mean_r = total_rewards.mean()
    std_r = total_rewards.std() + 1e-8
    A_ep = (total_rewards - mean_r) / std_r  # Shape (G,)

    # --- 3. Broadcast A_ep to all timesteps ---
    A_batch = np.zeros((G, max_len), dtype=np.float32)
    M = np.zeros((G, max_len), dtype=np.float32)

    for i, ep in enumerate(episodes):
        L = len(ep["r"])
        A_batch[i, :L] = A_ep[i]  # Assign same advantage to all steps
        M[i, :L] = 1.0

    # --- 4. Final batch normalization ---
    # This is crucial for PPO's loss stability.
    mask_vals = M > 0
    meanA = A_batch[mask_vals].mean()
    stdA = A_batch[mask_vals].std() + 1e-8
    A_final_normalized = (A_batch - meanA) / stdA
    
    return A_final_normalized * M


def flatten_batch(episodes: List[Dict[str, list]], A: np.ndarray):
    """Flatten (S, A, LOGP_OLD, ADV) aligned to per-step advantages."""
    S, ACT, LOGP, ADV = [], [], [], []
    for i, ep in enumerate(episodes):
        n = len(ep["s"])
        S.append(torch.tensor(ep["s"], dtype=torch.float32))
        ACT.append(torch.tensor(ep["a"], dtype=torch.int64))
        LOGP.append(torch.tensor(ep["logp"], dtype=torch.float32))
        ADV.append(torch.tensor(A[i, :n], dtype=torch.float32))
    return torch.cat(S), torch.cat(ACT), torch.cat(LOGP), torch.cat(ADV)


# ===========================================
# NEW: Per-Episode Advantage GRPO Trainer
# ===========================================

class PerEpAdvGRPOTrainer:
    """
    GRPO trainer variant using "Outcome Supervision" advantages.
    All steps in an episode receive the same advantage, which is the
    normalized total return of that episode.
    """

    def __init__(self, policy: nn.Module,
                 config: Optional[GRPOConfig] = None,
                 device: Optional[str] = None):
        self.cfg = config or GRPOConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- Build env spec (picklable) and friendly id ---
        self.env_spec, self.env_id_str = env_to_spec(self.cfg.env)

        # Create a master random number generator for the trainer
        base_seed = self.cfg.seed or int(time.time())
        self.np_random = np.random.default_rng(base_seed)

        # --- Probe spaces using a real instance (main process only) ---
        probe_env = make_env_from_spec(self.env_spec)
        self.obs_dim = probe_env.observation_space.shape[0]
        self.act_dim = probe_env.action_space.n
        probe_env.close()

        # --- Policy setup ---
        self.policy_class = policy.__class__
        self.policy_kwargs = getattr(policy, "_init_kwargs", {}) if hasattr(policy, "_init_kwargs") else {}
        self.pi = self.policy_class(self.obs_dim, self.act_dim, **self.policy_kwargs).to(self.device)
        self.pi.load_state_dict(policy.state_dict())
        self.pi_ref = self.policy_class(self.obs_dim, self.act_dim, **self.policy_kwargs).to(self.device)
        self.pi_ref.load_state_dict(policy.state_dict())
        self.opt = optim.Adam(self.pi.parameters(), lr=self.cfg.lr)

        # --- Logging ---
        now = datetime.now()
        self.timestamp = f"{now.hour:02d}h{now.minute:02d}_{now.day:02d}{now.month:02d}{now.year}"
        self.run_dir = Path(self.cfg.log_dir) / f"outcome_grpo_{self.timestamp}" # <-- Renamed
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.run_dir)
        self._save_hparams()

        self.global_steps = 0
        self.global_episodes = 0
        self.start_time = time.time()

        if self.cfg.n_workers is None:
            self.cfg.n_workers = min(self.cfg.G, os.cpu_count() or 1)
        if mp.get_start_method(allow_none=True) != "fork":
            try:
                mp.set_start_method("fork", force=True)
            except RuntimeError:
                pass

    # ------------------------------------
    def _save_hparams(self):
        txt_path = self.run_dir / f"outcome_grpo_{self.timestamp}.txt" # <-- Renamed
        with open(txt_path, "w") as f:
            f.write("=== Outcome GRPO Hyperparameters ===\n") # <-- Renamed
            for k, v in vars(self.cfg).items():
                if k != "env":
                    f.write(f"{k:20s}: {v}\n")
            f.write(f"{'env_id':20s}: {self.env_id_str}\n")
            f.write(f"{'device':20s}: {self.device}\n")
            f.write(f"{'run_dir':20s}: {self.run_dir}\n")
        print(f"📝 Saved hyperparameters to {txt_path}")

    # ------------------------------------
    def train(self, iters: int,
              eval_env: Optional[gym.Env] = None,
              eval_interval: int = 10,
              eval_episodes: int = 100,
              video_dir: Optional[str] = None,
              video_fps: int = 30,
              video_format: str = "mp4",
              video_episodes: int = 6):
        
        pbar = tqdm(range(iters), desc=f"OutcomeGRPO ({self.timestamp})", leave=True) # <-- Renamed
        for it in pbar:
            t0 = time.time()
            episodes, ep_returns = self._collect_group(self.pi)
            avgR, stdR = float(np.mean(ep_returns)), float(np.std(ep_returns))
            self.global_episodes += len(episodes)
            self.global_steps += sum(len(ep["r"]) for ep in episodes)

            # ✅ --- USE NEW ADVANTAGE FUNCTION ---
            A = compute_outcome_advantages(episodes, gamma=self.cfg.gamma)
            
            L_clip, L_kl, L_ent, total_loss, last_kl = self._update(episodes, A)
            self._adapt_beta(last_kl)

            if (it + 1) % self.cfg.ref_update_freq == 0:
                self.pi_ref.load_state_dict(self.pi.state_dict())

            it_time = time.time() - t0
            if self.cfg.verbose:
                msg = (f"Iter {it:04d} | avgR {avgR:7.2f} ±{stdR:6.2f} "
                       f"| KL {last_kl:.4f} (β={self.cfg.beta_kl:.4g}) "
                       f"| Lclip {L_clip:.4f} Lkl {L_kl:.4f} Lent {L_ent:.4f} "
                       f"| steps {self.global_steps} eps {self.global_episodes} "
                       f"| time {it_time:.2f}s")
                print(msg)
            pbar.set_postfix(avgR=f"{avgR:.1f}", KL=f"{last_kl:.3f}", beta=self.cfg.beta_kl, it_s=f"{it_time:.2f}")
            self._log_tb(avgR, stdR, last_kl, L_clip, L_kl, L_ent, total_loss, it_time)

            # --- periodic evaluation ---
            if (it + 1) % eval_interval == 0 and eval_env is not None:
                results = evaluate_model(self.pi,
                                         eval_env,
                                         n_episodes=eval_episodes,
                                         device=self.device,
                                         num_workers=self.cfg.n_workers,
                                         T=self.cfg.T,
                                         disable_tqdm=True)
                self.writer.add_scalar("eval/mean_reward", results["mean_reward"], self.global_steps)
                # Check if keys exist, as evaluate_model might not return them
                if "success_rate" in results:
                    self.writer.add_scalar("eval/success_rate", results["success_rate"], self.global_steps)
                if "mean_velocity" in results:
                    self.writer.add_scalar("eval/mean_velocity", results["mean_velocity"], self.global_steps)
                if "mean_distance" in results:
                    self.writer.add_scalar("eval/mean_distance", results["mean_distance"], self.global_steps)
                if "mean_legs" in results:
                    self.writer.add_scalar("eval/mean_legs", results["mean_legs"], self.global_steps)

        self.writer.flush()
        print(f"✅ Training finished. Logs saved to {self.run_dir}")

        # --- final evaluation ---
        if (eval_env is not None) and (iters % eval_interval != 0):
            print("\n🏁 Final evaluation after training:")
            evaluate_model(self.pi,
                           eval_env,
                           n_episodes=eval_episodes,
                           device=self.device,
                           num_workers=self.cfg.n_workers,
                           T=self.cfg.T,
                           disable_tqdm=True)

        # --- optional video recording ---
        if video_dir is not None:
            video_path = (Path(video_dir) / f"G{self.cfg.G}_gamma{self.cfg.gamma}_{self.timestamp}")
            print(f"🎥 Saving videos to {video_path}")
            record_videos(
                policy=self.pi,
                env=self.env_spec,
                video_dir=video_path,
                episodes=video_episodes,
                fps=video_fps,
                device=self.device,
                out_format=video_format,
                T=self.cfg.T
            )
            txt_path = video_path / f"outcome_grpo_{self.timestamp}.txt" # <-- Renamed
            video_path.mkdir(parents=True, exist_ok=True)
            with open(txt_path, "w") as f:
                f.write("=== Outcome GRPO Hyperparameters ===\n") # <-- Renamed
                for k, v in vars(self.cfg).items():
                    if k != "env":
                        f.write(f"{k:20s}: {v}\n")
                f.write(f"{'env_id':20s}: {self.env_id_str}\n")
                f.write(f"{'device':20s}: {self.device}\n")
                f.write(f"{'video_dir':20s}: {video_path}\n")
            print(f"📝 Saved video run params to {txt_path}")

        return self.pi

    # ------------------------------------
    def _collect_group(self, policy: nn.Module):
        state_dict_cpu = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        
        # ✅ --- REVERTED TO ORIGINAL ---
        # For Outcome Supervision, we WANT random seeds for each worker
        # to get a diverse batch of outcomes to normalize.
        # Generate ONE seed for the entire batch
        batch_seed = int(self.np_random.integers(2**31 - 1))
        # Give all G workers the EXACT SAME seed
        seeds = [batch_seed] * self.cfg.G
        
        with mp.Pool(processes=self.cfg.n_workers) as pool:
            results = pool.starmap(
                _rollout_worker,
                [(self.env_spec, self.policy_class, self.policy_kwargs,
                  state_dict_cpu, self.cfg.T, self.cfg.gamma, seeds[i]) for i in range(self.cfg.G)]
            )
        episodes, ep_returns = zip(*results)
        return list(episodes), np.asarray(ep_returns, dtype=np.float32)

    # (The _update, _adapt_beta, and _log_tb methods are identical to grpo.py)
    # ------------------------------------

    def _update(self, episodes: List[Dict[str, list]], A: np.ndarray):
        S, ACT, LOGP_OLD, ADV = flatten_batch(episodes, A)
        device = self.device
        S, ACT, LOGP_OLD, ADV = S.to(device), ACT.to(device), LOGP_OLD.to(device), ADV.to(device)
        idx = torch.randperm(len(S), device=device)
        S, ACT, LOGP_OLD, ADV = S[idx], ACT[idx], LOGP_OLD[idx], ADV[idx]

        total_clip = total_ent = total_kl = total_loss = 0.0
        batches = list(torch.chunk(torch.arange(len(S), device=device), self.cfg.minibatches))

        for _ in range(self.cfg.epochs):
            for mb in batches:
                s, a, logp_old, adv = S[mb], ACT[mb], LOGP_OLD[mb], ADV[mb]
                dist = self.pi.dist(s)
                logp = dist.log_prob(a)
                ratio = torch.exp(logp - logp_old)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv
                L_clip = torch.min(unclipped, clipped).mean()
                ent = dist.entropy().mean()

                with torch.no_grad():
                    dist_ref = self.pi_ref.dist(s)
                dkl = kl_divergence(dist, dist_ref).mean()

                loss = -(L_clip - self.cfg.beta_kl * dkl + self.cfg.ent_coef * ent)

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                total_clip += float(L_clip.item())
                total_ent += float(ent.item())
                total_kl += float(dkl.item())
                total_loss += float(loss.item())

        k = max(self.cfg.epochs * len(batches), 1)
        return total_clip / k, total_kl / k, total_ent / k, total_loss / k, total_kl / k

    def _adapt_beta(self, measured_kl: float):
        if measured_kl > self.cfg.target_kl * 1.5:
            self.cfg.beta_kl *= self.cfg.kl_adjust_up
        elif measured_kl < self.cfg.target_kl / 1.5:
            self.cfg.beta_kl *= self.cfg.kl_adjust_down
        self.cfg.beta_kl = float(np.clip(self.cfg.beta_kl, 1e-4, 1.0))

    def _log_tb(self, avgR, stdR, kl, L_clip, L_kl, L_ent, total_loss, it_time):
        self.writer.add_scalar("reward/avg", avgR, self.global_steps)
        self.writer.add_scalar("reward/std", stdR, self.global_steps)
        self.writer.add_scalar("kl/value", kl, self.global_steps)
        self.writer.add_scalar("kl/beta", self.cfg.beta_kl, self.global_steps)
        self.writer.add_scalar("loss/clip", L_clip, self.global_steps)
        self.writer.add_scalar("loss/entropy", L_ent, self.global_steps)
        self.writer.add_scalar("loss/kl", L_kl, self.global_steps)
        self.writer.add_scalar("loss/total", total_loss, self.global_steps)
        self.writer.add_scalar("time/iter_sec", it_time, self.global_steps)
        self.writer.add_scalar("progress/episodes", self.global_episodes, self.global_steps)