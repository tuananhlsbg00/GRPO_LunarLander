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

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = policy_class(obs_dim, act_dim, **policy_kwargs)
    policy.load_state_dict({k: v.cpu() for k, v in state_dict.items()})
    policy.eval()

    # Call reset() ONCE and pass the seed to it
    s, _ = env.reset(seed=seed)
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
# Advantage computation Functions
# ===========================================

def compute_process_advantages(episodes: List[Dict[str, list]], gamma: float) -> np.ndarray:
    """(Process Supervision) Per-step group normalization + discounted sum of normalized future rewards."""
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
    counts[counts == 0] = 1.0 # Avoid division by zero
    means = sums / counts
    var = ((R - means) * M) ** 2
    stds = np.sqrt(var.sum(axis=0) / counts)
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
    stdA = A[mask_vals].std() + 1e-8
    A_final = (A - meanA) / stdA
    return A_final * M


def compute_outcome_advantages(episodes: List[Dict[str, list]], gamma: float) -> np.ndarray:
    """(Outcome Supervision) Normalized total episode return, broadcast to all steps."""
    G = len(episodes)
    max_len = max(len(ep["r"]) for ep in episodes)
    
    total_rewards = np.zeros(G, dtype=np.float32)
    for i, ep in enumerate(episodes):
        total_rewards[i] = float(sum(ep["r"]))

    mean_r = total_rewards.mean()
    std_r = total_rewards.std() + 1e-8
    A_ep = (total_rewards - mean_r) / std_r  # Shape (G,)

    A_batch = np.zeros((G, max_len), dtype=np.float32)
    M = np.zeros((G, max_len), dtype=np.float32)

    for i, ep in enumerate(episodes):
        L = len(ep["r"])
        A_batch[i, :L] = A_ep[i]  # Assign same advantage to all steps
        M[i, :L] = 1.0

    mask_vals = M > 0
    meanA = A_batch[mask_vals].mean()
    stdA = A_batch[mask_vals].std() + 1e-8
    A_final_normalized = (A_batch - meanA) / stdA
    
    return A_final_normalized * M


def compute_hybrid_advantages(episodes: List[Dict[str, list]], 
                              gamma: float,
                              alpha: float = 1.0,
                              beta: float = 1.0,
                              clip_mag: Optional[float] = None) -> np.ndarray:
    """(Hybrid) Multiplies magnitudes of Process and Outcome adv, with logical AND for sign."""
    G = len(episodes)
    T_max = max(len(ep["r"]) for ep in episodes)
    R = np.zeros((G, T_max), dtype=np.float32)
    M = np.zeros((G, T_max), dtype=np.float32)
    ep_returns = np.zeros(G, dtype=np.float32)

    for i, ep in enumerate(episodes):
        L = len(ep["r"])
        R[i, :L] = np.asarray(ep["r"], dtype=np.float32)
        M[i, :L] = 1.0
        ep_returns[i] = float(np.sum(ep["r"]))

    # --- 1. Calculate Step Advantage (A_step) ---
    sums = (R * M).sum(axis=0)
    counts = M.sum(axis=0)
    counts[counts == 0] = 1.0
    means = sums / counts
    var = ((R - means) * M) ** 2
    stds = np.sqrt(var.sum(axis=0) / counts)
    stds[stds < 1e-8] = 1e-8
    normR = (R - means) / stds

    A_step = np.zeros_like(normR, dtype=np.float32)
    for i in range(G):
        run = 0.0
        for t in range(T_max - 1, -1, -1):
            if M[i, t] == 0:
                continue
            run = normR[i, t] + gamma * run
            A_step[i, t] = run

    # --- 2. Calculate Episodic Advantage (A_ep) ---
    ep_mean = ep_returns.mean()
    ep_std = ep_returns.std() + 1e-8
    A_ep = (ep_returns - ep_mean) / ep_std  # shape (G,)

    # --- 3. Combine with 'only-positive-if-both-positive' sign rule ---
    sign_mat = np.where(
        (A_ep[:, None] > 0.0) & (A_step > 0.0),
        1.0,
        -1.0
    ).astype(np.float32)

    mag = (np.abs(A_ep)[:, None] ** alpha) * (np.abs(A_step) ** beta)
    A_hybrid = sign_mat * mag
    A_hybrid *= M

    if clip_mag is not None:
        A_hybrid = np.clip(A_hybrid, -clip_mag, clip_mag)

    # --- 4. Final z-normalization ---
    valid = M > 0
    # meanA = A_hybrid[valid].mean()
    # stdA = A_hybrid[valid].std() + 1e-8
    # A_hybrid = (A_hybrid - meanA) / stdA

    return A_hybrid * M


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
# Refactored Config
# ===========================================

@dataclass
class GRPOConfig:
    env: Union[str, gym.Env] = "LunarLander-v3"
    
    # ✅ --- New Control Flag ---
    identical_G: bool = False  # If True, all G workers get the same seed
    
    # --- Training Params ---
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
    
    # --- KL Control ---
    target_kl: float = 0.015
    beta_kl: float = 0.02
    kl_adjust_up: float = 1.5
    kl_adjust_down: float = 1 / 1.5
    ref_update_freq: int = 1

    # --- Hybrid Advantage Params (used by HybridAdvGRPOTrainer) ---
    hybrid_alpha: float = 1.0
    hybrid_beta: float = 1.0
    hybrid_clip: Optional[float] = 10.0

    # --- Logging ---
    log_dir: str = "./runs/GRPO"
    seed: Optional[int] = None
    verbose: int = 1


# ===========================================
# Base GRPO Trainer Class
# ===========================================

class GRPOBaseTrainer:
    """
    Base class for GRPO trainers.
    Contains all shared logic for init, data collection, updates, and logging.
    Child classes must implement _compute_advantages() and set _trainer_name.
    """
    
    _trainer_name: str = "BaseGRPO" # Child classes should override this

    def __init__(self, policy: nn.Module,
                 config: Optional[GRPOConfig] = None,
                 device: Optional[str] = None):
        self.cfg = config or GRPOConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        base_seed = self.cfg.seed or int(time.time())
        self.np_random = np.random.default_rng(base_seed)

        self.env_spec, self.env_id_str = env_to_spec(self.cfg.env)

        probe_env = make_env_from_spec(self.env_spec)
        self.obs_dim = probe_env.observation_space.shape[0]
        self.act_dim = probe_env.action_space.n
        probe_env.close()

        self.policy_class = policy.__class__
        self.policy_kwargs = getattr(policy, "_init_kwargs", {}) if hasattr(policy, "_init_kwargs") else {}
        self.pi = self.policy_class(self.obs_dim, self.act_dim, **self.policy_kwargs).to(self.device)
        self.pi.load_state_dict(policy.state_dict())
        self.pi_ref = self.policy_class(self.obs_dim, self.act_dim, **self.policy_kwargs).to(self.device)
        self.pi_ref.load_state_dict(policy.state_dict())
        self.opt = optim.Adam(self.pi.parameters(), lr=self.cfg.lr)

        now = datetime.now()
        self.timestamp = f"{now.hour:02d}h{now.minute:02d}_{now.day:02d}{now.month:02d}{now.year}"
        self.run_dir = Path(self.cfg.log_dir) / f"grpo_{self._trainer_name.lower()}_{self.timestamp}"
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

    def _save_hparams(self):
        txt_path = self.run_dir / f"grpo_config_{self.timestamp}.txt"
        with open(txt_path, "w") as f:
            f.write(f"=== GRPO ({self._trainer_name}) Hyperparameters ===\n")
            for k, v in vars(self.cfg).items():
                if k != "env":
                    f.write(f"{k:20s}: {v}\n")
            f.write(f"{'env_id':20s}: {self.env_id_str}\n")
            f.write(f"{'device':20s}: {self.device}\n")
            f.write(f"{'run_dir':20s}: {self.run_dir}\n")
        print(f"📝 Saved hyperparameters to {txt_path}")

    def _collect_group(self, policy: nn.Module):
        state_dict_cpu = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        
        # ✅ --- Use identical_G flag to control seeding ---
        if self.cfg.identical_G:
            batch_seed = int(self.np_random.integers(2**31 - 1))
            seeds = [batch_seed] * self.cfg.G
        else:
            seeds = [int(s) for s in self.np_random.integers(2**31 - 1, size=self.cfg.G)]
        
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

    def _compute_advantages(self, episodes: List[Dict[str, list]]) -> np.ndarray:
        """Abstract method. Child classes must implement this."""
        raise NotImplementedError("Child class must implement _compute_advantages")

    def train(self, iters: int,
              eval_env: Optional[gym.Env] = None,
              eval_interval: int = 10,
              eval_episodes: int = 100,
              video_dir: Optional[str] = None,
              video_fps: int = 30,
              video_format: str = "mp4",
              video_episodes: int = 6):
        
        pbar_desc = f"GRPO-{self._trainer_name} ({self.timestamp})"
        pbar = tqdm(range(iters), desc=pbar_desc, leave=True)
        
        for it in pbar:
            t0 = time.time()
            episodes, ep_returns = self._collect_group(self.pi)
            avgR, stdR = float(np.mean(ep_returns)), float(np.std(ep_returns))
            self.global_episodes += len(episodes)
            self.global_steps += sum(len(ep["r"]) for ep in episodes)

            # --- Call the advantage function defined by the child class ---
            A = self._compute_advantages(episodes)
            
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
            video_path = (Path(video_dir) / f"{self._trainer_name.lower()}_G{self.cfg.G}_{self.timestamp}")
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
            txt_path = video_path / f"grpo_config_{self.timestamp}.txt"
            video_path.mkdir(parents=True, exist_ok=True)
            with open(txt_path, "w") as f:
                f.write(f"=== GRPO ({self._trainer_name}) Hyperparameters ===\n")
                for k, v in vars(self.cfg).items():
                    if k != "env":
                        f.write(f"{k:20s}: {v}\n")
                f.write(f"{'env_id':20s}: {self.env_id_str}\n")
                f.write(f"{'device':20s}: {self.device}\n")
                f.write(f"{'video_dir':20s}: {video_path}\n")
            print(f"📝 Saved video run params to {txt_path}")

        return self.pi


# ===========================================
# Child Trainer Implementations
# ===========================================

class PerStepAdvGRPOTrainer(GRPOBaseTrainer):
    """
    GRPO trainer using "Process Supervision" advantages (Image 4.1.3).
    Standard use assumes all G workers start from the same state.
    """
    _trainer_name = "Process"

    def __init__(self, policy: nn.Module,
                 config: Optional[GRPOConfig] = None,
                 device: Optional[str] = None):
        
        super().__init__(policy, config, device)
        
        # ✅ REMOVED: No longer forcing config.identical_G
        #    Instead, just print a warning if the user chooses a non-standard value.
        if not self.cfg.identical_G:
            print(f"\n{'='*20} WARNING {'='*20}")
            print(f"GRPO 'process' mode typically requires identical_G=True.")
            print(f"You have set identical_G=False. This will run G rollouts")
            print(f"from different states, which may destabilize 'process' advantages.")
            print(f"{'='*50}\n")

    def _compute_advantages(self, episodes: List[Dict[str, list]]) -> np.ndarray:
        return compute_process_advantages(episodes, gamma=self.cfg.gamma)


class PerEpAdvGRPOTrainer(GRPOBaseTrainer):
    """
    GRPO trainer using "Outcome Supervision" advantages (Image 4.1.2).
    Standard use assumes diverse starting states (identical_G=False).
    """
    _trainer_name = "Outcome"

    def __init__(self, policy: nn.Module,
                 config: Optional[GRPOConfig] = None,
                 device: Optional[str] = None):
        
        super().__init__(policy, config, device)
        
        # ✅ REMOVED: No longer forcing config.identical_G
        #    Instead, just print a warning if the user chooses a non-standard value.
        if self.cfg.identical_G:
            print(f"\n{'='*20} WARNING {'='*20}")
            print(f"GRPO 'outcome' mode typically requires identical_G=False.")
            print(f"You have set identical_G=True. This will run G rollouts")
            print(f"from the same state, which may reduce diversity and harm learning.")
            print(f"{'='*50}\n")

    def _compute_advantages(self, episodes: List[Dict[str, list]]) -> np.ndarray:
        return compute_outcome_advantages(episodes, gamma=self.cfg.gamma)


class HybridAdvGRPOTrainer(GRPOBaseTrainer):
    """
    GRPO trainer using your custom hybrid advantage.
    Standard use assumes diverse starting states (identical_G=False).
    """
    _trainer_name = "Hybrid"

    def __init__(self, policy: nn.Module,
                 config: Optional[GRPOConfig] = None,
                 device: Optional[str] = None):

        super().__init__(policy, config, device)
        
        # ✅ REMOVED: No longer forcing config.identical_G
        #    Instead, just print a warning if the user chooses a non-standard value.
        if self.cfg.identical_G:
            print(f"\n{'='*20} WARNING {'='*20}")
            print(f"GRPO 'hybrid' mode typically requires identical_G=False.")
            print(f"You have set identical_G=True. This will run G rollouts")
            print(f"from the same state, which may reduce diversity and harm learning.")
            print(f"{'='*50}\n")

    def _compute_advantages(self, episodes: List[Dict[str, list]]) -> np.ndarray:
        return compute_hybrid_advantages(
            episodes,
            gamma=self.cfg.gamma,
            alpha=self.cfg.hybrid_alpha,
            beta=self.cfg.hybrid_beta,
            clip_mag=self.cfg.hybrid_clip
        )



@dataclass
class TreeSearchGRPOConfig:
    env_spec = "LunarLander-v3"
    n_rollout_steps: int = 100 # N-step return to estimate Q(s,a)
    gamma: float = 0.99
    lr: float = 3e-4
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    G: int = 32      # Number of actions to sample at each step
    epochs: int = 4  # Optimization epochs per *timestep*
    minibatches: int = 4 # Minibatches for the G samples
    n_workers: Optional[int] = None
    target_kl: float = 0.015
    beta_kl: float = 0.02
    kl_adjust_up: float = 1.5
    kl_adjust_down: float = 1 / 1.5
    log_dir: str = "./runs/MCTS_GRPO"
    seed: Optional[int] = None
    verbose: int = 1
    ref_update_freq: int = 1 # In *policy updates* (timesteps)

class TreeSearchGRPOTrainer:
    """
    Monte Carlo Tree Search (MCTS) style GRPO Trainer.
    
    At each step of an episode, this trainer:
    1. Samples G actions from the policy.
    2. Runs G parallel N-step rollouts to estimate Q(s, a) for each action.
    3. Normalizes these Q-values to get advantages.
    4. Updates the policy on this (s, a_i, adv_i) batch.
    5. Takes the action with the highest estimated Q-value in the real env.
    
    This is an ONLINE, step-by-step algorithm and is FUNDAMENTALLY DIFFERENT
    from the batch-based GRPO trainers above. It requires a Stateful environment.
    """
    _trainer_name = "MCTS"

    def __init__(self, policy: nn.Module,
                 config: Optional[TreeSearchGRPOConfig] = None,
                 device: Optional[str] = None):
        
        self.cfg = config or MCTS_GRPOConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        base_seed = self.cfg.seed or int(time.time())
        self.np_random = np.random.default_rng(base_seed)
        # Master seed for env terrain
        self.master_seed = int(self.np_random.integers(2**31 - 1)) 

        self.env_spec, self.env_id_str = env_to_spec(self.cfg.env_spec)
        
        self.main_env = make_env_from_spec(self.env_spec)
        if not isinstance(self.main_env, StatefulLunarLander):
             raise TypeError(f"MCTS_GRPOTrainer requires a StatefulLunarLander, but got {type(self.main_env)}")
        self.main_env.reset(seed=self.master_seed) # Set terrain

        self.obs_dim = self.main_env.observation_space.shape[0]
        self.act_dim = self.main_env.action_space.n

        self.policy_class = policy.__class__
        self.policy_kwargs = getattr(policy, "_init_kwargs", {}) if hasattr(policy, "_init_kwargs") else {}
        self.pi = self.policy_class(self.obs_dim, self.act_dim, **self.policy_kwargs).to(self.device)
        self.pi.load_state_dict(policy.state_dict())
        self.pi_ref = self.policy_class(self.obs_dim, self.act_dim, **self.policy_kwargs).to(self.device)
        self.pi_ref.load_state_dict(policy.state_dict())
        self.opt = optim.Adam(self.pi.parameters(), lr=self.cfg.lr)

        now = datetime.now()
        self.timestamp = f"{now.hour:02d}h{now.minute:02d}_{now.day:02d}{now.month:02d}{now.year}"
        self.run_dir = Path(self.cfg.log_dir) / f"mcts_grpo_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.run_dir)
        self._save_hparams()

        self.global_steps = 0
        self.start_time = time.time()

        if self.cfg.n_workers is None:
            self.cfg.n_workers = min(self.cfg.G, os.cpu_count() or 1)
        
        if mp.get_start_method(allow_none=True) != "fork":
            try:
                mp.set_start_method("fork", force=True)
            except RuntimeError:
                pass
        
        # Pass the master seed to all workers so they create identical terrain
        init_args_list = [
            (i, self.env_spec, self.policy_class, self.policy_kwargs, self.master_seed) 
            for i in range(self.cfg.n_workers)
        ]
        self.pool = mp.Pool(processes=self.cfg.n_workers, 
                            initializer=_mcts_init_worker, 
                            initargs=[args for args in init_args_list])
        
    def _save_hparams(self):
        txt_path = self.run_dir / f"mcts_grpo_config_{self.timestamp}.txt"
        with open(txt_path, "w") as f:
            f.write(f"=== GRPO ({self._trainer_name}) Hyperparameters ===\n")
            for k, v in vars(self.cfg).items():
                if k != "env_spec":
                    f.write(f"{k:20s}: {v}\n")
            f.write(f"{'env_id':20s}: {self.env_id_str}\n")
            f.write(f"{'device':20s}: {self.device}\n")
            f.write(f"{'run_dir':20s}: {self.run_dir}\n")
        print(f"📝 Saved hyperparameters to {txt_path}")

    def _update_batch(self, S, A, LOGP_OLD, ADV):
        """Performs a GRPO update on a single batch of data."""
        S, A, LOGP_OLD, ADV = S.to(self.device), A.to(self.device), LOGP_OLD.to(self.device), ADV.to(self.device)
        
        total_clip = total_ent = total_kl = total_loss = 0.0
        n_updates = 0
        
        batch_size = max(1, len(S) // self.cfg.minibatches)
        
        for _ in range(self.cfg.epochs):
            idx = torch.randperm(len(S), device=self.device)
            for start in range(0, len(S), batch_size):
                mb = idx[start : start + batch_size]
                s, a, logp_old, adv = S[mb], A[mb], LOGP_OLD[mb], ADV[mb]

                dist = self.pi.dist(s)
                logp = dist.log_prob(a)
                ratio = torch.exp(logp - logp_old)
                
                L_clip = torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv
                ).mean()
                
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
                n_updates += 1

        k = max(n_updates, 1)
        return total_clip / k, total_kl / k, total_ent / k, total_loss / k

    def _adapt_beta(self, measured_kl: float):
        if measured_kl > self.cfg.target_kl * 1.5:
            self.cfg.beta_kl *= self.cfg.kl_adjust_up
        elif measured_kl < self.cfg.target_kl / 1.5:
            self.cfg.beta_kl *= self.cfg.kl_adjust_down
        self.cfg.beta_kl = float(np.clip(self.cfg.beta_kl, 1e-4, 1.0))
        
    def _log_tb(self, avgR, stdR, kl, L_clip, L_ent, total_loss, ep_len, it_time):
        s = self.global_steps
        self.writer.add_scalar("episode/reward_avg", avgR, s)
        self.writer.add_scalar("episode/reward_std", stdR, s)
        self.writer.add_scalar("episode/length", ep_len, s)
        self.writer.add_scalar("kl/value", kl, s)
        self.writer.add_scalar("kl/beta", self.cfg.beta_kl, s)
        self.writer.add_scalar("loss/clip", L_clip, s)
        self.writer.add_scalar("loss/entropy", L_ent, s)
        self.writer.add_scalar("loss/total", total_loss, s)
        self.writer.add_scalar("time/step_sec", it_time, s)

    def train(self, num_episodes: int, 
              video_dir: Optional[str] = None,
              video_fps: int = 30,
              video_format: str = "mp4",
              video_episodes: int = 6):
        """
        Main training loop. Runs for a fixed number of episodes.
        """
        pbar = tqdm(range(num_episodes), desc=f"MCTS-GRPO ({self.timestamp})", leave=True)
        
        all_episode_rewards = []
        
        for ep_idx in pbar:
            s_obs, _ = self.main_env.reset(seed=self.master_seed)
            ep_done = False
            ep_len = 0
            ep_ret = 0.0
            t_start_ep = time.time()
            
            # Store last metrics for logging
            L_kl, L_clip, L_ent, L_total = 0.0, 0.0, 0.0, 0.0

            while not ep_done:
                t_start_step = time.time()
                
                # 1. Get current state and sample G actions
                s_t_state = self.main_env.get_state()
                s_t_tensor = torch.tensor(s_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    dist = self.pi.dist(s_t_tensor)
                    actions = dist.sample((self.cfg.G,))
                    log_probs = dist.log_prob(actions)
                
                policy_state_cpu = {k: v.cpu() for k, v in self.pi.state_dict().items()}

                # 2. Farm out rollouts to workers
                worker_args = []
                for i in range(self.cfg.G):
                    worker_id = i % self.cfg.n_workers
                    worker_args.append((
                        worker_id, s_t_state, actions[i].item(), 
                        policy_state_cpu, self.cfg.n_rollout_steps, self.cfg.gamma
                    ))
                
                Returns = np.array(self.pool.starmap(_mcts_rollout_worker, worker_args))
                
                # 3. Calculate advantages
                Returns_t = torch.tensor(Returns, dtype=torch.float32)
                Adv = (Returns_t - Returns_t.mean()) / (Returns_t.std() + 1e-8)
                
                # 4. Perform policy update
                S_batch = s_t_tensor.repeat(self.cfg.G, 1)
                L_clip, L_kl, L_ent, L_total = self._update_batch(
                    S_batch, actions, log_probs, Adv
                )
                
                # 5. Adapt KL beta
                self._adapt_beta(L_kl)
                if (self.global_steps + 1) % self.cfg.ref_update_freq == 0:
                    self.pi_ref.load_state_dict(self.pi.state_dict())

                # 6. Take the best action in the main environment
                best_action_idx = np.argmax(Returns)
                best_action = actions[best_action_idx].item()
                
                s_obs, r, term, trunc, _ = self.main_env.step(best_action)
                ep_done = term or trunc
                ep_ret += r
                ep_len += 1
                self.global_steps += 1
                
                if self.cfg.verbose > 1:
                    print(f"  Step {ep_len}: R_mean={Returns.mean():.2f}, R_std={Returns.std():.2f}, Best_R={Returns.max():.2f}")
            
            # --- End of Episode Logging ---
            t_end_ep = time.time()
            all_episode_rewards.append(ep_ret)
            avgR = np.mean(all_episode_rewards[-50:]) # 50-episode moving avg
            
            self._log_tb(avgR, np.std(all_episode_rewards[-50:]), L_kl, L_clip, L_ent, L_total, ep_len, t_end_ep - t_start_ep)
            pbar.set_postfix(avgR=f"{avgR:.1f}", KL=f"{L_kl:.3f}", len=ep_len)
            
            if self.cfg.verbose > 0:
                 print(f"Ep {ep_idx} | Reward {ep_ret:7.2f} | AvgR(50) {avgR:7.2f} | Len {ep_len:4d} "
                       f"| KL {L_kl:.4f} (β={self.cfg.beta_kl:.4g}) "
                       f"| Lclip {L_clip:.4f} Lent {L_ent:.4f} "
                       f"| steps {self.global_steps} | time {t_end_ep - t_start_ep:.2f}s")

        print(f"✅ Training finished. Logs saved to {self.run_dir}")
        self.writer.flush()
        self.pool.close()
        self.pool.join()
        
        # --- Final video recording ---
        if video_dir and record_videos:
            print("\n🏁 Recording final policy videos...")
            video_path = (Path(video_dir) / f"{self._trainer_name}_G{self.cfg.G}_{self.timestamp}")
            record_videos(
                policy=self.pi,
                env=self.env_spec, 
                video_dir=video_path,
                episodes=video_episodes,
                fps=video_fps,
                device=self.device,
                out_format=video_format,
                T=1000 
            )
            # Save hparams
            txt_path = video_path / f"mcts_grpo_config_{self.timestamp}.txt"
            video_path.mkdir(parents=True, exist_ok=True)
            with open(txt_path, "w") as f:
                f.write(f"=== GRPO ({self._trainer_name}) Hyperparameters ===\n")
                for k, v in vars(self.cfg).items():
                    if k != "env_spec":
                        f.write(f"{k:20s}: {v}\n")
                f.write(f"{'env_id':20s}: {self.env_id_str}\n")  
                f.write(f"{'device':20s}: {self.device}\n")
                f.write(f"{'video_dir':20s}: {video_path}\n")
            print(f"📝 Saved video run params to {txt_path}")

        return self.pi