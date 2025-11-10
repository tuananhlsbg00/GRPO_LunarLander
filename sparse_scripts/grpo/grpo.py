"""
Pure GRPO Implementation - 100% Faithful to DeepSeek-R1 Paper

This is a clean-slate implementation that does NOT inherit from Stable Baselines3.
Every aspect follows the paper exactly:
  ✓ Group sampling from SAME state (G trajectories per state)
  ✓ NO value function (policy-only network)
  ✓ Pure Monte Carlo returns (no bootstrapping)
  ✓ Group-relative advantages (no per-minibatch normalization)
  ✓ Direct KL penalty in loss
  ✓ Trajectory-level advantages (all steps get same advantage)

Algorithm from DeepSeek-R1 Section 4.1:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each iteration:
    1. π_ref ← π_θ (save reference policy)
    2. For each state s:
        a. Sample G trajectories from s using π_ref
        b. Compute returns {R_1, ..., R_G}
        c. Normalize: Â_i = (R_i - mean(R)) / std(R)
    3. Update π_θ by maximizing:
       J = E[min(r·Â, clip(r,1-ε,1+ε)·Â)] - β·D_KL[π_θ||π_ref] + α·H[π_θ]
"""

from typing import Any, Callable, Dict, List, Tuple, Optional, Union
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Categorical, Normal
import gymnasium as gym
from collections import deque


DEFAULT_GRPO_CONFIG: Dict[str, Any] = {
    "group_size": 32,
    "learning_rate": 2e-4,
    "gamma": 1.0,
    "beta": 0.04,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "n_epochs": 10,
    "batch_size": 64,
    "device": "cpu",
    "verbose": 0,
    "hidden_dims": [64, 64],
}


def default_grpo_config(**overrides: Any) -> Dict[str, Any]:
    """Return a copy of the GRPO defaults with optional overrides."""

    config = DEFAULT_GRPO_CONFIG.copy()
    config.update(overrides)
    if "hidden_dims" in config and config["hidden_dims"] is not None:
        config["hidden_dims"] = list(config["hidden_dims"])
    return config


class PolicyNetwork(nn.Module):
    """
    Policy-only network (NO value head!).
    
    For discrete actions: outputs action logits
    For continuous actions: outputs (mean, log_std)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],
        action_space_type: str = "discrete",
    ):
        super().__init__()
        
        self.action_space_type = action_space_type
        self._init_kwargs = {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dims": list(hidden_dims) if hidden_dims is not None else None,
            "action_space_type": action_space_type,
        }
        
        # Shared feature extractor
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        self.features = nn.Sequential(*layers)
        
        # Policy head (NO value head!)
        if action_space_type == "discrete":
            self.action_logits = nn.Linear(prev_dim, action_dim)
        elif action_space_type == "continuous":
            self.action_mean = nn.Linear(prev_dim, action_dim)
            self.action_log_std = nn.Parameter(th.zeros(action_dim))
        else:
            raise ValueError(f"Unknown action space type: {action_space_type}")
    
    def forward(self, obs: th.Tensor) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Forward pass.
        
        Returns:
            - discrete: action logits
            - continuous: (mean, log_std)
        """
        features = self.features(obs)
        
        if self.action_space_type == "discrete":
            return self.action_logits(features)
        else:
            mean = self.action_mean(features)
            log_std = self.action_log_std.expand_as(mean)
            return mean, log_std
    
    def get_action_and_log_prob(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor]:
        """
        Sample action and compute log probability.
        
        Returns:
            action: sampled action
            log_prob: log probability of the action
        """
        if self.action_space_type == "discrete":
            logits = self.forward(obs)
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob
        else:
            mean, log_std = self.forward(obs)
            std = th.exp(log_std)
            dist = Normal(mean, std)
            if deterministic:
                action = mean
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob
    
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Evaluate log probability and entropy for given state-action pairs.
        
        Returns:
            log_prob: log probability of actions
            entropy: entropy of the policy distribution
        """
        if self.action_space_type == "discrete":
            logits = self.forward(obs)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            mean, log_std = self.forward(obs)
            std = th.exp(log_std)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class Trajectory:
    """Store a single trajectory (episode)."""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
    
    def add(self, obs, action, reward, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_return(self, gamma: float) -> float:
        """Compute discounted return (pure Monte Carlo)."""
        returns = 0.0
        for t, reward in enumerate(self.rewards):
            returns += (gamma ** t) * reward
        return returns
    
    def to_tensors(self, device):
        """Convert lists to tensors."""
        return {
            'observations': th.tensor(np.array(self.observations), dtype=th.float32, device=device),
            'actions': th.tensor(np.array(self.actions), dtype=th.long if len(self.actions) > 0 and isinstance(self.actions[0], (int, np.integer)) else th.float32, device=device),
            'log_probs': th.tensor(np.array(self.log_probs), dtype=th.float32, device=device),
        }
    
    def __len__(self):
        return len(self.observations)


class TrajectoryGroup:
    """Store G trajectories sampled from the same state."""
    
    def __init__(self, trajectories: List[Trajectory], returns: List[float], initial_state):
        self.trajectories = trajectories
        self.returns = np.array(returns)
        self.initial_state = initial_state
        self.advantages = None
    
    def compute_group_advantages(self):
        """
        Compute group-relative advantages.
        
        GRPO Algorithm (Equation in paper):
            Â_i = (R_i - mean(R_group)) / std(R_group)
        
        All timesteps in trajectory i get the SAME advantage Â_i.
        """
        mean_return = self.returns.mean()
        std_return = self.returns.std() + 1e-8
        
        # Normalize returns within this group
        self.advantages = (self.returns - mean_return) / std_return
        
        return self.advantages


class GRPO:
    """
    Pure Group Relative Policy Optimization.
    
    100% faithful to DeepSeek-R1 paper - no compromises.
    """
    
    def __init__(
        self,
        env: gym.Env,
        policy_network: PolicyNetwork,
        group_size: int = 32,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        beta: float = 0.04,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
        verbose: int = 1,
        identical_group_seed: bool = True,
    ):
        self.env = env
        self.policy = policy_network.to(device)
        self.device = device
        
        # GRPO hyperparameters
        self.group_size = group_size  # G in paper
        self.gamma = gamma
        self.beta = beta  # KL penalty coefficient
        self.clip_range = clip_range  # ε in paper
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.identical_group_seed = bool(identical_group_seed)
        self._rng = np.random.default_rng()
        
        # Optimizer (policy only, no value function!)
        self.optimizer = th.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.num_timesteps = 0
        self.num_iterations = 0
    
    def _reset_env(self, seed: Optional[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            reset_res = self.env.reset(seed=seed)
        except TypeError:
            reset_res = self.env.reset()

        obs, info = reset_res if isinstance(reset_res, tuple) else (reset_res, {})
        obs_array = np.asarray(obs, dtype=np.float32)
        return obs_array, info

    def collect_trajectory(
        self,
        initial_state: Optional[np.ndarray] = None,
        deterministic: bool = False,
        seed: Optional[int] = None,
    ) -> Trajectory:
        """Collect a single trajectory, optionally resetting to a provided state."""

        trajectory = Trajectory()

        obs, _info = self._reset_env(seed)
        if initial_state is not None:
            expected = np.asarray(initial_state, dtype=np.float32)
            if not np.allclose(obs, expected, atol=1e-6):
                raise RuntimeError(
                    "Environment reset did not reproduce the requested initial state."
                )
            obs = obs.copy()

        done = False
        truncated = False

        with th.no_grad():
            while not (done or truncated):
                obs_tensor = th.tensor(obs, dtype=th.float32, device=self.device).unsqueeze(0)
                action_tensor, log_prob_tensor = self.policy.get_action_and_log_prob(
                    obs_tensor, deterministic=deterministic
                )

                if self.policy.action_space_type == "discrete":
                    env_action = int(action_tensor.squeeze(0).cpu().item())
                    stored_action = env_action
                else:
                    env_action = action_tensor.squeeze(0).cpu().numpy()
                    stored_action = np.asarray(env_action, dtype=np.float32)

                step_result = self.env.step(env_action)
                if len(step_result) == 5:
                    next_obs, reward, done, truncated, _ = step_result
                else:
                    next_obs, reward, done, _ = step_result
                    truncated = False

                trajectory.add(
                    obs.copy(),
                    stored_action,
                    reward,
                    float(log_prob_tensor.squeeze(0).cpu().item()),
                    done or truncated,
                )

                obs = np.asarray(next_obs, dtype=np.float32)
                self.num_timesteps += 1

        return trajectory

    def collect_group(
        self,
        initial_state: Optional[np.ndarray] = None,
        num_trajectories: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> TrajectoryGroup:
        """
        Collect G trajectories from the SAME initial state.

        If ``initial_state`` is not provided, the environment is reset once and the
        resulting observation is used as the shared starting point. Identical resets
        are enforced by reusing a single ``seed`` (generated automatically when
        ``identical_group_seed`` is True).
        """

        total_traj = num_trajectories or self.group_size
        if total_traj <= 0:
            raise ValueError("num_trajectories must be positive")

        shared_seed = seed
        if initial_state is None:
            if shared_seed is None and self.identical_group_seed:
                shared_seed = int(self._rng.integers(0, 2**31 - 1))
            obs, _ = self._reset_env(shared_seed)
            initial_state = obs.copy() if hasattr(obs, "copy") else obs
        elif shared_seed is None and self.identical_group_seed:
            shared_seed = int(self._rng.integers(0, 2**31 - 1))

        trajectories: List[Trajectory] = []
        returns: List[float] = []

        for _ in range(total_traj):
            trajectory = self.collect_trajectory(
                initial_state=initial_state,
                deterministic=False,
                seed=shared_seed,
            )
            discounted_return = trajectory.compute_return(self.gamma)

            trajectories.append(trajectory)
            returns.append(discounted_return)

            self.episode_rewards.append(discounted_return)
            self.episode_lengths.append(len(trajectory))

        group = TrajectoryGroup(
            trajectories,
            returns,
            initial_state.copy() if hasattr(initial_state, "copy") else initial_state,
        )
        group.compute_group_advantages()
        return group
    
    def train_on_groups(self, groups: List[TrajectoryGroup]):
        """
        Update policy using collected trajectory groups.
        
        GRPO Loss (from paper):
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        L = L_clip + β·L_KL + α·L_entropy
        
        where:
            L_clip = -E[min(r·Â, clip(r,1-ε,1+ε)·Â)]
            L_KL = E[log π_θ - log π_old]  (Direct KL penalty)
            L_entropy = -E[H(π_θ)]
            
        NO VALUE FUNCTION LOSS!
        """
        # Flatten all trajectories from all groups
        all_observations = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        
        for group in groups:
            for traj_idx, trajectory in enumerate(group.trajectories):
                advantage = group.advantages[traj_idx]
                
                # ALL steps in trajectory get SAME advantage (trajectory-level)
                for t in range(len(trajectory)):
                    all_observations.append(trajectory.observations[t])
                    all_actions.append(trajectory.actions[t])
                    all_old_log_probs.append(trajectory.log_probs[t])
                    all_advantages.append(advantage)
        
        # Convert to tensors
        observations = th.tensor(np.array(all_observations), dtype=th.float32, device=self.device)
        actions = th.tensor(np.array(all_actions), dtype=th.long if self.policy.action_space_type == "discrete" else th.float32, device=self.device)
        old_log_probs = th.tensor(np.array(all_old_log_probs), dtype=th.float32, device=self.device)
        advantages = th.tensor(np.array(all_advantages), dtype=th.float32, device=self.device)
        
        # Create dataset
        dataset_size = len(observations)
        indices = np.arange(dataset_size)
        
        # Training statistics
        pg_losses, ent_losses, kl_losses, total_losses = [], [], [], []
        approx_kls, clip_fractions = [], []
        
        # Train for n_epochs
        for epoch in range(self.n_epochs):
            # Shuffle data
            np.random.shuffle(indices)
            
            # Mini-batch training
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # NO PER-MINIBATCH NORMALIZATION!
                # We keep the group-relative advantages as-is
                
                # Evaluate actions with current policy
                log_probs, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)
                
                # ============================================================
                # 1. CLIPPED POLICY LOSS
                # ============================================================
                ratio = th.exp(log_probs - batch_old_log_probs)
                
                policy_loss_1 = batch_advantages * ratio
                policy_loss_2 = batch_advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                
                # ============================================================
                # 2. KL PENALTY (Direct in loss, not in reward!)
                # ============================================================
                # Unbiased KL estimator: E[(r - 1) - log(r)] where r = π_θ/π_old
                with th.no_grad():
                    log_ratio = log_probs - batch_old_log_probs
                    approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio)
                    approx_kls.append(approx_kl.item())
                    
                    clip_fraction = th.mean((th.abs(ratio - 1) > self.clip_range).float())
                    clip_fractions.append(clip_fraction.item())
                
                # Add KL penalty to loss (not to reward!)
                kl_loss = self.beta * th.mean(log_ratio)
                
                # ============================================================
                # 3. ENTROPY LOSS
                # ============================================================
                entropy_loss = -th.mean(entropy)
                
                # ============================================================
                # 4. TOTAL LOSS (NO VALUE FUNCTION!)
                # ============================================================
                loss = policy_loss + kl_loss + self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Logging
                pg_losses.append(policy_loss.item())
                kl_losses.append(kl_loss.item())
                ent_losses.append(entropy_loss.item())
                total_losses.append(loss.item())
            
            # Early stopping based on KL divergence
            mean_kl = np.mean(approx_kls)
            if mean_kl > 0.15:
                if self.verbose >= 1:
                    print(f"  Early stopping at epoch {epoch+1}/{self.n_epochs} (KL={mean_kl:.4f})")
                break
        
        # Return training statistics
        return {
            'policy_loss': np.mean(pg_losses),
            'kl_loss': np.mean(kl_losses),
            'entropy_loss': np.mean(ent_losses),
            'total_loss': np.mean(total_losses),
            'approx_kl': np.mean(approx_kls),
            'clip_fraction': np.mean(clip_fractions),
        }
    
    def learn(
        self,
        total_timesteps: int,
        num_groups_per_iteration: int = 10,
        log_interval: int = 1,
    ):
        """
        Main training loop for GRPO.
        
        Args:
            total_timesteps: Total number of environment steps
            num_groups_per_iteration: Number of trajectory groups to collect per iteration
            log_interval: Log every N iterations
        """
        if self.verbose >= 1:
            print("=" * 70)
            print("PURE GRPO Training (100% faithful to paper)")
            print("=" * 70)
            print(f"Group Size (G):           {self.group_size}")
            print(f"KL Penalty (β):           {self.beta}")
            print(f"Clip Range (ε):           {self.clip_range}")
            print(f"Groups per iteration:     {num_groups_per_iteration}")
            print(f"Trajectories per iter:    {num_groups_per_iteration * self.group_size}")
            print(f"NO VALUE FUNCTION:        ✓")
            print(f"Pure Monte Carlo:         ✓")
            print(f"Group-relative advantages: ✓")
            print(f"Same-state sampling:      ✓")
            print("=" * 70)
        
        while self.num_timesteps < total_timesteps:
            self.num_iterations += 1
            
            # ================================================================
            # PHASE 1: COLLECT TRAJECTORY GROUPS
            # ================================================================
            # Collect multiple groups (each group has G trajectories from same state)
            groups = []
            for _ in range(num_groups_per_iteration):
                group = self.collect_group()
                groups.append(group)
            
            # ================================================================
            # PHASE 2: POLICY UPDATE
            # ================================================================
            train_stats = self.train_on_groups(groups)
            
            # ================================================================
            # LOGGING
            # ================================================================
            if self.num_iterations % log_interval == 0 and self.verbose >= 1:
                mean_reward = np.mean(self.episode_rewards) if len(self.episode_rewards) > 0 else 0
                mean_length = np.mean(self.episode_lengths) if len(self.episode_lengths) > 0 else 0
                
                print(f"\nIteration {self.num_iterations}")
                print(f"  Timesteps:       {self.num_timesteps}/{total_timesteps}")
                print(f"  Ep Reward (100): {mean_reward:.2f}")
                print(f"  Ep Length (100): {mean_length:.1f}")
                print(f"  Policy Loss:     {train_stats['policy_loss']:.4f}")
                print(f"  KL Loss:         {train_stats['kl_loss']:.4f}")
                print(f"  Entropy Loss:    {train_stats['entropy_loss']:.4f}")
                print(f"  Approx KL:       {train_stats['approx_kl']:.4f}")
                print(f"  Clip Fraction:   {train_stats['clip_fraction']:.3f}")
        
        if self.verbose >= 1:
            print("\nTraining completed!")
    
    def save(self, path: str):
        """Save policy network."""
        th.save(self.policy.state_dict(), path)
    
    def load(self, path: str):
        """Load policy network."""
        self.policy.load_state_dict(th.load(path))


def make_grpo_agent(env: gym.Env, **kwargs) -> GRPO:
    """Factory function to create a GRPO agent using shared defaults."""

    config = default_grpo_config(**kwargs)
    policy_network = config.pop("policy_network", None)
    hidden_dims = config.pop("hidden_dims", [64, 64])
    if hidden_dims is None:
        hidden_dims = [64, 64]

    if policy_network is None:
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_space_type = "discrete"
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):
            action_space_type = "continuous"
            action_dim = env.action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space: {type(env.action_space)}")

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(f"Unsupported observation space: {type(env.observation_space)}")

        obs_dim = env.observation_space.shape[0]
        policy_network = PolicyNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            action_space_type=action_space_type,
        )

    return GRPO(env=env, policy_network=policy_network, **config)


def load_trained_grpo(model_path: str, env_factory: Callable[[], gym.Env], **kwargs) -> GRPO:
    """Load a trained GRPO agent using an environment factory."""

    env = env_factory()
    agent = make_grpo_agent(env, **kwargs)
    agent.load(model_path)
    agent.policy.eval()
    return agent


def evaluate_policy(
    agent: GRPO,
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = False,
    render: bool = False,
) -> Dict[str, Any]:
    """Run evaluation episodes and return aggregated metrics."""

    successes: List[float] = []
    rewards: List[float] = []
    lengths: List[int] = []
    crash_details: List[Dict[str, Any]] = []

    for _ in range(n_episodes):
        reset_res = env.reset()
        obs, info = reset_res if isinstance(reset_res, tuple) else (reset_res, {})
        done = False
        truncated = False
        episode_reward = 0.0
        episode_length = 0
        last_info: Dict[str, Any] = info

        while not (done or truncated):
            obs_tensor = th.tensor(np.array([obs], dtype=np.float32), device=agent.device)
            action_tensor, _ = agent.policy.get_action_and_log_prob(obs_tensor, deterministic=deterministic)

            if agent.policy.action_space_type == "discrete":
                env_action = int(action_tensor.squeeze(0).cpu().item())
            else:
                env_action = action_tensor.squeeze(0).cpu().numpy()

            step_result = env.step(env_action)
            if len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                truncated = False

            last_info = info
            episode_reward += float(reward)
            episode_length += 1

            if render:
                env.render()

        success = float(1.0 if last_info.get("landing_success", False) else 0.0)
        successes.append(success)
        rewards.append(episode_reward)
        lengths.append(episode_length)

        if not success:
            crash_details.append(
                {
                    "velocity": last_info.get("crash_velocity", 0),
                    "legs_touching": last_info.get("legs_touching", 0),
                    "penalty": last_info.get("crash_penalty", 0),
                }
            )

    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "success_std": float(np.std(successes)) if successes else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "length_std": float(np.std(lengths)) if lengths else 0.0,
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "successes": successes,
        "n_episodes": n_episodes,
        "crash_details": crash_details,
    }


def load_trained_grpo(model_path: str, env_factory: Callable[[], gym.Env], **kwargs) -> GRPO:
    """Load a trained GRPO agent using an environment factory."""

    env = env_factory()
    agent = make_grpo_agent(env, **kwargs)
    agent.load(model_path)
    agent.policy.eval()
    return agent


def evaluate_policy(
    agent: GRPO,
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = False,
    render: bool = False,
) -> Dict[str, Any]:
    """Run evaluation episodes for a GRPO agent and return aggregated metrics."""

    successes: List[float] = []
    rewards: List[float] = []
    lengths: List[int] = []
    crash_details: List[Dict[str, Any]] = []

    for _ in range(n_episodes):
        reset_res = env.reset()
        obs, info = reset_res if isinstance(reset_res, tuple) else (reset_res, {})
        done = False
        truncated = False
        ep_reward = 0.0
        ep_length = 0
        last_info: Dict[str, Any] = info

        with th.no_grad():
            while not (done or truncated):
                obs_tensor = th.tensor(np.array([obs]), dtype=th.float32, device=agent.device)
                action_tensor, _ = agent.policy.get_action_and_log_prob(obs_tensor, deterministic=deterministic)

                if agent.policy.action_space_type == "discrete":
                    action = int(action_tensor.cpu().numpy()[0])
                else:
                    action = action_tensor.cpu().numpy()[0]

                step_res = env.step(action)
                if len(step_res) == 5:
                    obs, reward, done, truncated, info = step_res
                else:
                    obs, reward, done, info = step_res
                    truncated = False

                last_info = info
                ep_reward += float(reward)
                ep_length += 1

                if render:
                    env.render()

        success = float(1.0 if last_info.get("landing_success", False) else 0.0)
        successes.append(success)
        rewards.append(ep_reward)
        lengths.append(ep_length)

        if not success:
            crash_details.append(
                {
                    "velocity": last_info.get("crash_velocity", 0),
                    "legs_touching": last_info.get("legs_touching", 0),
                    "penalty": last_info.get("crash_penalty", 0),
                }
            )

    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "success_std": float(np.std(successes)) if successes else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "length_std": float(np.std(lengths)) if lengths else 0.0,
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "successes": successes,
        "n_episodes": n_episodes,
        "crash_details": crash_details,
    }
