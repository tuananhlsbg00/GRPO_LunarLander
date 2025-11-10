"""
Training Comparison: GRPO vs PPO on Sparse Lunar Lander

This script trains both GRPO (pure implementation) and PPO algorithms on the sparse 
reward lunar lander environment and compares their performance, focusing on success rate.
"""

import os
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch as th
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Import custom implementations
from ..grpo.grpo import make_grpo_agent, GRPO
from ..ppo.ppo import make_ppo_agent, PPO
from .sparse_lunar_lander import SparseLunarLander, EnvConfig

__all__ = [
    "EvaluationMetrics",
    "EnvConfig",
    "TrainingComparisonConfig",
    "evaluate_model",
    "train_ppo",
    "train_grpo",
    "plot_comparison",
    "save_metrics_csv",
    "save_training_stats",
    "print_final_summary",
    "run_training_comparison",
]


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


class EvaluationMetrics:
    """
    Container for evaluation metrics during training.
    """
    def __init__(self):
        self.evaluations_timesteps = []
        self.evaluations_success_rate = []
        self.evaluations_length = []
        self.evaluations_rewards = []
        self.best_success_rate = 0.0
        self.best_avg_reward = -float('inf')
        self.training_time = 0.0
        self.final_evaluation = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            'evaluations_timesteps': self.evaluations_timesteps,
            'evaluations_success_rate': self.evaluations_success_rate,
            'evaluations_length': self.evaluations_length,
            'evaluations_rewards': self.evaluations_rewards,
            'best_success_rate': float(self.best_success_rate),
            'training_time': float(self.training_time),
            'final_evaluation': self.final_evaluation,
        }


@dataclass
class TrainingComparisonConfig:
    """Configuration options for running the GRPO vs PPO comparison experiment."""

    total_timesteps: int = 50_000
    eval_frequency: int = 10_000
    seed: int = 42
    base_output_dir: Path = Path(".")
    run_id: Optional[str] = None
    env_config: Optional[EnvConfig] = None

    def resolve_run_dir(self) -> Path:
        """Return the directory where outputs for this run should be stored."""

        identifier = self.run_id or str(int(time.time()))
        return Path(self.base_output_dir) / identifier
    
    def get_env_config(self) -> EnvConfig:
        """Return the environment configuration, using defaults if not provided."""
        return self.env_config or EnvConfig()


def evaluate_model(model, eval_env, n_episodes: int = 20, deterministic: bool = False) -> Dict[str, Any]:
    """
    Evaluate a trained model on the environment.
    
    Args:
        model: Trained PPO or GRPO agent
        eval_env: Environment to evaluate on
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic policy
        
    Returns:
        Dictionary with evaluation metrics
    """
    successes = []
    episode_lengths = []
    episode_rewards = []
    crash_details = []
    
    for _ in range(n_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get action from model
            if isinstance(model, PPO):
                obs_tensor = model._obs_to_tensor(obs)
                dist, _ = model.policy.dist(obs_tensor)
                if deterministic:
                    if model.action_space_type == "discrete":
                        action = th.argmax(dist.probs, dim=-1)
                        act = int(action.item())
                    else:
                        act = dist.mean.cpu().numpy()[0]
                else:
                    if model.action_space_type == "discrete":
                        act = int(dist.sample().item())
                    else:
                        act = dist.sample().cpu().numpy()[0]
            elif isinstance(model, GRPO):
                obs_tensor = th.tensor(np.array([obs]), dtype=th.float32, device=model.device)
                action, _ = model.policy.get_action_and_log_prob(obs_tensor, deterministic=deterministic)
                if model.policy.action_space_type == "discrete":
                    act = int(action.item())
                else:
                    act = action.cpu().numpy()[0]
            else:
                raise ValueError(f"Unknown model type: {type(model)}")
            
            obs, reward, terminated, truncated, info = eval_env.step(act)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            if done:
                success = info.get('landing_success', False)
                successes.append(1.0 if success else 0.0)
                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)
                
                if not success:
                    crash_details.append({
                        'velocity': info.get('crash_velocity', 0),
                        'angle': info.get('crash_angle', 0),
                        'distance': info.get('distance_from_pad', 0),
                        'legs_touching': info.get('legs_touching', 0),
                        'penalty': info.get('crash_penalty', 0),
                    })
    
    # Calculate statistics
    return {
        'success_rate': float(np.mean(successes)),
        'success_std': float(np.std(successes)),
        'mean_reward': float(np.mean(episode_rewards)),
        'reward_std': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'length_std': float(np.std(episode_lengths)),
        'n_episodes': n_episodes,
        'crash_details': crash_details,
    }


def train_ppo(
    total_timesteps: int = 2_000_000,
    eval_freq: int = 10000,
    log_dir: str = "./logs/ppo",
    model_save_path: str = "./models/ppo_lunar_lander.pth",
    best_model_save_path: str = "./models/ppo_lunar_lander_best.pth",
    seed: int = 42,
    env_config: Optional[EnvConfig] = None
) -> Tuple[PPO, EvaluationMetrics]:
    """
    Train Pure PPO on Sparse Lunar Lander.
    
    Args:
        total_timesteps: Total training timesteps
        eval_freq: Evaluation frequency
        log_dir: Directory for logs
        model_save_path: Path to save final model
        best_model_save_path: Path to save best model
        seed: Random seed
        env_config: Environment configuration (uses defaults if None)
        
    Returns:
        Trained model and evaluation metrics
    """
    print("\n" + "="*70)
    print("TRAINING PURE PPO ON SPARSE LUNAR LANDER")
    print("="*70)
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Get environment configuration
    env_cfg = env_config or EnvConfig()
    env_kwargs = env_cfg.to_env_kwargs()
    
    # Create environment
    env = SparseLunarLander(**env_kwargs)
    eval_env = SparseLunarLander(**env_kwargs)
    
    # Set seeds
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)
    
    # Create PPO agent
    model = make_ppo_agent(
        env,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        beta=0.04,
        device='cpu',
        verbose=0,
    )
    
    # Tracking metrics
    metrics = EvaluationMetrics()
    
    # Custom training loop with evaluation
    print(f"\nConfiguration:")
    print(f"  Total timesteps:     {total_timesteps:,}")
    print(f"  Evaluation freq:     {eval_freq:,}")
    print(f"  n_steps:             {model.n_steps}")
    print(f"  batch_size:          {model.batch_size}")
    print(f"  n_epochs:            {model.n_epochs}")
    print(f"  learning_rate:       {model.lr}")
    print(f"  gamma:               {model.gamma}")
    print(f"  clip_range:          {model.clip_range}")
    print(f"  ent_coef:            {model.ent_coef}")
    print(f"  beta (KL penalty):   {model.beta}")
    print()
    
    start_time = time.time()
    last_eval_timesteps = 0
    
    iteration = 0
    while model.num_timesteps < total_timesteps:
        iteration += 1
        
        # Collect rollout and update
        model._collect_rollout()
        train_stats = model._update()
        
        # Check if we should evaluate
        if model.num_timesteps - last_eval_timesteps >= eval_freq:
            # Run evaluation
            eval_results = evaluate_model(model, eval_env, n_episodes=20, deterministic=False)
            
            success_rate = eval_results['success_rate']
            avg_length = eval_results['mean_length']
            avg_reward = eval_results['mean_reward']
            
            # Store metrics
            metrics.evaluations_timesteps.append(model.num_timesteps)
            metrics.evaluations_success_rate.append(success_rate)
            metrics.evaluations_length.append(avg_length)
            metrics.evaluations_rewards.append(avg_reward)
            
            # Track best and save best model
            if success_rate > metrics.best_success_rate or (success_rate == metrics.best_success_rate and avg_reward > metrics.best_avg_reward):
                metrics.best_success_rate = success_rate
                metrics.best_avg_reward = avg_reward
                model.save(best_model_save_path)
                print(f"  🌟 New best model saved! Success rate: {success_rate:.2%}")
            
            # Log progress
            mean_train_reward = np.mean(list(model.episode_rewards)[-100:]) if len(model.episode_rewards) > 0 else 0.0
            
            # Time estimation
            elapsed = time.time() - start_time
            progress = model.num_timesteps / total_timesteps
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = estimated_total - elapsed
            
            print(f"Iter {iteration:4d} | Step {model.num_timesteps:,}/{total_timesteps:,} ({progress:6.1%})")
            print(f"  Time:          {format_time(elapsed):>12s} elapsed, {format_time(remaining):>12s} remaining")
            print(f"  Train Reward:  {mean_train_reward:8.2f} (last 100 episodes)")
            print(f"  Eval Success:  {success_rate:8.2%}")
            print(f"  Eval Reward:   {avg_reward:8.2f}")
            print(f"  Eval Length:   {avg_length:8.1f}")
            print(f"  Best Success:  {metrics.best_success_rate:8.2%}")
            print(f"  Policy Loss:   {train_stats['policy_loss']:8.4f}")
            print(f"  Value Loss:    {train_stats['value_loss']:8.4f}")
            print(f"  Approx KL:     {train_stats['approx_kl']:8.4f}")
            print()
            
            last_eval_timesteps = model.num_timesteps
    
    train_time = time.time() - start_time
    metrics.training_time = train_time
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION (PPO)")
    print("="*70)
    final_eval = evaluate_model(model, eval_env, n_episodes=100, deterministic=False)
    metrics.final_evaluation = final_eval
    
    print(f"Success Rate:  {final_eval['success_rate']:.2%} ± {final_eval['success_std']:.2%}")
    print(f"Mean Reward:   {final_eval['mean_reward']:.2f} ± {final_eval['reward_std']:.2f}")
    print(f"Mean Length:   {final_eval['mean_length']:.1f} ± {final_eval['length_std']:.1f}")
    print(f"Episodes:      {final_eval['n_episodes']}")
    
    # Save final model
    model.save(model_save_path)
    print(f"\nPPO Training completed in {format_time(train_time)}")
    print(f"Final model saved to:     {model_save_path}")
    print(f"Best model saved to:      {best_model_save_path}")
    print(f"Best success rate:        {metrics.best_success_rate:.2%}")
    print("="*70)
    
    env.close()
    eval_env.close()
    
    return model, metrics


def train_grpo(
    total_timesteps: int = 2_000_000,
    eval_freq: int = 10000,
    log_dir: str = "./logs/grpo",
    model_save_path: str = "./models/grpo_lunar_lander.pth",
    best_model_save_path: str = "./models/grpo_lunar_lander_best.pth",
    seed: int = 42,
    env_config: Optional[EnvConfig] = None
) -> Tuple[GRPO, EvaluationMetrics]:
    """
    Train Pure GRPO on Sparse Lunar Lander.
    
    Args:
        total_timesteps: Total training timesteps
        eval_freq: Evaluation frequency
        log_dir: Directory for logs
        model_save_path: Path to save final model
        best_model_save_path: Path to save best model
        seed: Random seed
        env_config: Environment configuration (uses defaults if None)
        
    Returns:
        Trained model and evaluation metrics
    """
    print("\n" + "="*70)
    print("TRAINING PURE GRPO ON SPARSE LUNAR LANDER")
    print("="*70)
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Get environment configuration
    env_cfg = env_config or EnvConfig()
    env_kwargs = env_cfg.to_env_kwargs()
    
    # Create environment
    env = SparseLunarLander(**env_kwargs)
    eval_env = SparseLunarLander(**env_kwargs)
    
    # Set seeds
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)
    
    # Create GRPO agent
    model = make_grpo_agent(
        env,
        group_size=32,
        learning_rate=2e-4,
        gamma=1.0,
        beta=0.04,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64,
        device='cpu',
        verbose=0,
    )
    
    # Tracking metrics
    metrics = EvaluationMetrics()
    
    # Custom training loop with evaluation
    print(f"\nConfiguration:")
    print(f"  Total timesteps:     {total_timesteps:,}")
    print(f"  Evaluation freq:     {eval_freq:,}")
    print(f"  group_size:          {model.group_size}")
    print(f"  batch_size:          {model.batch_size}")
    print(f"  n_epochs:            {model.n_epochs}")
    print(f"  learning_rate:       {model.optimizer.param_groups[0]['lr']}")
    print(f"  gamma:               {model.gamma}")
    print(f"  clip_range:          {model.clip_range}")
    print(f"  ent_coef:            {model.ent_coef}")
    print(f"  beta (KL penalty):   {model.beta}")
    print()
    
    start_time = time.time()
    last_eval_timesteps = 0
    
    iteration = 0
    while model.num_timesteps < total_timesteps:
        iteration += 1
        
        # Collect and train on trajectory groups
        num_groups = 10
        groups = []
        for _ in range(num_groups):
            group = model.collect_group()
            groups.append(group)
        
        # Update policy
        train_stats = model.train_on_groups(groups)
        
        # Check if we should evaluate
        if model.num_timesteps - last_eval_timesteps >= eval_freq:
            # Run evaluation
            eval_results = evaluate_model(model, eval_env, n_episodes=20, deterministic=False)
            
            success_rate = eval_results['success_rate']
            avg_length = eval_results['mean_length']
            avg_reward = eval_results['mean_reward']
            
            # Store metrics
            metrics.evaluations_timesteps.append(model.num_timesteps)
            metrics.evaluations_success_rate.append(success_rate)
            metrics.evaluations_length.append(avg_length)
            metrics.evaluations_rewards.append(avg_reward)
            
            # Track best and save best model
            if success_rate > metrics.best_success_rate or (success_rate == metrics.best_success_rate and avg_reward > metrics.best_avg_reward):
                metrics.best_success_rate = success_rate
                metrics.best_avg_reward = avg_reward
                model.save(best_model_save_path)
                print(f"  🌟 New best model saved! Success rate: {success_rate:.2%}")
            
            # Log progress
            mean_train_reward = np.mean(list(model.episode_rewards)) if len(model.episode_rewards) > 0 else 0.0
            
            # Time estimation
            elapsed = time.time() - start_time
            progress = model.num_timesteps / total_timesteps
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = estimated_total - elapsed
            
            print(f"Iter {iteration:4d} | Step {model.num_timesteps:,}/{total_timesteps:,} ({progress:6.1%})")
            print(f"  Time:          {format_time(elapsed):>12s} elapsed, {format_time(remaining):>12s} remaining")
            print(f"  Train Reward:  {mean_train_reward:8.2f} (last 100 episodes)")
            print(f"  Eval Success:  {success_rate:8.2%}")
            print(f"  Eval Reward:   {avg_reward:8.2f}")
            print(f"  Eval Length:   {avg_length:8.1f}")
            print(f"  Best Success:  {metrics.best_success_rate:8.2%}")
            print(f"  Policy Loss:   {train_stats['policy_loss']:8.4f}")
            print(f"  Approx KL:     {train_stats['approx_kl']:8.4f}")
            print()
            
            last_eval_timesteps = model.num_timesteps
    
    train_time = time.time() - start_time
    metrics.training_time = train_time
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION (GRPO)")
    print("="*70)
    final_eval = evaluate_model(model, eval_env, n_episodes=100, deterministic=False)
    metrics.final_evaluation = final_eval
    
    print(f"Success Rate:  {final_eval['success_rate']:.2%} ± {final_eval['success_std']:.2%}")
    print(f"Mean Reward:   {final_eval['mean_reward']:.2f} ± {final_eval['reward_std']:.2f}")
    print(f"Mean Length:   {final_eval['mean_length']:.1f} ± {final_eval['length_std']:.1f}")
    print(f"Episodes:      {final_eval['n_episodes']}")
    
    # Save final model
    model.save(model_save_path)
    print(f"\nGRPO Training completed in {format_time(train_time)}")
    print(f"Final model saved to:     {model_save_path}")
    print(f"Best model saved to:      {best_model_save_path}")
    print(f"Best success rate:        {metrics.best_success_rate:.2%}")
    print("="*70)
    
    env.close()
    eval_env.close()
    
    return model, metrics


def plot_comparison(
    ppo_metrics: EvaluationMetrics,
    grpo_metrics: EvaluationMetrics,
    save_dir: str = "./results"
):
    """
    Plot comparison of PPO vs GRPO performance.
    
    Args:
        ppo_metrics: PPO evaluation metrics
        grpo_metrics: GRPO evaluation metrics
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Main comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Add training time to title
    title = f'GRPO vs PPO on Sparse Lunar Lander\n'
    title += f'PPO: {format_time(ppo_metrics.training_time)} | GRPO: {format_time(grpo_metrics.training_time)}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Success Rate
    axes[0, 0].plot(ppo_metrics.evaluations_timesteps, ppo_metrics.evaluations_success_rate, 
                    label='PPO', marker='o', linewidth=2, markersize=4, alpha=0.8)
    axes[0, 0].plot(grpo_metrics.evaluations_timesteps, grpo_metrics.evaluations_success_rate, 
                    label='GRPO', marker='s', linewidth=2, markersize=4, alpha=0.8)
    axes[0, 0].axhline(y=ppo_metrics.best_success_rate, color='blue', linestyle='--', 
                       alpha=0.5, label=f'PPO Best: {ppo_metrics.best_success_rate:.2%}')
    axes[0, 0].axhline(y=grpo_metrics.best_success_rate, color='orange', linestyle='--', 
                       alpha=0.5, label=f'GRPO Best: {grpo_metrics.best_success_rate:.2%}')
    axes[0, 0].set_xlabel('Timesteps', fontsize=12)
    axes[0, 0].set_ylabel('Success Rate', fontsize=12)
    axes[0, 0].set_title('Success Rate Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Average Reward
    axes[0, 1].plot(ppo_metrics.evaluations_timesteps, ppo_metrics.evaluations_rewards, 
                    label='PPO', marker='o', linewidth=2, markersize=4, alpha=0.8)
    axes[0, 1].plot(grpo_metrics.evaluations_timesteps, grpo_metrics.evaluations_rewards, 
                    label='GRPO', marker='s', linewidth=2, markersize=4, alpha=0.8)
    axes[0, 1].set_xlabel('Timesteps', fontsize=12)
    axes[0, 1].set_ylabel('Average Reward', fontsize=12)
    axes[0, 1].set_title('Average Reward Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Average Episode Length
    axes[1, 0].plot(ppo_metrics.evaluations_timesteps, ppo_metrics.evaluations_length, 
                    label='PPO', marker='o', linewidth=2, markersize=4, alpha=0.8)
    axes[1, 0].plot(grpo_metrics.evaluations_timesteps, grpo_metrics.evaluations_length, 
                    label='GRPO', marker='s', linewidth=2, markersize=4, alpha=0.8)
    axes[1, 0].set_xlabel('Timesteps', fontsize=12)
    axes[1, 0].set_ylabel('Average Episode Length', fontsize=12)
    axes[1, 0].set_title('Average Episode Length Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final Comparison Bar Chart
    ppo_final_success = ppo_metrics.evaluations_success_rate[-1] if ppo_metrics.evaluations_success_rate else 0.0
    grpo_final_success = grpo_metrics.evaluations_success_rate[-1] if grpo_metrics.evaluations_success_rate else 0.0
    ppo_final_reward = ppo_metrics.evaluations_rewards[-1] if ppo_metrics.evaluations_rewards else 0.0
    grpo_final_reward = grpo_metrics.evaluations_rewards[-1] if grpo_metrics.evaluations_rewards else 0.0
    
    metrics_names = ['Final\nSuccess', 'Best\nSuccess', 'Final\nReward']
    ppo_vals = [ppo_final_success, ppo_metrics.best_success_rate, ppo_final_reward / 200]  # Normalize reward
    grpo_vals = [grpo_final_success, grpo_metrics.best_success_rate, grpo_final_reward / 200]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, ppo_vals, width, label='PPO', alpha=0.8, color='tab:blue')
    bars2 = axes[1, 1].bar(x + width/2, grpo_vals, width, label='GRPO', alpha=0.8, color='tab:orange')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}',
                          ha='center', va='bottom', fontsize=9)
    
    axes[1, 1].set_xlabel('Metric', fontsize=12)
    axes[1, 1].set_ylabel('Value (Normalized)', fontsize=12)
    axes[1, 1].set_title('Final Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_names, fontsize=10)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1.1])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "comparison_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    plt.close()
    
    # Learning curves plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(ppo_metrics.evaluations_timesteps, ppo_metrics.evaluations_success_rate,
            label='PPO', linewidth=2.5, alpha=0.9)
    ax.plot(grpo_metrics.evaluations_timesteps, grpo_metrics.evaluations_success_rate,
            label='GRPO', linewidth=2.5, alpha=0.9)
    
    ax.set_xlabel('Training Timesteps', fontsize=14)
    ax.set_ylabel('Success Rate', fontsize=14)
    ax.set_title('Learning Curves: GRPO vs PPO', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Learning curves saved to: {save_path}")
    plt.close()


def save_metrics_csv(
    ppo_metrics: EvaluationMetrics,
    grpo_metrics: EvaluationMetrics,
    save_dir: str = "./results"
):
    """
    Save training metrics to CSV files.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Detailed metrics
    max_len = max(len(ppo_metrics.evaluations_timesteps), len(grpo_metrics.evaluations_timesteps))
    
    data = {
        'timestep_ppo': ppo_metrics.evaluations_timesteps + [None] * (max_len - len(ppo_metrics.evaluations_timesteps)),
        'success_rate_ppo': ppo_metrics.evaluations_success_rate + [None] * (max_len - len(ppo_metrics.evaluations_success_rate)),
        'reward_ppo': ppo_metrics.evaluations_rewards + [None] * (max_len - len(ppo_metrics.evaluations_rewards)),
        'length_ppo': ppo_metrics.evaluations_length + [None] * (max_len - len(ppo_metrics.evaluations_length)),
        
        'timestep_grpo': grpo_metrics.evaluations_timesteps + [None] * (max_len - len(grpo_metrics.evaluations_timesteps)),
        'success_rate_grpo': grpo_metrics.evaluations_success_rate + [None] * (max_len - len(grpo_metrics.evaluations_success_rate)),
        'reward_grpo': grpo_metrics.evaluations_rewards + [None] * (max_len - len(grpo_metrics.evaluations_rewards)),
        'length_grpo': grpo_metrics.evaluations_length + [None] * (max_len - len(grpo_metrics.evaluations_length)),
    }
    
    df = pd.DataFrame(data)
    metrics_path = os.path.join(save_dir, "metrics.csv")
    df.to_csv(metrics_path, index=False)
    print(f"Detailed metrics saved to: {metrics_path}")
    
    # Summary table
    summary_data = {
        'Algorithm': ['PPO', 'GRPO'],
        'Final Success Rate': [
            ppo_metrics.evaluations_success_rate[-1] if ppo_metrics.evaluations_success_rate else 0,
            grpo_metrics.evaluations_success_rate[-1] if grpo_metrics.evaluations_success_rate else 0
        ],
        'Best Success Rate': [
            ppo_metrics.best_success_rate,
            grpo_metrics.best_success_rate
        ],
        'Final Avg Reward': [
            ppo_metrics.evaluations_rewards[-1] if ppo_metrics.evaluations_rewards else 0,
            grpo_metrics.evaluations_rewards[-1] if grpo_metrics.evaluations_rewards else 0
        ],
        'Final Avg Length': [
            ppo_metrics.evaluations_length[-1] if ppo_metrics.evaluations_length else 0,
            grpo_metrics.evaluations_length[-1] if grpo_metrics.evaluations_length else 0
        ],
        'Training Time (s)': [
            ppo_metrics.training_time,
            grpo_metrics.training_time
        ],
    }
    
    # Add final evaluation metrics if available
    if ppo_metrics.final_evaluation and grpo_metrics.final_evaluation:
        summary_data['Final Eval Success Rate'] = [
            ppo_metrics.final_evaluation.get('success_rate', 0),
            grpo_metrics.final_evaluation.get('success_rate', 0)
        ]
        summary_data['Final Eval Std'] = [
            ppo_metrics.final_evaluation.get('success_std', 0),
            grpo_metrics.final_evaluation.get('success_std', 0)
        ]
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(save_dir, "evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to: {summary_path}")
    
    # Detailed evaluation results
    if ppo_metrics.final_evaluation and grpo_metrics.final_evaluation:
        eval_data = []
        
        # PPO crash details
        for i, crash in enumerate(ppo_metrics.final_evaluation.get('crash_details', [])):
            eval_data.append({
                'algorithm': 'PPO',
                'episode': i + 1,
                'crash_velocity': crash.get('velocity', 0),
                'crash_angle': crash.get('angle', 0),
                'distance_from_pad': crash.get('distance', 0),
                'legs_touching': crash.get('legs_touching', 0),
                'penalty': crash.get('penalty', 0),
            })
        
        # GRPO crash details
        for i, crash in enumerate(grpo_metrics.final_evaluation.get('crash_details', [])):
            eval_data.append({
                'algorithm': 'GRPO',
                'episode': i + 1,
                'crash_velocity': crash.get('velocity', 0),
                'crash_angle': crash.get('angle', 0),
                'distance_from_pad': crash.get('distance', 0),
                'legs_touching': crash.get('legs_touching', 0),
                'penalty': crash.get('penalty', 0),
            })
        
        if eval_data:
            eval_df = pd.DataFrame(eval_data)
            eval_path = os.path.join(save_dir, "evaluation_detailed.csv")
            eval_df.to_csv(eval_path, index=False)
            print(f"Detailed evaluation saved to: {eval_path}")


def save_training_stats(
    ppo_metrics: EvaluationMetrics,
    grpo_metrics: EvaluationMetrics,
    save_dir: str = "./results"
):
    """Save training statistics as JSON."""
    os.makedirs(save_dir, exist_ok=True)
    
    stats = {
        'ppo': ppo_metrics.to_dict(),
        'grpo': grpo_metrics.to_dict(),
        'comparison': {
            'ppo_better': ppo_metrics.best_success_rate > grpo_metrics.best_success_rate,
            'improvement': abs(ppo_metrics.best_success_rate - grpo_metrics.best_success_rate),
            'ppo_faster': ppo_metrics.training_time < grpo_metrics.training_time,
            'time_difference': abs(ppo_metrics.training_time - grpo_metrics.training_time),
        }
    }
    
    # Save PPO stats
    ppo_path = os.path.join(save_dir, "ppo_train_stats.json")
    with open(ppo_path, 'w') as f:
        json.dump(stats['ppo'], f, indent=2)
    print(f"PPO training stats saved to: {ppo_path}")
    
    # Save GRPO stats
    grpo_path = os.path.join(save_dir, "grpo_train_stats.json")
    with open(grpo_path, 'w') as f:
        json.dump(stats['grpo'], f, indent=2)
    print(f"GRPO training stats saved to: {grpo_path}")
    
    # Save combined learning curves data
    learning_curves = {
        'ppo': {
            'timesteps': ppo_metrics.evaluations_timesteps,
            'success_rates': ppo_metrics.evaluations_success_rate,
            'rewards': ppo_metrics.evaluations_rewards,
            'lengths': ppo_metrics.evaluations_length,
        },
        'grpo': {
            'timesteps': grpo_metrics.evaluations_timesteps,
            'success_rates': grpo_metrics.evaluations_success_rate,
            'rewards': grpo_metrics.evaluations_rewards,
            'lengths': grpo_metrics.evaluations_length,
        }
    }
    
    curves_path = os.path.join(save_dir, "learning_curves.json")
    with open(curves_path, 'w') as f:
        json.dump(learning_curves, f, indent=2)
    print(f"Learning curves data saved to: {curves_path}")


def print_final_summary(
    ppo_metrics: EvaluationMetrics,
    grpo_metrics: EvaluationMetrics
):
    """
    Print final training summary.
    """
    print("\n" + "="*70)
    print(" " * 20 + "TRAINING SUMMARY")
    print("="*70)
    
    print("\nPPO Results:")
    print("-" * 70)
    if ppo_metrics.evaluations_success_rate:
        print(f"  Final Success Rate:    {ppo_metrics.evaluations_success_rate[-1]:6.2%}")
        print(f"  Best Success Rate:     {ppo_metrics.best_success_rate:6.2%}")
        print(f"  Final Avg Reward:      {ppo_metrics.evaluations_rewards[-1]:8.2f}")
        print(f"  Final Avg Length:      {ppo_metrics.evaluations_length[-1]:8.1f}")
        print(f"  Training Time:         {format_time(ppo_metrics.training_time)}")
        
        if ppo_metrics.final_evaluation:
            print(f"\n  Final Evaluation (100 episodes):")
            print(f"    Success Rate:        {ppo_metrics.final_evaluation['success_rate']:6.2%} ± {ppo_metrics.final_evaluation['success_std']:6.2%}")
            print(f"    Mean Reward:         {ppo_metrics.final_evaluation['mean_reward']:8.2f} ± {ppo_metrics.final_evaluation['reward_std']:8.2f}")
            print(f"    Mean Length:         {ppo_metrics.final_evaluation['mean_length']:8.1f} ± {ppo_metrics.final_evaluation['length_std']:8.1f}")
    else:
        print("  No evaluations performed during training")
    
    print("\nGRPO Results:")
    print("-" * 70)
    if grpo_metrics.evaluations_success_rate:
        print(f"  Final Success Rate:    {grpo_metrics.evaluations_success_rate[-1]:6.2%}")
        print(f"  Best Success Rate:     {grpo_metrics.best_success_rate:6.2%}")
        print(f"  Final Avg Reward:      {grpo_metrics.evaluations_rewards[-1]:8.2f}")
        print(f"  Final Avg Length:      {grpo_metrics.evaluations_length[-1]:8.1f}")
        print(f"  Training Time:         {format_time(grpo_metrics.training_time)}")
        
        if grpo_metrics.final_evaluation:
            print(f"\n  Final Evaluation (100 episodes):")
            print(f"    Success Rate:        {grpo_metrics.final_evaluation['success_rate']:6.2%} ± {grpo_metrics.final_evaluation['success_std']:6.2%}")
            print(f"    Mean Reward:         {grpo_metrics.final_evaluation['mean_reward']:8.2f} ± {grpo_metrics.final_evaluation['reward_std']:8.2f}")
            print(f"    Mean Length:         {grpo_metrics.final_evaluation['mean_length']:8.1f} ± {grpo_metrics.final_evaluation['length_std']:8.1f}")
    else:
        print("  No evaluations performed during training")
    
    print("\nComparison:")
    print("-" * 70)
    ppo_best = ppo_metrics.best_success_rate
    grpo_best = grpo_metrics.best_success_rate
    
    if grpo_best > ppo_best:
        improvement = ((grpo_best - ppo_best) / ppo_best) * 100 if ppo_best > 0 else float('inf')
        print(f"  🏆 GRPO outperforms PPO by {improvement:.1f}% (best success rate)")
        print(f"     GRPO: {grpo_best:.2%} | PPO: {ppo_best:.2%}")
    elif ppo_best > grpo_best:
        improvement = ((ppo_best - grpo_best) / grpo_best) * 100 if grpo_best > 0 else float('inf')
        print(f"  🏆 PPO outperforms GRPO by {improvement:.1f}% (best success rate)")
        print(f"     PPO: {ppo_best:.2%} | GRPO: {grpo_best:.2%}")
    else:
        print("  Both algorithms achieved identical performance")
    
    # Training time comparison
    if ppo_metrics.training_time < grpo_metrics.training_time:
        time_diff = grpo_metrics.training_time - ppo_metrics.training_time
        print(f"\n  ⚡ PPO trained {format_time(time_diff)} faster than GRPO")
    elif grpo_metrics.training_time < ppo_metrics.training_time:
        time_diff = ppo_metrics.training_time - grpo_metrics.training_time
        print(f"\n  ⚡ GRPO trained {format_time(time_diff)} faster than PPO")
    
    print("="*70)


def run_training_comparison(config: Optional[TrainingComparisonConfig] = None) -> Dict[str, Any]:
    """Run the GRPO vs PPO comparison experiment with the provided configuration."""

    cfg = config or TrainingComparisonConfig()
    run_dir = cfg.resolve_run_dir()
    env_cfg = cfg.get_env_config()

    ppo_log_dir = run_dir / "logs" / "ppo"
    grpo_log_dir = run_dir / "logs" / "grpo"
    ppo_model_dir = run_dir / "models"
    grpo_model_dir = run_dir / "models"
    results_dir = run_dir / "results"

    print("\n" + "=" * 70)
    print(" " * 10 + "GRPO vs PPO on Sparse Lunar Lander")
    print("=" * 70)
    print("\nExperiment Configuration:")
    print(f"  Run ID:                {run_dir.name}")
    print(f"  Total timesteps:       {cfg.total_timesteps:,}")
    print(f"  Evaluation frequency:  {cfg.eval_frequency:,}")
    print(f"  Random seed:           {cfg.seed}")
    print(f"  Results directory:     {run_dir}/")
    print("\nEnvironment Configuration:")
    print(f"  Soft success condition:    {env_cfg.soft_success_condition}")
    print(f"  Random initial position:   {env_cfg.random_initial_position}")
    print(f"  Success reward:            {env_cfg.success_reward}")
    print(f"  Soft crash reward:         {env_cfg.soft_crash_reward}")
    print("=" * 70)

    ppo_model, ppo_metrics = train_ppo(
        total_timesteps=cfg.total_timesteps,
        eval_freq=cfg.eval_frequency,
        log_dir=str(ppo_log_dir),
        model_save_path=str(ppo_model_dir / "ppo_lunar_lander.pth"),
        best_model_save_path=str(ppo_model_dir / "ppo_lunar_lander_best.pth"),
        seed=cfg.seed,
        env_config=env_cfg,
    )

    grpo_model, grpo_metrics = train_grpo(
        total_timesteps=cfg.total_timesteps,
        eval_freq=cfg.eval_frequency,
        log_dir=str(grpo_log_dir),
        model_save_path=str(grpo_model_dir / "grpo_lunar_lander.pth"),
        best_model_save_path=str(grpo_model_dir / "grpo_lunar_lander_best.pth"),
        seed=cfg.seed,
        env_config=env_cfg,
    )

    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS AND SAVING RESULTS")
    print("=" * 70)
    plot_comparison(ppo_metrics, grpo_metrics, save_dir=str(results_dir))

    save_metrics_csv(ppo_metrics, grpo_metrics, save_dir=str(results_dir))
    save_training_stats(ppo_metrics, grpo_metrics, save_dir=str(results_dir))
    print_final_summary(ppo_metrics, grpo_metrics)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nAll results saved to: {run_dir}/")
    print("\nDirectory structure:")
    print(f"  {run_dir}/")
    print("  ├── models/")
    print("  │   ├── ppo_lunar_lander.pth")
    print("  │   ├── ppo_lunar_lander_best.pth")
    print("  │   ├── grpo_lunar_lander.pth")
    print("  │   └── grpo_lunar_lander_best.pth")
    print("  ├── results/")
    print("  │   ├── comparison_plot.png")
    print("  │   ├── learning_curves.png")
    print("  │   ├── metrics.csv")
    print("  │   ├── evaluation_summary.csv")
    print("  │   ├── evaluation_detailed.csv")
    print("  │   ├── ppo_train_stats.json")
    print("  │   ├── grpo_train_stats.json")
    print("  │   └── learning_curves.json")
    print("  └── logs/")
    print("      ├── ppo/")
    print("      └── grpo/")
    print("=" * 70)

    return {
        "run_dir": run_dir,
        "ppo": {"model": ppo_model, "metrics": ppo_metrics},
        "grpo": {"model": grpo_model, "metrics": grpo_metrics},
    }
