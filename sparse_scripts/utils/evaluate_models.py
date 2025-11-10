"""
Model Evaluation utilities for GRPO and PPO on Sparse Lunar Lander.

Provides helpers to load agents, run evaluation suites, compare results,
and persist plots/metrics for offline analysis. Designed for use as a
Python module rather than a standalone script.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch as th
import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

# Import custom implementations
from ..grpo.grpo import make_grpo_agent, GRPO, PolicyNetwork
from ..ppo.ppo import make_ppo_agent, PPO, ActorCritic
from .sparse_lunar_lander import SparseLunarLander, EnvConfig

__all__ = [
    "EvaluationConfig",
    "EnvConfig",
    "load_ppo_model",
    "load_grpo_model",
    "evaluate_agent",
    "save_videos",
    "plot_comparison",
    "plot_four_way_comparison",
    "save_results_to_csv",
    "save_four_way_results_to_csv",
    "perform_statistical_tests",
    "run_evaluation",
]


@dataclass
class EvaluationConfig:
    """Configuration options for evaluating GRPO and PPO agents."""

    ppo_model_path: Path = field(default_factory=lambda: Path("models/ppo_lunar_lander_best.pth"))
    grpo_model_path: Path = field(default_factory=lambda: Path("models/grpo_lunar_lander_best.pth"))
    num_episodes: int = 100
    num_video_episodes: int = 5
    output_dir: Path = field(default_factory=lambda: Path("./results"))
    device: str = "cpu"
    deterministic: bool = False
    compare_both_modes: bool = False  # If True, compare soft vs deterministic for both algorithms
    seed: int = 42
    env_config: Optional[EnvConfig] = None

    def resolved_output_dir(self) -> Path:
        """Return the output directory as a Path object."""

        return Path(self.output_dir)
    
    def get_env_config(self) -> EnvConfig:
        """Return the environment configuration, using defaults if not provided."""
        return self.env_config or EnvConfig()


def load_ppo_model(model_path: str, device: str = "cpu", env_config: Optional[EnvConfig] = None) -> PPO:
    """
    Load a trained PPO model.
    
    Args:
        model_path: Path to the saved model (.pth file)
        device: Device to load model on
        env_config: Environment configuration (uses defaults if None)
        
    Returns:
        Loaded PPO agent
    """
    print(f"Loading PPO model from: {model_path}")
    
    # Get environment configuration
    env_cfg = env_config or EnvConfig()
    env_kwargs = env_cfg.to_env_kwargs()
    
    # Create environment to get specs
    env = SparseLunarLander(**env_kwargs)
    
    # Create agent
    agent = make_ppo_agent(
        env,
        learning_rate=2e-4,
        device=device,
        verbose=0,
    )
    
    # Load weights
    agent.load(model_path)
    agent.policy.eval()
    
    env.close()
    print("PPO model loaded successfully!")
    return agent


def load_grpo_model(model_path: str, device: str = "cpu", env_config: Optional[EnvConfig] = None) -> GRPO:
    """
    Load a trained GRPO model.
    
    Args:
        model_path: Path to the saved model (.pth file)
        device: Device to load model on
        env_config: Environment configuration (uses defaults if None)
        
    Returns:
        Loaded GRPO agent
    """
    print(f"Loading GRPO model from: {model_path}")
    
    # Get environment configuration
    env_cfg = env_config or EnvConfig()
    env_kwargs = env_cfg.to_env_kwargs()
    
    # Create environment to get specs
    env = SparseLunarLander(**env_kwargs)
    
    # Create agent
    agent = make_grpo_agent(
        env,
        group_size=32,
        learning_rate=2e-4,
        device=device,
        verbose=0,
    )
    
    # Load weights
    agent.load(model_path)
    agent.policy.eval()
    
    env.close()
    print("GRPO model loaded successfully!")
    return agent


def evaluate_agent(
    agent,
    n_episodes: int = 100,
    deterministic: bool = True,
    render_mode: str = None,
    video_folder: str = None,
    env_config: Optional[EnvConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluate an agent for multiple episodes.
    
    Args:
        agent: PPO or GRPO agent
        n_episodes: Number of evaluation episodes
        deterministic: Whether to use deterministic policy
        render_mode: Render mode ('rgb_array' for video recording, None for no rendering)
        video_folder: Folder to save videos (if render_mode='rgb_array')
        env_config: Environment configuration (uses defaults if None)
        
    Returns:
        Dictionary with detailed evaluation results
    """
    # Get environment configuration
    env_cfg = env_config or EnvConfig()
    env_kwargs = env_cfg.to_env_kwargs()
    
    # Create environment
    if render_mode:
        env_kwargs['render_mode'] = render_mode
    
    eval_env = SparseLunarLander(**env_kwargs)
    
    # Storage for results
    successes = []
    episode_rewards = []
    episode_lengths = []
    crash_details = []
    landing_details = []
    
    # Video recording setup
    frames = [] if render_mode == 'rgb_array' else None
    
    print(f"\nEvaluating for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_frames = []
        
        while not done:
            # Get action from model
            if isinstance(agent, PPO):
                obs_tensor = agent._obs_to_tensor(obs)
                dist, _ = agent.policy.dist(obs_tensor)
                if deterministic:
                    if agent.action_space_type == "discrete":
                        action = th.argmax(dist.probs, dim=-1)
                        act = int(action.item())
                    else:
                        act = dist.mean.cpu().numpy()[0]
                else:
                    if agent.action_space_type == "discrete":
                        act = int(dist.sample().item())
                    else:
                        act = dist.sample().cpu().numpy()[0]
            elif isinstance(agent, GRPO):
                obs_tensor = th.tensor(np.array([obs]), dtype=th.float32, device=agent.device)
                with th.no_grad():
                    action, _ = agent.policy.get_action_and_log_prob(obs_tensor, deterministic=deterministic)
                if agent.policy.action_space_type == "discrete":
                    act = int(action.item())
                else:
                    act = action.cpu().numpy()[0]
            else:
                raise ValueError(f"Unknown agent type: {type(agent)}")
            
            # Step environment
            obs, reward, terminated, truncated, info = eval_env.step(act)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            # Record frame for video
            if render_mode == 'rgb_array' and video_folder and episode < 5:
                frame = eval_env.render()
                episode_frames.append(frame)
            
            if done:
                success = info.get('landing_success', False)
                successes.append(1.0 if success else 0.0)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                if success:
                    landing_details.append({
                        'episode': episode,
                        'reward': episode_reward,
                        'length': episode_length,
                    })
                else:
                    crash_details.append({
                        'episode': episode,
                        'velocity': info.get('crash_velocity', 0),
                        'angle': info.get('crash_angle', 0),
                        'distance': info.get('distance_from_pad', 0),
                        'legs_touching': info.get('legs_touching', 0),
                        'out_of_bounds': info.get('out_of_bounds', False),
                        'penalty': info.get('crash_penalty', 0),
                        'reward': episode_reward,
                        'length': episode_length,
                    })
        
        # Store episode frames
        if episode_frames and video_folder and episode < 5:
            frames.append(episode_frames)
        
        # Progress indicator
        if (episode + 1) % 10 == 0:
            current_success_rate = np.mean(successes)
            current_mean_reward = np.mean(episode_rewards)
            print(f"  Episode {episode + 1}/{n_episodes} | "
                  f"Success Rate: {current_success_rate:.2%} | "
                  f"Mean Reward: {current_mean_reward:.2f}")
    
    eval_env.close()
    
    # Calculate statistics
    rewards_array = np.array(episode_rewards)
    lengths_array = np.array(episode_lengths)
    successes_array = np.array(successes)
    
    results = {
        'n_episodes': n_episodes,
        'success_rate': float(np.mean(successes_array)),
        'success_std': float(np.std(successes_array)),
        'mean_reward': float(np.mean(rewards_array)),
        'std_reward': float(np.std(rewards_array)),
        'min_reward': float(np.min(rewards_array)),
        'max_reward': float(np.max(rewards_array)),
        'median_reward': float(np.median(rewards_array)),
        'mean_length': float(np.mean(lengths_array)),
        'std_length': float(np.std(lengths_array)),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'successes': successes,
        'crash_details': crash_details,
        'landing_details': landing_details,
        'frames': frames,
    }
    
    print(f"\nEvaluation complete!")
    print(f"  Success Rate:  {results['success_rate']:.2%} ± {results['success_std']:.4f}")
    print(f"  Mean Reward:   {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Median Reward: {results['median_reward']:.2f}")
    print(f"  Min Reward:    {results['min_reward']:.2f}")
    print(f"  Max Reward:    {results['max_reward']:.2f}")
    print(f"  Mean Length:   {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"  Successes:     {int(np.sum(successes_array))}/{n_episodes}")
    
    return results


def save_videos(frames_list: List[List[np.ndarray]], save_dir: str, prefix: str = "episode"):
    """
    Save episode frames as videos using imageio.
    
    Args:
        frames_list: List of episode frames
        save_dir: Directory to save videos
        prefix: Prefix for video filenames
    """
    try:
        import imageio
    except ImportError:
        print("Warning: imageio not installed. Cannot save videos.")
        print("Install with: pip install imageio imageio-ffmpeg")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    for i, frames in enumerate(frames_list):
        if not frames:
            continue
        
        video_path = os.path.join(save_dir, f"{prefix}_{i+1}.mp4")
        
        try:
            # Save as MP4 video
            writer = imageio.get_writer(video_path, fps=50)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"Video saved: {video_path}")
        except Exception as e:
            print(f"Error saving video {i+1}: {e}")


def plot_four_way_comparison(
    ppo_soft_results: Dict[str, Any],
    ppo_det_results: Dict[str, Any],
    grpo_soft_results: Dict[str, Any],
    grpo_det_results: Dict[str, Any],
    save_dir: str = "./results"
):
    """
    Generate comprehensive comparison plots for all four modes.
    
    Args:
        ppo_soft_results: PPO soft policy evaluation results
        ppo_det_results: PPO deterministic policy evaluation results
        grpo_soft_results: GRPO soft policy evaluation results
        grpo_det_results: GRPO deterministic policy evaluation results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # ========================================================================
    # FOUR-WAY COMPREHENSIVE COMPARISON PLOT
    # ========================================================================
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Complete Comparison: GRPO vs PPO × Soft vs Deterministic (100 Episodes Each)', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    all_results = {
        'PPO Soft': ppo_soft_results,
        'PPO Det': ppo_det_results,
        'GRPO Soft': grpo_soft_results,
        'GRPO Det': grpo_det_results,
    }
    
    algorithms = list(all_results.keys())
    colors = ['tab:blue', 'dodgerblue', 'tab:orange', 'darkorange']
    
    # Row 1, Col 1: Success Rate Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    success_rates = [res['success_rate'] for res in all_results.values()]
    
    bars = ax1.bar(algorithms, success_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Success Rate Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Row 1, Col 2: Average Reward
    ax2 = fig.add_subplot(gs[0, 1])
    mean_rewards = [res['mean_reward'] for res in all_results.values()]
    std_rewards = [res['std_reward'] for res in all_results.values()]
    
    bars = ax2.bar(algorithms, mean_rewards, yerr=std_rewards, capsize=8,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Average Reward (±1 std)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Success (100)', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.tick_params(axis='x', rotation=15)
    
    for i, (bar, val, std) in enumerate(zip(bars, mean_rewards, std_rewards)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Row 1, Col 3: Average Episode Length
    ax3 = fig.add_subplot(gs[0, 2])
    mean_lengths = [res['mean_length'] for res in all_results.values()]
    std_lengths = [res['std_length'] for res in all_results.values()]
    
    bars = ax3.bar(algorithms, mean_lengths, yerr=std_lengths, capsize=8,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Episode Length', fontsize=12, fontweight='bold')
    ax3.set_title('Average Episode Length (±1 std)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=15)
    
    for i, (bar, val, std) in enumerate(zip(bars, mean_lengths, std_lengths)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + std + 10,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Row 1, Col 4: Reward Box Plot
    ax4 = fig.add_subplot(gs[0, 3])
    data_to_plot = [res['episode_rewards'] for res in all_results.values()]
    bp = ax4.boxplot(data_to_plot, tick_labels=algorithms, patch_artist=True, showmeans=True,
                     meanprops=dict(marker='o', markerfacecolor='red', markersize=6, markeredgecolor='darkred'))
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.set_ylabel('Reward', fontsize=11, fontweight='bold')
    ax4.set_title('Reward Distribution', fontsize=13, fontweight='bold')
    ax4.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=15)
    
    # Row 2: Reward Histograms for each mode
    for idx, (name, res) in enumerate(all_results.items()):
        ax = fig.add_subplot(gs[1, idx])
        ax.hist(res['episode_rewards'], bins=20, color=colors[idx], alpha=0.7, edgecolor='black')
        ax.axvline(res['mean_reward'], color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {res["mean_reward"]:.1f}')
        ax.axvline(100, color='green', linestyle='--', linewidth=2, label='Success: 100', alpha=0.7)
        ax.set_xlabel('Reward', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.set_title(f'{name} Reward Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Episode Length Histograms
    for idx, (name, res) in enumerate(all_results.items()):
        ax = fig.add_subplot(gs[2, idx])
        ax.hist(res['episode_lengths'], bins=20, color=colors[idx], alpha=0.7, edgecolor='black')
        ax.axvline(res['mean_length'], color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {res["mean_length"]:.1f}')
        ax.set_xlabel('Episode Length', fontsize=10, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax.set_title(f'{name} Length Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Row 4, Cols 1-2: Within-Algorithm Comparison (PPO)
    ax_ppo = fig.add_subplot(gs[3, 0:2])
    ppo_modes = ['PPO Soft', 'PPO Det']
    ppo_success = [ppo_soft_results['success_rate'], ppo_det_results['success_rate']]
    ppo_reward = [ppo_soft_results['mean_reward'], ppo_det_results['mean_reward']]
    
    x = np.arange(len(ppo_modes))
    width = 0.35
    
    bars1 = ax_ppo.bar(x - width/2, ppo_success, width, label='Success Rate', 
                       color='tab:blue', alpha=0.7, edgecolor='black')
    ax_ppo2 = ax_ppo.twinx()
    bars2 = ax_ppo2.bar(x + width/2, ppo_reward, width, label='Avg Reward', 
                        color='tab:green', alpha=0.7, edgecolor='black')
    
    ax_ppo.set_ylabel('Success Rate', fontsize=11, fontweight='bold', color='tab:blue')
    ax_ppo2.set_ylabel('Average Reward', fontsize=11, fontweight='bold', color='tab:green')
    ax_ppo.set_xlabel('Policy Mode', fontsize=11, fontweight='bold')
    ax_ppo.set_title('PPO: Soft vs Deterministic', fontsize=13, fontweight='bold')
    ax_ppo.set_xticks(x)
    ax_ppo.set_xticklabels(ppo_modes)
    ax_ppo.set_ylim([0, 1.0])
    ax_ppo.tick_params(axis='y', labelcolor='tab:blue')
    ax_ppo2.tick_params(axis='y', labelcolor='tab:green')
    ax_ppo.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, ppo_success):
        height = bar.get_height()
        ax_ppo.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, ppo_reward):
        height = bar.get_height()
        ax_ppo2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Row 4, Cols 3-4: Within-Algorithm Comparison (GRPO)
    ax_grpo = fig.add_subplot(gs[3, 2:4])
    grpo_modes = ['GRPO Soft', 'GRPO Det']
    grpo_success = [grpo_soft_results['success_rate'], grpo_det_results['success_rate']]
    grpo_reward = [grpo_soft_results['mean_reward'], grpo_det_results['mean_reward']]
    
    x = np.arange(len(grpo_modes))
    
    bars1 = ax_grpo.bar(x - width/2, grpo_success, width, label='Success Rate', 
                        color='tab:orange', alpha=0.7, edgecolor='black')
    ax_grpo2 = ax_grpo.twinx()
    bars2 = ax_grpo2.bar(x + width/2, grpo_reward, width, label='Avg Reward', 
                         color='tab:green', alpha=0.7, edgecolor='black')
    
    ax_grpo.set_ylabel('Success Rate', fontsize=11, fontweight='bold', color='tab:orange')
    ax_grpo2.set_ylabel('Average Reward', fontsize=11, fontweight='bold', color='tab:green')
    ax_grpo.set_xlabel('Policy Mode', fontsize=11, fontweight='bold')
    ax_grpo.set_title('GRPO: Soft vs Deterministic', fontsize=13, fontweight='bold')
    ax_grpo.set_xticks(x)
    ax_grpo.set_xticklabels(grpo_modes)
    ax_grpo.set_ylim([0, 1.0])
    ax_grpo.tick_params(axis='y', labelcolor='tab:orange')
    ax_grpo2.tick_params(axis='y', labelcolor='tab:green')
    ax_grpo.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, grpo_success):
        height = bar.get_height()
        ax_grpo.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, grpo_reward):
        height = bar.get_height()
        ax_grpo2.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save comprehensive plot
    output_dir = save_dir
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'evaluation_comparison_four_way.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved four-way comparison plot to {plot_path}")
    plt.close()
    
    # Additional statistical summary table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Metric', 'PPO Soft', 'PPO Det', 'GRPO Soft', 'GRPO Det'],
        ['Success Rate', 
         f'{ppo_soft_results["success_rate"]:.1%}',
         f'{ppo_det_results["success_rate"]:.1%}',
         f'{grpo_soft_results["success_rate"]:.1%}',
         f'{grpo_det_results["success_rate"]:.1%}'],
        ['Mean Reward',
         f'{ppo_soft_results["mean_reward"]:.2f}',
         f'{ppo_det_results["mean_reward"]:.2f}',
         f'{grpo_soft_results["mean_reward"]:.2f}',
         f'{grpo_det_results["mean_reward"]:.2f}'],
        ['Std Reward',
         f'{ppo_soft_results["std_reward"]:.2f}',
         f'{ppo_det_results["std_reward"]:.2f}',
         f'{grpo_soft_results["std_reward"]:.2f}',
         f'{grpo_det_results["std_reward"]:.2f}'],
        ['Mean Length',
         f'{ppo_soft_results["mean_length"]:.1f}',
         f'{ppo_det_results["mean_length"]:.1f}',
         f'{grpo_soft_results["mean_length"]:.1f}',
         f'{grpo_det_results["mean_length"]:.1f}'],
        ['Min Reward',
         f'{ppo_soft_results["min_reward"]:.1f}',
         f'{ppo_det_results["min_reward"]:.1f}',
         f'{grpo_soft_results["min_reward"]:.1f}',
         f'{grpo_det_results["min_reward"]:.1f}'],
        ['Max Reward',
         f'{ppo_soft_results["max_reward"]:.1f}',
         f'{ppo_det_results["max_reward"]:.1f}',
         f'{grpo_soft_results["max_reward"]:.1f}',
         f'{grpo_det_results["max_reward"]:.1f}'],
        ['Successes',
         f'{int(np.sum(ppo_soft_results["successes"]))}/{ppo_soft_results["n_episodes"]}',
         f'{int(np.sum(ppo_det_results["successes"]))}/{ppo_det_results["n_episodes"]}',
         f'{int(np.sum(grpo_soft_results["successes"]))}/{grpo_soft_results["n_episodes"]}',
         f'{int(np.sum(grpo_det_results["successes"]))}/{grpo_det_results["n_episodes"]}'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#40466e')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(5):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('#ffffff')
            cell.set_text_props(fontsize=10)
            
            if j > 0:
                if j <= 2:
                    cell.set_text_props(color='tab:blue', weight='bold')
                else:
                    cell.set_text_props(color='tab:orange', weight='bold')
    
    ax.set_title('Four-Way Statistical Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'statistical_summary_four_way.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved four-way statistical summary to {summary_path}")
    plt.close()
    
    return plot_path


def plot_comparison(
    ppo_results: Dict[str, Any],
    grpo_results: Dict[str, Any],
    save_dir: str = "./results"
):
    """
    Generate comprehensive comparison plots matching the reference image layout.
    
    Args:
        ppo_results: PPO evaluation results
        grpo_results: GRPO evaluation results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    
    # ========================================================================
    # 1. COMPREHENSIVE EVALUATION PLOT (3x3 grid matching reference image)
    # ========================================================================
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('GRPO vs PPO - Comprehensive Evaluation (100 Episodes Each)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Row 1, Col 1: Success Rate Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    algorithms = ['PPO', 'GRPO']
    success_rates = [ppo_results['success_rate'], grpo_results['success_rate']]
    success_stds = [ppo_results['success_std'], grpo_results['success_std']]
    
    bars = ax1.bar(algorithms, success_rates, color=['tab:blue', 'tab:orange'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Success Rate Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, val in zip(bars, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.1%}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Row 1, Col 2: Average Reward with std
    ax2 = fig.add_subplot(gs[0, 1])
    mean_rewards = [ppo_results['mean_reward'], grpo_results['mean_reward']]
    std_rewards = [ppo_results['std_reward'], grpo_results['std_reward']]
    
    bars = ax2.bar(algorithms, mean_rewards, yerr=std_rewards, capsize=10,
                   color=['tab:blue', 'tab:orange'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Average Reward (±1 std)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Success Threshold (100)', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=9)
    
    # Add value labels
    for bar, val in zip(bars, mean_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std_rewards[algorithms.index('PPO' if bar.get_facecolor()[:3] == (0.12156862745098039, 0.4666666666666667, 0.7058823529411765) else 'GRPO')] + 5,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Row 1, Col 3: Average Episode Length with std
    ax3 = fig.add_subplot(gs[0, 2])
    mean_lengths = [ppo_results['mean_length'], grpo_results['mean_length']]
    std_lengths = [ppo_results['std_length'], grpo_results['std_length']]
    
    bars = ax3.bar(algorithms, mean_lengths, yerr=std_lengths, capsize=10,
                   color=['tab:blue', 'tab:orange'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Episode Length', fontsize=12, fontweight='bold')
    ax3.set_title('Average Episode Length (±1 std)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, mean_lengths):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + std_lengths[algorithms.index('PPO' if bar == bars[0] else 'GRPO')] + 10,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Row 2, Col 1: PPO Reward Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(ppo_results['episode_rewards'], bins=20, color='tab:blue', alpha=0.7, edgecolor='black')
    ax4.axvline(ppo_results['mean_reward'], color='red', linestyle='--', linewidth=2, label=f'Mean: {ppo_results["mean_reward"]:.1f}')
    ax4.axvline(100, color='green', linestyle='--', linewidth=2, label='Success: 100', alpha=0.7)
    ax4.set_xlabel('Reward', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('PPO Reward Distribution', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Row 2, Col 2: GRPO Reward Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(grpo_results['episode_rewards'], bins=20, color='tab:orange', alpha=0.7, edgecolor='black')
    ax5.axvline(grpo_results['mean_reward'], color='darkred', linestyle='--', linewidth=2, label=f'Mean: {grpo_results["mean_reward"]:.1f}')
    ax5.axvline(100, color='green', linestyle='--', linewidth=2, label='Success: 100', alpha=0.7)
    ax5.set_xlabel('Reward', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('GRPO Reward Distribution', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Row 2, Col 3: Reward Distribution Box Plot
    ax6 = fig.add_subplot(gs[1, 2])
    data_to_plot = [ppo_results['episode_rewards'], grpo_results['episode_rewards']]
    bp = ax6.boxplot(data_to_plot, labels=algorithms, patch_artist=True, showmeans=True, meanline=False,
                     meanprops=dict(marker='o', markerfacecolor='red', markersize=8, markeredgecolor='darkred'))
    
    colors = ['tab:blue', 'tab:orange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax6.set_ylabel('Reward', fontsize=11, fontweight='bold')
    ax6.set_title('Reward Distribution (Box Plot)', fontsize=13, fontweight='bold')
    ax6.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Success (100)', alpha=0.7)
    ax6.legend(fontsize=9, loc='upper left')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Row 3, Col 1: PPO Episode Length Distribution
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(ppo_results['episode_lengths'], bins=20, color='tab:blue', alpha=0.7, edgecolor='black')
    ax7.axvline(ppo_results['mean_length'], color='red', linestyle='--', linewidth=2, label=f'Mean: {ppo_results["mean_length"]:.1f}')
    ax7.set_xlabel('Episode Length', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax7.set_title('PPO Episode Length Distribution', fontsize=13, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Row 3, Col 2: GRPO Episode Length Distribution
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(grpo_results['episode_lengths'], bins=20, color='tab:orange', alpha=0.7, edgecolor='black')
    ax8.axvline(grpo_results['mean_length'], color='darkred', linestyle='--', linewidth=2, label=f'Mean: {grpo_results["mean_length"]:.1f}')
    ax8.set_xlabel('Episode Length', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax8.set_title('GRPO Episode Length Distribution', fontsize=13, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Row 3, Col 3: Statistical Summary Table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('tight')
    ax9.axis('off')
    
    # Prepare statistical summary data
    table_data = [
        ['Metric', 'PPO', 'GRPO'],
        ['Mean Reward', f'{ppo_results["mean_reward"]:.2f}', f'{grpo_results["mean_reward"]:.2f}'],
        ['Std Reward', f'{ppo_results["std_reward"]:.2f}', f'{grpo_results["std_reward"]:.2f}'],
        ['Success Rate', f'{ppo_results["success_rate"]:.1%}', f'{grpo_results["success_rate"]:.1%}'],
        ['Mean Length', f'{ppo_results["mean_length"]:.1f}', f'{grpo_results["mean_length"]:.1f}'],
        ['Std Length', f'{ppo_results["std_length"]:.1f}', f'{grpo_results["std_length"]:.1f}'],
        ['Min Reward', f'{ppo_results["min_reward"]:.1f}', f'{grpo_results["min_reward"]:.1f}'],
        ['Max Reward', f'{ppo_results["max_reward"]:.1f}', f'{grpo_results["max_reward"]:.1f}'],
    ]
    
    table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style header row
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#40466e')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Style data rows with alternating colors
    for i in range(1, len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('#ffffff')
            cell.set_text_props(fontsize=10)
            
            # Highlight values
            if j == 1:  # PPO column
                cell.set_text_props(color='tab:blue', weight='bold')
            elif j == 2:  # GRPO column
                cell.set_text_props(color='tab:orange', weight='bold')
    
    ax9.set_title('Statistical Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save comprehensive plot
    output_dir = save_dir  # Use save_dir directly as output_dir
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'evaluation_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comprehensive comparison plot to {plot_path}")
    plt.close()
    
    return plot_path
    
    sns.violinplot(data=data_for_violin, x='Algorithm', y='Reward', 
                   palette=['tab:blue', 'tab:orange'], ax=ax)
    ax.set_ylabel('Episode Reward', fontsize=14, fontweight='bold')
    ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax.set_title('Reward Distribution (Violin Plot)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'reward_violin.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Violin plot saved: {save_path}")
    plt.close()
    
    # 4. Success/Failure Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Landing Success Analysis', fontsize=16, fontweight='bold')
    
    # Success count pie chart
    ax = axes[0]
    ppo_successes = int(np.sum(ppo_results['successes']))
    ppo_failures = len(ppo_results['successes']) - ppo_successes
    
    ax.pie([ppo_successes, ppo_failures], labels=['Success', 'Failure'],
           autopct='%1.1f%%', startangle=90, colors=['#90EE90', '#FFB6C1'])
    ax.set_title(f'PPO: {ppo_successes}/{len(ppo_results["successes"])} Successful Landings',
                 fontsize=12, fontweight='bold')
    
    ax = axes[1]
    grpo_successes = int(np.sum(grpo_results['successes']))
    grpo_failures = len(grpo_results['successes']) - grpo_successes
    
    ax.pie([grpo_successes, grpo_failures], labels=['Success', 'Failure'],
           autopct='%1.1f%%', startangle=90, colors=['#90EE90', '#FFB6C1'])
    ax.set_title(f'GRPO: {grpo_successes}/{len(grpo_results["successes"])} Successful Landings',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'success_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success analysis plot saved: {save_path}")
    plt.close()


def save_four_way_results_to_csv(
    ppo_soft_results: Dict[str, Any],
    ppo_det_results: Dict[str, Any],
    grpo_soft_results: Dict[str, Any],
    grpo_det_results: Dict[str, Any],
    save_dir: str = "./results"
):
    """
    Save four-way evaluation results to CSV files.
    
    Args:
        ppo_soft_results: PPO soft policy evaluation results
        ppo_det_results: PPO deterministic policy evaluation results
        grpo_soft_results: GRPO soft policy evaluation results
        grpo_det_results: GRPO deterministic policy evaluation results
        save_dir: Directory to save CSV files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Episode-by-episode results
    episode_data = {
        'episode': list(range(1, len(ppo_soft_results['episode_rewards']) + 1)),
        'ppo_soft_reward': ppo_soft_results['episode_rewards'],
        'ppo_soft_length': ppo_soft_results['episode_lengths'],
        'ppo_soft_success': ppo_soft_results['successes'],
        'ppo_det_reward': ppo_det_results['episode_rewards'],
        'ppo_det_length': ppo_det_results['episode_lengths'],
        'ppo_det_success': ppo_det_results['successes'],
        'grpo_soft_reward': grpo_soft_results['episode_rewards'],
        'grpo_soft_length': grpo_soft_results['episode_lengths'],
        'grpo_soft_success': grpo_soft_results['successes'],
        'grpo_det_reward': grpo_det_results['episode_rewards'],
        'grpo_det_length': grpo_det_results['episode_lengths'],
        'grpo_det_success': grpo_det_results['successes'],
    }
    
    df_episodes = pd.DataFrame(episode_data)
    episodes_path = os.path.join(save_dir, 'episode_results_four_way.csv')
    df_episodes.to_csv(episodes_path, index=False)
    print(f"\nFour-way episode results saved: {episodes_path}")
    
    # Summary statistics
    summary_data = {
        'Metric': [
            'Success Rate', 'Success Std', 'Mean Reward', 'Std Reward',
            'Median Reward', 'Min Reward', 'Max Reward', 'Mean Length', 'Std Length',
            'Total Successes', 'Total Episodes'
        ],
        'PPO_Soft': [
            ppo_soft_results['success_rate'], ppo_soft_results['success_std'],
            ppo_soft_results['mean_reward'], ppo_soft_results['std_reward'],
            ppo_soft_results['median_reward'], ppo_soft_results['min_reward'],
            ppo_soft_results['max_reward'], ppo_soft_results['mean_length'],
            ppo_soft_results['std_length'], int(np.sum(ppo_soft_results['successes'])),
            ppo_soft_results['n_episodes']
        ],
        'PPO_Deterministic': [
            ppo_det_results['success_rate'], ppo_det_results['success_std'],
            ppo_det_results['mean_reward'], ppo_det_results['std_reward'],
            ppo_det_results['median_reward'], ppo_det_results['min_reward'],
            ppo_det_results['max_reward'], ppo_det_results['mean_length'],
            ppo_det_results['std_length'], int(np.sum(ppo_det_results['successes'])),
            ppo_det_results['n_episodes']
        ],
        'GRPO_Soft': [
            grpo_soft_results['success_rate'], grpo_soft_results['success_std'],
            grpo_soft_results['mean_reward'], grpo_soft_results['std_reward'],
            grpo_soft_results['median_reward'], grpo_soft_results['min_reward'],
            grpo_soft_results['max_reward'], grpo_soft_results['mean_length'],
            grpo_soft_results['std_length'], int(np.sum(grpo_soft_results['successes'])),
            grpo_soft_results['n_episodes']
        ],
        'GRPO_Deterministic': [
            grpo_det_results['success_rate'], grpo_det_results['success_std'],
            grpo_det_results['mean_reward'], grpo_det_results['std_reward'],
            grpo_det_results['median_reward'], grpo_det_results['min_reward'],
            grpo_det_results['max_reward'], grpo_det_results['mean_length'],
            grpo_det_results['std_length'], int(np.sum(grpo_det_results['successes'])),
            grpo_det_results['n_episodes']
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(save_dir, 'summary_statistics_four_way.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"Four-way summary statistics saved: {summary_path}")


def save_results_to_csv(
    ppo_results: Dict[str, Any],
    grpo_results: Dict[str, Any],
    save_dir: str = "./results"
):
    """
    Save evaluation results to CSV files.
    
    Args:
        ppo_results: PPO evaluation results
        grpo_results: GRPO evaluation results
        save_dir: Directory to save CSV files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Episode-by-episode results
    episode_data = {
        'episode': list(range(1, len(ppo_results['episode_rewards']) + 1)),
        'ppo_reward': ppo_results['episode_rewards'],
        'ppo_length': ppo_results['episode_lengths'],
        'ppo_success': ppo_results['successes'],
        'grpo_reward': grpo_results['episode_rewards'],
        'grpo_length': grpo_results['episode_lengths'],
        'grpo_success': grpo_results['successes'],
    }
    
    df_episodes = pd.DataFrame(episode_data)
    episodes_path = os.path.join(save_dir, 'episode_results.csv')
    df_episodes.to_csv(episodes_path, index=False)
    print(f"\nEpisode results saved: {episodes_path}")
    
    # Summary statistics
    summary_data = {
        'Metric': [
            'Success Rate', 'Success Std', 'Mean Reward', 'Std Reward',
            'Median Reward', 'Min Reward', 'Max Reward', 'Mean Length', 'Std Length',
            'Total Successes', 'Total Episodes'
        ],
        'PPO': [
            ppo_results['success_rate'], ppo_results['success_std'],
            ppo_results['mean_reward'], ppo_results['std_reward'],
            ppo_results['median_reward'], ppo_results['min_reward'],
            ppo_results['max_reward'], ppo_results['mean_length'],
            ppo_results['std_length'], int(np.sum(ppo_results['successes'])),
            ppo_results['n_episodes']
        ],
        'GRPO': [
            grpo_results['success_rate'], grpo_results['success_std'],
            grpo_results['mean_reward'], grpo_results['std_reward'],
            grpo_results['median_reward'], grpo_results['min_reward'],
            grpo_results['max_reward'], grpo_results['mean_length'],
            grpo_results['std_length'], int(np.sum(grpo_results['successes'])),
            grpo_results['n_episodes']
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(save_dir, 'summary_statistics.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary statistics saved: {summary_path}")
    
    # Crash details
    if ppo_results['crash_details'] or grpo_results['crash_details']:
        crash_data = []
        
        for crash in ppo_results['crash_details']:
            crash_data.append({
                'algorithm': 'PPO',
                **crash
            })
        
        for crash in grpo_results['crash_details']:
            crash_data.append({
                'algorithm': 'GRPO',
                **crash
            })
        
        df_crashes = pd.DataFrame(crash_data)
        crashes_path = os.path.join(save_dir, 'crash_details.csv')
        df_crashes.to_csv(crashes_path, index=False)
        print(f"Crash details saved: {crashes_path}")


def perform_statistical_tests(
    ppo_results: Dict[str, Any],
    grpo_results: Dict[str, Any]
):
    """
    Perform statistical tests to compare PPO and GRPO.
    
    Args:
        ppo_results: PPO evaluation results
        grpo_results: GRPO evaluation results
    """
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*70)
    
    ppo_rewards = np.array(ppo_results['episode_rewards'])
    grpo_rewards = np.array(grpo_results['episode_rewards'])
    
    sufficient_samples = len(ppo_rewards) > 1 and len(grpo_rewards) > 1
    if sufficient_samples:
        t_stat, t_pval = stats.ttest_ind(ppo_rewards, grpo_rewards)
        print(f"\n1. Two-Sample T-Test (Rewards):")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value:     {t_pval:.4f}")
        if t_pval < 0.05:
            better = "GRPO" if grpo_results['mean_reward'] > ppo_results['mean_reward'] else "PPO"
            print(f"   Result:      {better} is significantly better (p < 0.05)")
        else:
            print(f"   Result:      No significant difference (p >= 0.05)")
    else:
        print("\n1. Two-Sample T-Test (Rewards): skipped (not enough samples)")
    
    if sufficient_samples:
        u_stat, u_pval = stats.mannwhitneyu(ppo_rewards, grpo_rewards, alternative='two-sided')
        print(f"\n2. Mann-Whitney U Test (Rewards):")
        print(f"   U-statistic: {u_stat:.4f}")
        print(f"   p-value:     {u_pval:.4f}")
        if u_pval < 0.05:
            better = "GRPO" if grpo_results['mean_reward'] > ppo_results['mean_reward'] else "PPO"
            print(f"   Result:      {better} is significantly better (p < 0.05)")
        else:
            print(f"   Result:      No significant difference (p >= 0.05)")
    else:
        print("\n2. Mann-Whitney U Test (Rewards): skipped (not enough samples)")
    
    ppo_successes = int(np.sum(ppo_results['successes']))
    ppo_failures = len(ppo_results['successes']) - ppo_successes
    grpo_successes = int(np.sum(grpo_results['successes']))
    grpo_failures = len(grpo_results['successes']) - grpo_successes

    total_successes = ppo_successes + grpo_successes
    total_failures = ppo_failures + grpo_failures

    if total_successes > 0 and total_failures > 0:
        contingency_table = np.array([[ppo_successes, ppo_failures],
                                       [grpo_successes, grpo_failures]])
        chi2, chi_pval, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\n3. Chi-Square Test (Success Rates):")
        print(f"   χ²-statistic: {chi2:.4f}")
        print(f"   p-value:      {chi_pval:.4f}")
        print(f"   DoF:          {dof}")
        if chi_pval < 0.05:
            better = "GRPO" if grpo_results['success_rate'] > ppo_results['success_rate'] else "PPO"
            print(f"   Result:       {better} has significantly better success rate (p < 0.05)")
        else:
            print(f"   Result:       No significant difference in success rates (p >= 0.05)")
    else:
        print("\n3. Chi-Square Test (Success Rates): skipped (insufficient success/failure counts)")
    
    pooled_variance = (ppo_results['std_reward']**2 + grpo_results['std_reward']**2) / 2
    if pooled_variance > 0:
        pooled_std = np.sqrt(pooled_variance)
        cohens_d = (grpo_results['mean_reward'] - ppo_results['mean_reward']) / pooled_std
        print(f"\n4. Effect Size (Cohen's d):")
        print(f"   d = {cohens_d:.4f}")
        if abs(cohens_d) < 0.2:
            print("   Interpretation: Negligible effect")
        elif abs(cohens_d) < 0.5:
            print("   Interpretation: Small effect")
        elif abs(cohens_d) < 0.8:
            print("   Interpretation: Medium effect")
        else:
            print("   Interpretation: Large effect")
    else:
        print("\n4. Effect Size (Cohen's d): skipped (zero variance in rewards)")
    
    print("="*70)


def run_evaluation(config: Optional[EvaluationConfig] = None) -> Dict[str, Any]:
    """Run the evaluation pipeline for trained PPO and GRPO agents."""

    cfg = config or EvaluationConfig()
    output_dir = cfg.resolved_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    env_cfg = cfg.get_env_config()

    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)

    print("\n" + "=" * 70)
    if cfg.compare_both_modes:
        print(" " * 5 + "FOUR-WAY EVALUATION: GRPO vs PPO × Soft vs Deterministic")
    else:
        print(" " * 15 + "MODEL EVALUATION: GRPO vs PPO")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  PPO Model:         {cfg.ppo_model_path}")
    print(f"  GRPO Model:        {cfg.grpo_model_path}")
    print(f"  Evaluation Episodes: {cfg.num_episodes}")
    print(f"  Video Episodes:    {cfg.num_video_episodes}")
    print(f"  Output Directory:  {output_dir}")
    print(f"  Device:            {cfg.device}")
    if cfg.compare_both_modes:
        print(f"  Comparison Mode:   Four-way (Soft vs Deterministic for both algorithms)")
    else:
        print(f"  Policy Mode:       {'Deterministic' if cfg.deterministic else 'Stochastic (Soft)'}")
    print("\nEnvironment Configuration:")
    print(f"  Soft success condition:    {env_cfg.soft_success_condition}")
    print(f"  Random initial position:   {env_cfg.random_initial_position}")
    print(f"  Success reward:            {env_cfg.success_reward}")
    print(f"  Soft crash reward:         {env_cfg.soft_crash_reward}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)
    ppo_agent = load_ppo_model(str(cfg.ppo_model_path), device=cfg.device, env_config=env_cfg)
    grpo_agent = load_grpo_model(str(cfg.grpo_model_path), device=cfg.device, env_config=env_cfg)

    if cfg.compare_both_modes:
        # Four-way comparison: evaluate all combinations
        print("\n" + "=" * 70)
        print("EVALUATING PPO (SOFT POLICY)")
        print("=" * 70)
        ppo_soft_results = evaluate_agent(
            ppo_agent,
            n_episodes=cfg.num_episodes,
            deterministic=False,
            render_mode=None,
            env_config=env_cfg,
        )

        print("\n" + "=" * 70)
        print("EVALUATING PPO (DETERMINISTIC POLICY)")
        print("=" * 70)
        ppo_det_results = evaluate_agent(
            ppo_agent,
            n_episodes=cfg.num_episodes,
            deterministic=True,
            render_mode=None,
            env_config=env_cfg,
        )

        print("\n" + "=" * 70)
        print("EVALUATING GRPO (SOFT POLICY)")
        print("=" * 70)
        grpo_soft_results = evaluate_agent(
            grpo_agent,
            n_episodes=cfg.num_episodes,
            deterministic=False,
            render_mode=None,
            env_config=env_cfg,
        )

        print("\n" + "=" * 70)
        print("EVALUATING GRPO (DETERMINISTIC POLICY)")
        print("=" * 70)
        grpo_det_results = evaluate_agent(
            grpo_agent,
            n_episodes=cfg.num_episodes,
            deterministic=True,
            render_mode=None,
            env_config=env_cfg,
        )

        video_outputs: Dict[str, Any] = {}
        if cfg.num_video_episodes > 0:
            print("\n" + "=" * 70)
            print("RECORDING VIDEOS")
            print("=" * 70)

            ppo_soft_video_dir = Path(output_dir) / "videos" / "ppo_soft"
            ppo_det_video_dir = Path(output_dir) / "videos" / "ppo_deterministic"
            grpo_soft_video_dir = Path(output_dir) / "videos" / "grpo_soft"
            grpo_det_video_dir = Path(output_dir) / "videos" / "grpo_deterministic"

            print("\nRecording PPO Soft episodes...")
            ppo_soft_video_results = evaluate_agent(
                ppo_agent,
                n_episodes=cfg.num_video_episodes,
                deterministic=False,
                render_mode="rgb_array",
                video_folder=str(ppo_soft_video_dir),
                env_config=env_cfg,
            )
            if ppo_soft_video_results["frames"]:
                save_videos(ppo_soft_video_results["frames"], save_dir=str(ppo_soft_video_dir), prefix="ppo_soft_episode")
            video_outputs["ppo_soft"] = ppo_soft_video_results

            print("\nRecording PPO Deterministic episodes...")
            ppo_det_video_results = evaluate_agent(
                ppo_agent,
                n_episodes=cfg.num_video_episodes,
                deterministic=True,
                render_mode="rgb_array",
                video_folder=str(ppo_det_video_dir),
                env_config=env_cfg,
            )
            if ppo_det_video_results["frames"]:
                save_videos(ppo_det_video_results["frames"], save_dir=str(ppo_det_video_dir), prefix="ppo_det_episode")
            video_outputs["ppo_det"] = ppo_det_video_results

            print("\nRecording GRPO Soft episodes...")
            grpo_soft_video_results = evaluate_agent(
                grpo_agent,
                n_episodes=cfg.num_video_episodes,
                deterministic=False,
                render_mode="rgb_array",
                video_folder=str(grpo_soft_video_dir),
                env_config=env_cfg,
            )
            if grpo_soft_video_results["frames"]:
                save_videos(grpo_soft_video_results["frames"], save_dir=str(grpo_soft_video_dir), prefix="grpo_soft_episode")
            video_outputs["grpo_soft"] = grpo_soft_video_results

            print("\nRecording GRPO Deterministic episodes...")
            grpo_det_video_results = evaluate_agent(
                grpo_agent,
                n_episodes=cfg.num_video_episodes,
                deterministic=True,
                render_mode="rgb_array",
                video_folder=str(grpo_det_video_dir),
                env_config=env_cfg,
            )
            if grpo_det_video_results["frames"]:
                save_videos(grpo_det_video_results["frames"], save_dir=str(grpo_det_video_dir), prefix="grpo_det_episode")
            video_outputs["grpo_det"] = grpo_det_video_results

        print("\n" + "=" * 70)
        print("GENERATING FOUR-WAY COMPARISON PLOTS")
        print("=" * 70)
        plot_four_way_comparison(ppo_soft_results, ppo_det_results, grpo_soft_results, grpo_det_results, save_dir=str(output_dir))

        print("\n" + "=" * 70)
        print("SAVING FOUR-WAY RESULTS TO CSV")
        print("=" * 70)
        save_four_way_results_to_csv(ppo_soft_results, ppo_det_results, grpo_soft_results, grpo_det_results, save_dir=str(output_dir))

        print("\n" + "=" * 70)
        print("FOUR-WAY COMPARISON SUMMARY")
        print("=" * 70)
        print("\nSuccess Rates:")
        print(f"  PPO Soft:          {ppo_soft_results['success_rate']:.2%}")
        print(f"  PPO Deterministic: {ppo_det_results['success_rate']:.2%}")
        print(f"  GRPO Soft:         {grpo_soft_results['success_rate']:.2%}")
        print(f"  GRPO Deterministic:{grpo_det_results['success_rate']:.2%}")
        
        # Find best overall
        all_success_rates = {
            'PPO Soft': ppo_soft_results['success_rate'],
            'PPO Deterministic': ppo_det_results['success_rate'],
            'GRPO Soft': grpo_soft_results['success_rate'],
            'GRPO Deterministic': grpo_det_results['success_rate'],
        }
        best_mode = max(all_success_rates, key=all_success_rates.get)
        print(f"\n🏆 Best Performance: {best_mode} ({all_success_rates[best_mode]:.2%} success rate)")
        
        print("\nMean Rewards:")
        print(f"  PPO Soft:          {ppo_soft_results['mean_reward']:.2f}")
        print(f"  PPO Deterministic: {ppo_det_results['mean_reward']:.2f}")
        print(f"  GRPO Soft:         {grpo_soft_results['mean_reward']:.2f}")
        print(f"  GRPO Deterministic:{grpo_det_results['mean_reward']:.2f}")

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE!")
        print("=" * 70)
        print(f"\nResults saved to: {output_dir}/")
        print("\nGenerated files:")
        print("  📊 Plots:")
        print("     - evaluation_comparison_four_way.png")
        print("     - statistical_summary_four_way.png")
        print("  📁 Data:")
        print("     - episode_results_four_way.csv")
        print("     - summary_statistics_four_way.csv")
        if cfg.num_video_episodes > 0:
            print("  🎥 Videos:")
            print("     - videos/ppo_soft/ppo_soft_episode_*.mp4")
            print("     - videos/ppo_deterministic/ppo_det_episode_*.mp4")
            print("     - videos/grpo_soft/grpo_soft_episode_*.mp4")
            print("     - videos/grpo_deterministic/grpo_det_episode_*.mp4")
        print("=" * 70)

        return {
            "config": cfg,
            "ppo_soft": ppo_soft_results,
            "ppo_deterministic": ppo_det_results,
            "grpo_soft": grpo_soft_results,
            "grpo_deterministic": grpo_det_results,
            "videos": video_outputs,
            "best_mode": best_mode,
        }

    else:
        # Original two-way comparison
        print("\n" + "=" * 70)
        print("EVALUATING PPO")
        print("=" * 70)
        ppo_results = evaluate_agent(
            ppo_agent,
            n_episodes=cfg.num_episodes,
            deterministic=cfg.deterministic,
            render_mode=None,
            env_config=env_cfg,
        )

        print("\n" + "=" * 70)
        print("EVALUATING GRPO")
        print("=" * 70)
        grpo_results = evaluate_agent(
            grpo_agent,
            n_episodes=cfg.num_episodes,
            deterministic=cfg.deterministic,
            render_mode=None,
            env_config=env_cfg,
        )

        video_outputs: Dict[str, Any] = {}
        if cfg.num_video_episodes > 0:
            print("\n" + "=" * 70)
            print("RECORDING VIDEOS")
            print("=" * 70)

            ppo_video_dir = Path(output_dir) / "videos" / "ppo"
            grpo_video_dir = Path(output_dir) / "videos" / "grpo"

            print("\nRecording PPO episodes...")
            ppo_video_results = evaluate_agent(
                ppo_agent,
                n_episodes=cfg.num_video_episodes,
                deterministic=cfg.deterministic,
                render_mode="rgb_array",
                video_folder=str(ppo_video_dir),
                env_config=env_cfg,
            )
            if ppo_video_results["frames"]:
                save_videos(ppo_video_results["frames"], save_dir=str(ppo_video_dir), prefix="ppo_episode")
            video_outputs["ppo"] = ppo_video_results

            print("\nRecording GRPO episodes...")
            grpo_video_results = evaluate_agent(
                grpo_agent,
                n_episodes=cfg.num_video_episodes,
                deterministic=cfg.deterministic,
                render_mode="rgb_array",
                video_folder=str(grpo_video_dir),
                env_config=env_cfg,
            )
            if grpo_video_results["frames"]:
                save_videos(grpo_video_results["frames"], save_dir=str(grpo_video_dir), prefix="grpo_episode")
            video_outputs["grpo"] = grpo_video_results

        print("\n" + "=" * 70)
        print("GENERATING COMPARISON PLOTS")
        print("=" * 70)
        plot_comparison(ppo_results, grpo_results, save_dir=str(output_dir))

        print("\n" + "=" * 70)
        print("SAVING RESULTS TO CSV")
        print("=" * 70)
        save_results_to_csv(ppo_results, grpo_results, save_dir=str(output_dir))

        perform_statistical_tests(ppo_results, grpo_results)

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE!")
        print("=" * 70)
        print(f"\nResults saved to: {output_dir}/")
        print("\nGenerated files:")
        print("  📊 Plots:")
        print("     - evaluation_comparison.png (comprehensive 3x3 grid)")
        print("  📁 Data:")
        print("     - episode_results.csv")
        print("     - summary_statistics.csv")
        print("     - crash_details.csv")
        if cfg.num_video_episodes > 0:
            print("  🎥 Videos:")
            print("     - videos/ppo/ppo_episode_*.mp4")
            print("     - videos/grpo/grpo_episode_*.mp4")
        print("=" * 70)

        return {
            "config": cfg,
            "ppo": ppo_results,
            "grpo": grpo_results,
            "videos": video_outputs,
        }
