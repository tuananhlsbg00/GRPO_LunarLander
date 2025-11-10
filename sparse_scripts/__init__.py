"""Utility package for training and evaluating agents on sparse Lunar Lander."""

from .grpo.grpo import (
    GRPO,
    PolicyNetwork,
    default_grpo_config,
    make_grpo_agent,
)
from .ppo.ppo import (
    PPO,
    ActorCritic,
    default_ppo_config,
    make_ppo_agent,
)
from .utils.sparse_lunar_lander import SparseLunarLander
from .utils.evaluate_models import EvaluationConfig, run_evaluation
from .utils.train_compare_sparse_lander import (
    EvaluationMetrics,
    TrainingComparisonConfig,
    run_training_comparison,
)

__all__ = [
    "GRPO",
    "PolicyNetwork",
    "default_grpo_config",
    "make_grpo_agent",
    "PPO",
    "ActorCritic",
    "default_ppo_config",
    "make_ppo_agent",
    "SparseLunarLander",
    "EvaluationConfig",
    "run_evaluation",
    "EvaluationMetrics",
    "TrainingComparisonConfig",
    "run_training_comparison",
]
