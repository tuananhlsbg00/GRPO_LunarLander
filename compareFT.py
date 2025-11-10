from sparse_scripts.utils.train_compare_sparse_lander import TrainingComparisonConfig, run_training_comparison
from sparse_scripts.utils.sparse_lunar_lander import EnvConfig
from sparse_scripts.utils.evaluate_models import EvaluationConfig, run_evaluation

env_cfg = EnvConfig(
    soft_success_condition=False,
    random_initial_position=True
)

config = TrainingComparisonConfig(
    total_timesteps=5_000_000,
    eval_frequency=20_000,
    base_output_dir="runs",
    run_id="soft_condition_false_random_pos_true",
    env_config=env_cfg
)

eval_cfg = EvaluationConfig(
    ppo_model_path="runs/soft_condition_false_random_pos_true/models/ppo_lunar_lander_best.pth",
    grpo_model_path="runs/soft_condition_false_random_pos_true/models/grpo_lunar_lander_best.pth",
    output_dir="runs/soft_condition_false_random_pos_true/results",
    env_config=env_cfg,
    compare_both_modes=True,
)

results = run_training_comparison(config)
evaluation_metrics = run_evaluation(eval_cfg)