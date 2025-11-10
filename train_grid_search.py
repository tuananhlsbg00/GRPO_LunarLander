"""
Grid Search Training Script with Multiprocessing

Runs training comparisons for all combinations of:
- soft_success_condition: (True, False)
- random_initial_position: (True, False)

Total: 2 × 2 = 4 configurations
"""

import multiprocessing as mp
from itertools import product
from pathlib import Path
import time
from datetime import datetime

from sparse_scripts.utils.train_compare_sparse_lander import (
    run_training_comparison,
    TrainingComparisonConfig,
)
from sparse_scripts.utils.sparse_lunar_lander import EnvConfig
from sparse_scripts.utils.evaluate_models import (
    EvaluationConfig,
    run_evaluation,
)


def train_single_config(config_params):
    """
    Train a single configuration.
    
    Args:
        config_params: Tuple of (config_id, soft_success, random_pos)
    """
    config_id, soft_success, random_pos = config_params
    
    print(f"\n{'='*80}")
    print(f"Starting Configuration {config_id}")
    print(f"  soft_success_condition: {soft_success}")
    print(f"  random_initial_position: {random_pos}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        # Create output directory for this configuration
        base_dir = Path("grid_search_results")
        config_name = f"soft_condition_{str(soft_success).lower()}_random_pos_{str(random_pos).lower()}"

        # Create environment configuration
        env_cfg = EnvConfig(
            soft_success_condition=soft_success,
            random_initial_position=random_pos,
        )

        # Create training configuration
        train_cfg = TrainingComparisonConfig(
            total_timesteps=5_000_000,
            eval_frequency=10_000,
            seed=42,
            base_output_dir=base_dir,
            run_id=config_name,
            env_config=env_cfg,
        )

        # Run training comparison
        results = run_training_comparison(train_cfg)

        elapsed_time = time.time() - start_time

        # Get the actual output directory
        output_dir = results['run_dir']

        # ------------------------------------------------------------------
        # Post-training evaluation (mirrors compareFT.py)
        # ------------------------------------------------------------------
        try:
            eval_cfg = EvaluationConfig(
                ppo_model_path=str(output_dir / "models/ppo_lunar_lander_best.pth"),
                grpo_model_path=str(output_dir / "models/grpo_lunar_lander_best.pth"),
                output_dir=str(output_dir / "results"),
                env_config=env_cfg,
                compare_both_modes=True,
            )
            eval_results = run_evaluation(eval_cfg)
            eval_ok = True
        except Exception as eval_err:
            print(f"Warning: Evaluation failed for Config {config_id}: {eval_err}")
            eval_results = None
            eval_ok = False

        # Save configuration summary
        summary_path = output_dir / "config_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Configuration {config_id}\n")
            f.write("=" * 80 + "\n\n")
            f.write("Environment Configuration:\n")
            f.write(f"  soft_success_condition:  {soft_success}\n")
            f.write(f"  random_initial_position: {random_pos}\n\n")
            f.write(f"  Total training time:     {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)\n")
            if results:
                ppo_metrics = results['ppo']['metrics']
                grpo_metrics = results['grpo']['metrics']
                f.write(f"  PPO final success rate:  {ppo_metrics.final_evaluation['success_rate']:.2%}\n")
                f.write(f"  GRPO final success rate: {grpo_metrics.final_evaluation['success_rate']:.2%}\n")
                f.write(f"  PPO best success rate:   {ppo_metrics.best_success_rate:.2%}\n")
                f.write(f"  GRPO best success rate:  {grpo_metrics.best_success_rate:.2%}\n")
            f.write("\nEvaluation:\n")
            if eval_ok:
                # Point to artifacts; detailed metrics are written by evaluation module
                f.write(f"  Results directory:       {output_dir / 'results'}\n")
                if eval_results and isinstance(eval_results, dict):
                    # Write a brief one-line summary if available
                    try:
                        if eval_results.get('four_way'):
                            fw = eval_results['four_way']
                            f.write("  Four-way success rates:\n")
                            f.write(
                                f"    PPO Soft: {fw['ppo_soft']['success_rate']:.2%}, "
                                f"PPO Det: {fw['ppo_det']['success_rate']:.2%}, "
                                f"GRPO Soft: {fw['grpo_soft']['success_rate']:.2%}, "
                                f"GRPO Det: {fw['grpo_det']['success_rate']:.2%}\n"
                            )
                    except Exception:
                        pass
            else:
                f.write("  Evaluation failed. See console logs for details.\n")

        print(f"\n{'='*80}")
        print(f"✓ Completed Configuration {config_id} in {elapsed_time:.2f}s")
        print(f"  Results saved to: {output_dir}")
        print(f"{'='*80}\n")

        return {
            'config_id': config_id,
            'success': True,
            'elapsed_time': elapsed_time,
            'soft_success': soft_success,
            'random_pos': random_pos,
            'output_dir': str(output_dir),
        }

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✗ FAILED Configuration {config_id} after {elapsed_time:.2f}s")
        print(f"  Error: {str(e)}")
        print(f"{'='*80}\n")

        return {
            'config_id': config_id,
            'success': False,
            'elapsed_time': elapsed_time,
            'error': str(e),
            'soft_success': soft_success,
            'random_pos': random_pos,
        }


def main():
    """Run grid search with multiprocessing."""
    
    print("\n" + "=" * 80)
    print("GRID SEARCH TRAINING: PPO vs GRPO on Sparse Lunar Lander")
    print("=" * 80)
    
    # Define parameter grid
    soft_success_values = [True, False]
    random_pos_values = [True, False]

    # Generate all combinations
    param_grid = list(product(soft_success_values, random_pos_values))

    # Create configuration tuples with IDs
    configs = [(i + 1, *params) for i, params in enumerate(param_grid)]

    print(f"\nTotal configurations: {len(configs)}")
    print(f"Number of processes:  2")
    print(f"\nParameter combinations:")
    for config_id, soft_success, random_pos in configs:
        print(f"  Config {config_id:2d}: soft_success={soft_success}, random_pos={random_pos}")
    
    print("\n" + "=" * 80)
    print(f"Starting grid search at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    overall_start_time = time.time()
    
    # Run with multiprocessing (2 at a time)
    with mp.Pool(processes=2) as pool:
        results = pool.map(train_single_config, configs)
    
    overall_elapsed = time.time() - overall_start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nTotal time: {overall_elapsed:.2f}s ({overall_elapsed/60:.2f}m, {overall_elapsed/3600:.2f}h)")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    print(f"Failed:     {len(failed)}/{len(results)}")
    
    if successful:
        print("\n" + "-" * 80)
        print("Successful Configurations:")
        print("-" * 80)
        for r in successful:
            print(f"  Config {r['config_id']:2d}: {r['elapsed_time']:7.1f}s | "
                  f"soft_success={r['soft_success']}, random_pos={r['random_pos']}")
    
    if failed:
        print("\n" + "-" * 80)
        print("Failed Configurations:")
        print("-" * 80)
        for r in failed:
            print(f"  Config {r['config_id']:2d}: {r.get('error', 'Unknown error')}")
    
    # Save overall summary
    summary_path = Path("grid_search_results") / "overall_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Grid Search Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {overall_elapsed:.2f}s ({overall_elapsed/60:.2f}m)\n")
        f.write(f"Successful: {len(successful)}/{len(results)}\n")
        f.write(f"Failed:     {len(failed)}/{len(results)}\n\n")
        
        if successful:
            f.write("Successful Configurations:\n")
            f.write("-" * 80 + "\n")
            for r in successful:
                f.write(f"Config {r['config_id']:2d}: {r['elapsed_time']:7.1f}s | "
                       f"soft_success={r['soft_success']}, random_pos={r['random_pos']}\n")
                f.write(f"  Output: {r['output_dir']}\n")
        
        if failed:
            f.write("\nFailed Configurations:\n")
            f.write("-" * 80 + "\n")
            for r in failed:
                f.write(f"Config {r['config_id']:2d}: {r.get('error', 'Unknown error')}\n")
    
    print(f"\nOverall summary saved to: {summary_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Set start method to 'spawn' for better compatibility
    mp.set_start_method('spawn', force=True)
    main()
