#!/usr/bin/env python3
"""
Upload estimated reward decomposition metrics to existing wandb runs.

Since parallel evaluation doesn't track step-by-step penalties, we estimate them from:
- crash_count_mean: number of crashes per episode -> crash_penalty = crash_count_mean * 5.0
- out_of_road_rate: 0 or 1 per episode -> out_of_road_penalty = out_of_road_rate * 5.0
- reward: total reward per episode
- estimated_driving_reward = reward + crash_penalty + out_of_road_penalty

Usage:
    python upload_reward_decomposition.py --json_path /data/caihy/bc_eval/eval_iql_step_001000.json --wandb_run_id xxx
    
Or process all JSON files in a directory:
    python upload_reward_decomposition.py --json_dir /data/caihy/bc_eval --wandb_project iql-eval-1M
"""

import argparse
import json
import numpy as np
from pathlib import Path


def compute_reward_decomposition(json_path):
    """Compute estimated reward decomposition from saved evaluation results."""
    with open(json_path) as f:
        data = json.load(f)
    
    per_seed_results = data.get('per_seed_results', [])
    if not per_seed_results:
        print(f"  Warning: No per_seed_results in {json_path}")
        return None
    
    # Estimate penalties from available data
    crash_penalties = []
    out_of_road_penalties = []
    driving_rewards = []
    
    for r in per_seed_results:
        # Crash penalty: crash_count_mean includes both vehicle and object crashes
        crash_count = r.get('crash_count_mean', 0)
        crash_penalty = crash_count * 5.0  # Each crash is 5.0 penalty
        crash_penalties.append(crash_penalty)
        
        # Out of road penalty: out_of_road_rate is 0 or 1
        # We assume each out_of_road event is 5.0 penalty
        # Since we only know if it happened (not count), use rate * 5.0
        out_of_road_rate = r.get('out_of_road_rate', 0)
        out_of_road_penalty = out_of_road_rate * 5.0
        out_of_road_penalties.append(out_of_road_penalty)
        
        # Estimated driving reward = observed_reward + penalties
        reward = r.get('reward', 0)
        estimated_driving = reward + crash_penalty + out_of_road_penalty
        driving_rewards.append(estimated_driving)
    
    # Compute means
    metrics = {
        'reward/crash_penalty_mean': float(np.mean(crash_penalties)),
        'reward/out_of_road_penalty_mean': float(np.mean(out_of_road_penalties)),
        'reward/estimated_driving_reward_mean': float(np.mean(driving_rewards)),
        'reward/total_step_rewards_mean': float(np.mean([r.get('reward', 0) for r in per_seed_results])),
    }
    
    return metrics


def upload_to_wandb(metrics, run_path=None, run_id=None, project=None, entity="victorique"):
    """Upload metrics to an existing wandb run."""
    import wandb
    
    if run_path:
        # Resume by run path (e.g., "victorique/iql-eval-1M/run_id")
        api = wandb.Api()
        run = api.run(run_path)
        run.summary.update(metrics)
        run.update()
        print(f"  Updated wandb run: {run_path}")
    elif run_id and project:
        # Resume by run_id and project
        run_path = f"{entity}/{project}/{run_id}"
        api = wandb.Api()
        run = api.run(run_path)
        run.summary.update(metrics)
        run.update()
        print(f"  Updated wandb run: {run_path}")
    else:
        print("  Error: Must provide either run_path or (run_id + project)")
        return False
    
    return True


def find_wandb_run_from_json(json_path, project="iql-eval-1M", entity="victorique"):
    """Try to find the wandb run that corresponds to this JSON file."""
    # Extract step number from filename
    # e.g., eval_iql_step_001000.json -> 001000
    import re
    match = re.search(r'step_(\d+)', str(json_path))
    if not match:
        return None
    
    step = match.group(1)
    
    # Search for runs with matching exp_name
    import wandb
    api = wandb.Api()
    
    try:
        runs = api.runs(f"{entity}/{project}")
        for run in runs:
            exp_name = run.config.get('exp_name', '')
            if f"step{step}" in exp_name:
                return f"{entity}/{project}/{run.id}"
    except Exception as e:
        print(f"  Warning: Could not search wandb runs: {e}")
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Upload reward decomposition to wandb")
    parser.add_argument("--json_path", type=str, help="Path to single JSON result file")
    parser.add_argument("--json_dir", type=str, help="Directory containing JSON result files")
    parser.add_argument("--wandb_run_id", type=str, help="Specific wandb run ID to update")
    parser.add_argument("--wandb_run_path", type=str, help="Full wandb run path (entity/project/run_id)")
    parser.add_argument("--wandb_project", type=str, default="iql-eval-1M", help="Wandb project")
    parser.add_argument("--wandb_entity", type=str, default="victorique", help="Wandb entity")
    parser.add_argument("--dry_run", action="store_true", help="Compute metrics but don't upload")
    parser.add_argument("--auto_find", action="store_true", help="Auto-find wandb run from JSON filename")
    args = parser.parse_args()
    
    json_files = []
    
    if args.json_path:
        json_files = [Path(args.json_path)]
    elif args.json_dir:
        json_dir = Path(args.json_dir)
        json_files = list(json_dir.glob("eval_iql_step_*.json"))
    else:
        print("Error: Must provide --json_path or --json_dir")
        return
    
    print(f"Processing {len(json_files)} JSON files...")
    
    for json_path in sorted(json_files):
        print(f"\n=== {json_path.name} ===")
        
        # Compute metrics
        metrics = compute_reward_decomposition(json_path)
        if metrics is None:
            continue
        
        print(f"  Computed metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.2f}")
        
        if args.dry_run:
            print("  [DRY RUN] Skipping upload")
            continue
        
        # Upload to wandb
        run_path = args.wandb_run_path
        if not run_path and args.wandb_run_id:
            run_path = f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}"
        
        if not run_path and args.auto_find:
            run_path = find_wandb_run_from_json(json_path, args.wandb_project, args.wandb_entity)
            if run_path:
                print(f"  Auto-found run: {run_path}")
        
        if run_path:
            try:
                upload_to_wandb(metrics, run_path=run_path)
            except Exception as e:
                print(f"  Error uploading: {e}")
        else:
            print("  Warning: No wandb run specified, skipping upload")


if __name__ == "__main__":
    main()
