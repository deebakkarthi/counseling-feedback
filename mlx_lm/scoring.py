#!/usr/bin/env python3
"""Generate predictions and evaluate using MLX CLI with checkpointing and parallelism."""

import json
import subprocess
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    """Load data from JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_checkpoint(results: list[dict], checkpoint_path: str):
    """Save current results to checkpoint file."""
    with open(checkpoint_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def load_checkpoint(checkpoint_path: str) -> list[dict]:
    """Load results from checkpoint file."""
    if not os.path.exists(checkpoint_path):
        return []
    return load_jsonl(checkpoint_path)


def generate_with_cli(prompt: str, model_path: str, adapter_path: str, max_tokens: int = 256) -> str:
    """Generate using mlx_lm.generate CLI."""
    result = subprocess.run(
        [
            "mlx_lm.generate",
            "--model", model_path,
            "--adapter-path", adapter_path,
            "--max-tokens", str(max_tokens),
            "--prompt", prompt,
        ],
        capture_output=True,
        text=True,
    )
    
    output = result.stdout
    lines = output.split("\n")
    
    in_response = False
    response_lines = []
    for line in lines:
        if line.strip() == "==========":
            if in_response:
                break
            in_response = True
            continue
        if in_response:
            response_lines.append(line)
    
    return "\n".join(response_lines).strip()


def process_single_example(args: tuple) -> dict:
    """Process a single example. Used for parallel processing."""
    idx, sample, model_path, adapter_path, max_tokens = args
    
    prompt = sample["prompt"]
    response = generate_with_cli(prompt, model_path, adapter_path, max_tokens)
    
    return {
        "idx": idx,
        "prompt": prompt,
        "prediction": response,
        "ground_truth": sample.get("completion", ""),
    }


def parse_feedback(text: str) -> dict | None:
    """Parse JSON feedback from model output."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return None


def compute_metrics(results: list[dict]) -> dict:
    """Compute evaluation metrics."""
    total = 0
    correct_perfect = 0
    valid_json = 0
    
    all_good_areas_pred = []
    all_good_areas_gt = []
    all_bad_areas_pred = []
    all_bad_areas_gt = []
    
    for result in results:
        gt = result.get("ground_truth", "")
        if not gt:
            continue
        
        total += 1
        pred_parsed = parse_feedback(result["prediction"])
        gt_parsed = parse_feedback(gt)
        
        if pred_parsed is None:
            continue
        
        valid_json += 1
        
        if gt_parsed is None:
            continue
        
        if pred_parsed.get("perfect") == gt_parsed.get("perfect"):
            correct_perfect += 1
        
        pred_good = set(pred_parsed.get("goodareas", []))
        gt_good = set(gt_parsed.get("goodareas", []))
        pred_bad = set(pred_parsed.get("badareas", []))
        gt_bad = set(gt_parsed.get("badareas", []))
        
        all_good_areas_pred.append(pred_good)
        all_good_areas_gt.append(gt_good)
        all_bad_areas_pred.append(pred_bad)
        all_bad_areas_gt.append(gt_bad)
    
    def set_metrics(preds: list[set], gts: list[set]) -> dict:
        tp = fp = fn = 0
        for pred, gt in zip(preds, gts):
            tp += len(pred & gt)
            fp += len(pred - gt)
            fn += len(gt - pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    good_area_metrics = set_metrics(all_good_areas_pred, all_good_areas_gt)
    bad_area_metrics = set_metrics(all_bad_areas_pred, all_bad_areas_gt)
    
    return {
        "total_evaluated": total,
        "valid_json_count": valid_json,
        "valid_json_rate": valid_json / total if total else 0,
        "perfect_accuracy": correct_perfect / valid_json if valid_json else 0,
        "good_areas": good_area_metrics,
        "bad_areas": bad_area_metrics,
    }


def print_metrics(metrics: dict):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nTotal evaluated:    {metrics['total_evaluated']}")
    print(f"Valid JSON outputs: {metrics['valid_json_count']} ({metrics['valid_json_rate']:.1%})")
    print(f"Perfect accuracy:   {metrics['perfect_accuracy']:.1%}")
    
    print("\nGood Areas (identifying strengths):")
    print(f"  Precision: {metrics['good_areas']['precision']:.3f}")
    print(f"  Recall:    {metrics['good_areas']['recall']:.3f}")
    print(f"  F1:        {metrics['good_areas']['f1']:.3f}")
    
    print("\nBad Areas (identifying weaknesses):")
    print(f"  Precision: {metrics['bad_areas']['precision']:.3f}")
    print(f"  Recall:    {metrics['bad_areas']['recall']:.3f}")
    print(f"  F1:        {metrics['bad_areas']['f1']:.3f}")
    
    print("=" * 60)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def main():
    # Configuration
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    adapter_path = "./adapters/counseling-lora"
    test_path = "./data/valid.jsonl"
    output_path = "./data/predictions.jsonl"
    checkpoint_path = "./data/checkpoint.jsonl"
    metrics_path = "./data/metrics.json"
    
    max_tokens = 256
    num_examples = None  # Set to None for all examples
    num_workers = 4  # Number of parallel workers
    checkpoint_every = 20  # Save checkpoint every N examples
    
    print("Loading test data...")
    test_data = load_jsonl(test_path)
    if num_examples:
        test_data = test_data[:num_examples]
    print(f"Total examples: {len(test_data)}")
    
    # Load checkpoint if exists
    completed_results = load_checkpoint(checkpoint_path)
    completed_indices = {r.get("idx", i) for i, r in enumerate(completed_results)}
    
    if completed_results:
        print(f"Resuming from checkpoint: {len(completed_results)} already completed")
    
    # Filter out already completed examples
    remaining = [
        (i, sample, model_path, adapter_path, max_tokens)
        for i, sample in enumerate(test_data)
        if i not in completed_indices
    ]
    
    print(f"Remaining examples: {len(remaining)}")
    print(f"Using {num_workers} parallel workers\n")
    
    if not remaining:
        print("All examples already processed!")
        results = completed_results
    else:
        results = list(completed_results)
        start_time = time.time()
        processed_count = len(completed_results)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_example, args): args[0] for args in remaining}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    processed_count += 1
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    completed_in_session = processed_count - len(completed_results)
                    if completed_in_session > 0:
                        avg_per_sample = elapsed / completed_in_session
                        remaining_count = len(test_data) - processed_count
                        eta = avg_per_sample * remaining_count / num_workers
                        
                        print(f"[{processed_count:4d}/{len(test_data)}] "
                              f"Elapsed: {format_time(elapsed)} | "
                              f"ETA: {format_time(eta)} | "
                              f"Pred: {result['prediction'][:50]}...")
                    
                    # Save checkpoint
                    if processed_count % checkpoint_every == 0:
                        save_checkpoint(results, checkpoint_path)
                        print(f"  >> Checkpoint saved ({processed_count} examples)")
                
                except Exception as e:
                    idx = futures[future]
                    print(f"Error processing example {idx}: {e}")
        
        total_time = time.time() - start_time
        print(f"\nGeneration complete! Total time: {format_time(total_time)}")
    
    # Sort results by index
    results.sort(key=lambda x: x.get("idx", 0))
    
    # Remove idx field before saving
    for r in results:
        r.pop("idx", None)
    
    print(f"\nSaving predictions to {output_path}...")
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint file removed")
    
    print("Computing metrics...")
    metrics = compute_metrics(results)
    print_metrics(metrics)
    
    print(f"Saving metrics to {metrics_path}...")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
