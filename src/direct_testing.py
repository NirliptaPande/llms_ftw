"""
Direct program generation test - minimal one-shot vs experience comparison.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# Import from existing modules
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from vlm_prompter import VLMPrompter
from vlm_client import VLMConfig, create_client, BaseVLMClient
from main import (
    create_phase1_cache_key,
    extract_code_from_response,
    test_program,
    TaskResult,
    sort_examples_by_size,
    save_results
)


def test_direct_generation(
    data_dir: str,
    cache_path: str,
    vlm_client: BaseVLMClient,
    exp: bool = False,
    use_programs: bool = False,
    use_images: bool = False,
    max_api_calls: int = 400,
    log_dir: str = "logs/direct_test",
    verbose: bool = True
) -> List[TaskResult]:
    """
    Test direct program generation with optional experience.
    
    Args:
        data_dir: Path to task JSON files
        cache_path: Path to phase1 cache JSON
        vlm_client: VLM client for API calls
        exp: Whether to use experience (similar tasks/programs)
        use_programs: Include similar program code (requires exp=True)
        use_images: Include similar task images (requires exp=True)
        max_api_calls: Max concurrent API calls
        log_dir: Directory for logs
        verbose: Print progress
    
    Returns:
        List of TaskResult objects
    """
    
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Load tasks
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Directory not found: {data_dir}", flush=True)
        return []
    
    json_files = sorted(data_path.glob('*.json'))
    if not json_files:
        print(f"No JSON files found in {data_dir}", flush=True)
        return []
    
    tasks_data = []
    for task_file in json_files:
        task_id = task_file.stem
        try:
            with open(task_file, 'r') as f:
                task = json.load(f)
            task = sort_examples_by_size(task)
            tasks_data.append((task_id, task))
        except Exception as e:
            print(f"✗ {task_id}: {e}", flush=True)
    
    if verbose:
        print(f"\n{'='*80}", flush=True)
        mode = "EXPERIENCE" if exp else "ONE-SHOT"
        details = []
        if exp and use_programs:
            details.append("programs")
        if exp and use_images:
            details.append("images")
        mode_detail = f" ({'+'.join(details)})" if details else ""
        print(f"DIRECT GENERATION TEST - {mode}{mode_detail}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Total tasks: {len(tasks_data)}", flush=True)
        print(f"{'='*80}\n", flush=True)
    
    # 2. Load phase1 cache
    phase1_cache = {}
    if exp and (use_programs or use_images):
        cache_file = Path(cache_path)
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    phase1_cache = json.load(f)
                if verbose:
                    print(f"Loaded phase1 cache: {len(phase1_cache)} entries", flush=True)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}", flush=True)
        else:
            print(f"Warning: Cache file not found: {cache_path}", flush=True)
    
    # 3. Setup
    prompter = VLMPrompter()
    all_prompts = []
    task_list = []
    
    # 4. Build prompts (INLINE)
    if verbose:
        print(f"Building prompts...", flush=True)
    
    for task_id, task in tasks_data:
        content_blocks = []
        
        # Header
        content_blocks.append({
            "type": "text",
            "text": "# ARC Puzzle Solver\n\nYou are an expert Python programmer solving an ARC (Abstraction and Reasoning Corpus) puzzle.\n\n"
        })
        
        # Training examples
        content_blocks.append({
            "type": "text",
            "text": "## Training Examples\n\nBelow are training examples showing input-output pairs:\n"
        })
        content_blocks.extend(prompter._format_training_examples(task['train'], include_images=True))
        
        # Test examples
        content_blocks.append({
            "type": "text",
            "text": "\n## Test Examples\n\nBelow are test inputs you need to solve:\n"
        })
        content_blocks.extend(prompter._format_test_examples(task['test'], include_images=True))
        
        # Experience (if enabled)
        if exp and (use_programs or use_images):
            cache_key = create_phase1_cache_key(
                task_id, 
                task, 
                similar=True, 
                library_modules=['formatted_solutions', 'solvers']
            )
            
            if cache_key in phase1_cache:
                cached_data = phase1_cache[cache_key]
                similar_programs = cached_data.get('similar_programs', [])
                
                if similar_programs:
                    content_blocks.append({
                        "type": "text",
                        "text": "\n## Reference Materials\n\nThe following similar tasks and/or solutions may help you:\n"
                    })
                    
                    if use_images:
                        image_refs = [p for p in similar_programs if p.get('source_module') == 'solvers']
                        if image_refs:
                            content_blocks.append({
                                "type": "text",
                                "text": "\n### Similar Tasks:\n"
                            })
                            for idx, ref in enumerate(image_refs[:3], 1):  # Limit to top 3
                                sim = ref.get('similarity', 0.0)
                                content_blocks.append({
                                    "type": "text",
                                    "text": f"\nSimilar Task {idx} (similarity: {sim:.2f}):\n"
                                })
                                train_examples = ref.get('train_examples', [])
                                if train_examples:
                                    content_blocks.extend(prompter._format_training_examples(train_examples, include_images=True))
                    
                    if use_programs:
                        program_refs = [p for p in similar_programs if p.get('source_module') == 'formatted_solutions']
                        if program_refs:
                            content_blocks.append({
                                "type": "text",
                                "text": "\n### Similar Solutions:\n"
                            })
                            for idx, ref in enumerate(program_refs[:3], 1):  # Limit to top 3
                                sim = ref.get('similarity', 0.0)
                                prog_code = ref.get('program', '')
                                content_blocks.append({
                                    "type": "text",
                                    "text": f"\nSimilar Solution {idx} (similarity: {sim:.2f}):\n```python\n{prog_code}\n```\n"
                                })
                    
                    content_blocks.append({
                        "type": "text",
                        "text": "\nYou may use the above similar tasks and solutions as inspiration or reference.\n"
                    })
        
        # Instruction
        content_blocks.append({
            "type": "text",
            "text": """
## Task

Write a Python function `def solve(I):` that:
- Takes input `I` as a tuple of tuples of integers (immutable 2D grid)
- Returns the output in the same format (tuple of tuples of ints)
- Uses pure Python and standard libraries (no external dependencies)
- Implements the transformation pattern you observe

Generate the `solve(I)` function now.
"""
        })
        
        all_prompts.append(content_blocks)
        task_list.append((task_id, task))
        
        # Log prompt
        prompt_log_path = Path(log_dir) / f"{task_id}_prompt.txt"
        with open(prompt_log_path, 'w') as f:
            f.write(f"Task ID: {task_id}\n{'='*80}\n\n")
            for block in content_blocks:
                if block['type'] == 'text':
                    f.write(block['text'])
                    f.write('\n')
    
    if verbose:
        print(f"Built {len(all_prompts)} prompts", flush=True)
    
    # 5. Batch API calls
    if verbose:
        print(f"Sending {len(all_prompts)} prompts to API...", flush=True)
    
    system_prompt = """You are an expert Python programmer solving ARC puzzles. 
Generate a function `def solve(I):` that transforms the input grid to the output grid.
Use pure Python and standard libraries. Be concise and correct."""
    
    time_start = time.time()
    
    with ThreadPoolExecutor(max_workers=min(max_api_calls, len(all_prompts))) as executor:
        futures = [executor.submit(vlm_client.query, p, system_prompt) for p in all_prompts]
        responses = [f.result() for f in futures]
    
    time_api = time.time()
    
    if verbose:
        print(f"API calls complete: {time_api - time_start:.1f}s", flush=True)
    
    # 6. Extract and test
    if verbose:
        print(f"Extracting and testing programs...", flush=True)
    
    results = []
    successful = 0
    total_score = 0.0
    
    for idx, ((task_id, task), response) in enumerate(zip(task_list, responses), 1):
        # Log response
        response_log_path = Path(log_dir) / f"{task_id}_response.txt"
        with open(response_log_path, 'w') as f:
            f.write(f"Task ID: {task_id}\n{'='*80}\n\n")
            f.write(response)
        
        # Extract code
        code = extract_code_from_response(response)
        
        if code:
            # Log code
            code_log_path = Path(log_dir) / f"{task_id}_code.py"
            with open(code_log_path, 'w') as f:
                f.write(code)
            
            # Test on test set
            try:
                score, test_results = test_program(code, task, testing='test')
                success = (score == 1.0)
                
                if success:
                    successful += 1
                total_score += score
                
                result = TaskResult(
                    task_id=task_id,
                    success=success,
                    score=score,
                    program=code
                )
                
                # Log results
                results_log_path = Path(log_dir) / f"{task_id}_results.txt"
                with open(results_log_path, 'w') as f:
                    f.write(f"Task ID: {task_id}\n{'='*80}\n")
                    f.write(f"Score: {score:.2f}\n")
                    f.write(f"Success: {success}\n\n")
                    f.write(f"Test Results:\n")
                    for tidx, (expected, actual, correct) in enumerate(test_results):
                        f.write(f"  Test {tidx+1}: {'✓' if correct else '✗'}\n")
                
            except Exception as e:
                result = TaskResult(
                    task_id=task_id,
                    success=False,
                    score=0.0,
                    program=code,
                    error=str(e)
                )
        else:
            result = TaskResult(
                task_id=task_id,
                success=False,
                score=0.0,
                error="No code extracted"
            )
        
        results.append(result)
        
        # Progress
        if verbose:
            status = "✓" if result.success else "✗"
            print(f"{status} [{idx}/{len(task_list)}] {task_id}: {result.score:.2f}", flush=True)
    
    time_end = time.time()
    
    # 7. Summary
    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total time: {time_end - time_start:.1f}s", flush=True)
    print(f"  API calls: {time_api - time_start:.1f}s", flush=True)
    print(f"  Testing: {time_end - time_api:.1f}s", flush=True)
    print(f"Successful: {successful}/{len(tasks_data)} ({100*successful/len(tasks_data):.1f}%)", flush=True)
    print(f"Average score: {total_score/len(tasks_data):.2f}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    return results


def main():
    """Entry point for direct generation experiments"""
    import argparse
    import yaml
    from dotenv import load_dotenv
    import os
    import wandb
    
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/direct_test_config.yaml', help='Config file')
    parser.add_argument('--exp-name', type=str, help='Experiment name (optional)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        mode = "exp" if config['test_direct']['exp'] else "oneshot"
        details = []
        if config['test_direct'].get('use_programs'):
            details.append("progs")
        if config['test_direct'].get('use_images'):
            details.append("imgs")
        detail_str = "_".join(details) if details else "none"
        model = config['model']['name'].split('/')[-1].replace('-', '')
        exp_name = f"direct_{mode}_{detail_str}_{model}"
    
    # Setup wandb
    wandb.init(
        project="arc-solver_icml",
        name=exp_name,
        config=config
    )
    
    # Setup paths
    PROJECT_ROOT = Path(__file__).parent.parent
    data_dir = config['test_direct']['data_dir']
    cache_path = PROJECT_ROOT / config['test_direct']['cache_path']
    log_dir = f"logs/{exp_name}"
    
    # Setup VLM client
    PROVIDER = config['provider']
    if PROVIDER == "grok":
        api_key = os.getenv('OPENROUTER_API_KEY')
    elif PROVIDER == "qwen":
        api_key = None
    elif PROVIDER == "gemini":
        api_key = os.getenv('GEMINI_API_KEY')
    else:
        api_key = None
    
    vlm_config = VLMConfig(
        api_key=api_key,
        model=config['model']['name'],
        api_base=config['model']['api_base'],
        max_tokens=config['test_direct']['max_tokens'],
        max_retries=config['test_direct']['max_retries'],
        suppress_errors=True
    )
    
    vlm_client = create_client(PROVIDER, config=vlm_config)
    
    # Run test
    results = test_direct_generation(
        data_dir=data_dir,
        cache_path=str(cache_path),
        vlm_client=vlm_client,
        exp=config['test_direct']['exp'],
        use_programs=config['test_direct']['use_programs'],
        use_images=config['test_direct']['use_images'],
        max_api_calls=config['test_direct']['max_api_calls'],
        log_dir=log_dir,
        verbose=config['test_direct']['verbose']
    )
    
    # Save results
    output_dir = f"results/{exp_name}"
    save_results(results, output_dir=output_dir)
    
    # Log to wandb
    successful = sum(1 for r in results if r.success)
    total_score = sum(r.score for r in results)
    
    wandb.log({
        "success_rate": successful / len(results) if results else 0,
        "avg_score": total_score / len(results) if results else 0,
        "total_tasks": len(results),
        "successful_tasks": successful
    })
    wandb.finish()


if __name__ == "__main__":
    main()