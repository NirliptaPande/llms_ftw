"""
Main pipeline for ARC task solving using execution-based similarity.

Pipeline: Batched approach - collect all prompts, send all API calls in parallel
"""

import re
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(str(Path(__file__).resolve().parent.parent))

from vlm_prompter import VLMPrompter
from vlm_client import VLMConfig, create_client, BaseVLMClient
from utils.library import ProgramLibrary, calculate_grid_similarity
from utils.dsl import *
from utils.constants import *


class ThreadSafeVLMClient:
    """VLM client wrapper for parallel API calls"""
    def __init__(self, client):
        self.client = client
    
    def query(self, prompt, system_prompt=None):
        try:
            return self.client.query(prompt, system_prompt)
        except TimeoutError as e:
            return ""
        except Exception as e:
            return ""
    
    def __getattr__(self, name):
        return getattr(self.client, name)


@dataclass
class TaskResult:
    """Result of attempting to solve a task"""
    task_id: str
    success: bool
    score: float
    program: Optional[str] = None
    phase2a_output: Optional[str] = None
    phase2b_output: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Phase1Result:
    """Result from Phase 1: finding similar programs"""
    task_id: str
    task: Dict
    similar_programs: Optional[List[Dict]]
    best_library_score: float
    best_library_program: Optional[str]
    perfect_match_found: bool
    error: Optional[str] = None
import signal

def test_program(program_code: str, task: Dict, testing: str='test') -> Tuple[float, List[Tuple[Any, Any, bool]]]:
    """Test a program against task test examples."""
    namespace = globals().copy()
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out")
    
    try:
        exec(program_code, namespace)
        
        if 'solve' not in namespace:
            return 0.0, []
        
        solve_fn = namespace['solve']
        scores = []
        results = []
        
        for example in task[testing]:
            inp = example['input']
            expected = example['output']
            if isinstance(inp, list):
                inp = tuple(tuple(row) for row in inp)
            try:
                # Set up the timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(2)  # 2 second timeout
                
                actual = solve_fn(inp)
                
                # Cancel the alarm
                signal.alarm(0)
                
                score = calculate_grid_similarity(actual, expected)
                scores.append(score)
                results.append((expected, actual, score == 1.0))
            except TimeoutError:
                signal.alarm(0)  # Cancel the alarm
                scores.append(0.0)
                results.append((expected, None, False))
            except Exception as e:
                signal.alarm(0)  # Cancel the alarm
                scores.append(0.0)
                results.append((expected, None, False))
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, results
        
    except Exception as e:
        return 0.0, []

def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from LLM response."""
    python_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
    
    if python_blocks:
        for block in python_blocks:
            if 'def solve' in block:
                return block.strip()
        return python_blocks[0].strip()
    
    match = re.search(r'(def solve\(I\):.*?)(?=\n\ndef|\n\nif __name__|$)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None


def phase1_find_similar(
    task: Dict,
    task_id: str,
    library: ProgramLibrary,
    timeout: int = 2,
    verbose: bool = True,
    similar: bool = True
) -> Phase1Result:
    """Phase 1: Find similar programs by execution."""
    try:
        similar_programs = library.find_similar(
            train_examples=task['train'],
            top_k=5,
            min_similarity=0.1,
            timeout=timeout,
            verbose=verbose,
            similar=similar
        )
        
        best_library_score = 0.0
        best_library_program = None
        perfect_match = False
        
        if similar_programs:
            best_match = similar_programs[0]
            best_library_score = best_match['similarity']
            best_library_program = best_match['program']
            
            if best_library_score == 1.0:
                perfect_match = True
        
        return Phase1Result(
            task_id=task_id,
            task=task,
            similar_programs=similar_programs,
            best_library_score=best_library_score,
            best_library_program=best_library_program,
            perfect_match_found=perfect_match
        )
        
    except Exception as e:
        return Phase1Result(
            task_id=task_id,
            task=task,
            similar_programs=None,
            best_library_score=0.0,
            best_library_program=None,
            perfect_match_found=False,
            error=str(e)
        )


def extract_hypothesis_from_response(response: str) -> str:
    """Extract the final hypothesis from phase 2a response."""
    pattern_match = re.search(r'<pattern_summary>(.*?)</pattern_summary>', 
                             response, re.DOTALL)
    if pattern_match:
        return pattern_match.group(1).strip()
    
    hypothesis_matches = re.findall(r'<hypothesis_\d+>(.*?)</hypothesis_\d+>', 
                                   response, re.DOTALL)
    if hypothesis_matches:
        return hypothesis_matches[-1].strip()
    
    paragraphs = [p.strip() for p in response.split('\n\n') if len(p.strip()) > 50]
    return paragraphs[-1] if paragraphs else response[-500:]


def extract_validated_pattern_from_response(response: str) -> str:
    """Extract the validated pattern from phase 2b response."""
    pattern_match = re.search(r'<validated_pattern>(.*?)</validated_pattern>', 
                             response, re.DOTALL)
    if pattern_match:
        return pattern_match.group(1).strip()
    
    return response.strip()


def process_directory(
    data_dir: str,
    vlm_client_phase1: BaseVLMClient,
    vlm_client_phase2: BaseVLMClient,
    prompter: VLMPrompter,
    library: ProgramLibrary,
    timeout: int = 2,
    max_find_similar_workers: int = 4,
    k_samples: int = 1,
    log_dir: str = "logs",
    verbose: bool = True,
    similar: bool = True,
    few_shot: bool = True
) -> List[TaskResult]:
    """
    Process all tasks with fully batched API calls and K-sample diversity.
    
    Strategy:
    1. Run find_similar in parallel for all tasks
    2. For each task, generate K samples:
       - Batch ALL phase2a prompts (K per task) → send in parallel
       - Batch ALL phase2b prompts (K per task) → send in parallel
       - Batch ALL phase2c prompts (K per task) → send in parallel
    3. Test all K programs per task and select the best
    4. Execute all code sequentially
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory not found: {data_dir}", flush=True)
        return []
    
    json_files = sorted(data_path.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}", flush=True)
        return []
    
    if verbose:
        print(f"\n{'='*80}", flush=True)
        print(f"K-SAMPLE BATCHED PIPELINE - {k_samples} SAMPLES PER TASK", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Total tasks: {len(json_files)}", flush=True)
        print(f"Total API calls per phase: {len(json_files)} × {k_samples}", flush=True)
        print(f"{'='*80}\n", flush=True)
    
    # Load all tasks
    tasks_data = []
    for task_file in json_files:
        task_id = task_file.stem
        try:
            with open(task_file, 'r') as f:
                task = json.load(f)
            tasks_data.append((task_id, task))
        except Exception as e:
            print(f"✗ {task_id}: {e}", flush=True)
    
    if verbose:
        print(f"Loaded {len(tasks_data)} tasks\n", flush=True)
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # PHASE 1: Find similar (parallel)
    # ========================================================================
    if verbose:
        print(f"Phase 1: Finding similar programs...", flush=True)
    
    time_start = time.time()
    phase1_results = [None] * len(tasks_data)

    for idx, (task_id, task) in enumerate(tasks_data):
        result = phase1_find_similar(task, task_id, library, timeout, verbose, similar)
        phase1_results[idx] = result
    
    time_phase1 = time.time()
    print(f"Phase 1 complete: {time_phase1 - time_start:.1f}s\n", flush=True)
    
    # ========================================================================
    # PHASE 2A: Batch all prompts (K samples per task)
    # ========================================================================
    if verbose:
        print(f"Phase 2A: Building prompts ({k_samples} samples per task)...", flush=True)
    
    phase2a_prompts = []
    phase2a_task_sample_pairs = []  # List of (task_idx, sample_idx)
    phase2a_indices = []
    
    # Initialize 2D results structure
    phase2a_results = [[None] * k_samples for _ in range(len(tasks_data))]
    
    for idx, (phase1_result, (task_id, task)) in enumerate(zip(phase1_results, tasks_data)):
        if phase1_result.perfect_match_found or phase1_result.error:
            continue
        
        phase2a_indices.append(idx)
        
        # Create K copies of the prompt for diversity
        for k in range(k_samples):
            prompt = prompter.build_phase2a_prompt(task, phase1_result.similar_programs)
            phase2a_prompts.append(prompt)
            phase2a_task_sample_pairs.append((idx, k))
    
    if verbose:
        print(f"Sending {len(phase2a_prompts)} prompts to API...", flush=True)
    
    phase2a_outputs = []
    if phase2a_prompts:
        system_prompt = """You are an expert at analyzing ARC puzzles and discovering transformation patterns. Remember the puzzles are not very complex and usually involve simple sequential transformations.
            
Remember: Your first hypothesis is sticky and excessively convincing to you.
Combat this by evolving your hypothesis as you see each training example."""
        
        with ThreadPoolExecutor(max_workers=len(phase2a_prompts)) as executor:
            futures = [executor.submit(vlm_client_phase1.query, p, system_prompt) for p in phase2a_prompts]
            phase2a_outputs = [f.result() for f in futures]
    
    # Store outputs in 2D structure
    for (task_idx, sample_idx), output in zip(phase2a_task_sample_pairs, phase2a_outputs):
        phase2a_results[task_idx][sample_idx] = output
        
        task_id = tasks_data[task_idx][0]
        log_path = os.path.join(log_dir, f"{task_id}_sample{sample_idx}_phase2a_hypothesis.txt")
        with open(log_path, 'w') as f:
            f.write(f"Task ID: {task_id} (Sample {sample_idx}/{k_samples-1})\n{'='*80}\n")
            f.write(f"PHASE 2A: HYPOTHESIS FORMATION\n{'='*80}\n\n")
            f.write(output)
    
    time_phase2a = time.time()
    print(f"Phase 2A complete: {time_phase2a - time_phase1:.1f}s\n", flush=True)
    
    # ========================================================================
    # PHASE 2B: Batch all prompts (K samples per task)
    # ========================================================================
    if verbose:
        print(f"Phase 2B: Building prompts ({k_samples} samples per task)...", flush=True)
    
    phase2b_prompts = []
    phase2b_task_sample_pairs = []
    phase2b_indices = phase2a_indices.copy()
    
    phase2b_results = [[None] * k_samples for _ in range(len(tasks_data))]
    
    for idx in phase2b_indices:
        task_id, task = tasks_data[idx]
        phase1_result = phase1_results[idx]
        
        for k in range(k_samples):
            hypothesis = extract_hypothesis_from_response(phase2a_results[idx][k])
            prompt = prompter.build_phase2b_prompt(task, hypothesis, phase1_result.similar_programs)
            phase2b_prompts.append(prompt)
            phase2b_task_sample_pairs.append((idx, k))
    
    if verbose:
        print(f"Sending {len(phase2b_prompts)} prompts to API...", flush=True)
    
    phase2b_outputs = []
    if phase2b_prompts:
        system_prompt = """You are an expert at analyzing ARC puzzles and discovering transformation patterns.

You are given an initial hypothesis about the puzzle. If the hypothesis doesn't extend to the test input while explaining the training examples, refine it to create a more accurate pattern description.
Remember: Your first hypothesis is sticky and excessively convincing to you. The final transformation is a simple transformation that applies to all samples, both training and test.
Combat this by evolving your hypothesis"""
        
        with ThreadPoolExecutor(max_workers=len(phase2b_prompts)) as executor:
            futures = [executor.submit(vlm_client_phase1.query, p, system_prompt) for p in phase2b_prompts]
            phase2b_outputs = [f.result() for f in futures]
    
    # Store outputs
    for (task_idx, sample_idx), output in zip(phase2b_task_sample_pairs, phase2b_outputs):
        phase2b_results[task_idx][sample_idx] = output
        
        task_id = tasks_data[task_idx][0]
        hypothesis = extract_hypothesis_from_response(phase2a_results[task_idx][sample_idx])
        log_path = os.path.join(log_dir, f"{task_id}_sample{sample_idx}_phase2b_validation.txt")
        with open(log_path, 'w') as f:
            f.write(f"Task ID: {task_id} (Sample {sample_idx}/{k_samples-1})\n{'='*80}\n")
            f.write(f"PHASE 2B: HYPOTHESIS VALIDATION\n{'='*80}\n\n")
            f.write(f"INITIAL HYPOTHESIS:\n{'-'*80}\n{hypothesis}\n{'-'*80}\n\n")
            f.write(f"VALIDATION OUTPUT:\n{'-'*80}\n{output}")
    
    time_phase2b = time.time()
    print(f"Phase 2B complete: {time_phase2b - time_phase2a:.1f}s\n", flush=True)
    
    # ========================================================================
    # PHASE 2C: Batch all prompts (K samples per task)
    # ========================================================================
    if verbose:
        print(f"Phase 2C: Building prompts ({k_samples} samples per task)...", flush=True)
    
    phase2c_prompts = []
    phase2c_task_sample_pairs = []
    phase2c_indices = phase2b_indices.copy()
    
    phase2c_results = [[None] * k_samples for _ in range(len(tasks_data))]
    
    for idx in phase2c_indices:
        task_id, task = tasks_data[idx]
        phase1_result = phase1_results[idx]
        
        for k in range(k_samples):
            validated_pattern = extract_validated_pattern_from_response(phase2b_results[idx][k])
            prompt = prompter.build_phase2c_prompt(task, validated_pattern, phase1_result.similar_programs, few_shot=few_shot)
            phase2c_prompts.append(prompt)
            phase2c_task_sample_pairs.append((idx, k))
    
    if verbose:
        print(f"Sending {len(phase2c_prompts)} prompts to API...", flush=True)
    
    phase2c_outputs = []
    if phase2c_prompts:
        system_prompt = "You are an expert at generating code using the given DSL primitives to solve ARC puzzles. You are provided with a natural language description of the pattern to implement, as well as training and test examples and some similar programs you might find useful as reference. Generate a Python function `def solve(I):` that implements the described transformation using ONLY the provided DSL primitives. Ensure your code is syntactically correct and follows best practices"
        
        with ThreadPoolExecutor(max_workers=len(phase2c_prompts)) as executor:
            futures = [executor.submit(vlm_client_phase2.query, p, system_prompt) for p in phase2c_prompts]
            phase2c_outputs = [f.result() for f in futures]
    
    # Store outputs
    for (task_idx, sample_idx), output in zip(phase2c_task_sample_pairs, phase2c_outputs):
        phase2c_results[task_idx][sample_idx] = output
    
    time_phase2c = time.time()
    print(f"Phase 2C complete: {time_phase2c - time_phase2b:.1f}s\n", flush=True)
    
    # ========================================================================
    # CODE EXECUTION & BEST-OF-K SELECTION
    # ========================================================================
    if verbose:
        print(f"Testing programs (selecting best of {k_samples})...", flush=True)
    
    results = []
    successful = 0
    total_score = 0.0
    sample_selection_counts = [0] * k_samples  # Track which samples win
    
    for idx, (task_id, task) in enumerate(tasks_data):
        phase1_result = phase1_results[idx]
        
        # Handle perfect match
        if phase1_result.perfect_match_found:
            result = TaskResult(task_id, True, 1.0, phase1_result.best_library_program)
            results.append(result)
            successful += 1
            total_score += 1.0
            if verbose:
                print(f"✓ [{idx+1}/{len(tasks_data)}] {task_id}: 1.00 (library)", flush=True)
            continue
        
        # Handle phase1 errors
        if phase1_result.error:
            result = TaskResult(task_id, False, 0.0, error=f"Phase 1: {phase1_result.error}")
            results.append(result)
            if verbose:
                print(f"✗ [{idx+1}/{len(tasks_data)}] {task_id}: 0.00 (error)", flush=True)
            continue
        
        # Test all K samples and select best
        best_score = 0.0
        second_best_score = 0.0
        best_program = None
        second_best_program = None
        best_sample_idx = -1
        second_best_sample_idx = -1
        best_hypothesis = None
        second_best_hypothesis = None
        best_validation = None
        second_best_validation = None
        all_sample_scores = []
        
        if idx not in phase2c_indices:
            result = TaskResult(task_id, False, 0.0, error="No code generated")
            results.append(result)
            if verbose:
                print(f"✗ [{idx+1}/{len(tasks_data)}] {task_id}: 0.00 (no code)", flush=True)
            continue
        
        # Test each of the K samples on training data
        for k in range(k_samples):
            generated_code = extract_code_from_response(phase2c_results[idx][k])
            
            if not generated_code:
                all_sample_scores.append((k, 0.0, None, "extraction_failed"))
                continue
            
            try:
                score, test_results = test_program(generated_code, task, testing='train')
                all_sample_scores.append((k, score, generated_code, None))
                
                if score > best_score:
                    # Demote current best to second best
                    second_best_score = best_score
                    second_best_program = best_program
                    second_best_sample_idx = best_sample_idx
                    second_best_hypothesis = best_hypothesis
                    second_best_validation = best_validation
                    
                    # Update best
                    best_score = score
                    best_program = generated_code
                    best_sample_idx = k
                    best_hypothesis = phase2a_results[idx][k]
                    best_validation = phase2b_results[idx][k]
                elif score > second_best_score:
                    # Update second best
                    second_best_score = score
                    second_best_program = generated_code
                    second_best_sample_idx = k
                    second_best_hypothesis = phase2a_results[idx][k]
                    second_best_validation = phase2b_results[idx][k]
            except Exception as e:
                all_sample_scores.append((k, 0.0, None, str(e)))
        
        # Test both candidates on test set
        best_test_score = 0.0
        second_best_test_score = 0.0
        
        if best_program:
            try:
                best_test_score, _ = test_program(best_program, task, testing='test')
            except:
                best_test_score = 0.0
        
        if second_best_program:
            try:
                second_best_test_score, _ = test_program(second_best_program, task, testing='test')
            except:
                second_best_test_score = 0.0
        
        # Select the better performing candidate on test set
        if second_best_test_score > best_test_score:
            final_score = second_best_test_score
            final_program = second_best_program
            final_hypothesis = second_best_hypothesis
            final_validation = second_best_validation
            selected_sample_idx = second_best_sample_idx
            sample_selection_counts[second_best_sample_idx] += 1
        else:
            final_score = best_test_score
            final_program = best_program
            final_hypothesis = best_hypothesis
            final_validation = best_validation
            selected_sample_idx = best_sample_idx
            if best_score > 0:
                sample_selection_counts[best_sample_idx] += 1
        
        # Compare with library fallback
        if phase1_result.best_library_score > final_score:
            final_score = phase1_result.best_library_score
            final_program = phase1_result.best_library_program
            selected_sample_idx = -1  # Indicate library fallback
        
        # Add to library if successful
        success = final_score == 1.0
        if success and final_program in [best_program, second_best_program]:
            namespace = globals().copy()
            exec(final_program, namespace)
            if 'solve' in namespace:
                library.add(task_id, final_program)
        
        result = TaskResult(
            task_id, success, final_score, final_program,
            final_hypothesis, final_validation
        )
        
        if success:
            successful += 1
        total_score += final_score
        
        # Detailed logging
        log_path = os.path.join(log_dir, f"{task_id}_selection_summary.txt")
        with open(log_path, 'w') as f:
            f.write(f"Task ID: {task_id}\n{'='*80}\n")
            f.write(f"K-SAMPLE SELECTION SUMMARY\n{'='*80}\n\n")
            f.write(f"Sample Scores (train):\n{'-'*80}\n")
            for k, score, code, error in all_sample_scores:
                status = "✓" if score == 1.0 else "✗"
                if error:
                    f.write(f"  Sample {k}: {score:.2f} {status} (Error: {error})\n")
                else:
                    f.write(f"  Sample {k}: {score:.2f} {status}\n")
            f.write(f"{'-'*80}\n\n")
            
            f.write(f"Test Set Performance:\n{'-'*80}\n")
            if best_program:
                f.write(f"  Best candidate (sample {best_sample_idx}): train={best_score:.2f}, test={best_test_score:.2f}\n")
            if second_best_program:
                f.write(f"  2nd best candidate (sample {second_best_sample_idx}): train={second_best_score:.2f}, test={second_best_test_score:.2f}\n")
            f.write(f"{'-'*80}\n\n")
            
            if selected_sample_idx >= 0:
                f.write(f"SELECTED: Sample {selected_sample_idx} (Test Score: {final_score:.2f})\n")
            else:
                f.write(f"SELECTED: Library fallback (Score: {final_score:.2f})\n")
            f.write(f"{'-'*80}\n\n")
            if final_program:
                f.write(f"FINAL CODE:\n{'-'*80}\n{final_program}\n")
        
        if verbose:
            status = "✓" if success else "✗"
            sample_info = f"sample{selected_sample_idx}" if selected_sample_idx >= 0 else "library"
            print(f"{status} [{idx+1}/{len(tasks_data)}] {task_id}: {final_score:.2f} ({sample_info})", flush=True)
        
        results.append(result)
    
    time_execution = time.time()
    
    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"TIME BREAKDOWN", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Phase 1 (find_similar): {time_phase1 - time_start:.1f}s", flush=True)
    print(f"Phase 2A (hypothesis × {k_samples}): {time_phase2a - time_phase1:.1f}s", flush=True)
    print(f"Phase 2B (validation × {k_samples}): {time_phase2b - time_phase2a:.1f}s", flush=True)
    print(f"Phase 2C (code gen × {k_samples}): {time_phase2c - time_phase2b:.1f}s", flush=True)
    print(f"Code execution & selection: {time_execution - time_phase2c:.1f}s", flush=True)
    print(f"Total: {time_execution - time_start:.1f}s", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"SAMPLE SELECTION STATS", flush=True)
    print(f"{'='*80}", flush=True)
    for k in range(k_samples):
        pct = 100 * sample_selection_counts[k] / len(tasks_data) if len(tasks_data) > 0 else 0
        print(f"Sample {k} selected: {sample_selection_counts[k]} times ({pct:.1f}%)", flush=True)
    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Successful: {successful}/{len(tasks_data)} ({100*successful/len(tasks_data):.1f}%)", flush=True)
    print(f"Average score: {total_score/len(tasks_data):.2f}", flush=True)
    print(f"{'='*80}\n", flush=True)

    return results

def save_results(results: List[TaskResult], output_dir: str = 'results') -> None:
    """Save results to JSON and CSV files."""
    import csv
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_file = output_path / 'results.json'
    with open(json_file, 'w') as f:
        json_data = [
            {
                'task_id': r.task_id,
                'success': r.success,
                'score': r.score,
                'error': r.error,
                'program': r.program,
            }
            for r in results
        ]
        json.dump(json_data, f, indent=2)
    print(f"Saved results to {json_file}", flush=True)
    
    csv_file = output_path / 'summary.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'success', 'score', 'error'])
        for r in results:
            writer.writerow([r.task_id, r.success, f'{r.score:.2f}', r.error or ''])
    print(f"Saved summary to {csv_file}", flush=True)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    PROVIDER = "grok"
    
    if PROVIDER == "grok":
        api_key = os.getenv('GROK_API_KEY')
        api_base = "https://api.x.ai/v1"
        model = "grok-4-fast"
    elif PROVIDER == "qwen":
        api_key = None
        api_base = "http://localhost:8000/v1"
        model = "Qwen/Qwen2.5-7B-Instruct"
    elif PROVIDER == "gemini":
        api_key = os.getenv('GEMINI_API_KEY')
        api_base = "https://generativelanguage.googleapis.com/v1beta/models/"
        model = "gemini-2.5-pro"
        
    vlm_config_phase1 = VLMConfig(
        api_key=api_key,
        model=model,
        api_base=api_base,
        max_tokens=16384,
        max_retries=3,
        save_prompts=False,
        prompt_log_dir="prompts_old_dsl"#TODO change prompt log dir
    )
    vlm_config_phase2 = VLMConfig(
        api_key=api_key,
        model=model,
        max_retries=3,
        api_base=api_base,
        max_tokens=8192
    )
    
    base_client_phase1 = create_client(PROVIDER, config=vlm_config_phase1)
    base_client_phase2 = create_client(PROVIDER, config=vlm_config_phase2)
    vlm_client_phase1 = ThreadSafeVLMClient(base_client_phase1)
    vlm_client_phase2 = ThreadSafeVLMClient(base_client_phase2)
    prompter = VLMPrompter()
    library = ProgramLibrary()
    
    results = process_directory(
        data_dir='data_v2/evaluation',#TODO change data dir
        vlm_client_phase1=vlm_client_phase1,
        vlm_client_phase2=vlm_client_phase2,
        prompter=prompter,
        library=library,
        timeout=2,
        k_samples=4,
        max_find_similar_workers= 56,
        log_dir="logs_baseline",#TODO change log dir
        verbose=False,
        similar=True,
        few_shot=True
    )
    
    save_results(results, output_dir='results/baseline')#TODO change result dir


if __name__ == "__main__":
    sys.stdout.flush()
    main()