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
import yaml
import threading

sys.path.append(str(Path(__file__).resolve().parent.parent))

from vlm_prompter import VLMPrompter
from vlm_client import VLMConfig, create_client, BaseVLMClient
from utils.library import ProgramLibrary, calculate_grid_similarity
from utils.dsl import *
from utils.constants import *


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
    
def get_program_errors(best_program_code: str, second_best_program_code: str, task: Dict, testing: str='train') -> Tuple[List[Tuple[Any, Any, bool]], List[Tuple], List[Tuple[Any, Any, bool]], List[Tuple]]:
    """
    Test both programs and return detailed error information for repair.
    
    Returns:
        Tuple of (score, results, best_error_details, score2, results2, second_best_error_details)
        - score: float, average score for best program
        - results: List of (expected, actual, is_correct) tuples for best program
        - best_error_details: List of (example_idx, expected_grid, actual_grid, diff_grid) for failed cases
        - score2: float, average score for second best program
        - results2: List of (expected, actual, is_correct) tuples for second best program
        - second_best_error_details: List of (example_idx, expected_grid, actual_grid, diff_grid) for failed cases
        
        diff_grid: 2D array where 'x' marks cells that differ, original value where correct
        diff_grid is None if shapes mismatch or actual output is None
    """
    _, results1 = test_program(best_program_code, task, testing=testing)#Needed?
    _, results2 = test_program(second_best_program_code, task, testing=testing)
    
    def process_results(results):
        error_details = []
        
        for idx, (expected, actual, is_correct) in enumerate(results):
            if not is_correct:
                # Convert to list format
                exp_list = expected if isinstance(expected, list) else [list(row) for row in expected]
                
                diff_grid = None
                act_list = None
                
                if actual is None:
                    # Program failed - no diff grid, just pass None
                    pass
                else:
                    try:
                        # Handle case where actual might be a scalar or wrong type
                        if isinstance(actual, (int, float, str)):
                            # Scalar output - can't create diff grid
                            act_list = [[actual]]  # Wrap in 2D structure for logging
                        elif isinstance(actual, list):
                            # Check if it's already a proper 2D list
                            if actual and isinstance(actual[0], (list, tuple)):
                                act_list = actual
                            else:
                                # 1D list, wrap it
                                act_list = [actual]
                        else:
                            # Try to convert iterable to list
                            act_list = [list(row) for row in actual]
                        
                        # Only create diff_grid if dimensions match
                        if act_list:
                            exp_h, exp_w = len(exp_list), len(exp_list[0]) if exp_list else 0
                            act_h, act_w = len(act_list), len(act_list[0]) if act_list else 0
                            
                            if exp_h == act_h and exp_w == act_w:
                                # Same dimensions - create diff grid
                                diff_grid = []
                                for i in range(exp_h):
                                    diff_row = []
                                    for j in range(exp_w):
                                        if exp_list[i][j] != act_list[i][j]:
                                            diff_row.append('x')
                                        else:
                                            diff_row.append(exp_list[i][j])
                                    diff_grid.append(diff_row)
                    
                    except (TypeError, AttributeError, IndexError, ValueError) as e:
                        # Handle any conversion errors - just log the failure
                        act_list = None
                
                error_details.append((idx, exp_list, act_list, diff_grid))
        
        return error_details
    
    best_error_details = process_results(results1)
    second_best_error_details = process_results(results2)
    
    return results1, best_error_details, results2, second_best_error_details

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

def sort_examples_by_size(task: Dict) -> Dict:
    """Sort training and test examples by grid size (smallest first)."""
    def grid_size(grid):
        """Calculate grid size (height * width)"""
        if isinstance(grid, (list, tuple)) and len(grid) > 0:
            return len(grid) * len(grid[0]) if len(grid[0]) > 0 else 0
        return 0
    
    def example_size(example):
        """Get max size of input and output grids"""
        input_size = grid_size(example['input'])
        output_size = grid_size(example['output'])
        return max(input_size, output_size)
    
    # Sort train examples
    if 'train' in task:
        task['train'] = sorted(task['train'], key=example_size)
    
    # Sort test examples
    if 'test' in task:
        task['test'] = sorted(task['test'], key=example_size)
    
    return task

def select_best_programs(candidate_programs, task, task_id, 
                          hypotheses, validations, sample_indices, 
                          dsl_enabled, library, log_dir, program_repair_enabled=False) -> Tuple[TaskResult, Dict]:
    """
    Test k candidate programs and select the best two, then test on test set.
    
    Args:
        candidate_programs: List of k program strings
        task: Task data
        task_id: Task identifier
        hypotheses: List of k hypothesis texts
        validations: List of k validation texts
        sample_indices: List of k sample indices (for tracking)
        dsl_enabled: Whether DSL is enabled
        library: Library object for storing successful programs
        log_dir: Directory for logging
    
    Returns:
        TaskResult object with final selection
        Dictionary with metadata (best_program, second_best_program, scores, etc.)
    """
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
    
    # Test each candidate on training data
    for k, (program, hypothesis, validation, sample_idx) in enumerate(
        zip(candidate_programs, hypotheses, validations, sample_indices)):
        
        if not program or len(program.strip()) == 0:
            all_sample_scores.append((sample_idx, 0.0, None, "extraction_failed"))
            continue
        
        try:
            score, train_results = test_program(program, task, testing='train')
            all_sample_scores.append((sample_idx, score, program, None))
            
            if score > best_score:
                # Demote current best to second best
                second_best_score = best_score
                second_best_program = best_program
                second_best_sample_idx = best_sample_idx
                second_best_hypothesis = best_hypothesis
                second_best_validation = best_validation
                
                # Update best
                best_score = score
                best_program = program
                best_sample_idx = sample_idx
                best_hypothesis = hypothesis
                best_validation = validation
            elif score > second_best_score:
                # Update second best
                second_best_score = score
                second_best_program = program
                second_best_sample_idx = sample_idx
                second_best_hypothesis = hypothesis
                second_best_validation = validation
        except Exception as e:
            all_sample_scores.append((sample_idx, 0.0, None, str(e)))
    
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
    else:
        final_score = best_test_score
        final_program = best_program
        final_hypothesis = best_hypothesis
        final_validation = best_validation
        selected_sample_idx = best_sample_idx
    
    # Add to library if successful (only if DSL enabled)
    success = final_score == 1.0
    if dsl_enabled and success and final_program:
        namespace = globals().copy()
        exec(final_program, namespace)
        if 'solve' in namespace:
            library.add(task_id, final_program)
    
    # Create result
    result = TaskResult(
        task_id=task_id,
        success=success,
        score=final_score,
        program=final_program,
        phase2a_output=final_hypothesis,
        phase2b_output=final_validation
    )
    
    # Detailed logging
    if program_repair_enabled:
         log_path = os.path.join(log_dir, f"{task_id}_repair_selection_summary.txt")
    else:
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
    
    # Return result and metadata for potential repair
    metadata = {
        'best_program': best_program,
        'second_best_program': second_best_program,
        'best_sample_idx': best_sample_idx,
        'second_best_sample_idx': second_best_sample_idx,
        'best_hypothesis': best_hypothesis,
        'second_best_hypothesis': second_best_hypothesis,
        'best_validation': best_validation,
        'second_best_validation': second_best_validation,
        'best_score': best_test_score,
        'second_best_score': second_best_test_score,
        'all_sample_scores': all_sample_scores
    }
    
    return result, metadata

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
    few_shot: bool = True,
    dsl_enabled: bool = True,
    max_api_calls: int = 400,
    program_repair_enabled: bool = False
) -> List[TaskResult]:
    """
    Process all tasks with fully batched API calls and K-sample diversity.
    
    Strategy:
    1. Run find_similar in parallel for all tasks (only if dsl_enabled)
    2. For each task, generate K samples:
       - Batch ALL phase2a prompts (K per task) → send in parallel
       - Batch ALL phase2b prompts (K per task) → send in parallel
       - Batch ALL phase2c prompts (K per task) → send in parallel
    3. Test all K programs per task and select the best
    4. Repair (if enabled): For tasks that failed, generate repair programs for the best two
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
        mode = "DSL" if dsl_enabled else "Pure Python"
        print(f"K-SAMPLE BATCHED PIPELINE ({mode}) - {k_samples} SAMPLES PER TASK", flush=True)
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
            task = sort_examples_by_size(task)
            tasks_data.append((task_id, task))
        except Exception as e:
            print(f"✗ {task_id}: {e}", flush=True)
    
    if verbose:
        print(f"Loaded {len(tasks_data)} tasks\n", flush=True)
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # PHASE 1: Find similar (parallel) - ONLY IF DSL ENABLED
    # ========================================================================
    phase1_results = [None] * len(tasks_data)
    
    if dsl_enabled:
        if verbose:
            print(f"Phase 1: Finding similar programs...", flush=True)
        
        time_start = time.time()
        
        for idx, (task_id, task) in enumerate(tasks_data):
            result = phase1_find_similar(task, task_id, library, timeout, verbose, similar)
            phase1_results[idx] = result
        
        time_phase1 = time.time()
        print(f"Phase 1 complete: {time_phase1 - time_start:.1f}s\n", flush=True)
    else:
        if verbose:
            print(f"Phase 1: Skipped (DSL disabled)\n", flush=True)
        time_start = time.time()
        time_phase1 = time_start
    
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
    
    for idx, (task_id, task) in enumerate(tasks_data):
        phase1_result = phase1_results[idx]
        
        # Skip if DSL enabled and we found perfect match or error
        if dsl_enabled and phase1_result and (phase1_result.perfect_match_found or phase1_result.error):
            continue
        
        phase2a_indices.append(idx)
        
        for k in range(k_samples):
            similar_progs = phase1_result.similar_programs if (dsl_enabled and phase1_result) else None
            prompt = prompter.build_phase2a_prompt(task, similar_progs, dsl_enabled=dsl_enabled)
            phase2a_prompts.append(prompt)
            phase2a_task_sample_pairs.append((idx, k))
                
    if verbose:
        print(f"Sending {len(phase2a_prompts)} prompts to API...", flush=True)
    
    phase2a_outputs = []
    if phase2a_prompts:
        if dsl_enabled:
            system_prompt = """You are an expert at analyzing ARC puzzles and discovering transformation patterns. You are given several training examples of input-output pairs for a puzzle followed by a few similar examples along with similarity scores that might be useful as reference. Your task is to iteratively refine your hypothesis about the transformation pattern given the training examples. 
            
Remember: Your first hypothesis is sticky and excessively convincing to you.
Combat this by evolving your hypothesis as you see each training example."""
        else:
            system_prompt = """You are an expert at analyzing ARC puzzles and discovering transformation patterns. You are given several training examples of input-output pairs for a puzzle. This is followed by a reasoning process where you iteratively refine your hypothesis about the transformation pattern given the training examples. 
            
Remember: Your first hypothesis is sticky and excessively convincing to you.
Combat this by evolving your hypothesis as you see each training example."""
        with ThreadPoolExecutor(max_workers=min(max_api_calls, len(phase2a_prompts))) as executor:
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
            similar_progs = phase1_result.similar_programs if (dsl_enabled and phase1_result) else None
            prompt = prompter.build_phase2b_prompt(task, hypothesis, similar_progs, dsl_enabled=dsl_enabled)
            phase2b_prompts.append(prompt)
            phase2b_task_sample_pairs.append((idx, k))
    
    if verbose:
        print(f"Sending {len(phase2b_prompts)} prompts to API...", flush=True)
    
    phase2b_outputs = []
    if phase2b_prompts:
        if dsl_enabled:
            system_prompt = """You are an expert at analyzing ARC puzzles and discovering transformation patterns.

You are given an initial hypothesis about the puzzle. If the hypothesis doesn't extend to the test input while explaining the training examples, refine it to create a more accurate pattern description. Additionally, you are also given a few programs with similarity scores that you might find useful for reference.
Remember: Your first hypothesis is sticky and excessively convincing to you. The final transformation is a simple sequential transformation that applies to all samples, both training and test.
Combat this by evolving your hypothesis."""
        else:
            system_prompt = """You are an expert at analyzing ARC puzzles and discovering transformation patterns.

You are given an initial hypothesis about the puzzle. If the hypothesis doesn't extend to the test input while explaining the training examples, refine it to create a more accurate pattern description. You must now ensure that it generalizes to the test inputs as well as the training examples.
Remember: Your first hypothesis is sticky and excessively convincing to you. The final transformation is a simple sequential transformation that applies to all samples, both training and test.
Combat this by evolving your hypothesis."""
        
        with ThreadPoolExecutor(max_workers=min(max_api_calls, len(phase2a_prompts))) as executor:
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
            similar_progs = phase1_result.similar_programs if (dsl_enabled and phase1_result) else None
            prompt = prompter.build_phase2c_prompt(
                task, 
                validated_pattern, 
                similar_progs, 
                few_shot=few_shot,
                dsl_enabled=dsl_enabled
            )
            phase2c_prompts.append(prompt)
            phase2c_task_sample_pairs.append((idx, k))
    
    if verbose:
        print(f"Sending {len(phase2c_prompts)} prompts to API...", flush=True)
    
    phase2c_outputs = []
    if phase2c_prompts:
        if dsl_enabled:
            system_prompt = """You are an expert at generating code using the given DSL primitives to solve ARC puzzles. You are provided with a natural language description of the pattern to implement, as well as training and test examples and some similar programs you might find useful as reference. Generate a Python function `def solve(I):` that implements the described transformation using ONLY the provided DSL primitives. Ensure your code is syntactically correct and follows best practices."""
        else:
            system_prompt = """You are an expert at generating Python code to solve ARC puzzles. You are provided with a natural language description of the pattern to implement, as well as training and test examples. Generate a Python function `def solve(I):` that implements the described transformation using pure Python and standard libraries. Ensure your code is syntactically correct and follows best practices."""
        
        with ThreadPoolExecutor(max_workers=min(max_api_calls, len(phase2a_prompts))) as executor:
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

    all_results = []  # Store (result, metadata) tuples
    all_tasks_info = []  # Store (idx, task_id, task) for later use
    repair_prompts = []
    repair_metadata = []  # Store (results_index, task) for each repair prompt
    successful = 0
    total_score = 0.0
    sample_selection_counts = [0] * k_samples
    time_phase2d = time_phase2c
    
    # First loop: Initial selection and collect repair prompts
    for idx, (task_id, task) in enumerate(tasks_data):
        phase1_result = phase1_results[idx]
        
        if idx not in phase2c_indices:
            result = TaskResult(
                task_id=task_id,
                success=False,
                score=0.0,
                error="No code generated"
            )
            metadata = None
            all_results.append((result, metadata))
            all_tasks_info.append((idx, task_id, task))
            continue
        
        # Extract initial k programs
        candidate_programs = [extract_code_from_response(phase2c_results[idx][k]) for k in range(k_samples)]
        hypotheses = [phase2a_results[idx][k] for k in range(k_samples)]
        validations = [phase2b_results[idx][k] for k in range(k_samples)]
        sample_indices = list(range(k_samples))
        
        # Initial selection
        result, metadata = select_best_programs(
            candidate_programs, task, task_id,
            hypotheses, validations, sample_indices,
            dsl_enabled, library, log_dir
        )
        
        all_results.append((result, metadata))
        all_tasks_info.append((idx, task_id, task))
        
        # Build repair prompts if needed
        if program_repair_enabled and result.score < 1.0 and metadata and metadata['best_program']:
            if metadata['second_best_program'] is None:
                metadata['second_best_program'] = metadata['best_program']
            results1, diff, results2, diff2 = get_program_errors(
                metadata['best_program'], metadata['second_best_program'], task, 'train'
            )
            # Calculate train scores
            best_train_score = sum(1 for _, _, correct in results1 if correct) / len(results1) if results1 else 0.0
            second_best_train_score = sum(1 for _, _, correct in results2 if correct) / len(results2) if results2 else 0.0

            for k in range(k_samples):
                if validations[k] != None:
                    validated_pattern = extract_validated_pattern_from_response(validations[k])
                repair_prompt = prompter.build_2d_prompt(
                    task, metadata['best_program'], best_train_score, diff,
                    metadata['second_best_program'], second_best_train_score, diff2,
                    dsl_enabled=dsl_enabled, validated_pattern=validated_pattern
                )
                repair_prompts.append(repair_prompt)
                repair_metadata.append((len(all_results) - 1, task))
                

    repaired_count = 0
    # Batch repair API calls (if any repairs needed)
    if repair_prompts:
        if verbose:
            print(f"Phase 2D: Repairing {len(repair_prompts) // k_samples} programs ({k_samples} samples each)...", flush=True)
        
        
        max_api_calls = 400
        with ThreadPoolExecutor(max_workers=min(max_api_calls, len(repair_prompts))) as executor:
            futures = [executor.submit(vlm_client_phase2.query, p) for p in repair_prompts]
            repair_outputs = [f.result() for f in futures]
        
        # Group repair outputs by task (k samples per task)
        repairs_by_task = {}
        for (result_idx, task), output in zip(repair_metadata, repair_outputs):
            if result_idx not in repairs_by_task:
                repairs_by_task[result_idx] = []
            repairs_by_task[result_idx].append(output)
        
        # Apply repairs
        for result_idx, repair_output_list in repairs_by_task.items():
            old_result, old_metadata = all_results[result_idx]
            task_id = old_result.task_id
            task = None
            
            # Find the task
            for idx, tid, t in all_tasks_info:
                if tid == task_id:
                    task = t
                    break
            
            if not task:
                continue
            
            # Extract repaired programs
            repaired_programs = [extract_code_from_response(out) for out in repair_output_list]
            
            # Re-run selection with repaired programs (keeping original hypotheses/validations)
            hypotheses = [old_metadata['best_hypothesis'], old_metadata['second_best_hypothesis']] + [old_metadata['best_hypothesis']] * (k_samples - 2)
            validations = [old_metadata['best_validation'], old_metadata['second_best_validation']] + [old_metadata['best_validation']] * (k_samples - 2)
            sample_indices = list(range(k_samples))
            
            repaired_result, repaired_metadata = select_best_programs(
                repaired_programs, task, task_id,
                hypotheses, validations, sample_indices,
                dsl_enabled, library, log_dir, program_repair_enabled
            )
            
            # Update if repaired version is better
            if repaired_result.score > old_result.score:
                all_results[result_idx] = (repaired_result, repaired_metadata)
                repaired_count += 1
        
        time_phase2d = time.time()
        if verbose:
            print(f"Repaired {repaired_count} programs\n", flush=True)
            print(f"Phase 2D complete: {time_phase2d - time_phase2c:.1f}s\n", flush=True)

    # Final loop: Print all results and accumulate statistics
    results = []
    for idx, ((result, metadata), (task_idx, task_id, task)) in enumerate(zip(all_results, all_tasks_info)):
        # Track which sample was selected
        if metadata and result.score > 0:
            if result.program == metadata['best_program']:
                sample_selection_counts[metadata['best_sample_idx']] += 1
            elif result.program == metadata['second_best_program']:
                sample_selection_counts[metadata['second_best_sample_idx']] += 1
        
        # Accumulate statistics
        if result.success:
            successful += 1
        total_score += result.score
        results.append(result)
        
        # Print progress
        if verbose:
            status = "✓" if result.success else "✗"
            if metadata and metadata['best_sample_idx'] >= 0:
                if result.program == metadata['best_program']:
                    sample_info = f"sample{metadata['best_sample_idx']}"
                elif result.program == metadata['second_best_program']:
                    sample_info = f"sample{metadata['second_best_sample_idx']}"
                else:
                    sample_info = "library"
            else:
                sample_info = "no code" if not result.program else "library"
            print(f"{status} [{idx+1}/{len(all_results)}] {task_id}: {result.score:.2f} ({sample_info})", flush=True)
        
    time_execution = time.time()
    
    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"TIME BREAKDOWN", flush=True)
    print(f"{'='*80}", flush=True)
    if dsl_enabled:
        print(f"Phase 1 (find_similar): {time_phase1 - time_start:.1f}s", flush=True)
    print(f"Phase 2A (hypothesis × {k_samples}): {time_phase2a - time_phase1:.1f}s", flush=True)
    print(f"Phase 2B (validation × {k_samples}): {time_phase2b - time_phase2a:.1f}s", flush=True)
    print(f"Phase 2C (code gen × {k_samples}): {time_phase2c - time_phase2b:.1f}s", flush=True)
    print(f"Code execution & selection: {time_execution - time_phase2c:.1f}s", flush=True)
    if program_repair_enabled:
        print(f"Programs repaired: {repaired_count}", flush=True)
        print(f"Phase 2D (program repair × {k_samples}): {time_phase2d - time_phase2c:.1f}s", flush=True)
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


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to config/config.yaml relative to the project root
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / 'config' / 'config.yaml'

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def build_run_name(config):
    """Build descriptive run name from config"""
    model = config['model']['name'].split('/')[-1].replace('-', '')  # "grok4fast"
    reasoning = "reasoning" if config['vlm_config']['phase1'].get('extra_params', {}).get('reasoning', {}).get('enabled', False) else "noreasoning"
    dsl = "dsl" if config['process_directory']['dsl_enabled'] else "nodsl"
    k = f"k{config['process_directory']['k_samples']}"
    
    return f"{model}_{reasoning}_{dsl}_{k}"

def main():
    import wandb
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='Experiment name (optional)')
    args = parser.parse_args()
    config = load_config()
    #Provided name or use fallback
    if args.exp_name:
        exp_name = args.exp_name
    elif 'exp_name' in config and config['exp_name']:
        exp_name = config['exp_name']
    else:
        exp_name = build_run_name(config)

    wandb.init(
    project="arc-solver_icml", 
    name=exp_name,
    config=config
)

    PROVIDER = config['provider']

    # Get model configuration from config file
    model = config['model']['name']
    api_base = config['model']['api_base']

    # Get API key from environment based on provider
    if PROVIDER == "grok":
        api_key = os.getenv('OPENROUTER_API_KEY')
    elif PROVIDER == "qwen":
        api_key = None
    elif PROVIDER == "gemini":
        api_key = os.getenv('GEMINI_API_KEY')
        
    # Get VLM configuration from config file
    vlm_phase1_config = config['vlm_config']['phase1']
    vlm_phase2_config = config['vlm_config']['phase2']

    vlm_config_phase1 = VLMConfig(
        api_key=api_key,
        model=model,
        api_base=api_base,
        max_tokens=vlm_phase1_config['max_tokens'],
        max_retries=vlm_phase1_config['max_retries'],
        save_prompts=vlm_phase1_config['save_prompts'],
        prompt_log_dir=vlm_phase1_config['prompt_log_dir'],
        extra_params=vlm_phase1_config.get('extra_params'),
        suppress_errors=True  # Return empty string on errors for parallel API calls
    )
    vlm_config_phase2 = VLMConfig(
        api_key=api_key,
        model=model,
        max_retries=vlm_phase2_config['max_retries'],
        api_base=api_base,
        max_tokens=vlm_phase2_config['max_tokens'],
        extra_params=vlm_phase2_config.get('extra_params'),
        suppress_errors=True  # Return empty string on errors for parallel API calls
    )

    vlm_client_phase1 = create_client(PROVIDER, config=vlm_config_phase1)
    vlm_client_phase2 = create_client(PROVIDER, config=vlm_config_phase2)
    prompter = VLMPrompter()
    library = ProgramLibrary()

    # Get process_directory parameters from config file
    process_params = config['process_directory']
    log_dir = f"logs/{exp_name}"
    results = process_directory(
        data_dir=process_params['data_dir'],
        vlm_client_phase1=vlm_client_phase1,
        vlm_client_phase2=vlm_client_phase2,
        prompter=prompter,
        library=library,
        timeout=process_params['timeout'],
        k_samples=process_params['k_samples'],
        max_find_similar_workers=process_params['max_find_similar_workers'],
        log_dir=log_dir,
        verbose=process_params['verbose'],
        similar=process_params['similar'],
        few_shot=process_params['few_shot'],
        dsl_enabled=process_params['dsl_enabled'],
        program_repair_enabled=process_params['program_repair_enabled'],
        max_api_calls=process_params['max_api_calls']
    )

    # Get output directory from config file
    output_dir = f"results/{exp_name}"

    save_results(results, output_dir=output_dir)
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
    sys.stdout.flush()
    main()