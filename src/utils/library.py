"""
Program library for storing and retrieving solved ARC tasks.
Uses execution-based similarity with parallelized evaluation.
"""

import re
import json
import os
import inspect
import threading
import time
import concurrent.futures
import multiprocessing as mp
from functools import partial
from typing import List, Dict, Set, Tuple, Any

from utils.dsl import *
from utils.constants import *
from . import solvers


def pad_grid(grid,height, width, fill):
    assert isinstance(grid, tuple)
    new_grid = []
    for j in range(height):
        new_row=[]
        for i in range(width):
            if len(grid[0])>i and len(grid)>j:
                new_row.append(grid[j][i])
            else:
                new_row.append(fill)
        new_grid.append(new_row)
    new_grid = tuple(tuple(row) if isinstance(row, list) else row for row in new_grid)
    if len(grid[0])==width and len(grid)==height:
        assert new_grid==grid
    else:
        assert len(new_grid)==height
        assert len(new_grid[0])==width
        for j in range(len(new_grid)):
            for i in range(len(new_grid[0])):
                if j<len(grid) and i<len(grid[0]):
                    assert new_grid[j][i]==grid[j][i]
                else:
                    assert new_grid[j][i]==fill
    return new_grid


def calculate_grid_similarity(predicted: Any, expected: Any) -> float:
    """
    Calculate similarity between two grids.
    
    Returns:
        Similarity score between 0.0 (no match) and 1.0 (exact match)
    """
    try:
        # Convert lists to tuples if needed
        if isinstance(predicted, list):
            predicted = tuple(tuple(row) if isinstance(row, list) else row for row in predicted)
        if isinstance(expected, list):
            expected = tuple(tuple(row) if isinstance(row, list) else row for row in expected)
        
        if not isinstance(predicted, tuple) or not isinstance(expected, tuple):
            return 0.0
        
        pred_h = len(predicted)
        exp_h = len(expected)
        
        if pred_h == 0 or exp_h == 0:
            return 0.0
        
        pred_w = len(predicted[0]) if pred_h > 0 else 0
        exp_w = len(expected[0]) if exp_h > 0 else 0
        
        # Dimension mismatch penalty
        if pred_h != exp_h or pred_w != exp_w:
            max_h = max(pred_h, exp_h)
            max_w = max(pred_w, exp_w)

            predicted=pad_grid(grid=predicted,height=max_h, width=max_w, fill=-1)
            expected=pad_grid(grid=expected,height=max_h, width=max_w, fill=-2)
            
            correct = 0
            total = max_h*max_w
            
            for i in range(max_h):
                for j in range(max_w):
                    if predicted[i][j] == expected[i][j]:
                        correct += 1
            
            cell_accuracy = correct / total if total > 0 else 0.0
            
            return cell_accuracy
        
        # Exact dimension match - calculate cell accuracy
        total_cells = pred_h * pred_w
        correct_cells = 0
        
        for i in range(pred_h):
            for j in range(pred_w):
                if predicted[i][j] == expected[i][j]:
                    correct_cells += 1
        
        return correct_cells / total_cells if total_cells > 0 else 0.0
        
    except Exception:
        return 0.0


def execute_program_safely(solve_func, input_grid: Tuple[Tuple[int]], timeout_seconds: int = 2) -> Tuple[Any, str]:
    """Execute program with timeout protection"""
    
    # Convert lists to tuples if needed
    if isinstance(input_grid, list):
        input_grid = tuple(tuple(row) for row in input_grid)
    
    result = {'output': None, 'error': None}
    
    def run_program():
        try:
            result['output'] = solve_func(input_grid)
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            result['error'] = f"{error_type}: {error_msg}"
    
    # Run in a thread with timeout
    thread = threading.Thread(target=run_program, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    # If thread still alive, it timed out
    if thread.is_alive():
        result['error'] = f"Timeout: Execution exceeded {timeout_seconds}s"
    
    return result['output'], result['error']


def extract_functions(code: str) -> Set[str]:
    """
    Extract DSL function names from code.
    
    Returns:
        Set of DSL function names found
    """
    DSL_FUNCTIONS = {
        'identity', 'compose', 'chain', 'fork', 'apply', 'mapply',
        'lbind', 'rbind', 'matcher', 'extract',
        'hmirror', 'vmirror', 'dmirror', 'cmirror',
        'rot90', 'rot180', 'rot270',
        'vconcat', 'hconcat', 'crop', 'upscale', 'downscale',
        'hsplit', 'vsplit', 'tophalf', 'bottomhalf', 'lefthalf', 'righthalf',
        'objects', 'colorfilter', 'sizefilter', 'ofcolor',
        'toobject', 'normalize', 'toindices', 'asindices',
        'fill', 'paint', 'replace', 'switch', 'shift', 'move', 'cover',
        'size', 'height', 'width', 'shape', 'palette', 'mostcolor', 'leastcolor',
        'ulcorner', 'urcorner', 'llcorner', 'lrcorner', 'center', 'corners',
        'color', 'index', 'occurrences',
        'box', 'inbox', 'outbox', 'backdrop', 'delta',
        'vfrontier', 'hfrontier', 'shoot', 'connect', 'position',
        'gravitate', 'compress', 'frontiers',
        'combine', 'intersection', 'difference', 'merge', 'dedupe',
        'sfilter', 'mfilter', 'contained', 'initset',
        'argmax', 'argmin', 'valmax', 'valmin', 'maximum', 'minimum',
        'mostcommon', 'leastcommon', 'order', 'repeat',
        'add', 'subtract', 'multiply', 'divide', 'invert',
        'double', 'halve', 'increment', 'decrement', 'sign',
        'toivec', 'tojvec',
        'equality', 'both', 'either', 'flip', 'positive', 'even', 'greater',
        'canvas', 'astuple', 'trim',
        'cellwise', 'hperiod', 'vperiod',
    }
    
    found = set()
    for func in DSL_FUNCTIONS:
        if re.search(rf'\b{func}\b', code):
            found.add(func)
    
    return found


def evaluate_single_program(prog_data: Tuple, train_examples: List[Dict], min_similarity: float, timeout: int = 2) -> Dict:
    """
    Evaluate a single program against training examples.
    Must be a module-level function for multiprocessing.
    
    Args:
        prog_data: Tuple of (task_id, solve_func, functions)
        train_examples: List of training examples
        min_similarity: Minimum similarity threshold
        timeout: Timeout per execution in seconds
    
    Returns:
        Dict with program results or None if below threshold
    """
    task_id, solve_func, functions = prog_data
    
    similarities = []
    error_count = 0
    has_perfect = False
    errors_list = []
    
    for example in train_examples:
        input_grid = example['input']
        expected_output = example['output']
        
        output, error = execute_program_safely(solve_func, input_grid, timeout_seconds=timeout)
        
        if error is not None:
            error_count += 1
            similarities.append(0.0)
            errors_list.append(error)
        else:
            sim = calculate_grid_similarity(output, expected_output)
            similarities.append(sim)
            if sim == 1.0:
                has_perfect = True
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    
    # Return results even if below threshold for debugging
    try:
        source = inspect.getsource(solve_func)
    except:
        source = ""
    
    return {
        'program': source,
        'task_id': task_id,
        'similarity': avg_similarity,
        'example_scores': similarities,
        'errors': error_count,
        'error_messages': errors_list[:1] if errors_list else [],  # Keep first error for debugging
        'functions': functions,
        'has_perfect_example': has_perfect,
        'above_threshold': avg_similarity >= min_similarity
    }


class ProgramLibrary:
    """Storage and retrieval of solved ARC programs with execution-based similarity"""
    
    def __init__(self):
        """Initialize program library by loading solvers from solvers.py"""
        self.programs = []
        self._load_solvers_from_module()
    
    def _load_solvers_from_module(self):
        """Load all solve_* functions from solvers.py"""
        try:
            from . import solvers
            
            for name, obj in inspect.getmembers(solvers):
                if name.startswith('solve_') and callable(obj):
                    task_id = name.replace('solve_', '')
                    
                    try:
                        source = inspect.getsource(obj)
                        functions = extract_functions(source)
                        
                        self.programs.append({
                            'task_id': task_id,
                            'solve_func': obj,
                            'functions': functions
                        })
                    except Exception as e:
                        print(f"DEBUG: Failed to add {name}: {e}", flush=True)
                        
        except Exception as e:
            print(f"DEBUG: Failed to import solvers: {e}", flush=True)
    
    def save(self, filepath: str):
        """Save library to disk (JSON format)."""
        data = []
        for prog in self.programs:
            try:
                source = inspect.getsource(prog['solve_func'])
            except:
                source = ""
            
            data.append({
                'task_id': prog['task_id'],
                'code': source,
                'functions': list(prog['functions'])
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str = None):
        """Reload solvers from solvers.py"""
        self.programs.clear()
        self._load_solvers_from_module()
    
    def add(self, task_id: str, program_source: str):
        """
        Store successful solution.
        
        Args:
            task_id: Unique identifier for the task
            program_source: The complete program source code as string
        """
        functions = extract_functions(program_source)
        
        self.programs.append({
            'task_id': task_id,
            'solve_func': program_source,
            'functions': functions
        })
    
    def find_similar(self, 
                    train_examples: List[Dict[str, Any]], 
                    top_k: int = 3,
                    min_similarity: float = 0.0,
                    n_workers: int = None,
                    timeout: int = 2,
                    verbose: bool = False) -> List[Dict]:
        """
        Find programs based on execution similarity (parallelized).
        
        Args:
            train_examples: Training examples to test against
            top_k: Number of top programs to return
            min_similarity: Minimum similarity threshold
            n_workers: Number of parallel workers (default: CPU count - 4)
            timeout: Timeout per program execution in seconds
        
        Returns:
            List of similar programs sorted by similarity
        """
        if verbose:
            print(f"\n=== DEBUG find_similar ===", flush=True)
            print(f"Number of train examples: {len(train_examples)}", flush=True)
            print(f"Number of programs in library: {len(self.programs)}", flush=True)
            print(f"min_similarity threshold: {min_similarity}", flush=True)
            print(f"top_k: {top_k}", flush=True)
        
        if not train_examples:
            print("WARNING: No train examples provided!", flush=True)
            return []
        
        # Determine number of workers
        if n_workers is None:
            cpu_count = mp.cpu_count()
            n_workers = max(1, cpu_count - 8)  # Leave some CPUs free
        if verbose:
            print(f"Using {n_workers} parallel workers (timeout: {timeout}s per program)", flush=True)
        
        # Prepare program data for parallel processing
        prog_data_list = [
            (prog['task_id'], prog['solve_func'], prog['functions'])
            for prog in self.programs
        ]
        
        perfect_programs = []
        other_programs = []
        all_results = []  # Track all results for debugging
        
        start_time = time.time()
        
        # Use ProcessPoolExecutor for true parallelism
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Create partial function with fixed arguments
            eval_func = partial(
                evaluate_single_program,
                train_examples=train_examples,
                min_similarity=min_similarity,
                timeout=timeout
            )
            
            # Submit all tasks
            future_to_idx = {
                executor.submit(eval_func, prog_data): idx 
                for idx, prog_data in enumerate(prog_data_list)
            }
            
            # Process results as they complete
            completed = 0
            failed = 0
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed += 1
                
                # Progress update every 50 programs
                if completed % 50 == 0 or completed == len(self.programs):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    print(f"Progress: {completed}/{len(self.programs)} programs ({rate:.1f}/s, {elapsed:.1f}s elapsed)", flush=True)
                
                try:
                    result = future.result(timeout=timeout + 1)  # Slightly longer than execution timeout
                    
                    if result is not None:
                        all_results.append(result)
                        
                        if result['above_threshold']:
                            if result['has_perfect_example']:
                                perfect_programs.append(result)
                            else:
                                other_programs.append(result)
                            
                except concurrent.futures.TimeoutError:
                    failed += 1
                    if failed <= 5:  # Only log first few
                        print(f"WARNING: Program {idx} ({prog_data_list[idx][0]}) timed out", flush=True)
                except Exception as e:
                    failed += 1
                    if failed <= 5:  # Only log first few errors
                        print(f"WARNING: Program {idx} ({prog_data_list[idx][0]}) failed: {e}", flush=True)
        
        total_time = time.time() - start_time
        print(f"\nEvaluation complete: {completed}/{len(self.programs)} programs in {total_time:.1f}s", flush=True)
        if failed > 0:
            print(f"Failed/timed out: {failed} programs", flush=True)
        
        # DEBUGGING: Show statistics about all results
        if len(all_results) > 0:
            error_counts = sum(1 for r in all_results if r['errors'] > 0)
            max_sim = max((r['similarity'] for r in all_results), default=0.0)
            avg_sim = sum(r['similarity'] for r in all_results) / len(all_results)
            
            print(f"\n=== Execution Statistics ===", flush=True)
            print(f"Programs with errors: {error_counts}/{len(all_results)}", flush=True)
            print(f"Max similarity achieved: {max_sim:.3f}", flush=True)
            print(f"Average similarity: {avg_sim:.3f}", flush=True)
            
            # Show sample errors
            programs_with_errors = [r for r in all_results if r['errors'] > 0 and len(r.get('error_messages', [])) > 0]
            if programs_with_errors:
                print(f"\nSample errors from first 3 programs:", flush=True)
                for r in programs_with_errors[:3]:
                    print(f"  {r['task_id']}: {r['error_messages'][0]}", flush=True)
            
            # Show programs closest to threshold
            sorted_results = sorted(all_results, key=lambda x: x['similarity'], reverse=True)
            print(f"\nTop 5 programs by similarity:", flush=True)
            for r in sorted_results[:5]:
                print(f"  {r['task_id']}: {r['similarity']:.3f} (errors: {r['errors']}/4)", flush=True)
        
        # Sort both lists by similarity
        perfect_programs.sort(key=lambda x: x['similarity'], reverse=True)
        other_programs.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"\n=== Results ===", flush=True)
        print(f"Programs with perfect examples: {len(perfect_programs)}", flush=True)
        print(f"Other programs above threshold: {len(other_programs)}", flush=True)
        
        # Return all perfect programs + top_k others
        if perfect_programs:
            result = perfect_programs + other_programs[:top_k]
        else:
            result = other_programs[:top_k]
        
        print(f"Returning {len(result)} programs", flush=True)
        if len(result) > 0:
            print(f"Top program: {result[0]['task_id']} (sim: {result[0]['similarity']:.3f})", flush=True)
            if len(result) > 1:
                print(f"Second program: {result[1]['task_id']} (sim: {result[1]['similarity']:.3f})", flush=True)
        
        return result
    
    def test_program(self, 
                    solve_func, 
                    train_examples: List[Dict[str, Any]],
                    timeout: int = 2) -> Dict:
        """
        Test a single program on training examples.
        
        Args:
            solve_func: Function to test
            train_examples: List of {'input': grid, 'output': grid}
            timeout: Timeout per execution in seconds
        
        Returns:
            Dict with keys: avg_similarity, example_results
        """
        results = []
        
        for i, example in enumerate(train_examples):
            input_grid = example['input']
            expected_output = example['output']
            
            output, error = execute_program_safely(solve_func, input_grid, timeout_seconds=timeout)
            
            if error is not None:
                results.append({
                    'example_idx': i,
                    'success': False,
                    'error': error,
                    'similarity': 0.0
                })
            else:
                sim = calculate_grid_similarity(output, expected_output)
                results.append({
                    'example_idx': i,
                    'success': True,
                    'similarity': sim,
                    'output': output
                })
        
        avg_sim = sum(r['similarity'] for r in results) / len(results) if results else 0.0
        
        return {
            'avg_similarity': avg_sim,
            'example_results': results
        }
    
    def get(self, task_id: str) -> Dict:
        """Get program by task ID"""
        for prog in self.programs:
            if prog['task_id'] == task_id:
                return prog
        return None
    
    def __len__(self):
        """Number of programs in library"""
        return len(self.programs)


def format_similar_programs_for_prompt(similar_programs: List[Dict]) -> List[Dict[str, Any]]:
    """
    Format library matches for insertion into prompts.
    
    Returns:
        List of dicts with keys: program, similarity, task_id
    """
    if not similar_programs:
        return []
    
    formatted = []
    
    for entry in similar_programs:
        formatted.append({
            'program': entry['program'],
            'similarity': entry['similarity'],
            'task_id': entry['task_id']
        })
    
    return formatted
