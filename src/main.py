"""
Main pipeline for ARC task solving using execution-based similarity.

Pipeline: Sequential phases per task, parallelized across tasks
- Each task: Phase 1 (find_similar) → Phase 2 (generate solution) sequentially
- Multiple tasks run in parallel
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import os
import sys
import threading
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
        return self.client.query(prompt, system_prompt)
    
    def __getattr__(self, name):
        return getattr(self.client, name)


@dataclass
class TaskResult:
    """Result of attempting to solve a task"""
    task_id: str
    success: bool
    score: float
    program: Optional[str] = None
    phase2a_output: Optional[str] = None  # Hypothesis formation
    phase2b_output: Optional[str] = None  # Hypothesis validation
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


def test_program(program_code: str, task: Dict) -> Tuple[float, List[Tuple[Any, Any, bool]]]:
    """
    Test a program against task training examples.
    
    Returns:
        - Average score across examples
        - List of (expected_output, actual_output, passed) tuples
    """
    namespace = globals().copy()
    
    try:
        exec(program_code, namespace)
        
        if 'solve' not in namespace:
            return 0.0, []
        
        solve_fn = namespace['solve']
        scores = []
        results = []
        
        for example in task['test']:
            inp = example['input']
            expected = example['output']
            if isinstance(inp, list):
                inp = tuple(tuple(row) for row in inp)
            try:
                actual = solve_fn(inp)
                score = calculate_grid_similarity(actual, expected)
                scores.append(score)
                results.append((expected, actual, score == 1.0))
            except Exception as e:
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
    timeout: int = 2
) -> Phase1Result:
    """
    Phase 1: Find similar programs by execution.
    
    Args:
        task: Task dictionary with 'train' examples
        task_id: Unique task identifier
        library: Program library
        timeout: Timeout per program execution
        
    Returns:
        Phase1Result with similar programs and library matches
    """
    try:
        # Find similar programs by execution (now sequential)
        similar_programs = library.find_similar(
            train_examples=task['train'],
            top_k=5,
            min_similarity=0.1,
            timeout=timeout
        )
        
        # Test library programs for perfect match
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


def phase2_generate_solution(
    phase1_result: Phase1Result,
    vlm_client_phase1: BaseVLMClient,
    vlm_client_phase2: BaseVLMClient,
    prompter: VLMPrompter,
    library: ProgramLibrary,
    log_dir: str = "logs"
) -> TaskResult:
    """
    Phase 2: Generate solution using two-stage pattern discovery and code generation.
    """
    task_id = phase1_result.task_id
    task = phase1_result.task
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Handle Phase 1 perfect match or errors
        if phase1_result.perfect_match_found:
            return TaskResult(
                task_id=task_id,
                success=True,
                score=1.0,
                program=phase1_result.best_library_program
            )
        
        if phase1_result.error:
            return TaskResult(
                task_id=task_id,
                success=False,
                score=0.0,
                error=f"Phase 1 failed: {phase1_result.error}"
            )
        
        # ====================================================================
        # PHASE 2A: Initial Hypothesis Formation (Training Only)
        # ====================================================================
        phase2a_prompt = prompter.build_phase2a_prompt(
            task, 
            phase1_result.similar_programs
        )
        
        phase2a_output = vlm_client_phase1.query(
            phase2a_prompt,
            system_prompt="""You are an expert at analyzing ARC puzzles and discovering transformation patterns.
            
Remember: Your first hypothesis is sticky and excessively convincing to you.
Combat this by evolving your hypothesis as you see each training example."""
        )
        
        # Log Phase 2A output
        phase2a_log_path = os.path.join(log_dir, f"{task_id}_phase2a_hypothesis.txt")
        with open(phase2a_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Task ID: {task_id}\n")
            f.write("="*80 + "\n")
            f.write("PHASE 2A: HYPOTHESIS FORMATION (TRAINING ONLY)\n")
            f.write("="*80 + "\n\n")
            f.write(phase2a_output)
        
        # Extract hypothesis from phase2a_output
        hypothesis = extract_hypothesis_from_response(phase2a_output)
        
        # ====================================================================
        # PHASE 2B: Hypothesis Validation (Training + Test)
        # ====================================================================
        phase2b_prompt = prompter.build_phase2b_prompt(
            task,
            hypothesis,
            phase1_result.similar_programs
        )
        
        phase2b_output = vlm_client_phase1.query(
            phase2b_prompt,
            system_prompt="""You are an expert at analyzing ARC puzzles and discovering transformation patterns.

You are given an initial hypothesis about the puzzle. If the hypothesis doesn't extend to the test input while explaining the training examples, refine it to create a more accurate pattern description.
Remember: Your first hypothesis is sticky and excessively convincing to you. The final transformation is a simple transformation that applies to all samples, both training and test.
Combat this by evolving your hypothesis"""
        )
        
        # Log Phase 2B output
        phase2b_log_path = os.path.join(log_dir, f"{task_id}_phase2b_validation.txt")
        with open(phase2b_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Task ID: {task_id}\n")
            f.write("="*80 + "\n")
            f.write("PHASE 2B: HYPOTHESIS VALIDATION (TRAINING + TEST)\n")
            f.write("="*80 + "\n\n")
            f.write("INITIAL HYPOTHESIS:\n")
            f.write("-"*80 + "\n")
            f.write(hypothesis + "\n")
            f.write("-"*80 + "\n\n")
            f.write("VALIDATION OUTPUT:\n")
            f.write("-"*80 + "\n")
            f.write(phase2b_output)
        
        # Extract validated pattern from phase2b_output
        validated_pattern = extract_validated_pattern_from_response(phase2b_output)
        
        # ====================================================================
        # PHASE 2C: Code Generation
        # ====================================================================
        phase2c_prompt = prompter.build_phase2c_prompt(
            task, 
            validated_pattern,  # Use validated pattern
            phase1_result.similar_programs
        )
        
        phase2c_output = vlm_client_phase2.query(
            phase2c_prompt,
            system_prompt="You are an expert at generating code using the given DSL primitives to solve ARC puzzles. You are provided with a natural language description of the pattern to implement, as well as training and test examples and some similar programs you might find useful as reference. Generate a Python function `def solve(I):` that implements the described transformation using ONLY the provided DSL primitives. Ensure your code is syntactically correct and follows best practices"
        )
        
        # ====================================================================
        # EXTRACT AND TEST GENERATED CODE
        # ====================================================================
        generated_code = extract_code_from_response(phase2c_output)
        
        if not generated_code:
            if phase1_result.best_library_program and phase1_result.best_library_score > 0.5:
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    score=phase1_result.best_library_score,
                    program=phase1_result.best_library_program,
                    phase2a_output=phase2a_output,
                    phase2b_output=phase2b_output,
                    error="Code extraction failed, using library fallback"
                )
            
            return TaskResult(
                task_id=task_id,
                success=False,
                score=0.0,
                phase2a_output=phase2a_output,
                phase2b_output=phase2b_output,
                error="Failed to extract code from response"
            )
        
        score, results = test_program(generated_code, task)
        
        # Log Phase 2C output with test results
        phase2c_log_path = os.path.join(log_dir, f"{task_id}_phase2c_results.txt")
        with open(phase2c_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Task ID: {task_id}\n")
            f.write("="*80 + "\n")
            f.write("PHASE 2C: CODE GENERATION & TEST RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write("GENERATED CODE:\n")
            f.write("-"*80 + "\n")
            f.write(generated_code + "\n")
            f.write("-"*80 + "\n\n")
            
            f.write(f"SCORE: {score:.2f}\n\n")
            
            f.write("TEST RESULTS:\n")
            f.write("-"*80 + "\n")
            for i, (expected, actual, passed) in enumerate(results, 1):
                f.write(f"\nExample {i}: {'✓ PASS' if passed else '✗ FAIL'}\n")
                f.write(f"Expected Output:\n")
                if expected:
                    f.write(f"{json.dumps([list(row) for row in expected], indent=2)}\n")
                else:
                    f.write("None\n")
                f.write(f"Actual Output:\n")
                if actual:
                    f.write(f"{json.dumps([list(row) for row in actual], indent=2)}\n")
                else:
                    f.write("None (execution failed)\n")
                f.write("-"*40 + "\n")
        
        # ====================================================================
        # DECIDE FINAL PROGRAM
        # ====================================================================
        success = score == 1.0
        final_program = generated_code
        final_score = score
        
        if not success and phase1_result.best_library_score > score:
            final_program = phase1_result.best_library_program
            final_score = phase1_result.best_library_score
        
        # ====================================================================
        # SAVE TO LIBRARY IF SUCCESSFUL
        # ====================================================================
        if success:
            namespace = globals().copy()
            exec(final_program, namespace)
            if 'solve' in namespace:
                library.add(task_id, final_program)
        
        return TaskResult(
            task_id=task_id,
            success=success,
            score=final_score,
            program=final_program,
            phase2a_output=phase2a_output,
            phase2b_output=phase2b_output
        )
        
    except Exception as e:
        return TaskResult(
            task_id=task_id,
            success=False,
            score=0.0,
            error=str(e)
        )


def extract_hypothesis_from_response(response: str) -> str:
    """Extract the final hypothesis from phase 2a response."""
    import re
    
    # Try to find pattern_summary first
    pattern_match = re.search(r'<pattern_summary>(.*?)</pattern_summary>', 
                             response, re.DOTALL)
    if pattern_match:
        return pattern_match.group(1).strip()
    
    # Fall back to last hypothesis tag
    hypothesis_matches = re.findall(r'<hypothesis_\d+>(.*?)</hypothesis_\d+>', 
                                   response, re.DOTALL)
    if hypothesis_matches:
        return hypothesis_matches[-1].strip()
    
    # If no tags found, return last substantial paragraph
    paragraphs = [p.strip() for p in response.split('\n\n') if len(p.strip()) > 50]
    return paragraphs[-1] if paragraphs else response[-500:]


def extract_validated_pattern_from_response(response: str) -> str:
    """Extract the validated pattern from phase 2b response."""
    import re
    
    # Try to find validated_pattern first
    pattern_match = re.search(r'<validated_pattern>(.*?)</validated_pattern>', 
                             response, re.DOTALL)
    if pattern_match:
        return pattern_match.group(1).strip()
    
    # If no tag found, return entire response (it's all the validated pattern)
    return response.strip()


def process_single_task(
    task_id: str,
    task: Dict,
    vlm_client_phase1: BaseVLMClient,
    vlm_client_phase2: BaseVLMClient,
    prompter: VLMPrompter,
    library: ProgramLibrary,
    timeout: int = 2,
    log_dir: str = "logs"
) -> TaskResult:
    """
    Process a single task through all phases sequentially.
    
    Args:
        task_id: Task identifier
        task: Task data
        vlm_client_phase1: VLM client for pattern discovery
        vlm_client_phase2: VLM client for code generation
        prompter: Prompt builder
        library: Program library
        timeout: Timeout per program execution
        log_dir: Directory for logs
    
    Returns:
        TaskResult
    """
    # Phase 1: Find similar programs
    phase1_result = phase1_find_similar(
        task=task,
        task_id=task_id,
        library=library,
        timeout=timeout
    )
    
    # Phase 2: Generate solution
    phase2_result = phase2_generate_solution(
        phase1_result=phase1_result,
        vlm_client_phase1=vlm_client_phase1,
        vlm_client_phase2=vlm_client_phase2,
        prompter=prompter,
        library=library,
        log_dir=log_dir
    )
    
    return phase2_result


def process_directory(
    data_dir: str,
    vlm_client_phase1: BaseVLMClient,
    vlm_client_phase2: BaseVLMClient,
    prompter: VLMPrompter,
    library: ProgramLibrary,
    timeout: int = 2,
    max_concurrent_tasks: int = 10,
    log_dir: str = "logs"
) -> List[TaskResult]:
    """
    Process all task files with sequential phases per task, parallelized across tasks.
    
    Args:
        data_dir: Directory containing task JSON files
        vlm_client_phase1: VLM client for pattern discovery
        vlm_client_phase2: VLM client for code generation
        prompter: Prompt builder
        library: Program library
        timeout: Timeout per program execution
        max_concurrent_tasks: Maximum number of tasks to process in parallel
        log_dir: Directory for logs
    
    Returns:
        List of TaskResult objects
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory not found: {data_dir}", flush=True)
        return []
    
    json_files = sorted(data_path.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}", flush=True)
        return []
    
    print(f"\n{'='*80}", flush=True)
    print(f"SEQUENTIAL PIPELINE (PARALLELIZED ACROSS TASKS)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total tasks: {len(json_files)}", flush=True)
    print(f"Max concurrent tasks: {max_concurrent_tasks}", flush=True)
    print(f"Timeout per program: {timeout}s", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    # Load all tasks
    print(f"Loading {len(json_files)} tasks...", flush=True)
    tasks_data = []
    
    for task_file in json_files:
        task_id = task_file.stem
        try:
            with open(task_file, 'r') as f:
                task = json.load(f)
            tasks_data.append((task_id, task))
        except Exception as e:
            print(f"✗ {task_id}: {e}", flush=True)
    
    print(f"Loaded {len(tasks_data)} valid tasks\n", flush=True)
    
    # Statistics tracking
    results = []
    results_lock = threading.Lock()
    stats = {
        'completed': 0,
        'successful': 0,
        'total_score': 0.0
    }
    stats_lock = threading.Lock()
    
    def worker(task_id, task):
        """Worker function to process a single task"""
        try:
            result = process_single_task(
                task_id=task_id,
                task=task,
                vlm_client_phase1=vlm_client_phase1,
                vlm_client_phase2=vlm_client_phase2,
                prompter=prompter,
                library=library,
                timeout=timeout,
                log_dir=log_dir
            )
            
            with stats_lock:
                stats['completed'] += 1
                if result.success:
                    stats['successful'] += 1
                stats['total_score'] += result.score
                
                status = "✓" if result.success else "✗"
                print(f"{status} [{stats['completed']}/{len(tasks_data)}] {result.task_id}: {result.score:.2f}", flush=True)
            
            with results_lock:
                results.append(result)
            
            return result
            
        except Exception as e:
            print(f"✗ Error processing {task_id}: {e}", flush=True)
            return TaskResult(
                task_id=task_id,
                success=False,
                score=0.0,
                error=str(e)
            )
    
    # Process tasks in parallel
    print(f"Starting task processing...\n", flush=True)
    
    with ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        futures = [
            executor.submit(worker, task_id, task)
            for task_id, task in tasks_data
        ]
        
        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Task failed with exception: {e}", flush=True)
    
    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total tasks: {len(tasks_data)}", flush=True)
    print(f"Successful: {stats['successful']}/{len(tasks_data)} ({100*stats['successful']/len(tasks_data):.1f}%)", flush=True)
    print(f"Average score: {stats['total_score']/len(tasks_data):.2f}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    return results


def save_results(results: List[TaskResult], output_dir: str = 'results') -> None:
    """Save results to JSON and CSV files."""
    import csv
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON
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
    print(f"Saved detailed results to {json_file}", flush=True)
    
    # CSV
    csv_file = output_path / 'summary.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'success', 'score', 'error'])
        for r in results:
            writer.writerow([r.task_id, r.success, f'{r.score:.2f}', r.error or ''])
    print(f"Saved summary to {csv_file}", flush=True)


def main():
    from dotenv import load_dotenv
    """Main entry point"""
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
    elif PROVIDER == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        api_base = "https://api.openai.com/v1"
        model = "gpt-4o-mini"  # or "gpt-4o-mini" for cheaper option
        
    vlm_config_phase1 = VLMConfig(
        api_key=api_key,
        model=model,
        api_base=api_base,
        max_tokens=16384,
        max_retries=3,
        save_prompts=False,
        prompt_log_dir="prompts_test_1"#TODO change dir if needed
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
    
    # Run simplified pipeline
    results = process_directory(
        data_dir='data_v1/eval_size_10',#TODO change dir if needed
        vlm_client_phase1=vlm_client_phase1,
        vlm_client_phase2=vlm_client_phase2,
        prompter=prompter,
        library=library,
        timeout=2,
        max_concurrent_tasks=10,  # Process 10 tasks in parallel
        log_dir="logs_test_1"#TODO change dir if needed
    )
    
    save_results(results, output_dir='results/test_1')#TODO change dir if needed


if __name__ == "__main__":
    sys.stdout.flush()
    main()