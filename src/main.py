"""
Main pipeline for ARC task solving using execution-based similarity.

Pipeline: Program Similarity → Pattern Discovery → Code Generation
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import os
import sys

print("Script starting...", flush=True)
sys.stdout.flush()
sys.path.append(str(Path(__file__).resolve().parent.parent))

from vlm_prompter import VLMPrompter
from vlm_client import VLMConfig, create_client, BaseVLMClient
from utils.library import ProgramLibrary, calculate_grid_similarity
from utils.dsl import *
from utils.constants import *

print("Imports done...", flush=True)
sys.stdout.flush()

@dataclass
class TaskResult:
    """Result of attempting to solve a task"""
    task_id: str
    success: bool
    score: float
    program: Optional[str] = None
    phase1_output: Optional[str] = None
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
        
        for example in task['train']:
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


def solve_task(
    task: Dict,
    task_id: str,
    vlm_client_phase1: BaseVLMClient,
    vlm_client_phase2: BaseVLMClient,
    prompter: VLMPrompter,
    library: ProgramLibrary,
    verbose: bool = True,
    n_workers: int = None,
    timeout: int = 2,
    log_dir: str = "logs"
) -> TaskResult:
    """
    Solve a single ARC task using execution-based pipeline.
    
    Pipeline:
    1. Find similar programs by execution (parallelized)
    2. Phase 1: Pattern discovery (natural language) with similar programs
    3. Phase 2: Code generation with pattern + similar programs
    
    Args:
        task: Task dictionary with 'train' examples
        task_id: Unique task identifier
        vlm_client: VLM client for queries
        prompter: Prompt builder
        library: Program library
        verbose: Print progress
        n_workers: Number of parallel workers (None = auto)
        timeout: Timeout per program execution in seconds
        log_dir: Directory to save logs (default: "logs")
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*80}", flush=True)
        print(f"Solving Task: {task_id}", flush=True)
        print(f"{'='*80}", flush=True)
    
    try:
        # ====================================================================
        # STEP 1: Find Similar Programs by Execution (PARALLELIZED)
        # ====================================================================
        if verbose:
            print("\n🔍 Finding similar programs by execution...", flush=True)
        
        similar_programs = library.find_similar(
            train_examples=task['train'],
            top_k=40,
            min_similarity=0.0,
            n_workers=n_workers,
            timeout=timeout
        )
        
        if verbose:
            if similar_programs:
                print(f"   Found {len(similar_programs)} similar programs:", flush=True)
                for i, prog in enumerate(similar_programs[:3], 1):
                    print(f"   {i}. Task {prog['task_id']}: {prog['similarity']:.2f}", flush=True)
            else:
                print("   No similar programs found", flush=True)
        
        # ====================================================================
        # TEST LIBRARY PROGRAMS: Try existing solutions first
        # ====================================================================
        best_library_score = 0.0
        best_library_program = None
        
        if similar_programs:
            best_match = similar_programs[0]
            best_library_score = best_match['similarity']
            best_library_program = best_match['program']
            if verbose:
                print(f"\n✓ Best library match: Task {best_match['task_id']} ({best_library_score:.2f})", flush=True)
            
            if best_library_score == 1.0:
                if verbose:
                    print(f"   → Perfect match found! Using library solution.", flush=True)
                return TaskResult(
                    task_id=task_id,
                    success=True,
                    score=1.0,
                    program=best_library_program
                )
                
        # ====================================================================
        # PHASE 1: Pattern Discovery (Natural Language)
        # ====================================================================
        if verbose:
            print("\n📝 Phase 1: Pattern Discovery...", flush=True)
        
        # Pass the raw similar_programs list - prompter will format it
        phase1_prompt = prompter.build_phase1_prompt(task, similar_programs)
        
        phase1_output = vlm_client_phase1.query(
            phase1_prompt,
            system_prompt=""""You are an expert at solving ARC puzzles by thinking like a human playing a visual puzzle game.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL COGNITIVE FRAME:

These puzzles exist in an OBJECT/ACTION space, not a pixel coordinate space.
Think compositionally like playing Pacman, Sokoban, or Tetris:

✓ GOOD: "Extract the largest red object, rotate it 90°, align with top border"
✓ GOOD: "Red pixels shoot downward in free vertical lanes until hitting a wall"
✓ GOOD: "Fill each enclosed region with the color of its boundary"

✗ BAD: "Pixels at positions where value > 0 and row == col become value 7"
✗ BAD: "Apply transformation matrix to coordinates meeting condition X"

Objects: Connected components, shapes, borders, regions, bounding boxes
Actions: Extract, rotate, shoot, bounce, fill, merge, filter, align, crop

Think as sequences: "First find X, then do Y to it, then place result at Z"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KNOWN FAILURE MODES:

1. Premature narrowing: You lock onto your first hypothesis from Example 1 
   and force-fit it to other examples, rationalizing discrepancies.

2. Imagined verification: You believe you've checked all examples thoroughly 
   when you've only done surface-level pattern matching on Example 1.

3. Pixel-space thinking: You fall back on coordinate transforms and 
   pixel-by-pixel operations instead of object-level reasoning.

4. Over-complexity: You create convoluted multi-case rules when a simpler 
   compositional sequence would work better.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CORE INSTRUCTIONS:

- Observe pixels, but REASON about objects and actions
- Generate multiple diverse hypotheses before committing (not just variations of the same idea)
- State explicit disproof criteria: "This hypothesis dies if Example 2 shows X"
- Before finalizing, actively ask: "Did I narrow prematurely? What else could explain these patterns?"
- Simpler is usually correct: if you can't explain it to a 10-year-old, revisit your logic
- The transformation should make intuitive visual sense, like a game mechanic

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Remember: Your first hypothesis is sticky and excessively convincing to you.
Combat this by generating alternatives and actively seeking evidence against your initial guess.
"""
        )
        
        # LOG PHASE 1 OUTPUT
        phase1_log_path = os.path.join(log_dir, f"{task_id}_phase1_output.txt")
        with open(phase1_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Task ID: {task_id}\n")
            f.write("="*80 + "\n")
            f.write("PHASE 1: PATTERN DISCOVERY OUTPUT\n")
            f.write("="*80 + "\n\n")
            f.write(phase1_output)
        
        if verbose:
            print(f"   ✓ Phase 1 complete ({len(phase1_output)} chars)", flush=True)
            print(f"   📄 Logged to: {phase1_log_path}", flush=True)
        
        # ====================================================================
        # PHASE 2: Code Generation
        # ====================================================================
        if verbose:
            print("\n⚙️  Phase 2: Code Generation...", flush=True)
        
        # Pass the raw similar_programs list - prompter will format it
        phase2_prompt = prompter.build_phase2_prompt(task, 
            phase1_output,
            similar_programs
        )
        
        phase2_output = vlm_client_phase2.query(
            phase2_prompt,
            system_prompt="You are an expert at generating Python code using the given DSL primitives to solve ARC puzzles. You are provided with a natural language description of the pattern to implement, as well as training examples. Generate a Python function `def solve(I):` that implements the described transformation using ONLY the provided DSL primitives. Ensure your code is syntactically correct and follows best practices."
        )
        
        if verbose:
            print(f"   ✓ Phase 2 complete ({len(phase2_output)} chars)", flush=True)
        
        # ====================================================================
        # EXTRACT AND TEST GENERATED CODE
        # ====================================================================
        if verbose:
            print("\n🧪 Testing generated program...", flush=True)
        
        generated_code = extract_code_from_response(phase2_output)
        
        if not generated_code:
            if verbose:
                print("   ✗ Failed to extract code", flush=True)
            
            if best_library_program and best_library_score > 0.5:
                if verbose:
                    print(f"   → Falling back to library (score: {best_library_score:.2f})", flush=True)
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    score=best_library_score,
                    program=best_library_program,
                    phase1_output=phase1_output,
                    error="Code extraction failed, using library fallback"
                )
            
            return TaskResult(
                task_id=task_id,
                success=False,
                score=0.0,
                phase1_output=phase1_output,
                error="Failed to extract code from response"
            )
        
        score, results = test_program(generated_code, task)
        
        if verbose:
            print(f"   Generated score: {score:.2f}", flush=True)
        
        # LOG PHASE 2 OUTPUT WITH TEST RESULTS
        phase2_log_path = os.path.join(log_dir, f"{task_id}_phase2_results.txt")
        with open(phase2_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Task ID: {task_id}\n")
            f.write("="*80 + "\n")
            f.write("PHASE 2: CODE GENERATION & TEST RESULTS\n")
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
        
        if verbose:
            print(f"   📄 Logged to: {phase2_log_path}", flush=True)
        
        # ====================================================================
        # DECIDE FINAL PROGRAM
        # ====================================================================
        success = score == 1.0
        final_program = generated_code
        final_score = score
        
        if not success and best_library_score > score:
            if verbose:
                print(f"   → Library program better ({best_library_score:.2f} > {score:.2f})", flush=True)
            final_program = best_library_program
            final_score = best_library_score
        
        # ====================================================================
        # SAVE TO LIBRARY IF SUCCESSFUL
        # ====================================================================
        if success:
            namespace = globals().copy()
            exec(final_program, namespace)
            if 'solve' in namespace:
                library.add(task_id, final_program)
                if verbose:
                    print(f"   ✓ Added to library", flush=True)
        
        return TaskResult(
            task_id=task_id,
            success=success,
            score=final_score,
            program=final_program,
            phase1_output=phase1_output
        )
        
    except Exception as e:
        if verbose:
            print(f"   ✗ Error: {e}", flush=True)
        
        return TaskResult(
            task_id=task_id,
            success=False,
            score=0.0,
            error=str(e)
        )


def process_directory(
    data_dir: str,
    vlm_client_phase1: BaseVLMClient,
    vlm_client_phase2: BaseVLMClient,
    prompter: VLMPrompter,
    library: ProgramLibrary,
    verbose: bool = True,
    n_workers: int = None,
    timeout: int = 2
) -> List[TaskResult]:
    """
    Process all task files in a directory.
    
    Args:
        data_dir: Directory containing task JSON files
        vlm_client: VLM client for queries
        prompter: Prompt builder
        library: Program library
        verbose: Print progress
        n_workers: Number of parallel workers for library search (None = auto)
        timeout: Timeout per program execution in seconds
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory not found: {data_dir}", flush=True)
        return []
    
    json_files = sorted(data_path.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}", flush=True)
        return []
    
    print(f"\nFound {len(json_files)} tasks in {data_dir}\n", flush=True)
    
    results = []
    successful = 0
    total_score = 0.0
    
    for i, task_file in enumerate(json_files, 1):
        task_id = task_file.stem
        
        try:
            with open(task_file, 'r') as f:
                task = json.load(f)
            
            result = solve_task(
                task=task,
                task_id=task_id,
                vlm_client_phase1=vlm_client_phase1,
                vlm_client_phase2=vlm_client_phase2,
                prompter=prompter,
                library=library,
                verbose=verbose,
                n_workers=n_workers,
                timeout=timeout
            )
            
            results.append(result)
            
            if result.success:
                successful += 1
            
            total_score += result.score
            
            status = "✓" if result.success else "✗"
            print(f"{status} [{i}/{len(json_files)}] {task_id}: {result.score:.2f}", flush=True)
            
        except json.JSONDecodeError as e:
            print(f"✗ [{i}/{len(json_files)}] {task_id}: Invalid JSON - {e}", flush=True)
        except Exception as e:
            print(f"✗ [{i}/{len(json_files)}] {task_id}: {e}", flush=True)
    
    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total tasks: {len(json_files)}", flush=True)
    print(f"Successful: {successful}/{len(json_files)} ({100*successful/len(json_files):.1f}%)", flush=True)
    print(f"Average score: {total_score/len(json_files):.2f}", flush=True)
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
    # print("Initializing components...", flush=True)
    load_dotenv()
    PROVIDER = "openai"
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
        api_base = "https://generativelanguage.googleapis.com/v1beta"
        model = "gemini-2.5-pro"
    elif PROVIDER == "openai":
        api_key = os.getenv('OPENAI_API_KEY')
        api_base = "https://api.openai.com/v1"
        model = "gpt-4o-mini"  # or "gpt-4o-mini" for cheaper option
        
    vlm_config_phase1 = VLMConfig(
        api_key=api_key,
        model=model,
        api_base=api_base,
        max_tokens=16384  # Longer for analysis
    )
    vlm_config_phase2 = VLMConfig(
        api_key=api_key,
        model=model,
        api_base=api_base,
        max_tokens=8192   # Shorter for code gen
    )
    
    vlm_client_phase1 = create_client(PROVIDER, config=vlm_config_phase1)
    # print("VLM client created", flush=True)
    
    vlm_client_phase2 = create_client(PROVIDER, config=vlm_config_phase2)
    prompter = VLMPrompter()
    # print("Prompter created", flush=True)
    
    library = ProgramLibrary()  # Auto-loads from solvers.py
    # print("Loaded library...", flush=True)
    #sanity check
    print(f"Loaded {len(library)} programs from library", flush=True)
    if len(library) > 0:
        print(f"First program: {library.programs[0]['task_id']}", flush=True)
    
    # Configure parallelization
    results = process_directory(
        data_dir='data_v1/eval_size_10',
        vlm_client_phase1=vlm_client_phase1,
        vlm_client_phase2=vlm_client_phase2,
        prompter=prompter,
        library=library,
        verbose=True,
        n_workers=None,  # Auto-detect CPUs (recommended)
        timeout=2        # 2 second timeout per program
    )
    
    save_results(results, output_dir='results/eval_new')


if __name__ == "__main__":
    # print("Starting main...", flush=True)
    sys.stdout.flush()
    main()