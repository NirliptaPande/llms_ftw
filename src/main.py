"""
Main pipeline for ARC task solving using two-phase LLM approach.

Phase 1: Pattern Discovery (analyze examples, extract pattern)
Phase 2: Code Generation (generate solve function)
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from vlm_prompter import VLMPrompter
from vlm_client import VLMClient, VLMConfig
from utils.library import ProgramLibrary, extract_functions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))



@dataclass
class TaskResult:
    """Result of attempting to solve a task"""
    task_id: str
    success: bool
    score: float  # 0.0 to 1.0
    program: Optional[str] = None
    phase1_output: Optional[str] = None
    error: Optional[str] = None


def hamming_distance(grid1: Tuple[Tuple[int]], grid2: Tuple[Tuple[int]]) -> float:
    """
    Calculate hamming distance between two grids (fraction of differing cells).
    
    Returns:
        0.0 = identical, 1.0 = completely different
    """
    if grid1 == grid2:
        return 0.0
    
    # Check dimensions
    if len(grid1) != len(grid2):
        return 1.0
    if len(grid1) == 0:
        return 0.0
    if len(grid1[0]) != len(grid2[0]):
        return 1.0
    
    h, w = len(grid1), len(grid1[0])
    total_cells = h * w
    
    if total_cells == 0:
        return 0.0
    
    diff_count = 0
    for i in range(h):
        for j in range(w):
            if grid1[i][j] != grid2[i][j]:
                diff_count += 1
    
    return diff_count / total_cells


def similarity_score(grid1: Tuple[Tuple[int]], grid2: Tuple[Tuple[int]]) -> float:
    """
    Calculate similarity score (inverse of hamming distance).
    
    Returns:
        1.0 = identical, 0.0 = completely different
    """
    return 1.0 - hamming_distance(grid1, grid2)


def test_program(program_code: str, task: Dict, dsl_globals: Dict) -> Tuple[float, List[bool]]:
    """
    Test a program against task training examples.
    
    Args:
        program_code: Python code defining solve(I) function
        task: Task dict with 'train' key
        dsl_globals: Dict with DSL functions (from dsl.py)
    
    Returns:
        (average_score, list of per-example results)
    """
    # Create isolated namespace with DSL
    namespace = dsl_globals.copy()
    
    try:
        # Execute the program to define solve function
        exec(program_code, namespace)
        
        if 'solve' not in namespace:
            return 0.0, []
        
        solve_fn = namespace['solve']
        
        # Test on training examples
        scores = []
        results = []
        
        for example in task['train']:
            inp = example['input']
            expected = example['output']
            
            try:
                actual = solve_fn(inp)
                score = similarity_score(actual, expected)
                scores.append(score)
                results.append(score == 1.0)
            except Exception as e:
                scores.append(0.0)
                results.append(False)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, results
        
    except Exception as e:
        return 0.0, []


def extract_code_from_response(response: str) -> Optional[str]:
    """
    Extract Python code from LLM response.
    Looks for code in ```python blocks or def solve patterns.
    """
    # Try to find ```python code blocks
    python_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
    
    if python_blocks:
        # Return the first block that contains 'def solve'
        for block in python_blocks:
            if 'def solve' in block:
                return block.strip()
        # If no block has 'def solve', return first block
        return python_blocks[0].strip()
    
    # Try to find def solve directly
    match = re.search(r'(def solve\(I\):.*?)(?=\n\ndef|\n\nif __name__|$)', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return None


def load_solvers(solvers_path: str, library: ProgramLibrary, dsl_globals: Dict):
    """
    Load existing solutions from solvers.py into library.
    
    Args:
        solvers_path: Path to solvers.py file
        library: ProgramLibrary instance to populate
        dsl_globals: Dict with DSL functions
    """
    try:
        with open(solvers_path, 'r') as f:
            solvers_code = f.read()
        
        # Execute to get all solve functions
        namespace = dsl_globals.copy()
        exec(solvers_code, namespace)
        
        # Find all solve_* functions
        count = 0
        for name, obj in namespace.items():
            if name.startswith('solve_') and callable(obj):
                task_id = name.replace('solve_', '')
                
                # Extract just this function's code
                pattern = rf'def {name}\(I\):.*?(?=\ndef |\Z)'
                match = re.search(pattern, solvers_code, re.DOTALL)
                
                if match:
                    func_code = match.group(0).strip()
                    # Normalize to 'def solve(I):'
                    func_code = func_code.replace(f'def {name}(', 'def solve(')
                    
                    # Create a simple pattern description for keywords
                    pattern_desc = func_code
                    
                    library.add(task_id, pattern_desc, func_code)
                    count += 1
        
        print(f" Loaded {count} solutions from {solvers_path}")
        
    except FileNotFoundError:
        print(f"  Solvers file not found: {solvers_path}")
    except Exception as e:
        print(f"  Error loading solvers: {e}")


def solve_task(
    task: Dict,
    task_id: str,
    vlm_client: VLMClient,
    prompter: VLMPrompter,
    library: ProgramLibrary,
    dsl_globals: Dict,
    verbose: bool = True
) -> TaskResult:
    """
    Solve a single ARC task using two-phase approach.
    
    Args:
        task: Task dict with 'train' and optionally 'test' keys
        task_id: Unique identifier for this task
        vlm_client: VLM client for API calls
        prompter: VLMPrompter instance
        library: ProgramLibrary with existing solutions
        dsl_globals: Dict with DSL functions
        verbose: Print progress
    
    Returns:
        TaskResult with solution and metrics
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Solving Task: {task_id}")
        print(f"{'='*80}")
    
    try:
        # ====================================================================
        # PHASE 1: Pattern Discovery
        # ====================================================================
        if verbose:
            print("\n Phase 1: Pattern Discovery...")
        
        phase1_prompt = prompter.build_phase1_prompt(task)
        
        phase1_output = vlm_client.query(
            phase1_prompt,
            system_prompt="You are an expert at analyzing ARC puzzles and discovering transformation patterns."
        )
        
        if verbose:
            print(f" Phase 1 complete ({len(phase1_output)} chars)")
            print(f"Pattern excerpt: {phase1_output[:200]}...")
        
        # ====================================================================
        # LIBRARY SEARCH: Find similar programs
        # ====================================================================
        if verbose:
            print("\n📚 Searching library for similar programs...")
        
        keywords = extract_functions(phase1_output)
        similar_programs = library.find_similar(keywords, top_k=5)
        
        if verbose:
            if similar_programs:
                print(f" Found {len(similar_programs)} similar programs:")
                for i, prog in enumerate(similar_programs[:3], 1):
                    sim = prog['similarity']
                    shared = prog['shared_functions']
                    print(f"   {i}. Similarity: {sim:.2f}, Shared: {shared}")
            else:
                print("   No similar programs found")
        
        # ====================================================================
        # TEST LIBRARY PROGRAMS: Try existing solutions first
        # ====================================================================
        if similar_programs and verbose:
            print("\n Testing library programs...")
        
        best_library_score = 0.0
        best_library_program = None
        
        for prog_entry in similar_programs:
            prog = prog_entry['program']
            code = prog['code']
            
            score, results = test_program(code, task, dsl_globals)
            
            if verbose:
                print(f"   Task {prog['task_id']}: score={score:.2f}")
            
            if score > best_library_score:
                best_library_score = score
                best_library_program = code
            
            # Early stop if perfect match
            if score == 1.0:
                if verbose:
                    print(f"   Perfect match found! Returning library solution.")
                
                return TaskResult(
                    task_id=task_id,
                    success=True,
                    score=1.0,
                    program=code,
                    phase1_output=phase1_output
                )
        
        if best_library_score > 0:
            if verbose:
                print(f"   Best library score: {best_library_score:.2f}")
        
        # ====================================================================
        # PHASE 2: Code Generation
        # ====================================================================
        if verbose:
            print("\n Phase 2: Code Generation...")
        
        # Format similar programs for prompt
        similar_for_prompt = []
        for prog_entry in similar_programs:
            similar_for_prompt.append({
                'program': prog_entry['program']['code'],
                'similarity': prog_entry['similarity'],
                'functions': list(prog_entry['shared_functions'])
            })
        
        phase2_prompt = prompter.build_phase2_prompt(
            phase1_output,
            similar_for_prompt
        )
        
        phase2_output = vlm_client.query(
            phase2_prompt,
            system_prompt="You are an expert at generating Python code using DSL primitives to solve ARC puzzles."
        )
        
        if verbose:
            print(f"Phase 2 complete ({len(phase2_output)} chars)")
        
        # ====================================================================
        # EXTRACT AND TEST GENERATED CODE
        # ====================================================================
        if verbose:
            print("\n Testing generated program...")
        
        generated_code = extract_code_from_response(phase2_output)
        
        if not generated_code:
            if verbose:
                print("Failed to extract code from response")
            
            # Fall back to best library program if available
            if best_library_program and best_library_score > 0.5:
                if verbose:
                    print(f"   Falling back to library program (score: {best_library_score:.2f})")
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
        
        # Test the generated code
        score, results = test_program(generated_code, task, dsl_globals)
        
        if verbose:
            print(f"   Generated program score: {score:.2f}")
            print(f"   Per-example results: {results}")
        
        # ====================================================================
        # DECIDE FINAL PROGRAM
        # ====================================================================
        success = score == 1.0
        final_program = generated_code
        final_score = score
        
        # If generated program isn't perfect, check if library was better
        if not success and best_library_score > score:
            if verbose:
                print(f"Library program better ({best_library_score:.2f} > {score:.2f})")
            final_program = best_library_program
            final_score = best_library_score
        
        # ====================================================================
        # SAVE TO LIBRARY IF SUCCESSFUL
        # ====================================================================
        if success:
            library.add(task_id, phase1_output, final_program)
            if verbose:
                print(f"Added to library")
            
            # Update cache file (optional - for persistence)
            try:
                library.save('library_cache.json')
            except:
                pass  # Don't fail if can't save cache
        
        return TaskResult(
            task_id=task_id,
            success=success,
            score=final_score,
            program=final_program,
            phase1_output=phase1_output
        )
        
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        
        return TaskResult(
            task_id=task_id,
            success=False,
            score=0.0,
            error=str(e)
        )


def main():
    """Main entry point"""
    import sys
    import os
    
    # Load DSL
    print("Loading DSL...")
    dsl_globals = {}
    try:
        with open('./src/utils/dsl.py', 'r') as f:
            dsl_code = f.read()
        exec(dsl_code, dsl_globals)
        print(f"Loaded DSL with {len([k for k in dsl_globals.keys() if not k.startswith('_')])} functions")
    except Exception as e:
        print(f"Failed to load DSL: {e}")
        return
    
    # Initialize components
    print("\nInitializing components...")
    vlm_client = VLMClient()
    prompter = VLMPrompter()
    library = ProgramLibrary()
    
    # Try to load library from cache first (much faster!)
    library_cache = 'library_cache.json'
    if os.path.exists(library_cache):
        print(f"Loading library from cache: {library_cache}")
        library.load(library_cache)
        print(f"Loaded {len(library)} programs from cache")
    
    # If cache doesn't exist or is empty, load from solvers.py
    if len(library) == 0 and os.path.exists('../utils/solvers.py'):
        print("Building library from solvers.py (first time)...")
        load_solvers('../utils/solvers.py', library, dsl_globals)
        # Save to cache for next time
        library.save(library_cache)
        print(f"Saved library cache to {library_cache}")
    
    # Example task (replace with actual task loading)
    task = {
        'train': [
            {
                'input': ((1, 2), (3, 4)),
                'output': ((4, 3), (2, 1))
            },
            {
                'input': ((5, 6, 7), (8, 9, 0)),
                'output': ((0, 9, 8), (7, 6, 5))
            }
        ]
    }
    
    # Solve task
    result = solve_task(
        task=task,
        task_id='example_hmirror',
        vlm_client=vlm_client,
        prompter=prompter,
        library=library,
        dsl_globals=dsl_globals,
        verbose=True
    )
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Task ID: {result.task_id}")
    print(f"Success: {result.success}")
    print(f"Score: {result.score:.2f}")
    
    if result.program:
        print(f"\nGenerated Program:")
        print(result.program)
    
    if result.error:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    main()