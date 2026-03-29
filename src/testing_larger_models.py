"""
Testing Larger Models for Phase 2c Code Generation

This script tests different (larger) models on Phase 2c code generation by:
1. Loading existing Phase 2b validation outputs from a log directory
2. Rerunning Phase 1 to find similar programs
3. Generating K Phase 2c samples using a new model (in parallel)
4. Testing and selecting the best program per task
5. Comparing results against baseline
"""

# ============================================================================
# 1. Setup and Imports
# ============================================================================

import sys
import os
import json
import yaml
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import from main.py
from main import (
    test_program,
    extract_code_from_response,
    extract_validated_pattern_from_response,
    phase1_find_similar,
    select_best_programs,
    sort_examples_by_size
)

# Import VLM components
from vlm_prompter import VLMPrompter
from vlm_client import VLMConfig, create_client

# Import utils
from utils.library import ProgramLibrary, calculate_grid_similarity
from utils.dsl import *
from utils.constants import *

# Load environment variables
load_dotenv()

# Load configuration
config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("✓ Configuration loaded")


# ============================================================================
# 2. Configuration - MODIFY THESE AS NEEDED
# ============================================================================

# Input: Directory with existing Phase 2b validation logs
INPUT_LOG_DIR = "../logs/grok4.1fast_grokcodefast1_reasoning_dsl_k4_80703"

# Output: New log directory for larger model results
OUTPUT_LOG_DIR = "../logs/gpt5.2_80703_phase2c_nodsl"

# Data directory with task JSON files
DATA_DIR = "../data_v2/evaluation"

# Model configuration for Phase 2c
TEST_MODEL = "openai/gpt-5.2"
TEST_MODEL_API_BASE = "https://openrouter.ai/api/v1"
TEST_MODEL_MAX_TOKENS = 32768

# Processing parameters
K_SAMPLES = 4
MAX_API_CALLS = 400
TIMEOUT = 2
DSL_ENABLED = False
FEW_SHOT = False
SIMILAR = False

# Verbose output
VERBOSE = True

print(f"Input logs: {INPUT_LOG_DIR}")
print(f"Output logs: {OUTPUT_LOG_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Test model: {TEST_MODEL}")
print(f"K samples: {K_SAMPLES}")

# Create output directory
Path(OUTPUT_LOG_DIR).mkdir(parents=True, exist_ok=True)
print(f"✓ Output directory created: {OUTPUT_LOG_DIR}")


# ============================================================================
# 3. Load Phase 2b Validation Files
# ============================================================================

def load_phase2b_validations(log_dir: str) -> Dict[str, Dict[int, str]]:
    """
    Load all Phase 2b validation files from log directory.
    
    Returns:
        Dictionary mapping {task_id: {sample_k: validated_pattern}}
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    # Find all phase2b validation files
    pattern = "*_sample*_phase2b_validation.txt"
    validation_files = list(log_path.glob(pattern))
    
    if not validation_files:
        raise FileNotFoundError(f"No phase2b validation files found in {log_dir}")
    
    print(f"Found {len(validation_files)} phase2b validation files")
    
    # Parse and organize by task_id and sample
    validations = defaultdict(dict)
    
    for file_path in validation_files:
        # Parse filename: {task_id}_sample{k}_phase2b_validation.txt
        filename = file_path.name
        match = re.match(r'([a-f0-9]+)_sample(\d+)_phase2b_validation\.txt', filename)
        
        if not match:
            print(f"Warning: Could not parse filename: {filename}")
            continue
        
        task_id = match.group(1)
        sample_k = int(match.group(2))
        
        # Read file and extract validated pattern
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            validated_pattern = extract_validated_pattern_from_response(content)
            validations[task_id][sample_k] = validated_pattern
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    print(f"Loaded validations for {len(validations)} unique tasks")
    
    # Print sample distribution
    samples_per_task = [len(samples) for samples in validations.values()]
    if samples_per_task:
        print(f"Samples per task: min={min(samples_per_task)}, max={max(samples_per_task)}, avg={sum(samples_per_task)/len(samples_per_task):.1f}")
    
    return dict(validations)


def load_tasks(task_ids: List[str], data_dir: str) -> Dict[str, Dict]:
    """Load task JSON files for given task IDs."""
    data_path = Path(data_dir)
    tasks = {}
    
    for task_id in task_ids:
        task_file = data_path / f"{task_id}.json"
        
        if not task_file.exists():
            print(f"Warning: Task file not found: {task_file}")
            continue
        
        try:
            with open(task_file, 'r') as f:
                task = json.load(f)
            
            # Sort examples by size (as in main.py)
            task = sort_examples_by_size(task)
            tasks[task_id] = task
            
        except Exception as e:
            print(f"Error loading {task_id}: {e}")
    
    return tasks


# Load validations
validations = load_phase2b_validations(INPUT_LOG_DIR)
task_ids = sorted(validations.keys())

print(f"\n✓ Loaded validations for {len(task_ids)} tasks")
print(f"Task IDs: {task_ids[:5]}..." if len(task_ids) > 5 else f"Task IDs: {task_ids}")

# Load task JSON files
tasks = load_tasks(task_ids, DATA_DIR)
print(f"✓ Loaded {len(tasks)} task files")


# ============================================================================
# 4. Run Phase 1: Find Similar Programs
# ============================================================================

# Initialize library
library = ProgramLibrary()

print("Running Phase 1: Finding similar programs...")
time_start = time.time()

phase1_results = {}

for task_id in task_ids:
    if task_id not in tasks:
        continue
    
    task = tasks[task_id]
    
    result = phase1_find_similar(
        task=task,
        task_id=task_id,
        library=library,
        timeout=TIMEOUT,
        verbose=False,  # Set to True for detailed output
        similar=SIMILAR
    )
    
    phase1_results[task_id] = result
    
    if VERBOSE and result.similar_programs:
        print(f"  {task_id}: Found {len(result.similar_programs)} similar programs (best: {result.best_library_score:.2f})")

time_phase1 = time.time()
print(f"\n✓ Phase 1 complete: {time_phase1 - time_start:.1f}s")


# ============================================================================
# 5. Setup VLM Client for Larger Model
# ============================================================================

# Get provider and API key from config/env
PROVIDER = config['provider']

if PROVIDER == "grok":
    api_key = os.getenv('OPENROUTER_API_KEY')
elif PROVIDER == "qwen":
    api_key = None
elif PROVIDER == "gemini":
    api_key = os.getenv('GEMINI_API_KEY')
else:
    api_key = None

# Create VLM config for the test model
vlm_config_phase2c = VLMConfig(
    api_key=api_key,
    model=TEST_MODEL,
    api_base=TEST_MODEL_API_BASE,
    max_tokens=TEST_MODEL_MAX_TOKENS,
    max_retries=3,
    extra_params=config['vlm_config']['phase2'].get('extra_params'),
    suppress_errors=True
)

# Create client
vlm_client = create_client(PROVIDER, config=vlm_config_phase2c)
print(f"✓ VLM client created: {TEST_MODEL}")

# Create prompter
prompter = VLMPrompter()
print("✓ Prompter initialized")


# ============================================================================
# 6. Build and Execute Phase 2c Prompts (Batched)
# ============================================================================

print("Building Phase 2c prompts...")

phase2c_prompts = []
phase2c_task_sample_pairs = []  # List of (task_id, sample_k)

# Build all prompts
for task_id in task_ids:
    if task_id not in tasks or task_id not in validations:
        continue
    
    task = tasks[task_id]
    phase1_result = phase1_results.get(task_id)
    task_validations = validations[task_id]
    
    # For each sample that has a validation
    for sample_k in sorted(task_validations.keys()):
        validated_pattern = task_validations[sample_k]
        
        # Get similar programs if available
        similar_progs = phase1_result.similar_programs if (DSL_ENABLED and phase1_result) else None
        
        # Build prompt
        prompt = prompter.build_phase2c_prompt(
            task=task,
            validated_pattern=validated_pattern,
            similar_programs=similar_progs,
            few_shot=FEW_SHOT,
            dsl_enabled=DSL_ENABLED
        )
        
        phase2c_prompts.append(prompt)
        phase2c_task_sample_pairs.append((task_id, sample_k))

print(f"Built {len(phase2c_prompts)} Phase 2c prompts")
print(f"Sending to API ({TEST_MODEL})...")

time_start_phase2c = time.time()

# Send all prompts in parallel
if DSL_ENABLED:
    system_prompt = """You are an expert at implementing ARC puzzle solutions using a domain-specific language (DSL). You will receive a validated transformation pattern and similar example programs. Generate a precise `solve(I)` function using the DSL primitives provided."""
else:
    system_prompt = """You are an expert at implementing ARC puzzle solutions in pure Python. You will receive a validated transformation pattern. Generate a precise `solve(I)` function that implements this pattern."""

phase2c_outputs = []
if phase2c_prompts:
    with ThreadPoolExecutor(max_workers=min(MAX_API_CALLS, len(phase2c_prompts))) as executor:
        futures = [executor.submit(vlm_client.query, p, system_prompt) for p in phase2c_prompts]
        phase2c_outputs = [f.result() for f in futures]

time_phase2c = time.time()
print(f"✓ Phase 2c complete: {time_phase2c - time_start_phase2c:.1f}s")
print(f"Received {len(phase2c_outputs)} responses")

# Organize outputs by task_id
phase2c_results = defaultdict(dict)

for (task_id, sample_k), output in zip(phase2c_task_sample_pairs, phase2c_outputs):
    # Extract code from response
    code = extract_code_from_response(output)
    phase2c_results[task_id][sample_k] = code

print(f"✓ Organized results for {len(phase2c_results)} tasks")

# Quick check: how many codes were successfully extracted?
total_codes = sum(len(samples) for samples in phase2c_results.values())
extracted_codes = sum(1 for samples in phase2c_results.values() for code in samples.values() if code)
print(f"Successfully extracted code: {extracted_codes}/{total_codes}")


# ============================================================================
# 7. Test Programs and Select Best
# ============================================================================

print("Testing programs and selecting best...")

results = []
successful = 0
total_score = 0.0
sample_selection_counts = defaultdict(int)

time_start_testing = time.time()

for task_id in sorted(phase2c_results.keys()):
    if task_id not in tasks:
        continue
    
    task = tasks[task_id]
    task_codes = phase2c_results[task_id]
    
    # Prepare inputs for select_best_programs
    sample_indices = sorted(task_codes.keys())
    candidate_programs = [task_codes[k] for k in sample_indices]
    
    # Get hypotheses and validations
    hypotheses = [validations[task_id].get(k, "") for k in sample_indices]
    task_validations_list = [validations[task_id].get(k, "") for k in sample_indices]
    
    # Select best programs
    result, metadata = select_best_programs(
        candidate_programs=candidate_programs,
        task=task,
        task_id=task_id,
        hypotheses=hypotheses,
        validations=task_validations_list,
        sample_indices=sample_indices,
        dsl_enabled=DSL_ENABLED,
        library=library,
        log_dir=OUTPUT_LOG_DIR,
        program_repair_enabled=False
    )
    
    results.append(result)
    
    # Track statistics
    if result.success:
        successful += 1
    total_score += result.score
    
    # Track which sample was selected
    if metadata and metadata.get('best_sample_idx') is not None:
        sample_selection_counts[metadata['best_sample_idx']] += 1
    
    # Print progress
    status = "✓" if result.success else "✗"
    if VERBOSE:
        print(f"  {status} {task_id}: score={result.score:.2f}")

time_testing = time.time()

print(f"\n✓ Testing complete: {time_testing - time_start_testing:.1f}s")


# ============================================================================
# 8. Results Summary
# ============================================================================

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print(f"\nModel: {TEST_MODEL}")
print(f"Total tasks: {len(results)}")
print(f"Successful (score=1.0): {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
print(f"Average test score: {total_score/len(results):.3f}")

print("\n" + "="*80)
print("SAMPLE SELECTION STATISTICS")
print("="*80)

# Get max sample index seen
max_sample = max(sample_selection_counts.keys()) if sample_selection_counts else 0

for k in range(max_sample + 1):
    count = sample_selection_counts[k]
    pct = 100 * count / len(results) if len(results) > 0 else 0
    print(f"Sample {k} selected: {count} times ({pct:.1f}%)")

print("\n" + "="*80)
print("TIME BREAKDOWN")
print("="*80)
print(f"Phase 1 (find_similar): {time_phase1 - time_start:.1f}s")
print(f"Phase 2c (code gen × {len(phase2c_prompts)}): {time_phase2c - time_start_phase2c:.1f}s")
print(f"Testing & selection: {time_testing - time_start_testing:.1f}s")
print(f"Total: {time_testing - time_start:.1f}s")

print("\n" + "="*80)


# ============================================================================
# 9. Detailed Results Table
# ============================================================================

# Create results DataFrame
results_data = []
for result in results:
    results_data.append({
        'task_id': result.task_id,
        'success': result.success,
        'test_score': result.score,
        'error': result.error if result.error else ""
    })

df = pd.DataFrame(results_data)

# Display sorted by score (descending)
print("\nResults sorted by test score:")
print(df.sort_values('test_score', ascending=False).to_string(index=False))

# Show successful tasks
successful_tasks = df[df['success'] == True]['task_id'].tolist()
print(f"\n✓ Successful tasks ({len(successful_tasks)}):")
for task_id in successful_tasks:
    print(f"  - {task_id}")

# Show failed tasks
failed_tasks = df[df['success'] == False]['task_id'].tolist()
print(f"\n✗ Failed tasks ({len(failed_tasks)}):")
for task_id in failed_tasks[:20]:  # Show first 20
    score = df[df['task_id'] == task_id]['test_score'].values[0]
    print(f"  - {task_id} (score: {score:.2f})")
if len(failed_tasks) > 20:
    print(f"  ... and {len(failed_tasks) - 20} more")


# ============================================================================
# 10. Save Results
# ============================================================================

# Save results to JSON
results_dir = Path(OUTPUT_LOG_DIR).parent / "results" / Path(OUTPUT_LOG_DIR).name
results_dir.mkdir(parents=True, exist_ok=True)

json_file = results_dir / 'results.json'
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

print(f"✓ Saved results to {json_file}")

# Save summary CSV
csv_file = results_dir / 'summary.csv'
df.to_csv(csv_file, index=False)
print(f"✓ Saved summary to {csv_file}")

# Save configuration
config_file = results_dir / 'config.txt'
with open(config_file, 'w') as f:
    f.write(f"Model: {TEST_MODEL}\n")
    f.write(f"API Base: {TEST_MODEL_API_BASE}\n")
    f.write(f"Max Tokens: {TEST_MODEL_MAX_TOKENS}\n")
    f.write(f"K Samples: {K_SAMPLES}\n")
    f.write(f"DSL Enabled: {DSL_ENABLED}\n")
    f.write(f"Few Shot: {FEW_SHOT}\n")
    f.write(f"Input Log Dir: {INPUT_LOG_DIR}\n")
    f.write(f"Output Log Dir: {OUTPUT_LOG_DIR}\n")
    f.write(f"\nSuccess Rate: {100*successful/len(results):.1f}%\n")
    f.write(f"Average Score: {total_score/len(results):.3f}\n")

print(f"✓ Saved configuration to {config_file}")

print("\n" + "="*80)
print("DONE!")
print("="*80)
