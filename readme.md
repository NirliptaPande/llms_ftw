# ARC Solver - Multi-Phase LLM Pipeline

A pipeline for solving ARC tasks using execution-based similarity search, multi-phase LLM reasoning, and K-sample diversity.

## Architecture

```
Configuration (config.yaml)
    │
    ├─> Phase 1: Find Similar Programs
    │   - Search library by execution similarity
    │   - Test on training examples
    │   - Early exit if perfect match found
    │
    └─> Phase 2: LLM Generation (K samples per task)
        │
        ├─> 2A: Hypothesis Formation
        │   - Analyze training examples
        │   - Identify transformation patterns
        │
        ├─> 2B: Hypothesis Validation
        │   - Check pattern extends to test input
        │   - Refine if needed
        │
        ├─> 2C: Code Generation
        │   - Generate Python code using DSL
        │   - Include similar programs as reference
        │
        └─> Selection: Best-of-K
            - Test all K programs on training set
            - Select top 2 candidates
            - Test both on test set
            - Choose best performer
```

## Key Files

**Main Pipeline**
- `src/main.py` - Pipeline orchestration, batched execution, K-sample selection
- `config/config.yaml` - All hyperparameters (provider, model, k_samples, etc.)

**VLM Client**
- `src/vlm_client.py` - Multi-provider client (Grok, Qwen, Gemini)
  - `OpenAICompatibleClient` - For Grok, Qwen, and other OpenAI-compatible APIs
  - `GeminiClient` - For Google Gemini (different API structure)
  - `VLMConfig` - Configuration dataclass with suppress_errors flag

**Prompting & Search**
- `src/vlm_prompter.py` - Prompt builders for Phase 2A/2B/2C
- `src/utils/library.py` - Program storage and execution-based similarity search
- `src/utils/dsl.py` - Domain-Specific Language (100+ grid transformation primitives)

**Execution**
- `run_main.slurm` - Basic SLURM submission script
- `run_with_config.slurm` - Multi-config support for experiments
- `RUN_INSTRUCTIONS.md` - Detailed execution guide

## Configuration

All settings in `config/config.yaml`:

```yaml
provider: "grok"                    # grok, qwen, or gemini
model:
  name: "grok-4-fast"
  api_base: "https://api.x.ai/v1"

process_directory:
  data_dir: "data_v2/evaluation"
  k_samples: 4                      # Number of diverse samples per task
  timeout: 2                        # Execution timeout (seconds)
  log_dir: "logs_baseline"
  verbose: false

output:
  results_dir: "results/baseline"
```

## Running

**Local:**
```bash
python src/main.py
```

**SLURM:**
```bash
sbatch run_main.slurm
```

See `RUN_INSTRUCTIONS.md` for details.

## Client Architecture

**Before:** Multiple duplicate classes + wrapper layer
```
GrokClient, QwenClient, GeminiClient → ThreadSafeVLMClient (wrapper)
```

**Now:** Unified OpenAI-compatible client + built-in error handling
```
OpenAICompatibleClient (Grok/Qwen/etc.) + GeminiClient
```

Key simplifications:
- Merged duplicate OpenAI-compatible clients (GrokClient, QwenClient) into one
- Removed ThreadSafeVLMClient wrapper (error suppression now in VLMConfig)
- All configuration in YAML instead of hardcoded

## K-Sample Diversity

Problem: Single LLM call can produce wrong hypothesis.

Solution: Generate K diverse solutions, test all, select best:
1. Phase 2A generates K hypotheses (temperature-based diversity)
2. Phase 2B validates each hypothesis
3. Phase 2C generates K programs
4. Test all K on training set, select top 2
5. Test top 2 on test set, pick winner

Different samples win on different tasks - diversity improves overall accuracy.

## Pipeline Flow

1. Load config from YAML
2. Initialize VLM clients (phase1 and phase2 configs)
3. For each task:
   - Phase 1: Search library for similar programs
   - If perfect match found, done
   - Otherwise: Generate K samples through Phase 2A/2B/2C (batched, parallel)
   - Test all K programs, select best
   - Add to library if score = 1.0
4. Save results to JSON and CSV

## Adding Providers

For OpenAI-compatible APIs (no code changes needed):
```yaml
provider: "claude"
model:
  name: "claude-sonnet-4"
  api_base: "https://api.anthropic.com/v1"
```

Then set API key: `export CLAUDE_API_KEY=your_key`
