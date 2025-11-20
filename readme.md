# ARC Solver - Multi-Provider LLM System with K-Sample Diversity

A complete pipeline for solving ARC (Abstraction and Reasoning Corpus) tasks using a multi-phase LLM approach with DSL-based code generation and diversity sampling.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CONFIGURATION                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  config/config.yaml                                              │  │
│  │  • Provider selection (Grok/Qwen/Gemini)                         │  │
│  │  • Model parameters (max_tokens, retries, etc.)                  │  │
│  │  • Pipeline settings (k_samples, timeout, etc.)                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: Find Similar                           │
│                  Execution-Based Similarity Search                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  • Search library for programs with similar execution patterns   │  │
│  │  • Test programs on training examples                            │  │
│  │  • Calculate grid similarity scores                              │  │
│  │  • Return top-K similar programs (similarity > 0.1)              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                            Perfect Match (1.0)?                         │
│                         ┌──────────┴──────────┐                         │
│                        YES                    NO                        │
│                         │                      │                        │
│                       DONE                     ▼                        │
└────────────────────────────────────────────────┼─────────────────────────┘
                                                 │
┌────────────────────────────────────────────────┼─────────────────────────┐
│                    PHASE 2A: Hypothesis Formation                       │
│               Generate K diverse hypotheses per task                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  For each task, generate K samples (e.g., K=4):                  │  │
│  │  • Analyze training examples sequentially                        │  │
│  │  • Discover transformation patterns                              │  │
│  │  • Output pattern summary in natural language                    │  │
│  │  • Diversity through temperature sampling                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                          Batch all prompts → Parallel API calls         │
└────────────────────────────────────────────────┼─────────────────────────┘
                                                 │
┌────────────────────────────────────────────────┼─────────────────────────┐
│                    PHASE 2B: Hypothesis Validation                      │
│                  Validate & refine hypotheses                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  For each hypothesis from 2A:                                    │  │
│  │  • Check if pattern extends to test input                        │  │
│  │  • Refine hypothesis if needed                                   │  │
│  │  • Output validated pattern description                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                          Batch all prompts → Parallel API calls         │
└────────────────────────────────────────────────┼─────────────────────────┘
                                                 │
┌────────────────────────────────────────────────┼─────────────────────────┐
│                    PHASE 2C: Code Generation                            │
│              Generate executable code from patterns                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  For each validated pattern:                                     │  │
│  │  • Generate Python code using DSL primitives                     │  │
│  │  • Include similar programs as reference                         │  │
│  │  • Output: def solve(I): ...                                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                          Batch all prompts → Parallel API calls         │
└────────────────────────────────────────────────┼─────────────────────────┘
                                                 │
┌────────────────────────────────────────────────┼─────────────────────────┐
│                    BEST-OF-K SELECTION                                  │
│              Test all K programs, select best                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  1. Test all K programs on TRAINING set                          │  │
│  │  2. Select top 2 candidates                                      │  │
│  │  3. Test both on TEST set                                        │  │
│  │  4. Select best performer on test set                            │  │
│  │  5. Compare with library fallback                                │  │
│  │  6. Return final solution (score = 1.0 → add to library)         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📦 Project Structure

```
llms_ftw/
├── config/
│   └── config.yaml              # Main configuration file
├── src/
│   ├── main.py                  # Main pipeline orchestration
│   ├── vlm_client.py            # Multi-provider VLM client
│   ├── vlm_prompter.py          # Prompt builders for each phase
│   └── utils/
│       ├── library.py           # Program library & similarity search
│       ├── dsl.py               # Domain-Specific Language (100+ primitives)
│       └── constants.py         # DSL constants (colors, directions)
├── run_main.slurm              # Basic SLURM submission script
├── run_with_config.slurm       # Advanced SLURM with config selection
├── RUN_INSTRUCTIONS.md         # Detailed execution guide
└── readme.md                   # This file
```

## 🎯 Key Features

### 1. **YAML-Based Configuration**
All hyperparameters in one place - no more hardcoded values!
- Provider selection (Grok, Qwen, Gemini)
- Model settings (tokens, retries, temperature)
- Pipeline parameters (k_samples, timeout, workers)
- Input/output directories

### 2. **Multi-Provider Support**
Unified interface for multiple LLM providers:
- **Grok** (X.AI's API)
- **Qwen** (local vLLM server)
- **Gemini** (Google's API)
- Easy to add more OpenAI-compatible providers

### 3. **K-Sample Diversity**
Generate K diverse solutions per task, select the best:
- Reduces risk of single bad hypothesis
- Temperature-based diversity
- Best-of-K selection on test set
- Configurable K value (default: 4)

### 4. **Simplified Architecture**
Clean, maintainable codebase:
- `OpenAICompatibleClient` for Grok/Qwen/etc.
- `GeminiClient` for Gemini's different API
- Built-in error suppression (no wrapper classes)
- Direct configuration → client flow

### 5. **SLURM Support**
Ready for cluster execution:
- Pre-configured SLURM scripts
- Multiple config file support
- Job monitoring and logging
- Easy customization for your cluster

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install pyyaml requests python-dotenv

# Set API keys in .env file
echo "GROK_API_KEY=your_key_here" > .env
# echo "GEMINI_API_KEY=your_key_here" >> .env  # if using Gemini
```

### 2. Configure Settings

Edit `config/config.yaml`:

```yaml
# Choose provider
provider: "grok"  # or "qwen", "gemini"

# Model configuration
model:
  name: "grok-4-fast"
  api_base: "https://api.x.ai/v1"

# Pipeline settings
process_directory:
  data_dir: "data_v2/evaluation"
  k_samples: 4              # Number of diverse samples per task
  timeout: 2                # Execution timeout (seconds)
  max_find_similar_workers: 56
  log_dir: "logs_baseline"
  verbose: false
  similar: true
  few_shot: true

# Output
output:
  results_dir: "results/baseline"
```

### 3. Run the Solver

#### Local/Interactive:
```bash
python src/main.py
```

#### SLURM Cluster:
```bash
# Basic submission
sbatch run_main.slurm

# With specific config
sbatch run_with_config.slurm baseline
```

See `RUN_INSTRUCTIONS.md` for detailed execution guide.

## 🔧 Core Components

### VLM Client (`vlm_client.py`)

**Unified multi-provider client with two classes:**

```python
# OpenAI-compatible providers (Grok, Qwen, Claude, GPT, etc.)
class OpenAICompatibleClient(BaseVLMClient):
    - Handles standard chat/completions endpoint
    - Conditional API key (supports local servers)
    - Automatic retry with exponential backoff
    - Built-in error suppression

# Google Gemini (different API structure)
class GeminiClient(BaseVLMClient):
    - Custom endpoint format
    - Different payload structure
    - Same retry/error handling
```

**Configuration:**
```python
config = VLMConfig(
    api_key="your_key",
    model="grok-4-fast",
    api_base="https://api.x.ai/v1",
    max_tokens=16384,
    max_retries=3,
    suppress_errors=True  # Return empty string on errors
)

client = create_client("grok", config=config)
response = client.query(prompt, system_prompt)
```

### Pipeline (`main.py`)

**Main orchestration with batched execution:**

1. **Load Config:** Read YAML configuration
2. **Phase 1:** Find similar programs (parallel)
3. **Phase 2A:** Generate K hypotheses (batched, parallel)
4. **Phase 2B:** Validate hypotheses (batched, parallel)
5. **Phase 2C:** Generate code (batched, parallel)
6. **Selection:** Test all K programs, select best

**Key Function:**
```python
def process_directory(
    data_dir: str,
    vlm_client_phase1: BaseVLMClient,
    vlm_client_phase2: BaseVLMClient,
    prompter: VLMPrompter,
    library: ProgramLibrary,
    timeout: int = 2,
    k_samples: int = 1,
    max_find_similar_workers: int = 4,
    log_dir: str = "logs",
    verbose: bool = True,
    similar: bool = True,
    few_shot: bool = True
) -> List[TaskResult]
```

### Program Library (`utils/library.py`)

**Execution-based similarity search:**

```python
library = ProgramLibrary()

# Find similar programs by execution
similar = library.find_similar(
    train_examples=task['train'],
    top_k=5,
    min_similarity=0.1,
    timeout=2
)

# Add successful solutions
if score == 1.0:
    library.add(task_id, program_code)
```

### DSL (`utils/dsl.py`)

**100+ primitives for grid transformations:**

```python
# Transforms
hmirror(grid)          # horizontal flip
vmirror(grid)          # vertical flip
rot90/180/270(grid)    # rotations

# Composition
vconcat(a, b)          # stack vertically
hconcat(a, b)          # stack horizontally

# Objects
objects(grid, T, F, T) # find connected regions
colorfilter(objs, c)   # filter by color

# Functional
compose(f, g)          # f(g(x))
fork(combine, f, g)    # combine(f(x), g(x))
```

## 📊 How K-Sample Diversity Works

### The Problem
Single LLM call can produce incorrect hypothesis → wrong code → failure

### The Solution
Generate K diverse hypotheses, test all, select best:

```
Task → Phase 2A (K samples) → K hypotheses
     → Phase 2B (K samples) → K validated patterns
     → Phase 2C (K samples) → K programs
     → Test all K programs → Select best on test set
```

### Selection Strategy

1. **Training Set:** Test all K programs on training examples
2. **Top 2:** Select two best performers on training set
3. **Test Set:** Test both on test examples
4. **Final:** Select whichever performs better on test set
5. **Fallback:** Compare with library, use library if better

### Example Results
With K=4:
- Sample 0 selected: 40% of tasks
- Sample 1 selected: 25% of tasks
- Sample 2 selected: 20% of tasks
- Sample 3 selected: 15% of tasks

**Key Insight:** Different samples win on different tasks!

## 🎛️ Configuration Reference

### Provider Settings

```yaml
provider: "grok"  # Options: grok, qwen, gemini

model:
  name: "grok-4-fast"
  api_base: "https://api.x.ai/v1"
```

**Provider-specific defaults:**

| Provider | Model | API Base | Auth |
|----------|-------|----------|------|
| Grok | grok-4-fast | https://api.x.ai/v1 | GROK_API_KEY |
| Qwen | Qwen/Qwen2.5-7B-Instruct | http://localhost:8000/v1 | None (local) |
| Gemini | gemini-2.5-pro | https://generativelanguage.googleapis.com/v1beta | GEMINI_API_KEY |

### VLM Config

```yaml
vlm_config:
  phase1:  # Hypothesis & validation
    max_tokens: 16384
    max_retries: 3
    save_prompts: false
    prompt_log_dir: "prompts_old_dsl"

  phase2:  # Code generation
    max_tokens: 8192
    max_retries: 3
```

### Pipeline Parameters

```yaml
process_directory:
  data_dir: "data_v2/evaluation"    # Input task directory
  timeout: 2                        # Execution timeout (seconds)
  k_samples: 4                      # Number of samples per task
  max_find_similar_workers: 56      # Parallel workers for phase 1
  log_dir: "logs_baseline"          # Log output directory
  verbose: false                    # Print detailed progress
  similar: true                     # Use similarity search
  few_shot: true                    # Include similar programs in prompts
```

### Output Settings

```yaml
output:
  results_dir: "results/baseline"
```

## 📈 Performance & Costs

### Batched Execution
All API calls within each phase run in parallel:
- Phase 2A: All tasks × K samples in parallel
- Phase 2B: All tasks × K samples in parallel
- Phase 2C: All tasks × K samples in parallel

### Token Usage (per task with K=4)
- Phase 2A: ~4 calls × 16K tokens = 64K tokens
- Phase 2B: ~4 calls × 16K tokens = 64K tokens
- Phase 2C: ~4 calls × 8K tokens = 32K tokens
- **Total: ~160K tokens per task**

### Approximate Costs (K=4)
With Grok API (~$5/1M input, $15/1M output tokens):
- ~$0.80-$2.40 per task (depending on output length)
- 100 tasks: ~$80-$240

*Note: Costs vary by provider and token usage*

## 🔍 Debugging & Monitoring

### Enable Verbose Mode
```yaml
process_directory:
  verbose: true
```

Shows:
- Phase timing breakdown
- Per-task progress with scores
- Sample selection statistics
- Library matches

### Logs
All outputs saved to configured directories:
- **Pipeline logs:** `<log_dir>/<task_id>_phase2*_*.txt`
- **Selection logs:** `<log_dir>/<task_id>_selection_summary.txt`
- **SLURM logs:** `logs/slurm_<job_id>.out`
- **Results:** `<results_dir>/results.json` and `summary.csv`

### Common Issues

**Config file not found:**
```bash
ls config/config.yaml  # Make sure it exists
```

**API key errors:**
```bash
cat .env  # Check API keys are set
```

**Import errors:**
```bash
# Make sure you're in project root
cd /home/user/llms_ftw
python src/main.py
```

## 🆚 Architecture Evolution

### Before (Old)
```
VLMConfig → create_client() → GrokClient/QwenClient/GeminiClient
                                        ↓
                             ThreadSafeVLMClient (wrapper)
                                        ↓
                                  Catches errors
```
- 3 separate client classes (GrokClient, QwenClient, GeminiClient)
- Wrapper class for error handling
- Hardcoded hyperparameters in main.py

### After (Current)
```
config.yaml → load_config() → VLMConfig → create_client()
                                              ↓
                              OpenAICompatibleClient OR GeminiClient
                              (built-in error suppression)
```
- 2 client classes (merged OpenAI-compatible ones)
- No wrapper needed (built-in suppress_errors)
- All configuration in YAML

**Result:**
- 40+ lines of duplicate code removed
- Cleaner architecture
- Easier to add new providers
- All settings in one place

## 🔮 Future Enhancements

### Planned Features
- [ ] Command-line config file selection: `python main.py --config exp1.yaml`
- [ ] Adaptive K sampling (increase K for harder tasks)
- [ ] Semantic library search with embeddings
- [ ] LLM-guided program repair with error feedback
- [ ] Multi-model ensembles (combine Grok + Gemini)

### Adding New Providers

For OpenAI-compatible APIs (Claude, GPT, etc.):
```yaml
# Just update config.yaml - no code changes needed!
provider: "claude"
model:
  name: "claude-sonnet-4"
  api_base: "https://api.anthropic.com/v1"
```

Then add to environment:
```bash
export CLAUDE_API_KEY=your_key
```

## 📚 References

- **ARC Dataset:** https://github.com/fchollet/ARC
- **DSL Design:** Based on functional programming primitives
- **K-Sample Diversity:** Inspired by best-of-N sampling strategies

## 📄 License

Your project - use as you wish!

---

**Ready to solve ARC tasks with configurable multi-provider LLMs!** 🚀
