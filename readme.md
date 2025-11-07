# ARC Solver - Two-Phase LLM System

## I have no idea if this up-to-date

Complete pipeline for solving ARC (Abstraction and Reasoning Corpus) tasks using a two-phase LLM approach with DSL-based code generation.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 1                                 │
│                    Pattern Discovery                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐     │
│  │   Task   │───▶│   VLM    │───▶│  Pattern Analysis    │     │
│  │ Examples │    │ (Sonnet) │    │  (DSL operations)    │     │
│  └──────────┘    └──────────┘    └──────────────────────┘     │
│                                             │                   │
└─────────────────────────────────────────────┼───────────────────┘
                                              │
                    ┌─────────────────────────┼───────────────┐
                    │  LIBRARY SEARCH         ▼               │
                    │  ┌──────────────────────────────────┐   │
                    │  │  Extract keywords from pattern   │   │
                    │  │  Search solvers.py for similar   │   │
                    │  │  Test library programs           │   │
                    │  └──────────────────────────────────┘   │
                    │             │          │                 │
                    │     Perfect Match?  Top-K Similar       │
                    │             │          │                 │
                    │            YES        NO                 │
                    │             │          │                 │
                    │           DONE         ▼                 │
                    └────────────────────────┼─────────────────┘
                                             │
┌────────────────────────────────────────────┼──────────────────┐
│                         PHASE 2            ▼                   │
│                    Code Generation                             │
│  ┌──────────────┐    ┌──────────┐    ┌────────────────────┐  │
│  │ Pattern +    │───▶│   VLM    │───▶│  Python Code       │  │
│  │ Similar Progs│    │ (Haiku)  │    │  def solve(I): ... │  │
│  └──────────────┘    └──────────┘    └────────────────────┘  │
│                                             │                  │
└─────────────────────────────────────────────┼──────────────────┘
                                              │
                    ┌─────────────────────────┼───────────┐
                    │  TEST & EVALUATE        ▼           │
                    │  ┌──────────────────────────────┐   │
                    │  │  Execute on training examples│   │
                    │  │  Calculate hamming distance  │   │
                    │  │  Compare with library        │   │
                    │  │  Select best solution        │   │
                    │  └──────────────────────────────┘   │
                    │             │                        │
                    │      Perfect Score?                  │
                    │             │                        │
                    │            YES                       │
                    │             │                        │
                    │  ┌──────────▼──────────┐            │
                    │  │  Add to Library     │            │
                    │  └─────────────────────┘            │
                    └──────────────────────────────────────┘
```

## 📦 Components

### Core Files

1. **`main.py`** - Main orchestration pipeline
   - Coordinates Phase 1 and Phase 2
   - Tests programs against training examples
   - Manages library search and fallback logic

2. **`vlm_prompter.py`** - Prompt builder
   - `build_phase1_prompt()` - Pattern discovery prompt
   - `build_phase2_prompt()` - Code generation prompt
   - Includes full DSL reference (~1,500-2,500 tokens)

3. **`vlm_client.py`** - API client
   - Handles Grok API calls
   - Retry logic with exponential backoff
   - Rate limiting support

4. **`library.py`** - Program storage
   - `ProgramLibrary` class for storing solutions
   - Keyword-based similarity search
   - Jaccard similarity scoring

5. **`dsl.py`** - Domain-Specific Language
   - 100+ primitives for grid transformations
   - Functional programming support
   - Object manipulation functions

6. **`constants.py`** - DSL constants
   - Colors (ZERO-NINE)
   - Directions (UP, DOWN, LEFT, RIGHT)
   - Special values (T, F, ORIGIN)

7. **`task_loader.py`** - Task I/O utilities
   - Load tasks from JSON files
   - Convert between list/tuple formats
   - Batch loading from directories

8. **`solvers.py`** - Pre-solved tasks (your file)
   - Collection of `solve_*` functions
   - Automatically loaded into library

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
module load arch/h100
module load python/3.11.5

module load cuda/12.8.0

conda create -n arcn python=3.12 -y
conda activate arcn
pip install "sglang" 

pip install -e .

# to dl eval task
mkdir tasks
cd tasks
curl -L https://github.com/arcprize/ARC-AGI-2/archive/refs/heads/main.tar.gz | tar xz ARC-AGI-2-main/data/evaluation --strip-components=2 -C .


# if bug with sglang this can be usefull:
conda install -c conda-forge libstdcxx-ng --update-deps

# Set API key
export GROK_API_KEY=your_grok_api_key_here
```


### 2. Prepare Your Files

Ensure you have:
```
your_project/
├── main.py
├── vlm_prompter.py
├── vlm_client.py
├── library.py
├── dsl.py
├── constants.py
├── task_loader.py
├── solvers.py          # Your existing solutions
└── tasks/              # Directory with ARC JSON files
    ├── task1.json
    ├── task2.json
    └── ...
```

### 3. Run the Solver

#### Option A: Single Task (Programmatic)

```python
from main import solve_task
from vlm_client import VLMClient
from vlm_prompter import VLMPrompter
from library import ProgramLibrary
import dsl

# Load DSL
with open('dsl.py', 'r') as f:
    dsl_globals = {}
    exec(f.read(), dsl_globals)

# Initialize
client = VLMClient()
prompter = VLMPrompter()
library = ProgramLibrary()

# Your task
task = {
    'train': [
        {
            'input': ((1, 2), (3, 4)),
            'output': ((2, 1), (4, 3))
        }
    ]
}

# Solve
result = solve_task(
    task=task,
    task_id='my_task',
    vlm_client=client,
    prompter=prompter,
    library=library,
    dsl_globals=dsl_globals,
    verbose=True
)

print(f"Success: {result.success}")
print(f"Score: {result.score:.2f}")
print(f"Program:\n{result.program}")
```

#### Option B: Batch Processing

```python
from task_loader import load_tasks_from_directory
from main import solve_task, load_solvers
# ... (same initialization as above)

# Load existing solutions
load_solvers('solvers.py', library, dsl_globals)

# Load all tasks
tasks = load_tasks_from_directory('tasks/')

# Solve each task
results = {}
for task_id, task in tasks.items():
    result = solve_task(
        task=task,
        task_id=task_id,
        vlm_client=client,
        prompter=prompter,
        library=library,
        dsl_globals=dsl_globals,
        verbose=True
    )
    results[task_id] = result

# Summary
solved = sum(1 for r in results.values() if r.success)
print(f"\n✅ Solved {solved}/{len(results)} tasks")
```

## 📊 How It Works

### Phase 1: Pattern Discovery (~1,561 tokens)

**Input:** Training examples (input/output pairs)

**Process:**
1. LLM analyzes each example sequentially
2. Identifies transformations in DSL terms
3. Synthesizes final pattern with operations

**Output:** Structured pattern analysis
```
<final_pattern>
SIZE: (h,w) → (2h,w)
OPS:
x1 = hmirror(I)
O = vconcat(I, x1)
LOGIC: stack horizontally mirrored version below original
CONDITIONS: none
</final_pattern>
```

### Library Search

**Process:**
1. Extract DSL function keywords from Phase 1 output
2. Search `solvers.py` using Jaccard similarity
3. Test top-K similar programs on training examples
4. If perfect match found (score=1.0), return immediately

### Phase 2: Code Generation (~2,497 tokens)

**Input:**
- Phase 1 pattern analysis
- Top-5 similar programs from library

**Process:**
1. LLM generates Python code using DSL primitives
2. Includes functional programming patterns
3. Follows solve(I) function signature

**Output:** Python code
```python
def solve(I):
    # Mirror horizontally
    x1 = hmirror(I)
    
    # Stack vertically
    O = vconcat(I, x1)
    
    return O
```

### Testing & Evaluation

**Process:**
1. Execute generated code on training examples
2. Calculate hamming distance (per-cell comparison)
3. Compute similarity score (1.0 = perfect match)
4. Compare with best library program
5. Select highest-scoring solution

**Scoring:**
- `1.0` = Perfect match (all cells identical)
- `0.8` = 80% of cells match
- `0.0` = Completely different

### Library Update

If score = 1.0:
- Add solution to library
- Available for future similarity searches

## 🔧 Configuration

### VLM Settings

```python
from vlm_client import VLMConfig

config = VLMConfig(
    api_key="your_key",
    model="grok-beta",  # or "claude-sonnet-3.5"
    max_tokens=4096,
    temperature=0.7,  # 0.0 for Phase 1, 0.7 for Phase 2
    max_retries=3
)

client = VLMClient(config)
```

### Library Settings

```python
# Change number of similar programs
similar_programs = library.find_similar(keywords, top_k=3)  # Default: 5
```

## 📈 Performance Optimization

### Early Stopping

The system stops early when:
1. ✅ Library has perfect match (score=1.0)
2. ✅ Generated code scores 1.0
3. ⏭️ Skip Phase 2 if library match is perfect

### Token Efficiency

| Component | Tokens | % of Context |
|-----------|--------|--------------|
| Phase 1   | ~1,561 | 0.8%        |
| Phase 2   | ~2,497 | 1.2%        |
| **Total** | ~4,058 | **2.0%**    |

### Cost Per Task

With Grok API (~$3/1M tokens):
- Phase 1: ~$0.0047
- Phase 2: ~$0.0075
- **Total: ~$0.012 per task**

## 🐛 Debugging

### Enable Verbose Output

```python
result = solve_task(..., verbose=True)
```

Shows:
- Phase 1 completion
- Library search results
- Program test scores
- Final decision logic

### Common Issues

**Issue:** "GROK_API_KEY environment variable not set"
```bash
export GROK_API_KEY=your_key_here
```

**Issue:** "Failed to extract code from response"
- Phase 2 output didn't contain valid Python code
- Check LLM response format
- May need to adjust temperature or prompt

**Issue:** "No similar programs found"
- Library is empty or keywords don't match
- Normal for first few tasks
- Library will grow as you solve more

## 📚 DSL Reference

### Most Common Functions

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
argmax(objs, size)     # largest object

# Functional
compose(f, g)          # f(g(x))
chain(f, g, h)         # f(g(h(x)))
fork(combine, f, g)    # combine(f(x), g(x))
rbind(func, arg)       # partial application
```

Full reference in `dsl.py` (100+ functions)

## 🔮 Future Enhancements

### Phase 2b: Evolution (Not Yet Implemented)

```python
# Planned mutation strategies:
1. Function substitution (hmirror ↔ vmirror)
2. Parameter tweaking (objects(I, T, F, T) → objects(I, T, T, T))
3. Add/remove steps
4. Control flow changes
5. LLM-guided repair with error feedback
```

### Semantic Library Search

Replace keyword matching with embeddings:
```python
from sentence_transformers import SentenceTransformer

library.find_similar_semantic(pattern, top_k=5)
```

### Test-Time Compute

Allocate more retries for hard tasks:
```python
solve_task(..., max_attempts=10, adaptive_budget=True)
```

## 📄 License

Your project - use as you wish!

## 🤝 Contributing

This is your personal ARC solver. Customize as needed!

---

**Ready to solve some ARC tasks!** 🎯