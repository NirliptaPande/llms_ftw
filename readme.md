# HINT: Task Demonstrations for Hierarchical Inference in Abstract Reasoning

> **COLM 2026** | [Paper](#) | [Dataset](https://arcprize.org)

HINT is a multi-phase inference framework for ARC-AGI. Recent inductive approaches treat each task in isolation and struggle to acquire new skills efficiently. HINT addresses this through principled reuse of prior experience and hierarchical reasoning, without any model fine-tuning or additional training data. 

Best performance of **39.2% accuracy at $0.83/task** with Gemini 3, comparable performance of **37.5% at $0.63/task** with a hybrid approach (Gemini 3 Flash for hypothesis + Grok 4.1-Fast for synthesis) and represenatative performance **20.8% accuracy at $0.28/task** with Grok 4.1-Fast on ARC-AGI-2 public evaluation, compared to 5% for direct synthesis with the same backbone model.

---

## How it works

HINT structures inference as a three-phase hierarchy:

1. **Task Demonstration** — retrieves functionally similar programs from a library of 934 previously solved tasks by executing them on the current task's training examples.
2. **Hypothesis Search** — uses the retrieved demonstrations to ground natural-language hypothesis formation and refinement about the underlying transformation.
3. **Program Search** — synthesises and repairs a `solve(I)` function conditioned on the evolved hypothesis and similar programs.

## Setup

```bash
pip install -r requirements.txt
```

Add your API keys to a `.env` file in the project root:
```
GROK_API_KEY=your_key
GEMINI_API_KEY=your_key
OPENROUTER_API_KEY=your_key
```

Edit `config/config.yaml` to set your provider, model, and number of parallel samples `k`.

Run HINT:
```bash
python src/main.py --exp-name my_run
```

Run the direct synthesis baseline:
```bash
python src/direct_testing.py --config config/direct_test_config.yaml
```

Results are saved to `results/<exp_name>/` and logs to `logs/<exp_name>/`. Runs are tracked via Weights & Biases under the `arc-solver_icml` project.

## Citation

```bibtex
@inproceedings{hint2026,
  title     = {HINT: Task Demonstrations for Hierarchical Inference in Abstract Reasoning},
  author    = {Anonymous},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026},
  note      = {Under review}
}
```