"""
Refactored VLM Prompter for ARC DSL-based solver.
New pipeline: Program Similarity → Pattern Discovery → Code Generation
"""

from typing import List, Tuple, Dict, Any


class VLMPrompter:
    """Builds prompts for the program-first ARC solving process"""
    
    def __init__(self):
        self.phase1_template = self._load_phase1_template()
        self.phase2_template = self._load_phase2_template()
    
    def build_phase1_prompt(self, 
                           task: Dict[str, Any],
                           similar_programs: List[Dict[str, Any]] = None) -> str:
        """
        Build Phase 1 prompt: Natural Language Pattern Discovery
        Analyzes training examples AND similar programs to discover transformation pattern.
        
        Args:
            task: Dict with 'train' key containing list of {'input': grid, 'output': grid}
            similar_programs: List of similar programs from library (found via execution)
                             Each dict has: {'program': str, 'similarity': float, 'task_id': str}
            
        Returns:
            Complete prompt string for natural language pattern analysis
        """
        # Format training examples
        examples_str = self._format_training_examples(task['train'])
        
        # Format similar programs
        if similar_programs:
            similar_str = self._format_similar_programs_for_phase1(similar_programs)
        else:
            similar_str = "[No similar programs found in library]"
        
        # Insert into template
        prompt = self.phase1_template.replace('[INSERT TASK EXAMPLES HERE]', examples_str)
        prompt = prompt.replace('[INSERT SIMILAR PROGRAMS HERE]', similar_str)
        
        return prompt
    
    def build_phase2_prompt(self,
                            task: Dict[str, Any], 
                           phase1_output: str,
                           similar_programs: List[Dict[str, Any]] = None) -> str:
        """
        Build Phase 2 prompt: Code Generation
        Generates solve() function based on natural language pattern and similar programs.
        
        Args:
            phase1_output: The natural language pattern description from Phase 1
            similar_programs: Same list of similar programs (for reference during coding)
            
        Returns:
            Complete prompt string for code generation
        """
        # Format training examples
        examples_str = self._format_training_examples(task['train'])
        prompt = self.phase2_template.replace('[INSERT TASK EXAMPLES HERE]', examples_str)
        # Extract the key sections from Phase 1 output
        extracted_pattern = self._extract_key_pattern_from_phase1(phase1_output)
        
        # Insert extracted pattern
        prompt = self.phase2_template.replace('[PHASE 1 PATTERN DESCRIPTION]', extracted_pattern)
        
        # Insert similar programs if available
        if similar_programs:
            similar_str = self._format_similar_programs_for_phase2(similar_programs)
        else:
            similar_str = "[No similar programs available for reference]"
        
        prompt = prompt.replace('[SIMILAR PROGRAMS FOR REFERENCE]', similar_str)
        
        return prompt
    
    def _extract_key_pattern_from_phase1(self, phase1_output: str) -> str:
        """
        Extract the most relevant sections from Phase 1 for Phase 2.
        Focuses on the final transformation rule and decision.
        """
        sections = []
        
        # Try to extract STEP 6: Final Transformation Rule
        if "### STEP 6: Final Transformation Rule" in phase1_output:
            step6_start = phase1_output.find("### STEP 6: Final Transformation Rule")
            step6_end = phase1_output.find("━━━━━━━━━━", step6_start + 1)
            if step6_end == -1:
                step6_end = len(phase1_output)
            sections.append(phase1_output[step6_start:step6_end].strip())
        
        # Try to extract STEP 5: Lock-in Checkpoint (the decision)
        if "**Decision:**" in phase1_output:
            decision_start = phase1_output.find("**Decision:**")
            decision_end = phase1_output.find("\n\n", decision_start)
            if decision_end == -1:
                decision_end = phase1_output.find("━━━━━━━━━━", decision_start)
            if decision_end != -1:
                sections.append(phase1_output[decision_start:decision_end].strip())
        
        # If we got specific sections, use them
        if sections:
            return "\n\n".join(sections)
        
        # Fallback: use the entire output
        return phase1_output
    
    def _format_training_examples(self, train_examples: List[Dict[str, Any]]) -> str:
        """Format training examples for display"""
        lines = []
        
        for idx, example in enumerate(train_examples, 1):
            lines.append(f"Example {idx}:")
            lines.append("Input:")
            lines.append(self._format_grid(example['input']))
            lines.append("Output:")
            lines.append(self._format_grid(example['output']))
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_grid(self, grid: Tuple[Tuple[int]]) -> str:
        """Format grid as readable ASCII"""
        return "\n".join("".join(str(cell) for cell in row) for row in grid)
    
    def _format_similar_programs_for_phase1(self, similar_programs: List[Dict[str, Any]]) -> str:
        """Format similar programs for Phase 1 (natural language discovery)"""
        lines = []
        lines.append("The following programs from the library solved similar tasks:\n")
        
        for idx, prog in enumerate(similar_programs, 1):
            similarity = prog.get('similarity', 0.0)
            program_code = prog.get('program', '')
            task_id = prog.get('task_id', 'unknown')
            
            lines.append(f"Similar Program {idx} (similarity: {similarity:.2f}, task: {task_id}):")
            lines.append("```python")
            lines.append(program_code)
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_similar_programs_for_phase2(self, similar_programs: List[Dict[str, Any]]) -> str:
        """Format similar programs for Phase 2 (code generation reference)"""
        lines = []
        
        for idx, prog in enumerate(similar_programs, 1):
            similarity = prog.get('similarity', 0.0)
            program_code = prog.get('program', '')
            task_id = prog.get('task_id', 'unknown')
            
            lines.append(f"```python")
            lines.append(f"# Similar program {idx} (similarity: {similarity:.2f}, task: {task_id})")
            lines.append(program_code)
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)
    
    def _load_phase1_template(self) -> str:
        return r"""## Training Examples
[INSERT TASK EXAMPLES HERE]

## Similar Programs from Library
[INSERT SIMILAR PROGRAMS HERE]

## Analysis Protocol

You will analyze these examples systematically, allowing your hypothesis to evolve 
naturally as you see more data - like a human solving a puzzle.

**Core principle:** Look for DIFFERENCES within each example (input→output changes) 
and SIMILARITIES across all examples (the consistent pattern).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### STEP 1: First Observations & Comparison

For each example, describe what you see (objects, not pixels):

**Example 1:**
- Input: [objects, colors, spatial layout]
- Output: [objects, colors, spatial layout]
- What CHANGES: [list differences]
- What STAYS SAME: [list constants]

**Example 2:**
- Input: [objects, colors, spatial layout]
- Output: [objects, colors, spatial layout]
- What CHANGES: [list differences]
- What STAYS SAME: [list constants]

**Example 3:**
- Input: [objects, colors, spatial layout]
- Output: [objects, colors, spatial layout]
- What CHANGES: [list differences]
- What STAYS SAME: [list constants]

**Comparison Matrix** (find patterns):

| Feature | Ex1 | Ex2 | Ex3 | Pattern? |
|---------|-----|-----|-----|----------|
| # input objects | | | | [Varies / Always X / ...] |
| # output objects | | | | [Pattern description] |
| Grid size changes? | | | | [Y/N and how] |
| Colors preserved? | | | | [Pattern description] |
| Spatial relationship | | | | [Pattern description] |
| [Add custom rows] | | | | |

**Key similarities across ALL examples:**
[What is CONSISTENT in all transformations?]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### STEP 2: Hypothesis Evolution Through Examples

**From Example 1:**
Initial hypothesis: [Describe transformation as object/action sequence]
Confidence: Low/Medium/High - [Why?]

**Testing on Example 2:**
Does initial hypothesis explain Example 2? ✓ / ✗
- If ✗: What's wrong or missing?
- If ✓: What additional details does Example 2 reveal?

Evolved hypothesis: [Refined transformation description]
Confidence: Low/Medium/High - [Why?]

**Testing on Example 3:**
Does evolved hypothesis explain Example 3? ✓ / ✗
- If ✗: What's wrong or missing?
- If ✓: What additional constraints does Example 3 reveal?

Refined hypothesis: [Final refined transformation description]
Confidence: Low/Medium/High - [Why?]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### STEP 3: Similar Programs Analysis

**Review the library programs** (shown above with similarity scores):

For each similar program:
- Task ID: [X] - Similarity: [score]
- What transformation does it do? [describe in object/action terms]
- WHY might it be similar? [what execution pattern is shared?]
- Does it suggest anything about your hypothesis? [insights]

**Important:** High similarity ≠ same logic. These may solve DIFFERENT patterns 
that happen to produce similar input/output relationships. Use as inspiration, 
not as a template.

**Key insight from similar programs:**
[Do they suggest your hypothesis, an alternative, or something you missed?]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### STEP 4: Generate Alternative Hypotheses

Now generate at least 2 alternatives that ALSO explain all examples but use 
different reasoning:

**Primary (evolved from Steps 1-2):** 
[Your refined hypothesis]

**Alternative A:** 
[Different object/action interpretation]
- Why it could work: [evidence from examples]
- Why it might be better/worse than primary: [reasoning]
- Key difference from primary: [what's fundamentally different?]

**Alternative B:** 
[Another different interpretation]
- Why it could work: [evidence from examples]
- Why it might be better/worse than primary: [reasoning]
- Key difference from primary: [what's fundamentally different?]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### STEP 5: Lock-in Checkpoint ⚠️ (where you usually fail)

Before committing, force these checks:

**A. Evolution check:**
Looking back at Steps 1-2, did you genuinely evolve your hypothesis based on 
new evidence, or did you rationalize away problems to keep your first guess?

Red flags: "it mostly works", "except for this one case", "basically the same"

Honest assessment: [Did I force-fit? Y/N and why]

**B. Alternative comparison:**
Test each hypothesis fairly:

| Hypothesis | Ex1 Match | Ex2 Match | Ex3 Match | # Special Cases | Simplicity |
|------------|-----------|-----------|-----------|-----------------|------------|
| Primary | ✓/✗ | ✓/✗ | ✓/✗ | [count] | [1-5] |
| Alternative A | ✓/✗ | ✓/✗ | ✓/✗ | [count] | [1-5] |
| Alternative B | ✓/✗ | ✓/✗ | ✓/✗ | [count] | [1-5] |

**C. Confidence breakdown:**
Rate your confidence on different aspects:

- Pattern explains Example 1: [Low/Med/High] - [Why?]
- Pattern explains Example 2: [Low/Med/High] - [Why?]
- Pattern explains Example 3: [Low/Med/High] - [Why?]
- Pattern is simple/intuitive: [Low/Med/High] - [Why?]
- Pattern captures ALL similarities: [Low/Med/High] - [Why?]

Overall confidence: [X%]
What would increase confidence: [what evidence/test?]
What would decrease confidence: [what would disprove it?]

**D. Similarity verification:**
Does your chosen hypothesis explain:
- The DIFFERENCES within each example? ✓ / ✗
- The SIMILARITIES across all examples? ✓ / ✗

The core pattern across all examples is: [state it clearly]
My hypothesis captures this because: [explain connection]

**E. Simplicity test:**
Explain your hypothesis in ONE sentence (like explaining to a friend):
[State it]

If this explanation is convoluted or has many "except when" clauses, 
it's probably wrong.

**F. Disconfirming evidence hunt:**
Actively search for evidence AGAINST your primary hypothesis:
- What doesn't quite fit? [list]
- What should be there but isn't? [list]
- Which example fits least well? [Example X because...]

**Decision:** Stick with primary or switch to alternative? 
[Choice] because [specific reasoning based on checks A-F]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### STEP 6: Final Transformation Rule

**Pattern:** [1-2 sentence description in object/action language]

**Reasoning:**
[Explain WHY this pattern makes sense - what's the underlying logic?]

**Step-by-step process:**
1. [First: what to identify/find]
2. [Second: what action to perform]
3. [Third: how to produce output]
[Add more steps if needed for multi-stage transformations]

**Conditions/Constraints:** 
[Any "if-then" rules or special cases: "if X then Y, otherwise Z"]

**Why this works:**
[Brief explanation of why this pattern explains all examples and their similarities]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Begin your analysis:"""
    
    def _load_phase2_template(self) -> str:
        """Phase 2: Code Generator Template"""
        return """# DSL Code Generator
Generate a Python `solve(I)` function using ONLY the DSL primitives below.

## Training Examples
[INSERT TASK EXAMPLES HERE]

## Natural Language Pattern Description
[PHASE 1 PATTERN DESCRIPTION]

## Similar Programs for Reference
The following programs solved similar tasks and may provide useful patterns or approaches:

[SIMILAR PROGRAMS FOR REFERENCE]

## DSL Primitives

**Type Definitions:**
Grid: Tuple[Tuple[int, ...], ...] - Immutable 2D array (tuple of tuples)
Object: FrozenSet[(int, (int, int))] - Set of (color, location) pairs
Patch: FrozenSet[(int, int)] or Object - Set of indices or colored object
Indices: FrozenSet[(int, int)] - Set of (row, col) positions
Objects: FrozenSet[Object] - Set of objects
Container: Tuple or FrozenSet - Generic container type
IntegerTuple: (int, int) - Tuple of integers (usually coordinates or dimensions)
Callable: Function type

**Functional Programming:**
```python
compose(f, g) -> Callable                    # f(g(x)) - function composition
chain(f, g, h, ...) -> Callable              # f(g(h(...(x)))) - chain multiple functions
fork(combiner, f, g) -> Callable             # combiner(f(x), g(x)) - apply two functions and combine
apply(f, container) -> Container             # apply f to each element
mapply(f, container) -> FrozenSet            # apply f over container followed by merge results
lbind(f, arg) -> Callable                    # f(arg, x) - left partial application
rbind(f, arg) -> Callable                    # f(x, arg) - right partial application
identity(x) -> Any                           # returns x unchanged
matcher(f, container) -> Callable            # find element in container where f matches
extract(container, f) -> Any                 # first element satisfying predicate f
```

**Grid Transforms:**
```python
hmirror(grid) -> Grid/Patch                  # flip left-right
vmirror(grid) -> Grid/Patch                  # flip up-down
dmirror(grid) -> Grid/Patch                  # diagonal \\ mirror
cmirror(grid) -> Grid/Patch                  # diagonal / mirror
rot90(grid) -> Grid                          # rotate 90° clockwise
rot180(grid) -> Grid                         # rotate 180°
rot270(grid) -> Grid                         # rotate 270° clockwise
vconcat(a, b) -> Grid                        # stack vertically [a; b]
hconcat(a, b) -> Grid                        # stack horizontally [a, b]
crop(grid, (i,j), (h,w)) -> Grid             # extract subgrid starting at (i,j) with dimensions (h,w)
upscale(grid, n) -> Grid/Object              # enlarge by factor n
downscale(grid, n) -> Grid                   # shrink by factor n
hsplit(grid, n) -> Tuple                     # split horizontally into n parts
vsplit(grid, n) -> Tuple                     # split vertically into n parts
tophalf/bottomhalf -> Grid                   # get top/bottom half
lefthalf/righthalf -> Grid                   # get left/right half
trim(grid) -> Grid                           # remove border
```

**Objects:**
```python
objects(grid, univalued, diagonal, without_bg) -> FrozenSet[Object]
# Find connected components (separate shapes/regions)
# univalued: T = single-color objects, F = multicolor allowed
# diagonal: T = diagonal connects (8-connected), F = only orthogonal (4-connected)
# without_bg: T = ignore background (most common color)
# Returns: frozenset of objects

colorfilter(objects, color) -> FrozenSet[Object]  # Keep only objects of specified color
sizefilter(objects, n) -> FrozenSet               # Keep only objects with exactly n cells
ofcolor(grid, color) -> Indices                   # get all indices of specified color
toobject(patch, grid) -> Object                   # convert patch to object with colors
normalize(obj) -> Patch                           # move object to origin (0,0)
toindices(obj) -> Indices                         # extract just the (i,j) positions
asindices(grid) -> Indices                        # all non-background cell positions
```

**Modifications:**
```python
fill(grid, color, patch) -> Grid                  # color cells in patch/indices
paint(grid, obj) -> Grid                          # draw object onto grid
replace(grid, old, new) -> Grid                   # swap all instances of old color with new
switch(grid, a, b) -> Grid                        # swap two colors
move(grid, obj, offset) -> Grid                   # move object by offset
shift(patch, (di, dj)) -> Patch                   # translate patch by offset
cover(grid, patch) -> Grid                        # remove object (fill with background)
```

**Spatial Queries:**
```python
ulcorner(obj) -> IntegerTuple                     # upper-left corner
urcorner(obj) -> IntegerTuple                     # upper-right corner
llcorner(obj) -> IntegerTuple                     # lower-left corner
lrcorner(obj) -> IntegerTuple                     # lower-right corner
center(patch) -> IntegerTuple                     # center point of patch
corners(patch) -> Indices                         # all 4 corner positions
position(a, b) -> IntegerTuple                    # relative position between patches
box(patch) -> Indices                             # outline of bounding box
inbox(patch) -> Indices                           # interior outline of box
outbox(patch) -> Indices                          # exterior outline of box
backdrop(patch) -> Indices                        # all indices in bounding box
delta(patch) -> Indices                           # bounding box minus the patch
vfrontier(loc) -> Indices                         # vertical line through location
hfrontier(loc) -> Indices                         # horizontal line through location
shoot(start, direction) -> Indices                # ray from point in direction
connect(a, b) -> Indices                          # line between two points
```

**Properties:**
```python
height(grid) -> Integer                           # number of rows
width(grid) -> Integer                            # number of columns
shape(grid/obj) -> IntegerTuple                   # (height, width) tuple
size(container) -> Integer                        # number of elements
palette(grid) -> FrozenSet[Integer]               # set of colors used
mostcolor(grid) -> Integer                        # most frequent color (background)
leastcolor(grid) -> Integer                       # least frequent color
color(obj) -> Integer                             # color of single-color object
index(grid, (i, j)) -> Integer/None               # value at location
occurrences(grid, obj) -> Indices                 # locations where obj appears
compress(grid) -> Grid                            # remove uniform rows/columns
frontiers(grid) -> FrozenSet[Object]              # get uniform rows/columns
hperiod(obj) -> Integer                           # horizontal period
vperiod(obj) -> Integer                           # vertical period
gravitate(src, dst) -> IntegerTuple               # offset to move src next to dst
```

**Set Operations:**
```python
combine(a, b) -> Container                        # union of containers
intersection(a, b) -> FrozenSet                   # intersection of sets
difference(a, b) -> FrozenSet                     # a - b (set difference)
merge(containers) -> Container                    # flatten container of containers
dedupe(tuple) -> Tuple                            # remove duplicates
sfilter(container, f) -> Container                # filter by predicate
mfilter(container, f) -> FrozenSet                # filter and merge
contained(val, set) -> Boolean                    # membership test (val in set)
initset(value) -> FrozenSet                       # create singleton frozenset
```

**Aggregation:**
```python
argmax(container, f) -> Any                       # item with maximum f(item)
argmin(container, f) -> Any                       # item with minimum f(item)
valmax(container, f) -> Integer                   # maximum value of f over container
valmin(container, f) -> Integer                   # minimum value of f over container
maximum(container) -> Integer                     # max element
minimum(container) -> Integer                     # min element
mostcommon(container) -> Any                      # most frequent element
leastcommon(container) -> Any                     # least frequent element
order(container, key) -> Tuple                    # sort by key function
```

**Arithmetic:**
```python
add(a, b) -> Numerical                            # addition (works on ints and tuples)
subtract(a, b) -> Numerical                       # subtraction
multiply(a, b) -> Numerical                       # multiplication
divide(a, b) -> Numerical                         # floor division
invert(n) -> Numerical                            # negation
double(n) -> Numerical                            # multiply by 2
halve(n) -> Numerical                             # divide by 2
increment(x) -> Numerical                         # add 1
decrement(x) -> Numerical                         # subtract 1
sign(x) -> Numerical                              # -1, 0, or 1 based on sign
toivec(i) -> IntegerTuple                         # (i, 0) - vertical vector
tojvec(j) -> IntegerTuple                         # (0, j) - horizontal vector
```

**Constructors:**
```python
canvas(color, (height, width)) -> Grid            # Create blank grid
astuple(a, b, ...) -> IntegerTuple                # create tuple from elements
repeat(item, n) -> Tuple                          # tuple with item repeated n times
```

**Boolean:**
```python
equality(a, b) -> Boolean                         # a == b
both(a, b) -> Boolean                             # a and b
either(a, b) -> Boolean                           # a or b
flip(b) -> Boolean                                # not b
contained(val, set) -> Boolean                    # val in set
positive(n) -> Boolean                            # n > 0
even(n) -> Boolean                                # n % 2 == 0
greater(a, b) -> Boolean                          # a > b
```

**Advanced:**
```python
cellwise(a, b, fallback) -> Grid                  # compare grids cell-by-cell
```

**Constants:**
```python
# Colors: ZERO=0, ONE=1, TWO=2, THREE=3, FOUR=4, FIVE=5, SIX=6, SEVEN=7, EIGHT=8, NINE=9
# Directions: UP=(-1,0), DOWN=(1,0), LEFT=(0,-1), RIGHT=(0,1)
# Diagonals: UNITY=(1,1), NEG_UNITY=(-1,-1), UP_RIGHT=(-1,1), DOWN_LEFT=(1,-1)
# Special: ORIGIN=(0,0), T=True, F=False
```

## Control Flow Allowed
- Single-level `for` loops (no nesting)
- `if/else` conditionals
- List comprehensions

## Requirements
- Function signature: `def solve(I):`
- Input `I` is tuple of tuples of ints (Grid)
- Return same format
- Use ONLY DSL primitives listed above
- Add brief comments for clarity
- Use functional programming patterns when appropriate (compose, chain, fork, lbind/rbind, mapply)
- You can adapt patterns from similar programs but adjust them to match the pattern description

## Output Format
```python
def solve(I):
    # [comment explaining step]
    x1 = dsl_function(I)
    
    # [comment]
    x2 = another_function(x1, args)
    
    # [final step]
    return O # O is the output grid
```

Generate the solve function now:"""