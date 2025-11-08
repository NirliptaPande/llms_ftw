"""
Refactored VLM Prompter for ARC DSL-based solver with image support.
New pipeline: Program Similarity → Pattern Discovery (2 phases) → Code Generation
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from utils.render_legacy import grid_to_base64_png_oai_content


class VLMPrompter:
    """Builds prompts for the program-first ARC solving process"""
    
    def __init__(self):
        self.phase2a_template = self._load_phase2a_template()
        self.phase2b_template = self._load_phase2b_template()
        self.phase2c_dsl_section = self._get_phase2c_dsl_section()
    
    def build_phase2a_prompt(self, 
                             task: Dict[str, Any],
                             similar_programs: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Build Phase 2A prompt: Hypothesis Formation from Training Only.
        
        Args:
            task: Dict with 'train' key containing list of {'input': grid, 'output': grid}
            similar_programs: List of similar programs from library (found via execution)
                             Each dict has: {'program': str, 'similarity': float, 'task_id': str}
            
        Returns:
            List of content blocks (dicts) for multimodal prompt
        """
        content_blocks = []
        
        # Add header
        content_blocks.append({
            "type": "text",
            "text": "## Training Examples\n"
        })
        
        # Format training examples ONLY (with images)
        content_blocks.extend(self._format_training_examples(task['train']))
        
        # Add similar programs section
        content_blocks.append({
            "type": "text",
            "text": "\n## Similar Programs from Library\n"
        })
        
        if similar_programs:
            similar_str = self._format_similar_programs(similar_programs)
        else:
            similar_str = "[No similar programs found in library]"
        
        content_blocks.append({
            "type": "text",
            "text": similar_str
        })
        
        # Add the analysis protocol template (hypothesis formation)
        content_blocks.append({
            "type": "text",
            "text": self.phase2a_template
        })
        
        return content_blocks
    
    def build_phase2b_prompt(self,
                             task: Dict[str, Any],
                             hypothesis: str,
                             similar_programs: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Build Phase 2B prompt: Hypothesis Validation with Training + Test.
        
        Args:
            task: Dict with 'train' and 'test' keys
            hypothesis: The hypothesis from Phase 2A
            similar_programs: List of similar programs from library
            
        Returns:
            List of content blocks for multimodal prompt
        """
        content_blocks = []
        
        # Add the initial hypothesis
        content_blocks.append({
            "type": "text",
            "text": f"""## Initial Hypothesis

{hypothesis}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## All Examples (Test + Train)

Your task: Test if this hypothesis works for test examples and train input output pairs below.
If it doesn't fit perfectly, identify what needs to be refined.

"""
        })
        content_blocks.append({
            "type": "text",
            "text": "\n### Test Examples (inputs only)\n"
        })
        content_blocks.extend(self._format_test_examples(task['test'], include_images=True))
        
        # Show ALL examples now (training + test)
        content_blocks.append({
            "type": "text",
            "text": "### Training Examples\n"
        })
        content_blocks.extend(self._format_training_examples(task['train'], include_images=True))
        

        
        # Add validation template
        content_blocks.append({
            "type": "text",
            "text": self.phase2b_template
        })
        
        return content_blocks
    
    def build_phase2c_prompt(self,
                             task: Dict[str, Any], 
                             validated_pattern: str,
                             similar_programs: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Build Phase 2C prompt: Code Generation from Validated Pattern.
        This matches the original build_phase2_prompt structure exactly.
        
        Args:
            task: Dict with 'train' and 'test' keys
            validated_pattern: The validated pattern description from Phase 2B
            similar_programs: Same list of similar programs (for reference during coding)
            
        Returns:
            List of content blocks (dicts) for multimodal prompt
        """
        content_blocks = []
        
        # Add header (same as original)
        content_blocks.append({
            "type": "text",
            "text": "# DSL Code Generator\nGenerate a Python `solve(I)` function using ONLY the DSL primitives below.\n\n"
        })
        
        # Add training examples section (same as original)
        content_blocks.append({
            "type": "text",
            "text": "## Training Examples\n"
        })
        
        # Format training examples (with images)
        content_blocks.extend(self._format_training_examples(task['train'], include_images=True))
        
        # Format test examples (input only, no output)
        content_blocks.extend(self._format_test_examples(task['test'], include_images=True))
        
        # Add pattern description (using validated pattern instead of extracting from phase1)
        content_blocks.append({
            "type": "text",
            "text": f"\n## Natural Language Pattern Description\n{validated_pattern}\n"
        })
        
        # Add similar programs section (same as original)
        content_blocks.append({
            "type": "text",
            "text": "\n## Similar Programs for Reference\nThe following programs solved similar tasks and may provide useful patterns or approaches:\n\n"
        })
        
        if similar_programs:
            similar_str = self._format_similar_programs(similar_programs)
        else:
            similar_str = "[No similar programs available for reference]"
        
        content_blocks.append({
            "type": "text",
            "text": similar_str
        })
        
        # Add DSL primitives and the rest of the template (same as original)
        content_blocks.append({
            "type": "text",
            "text": self.phase2c_dsl_section
        })
        
        return content_blocks
    
    def _format_test_examples(self, test_examples: List[Dict[str, Any]], include_images: bool = True) -> List[Dict[str, Any]]:
        """Format test examples as content blocks with images (input only, no output)"""
        content_blocks = []
        
        # Header
        content_blocks.append({
            "type": "text",
            "text": f"\n{'='*60}\nTEST EXAMPLES (to solve)\n{'='*60}\n"
        })
        
        content_blocks.append({
            "type": "text",
            "text": f"Below are {len(test_examples)} test example(s) you need to solve:\nFor each test example, only the input grid is provided. You must determine the output.\n"
        })
        
        for idx, example in enumerate(test_examples, 1):
            # Test example header
            content_blocks.append({
                "type": "text",
                "text": f"\nTest Example {idx}:\n"
            })
            
            # Input section
            content_blocks.append({
                "type": "text",
                "text": "Input:\n"
            })
            
            # Input image - conditional
            if include_images:
                input_grid = np.array(example['input'])
                content_blocks.append(grid_to_base64_png_oai_content(input_grid))
            
            # Input ASCII
            content_blocks.append({
                "type": "text",
                "text": f"\nASCII representation:\n{self._format_grid(example['input'], separator='|')}\n"
            })
            
            # Placeholder for output
            content_blocks.append({
                "type": "text",
                "text": "\nOutput: [TO BE DETERMINED]\n"
            })
        
        return content_blocks

    def _format_training_examples(self, train_examples: List[Dict[str, Any]], include_images: bool = True) -> List[Dict[str, Any]]:
        """Format training examples as content blocks with images"""
        content_blocks = []
        content_blocks.append({
            "type": "text",
            "text": f"Below are {len(train_examples)} training example(s):\nFor each example, the input grid is shown first, followed by the output grid.\n"
        })
        
        for idx, example in enumerate(train_examples, 1):
            # Example header
            content_blocks.append({
                "type": "text",
                "text": f"\nExample {idx}:\n"
            })
            
            # Input section
            content_blocks.append({
                "type": "text",
                "text": "Input:\n"
            })
            
            # Input image - conditional
            if include_images:
                input_grid = np.array(example['input'])
                content_blocks.append(grid_to_base64_png_oai_content(input_grid))
            
            # Input ASCII (optional, for reference)
            content_blocks.append({
                "type": "text",
                "text": f"\nASCII representation:\n{self._format_grid(example['input'], separator='|')}\n"
            })
            
            # Output section
            content_blocks.append({
                "type": "text",
                "text": "\nOutput:\n"
            })
            
            # Output image - conditional
            if include_images:
                output_grid = np.array(example['output'])
                content_blocks.append(grid_to_base64_png_oai_content(output_grid))
            
            # Output ASCII (optional, for reference)
            content_blocks.append({
                "type": "text",
                "text": f"\nASCII representation:\n{self._format_grid(example['output'], separator='|')}\n"
            })
        
        return content_blocks   
    
    def _format_grid(self, grid: Tuple[Tuple[int]], separator: str = "|") -> str:
        return "\n".join(separator.join(str(cell) for cell in row) for row in grid)
    
    def _format_similar_programs(self, similar_programs: List[Dict[str, Any]]) -> str:
        """Format similar programs for pattern discovery and code generation"""
        lines = []
        lines.append("The following programs may be useful to solve the current task, feel free to use parts of each program, and/or combine them. Most importantly, use them as guidance:\n")
        
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
    
    def _load_phase2a_template(self) -> str:
        """Template for hypothesis formation (training only)"""
        return r"""
## Analysis Protocol

You will analyze the examples systematically, allowing your hypothesis to evolve naturally as you see more data - like a human solving a puzzle.

**Core principle:** Look for DIFFERENCES within each example (input→output changes) and SIMILARITIES across all examples (the consistent pattern).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Work through examples sequentially, reasoning in simple English about what you observe.

**Step 1: First Example Analysis**
Given: Input 1 → Output 1

<observation_1>
Describe what you see:
- What's the size/shape change?
- What colors changed? 
- What geometric transformations occurred?
- What patterns do you notice?
</observation_1>

<hypothesis_1>
State your initial guess about the transformation rule in natural language.
Example: "The grid appears to be flipped horizontally"
</hypothesis_1>

**Step 2: Second Example Validation**
Given: Input 2 → Output 2

<observation_2>
- Does your hypothesis from Example 1 still hold?
- What's similar to Example 1? 
- What's different from Example 1?
</observation_2>

<hypothesis_2>
Refine your hypothesis:
- If it still works: Confirm and strengthen
- If it breaks: Revise with a more general pattern
Example: "Actually, it's mirrored horizontally, THEN the original is stacked on top"
</hypothesis_2>

**Step 3: Third Example Confirmation**
Given: Input 3 → Output 3

<observation_3>
- Does hypothesis_2 work here?
- Any new edge cases or variations?
</observation_3>

<hypothesis_3>
Final refined hypothesis in natural language.
</hypothesis_3>

**Step N: Additional Examples**
Continue for all remaining examples...

**Final Step: Pattern Synthesis**
<pattern_summary>
In plain English, the transformation rule is:
- [Short description of relevant observations]
- [First operation in natural language]
- [Second operation in natural language]
- [Any conditions or special cases]

Edge cases to consider:
- [Any variations you noticed]

Why this works:
- [Brief explanation about WHY this pattern makes sense]

Confidence level: [0-100%]
</pattern_summary>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Begin your analysis:"""
    
    def _load_phase2b_template(self) -> str:
        """Template for hypothesis validation (all examples)"""
        return r"""
## Validation Protocol

Check if the initial hypothesis fits ALL examples (training + test inputs).
Your goal: Confirm the hypothesis or refine it as needed.

**For each test input:**
<test_check_N>
Does the hypothesis make sense for Test Input N?
- Consider: size, colors, patterns, edge cases
- Any concerns or refinements needed?
</test_check_N>

Does it explain all the test and training examples perfectly?

**Final Validation:**
<validated_pattern>
After checking all examples:

**Status:** [CONFIRMED / NEEDS REFINEMENT]

**Final Pattern Description:**
[If confirmed: restate the pattern cleanly]
[If refined: provide the improved pattern with explanations of what changed]

**Confidence:** [0-100%]

**Reasoning:** [Why this pattern works for all examples]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Begin validation:"""
    
    def _get_phase2c_dsl_section(self) -> str:
        """Phase 2C: DSL Primitives and Code Generation Instructions (EXACT copy from original)"""
        return """
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

Generate the solve function now:
"""