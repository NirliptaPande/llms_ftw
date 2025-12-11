from typing import List, Tuple, Dict, Any, Union
import numpy as np
from utils.render_legacy import grid_to_base64_png_oai_content


class VLMPrompter:
    """Builds prompts for the program-first ARC solving process"""
    
    def __init__(self):
        # Don't cache templates that depend on dsl_enabled
        self.phase2b_template = self._load_phase2b_template()
    
    def build_phase2a_prompt(self, 
                             task: Dict[str, Any],
                             similar_programs: List[Dict[str, Any]] = None,
                             dsl_enabled: bool = True) -> List[Dict[str, Any]]:
        """
        Build Phase 2A prompt: Hypothesis Formation from Training Only.
        """
        content_blocks = []
        
        # Add header
        content_blocks.append({
            "type": "text",
            "text": "## Training Examples\n"
        })
        
        # Format training examples ONLY (with images)
        content_blocks.extend(self._format_training_examples(task['train']))

        if dsl_enabled:  
            # Add similar programs section
            content_blocks.append({
                "type": "text",
                "text": "\n## Similar Programs for Reference.\nThe following programs solved similar tasks and may provide useful patterns or approaches:\n"
            })
      
            if similar_programs:
                similar_str = self._format_similar_programs(similar_programs)
            else:
                similar_str = "[No similar programs found]"
            
            content_blocks.append({
                "type": "text",
                "text": similar_str
            })
        
        # Get template with dsl_enabled parameter
        template = self._load_phase2a_template(dsl_enabled)
        
        # Add the analysis protocol template (hypothesis formation)
        content_blocks.append({
            "type": "text",
            "text": template
        })
        
        return content_blocks
    
    def build_phase2b_prompt(self,
                             task: Dict[str, Any],
                             hypothesis: str,
                             similar_programs: List[Dict[str, Any]] = None,
                             dsl_enabled: bool = True) -> List[Dict[str, Any]]:
        """
        Build Phase 2B prompt: Hypothesis Validation with Training + Test.
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
        
        if dsl_enabled:
            content_blocks.append({
                "type": "text",
                "text": "\n## Similar Programs for Reference\n"
            })
            
            if similar_programs:
                similar_str = self._format_similar_programs(similar_programs)
            else:
                similar_str = "[No similar programs found]"
                
            content_blocks.append({
                "type": "text",
                "text": similar_str
            })
        
        # Add validation template
        content_blocks.append({
            "type": "text",
            "text": self.phase2b_template
        })
        
        return content_blocks
    
    def build_phase2c_prompt(self,
                             task: Dict[str, Any], 
                             validated_pattern: str,
                             similar_programs: List[Dict[str, Any]] = None,
                             few_shot: bool = True,
                             dsl_enabled: bool = True) -> List[Dict[str, Any]]:
        """
        Build Phase 2C prompt: Code Generation from Validated Pattern.
        """
        content_blocks = []
        
        # Add header
        if dsl_enabled:
            content_blocks.append({
                "type": "text",
                "text": "# DSL Code Generator\nGenerate a Python `solve(I)` function using ONLY the DSL primitives below.\n\n"
            })
        else:
            content_blocks.append({
                "type": "text",
                "text": "# Python Code Generator\nGenerate a Python `solve(I)` function.\n\n"
            })
        
        # Add training examples section
        content_blocks.append({
            "type": "text",
            "text": "## Training Examples\n"
        })
        
        # Format training examples (with images)
        content_blocks.extend(self._format_training_examples(task['train'], include_images=True))
        
        # Format test examples (input only, no output)
        content_blocks.extend(self._format_test_examples(task['test'], include_images=True))
        
        # Add pattern description
        content_blocks.append({
            "type": "text",
            "text": f"\n## Natural Language Pattern Description\n{validated_pattern}\n"
        })
        
        if dsl_enabled:
            # Add similar programs section
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
        
        # Add DSL primitives or Python instructions (dynamically generated)
        content_blocks.append({
            "type": "text",
            "text": self._get_phase2c_dsl_section(dsl_enabled)
        })
        
        # Add few-shot examples only if DSL is enabled
        if few_shot and dsl_enabled:
            content_blocks.append({
                "type": "text",
                "text": self._get_phase2c_fewshot_section()
            })
            
        content_blocks.append({
            "type": "text",
            "text": "Generate the `solve(I)` function now.\n"
        })
        
        return content_blocks

    def build_2d_prompt(self,
                        task: Dict[str, Any],
                        best_program_code: str,
                        diff_grid:  List[Tuple],
                        second_best_program_code: str,
                        diff_grid2:  List[Tuple],
                        dsl_enabled: bool = True) -> List[Dict[str, Any]]:
        """
        Build 2D prompt: Single-step prompt for combined hypothesis formation and code generation.
        """
        content_blocks = []
        content_blocks.append({
            "type": "text",
            "text": "You are an expert at debugging and repairing Python code for ARC puzzles.\n"
        })
        
        content_blocks.append({
            "type": "text",
            "text": "You are given a 2D ARC task with training and test samples. You are also given 2 programs that were generated in response to the ARC task. However, they are incorrect. You are given the corresponding output of the incorrect programs and the difference with the actual output from the training samples if available. Your goal is to generate a Python `solve(I)` to solve these tasks."
        })
        if dsl_enabled:
            content_blocks.append({
                "type": "text",
                "text": " using ONLY the DSL primitives below.\n\n"
            })
        else:
            content_blocks.append({
                "type": "text",
                "text": ".\n\n"
            })
        
        # Add training examples section
        content_blocks.append({
            "type": "text",
            "text": "## Training Examples\n"
        })
        
        # Format training examples (with images)
        content_blocks.extend(self._format_training_examples(task['train'], include_images=True))
        
        # Format test examples (input only, no output)
        content_blocks.extend(self._format_test_examples(task['test'], include_images=True))
        
        # Add first incorrect program and diff grid
        content_blocks.append({
            "type": "text",
            "text": f"\n## First Incorrect Program\n```python\n{best_program_code}\n```\n"
        })
        
        content_blocks.append({
            "type": "text",
            "text": "### Difference Grid from Actual Output\n"
        })
        content_blocks.append({
            "type": "text",
            "text": f"\nASCII representation:\n{self._format_grid(diff_grid, separator='|')}\n"
        })
        
        # Add second incorrect program and diff grid
        content_blocks.append({
            "type": "text",
            "text": f"\n## Second Incorrect Program\n```python\n{second_best_program_code}\n```\n"
        })
        
        content_blocks.append({
            "type": "text",
            "text": "### Difference Grid from Actual Output\n"
        })
        content_blocks.append({
            "type": "text",
            "text": f"\nASCII representation:\n{self._format_grid(diff_grid2, separator='|')}\n"
        })
        # Add DSL primitives or Python instructions (dynamically generated)
        if dsl_enabled:
            content_blocks.append({
                "type": "text",
                "text": self._get_phase2c_dsl_section(dsl_enabled)
            })
        
        content_blocks.append({
            "type": "text",
            "text": "Generate the `solve(I)` function now.\n"
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
    
    def _format_grid(self, grid: Union[Tuple[Tuple[int]], List[List]], separator: str = "|") -> str:
        return "\n".join(separator.join(str(cell) for cell in row) for row in grid)
    
    def _format_similar_programs(self, similar_programs: List[Dict[str, Any]]) -> str:
        """Format similar programs for pattern discovery and code generation"""
        lines = []
        lines.append("The following programs may be useful to solve the current task, feel free to use parts of each program, and/or combine them. Most importantly, use them as guidance:\n")
        
        for idx, prog in enumerate(similar_programs, 1):
            similarity = prog.get('similarity', 0.0)
            program_code = prog.get('program', '')
            task_id = prog.get('task_id', 'unknown')
            example_scores = prog.get('example_scores', [])
            perfect_count = sum(1 for s in example_scores if s == 1.0)
            total_examples = len(example_scores)
            
            lines.append(f"Similar Program {idx} (similarity: {similarity:.2f}, task: {task_id}):")
            if total_examples > 0:
                lines.append(f"# This program solved {perfect_count}/{total_examples} training examples perfectly.")
            lines.append("```python")
            lines.append(program_code)
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)
    
    def _load_phase2a_template(self, dsl_enabled: bool = True) -> str:
        """Template for hypothesis formation (training only)"""
        
        if dsl_enabled:
            intro = "\n## Analysis Protocol\n\nThe similar programs shown above may provide useful patterns, but focus primarily on understanding the transformation through the training examples.\n\n"
        else:
            intro = "\n## Analysis Protocol\n\n"
        
        body = """You will analyze the examples systematically, allowing your hypothesis to evolve naturally as you see more data - like a human solving a puzzle.

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
        
        return intro + body

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
    
    def _get_phase2c_dsl_section(self, dsl_enabled: bool = True) -> str:
        """Phase 2C: DSL Primitives and Code Generation Instructions"""
        if dsl_enabled:
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
"""
        else:
            return """
## Requirements
- Function signature: `def solve(I):`
- Input `I` is `Tuple[Tuple[int, ...], ...]` - Immutable 2D array (tuple of tuples)
- Return same format: tuple of tuples of int
- Write the solution in **pure Python** using standard libraries
- You can use: numpy, itertools, collections, or any standard Python constructs
- Add clear comments explaining your logic
- Use any functions, loops, and data structures you need

## Approach
Think about the transformation step-by-step:
1. Analyze the input grid structure
2. Identify the pattern/transformation rule
3. Implement the logic to apply this transformation
4. Return the output grid in the same tuple-of-tuples format

## Output Format
```python
def solve(I):
    # Convert to working format if needed
    # (e.g., list of lists, numpy array, etc.)
    
    # [Your transformation logic here]
    # Use clear variable names and comments
    
    # Convert back to tuple of tuples
    O = tuple(tuple(row) for row in result)
    return O
```

**Example Structure:**
```python
def solve(I):
    # Convert input to mutable format
    grid = [list(row) for row in I]
    height, width = len(grid), len(grid[0])
    
    # Apply transformation logic
    for i in range(height):
        for j in range(width):
            # Your logic here
            pass
    
    # Return as tuple of tuples
    return tuple(tuple(row) for row in grid)
```
"""
        
    def _get_phase2c_fewshot_section(self) -> str:
        """Few-shot examples for Phase 2C (if needed)"""
        return r"""
    Here are some examples of ARC tasks and their corresponding `solve(I)` functions:
    ---------------------------
# Task 1:

Below are 2 training examples follwed by the test example(s) you have to generalize to, for each example, the input grid is shown first, followed by the output grid. 

Example 1:
Input:

ASCII representation:
0|0|0|0|0|0|0|0|0|0|0
1|0|0|0|0|0|0|0|0|0|2
0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0

Output:

ASCII representation:
0|0|0|0|0|0|0|0|0|0|0
1|1|1|1|1|5|2|2|2|2|2
0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0

Example 2:
Input:

ASCII representation:
0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0
3|0|0|0|0|0|0|0|0|0|7
0|0|0|0|0|0|0|0|0|0|0

Output:

ASCII representation:
0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0
3|3|3|3|3|5|7|7|7|7|7
0|0|0|0|0|0|0|0|0|0|0


============================================================
TEST EXAMPLES (to solve)
============================================================
Below are 1 test example(s) you need to solve:
For each test example, only the input grid is provided. You must determine the output.

Test Example 1:
Input:

ASCII representation:
0|0|0|0|0|0|0|0|0|0|0
4|0|0|0|0|0|0|0|0|0|8
0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0
6|0|0|0|0|0|0|0|0|0|9

Output: [TO BE DETERMINED]

Solution:
```python
def solve(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = objects(x2, T, F, T)
    x4 = objects(x1, T, F, T)
    x5 = compose(hfrontier, center)
    x6 = fork(recolor, color, x5)
    x7 = mapply(x6, x4)
    x8 = paint(x1, x7)
    x9 = mapply(x6, x3)
    x10 = paint(I, x9)
    x11 = objects(x8, T, F, T)
    x12 = apply(urcorner, x11)
    x13 = shift(x12, RIGHT)
    x14 = merge(x11)
    x15 = paint(x10, x14)
    O = fill(x15, FIVE, x13)
    return O
```

---------------------------
# Task 2:

Below are 3 training examples follwed by the test example(s) you have to generalize to, for each example, the input grid is shown first, followed by the output grid. 

Example 1:
Input:

ASCII representation:
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|1|0|1|0|0|1|1|0|1|0|0
0|0|1|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|1|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|1|0|0
0|0|1|0|0|0|0|0|0|0|1|0|0
0|0|1|1|0|0|1|1|0|1|1|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0

Output:

ASCII representation:
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|1|2|1|2|2|1|1|2|1|0|0
0|0|1|0|0|0|0|0|0|0|2|0|0
0|0|2|0|0|0|0|0|0|0|1|0|0
0|0|2|0|0|0|0|0|0|0|2|0|0
0|0|2|0|0|0|0|0|0|0|1|0|0
0|0|1|0|0|0|0|0|0|0|1|0|0
0|0|1|1|2|2|1|1|2|1|1|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0

Example 2:
Input:

ASCII representation:
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|1|1|1|0|0|1|1|0|0|0|0
0|0|1|0|0|0|0|0|1|0|0|0|0
0|0|0|0|1|0|0|0|0|0|0|0|0
0|0|1|0|1|0|0|0|1|0|0|0|0
0|0|1|0|0|0|0|0|1|0|0|0|0
0|0|0|0|1|0|0|0|1|0|0|0|0
0|0|1|1|1|1|0|1|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0

Output:

ASCII representation:
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|1|1|1|2|2|1|1|0|0|0|0
0|0|1|0|2|0|0|0|1|0|0|0|0
0|0|2|0|1|0|0|0|2|0|0|0|0
0|0|1|0|1|0|0|0|1|0|0|0|0
0|0|1|0|2|0|0|0|1|0|0|0|0
0|0|2|0|1|0|0|0|1|0|0|0|0
0|0|1|1|1|1|2|1|2|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0

Example 3:
Input:

ASCII representation:
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|1|1|0|1|1|0|1|1|1|0|0
0|0|1|0|0|0|0|0|0|0|1|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|1|0|0|0|0|0|0|0|1|0|0
0|0|1|1|0|1|0|1|1|0|0|0|0
0|0|1|0|0|0|0|0|0|0|1|0|0
0|0|0|0|0|0|0|0|0|0|1|0|0
0|0|1|1|0|1|1|0|0|1|1|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0

Output:

ASCII representation:
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|1|1|2|1|1|2|1|1|1|0|0
0|0|1|0|0|0|0|0|0|0|1|0|0
0|0|2|0|0|0|0|0|0|0|2|0|0
0|0|1|0|0|0|0|0|0|0|1|0|0
0|0|1|1|2|1|2|1|1|2|2|0|0
0|0|1|0|0|0|0|0|0|0|1|0|0
0|0|2|0|0|0|0|0|0|0|1|0|0
0|0|1|1|2|1|1|2|2|1|1|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0


============================================================
TEST EXAMPLES (to solve)
============================================================
Below are 1 test example(s) you need to solve:
For each test example, only the input grid is provided. You must determine the output.

Test Example 1:
Input:

ASCII representation:
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|1|0|1|1|0|1|0|1|1|0|0
0|0|1|0|0|0|0|0|0|0|1|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|1|0|0|0|0|0|0|0|1|0|0
0|0|1|0|1|0|1|0|0|1|1|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|1|0|0|0|0|0|0|0|1|0|0
0|0|1|0|1|1|0|1|0|1|1|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0
0|0|0|0|0|0|0|0|0|0|0|0|0

Output: [TO BE DETERMINED]

Solution:
```python
def solve(I):
    x1 = ofcolor(I, ONE)
    x2 = box(x1)
    x3 = fill(I, TWO, x2)
    x4 = subgrid(x1, x3)
    x5 = ofcolor(x4, ONE)
    x6 = mapply(vfrontier, x5)
    x7 = mapply(hfrontier, x5)
    x8 = size(x6)
    x9 = size(x7)
    x10 = greater(x8, x9)
    x11 = branch(x10, x7, x6)
    x12 = fill(x4, TWO, x11)
    x13 = ofcolor(x12, TWO)
    x14 = ulcorner(x1)
    x15 = shift(x13, x14)
    O = underfill(I, TWO, x15)
    return O
```

---------------------------
# Task 3:

Below are 2 training examples follwed by the test example(s) you have to generalize to:
 for each example, the input grid is shown first, followed by the output grid. 
.
Example 1:
Input:

ASCII representation:
8|0|0|0|0|0|8|8|8|8|8|8|0|8|8|8|0|8|8|0|8|8|8|0
0|0|8|8|8|0|0|0|0|0|0|8|0|0|0|8|0|8|0|0|8|0|8|0
8|8|8|0|8|0|8|8|8|8|0|8|8|8|0|8|0|8|8|8|8|0|8|0
8|0|0|0|8|0|8|0|0|8|0|0|0|8|0|8|0|0|0|0|0|0|8|0
8|0|8|8|8|0|8|8|0|8|0|8|8|8|0|8|8|0|8|8|8|8|8|0
8|0|8|0|0|0|0|8|0|8|0|8|0|0|0|0|8|0|8|0|0|0|0|0
8|0|8|8|8|8|8|8|0|8|0|8|8|8|8|8|8|3|8|8|8|8|8|0
8|0|0|0|0|0|0|0|0|8|0|0|0|0|0|0|3|2|3|0|0|0|8|0
8|8|0|8|8|8|0|8|8|8|0|8|8|8|8|8|8|3|8|8|8|0|8|0
0|8|0|8|0|8|0|8|0|0|0|8|0|0|0|0|8|0|8|0|8|0|8|0
0|8|8|8|0|8|8|8|0|8|8|8|0|8|8|0|8|8|8|0|8|8|8|0

Output:

ASCII representation:
8|3|2|3|2|3|8|8|8|8|8|8|0|8|8|8|2|8|8|0|8|8|8|0
3|2|8|8|8|2|3|2|3|2|3|8|0|0|0|8|3|8|0|0|8|2|8|0
8|8|8|0|8|3|8|8|8|8|2|8|8|8|0|8|2|8|8|8|8|3|8|0
8|0|0|0|8|2|8|0|0|8|3|2|3|8|0|8|3|2|3|2|3|2|8|0
8|0|8|8|8|3|8|8|0|8|2|8|8|8|0|8|8|3|8|8|8|8|8|0
8|0|8|2|3|2|3|8|0|8|3|8|0|0|0|0|8|2|8|0|0|0|0|0
8|0|8|8|8|8|8|8|0|8|2|8|8|8|8|8|8|3|8|8|8|8|8|0
8|0|0|0|0|0|0|0|0|8|3|2|3|2|3|2|3|2|3|2|3|2|8|0
8|8|0|8|8|8|0|8|8|8|2|8|8|8|8|8|8|3|8|8|8|3|8|0
0|8|0|8|0|8|0|8|3|2|3|8|0|0|0|0|8|2|8|0|8|2|8|0
0|8|8|8|0|8|8|8|2|8|8|8|0|8|8|0|8|8|8|0|8|8|8|0

Example 2:
Input:

ASCII representation:
0|0|0|8|0|0|0|8|0|0|0|0|0|8
8|8|0|8|8|8|0|8|0|8|8|8|0|8
0|8|0|0|0|8|0|8|0|8|0|8|8|8
0|8|8|8|8|8|0|8|0|8|0|0|0|0
0|0|0|0|0|0|0|8|0|8|8|8|0|8
8|8|8|8|8|8|0|8|0|0|0|8|0|8
8|0|0|0|0|8|0|8|8|8|0|8|0|8
8|8|8|8|0|8|0|0|0|8|0|8|0|0
0|0|0|8|1|8|8|8|8|8|0|8|8|0
8|8|0|8|4|1|0|0|0|0|0|0|8|0
0|8|0|8|1|8|8|8|8|8|8|8|8|0
0|8|8|8|0|8|0|0|0|0|0|0|0|0
0|0|0|0|0|8|0|8|8|8|8|8|8|8

Output:

ASCII representation:
0|0|0|8|0|0|0|8|1|4|1|4|1|8
8|8|0|8|8|8|0|8|4|8|8|8|4|8
0|8|0|0|0|8|0|8|1|8|0|8|8|8
0|8|8|8|8|8|0|8|4|8|0|0|0|0
0|0|0|0|0|0|0|8|1|8|8|8|0|8
8|8|8|8|8|8|0|8|4|1|4|8|0|8
8|4|1|4|1|8|0|8|8|8|1|8|0|8
8|8|8|8|4|8|0|0|0|8|4|8|0|0
0|0|0|8|1|8|8|8|8|8|1|8|8|0
8|8|0|8|4|1|4|1|4|1|4|1|8|0
1|8|0|8|1|8|8|8|8|8|8|8|8|0
4|8|8|8|4|8|0|0|0|0|0|0|0|0
1|4|1|4|1|8|0|8|8|8|8|8|8|8


============================================================
TEST EXAMPLES (to solve)
============================================================
Below are 1 test example(s) you need to solve:
For each test example, only the input grid is provided. You must determine the output.

Test Example 1:
Input:

ASCII representation:
8|8|0|8|0|0|8|0|0|0|0|0|0|0|0
0|8|0|8|8|8|8|4|8|8|8|8|8|8|8
0|8|0|0|0|0|4|3|8|0|0|0|0|0|8
0|8|8|8|8|8|8|4|8|8|8|0|8|8|8
0|0|0|0|0|0|8|0|0|0|8|0|8|0|0
8|8|8|8|8|0|8|8|8|0|8|0|8|0|8
0|0|0|0|8|0|0|0|8|0|8|0|8|0|8
8|8|8|0|8|8|8|0|8|0|8|0|8|8|8
0|0|8|0|0|0|8|0|8|0|8|0|0|0|0
8|0|8|8|8|0|8|8|8|0|8|8|8|0|8
8|0|0|0|8|0|0|0|0|0|0|0|8|0|8
8|8|8|0|8|0|8|8|8|8|8|8|8|0|8
0|0|8|0|8|0|8|0|0|0|0|0|0|0|8
8|0|8|8|8|0|8|0|8|8|8|8|8|8|8
8|0|0|0|0|0|8|0|8|0|0|0|0|0|0

Output: [TO BE DETERMINED]

Solution:
```python
def solve_b782dc8a(I):
    x1 = least_common_color(I)
    x2 = as_objects(I, True, False, False)
    x3 = of_color(I, x1)
    x4 = get_first(x3)
    x5 = direct_neighbors(x4)
    x6 = to_object(x5, I)
    x7 = most_common_color(x6)
    x8 = of_color(I, x7)
    x9 = color_filter(x2, COLOR_ZERO)
    x10 = fix_last_argument(adjacent, x8)
    x11 = keep_if_condition_and_flatten(x9, x10)
    x12 = to_indices(x11)
    x13 = fix_last_argument(manhattan_distance, x3)
    x14 = chain(is_even, x13, initset)
    x15 = keep_if_condition(x12, x14)
    x16 = difference(x12, x15)
    x17 = fill(I, x1, x15)
    O = fill(x17, x7, x16)
    return O
```

---------------------------
    """