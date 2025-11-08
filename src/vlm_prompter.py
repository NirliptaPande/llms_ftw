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
compose(outer, inner) -> Callable                    # outer(inner(x))
chain(h, g, f) -> Callable                           # h(g(f(x)))
combine_two_function_results(outer, f1, f2) -> Callable  # outer(f1(x), f2(x))
transform(f, container) -> Container                 # map
transform_and_flatten(f, container) -> FrozenSet     # map + flatten
transform_both(f, a, b) -> Tuple                     # pairwise map over tuples
transform_both_and_flatten(f, a, b) -> Tuple         # pairwise map + flatten
apply_each_function(funcs, value) -> Container       # apply each function to the same value
apply_function_on_cartesian_product(f, a, b) -> FrozenSet
fix_first_argument(f, arg) -> Callable                # left partial
fix_last_argument(f, arg) -> Callable                 # right partial
equals(f, target) -> Callable                         # x -> f(x) == target
extract_first_matching(container, pred) -> Any        # first element satisfying pred
power(f, n) -> Callable                               # f applied n times
identity(x) -> Any                                    # returns x unchanged
```

**Grid Transforms:**
```python
horizontal_mirror(grid) -> Grid/Patch                # flip up-down
vertical_mirror(grid) -> Grid/Patch                  # flip left-right
diagonal_mirror(grid) -> Grid/Patch                  # diagonal \ mirror
counterdiagonal_mirror(grid) -> Grid/Patch           # diagonal / mirror
rot90(grid) -> Grid                                  # rotate 90° clockwise
rot180(grid) -> Grid                                 # rotate 180°
rot270(grid) -> Grid                                 # rotate 270° clockwise
vertical_concat(a, b) -> Grid                        # stack vertically [a; b]
horizontal_concat(a, b) -> Grid                      # stack horizontally [a, b]
crop(grid, (i,j), (h,w)) -> Grid                     # extract subgrid
upscale(element, n) -> Grid/Object                   # enlarge by factor n
downscale(grid, n) -> Grid                           # shrink by factor n
horizontal_upscale(grid, n) -> Grid                  # widen by n
vertical_upscale(grid, n) -> Grid                    # height by n
horizontal_split(grid, n) -> Tuple                   # split into n vertical slabs
vertical_split(grid, n) -> Tuple                     # split into n horizontal slabs
top_half/bottom_half/left_half/right_half -> Grid    # halves
trim_border(grid) -> Grid                            # remove 1-cell border
smallest_subgrid_containing(patch, grid) -> Grid     # tight crop around patch
```

**Objects & Indices:**
```python
as_objects(grid, each_object_single_color, include_diagonal_neighbors, discard_background) -> FrozenSet[Object]
partition(grid) -> Objects                           # objects by color
partition_only_foreground(grid) -> Objects           # objects by color w/o background

color_filter(objects, color) -> Objects              # filter by color
size_filter(objects, n) -> FrozenSet                 # keep of size n
of_color(grid, color) -> Indices                     # indices of a color
to_object(patch, grid) -> Object                     # patch -> colored object
shift_to_origin(obj) -> Patch                        # move object to (0,0)
to_indices(patch_or_obj) -> Indices                  # indices only
as_indices(grid) -> Indices                          # all grid indices
as_object(grid) -> Object                            # grid -> object
recolor(color, patch) -> Object                      # recolor patch
shift_by_vector(patch, (di, dj)) -> Patch            # translate
paint_onto_grid(grid, obj) -> Grid                   # draw object
paint_onto_grid_background(grid, obj) -> Grid        # draw only onto background
fill(grid, color, patch) -> Grid                     # color cells in indices/patch
erase_patch(grid, patch) -> Grid                     # fill patch with background
move_object(grid, obj, offset) -> Grid               # move obj by offset
```

**Spatial Queries:**
```python
upper_left_corner/upper_right_corner/lower_left_corner/lower_right_corner(obj) -> IntegerTuple
center(patch) -> IntegerTuple
corner_indices(patch) -> Indices
position(a, b) -> IntegerTuple                       # relative (-1/0/1, -1/0/1)
box(patch) -> Indices                                # bounding-box outline
inbox(patch) -> Indices                              # outline 1 inside
outbox(patch) -> Indices                             # outline 1 outside
bounding_box_indices(patch) -> Indices               # all indices in box
bounding_box_delta(patch) -> Indices                 # box minus patch
vertical_line(loc) -> Indices                        # vertical line through loc
horizontal_line(loc) -> Indices                      # horizontal line through loc
shoot(start, direction) -> Indices                   # ray from point in direction
line_between(a, b) -> Indices                        # line between two points
adjacent(a, b) -> Boolean                            # patches touch
horizontal_matching(a, b) -> Boolean                 # share a row
vertical_matching(a, b) -> Boolean                   # share a column
manhattan_distance(a, b) -> Integer                  # min Manhattan distance
move_until_touching(src, dst) -> IntegerTuple        # offset to touch
```

**Shape/Geometry Predicates:**
```python
is_portrait(piece) -> Boolean
is_square(piece) -> Boolean
is_vertical_line(patch) -> Boolean
is_horizontal_line(patch) -> Boolean
```

**Properties:**
```python
get_height(grid_or_patch) -> Integer
get_width(grid_or_patch) -> Integer
get_shape(grid_or_patch) -> IntegerTuple
size(container) -> Integer
palette(element) -> FrozenSet[int]
count_colors(element) -> Integer
most_common_color(element) -> Integer
least_common_color(element) -> Integer
get_color(obj) -> Integer
color_at_location(grid, (i,j)) -> Integer/None
occurrences(grid, obj) -> Indices
cellwise(a, b, fallback) -> Grid
horizontal_periodicity(obj) -> Integer
vertical_periodicity(obj) -> Integer
solid_color_strips_in_grid(grid) -> Objects          # formerly frontiers
remove_solid_color_strips_from_grid(grid) -> Grid    # formerly compress
```

**Set Operations:**
```python
union(a, b) -> Container                             # concatenate/union
intersection(a, b) -> FrozenSet
difference(a, b) -> FrozenSet
flatten(containers) -> Container                     # flatten nested
remove_duplicates(t) -> Tuple
keep_if_condition(container, pred) -> Container
keep_if_condition_and_flatten(container, pred) -> FrozenSet
contains(value, container) -> Boolean
initset(value) -> FrozenSet
insert(value, frozenset) -> FrozenSet
remove(value, container) -> Container
to_tuple(frozenset) -> Tuple
cartesian_product(a, b) -> FrozenSet
pairwise(a, b) -> TupleTuple
as_tuple(a, b) -> IntegerTuple
as_generic_tuple(a, b) -> Tuple
make_cell(color, (i,j)) -> Tuple                     # (color, (i,j))
interval(start, stop, step) -> Tuple
```

**Aggregation:**
```python
argmax(container, f) -> Any
argmin(container, f) -> Any
valmax(container, f) -> Integer
valmin(container, f) -> Integer
maximum(container) -> Integer
minimum(container) -> Integer
most_common(container) -> Any
least_common(container) -> Any
sort(container, key_fn) -> Tuple
```

**Arithmetic & Vectors:**
```python
add(a, b) -> Numerical
subtract(a, b) -> Numerical
multiply(a, b) -> Numerical
divide(a, b) -> Numerical
negate(n) -> Numerical
double(n) -> Numerical
halve(n) -> Numerical
increment(x) -> Numerical
decrement(x) -> Numerical
sign(x) -> Numerical or IntegerTuple
to_vertical_vec(i) -> IntegerTuple
to_horizontal_vec(j) -> IntegerTuple
```

**Boolean:**
```python
is_equal(a, b) -> Boolean
logical_and(a, b) -> Boolean
logical_or(a, b) -> Boolean
logical_not(b) -> Boolean
contains(value, container) -> Boolean
is_positive(n) -> Boolean
is_even(n) -> Boolean
greater_than(a, b) -> Boolean
```

**Constants:**
```python
# Colors: ZERO=0, ONE=1, TWO=2, THREE=3, FOUR=4, FIVE=5, SIX=6, SEVEN=7, EIGHT=8, NINE=9
# Directions: UP=(-1,0), DOWN=(1,0), LEFT=(0,-1), RIGHT=(0,1)
# Diagonals: UNITY=(1,1), NEG_UNITY=(-1,-1), UP_RIGHT=(-1,1), DOWN_LEFT=(1,-1)
# Special: ORIGIN=(0,0), True=True, False=False
```

## Control Flow Allowed
- Single-level `for` loops (no nesting)
- `if/else` conditionals
- List comprehensions

## Requirements
- Function signature: `def solve(I):`
- Input `I` is tuple of tuples of ints (Grid)
- Return same format
- Use ONLY the DSL primitives listed above (arc-dsl-llm names)
- Add brief comments for clarity
- Prefer functional patterns (compose, chain, combine_two_function_results, fix_first_argument/fix_last_argument, transform, transform_and_flatten)
- You can adapt patterns from similar programs but adjust them to match the pattern description

## Output Format
```python
def solve(I):
    # [comment explaining step]
    x1 = some_dsl_function(I)

    # [comment]
    x2 = another_function(x1, args)

    # [final step]
    return O  # O is the output grid
```

Generate the solve function now:
"""