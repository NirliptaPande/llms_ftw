"""
Refactored VLM Prompter for ARC DSL-based solver with image support.
New pipeline: Program Similarity → Pattern Discovery → Code Generation
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from src.utils.render_legacy import grid_to_base64_png_oai_content


class VLMPrompter:
    """Builds prompts for the program-first ARC solving process"""
    
    def __init__(self,use_vision: bool = True):
        self.phase1_template = self._load_phase1_template()
        self.use_vision = use_vision
        # self.phase2_template = self._load_phase2_template()
    
    def build_phase1_prompt(self, 
                           task: Dict[str, Any],
                           similar_programs: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Build Phase 1 prompt: Natural Language Pattern Discovery
        Analyzes training examples AND similar programs to discover transformation pattern.
        
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
        
        # Format training examples (with images)
        content_blocks.extend(self._format_training_examples(task['train']))
        content_blocks.extend(self._format_test_examples(task['test']))

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
        
        # Add the analysis protocol template
        content_blocks.append({
            "type": "text",
            "text": self.phase1_template
        })
        
        return content_blocks
    
    def build_phase2_prompt(self,
                            task: Dict[str, Any], 
                           phase1_output: str,
                           similar_programs: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Build Phase 2 prompt: Code Generation
        Generates solve() function based on natural language pattern and similar programs.
        
        Args:
            task: Dict with 'train' key containing list of {'input': grid, 'output': grid}
            phase1_output: The natural language pattern description from Phase 1
            similar_programs: Same list of similar programs (for reference during coding)
            
        Returns:
            List of content blocks (dicts) for multimodal prompt
        """
        content_blocks = []
        
        # Add header
        content_blocks.append({
            "type": "text",
            "text": "# DSL Code Generator\nGenerate a Python `solve(I)` function using ONLY the DSL primitives below.\n\n"
        })
        
        # Add training examples section
        content_blocks.append({
            "type": "text",
            "text": "## Training Examples\n"
        })
        
        # Format training examples (with images)
        content_blocks.extend(self._format_training_examples(task['train']))
        
        # Extract the key sections from Phase 1 output
        extracted_pattern = self._extract_key_pattern_from_phase1(phase1_output)
        
        # Add pattern description
        content_blocks.append({
            "type": "text",
            "text": f"\n## Natural Language Pattern Description\n{extracted_pattern}\n"
        })
        
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
        
        # Add DSL primitives and the rest of the template
        content_blocks.append({
            "type": "text",
            "text": self._get_phase2_dsl_section()
        })
        
        return content_blocks
    
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
    
    def _format_training_examples(self, train_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format training examples as content blocks with images"""
        include_images = self.use_vision
        content_blocks = []
        content_blocks.append({
            "type": "text",
            "text": f"Below are {len(train_examples)} training examples follwed by the test example(s) you have to generalize to, for each example, the input grid is shown first, followed by the output grid. \n"

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
    
    def _format_test_examples(self, test_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format test examples as content blocks with images (input only, no output)"""
        include_images = self.use_vision
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

    def _format_grid(self, grid: Tuple[Tuple[int]], separator: str = "|") -> str:
        return "\n".join(separator.join(str(cell) for cell in row) for row in grid)
    
    def _format_similar_programs(self, similar_programs: List[Dict[str, Any]]) -> str:
        """Format similar programs for Phase 1 (natural language discovery)"""
        lines = []
        lines.append("The following programs may be useful to solve the current tasks, feel free to use parts of each program, and/or combine them. Most importanly, use them as guidance:\n")
        
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
    
    def _load_phase1_template(self) -> str:
        return r"""
## Analysis Protocol

You will analyze these examples and the test sample systematically, allowing your hypothesis to evolve 
naturally as you see more data - like a human solving a puzzle.

**Core principle:** Look for DIFFERENCES within each example (input→output changes)
and SIMILARITIES across all examples (the consistent pattern).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Work through examples sequentially, reasoning in plain English about what you observe.

Step 1.1: First Example Analysis
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

Step 1.2: Second Example Validation
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

Step 1.3: Third Example Confirmation
Given: Input 3 → Output 3

<observation_3>
- Does hypothesis_2 work here?
- Any new edge cases or variations?
</observation_3>

<hypothesis_3>
Final refined hypothesis in natural language.
</hypothesis_3>

Step 1.N: Additional Examples
Continue for all remaining examples...

Step 2.1: First Test Example Confirmation
Given: Test Input 1
<observation_test1>
- Does hypothesis_training work here?
- Any new edge cases or variations?
</observation_test1>

<hypothesis_test1>
Test hypothesis after seeing test example 1.
</hypothesis_test1>

Step 2.N: Additional Test Examples, if present
Continue for all remaining test cases...

Step 2.Final: Pattern Synthesis
<pattern_summary>
In plain English, the transformation rule is:
- [First operation in natural language]
- [Second operation in natural language]
- [Any conditions or special cases]

Edge cases to consider:
- [Any variations you noticed]

Why this works:
- [Brief explanation about WHY this pattern makes sense]
</pattern_summary>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Begin your analysis:"""
    
    def _get_phase2_dsl_section(self) -> str:
        """Phase 2: DSL Primitives and Code Generation Instructions"""
        return """
        
## DSL Primitives

**Type Definitions:**
Grid: Tuple[Tuple[int, ...], ...] – immutable 2D array  
Object: FrozenSet[(int, (int, int))] – set of (color, location) pairs  
Patch: FrozenSet[(int, int)] or Object – set of indices or colored object  
Indices: FrozenSet[(int, int)] – set of (row, col) positions  
Objects: FrozenSet[Object] – set of objects  
Element: Grid or Object – generic colored structure  
Piece: Grid, Patch or Object – anything with a bounding box  
Container: Tuple or FrozenSet – generic container type  
ColorSet: FrozenSet[int] – set of colors  
IntegerTuple: (int, int) – usually coordinates or dimensions  
Callable: Function type  

---

### Functional Programming

identity(x) -> Any                          # return x unchanged
compose(outer, inner) -> Callable           # outer(inner(x)) - compose 2 functions
chain(h, g, f) -> Callable                  # h(g(f(x))) - compose 3 functions
equals(f, target) -> Callable               # predicate: f(x) == target
fix_first_argument(f, a) -> Callable        # partially apply first argument
fix_last_argument(f, b) -> Callable         # partially apply last argument
power(f, n) -> Callable                     # apply f n times to its argument
combine_two_function_results(o, f, g)       # x ↦ o(f(x), g(x)) - fork/combiner
transform(f, container) -> Container        # map f over container
apply_each_function(funcs, value) -> Cont   # apply each function to same value
transform_and_flatten(f, nested) -> FrozenSet # map f then flatten nested containers
transform_both(f, a, b) -> Tuple            # pairwise map over two tuples
transform_both_and_flatten(f, a, b) -> Tuple # pairwise map then flatten
apply_function_on_cartesian_product(f, a, b) -> FrozenSet # apply f to all pairs from a×b

---

### Arithmetic & Vectors

add(a, b) -> Numerical                      # add ints or (i,j) tuples elementwise
subtract(a, b) -> Numerical                 # subtract elementwise
multiply(a, b) -> Numerical                 # multiply elementwise
divide(a, b) -> Numerical                   # floor-division elementwise
double(n) -> Numerical                      # multiply by 2
halve(n) -> Numerical                       # divide by 2 (floor)
negate(n) -> Numerical                      # additive inverse (−n)
increment(x) -> Numerical                   # add 1 to each component
decrement(x) -> Numerical                   # subtract 1 from each component
crement(x) -> Numerical                     # move each component one step toward 0
sign(x) -> Numerical                        # componentwise sign: −1/0/1
to_vertical_vec(i) -> IntegerTuple          # (i, 0)
to_horizontal_vec(j) -> IntegerTuple        # (0, j)

---

### Boolean & Comparisons

is_even(n) -> Boolean                       # n % 2 == 0
is_positive(x) -> Boolean                   # x > 0
logical_not(b) -> Boolean                   # not b
logical_and(a, b) -> Boolean                # a and b
logical_or(a, b) -> Boolean                 # a or b
is_equal(a, b) -> Boolean                   # a == b
greater_than(a, b) -> Boolean               # a > b
contains(value, container) -> Boolean       # membership test
condition_if_else(cond, a, b) -> Any        # a if cond else b

---

### Set / Container Operations

union(a, b) -> Container                    # concatenation/union of same-type containers
intersection(a, b) -> FrozenSet             # set intersection
difference(a, b) -> FrozenSet               # elements in a but not in b
flatten(containers) -> Container            # flatten one level of nesting
remove_duplicates(tup) -> Tuple             # remove duplicates, preserve order
keep_if_condition(container, pred) -> Cont  # filter elements by predicate
keep_if_condition_and_flatten(container, pred) -> FrozenSet # filter then flatten results
extract_first_matching(container, pred) -> Any # first element satisfying predicate
initset(value) -> FrozenSet                 # singleton frozenset
insert(value, s) -> FrozenSet               # add value to frozenset
remove(value, container) -> Container       # remove all occurrences of value
get_first(container) -> Any                 # first element (iteration order)
get_last(container) -> Any                  # last element (iteration order)
get_other(container, value) -> Any          # the other element in 2-element container
to_tuple(fset) -> Tuple                     # cast frozenset to tuple
cartesian_product(a, b) -> FrozenSet        # all pairs (x,y) in a×b
pairwise(a, b) -> Tuple                     # zip two tuples into pairs

---

### Aggregation

size(container) -> Integer                  # number of elements
maximum(container) -> Integer               # max value (0 if empty)
minimum(container) -> Integer               # min value (0 if empty)
valmax(container, key) -> Integer           # max key(item) value
valmin(container, key) -> Integer           # min key(item) value
argmax(container, key) -> Any               # item with largest key(item)
argmin(container, key) -> Any               # item with smallest key(item)
most_common(container) -> Any               # most frequent element
least_common(container) -> Any              # least frequent element
sort(container, key) -> Tuple               # sort by key function
most_common_color(elem) -> Color            # most frequent color in grid/object
least_common_color(elem) -> Color           # least frequent color in grid/object

---

### Tuple & Cell Constructors

as_generic_tuple(a, b) -> Tuple             # generic pair (a, b)
as_tuple(i, j) -> IntegerTuple              # coordinate pair (i, j)
make_cell(color, loc) -> (int, (int,int))   # single colored cell as (color, loc)
repeat(item, times) -> Tuple                # item repeated times times
interval(start, stop, step) -> Tuple        # integer range as tuple

---

### Color & Palette Operations

palette(elem) -> ColorSet                   # set of colors in grid or object
count_colors(elem) -> Integer               # number of distinct colors
color_count(elem, color) -> Integer         # count of given color
get_color(obj) -> Color                     # color of single-color object
color_filter(objs, color) -> Objects        # keep objects with given color
size_filter(container, n) -> FrozenSet      # keep items of size n

---

### Grid / Object Indexing & Conversion

as_indices(grid) -> Indices                 # all (i,j) positions in grid
of_color(grid, color) -> Indices            # positions with given color
to_indices(patch) -> Indices                # drop colors, keep locations
recolor(color, patch) -> Object             # patch with all cells set to color
to_object(patch, grid) -> Object            # colored object from patch and grid
as_object(grid) -> Object                   # entire grid as colored object
color_at_location(grid, loc) -> Color|None  # color at (i,j) or None if out of bounds

---

### Geometry & Spatial Queries

upper_left_corner(patch) -> IntegerTuple    # (min_row, min_col)
upper_right_corner(patch) -> IntegerTuple   # (min_row, max_col)
lower_left_corner(patch) -> IntegerTuple    # (max_row, min_col)
lower_right_corner(patch) -> IntegerTuple   # (max_row, max_col)
uppermost(patch) -> Integer                 # smallest row index
lowermost(patch) -> Integer                 # largest row index
leftmost(patch) -> Integer                  # smallest column index
rightmost(patch) -> Integer                 # largest column index
get_height(piece) -> Integer                # height of grid/patch/object
get_width(piece) -> Integer                 # width of grid/patch/object
get_shape(piece) -> IntegerTuple            # (height, width)
is_portrait(piece) -> Boolean               # height > width
is_square(piece) -> Boolean                 # square footprint
center(patch) -> IntegerTuple               # geometric mid-point of bounding box
centerofmass(patch) -> IntegerTuple         # average of cell positions
position(a, b) -> IntegerTuple              # coarse relative direction (−1/0/1, −1/0/1)
is_vertical_line(patch) -> Boolean          # 1 column, contiguous
is_horizontal_line(patch) -> Boolean        # 1 row, contiguous
horizontal_matching(a, b) -> Boolean        # share some row index
vertical_matching(a, b) -> Boolean          # share some column index
manhattan_distance(a, b) -> Integer         # minimum L1 distance between patches
adjacent(a, b) -> Boolean                   # manhattan_distance == 1
bordering(patch, grid) -> Boolean           # touches outer border of grid

---

### Neighborhoods & Connectivity

direct_neighbors(loc) -> Indices            # 4-connected neighbors (up,down,left,right)
diagonal_neighbors(loc) -> Indices          # 4 diagonal neighbors
neighbors(loc) -> Indices                   # 8-connected neighborhood
as_objects(grid, single_color, diag, no_bg) -> Objects # connected components
partition(grid) -> Objects                  # one object per color (include background)
partition_only_foreground(grid) -> Objects  # one object per non-background color

---

### Bounding Boxes & Lines

bounding_box_indices(patch) -> Indices      # all cells in bounding box
bounding_box_delta(patch) -> Indices        # box cells not in patch
inbox(patch) -> Indices                     # box 1 cell inside bounding box
outbox(patch) -> Indices                    # box 1 cell outside bounding box
box(patch) -> Indices                       # outline along bounding box edges
corner_indices(patch) -> Indices            # set of all four corner coordinates
line_between(a, b) -> Indices               # straight line between points
shoot(start, direction) -> Indices          # long ray from start in given direction
vertical_line(loc) -> Indices               # vertical line through loc
horizontal_line(loc) -> Indices             # horizontal line through loc
move_until_touching(src, dst) -> IntegerTuple # offset to move src until adjacent to dst

---

### Grid Transforms

rot90(grid) -> Grid                         # rotate 90° clockwise
rot180(grid) -> Grid                        # rotate 180°
rot270(grid) -> Grid                        # rotate 270° clockwise
horizontal_mirror(piece) -> Piece           # flip along horizontal axis
vertical_mirror(piece) -> Piece             # flip along vertical axis
diagonal_mirror(piece) -> Piece             # mirror along main diagonal
counterdiagonal_mirror(piece) -> Piece      # mirror along anti-diagonal

---

### Grid Composition & Resizing

horizontal_concat(a, b) -> Grid             # [a | b] – concat columns
vertical_concat(a, b) -> Grid               # stack grids vertically
horizontal_upscale(grid, n) -> Grid         # stretch columns by factor n
vertical_upscale(grid, n) -> Grid           # stretch rows by factor n
upscale(element, n) -> Element              # isotropic upscale grid or object
downscale(grid, n) -> Grid                  # sample every n-th row/column
smallest_subgrid_containing(patch, grid) -> Grid # tight crop around patch
horizontal_split(grid, n) -> Tuple[Grid]    # split into n vertical slices
vertical_split(grid, n) -> Tuple[Grid]      # split into n horizontal slices
top_half(grid) -> Grid                      # upper half
bottom_half(grid) -> Grid                   # lower half
left_half(grid) -> Grid                     # left half
right_half(grid) -> Grid                    # right half
crop(grid, start, dims) -> Grid		    # extract rectangular region from grid starting at start with size dims


---

### Painting & Editing Grids

fill(grid, color, patch) -> Grid            # set all cells in patch to color
fill_background(grid, color, patch) -> Grid # fill only where grid has background
paint_onto_grid(grid, obj) -> Grid          # draw colored object onto grid
paint_onto_grid_background(grid, obj) -> Grid # paint only on background cells
replace(grid, old, new) -> Grid             # change all old color to new
switch(grid, a, b) -> Grid                  # swap two colors everywhere
erase_patch(grid, patch) -> Grid            # overwrite patch with background color
trim_border(grid) -> Grid                   # remove 1-cell border
move_object(grid, obj, offset) -> Grid      # erase obj and redraw at offset
shift_by_vector(patch, offset) -> Grid	    # shift all cells of patch by offset vector
shift_to_origin(patch) -> Grid		    # translate patch so that top-left filled cell moves to origin

---

### Grid Construction & Matching

create_grid(color, (h, w)) -> Grid          # blank grid of size (h,w) with color
cellwise(a, b, fallback) -> Grid            # keep matching cells, else fallback color
occurrences(grid, obj) -> Indices           # top-left positions where obj occurs
solid_color_strips_in_grid(grid) -> Objects # uniform rows/columns as objects
remove_solid_color_strips_from_grid(grid) -> Grid # delete uniform rows/columns
horizontal_periodicity(obj) -> Integer      # horizontal repetition period
vertical_periodicity(obj) -> Integer        # vertical repetition period

"""