"""
Simple VLM Prompter for ARC DSL-based solver.
Two-phase approach: Pattern Discovery → Code Generation
"""

from typing import List, Tuple, Dict, Any


class VLMPrompter:
    """Builds prompts for the two-phase ARC solving process"""
    
    def __init__(self):
        self.phase1_template = self._load_phase1_template()
        self.phase2_template = self._load_phase2_template()
    
    def build_phase1_prompt(self, task: Dict[str, Any]) -> str:
        """
        Build Phase 1 prompt: Pattern Discovery
        Analyzes training examples to discover transformation pattern.
        
        Args:
            task: Dict with 'train' key containing list of {'input': grid, 'output': grid}
            
        Returns:
            Complete prompt string for pattern analysis
        """
        # Format training examples
        examples_str = self._format_training_examples(task['train'])
        
        # Insert into template
        prompt = self.phase1_template.replace('[INSERT TASK EXAMPLES HERE]', examples_str)
        
        return prompt
    
    def build_phase2_prompt(self, 
                           phase1_output: str,
                           similar_programs: List[Dict[str, Any]] = None) -> str:
        """
        Build Phase 2 prompt: Code Generation
        Generates solve() function based on discovered pattern.
        
        Args:
            phase1_output: The pattern analysis from Phase 1
            similar_programs: Optional list of similar programs from library
                             Each dict has: {'program': str, 'similarity': float, 'functions': list}
            
        Returns:
            Complete prompt string for code generation
        """
        # Insert Phase 1 output
        prompt = self.phase2_template.replace('[PHASE 1 OUTPUT HERE]', phase1_output)
        
        # Insert similar programs if available
        if similar_programs:
            similar_str = self._format_similar_programs(similar_programs)
        else:
            similar_str = "[No similar programs found in library]"
        
        prompt = prompt.replace('[IF LIBRARY HAS MATCHES:]', similar_str)
        
        return prompt
    
    def _format_training_examples(self, train_examples: List[Dict[str, Any]]) -> str:
        """Format training examples for Phase 1 prompt"""
        lines = []
        
        for idx, example in enumerate(train_examples, 1):
            lines.append(f"Example {idx}:")
            lines.append("Input:")
            lines.append(self._format_grid(example['input']))
            lines.append("Output:")
            lines.append(self._format_grid(example['output']))
            lines.append("")  # Blank line between examples
        
        return "\n".join(lines)
    
    def _format_grid(self, grid: Tuple[Tuple[int]]) -> str:
        """Format grid as readable ASCII"""
        return "\n".join("".join(str(cell) for cell in row) for row in grid)
    
    def _format_similar_programs(self, similar_programs: List[Dict[str, Any]]) -> str:
        """Format similar programs from library"""
        lines = []
        
        for prog in similar_programs:
            similarity = prog.get('similarity', 0.0)
            program_code = prog.get('program', '')
            shared_funcs = prog.get('functions', [])
            
            lines.append(f"```python")
            lines.append(f"# Similar program (similarity: {similarity:.2f})")
            lines.append(f"# Shared functions: {{{', '.join(shared_funcs)}}}")
            lines.append(program_code)
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)
    
    def _load_phase1_template(self) -> str:
        """Phase 1: Pattern Discovery Template"""
        return r"""# ARC Pattern Discovery
Analyze training examples sequentially to discover the transformation pattern.

## DSL Functions Reference

**Functional Programming:**
```
compose(f, g)          # f(g(x))
chain(f, g, h, ...)    # f(g(h(...(x))))
fork(combiner, f, g)   # combiner(f(x), g(x))
apply(f, container)    # apply f to each element
mapply(f, container)   # map and merge results
lbind(f, arg)          # partial application: f(arg, x)
rbind(f, arg)          # partial application: f(x, arg)
```

**Grid Transforms:**
```
hmirror(grid)          # horizontal flip
vmirror(grid)          # vertical flip
dmirror(grid)          # diagonal \ flip
cmirror(grid)          # diagonal / flip
rot90/rot180/rot270    # rotations
vconcat(a, b)          # vertical stack
hconcat(a, b)          # horizontal stack
crop(grid, loc, dims)  # extract subgrid
upscale(grid, n)       # scale up
downscale(grid, n)     # scale down
```

**Objects:**
```
objects(grid, univalued, diagonal, without_bg)
  # Find connected components
  # univalued: T = single-color, F = multicolor
  # diagonal: T = 8-connected, F = 4-connected  
  # without_bg: T = ignore most common color
colorfilter(objs, color)   # keep only specified color
sizefilter(objs, n)        # keep only size n objects
ofcolor(grid, color)       # indices of color
toobject(patch, grid)      # patch to object
normalize(obj)             # move to origin
toindices(obj/grid)        # extract indices
asindices(grid)            # all non-background indices
```

**Modifications:**
```
fill(grid, color, indices) # color specified cells
paint(grid, obj)           # draw object
replace(grid, old, new)    # swap colors
switch(grid, a, b)         # swap two colors
move(grid, obj, offset)    # move object
shift(patch, offset)       # translate patch
cover(grid, patch)         # remove object
```

**Spatial:**
```
ulcorner/urcorner/llcorner/lrcorner  # corners
center(patch)              # center point
corners(patch)             # all 4 corners
position(a, b)             # relative position
box(patch)                 # outline
inbox/outbox(patch)        # inner/outer box
backdrop(patch)            # bounding box fill
delta(patch)               # bounding box - patch
vfrontier/hfrontier(loc)   # infinite lines
shoot(start, direction)    # ray from point
connect(a, b)              # line between points
```

**Queries:**
```
height/width(grid)         # dimensions
shape(obj/grid)            # (height, width)
size(container)            # length/cardinality
palette(grid)              # set of colors
mostcolor/leastcolor       # most/least common
color(obj)                 # object's color
index(grid, loc)           # value at location
occurrences(grid, obj)     # where obj appears
```

**Set Operations:**
```
combine(a, b)              # union
intersection(a, b)         # intersection
difference(a, b)           # set difference
merge(containers)          # flatten
dedupe(tuple)              # remove duplicates
sfilter(container, pred)   # filter by predicate
extract(container, pred)   # first match
contained(val, container)  # membership
```

**Aggregation:**
```
argmax/argmin(container, f)    # item with max/min f(item)
valmax/valmin(container, f)    # max/min value of f
maximum/minimum(container)     # max/min value
mostcommon/leastcommon         # frequency
order(container, key)          # sort by key
```

**Arithmetic:**
```
add/subtract/multiply/divide   # operations on ints/tuples
invert(x)                      # negation
double/halve(x)                # scale by 2 or 1/2
increment/decrement(x)         # +1/-1
sign(x)                        # -1/0/1
toivec(i)/tojvec(j)            # to directional vector
```

**Constructors:**
```
canvas(color, (h, w))      # blank grid
initset(value)             # singleton set
astuple(a, b, ...)         # create tuple
repeat(item, n)            # repeat n times
```

**Boolean:**
```
equality(a, b)             # ==
both/either(a, b)          # and/or
flip(b)                    # not
positive/even(n)           # predicates
greater(a, b)              # >
```

**Advanced:**
```
matcher(f, container)      # element where f matches
gravitate(src, dst)        # movement vector
compress(grid)             # remove uniform rows/cols
frontiers(grid)            # uniform rows/cols
hperiod/vperiod(obj)       # periodicity
cellwise(a, b, fallback)   # compare grids
hsplit/vsplit(grid, n)     # split grid
tophalf/bottomhalf         # halve grid
lefthalf/righthalf         # halve grid
trim(grid)                 # remove border
```

## Common Patterns
```
# Mirror and stack
x1 = hmirror(I)
O = vconcat(I, x1)

# Find largest object
objs = objects(I, T, F, T)
largest = argmax(objs, size)

# Functional composition
x1 = rbind(upscale, TWO)       # upscale by 2
x2 = compose(x1, vmirror)      # upscale then mirror
x3 = chain(invert, halve, size) # size then halve then invert

# Fork pattern
x1 = fork(combine, hfrontier, vfrontier)  # combine results of two functions

# Object manipulation with mapping
objs = objects(I, T, F, T)
x1 = lbind(shift, (1, 0))      # shift by (1,0)
shifted = mapply(x1, objs)      # apply to all objects

# Normalize and match
x1 = apply(normalize, objs)     # normalize all
x2 = lbind(matcher, normalize)  # create matcher function
```

## Analysis Format
For each example, write:
```
<example_N>
INPUT: (h,w), colors [...]
OUTPUT: (h',w'), colors [...]
OBS: [what changed - be specific]
HYP: [DSL operations to achieve this]
</example_N>
```

After all examples:
```
<final_pattern>
OPS:
x1 = func1(I)
x2 = func2(x1, ...)
O = func3(x2)
LOGIC: [one-line summary]
CONDITIONS: [for loops, if statements, or "none"]
</final_pattern>
```

## Rules
- Use DSL function names exactly as listed
- Keep observations concise but precise
- Update hypothesis if pattern changes
- Final pattern must be implementable

Now analyze this task:

**Training Examples:**
[INSERT TASK EXAMPLES HERE]"""
    
    def _load_phase2_template(self) -> str:
        """Phase 2: Code Generator Template"""
        return """# DSL Code Generator
Generate a Python `solve(I)` function using ONLY the DSL primitives below.

## DSL Primitives

**Functional Programming:**
```python
compose(f, g)          # f(g(x)) - function composition
chain(f, g, h, ...)    # f(g(h(...(x)))) - chain multiple functions
fork(combiner, f, g)   # combiner(f(x), g(x)) - apply two functions and combine
apply(f, container)    # apply f to each element
mapply(f, container)   # map f over container and merge results
lbind(f, arg)          # f(arg, x) - left partial application
rbind(f, arg)          # f(x, arg) - right partial application
identity(x)            # returns x unchanged
matcher(f, container)  # find element in container where f matches
extract(container, f)  # first element satisfying predicate f
```

**Grid Transforms:**
```python
hmirror(grid)          # flip left-right
vmirror(grid)          # flip up-down
dmirror(grid)          # diagonal \\ mirror
cmirror(grid)          # diagonal / mirror
rot90(grid)            # rotate 90° clockwise
rot180(grid)           # rotate 180°
rot270(grid)           # rotate 270° clockwise
vconcat(a, b)          # stack vertically [a; b]
hconcat(a, b)          # stack horizontally [a, b]
crop(grid, (i,j), (h,w))  # extract subgrid starting at (i,j) with dimensions (h,w)
upscale(grid, n)       # enlarge by factor n
downscale(grid, n)     # shrink by factor n
hsplit(grid, n)        # split horizontally into n parts
vsplit(grid, n)        # split vertically into n parts
tophalf/bottomhalf     # get top/bottom half
lefthalf/righthalf     # get left/right half
trim(grid)             # remove border
```

**Objects:**
```python
objects(grid, univalued, diagonal, without_bg)
# Find connected components (separate shapes/regions)
# univalued: T = single-color objects, F = multicolor allowed
# diagonal: T = diagonal connects (8-connected), F = only orthogonal (4-connected)
# without_bg: T = ignore background (most common color)
# Returns: frozenset of objects (each object is frozenset of (color, (i,j)) tuples)
# Example: objects(I, T, F, T) finds separate single-colored regions

colorfilter(objects, color)
# Keep only objects of specified color
# Example: colorfilter(objs, BLUE) → only blue objects

sizefilter(objects, n)
# Keep only objects with exactly n cells
# Example: sizefilter(objs, 4) → only 4-cell objects

ofcolor(grid, color)   # get all indices of specified color
toobject(patch, grid)  # convert patch to object with colors
normalize(obj)         # move object to origin (0,0)
toindices(obj)         # extract just the (i,j) positions from object
asindices(grid)        # all non-background cell positions
```

**Modifications:**
```python
fill(grid, color, patch)    # color cells in patch/indices
paint(grid, obj)            # draw object onto grid
replace(grid, old, new)     # swap all instances of old color with new
switch(grid, a, b)          # swap two colors
move(grid, obj, offset)     # move object by offset
shift(patch, (di, dj))      # translate patch by offset
cover(grid, patch)          # remove object (fill with background)
```

**Spatial Queries:**
```python
ulcorner(obj)          # upper-left corner (min_i, min_j)
urcorner(obj)          # upper-right corner
llcorner(obj)          # lower-left corner
lrcorner(obj)          # lower-right corner (max_i, max_j)
center(patch)          # center point of patch
corners(patch)         # all 4 corner positions
position(a, b)         # relative position between patches
box(patch)             # outline of bounding box
inbox(patch)           # interior outline of box
outbox(patch)          # exterior outline of box
backdrop(patch)        # all indices in bounding box
delta(patch)           # bounding box minus the patch
vfrontier(loc)         # vertical line through location
hfrontier(loc)         # horizontal line through location
shoot(start, direction)# ray from point in direction
connect(a, b)          # line between two points
```

**Properties:**
```python
height(grid)           # number of rows
width(grid)            # number of columns
shape(grid/obj)        # (height, width) tuple
size(container)        # number of elements
palette(grid)          # set of colors used
mostcolor(grid)        # most frequent color (background)
leastcolor(grid)       # least frequent color
color(obj)             # color of single-color object
index(grid, (i, j))    # value at location
occurrences(grid, obj) # locations where obj appears
compress(grid)         # remove uniform rows/columns
frontiers(grid)        # get uniform rows/columns
hperiod(obj)           # horizontal period
vperiod(obj)           # vertical period
gravitate(src, dst)    # offset to move src next to dst
```

**Set Operations:**
```python
combine(a, b)          # union of containers
intersection(a, b)     # intersection of sets
difference(a, b)       # a - b (set difference)
merge(containers)      # flatten container of containers
dedupe(tuple)          # remove duplicates
sfilter(container, f)  # filter by predicate
mfilter(container, f)  # filter and merge
contained(val, set)    # membership test (val in set)
initset(value)         # create singleton frozenset
```

**Aggregation:**
```python
argmax(container, f)   # item with maximum f(item)
argmin(container, f)   # item with minimum f(item)
valmax(container, f)   # maximum value of f over container
valmin(container, f)   # minimum value of f over container
maximum(container)     # max element
minimum(container)     # min element
mostcommon(container)  # most frequent element
leastcommon(container) # least frequent element
order(container, key)  # sort by key function
```

**Arithmetic:**
```python
add(a, b)              # addition (works on ints and tuples)
subtract(a, b)         # subtraction
multiply(a, b)         # multiplication
divide(a, b)           # floor division
invert(n)              # negation
double(n)              # multiply by 2
halve(n)               # divide by 2
increment(x)           # add 1
decrement(x)           # subtract 1
sign(x)                # -1, 0, or 1
toivec(i)              # (i, 0) - vertical vector
tojvec(j)              # (0, j) - horizontal vector
```

**Constructors:**
```python
canvas(color, (height, width))
# Create blank grid filled with one color
# Example: canvas(ZERO, (5,5)) → 5×5 grid of zeros

astuple(a, b, ...)     # create tuple from elements
repeat(item, n)        # tuple with item repeated n times
```

**Boolean:**
```python
equality(a, b)         # a == b
both(a, b)             # a and b
either(a, b)           # a or b
flip(b)                # not b
contained(val, set)    # val in set
positive(n)            # n > 0
even(n)                # n % 2 == 0
greater(a, b)          # a > b
```

**Advanced:**
```python
cellwise(a, b, fallback)   # compare grids cell-by-cell
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

## Input Pattern
```
[PHASE 1 OUTPUT HERE]
```

## Similar Programs for Reference
[IF LIBRARY HAS MATCHES:]

## Requirements
- Function signature: `def solve(I):`
- Input `I` is tuple of tuples of ints (Grid)
- Return same format
- Use ONLY DSL primitives listed above
- Add brief comments for clarity
- No raw loops over cells - use DSL functions
- Use functional programming when appropriate:
  * compose/chain for sequential operations
  * fork for parallel operations that combine
  * lbind/rbind for partial application
  * mapply for operations over collections

## Output Format
```python
def solve(I):
    # [comment explaining step]
    x1 = dsl_function(I)
    
    # [comment with functional composition]
    x2 = compose(func_a, func_b)
    x3 = x2(x1)
    
    # [comment]
    x4 = mapply(some_func, x3)
    
    return x4
```

## Example with Functional Programming
```python
def solve(I):
    # Extract objects
    x1 = objects(I, T, F, T)
    
    # Create upscaler and apply with mirror
    x2 = rbind(upscale, TWO)  # partial: upscale(_, 2)
    x3 = compose(x2, vmirror)  # upscale then mirror
    
    # Get shape information with chain
    x4 = chain(invert, halve, shape)  # shape -> halve -> invert
    
    # Fork to get both frontiers
    x5 = fork(combine, hfrontier, vfrontier)  # combine both results
    
    # Apply to all objects
    x6 = mapply(x3, x1)
    
    return paint(I, x6)
```

Generate the solve function now:"""


# Quick test
if __name__ == "__main__":
    prompter = VLMPrompter()
    
    # Mock task
    task = {
        'train': [
            {
                'input': ((1, 2), (3, 4)),
                'output': ((1, 2), (3, 4), (4, 3), (2, 1))
            },
            {
                'input': ((5, 6, 7), (8, 9, 0)),
                'output': ((5, 6, 7), (8, 9, 0), (0, 9, 8), (7, 6, 5))
            }
        ]
    }
    
    # Test Phase 1
    print("=== PHASE 1 PROMPT ===")
    phase1 = prompter.build_phase1_prompt(task)
    print(phase1[:500] + "...\n")
    
    # Test Phase 2
    print("=== PHASE 2 PROMPT ===")
    mock_phase1_output = """<final_pattern>
SIZE: (h,w) → (2h,w)
OPS:
x1 = hmirror(I)
O = vconcat(I, x1)
LOGIC: stack horizontally mirrored version below original
CONDITIONS: none
</final_pattern>"""
    
    similar = [
        {
            'program': 'def solve(I):\n    x1 = vmirror(I)\n    return vconcat(I, x1)',
            'similarity': 0.85,
            'functions': ['vmirror', 'vconcat']
        }
    ]
    
    phase2 = prompter.build_phase2_prompt(mock_phase1_output, similar)
    print(phase2[:500] + "...")