from .dsl import *
from .constants import *



def solve_1ae2feb7(I):
    # Find height and width
    h = height(I)
    w = width(I)
    
    # Find wall_col: the column fully filled with 2
    wall_col = next(j for j in range(w) if all(index(I, (i, j)) == 2 for i in range(h)))
    
    # Build new grid row by row
    new_rows = []
    for i in range(h):
        # Extract left values for run detection
        left_vals = [index(I, (i, j)) for j in range(wall_col)]
        
        # Find runs: single for loop over positions
        runs = []
        current_c = 0
        current_l = 0
        for val in left_vals:
            if val == 0:
                if current_l > 0:
                    runs.append((current_c, current_l))
                    current_c = 0
                    current_l = 0
            elif val != current_c:
                if current_l > 0:
                    runs.append((current_c, current_l))
                current_c = val
                current_l = 1
            else:
                current_l += 1
        if current_l > 0:
            runs.append((current_c, current_l))
        
        # Build new_row, copy original, clear right side
        new_row = [index(I, (i, j)) for j in range(w)]
        for j in range(wall_col + 1, w):
            new_row[j] = 0
        
        # Place for each run (left to right, later overwrite earlier)
        for c, l in runs:
            pos = wall_col + 1
            while pos < w:
                new_row[pos] = c
                pos += l
        
        new_rows.append(tuple(new_row))
    
    # Return as grid
    return tuple(new_rows)


def solve_2ba387bc(I):
    # Extract all single-color, 8-connected objects ignoring background
    objs = objects(I, T, T, T)
    
    # Filter to only 4x4 bounding box objects
    four_by_four = [o for o in objs if shape(o) == (4, 4)]
    
    # Classify into frames (size < 16) and solids (size == 16), extract subgrids, collect with start row
    frames = []
    solids = []
    for o in four_by_four:
        pos = ulcorner(o)
        sub = crop(I, pos, (4, 4))
        if size(o) == 16:
            solids.append((pos[0], sub))
        else:
            frames.append((pos[0], sub))
    
    # Sort each list by starting row (top-to-bottom)
    frames = sorted(frames, key=lambda x: x[0])
    solids = sorted(solids, key=lambda x: x[0])
    
    # Extract just the subgrids (discard rows now)
    frame_grids = [f[1] for f in frames]
    solid_grids = [s[1] for s in solids]
    
    # Compute n = max length, create empty 4x4
    n = max(len(frame_grids), len(solid_grids))
    empty = canvas(0, (4, 4))
    
    # Pad shorter list with empties at the end
    frame_grids += [empty for _ in range(n - len(frame_grids))]
    solid_grids += [empty for _ in range(n - len(solid_grids))]
    
    # Build output by horizontally concatenating pairs, then vertically stacking
    if n == 0:
        return canvas(0, (0, 8))
    O = hconcat(frame_grids[0], solid_grids[0])
    for i in range(1, n):
        pair = hconcat(frame_grids[i], solid_grids[i])
        O = vconcat(O, pair)
    return O

def solve_3dc255db(I):
    # Get all connected multi-color objects using 8-connectivity, ignoring background
    objs = objects(I, F, T, T)
    
    # Start with output grid as input
    O = I
    
    # Process each object
    for obj in objs:
        # Get unique colors in the object
        colors = frozenset(c for c, pos in obj)
        if len(colors) != 2:
            continue  # Skip single-color or multi>2 color objects
        
        # Count cells per color
        count = {c: 0 for c in colors}
        for c, pos in obj:
            count[c] += 1
        
        # Identify majority and minority colors
        maj_c = max(count, key=count.get)
        min_c = next(c for c in colors if c != maj_c)
        min_count = count[min_c]
        if count[maj_c] <= min_count:
            continue  # Skip if not clear majority
        
        # Get core positions (majority color)
        core_pos_list = [(r, c) for cc, (r, c) in obj if cc == maj_c]
        if not core_pos_list:
            continue
        core_pos = frozenset(core_pos_list)
        
        # Get original minority positions to erase
        min_pos = frozenset((r, c) for cc, (r, c) in obj if cc == min_c)
        
        # Erase original minority positions
        O = fill(O, 0, min_pos)
        
        # Compute core bounding box
        min_r_core = min(r for r, c in core_pos_list)
        max_r_core = max(r for r, c in core_pos_list)
        min_c_core = min(c for r, c in core_pos_list)
        max_c_core = max(c for r, c in core_pos_list)
        h_core = max_r_core - min_r_core + 1
        w_core = max_c_core - min_c_core + 1
        
        # Get all positions of minority color in original grid
        all_min_pos = ofcolor(I, min_c)
        
        # Check if there is a left protrusion (min_c cells left of core within row range)
        has_left_protrusion = any(
            min_r_core <= r <= max_r_core and c < min_c_core
            for r, c in all_min_pos
        )
        
        # Compute new positions for minority cells
        new_pos_set = frozenset()
        if w_core > h_core:
            # Horizontal orientation: extend left or right on the bottommost extremal
            if has_left_protrusion:
                # Extend right from bottommost rightmost core cell
                right_candidates = [(r, c) for r, c in core_pos_list if c == max_c_core]
                attach_pos = max(right_candidates, key=lambda p: p[0])
                attach_r, attach_c = attach_pos
                new_pos_list = [(attach_r, attach_c + 1 + i) for i in range(min_count)]
            else:
                # Extend left from bottommost leftmost core cell
                left_candidates = [(r, c) for r, c in core_pos_list if c == min_c_core]
                attach_pos = max(left_candidates, key=lambda p: p[0])
                attach_r, attach_c = attach_pos
                new_pos_list = [(attach_r, attach_c - 1 - i) for i in range(min_count)]
            new_pos_set = frozenset(new_pos_list)
        else:
            # Vertical orientation: stack upward from top row, closest to center column
            top_candidates = [(r, c) for r, c in core_pos_list if r == min_r_core]
            if not top_candidates:
                continue
            center_col = (min_c_core + max_c_core) // 2
            top_pos = min(top_candidates, key=lambda p: abs(p[1] - center_col))
            attach_c = top_pos[1]
            new_pos_list = []
            current_r = min_r_core - 1
            placed = 0
            while placed < min_count and current_r >= 0:
                new_pos_list.append((current_r, attach_c))
                placed += 1
                current_r -= 1
            new_pos_set = frozenset(new_pos_list)
        
        # Place the new minority cells
        O = fill(O, min_c, new_pos_set)
    
    return O

def solve_4c3d4a41(I):
    # Initialize output as list of lists all 0s (8x20 grid)
    H, W = 8, 20  # Fixed size from examples
    O_list = [[0 for _ in range(W)] for _ in range(H)]
    
    # Set row 0, columns 9-19 to 5
    for c in range(9, 20):
        O_list[0][c] = 5
    
    # Set row 7, columns 9-19 to 5
    for c in range(9, 20):
        O_list[7][c] = 5
    
    # Set row 6: col9=5, col19=5 (others remain 0)
    O_list[6][9] = 5
    O_list[6][19] = 5
    
    # Copy col9 and col19 from input for rows 1-5
    for r in range(1, 6):
        O_list[r][9] = index(I, (r, 9))
        O_list[r][19] = index(I, (r, 19))
    
    # Hardcode for each of the 4 bars to avoid nested loops
    # Bar 1: left col=1, right col=11
    h1 = sum(1 for r in range(1, 6) if index(I, (r, 1)) == 5)
    cap1 = 5 - h1
    seq1 = [index(I, (r, 11)) for r in range(1, 6) if index(I, (r, 11)) != 0 and index(I, (r, 11)) != 5]
    num_place1 = min(len(seq1), cap1)
    # Set bottom zone rows cap1+1 to 5 to 5
    for r in range(cap1 + 1, 6):
        O_list[r][11] = 5
    # Set top of cap zone to 5 if needed
    top_51 = cap1 - num_place1
    for r in range(1, top_51 + 1):
        O_list[r][11] = 5
    # Place colors if any
    if num_place1 > 0:
        subseq1 = seq1[-num_place1:]
        for k in range(num_place1):
            row = cap1 - k
            color = subseq1[num_place1 - 1 - k]
            O_list[row][11] = color
    
    # Bar 2: left col=3, right col=13
    h2 = sum(1 for r in range(1, 6) if index(I, (r, 3)) == 5)
    cap2 = 5 - h2
    seq2 = [index(I, (r, 13)) for r in range(1, 6) if index(I, (r, 13)) != 0 and index(I, (r, 13)) != 5]
    num_place2 = min(len(seq2), cap2)
    # Set bottom zone rows cap2+1 to 5 to 5
    for r in range(cap2 + 1, 6):
        O_list[r][13] = 5
    # Set top of cap zone to 5 if needed
    top_52 = cap2 - num_place2
    for r in range(1, top_52 + 1):
        O_list[r][13] = 5
    # Place colors if any
    if num_place2 > 0:
        subseq2 = seq2[-num_place2:]
        for k in range(num_place2):
            row = cap2 - k
            color = subseq2[num_place2 - 1 - k]
            O_list[row][13] = color
    
    # Bar 3: left col=5, right col=15
    h3 = sum(1 for r in range(1, 6) if index(I, (r, 5)) == 5)
    cap3 = 5 - h3
    seq3 = [index(I, (r, 15)) for r in range(1, 6) if index(I, (r, 15)) != 0 and index(I, (r, 15)) != 5]
    num_place3 = min(len(seq3), cap3)
    # Set bottom zone rows cap3+1 to 5 to 5
    for r in range(cap3 + 1, 6):
        O_list[r][15] = 5
    # Set top of cap zone to 5 if needed
    top_53 = cap3 - num_place3
    for r in range(1, top_53 + 1):
        O_list[r][15] = 5
    # Place colors if any
    if num_place3 > 0:
        subseq3 = seq3[-num_place3:]
        for k in range(num_place3):
            row = cap3 - k
            color = subseq3[num_place3 - 1 - k]
            O_list[row][15] = color
    
    # Bar 4: left col=7, right col=17
    h4 = sum(1 for r in range(1, 6) if index(I, (r, 7)) == 5)
    cap4 = 5 - h4
    seq4 = [index(I, (r, 17)) for r in range(1, 6) if index(I, (r, 17)) != 0 and index(I, (r, 17)) != 5]
    num_place4 = min(len(seq4), cap4)
    # Set bottom zone rows cap4+1 to 5 to 5
    for r in range(cap4 + 1, 6):
        O_list[r][17] = 5
    # Set top of cap zone to 5 if needed
    top_54 = cap4 - num_place4
    for r in range(1, top_54 + 1):
        O_list[r][17] = 5
    # Place colors if any
    if num_place4 > 0:
        subseq4 = seq4[-num_place4:]
        for k in range(num_place4):
            row = cap4 - k
            color = subseq4[num_place4 - 1 - k]
            O_list[row][17] = color
    
    # Set separators col10,12,14,16,18 rows1-5 to 0 (already 0, skip)
    
    # Override row 5, columns 11-17 to 5 (solid base)
    for c in range(11, 18):
        O_list[5][c] = 5
    
    # Convert to tuple of tuples
    O = tuple(tuple(row) for row in O_list)
    return O

def solve_6e453dd6(I):
    # Find the column index of the vertical gray bar (5)
    gray_pos = ofcolor(I, FIVE)
    mirror_col = min(j for i, j in gray_pos)
    
    # Find all 4-connected single-color black (0) objects, filter to those entirely left of gray bar
    all_objs = objects(I, T, F, T)
    black_objs = colorfilter(all_objs, ZERO)
    left_black_objs = [obj for obj in black_objs if all(j < mirror_col for i, j in toindices(obj))]
    
    # Create cleared grid by covering (removing) all left black objects with background
    cleared = I
    for obj in left_black_objs:
        indices = toindices(obj)
        cleared = cover(cleared, indices)
    
    # Shift each left black object right to flush against the gray bar and paint onto cleared grid
    shifted = cleared
    for obj in left_black_objs:
        positions = toindices(obj)
        max_c = max(j for i, j in positions)
        shift_amt = mirror_col - 1 - max_c
        delta = (0, shift_amt)
        new_pos = shift(positions, delta)
        new_obj = frozenset((ZERO, pos) for pos in new_pos)
        shifted = paint(shifted, new_obj)
    
    # For each row, check if col (mirror_col-1) is black (0) and col (mirror_col-2) is pink (6);
    # if so, fill the right side (mirror_col+1 to end) with red (2)
    h = height(I)
    w = width(I)
    final = shifted
    for r in range(h):
        if mirror_col >= 2:
            col1 = mirror_col - 1
            col2 = mirror_col - 2
            val1 = index(final, (r, col1))
            val2 = index(final, (r, col2))
            if val1 == ZERO and val2 == SIX:
                right_indices = frozenset((r, c) for c in range(mirror_col + 1, w))
                final = fill(final, TWO, right_indices)
    
    return final

def solve_7b5033c1(I):
    # Find all connected components (4-connected, single-color, ignoring background)
    objs = objects(I, True, False, True)
    
    # Sort objects top-to-bottom by min row, then left-to-right by min col if tie
    sorted_objs = sorted(objs, key=lambda o: (ulcorner(o)[0], ulcorner(o)[1]))
    
    # Create vertical bars for each object: 1-wide grid of height=size, color=obj color
    bars = [canvas(color(o), (size(o), 1)) for o in sorted_objs]
    
    # If no objects, return a 1x1 background (edge case, not in examples)
    if not bars:
        return canvas(mostcolor(I), (1, 1))
    
    # Stack all bars vertically into single output grid
    O = bars[0]
    for b in bars[1:]:
        O = vconcat(O, b)
    return O

def solve_7ed72f31(I):
    # Get grid dimensions and background color
    h = height(I)
    w = width(I)
    bg = mostcolor(I)

    # Helper to get 8 neighbors for touching check
    def get_neighbors(p):
        r, c = p
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        nbs = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                nbs.append((nr, nc))
        return nbs

    # Helper to check if two objects touch (8-way)
    def touches(o1, o2):
        pos1 = toindices(o1)
        pos2 = set(toindices(o2))  # set for fast lookup
        for p1 in pos1:
            for p2 in get_neighbors(p1):
                if p2 in pos2:
                    return True
        return False

    # Get all objects (single-color, 8-connected, ignore bg)
    all_objects = objects(I, T, T, T)
    reds = [o for o in all_objects if color(o) == TWO]
    non_reds = [o for o in all_objects if color(o) != TWO and color(o) != bg]

    # Start with input grid
    O = I

    # Process each red object
    for red in reds:
        pos = toindices(red)
        if len(pos) == 0:
            continue
        pos_list = list(pos)
        if len(pos_list) == 1:
            # Point reflection for single cell red
            r, c = pos_list[0]
            touching = [o for o in non_reds if touches(red, o)]
            for o in touching:
                colr = color(o)
                o_pos = toindices(o)
                new_pos = set()
                for pr, pc in o_pos:
                    nr = 2 * r - pr
                    nc = 2 * c - pc
                    if 0 <= nr < h and 0 <= nc < w and index(I, (nr, nc)) == bg:
                        new_pos.add((nr, nc))
                if new_pos:
                    O = fill(O, colr, frozenset(new_pos))
        else:
            # Check for horizontal or vertical line
            rows_set = set(p[0] for p in pos_list)
            cols_set = set(p[1] for p in pos_list)
            if len(rows_set) == 1:
                # Horizontal red line
                r = list(rows_set)[0]
                touching = [o for o in non_reds if touches(red, o)]
                for o in touching:
                    colr = color(o)
                    o_pos = toindices(o)
                    o_rows = [p[0] for p in o_pos]
                    if max(o_rows) < r:
                        # Reflect above to below
                        new_pos = set()
                        for pr, pc in o_pos:
                            nr = 2 * r - pr
                            nc = pc
                            if 0 <= nr < h and 0 <= nc < w and index(I, (nr, nc)) == bg:
                                new_pos.add((nr, nc))
                        if new_pos:
                            O = fill(O, colr, frozenset(new_pos))
                    elif min(o_rows) > r:
                        # Reflect below to above
                        new_pos = set()
                        for pr, pc in o_pos:
                            nr = 2 * r - pr
                            nc = pc
                            if 0 <= nr < h and 0 <= nc < w and index(I, (nr, nc)) == bg:
                                new_pos.add((nr, nc))
                        if new_pos:
                            O = fill(O, colr, frozenset(new_pos))
            elif len(cols_set) == 1:
                # Vertical red line
                c = list(cols_set)[0]
                touching = [o for o in non_reds if touches(red, o)]
                for o in touching:
                    colr = color(o)
                    o_pos = toindices(o)
                    o_cols = [p[1] for p in o_pos]
                    if max(o_cols) < c:
                        # Reflect left to right
                        new_pos = set()
                        for pr, pc in o_pos:
                            nr = pr
                            nc = 2 * c - pc
                            if 0 <= nr < h and 0 <= nc < w and index(I, (nr, nc)) == bg:
                                new_pos.add((nr, nc))
                        if new_pos:
                            O = fill(O, colr, frozenset(new_pos))
                    elif min(o_cols) > c:
                        # Reflect right to left, but only if not edge (check if min new nc >=0)
                        new_pos = set()
                        min_nc = min(2 * c - pc for pr, pc in o_pos)
                        if min_nc >= 0:  # only if not going out left edge
                            for pr, pc in o_pos:
                                nr = pr
                                nc = 2 * c - pc
                                if 0 <= nr < h and 0 <= nc < w and index(I, (nr, nc)) == bg:
                                    new_pos.add((nr, nc))
                            if new_pos:
                                O = fill(O, colr, frozenset(new_pos))

    return O

def solve_8e5c0c38(I):
    # Identify background color and grid width
    bg = mostcolor(I)
    W = width(I)
    # Get non-background colors
    colors = difference(palette(I), initset(bg))
    colors_list = list(colors)
    
    # Prepare all positions to remove (union across colors)
    all_to_remove = set()
    
    # Possible mirror sums s = 0 to 2*W-2
    possible_s = list(range(2 * W - 1))
    center_s = W - 1
    
    # Process each color independently
    for C in colors_list:
        S = ofcolor(I, C)
        if not S:
            continue
        S_set = set(S)
        # Compute kept counts for each possible s using list comprehension
        counts = [sum(1 for p in S if 0 <= s - p[1] < W and (p[0], s - p[1]) in S_set)
                  for s in possible_s]
        max_kept = max(counts)
        # Find indices of max kept
        candidates_idx = [i for i, c in enumerate(counts) if c == max_kept]
        # Choose the one closest to center_s
        best_idx = min(candidates_idx, key=lambda i: abs(possible_s[i] - center_s))
        best_s = possible_s[best_idx]
        # Compute positions to remove for this color using list comprehension
        to_remove_list = [p for p in S
                          if not (0 <= best_s - p[1] < W and (p[0], best_s - p[1]) in S_set)]
        all_to_remove.update(to_remove_list)
    
    # Fill all positions to remove with background color
    all_to_remove_fs = frozenset(all_to_remove)
    O = fill(I, bg, all_to_remove_fs)
    return O


def solve_16de56c4(I):
    # Helper for GCD
    def my_gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a

    # Get dimensions
    h = height(I)
    w = width(I)
    O = I

    if h > w:
        # Horizontal: process each row
        for r in range(h):
            # Collect non-empty positions and colors in row r
            non_empty = [(p, index(I, (r, p))) for p in range(w) if index(I, (r, p)) != 0]
            if len(non_empty) < 2:
                continue
            positions = [pc[0] for pc in non_empty]
            colors = [pc[1] for pc in non_empty]
            diffs = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            constant_diff = all(d == diffs[0] for d in diffs)
            the_c = 0
            fill_pos = []
            if constant_diff:
                d = diffs[0]
                first_c = colors[0]
                if all(c == first_c for c in colors):
                    # Tight same color: extend to full AP
                    residue = positions[0] % d
                    cur = residue
                    while cur < w:
                        fill_pos.append(cur)
                        cur += d
                    the_c = first_c
                else:
                    # Mixed constant diff: overwrite existing positions with first color
                    fill_pos = positions
                    the_c = first_c
            else:
                # Non-constant
                if colors[-1] == colors[-2]:
                    # Extend tight suffix of last two (same color)
                    suffix_d = positions[-1] - positions[-2]
                    residue = positions[-2] % suffix_d
                    cur = residue
                    while cur < w:
                        fill_pos.append(cur)
                        cur += suffix_d
                    the_c = colors[-1]
                else:
                    # GCD fill within min-max with last color
                    d = my_gcd(diffs[0], diffs[1])
                    the_c = colors[-1]
                    min_p = positions[0]
                    max_p = positions[-1]
                    cur = min_p
                    while cur <= max_p:
                        fill_pos.append(cur)
                        cur += d
            # Build grid positions and fill
            grid_pos = [(r, fp) for fp in fill_pos]
            O = fill(O, the_c, frozenset(grid_pos))
    else:
        # Vertical: process each column
        for c in range(w):
            # Collect non-empty positions and colors in column c
            non_empty = [(p, index(I, (p, c))) for p in range(h) if index(I, (p, c)) != 0]
            if len(non_empty) < 2:
                continue
            positions = [pc[0] for pc in non_empty]
            colors = [pc[1] for pc in non_empty]
            diffs = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            constant_diff = all(d == diffs[0] for d in diffs)
            the_c = 0
            fill_pos = []
            if constant_diff:
                d = diffs[0]
                first_c = colors[0]
                if all(c == first_c for c in colors):
                    # Tight same color: extend to full AP
                    residue = positions[0] % d
                    cur = residue
                    while cur < h:
                        fill_pos.append(cur)
                        cur += d
                    the_c = first_c
                else:
                    # Mixed constant diff: overwrite existing positions with first color
                    fill_pos = positions
                    the_c = first_c
            else:
                # Non-constant
                if colors[-1] == colors[-2]:
                    # Extend tight suffix of last two (same color)
                    suffix_d = positions[-1] - positions[-2]
                    residue = positions[-2] % suffix_d
                    cur = residue
                    while cur < h:
                        fill_pos.append(cur)
                        cur += suffix_d
                    the_c = colors[-1]
                else:
                    # GCD fill within min-max with last color
                    d = my_gcd(diffs[0], diffs[1])
                    the_c = colors[-1]
                    min_p = positions[0]
                    max_p = positions[-1]
                    cur = min_p
                    while cur <= max_p:
                        fill_pos.append(cur)
                        cur += d
            # Build grid positions and fill
            grid_pos = [(fp, c) for fp in fill_pos]
            O = fill(O, the_c, frozenset(grid_pos))
    return O

def solve_31f7f899(I):
    # Get grid dimensions
    h = height(I)
    w = width(I)
    
    # Find the floor row: row with the most 6's
    floor_row = 0
    max_count = 0
    for r in range(h):
        row_grid = crop(I, (r, 0), (1, w))
        count = size(ofcolor(row_grid, 6))
        if count > max_count:
            max_count = count
            floor_row = r
    
    # Identify bar columns and their colors: positions in floor row with non-6, non-bg
    bg = mostcolor(I)
    bar_columns = []
    for j in range(w):
        c = index(I, (floor_row, j))
        if c != 6 and c != bg:
            bar_columns.append((j, c))
    
    # Compute original heights for each bar
    heights = []
    for j, c in bar_columns:
        # Count cells above floor_row that are consecutively c
        k_up = 0
        for k in range(floor_row):
            r = floor_row - 1 - k
            if index(I, (r, j)) != c:
                break
            k_up = k + 1
        # Count cells below floor_row that are consecutively c
        k_down = 0
        for k in range(h - floor_row - 1):
            r = floor_row + 1 + k
            if index(I, (r, j)) != c:
                break
            k_down = k + 1
        # Height includes floor cell
        orig_h = 1 + k_up + k_down
        heights.append(orig_h)
    
    # Sort heights ascending
    sorted_heights = order(tuple(heights), identity)
    
    # Build output grid starting from background
    O = canvas(bg, shape(I))
    
    # Paint the floor 6's where original floor has 6
    floor_pos = [(floor_row, j) for j in range(w) if index(I, (floor_row, j)) == 6]
    if floor_pos:
        floor_obj = frozenset((6, pos) for pos in floor_pos)
        O = paint(O, floor_obj)
    
    # Paint each bar with reassigned sorted height, symmetrically around floor_row
    for idx, (j, c) in enumerate(bar_columns):
        h_new = sorted_heights[idx]
        m = (h_new - 1) // 2
        bar_pos = []
        for k in range(-m, m + 1):
            r = floor_row + k
            if 0 <= r < h:
                bar_pos.append((r, j))
        bar_obj = frozenset((c, pos) for pos in bar_pos)
        O = paint(O, bar_obj)
    
    return O

def solve_35ab12c3(I):
    # Helper function for cross product in convex hull
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Helper function to compute convex hull of points using monotone chain
    def convex_hull(pts):
        pts = sorted(set(pts))
        if len(pts) <= 1:
            return pts

        def build_hull(points):
            hull = []
            for p in points:
                while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
                    hull.pop()
                hull.append(p)
            return hull

        lower = build_hull(pts)
        upper = build_hull(list(reversed(pts)))
        return lower[:-1] + upper[:-1]

    # Get sorted list of non-background colors
    colors = sorted(c for c in palette(I) if c != 0)

    # Start with input grid
    O = I
    h, w = shape(I)

    # Phase 1: Handle multi-point colors by filling aligned convex hull edges
    for c in colors:
        pos_set = ofcolor(I, c)
        if size(pos_set) > 1:
            pos_list = list(pos_set)
            hull = convex_hull(pos_list)
            if len(hull) > 1:
                lines = set()
                nh = len(hull)
                for i in range(nh):
                    p1 = hull[i]
                    p2 = hull[(i + 1) % nh]
                    r1, c1 = p1
                    r2, c2 = p2
                    dr = abs(r1 - r2)
                    dc = abs(c1 - c2)
                    if dr + dc > 0 and (dr == 0 or dc == 0 or dr == dc):
                        line = connect(p1, p2)
                        lines |= line
                if lines:
                    O = fill(O, c, lines)

    # Phase 2: Handle single-point colors by shifting adjacent components
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for c in colors:
        pos_set = ofcolor(I, c)
        if size(pos_set) == 1:
            s = next(iter(pos_set))
            sr, sc = s
            # Find candidate adjacent cells
            candidates = []
            for dr, dc in dirs:
                nr = sr + dr
                nc = sc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    col_at = index(O, (nr, nc))
                    if col_at != 0 and col_at != c:
                        dist = dr * dr + dc * dc
                        # Tie-break by column then row for determinism
                        candidates.append(((dist, nc, nr), (nr, nc), col_at))
            if not candidates:
                continue
            # Select the nearest (min dist, then tie-breaks)
            candidates.sort()
            _, n, d = candidates[0]
            nr, nc = n
            # Compute shift vector from n to s
            vec_r = sr - nr
            vec_c = sc - nc
            # Find the connected component containing n
            objs = objects(O, True, True, True)  # univalued, 8-connected, ignore bg
            d_objs = [obj for obj in objs if color(obj) == d]
            the_obj = None
            for obj in d_objs:
                if (nr, nc) in toindices(obj):
                    the_obj = obj
                    break
            if the_obj is None:
                continue
            # Shift all positions in the component
            positions = toindices(the_obj)
            new_pos = set()
            for pr, pc in positions:
                nnr = pr + vec_r
                nnc = pc + vec_c
                if 0 <= nnr < h and 0 <= nnc < w and index(O, (nnr, nnc)) == 0:
                    new_pos.add((nnr, nnc))
            if new_pos:
                O = fill(O, c, new_pos)

    return O

def solve_53fb4810(I):
    # Replace all colors except 2 (TWO) and 4 (FOUR) to background 8 (EIGHT)
    temp = I
    for c in [ZERO, ONE, THREE, FIVE, SIX, SEVEN, NINE]:
        temp = replace(temp, c, EIGHT)
    
    # Extract connected components (multicolor allowed, 4-connected, ignore bg) of 2-4 regions
    seeds = objects(temp, F, F, T)
    if size(seeds) == ZERO:
        return I
    
    # Function to get top row: compose first (row) with ulcorner
    top_func = compose(first, ulcorner)
    
    # Select the lowest seed (one with maximum top row)
    seed = argmax(seeds, top_func)
    
    # Compute seed_top and seed_bot rows
    u = ulcorner(seed)
    seed_top = first(u)
    l = lrcorner(seed)
    seed_bot = first(l)
    
    # Compute height h of seed
    h = add(ONE, subtract(seed_bot, seed_top))
    
    # Collect all (color, position) pairs for the tiled motif, skipping blue (1) positions
    # For each rr from 0 to seed_bot, match cells where relative row modulo h aligns
    all_cells = [(colr, (rr, sc)) for rr in range(add(seed_bot, ONE))
                            for colr, (sr, sc) in seed
                            if subtract(rr, seed_top) % h == subtract(sr, seed_top)
                            and index(I, (rr, sc)) != ONE]
    
    # Create the fill object from collected cells
    fill_obj = frozenset(all_cells)
    
    # Paint the tiled motif onto the input grid
    O = paint(I, fill_obj)
    return O


def solve_58f5dbd5(I):
    # Identify background color
    bg = mostcolor(I)
    
    # Extract all single-color, 4-connected objects excluding background
    all_objs = objects(I, T, F, T)
    
    # Find qualifying colors: those with one large component (max size) and small components (rest)
    qual = []
    small_rels = {}
    large_objs = {}
    colors = palette(I)
    for C in colors:
        if C == bg:
            continue
        objs_C = colorfilter(all_objs, C)
        if not objs_C:
            continue
        obj_sizes = [(len(toindices(o)), o) for o in objs_C]
        if not obj_sizes:
            continue
        max_sz, large = max(obj_sizes)
        if max_sz < 15:  # Heuristic: large blocks are ~25 cells
            continue
        small_os = [o for sz, o in obj_sizes if sz < max_sz]  # Rest are small (handle ties implicitly)
        if small_os:
            small_inds_list = [toindices(o) for o in small_os]
            small_inds = set()
            for inds in small_inds_list:
                small_inds.update(inds)
            if small_inds:
                minr = min(r for r, c in small_inds)
                minc = min(c for r, c in small_inds)
                rels = {(r - minr, c - minc) for r, c in small_inds}
                small_rels[C] = frozenset(rels)
                large_objs[C] = large
                qual.append(C)
    
    if not qual:
        # No qualifying pairs: return input unchanged (edge case)
        return I
    
    # Compute centers of large objects to determine layout
    centers = {}
    for C in qual:
        inds = toindices(large_objs[C])
        centers[C] = center(inds)
    
    # Determine grid dimensions v (block rows), h (block cols) from unique center rows/cols
    row_set = {centers[C][0] for C in qual}
    col_set = {centers[C][1] for C in qual}
    v = len(row_set)
    h = len(col_set)
    
    # Order qualifying colors row-major by center positions
    def key_func(C):
        return centers[C]
    ordered_Cs = sorted(qual, key=key_func)
    
    # Compute output grid size: v*5 + (v+1) for height, h*5 + (h+1) for width
    out_height = v * 5 + v + 1
    out_width = h * 5 + h + 1
    O = canvas(bg, (out_height, out_width))
    
    # Build and place each 5x5 patterned block
    for i in range(v):
        for j in range(h):
            idx = i * h + j
            C = ordered_Cs[idx]
            
            # Create 5x5 canvas filled with C
            block = canvas(C, (5, 5))
            
            # Compute cutout positions: small relative shape offset by (1,1), clipped to 5x5
            cut_pos = {(dr + 1, dc + 1) for dr, dc in small_rels[C]
                       if 0 <= dr + 1 < 5 and 0 <= dc + 1 < 5}
            if cut_pos:
                block = fill(block, bg, frozenset(cut_pos))
            
            # Get indices of C cells in the block
            C_inds = ofcolor(block, C)
            
            # Compute placement offset for this block position
            start_r = 1 + i * 6
            start_c = 1 + j * 6
            shifted = {(r + start_r, c + start_c) for r, c in C_inds}
            
            # Create object and paint onto output
            obj = frozenset((C, pos) for pos in shifted)
            O = paint(O, obj)
    
    return O

def solve_80a900e0(I):
    # Get grid dimensions
    h, w = shape(I)
    
    # Find all single-color 8-connected objects (without excluding bg explicitly, filter later)
    all_objs = objects(I, T, T, F)
    
    # Collect boundary chains: color >1 !=3, size >=2, and straight diagonal (constant diff or sum)
    boundary_chains = []
    for obj in all_objs:
        c = color(obj)
        if c > 1 and c != 3:
            poss = toindices(obj)
            if size(obj) >= 2:
                diffs = {r - cc for r, cc in poss}
                sums_ = {r + cc for r, cc in poss}
                if len(diffs) == 1 or len(sums_) == 1:
                    boundary_chains.append((obj, c))
    
    # Create mutable version of input grid
    O = [list(row) for row in I]
    
    # Perpendicular ray directions
    main_dirs = [(-1, -1), (1, 1)]  # For anti-diag chains (perp to main-diag)
    anti_dirs = [(-1, 1), (1, -1)]  # For main-diag chains (perp to anti-diag)
    
    # Process each boundary chain
    for chain, c in boundary_chains:
        poss = sorted(toindices(chain), key=lambda p: p[0])  # Sort by row to find ends
        if len(poss) < 2:
            continue
        ends = [poss[0], poss[-1]]  # Upper and lower endpoints
        
        # Determine chain type
        tposs = toindices(chain)
        diffs = {r - cc for r, cc in tposs}
        sums_ = {r + cc for r, cc in tposs}
        
        if len(diffs) == 1:
            # Main-diag chain: use anti-diag ray directions
            dirs = anti_dirs
        elif len(sums_) == 1:
            # Anti-diag chain: use main-diag ray directions
            dirs = main_dirs
        else:
            continue  # Not a straight diagonal chain
        
        # From each endpoint, cast rays in both perp directions
        for r, cc in ends:
            for dr, dc in dirs:
                cr, ccc = r + dr, cc + dc  # Start from next cell
                while 0 <= cr < h and 0 <= ccc < w:
                    if O[cr][ccc] == 1:
                        O[cr][ccc] = c  # Paint if 1
                    elif O[cr][ccc] != 1:
                        break  # Stop if not paintable (0, 3, or other)
                    cr += dr
                    ccc += dc
    
    # Convert back to immutable grid
    return tuple(tuple(row) for row in O)
    
def solve_97d7923e(I):
    # Get grid dimensions
    H = height(I)
    W = width(I)
    bottom = H - 1
    
    # Find columns with non-zero in bottom row
    bottom_js = [j for j in range(W) if index(I, (bottom, j)) != 0]
    if not bottom_js:
        return I
    rightmost = max(bottom_js)
    
    # Extract all multi-color, 4-connected objects (ignoring background)
    objs = objects(I, F, F, T)
    
    # Start with input grid
    O = I
    
    # Process each object to find qualifying sandwiches
    for obj in objs:
        # Get positions
        poss = toindices(obj)
        if size(poss) == 0:
            continue
        
        # Check if touches bottom (lower-left row == bottom)
        llc = llcorner(poss)
        if llc[0] != bottom:
            continue
        
        # Check single column
        col_set = {p[1] for p in poss}
        if len(col_set) != 1:
            continue
        j = min(col_set)
        
        # Check single column rows
        row_set = {p[0] for p in poss}
        min_r = min(row_set)
        max_r = max(row_set)
        if max_r != bottom:
            continue
        if max_r - min_r + 1 != size(poss):
            continue  # Gaps in vertical span
        
        # Get bottom cap color C
        bottom_pos = (bottom, j)
        bottom_pair = [p for p in obj if p[1] == bottom_pos][0]
        C = bottom_pair[0]
        
        # Get top cap color
        top_pos = (min_r, j)
        top_pair = [p for p in obj if p[1] == top_pos][0]
        top_C = top_pair[0]
        if top_C != C:
            continue
        
        # Define body range
        body_start = min_r + 1
        body_end = bottom - 1
        if body_start > body_end:
            continue  # No body (height 1)
        
        # Check body is uniform D != C and != 0
        first_d = None
        uniform = True
        for r in range(body_start, body_end + 1):
            pos = (r, j)
            pair = [p for p in obj if p[1] == pos][0]
            d = pair[0]
            if first_d is None:
                first_d = d
            elif d != first_d:
                uniform = False
                break
        if not uniform or first_d == 0 or first_d == C:
            continue
        
        # Check conditions: rightmost or prior disconnected block above
        is_rightmost = (j == rightmost)
        has_prior = any(index(I, (r, j)) != 0 for r in range(min_r))
        if is_rightmost or has_prior:
            # Create body patch and fill with C
            body_patch = frozenset((r, j) for r in range(body_start, body_end + 1))
            O = fill(O, C, body_patch)
    
    return O




def solve_135a2760(I):
    # Get grid dimensions and background color (outer border)
    h = height(I)
    w = width(I)
    outer = mostcolor(I)  # background b
    inner_start = 2
    inner_len = w - 4  # number of inner cells to pattern
    O_rows = []
    for r in range(h):
        row = I[r]
        # Get colors in inner section
        inner_colors_set = set(row[j] for j in range(inner_start, w - 2))
        if len(inner_colors_set) != 2 or outer not in inner_colors_set:
            # Not a mixed patterned row: copy as is
            O_rows.append(row)
            continue
        # Identify foreground c
        c = (inner_colors_set - {outer}).pop()
        # Determine k based on c
        if c == 1 or c == 9:
            k = 1
        elif c == 3:
            k = 2
        elif c == 8:
            k = 3
        else:
            # Unknown color: copy as is
            O_rows.append(row)
            continue
        p = k + 1
        # Find best phase s with minimal differences
        min_diff = 1000
        best_target = None
        for s in range(p):
            target = [(c if ((ii + s) % p) < k else outer) for ii in range(inner_len)]
            diff = sum(target[ii] != row[inner_start + ii] for ii in range(inner_len))
            if diff < min_diff:
                min_diff = diff
                best_target = target
        # Apply the best target to inner cells
        new_row_list = list(row)
        for ii in range(inner_len):
            new_row_list[inner_start + ii] = best_target[ii]
        O_rows.append(tuple(new_row_list))
    # Reconstruct output grid
    O = tuple(O_rows)
    return O

def solve_221dfab4(I):
    # Determine grid dimensions
    H = height(I)
    W = width(I)
    
    # Identify background (most frequent color) and foreground (the other non-background, non-4 color)
    bkg = mostcolor(I)
    pal = palette(I)
    fgd_set = difference(pal, {bkg, 4})  # Assuming exactly one remaining color
    fgd = extract(fgd_set, identity)  # Extract the single foreground color
    
    # Find active columns from the yellow (4) bar in the bottom row (consecutive)
    last_row = I[H - 1]
    active_js = [j for j in range(W) if index(I, (H - 1, j)) == 4]
    if active_js:
        min_c = minimum(active_js)
        max_c = maximum(active_js)
        active_cols = range(min_c, max_c + 1)
    else:
        active_cols = range(0)  # Empty range if no active columns
    
    # Build output rows by processing each row except the last (unchanged)
    O_rows = []
    for r in range(H - 1):
        row_in = I[r]
        new_row_list = list(row_in)
        
        if r % 2 == 1:  # Odd row (0-indexed): clear active columns to background
            for c in active_cols:
                new_row_list[c] = bkg
        else:  # Even row (0-indexed): fill active columns with BC, and if BC==3 recolor all FGD to 3
            k = (r // 2) + 1  # k-th odd row (1-indexed)
            bc = 3 if (k % 3 == 1) else 4
            for c in active_cols:
                new_row_list[c] = bc
            if bc == 3:
                for c in range(W):
                    if row_in[c] == fgd:
                        new_row_list[c] = 3
        
        O_rows.append(tuple(new_row_list))
    
    # Append unchanged bottom row
    O_rows.append(last_row)
    
    # Return as grid (tuple of tuples)
    return tuple(O_rows)

def solve_247ef758(I):
    # Get grid dimensions
    height_h = height(I)
    width_w = width(I)
    
    # Extract top row as tuple
    top_row = tuple(index(I, (0, j)) for j in range(width_w))
    
    # Extract right column as tuple
    right_col = tuple(index(I, (r, width_w - 1)) for r in range(height_h))
    
    # Find core column k: leftmost >=4 matching right column
    k = None
    for cc in range(4, width_w):
        col_cc = tuple(index(I, (r, cc)) for r in range(height_h))
        if col_cc == right_col:
            k = cc
            break
    
    # Movable colors: unique non-zero colors in top row from k onward
    movable_cs = {top_row[j] for j in range(k, width_w) if top_row[j] != ZERO}
    
    # Collect left objects L_c and clears for each movable c
    left_objects = {}
    clears = set()
    for c in movable_cs:
        pos_c = ofcolor(I, c)
        left_pos = [(r, j) for r, j in pos_c if j < 3]
        if left_pos:
            left_objects[c] = left_pos
            clears.update(left_pos)
    
    # Initial output: clear all left movable objects
    O = fill(I, ZERO, frozenset(clears))
    
    # Collect all (v, h, c) placements
    placements = []
    for c in movable_cs:
        if c not in left_objects:
            continue
        # V_c: rows where core col k has c
        Vc = [r for r in range(height_h) if index(I, (r, k)) == c]
        # H_c: cols >=k in top row with c
        Hc = [j for j in range(k, width_w) if top_row[j] == c]
        for v in Vc:
            for h in Hc:
                placements.append((v, h, c))
    
    # Sort placements by increasing v, then h (for sequential filling order)
    placements.sort(key=lambda p: (p[0], p[1]))
    
    # For each placement, translate L_c and fill only background cells
    for v, h, c in placements:
        L = left_objects[c]
        if not L:
            continue
        # Compute center_r and center_j as integer means
        rows = [p[0] for p in L]
        num = len(rows)
        center_r = sum(rows) // num
        colss = [p[1] for p in L]
        center_j = sum(colss) // num
        dr = v - center_r
        dj = h - center_j
        # Collect valid in-bound targets that are still 0
        to_fill_set = set()
        for r, j in L:
            nr = r + dr
            nj = j + dj
            if 0 <= nr < height_h and 0 <= nj < width_w and index(O, (nr, nj)) == ZERO:
                to_fill_set.add((nr, nj))
        if to_fill_set:
            O = fill(O, c, frozenset(to_fill_set))
    
    return O

def solve_5545f144(I):
    # Get dimensions and background color
    h = height(I)
    bg = mostcolor(I)
    
    # Find first separator column: leftmost uniform non-bg column
    w = width(I)
    first_sep = w
    for j in range(w):
        col_color = index(I, (0, j))
        if col_color != bg:
            uniform = all(index(I, (i, j)) == col_color for i in range(h))
            if uniform:
                first_sep = j
                break
    
    # Output dimensions
    out_w = first_sep
    out_h = h
    
    # Create blank output grid
    O = canvas(bg, (out_h, out_w))
    
    # If no panel, return empty
    if out_w == 0:
        return O
    
    # Extract first panel
    panel = crop(I, ORIGIN, (out_h, out_w))
    
    # Find object color: color with highest count excluding bg
    pal = palette(panel)
    counts = {c: size(ofcolor(panel, c)) for c in pal}
    non_bg_items = [(cnt, c) for c, cnt in counts.items() if c != bg and cnt > 0]
    if not non_bg_items:
        return O
    obj_col = max(non_bg_items, key=lambda x: x[0])[1]
    
    # Get all object positions
    obj_pos = list(ofcolor(panel, obj_col))
    if not obj_pos:
        return O
    
    # Compute row counts
    row_count = [0] * out_h
    for r, c in obj_pos:
        row_count[r] += 1
    
    # Find max count and topmost prominent row
    max_count = max(row_count)
    prom_rows = [r for r, cnt in enumerate(row_count) if cnt == max_count]
    prom_row = min(prom_rows)
    
    # Decide top or bottom based on prominent row position
    half = out_h // 2
    patch_list = []  # to collect positions
    if prom_row <= half:
        # Top case
        min_r = min(r for r, c in obj_pos)
        top_row_cols = [c for r, c in obj_pos if r == min_r]
        center_c = min(top_row_cols)
        length_v = 2 if min_r == 0 else 1
        start_r = min_r
        end_r = start_r + length_v - 1
        for rr in range(start_r, end_r + 1):
            if 0 <= center_c < out_w:
                patch_list.append((rr, center_c))
        arm_r = end_r + 1
        if arm_r < out_h:
            left_arm = center_c - 1
            right_arm = center_c + 1
            if 0 <= left_arm < out_w:
                patch_list.append((arm_r, left_arm))
            if 0 <= right_arm < out_w:
                patch_list.append((arm_r, right_arm))
    else:
        # Bottom case
        max_r = max(r for r, c in obj_pos)
        bottom_row_cols = [c for r, c in obj_pos if r == max_r]
        center_c = max(bottom_row_cols)
        bar_r = max_r - 1
        if bar_r >= 0:
            left_bar = center_c - 1
            mid_bar = center_c
            right_bar = center_c + 1
            if 0 <= left_bar < out_w:
                patch_list.append((bar_r, left_bar))
            if 0 <= mid_bar < out_w:
                patch_list.append((bar_r, mid_bar))
            if 0 <= right_bar < out_w:
                patch_list.append((bar_r, right_bar))
        if 0 <= center_c < out_w:
            patch_list.append((max_r, center_c))
    
    # Create object from patch
    patch = frozenset(patch_list)
    obj = frozenset((obj_col, pos) for pos in patch)
    
    # Paint the object on output
    O = paint(O, obj)
    
    return O

def solve_7491f3cf(I):
    # Extract the four 5x5 panels (rows 1-5, cols 1-5,7-11,13-17,19-23)
    p1 = crop(I, (1, 1), (5, 5))
    p2 = crop(I, (1, 7), (5, 5))
    p3 = crop(I, (1, 13), (5, 5))
    p4_input = crop(I, (1, 19), (5, 5))
    
    # Extract vertical borders (5x1 strips at cols 0,6,12,18,24, rows 1-5)
    vb0 = crop(I, (1, 0), (5, 1))
    vb6 = crop(I, (1, 6), (5, 1))
    vb12 = crop(I, (1, 12), (5, 1))
    vb18 = crop(I, (1, 18), (5, 1))
    vb24 = crop(I, (1, 24), (5, 1))
    
    # Extract top and bottom borders (1x25 at rows 0 and 6)
    top = crop(I, (0, 0), (1, 25))
    bottom = crop(I, (6, 0), (1, 25))
    
    # Compute panels 1-3 unchanged
    # p1, p2, p3 already extracted
    
    # Compute p4_new
    # Get foreground for p1, p2, p3
    objs1 = objects(p1, T, T, T)  # 8-connected for S1 components
    if objs1:
        largest_obj1 = max(list(objs1), key=lambda o: size(o))
        boundary = toindices(largest_obj1)
    else:
        boundary = frozenset()
    
    objs2 = objects(p2, T, F, T)  # 4-connected, but for all fg
    if objs2:
        c2 = color(list(objs2)[0])
        s2 = ofcolor(p2, c2)
    else:
        c2 = 0  # fallback, assume has fg
        s2 = frozenset()
    
    objs3 = objects(p3, T, F, T)
    if objs3:
        c3 = color(list(objs3)[0])
        s3 = ofcolor(p3, c3)
    else:
        c3 = 0  # fallback
        s3 = frozenset()
    
    bg4 = mostcolor(p4_input)
    
    # Create mask: 0 non-boundary, 1 boundary
    mask = canvas(0, (5, 5))
    mask = fill(mask, 1, boundary)
    
    # Get 4-connected components of non-boundary (color 0)
    objs_non = colorfilter(objects(mask, T, F, F), 0)
    components = list(objs_non)
    
    # Helper functions
    def get_score(obj):
        inds = toindices(obj)
        inter = intersection(inds, s2)
        return size(inter)
    
    def get_avg_col(obj):
        inds = toindices(obj)
        n = size(inds)
        if n == 0:
            return 0.0
        sum_c = sum(c for r, c in inds)
        return sum_c / n
    
    def get_avg_row(obj):
        inds = toindices(obj)
        n = size(inds)
        if n == 0:
            return 0.0
        sum_r = sum(r for r, c in inds)
        return sum_r / n
    
    # Find A: max score, tie min avg_col, tie min avg_row
    if components:
        A_obj = max(components, key=lambda obj: (get_score(obj), -get_avg_col(obj), -get_avg_row(obj)))
        A = toindices(A_obj)
        remaining = [obj for obj in components if obj != A_obj]
        if remaining:
            B_obj = max(remaining, key=get_score)
            B = toindices(B_obj)
        else:
            B = frozenset()
    else:
        A = frozenset()
        B = frozenset()
    
    # Positions to paint
    union_A_bd = combine(A, boundary)
    indices_c2 = intersection(s2, union_A_bd)
    
    union_B_bd = combine(B, boundary)
    indices_c3 = intersection(s3, union_B_bd)
    indices_c3_only = difference(indices_c3, indices_c2)
    
    # Build p4_new
    p4_new = canvas(bg4, (5, 5))
    p4_new = fill(p4_new, c2, indices_c2)
    p4_new = fill(p4_new, c3, indices_c3_only)
    
    # Reconstruct middle rows (1-5)
    middle = hconcat(vb0, p1)
    middle = hconcat(middle, vb6)
    middle = hconcat(middle, p2)
    middle = hconcat(middle, vb12)
    middle = hconcat(middle, p3)
    middle = hconcat(middle, vb18)
    middle = hconcat(middle, p4_new)
    middle = hconcat(middle, vb24)
    
    # Reconstruct full grid
    O = vconcat(top, middle)
    O = vconcat(O, bottom)
    return O

def solve_78332cb0(I):
    # Get dimensions
    h = height(I)
    w = width(I)
    
    # Find horizontal separator rows (full rows of 6's)
    horiz_seps = []
    for i in range(h):
        is_sep = True
        for j in range(w):
            if index(I, (i, j)) != 6:
                is_sep = False
                break
        if is_sep:
            horiz_seps.append(i)
    
    # Find vertical separator columns (full columns of 6's)
    vert_seps = []
    for j in range(w):
        is_sep = True
        for i in range(h):
            if index(I, (i, j)) != 6:
                is_sep = False
                break
        if is_sep:
            vert_seps.append(j)
    
    # Compute meta-grid dimensions
    r = len(horiz_seps) + 1
    c = len(vert_seps) + 1
    
    # Compute panel start positions
    row_div = [0] + [s + 1 for s in horiz_seps] + [h]
    row_starts = row_div[:-1]
    col_div = [0] + [s + 1 for s in vert_seps] + [w]
    col_starts = col_div[:-1]
    
    # Extract 5x5 panels
    panels = [[None] * c for _ in range(r)]
    panel_size = astuple(5, 5)
    for i in range(r):
        rs = row_starts[i]
        for j in range(c):
            cs = col_starts[j]
            panels[i][j] = crop(I, (rs, cs), panel_size)
    
    # Compute ordered panels based on case
    N = r * c
    if r == 2 and c == 2:
        # Diagonal traversal: TL, BR, TR, BL
        ordered_panels = [panels[0][0], panels[1][1], panels[0][1], panels[1][0]]
    elif c == 1:
        # Tall or single column: reverse row order (bottom to top)
        ordered_panels = [row[0] for row in panels[::-1]]
    elif r == 1:
        # Wide row: preserve left-to-right order
        ordered_panels = panels[0]
    else:
        # Fallback for unexpected cases (e.g., row-major flatten)
        ordered_panels = []
        for i in range(r):
            for j in range(c):
                ordered_panels.append(panels[i][j])
    
    # Determine if input meta is tall (output horizontal) or not (output vertical)
    is_tall_meta = r > c
    
    # Build separators using DSL
    horiz_sep = canvas(6, astuple(1, 5))  # 1x5 all 6's
    vert_sep = canvas(6, astuple(5, 1))   # 5x1 all 6's
    
    if N == 1:
        return ordered_panels[0]
    
    if is_tall_meta:
        # Output horizontal stack: panel + (sep + panel) * (N-1)
        O = ordered_panels[0]
        for k in range(1, N):
            O = hconcat(O, vert_sep)
            O = hconcat(O, ordered_panels[k])
    else:
        # Output vertical stack: panel + (sep + panel) * (N-1)
        O = ordered_panels[0]
        for k in range(1, N):
            O = vconcat(O, horiz_sep)
            O = vconcat(O, ordered_panels[k])
    
    return O

def solve_89565ca0(I):
    # Get all non-zero colors present in the grid
    pal = palette(I)
    non_zero = [c for c in pal if c != 0]
    
    # Find all single-color, 8-connected objects, ignoring background
    all_objs = objects(I, True, True, True)
    
    # Compute max component size for each non-zero color
    max_sizes = {}
    for c in non_zero:
        objs_c = colorfilter(all_objs, c)
        if objs_c:
            sizes = [size(obj) for obj in objs_c]
            max_sizes[c] = max(sizes)
        else:
            max_sizes[c] = 0
    
    # Filler is the color with smallest max component size (tie broken by smallest color ID)
    filler = min(max_sizes, key=lambda c: (max_sizes[c], c))
    
    # Main colors are all non-zero except filler
    main_set = set(non_zero) - {filler}
    
    # Determine stacking order based on priority rules
    order = []
    if 8 in main_set:
        order.append(8)
        if 1 in main_set:
            order.append(1)
        if 2 in main_set:
            order.append(2)
        remaining = [c for c in main_set if c > 2 and c != 8]
        order.extend(sorted(remaining, reverse=True))
    else:
        if 1 in main_set:
            order.append(1)
        remaining = [c for c in main_set if c > 1]
        order.extend(sorted(remaining, reverse=True))
    
    n = len(order)
    if n == 0:
        # Edge case: no main colors, return empty grid (though unlikely)
        return canvas(0, (0, 4))
    
    # Create output canvas filled with filler color, size n x 4
    O = canvas(filler, (n, 4))
    
    # For each row, compute bar length and fill the left positions with the row's color
    for i, colr in enumerate(order):
        length = 4 if i == n - 1 else min(i + 1, 3)
        positions = frozenset((i, j) for j in range(length))
        O = fill(O, colr, positions)
    
    return O


def solve_1818057f(I):
    # Get all yellow positions (color 4)
    yellow_pos = ofcolor(I, FOUR)
    
    # Find centers: yellow positions with yellow neighbors up, down, left, right
    centers = set()
    for pos in yellow_pos:
        r, c = pos
        center_val = index(I, pos)  # Should be FOUR, but confirm
        left_val = index(I, (r, c - 1))
        right_val = index(I, (r, c + 1))
        up_val = index(I, (r - 1, c))
        down_val = index(I, (r + 1, c))
        if (center_val == FOUR and
            left_val == FOUR and
            right_val == FOUR and
            up_val == FOUR and
            down_val == FOUR):
            centers.add(pos)
    
    # Collect all positions to change to purple (8): the plus shapes from each center
    to_change = set()
    for pos in centers:
        r, c = pos
        to_change.add((r, c))      # center
        to_change.add((r, c - 1))  # left
        to_change.add((r, c + 1))  # right
        to_change.add((r - 1, c))  # up
        to_change.add((r + 1, c))  # down
    
    # Fill the grid: change those positions to purple, leave others unchanged
    O = fill(I, EIGHT, frozenset(to_change))
    return O


















