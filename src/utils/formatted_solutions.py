
def solve_0(I):
    A = I[0][0]
    B = I[0][1]
    C = I[1][0]
    D = I[1][1]
    
    normal1 = [A, B] * 3
    normal2 = [C, D] * 3
    rev1 = [B, A] * 3
    rev2 = [D, C] * 3
    
    output = [
        normal1,
        normal2,
        rev1,
        rev2,
        normal1,
        normal2
    ]
    return output

import numpy as np

def solve_1(I):
    grid_np = np.array(I)
    out = np.zeros((9, 9), dtype=int)
    for i in range(3):
        for j in range(3):
            if grid_np[i, j] != 0:
                out[3*i:3*i+3, 3*j:3*j+3] = grid_np
    return out.tolist()

import copy

def solve_2(I):
    # Find positions of all 1's
    positions = []
    for r in range(len(I)):
        for c in range(len(I[r])):
            if I[r][c] == 1:
                positions.append((r, c))
    
    if not positions:
        # If no 1's, perhaps no change, but assume there are
        return I
    
    # Find min_r and min_c
    min_r = min(p[0] for p in positions)
    min_c = min(p[1] for p in positions)
    
    # Compute relative positions
    rel_pos = set((r - min_r, c - min_c) for r, c in positions)
    
    # Mapping from pattern to color
    pattern_to_color = {
        frozenset([(0,0), (0,1), (0,2), (1,0), (1,2), (2,1)]): 7,
        frozenset([(0,0), (0,2), (1,1), (2,0), (2,1), (2,2)]): 3,
        frozenset([(0,1), (1,0), (1,1), (1,2), (2,1)]): 2
    }
    
    if frozenset(rel_pos) not in pattern_to_color:
        # If unknown pattern, perhaps default, but assume it's known
        raise ValueError("Unknown blue pattern")
    
    color = pattern_to_color[frozenset(rel_pos)]
    
    # Create output
    output = copy.deepcopy(I)
    for r in range(len(output)):
        for c in range(len(output[r])):
            if output[r][c] == 8:
                output[r][c] = color
            elif output[r][c] == 1:
                output[r][c] = 0
    
    return output

from collections import deque

def solve_3(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    q = deque()
    # Add all border 0's to queue
    for r in range(rows):
        for c in [0, cols - 1]:
            if I[r][c] == 0 and not visited[r][c]:
                q.append((r, c))
                visited[r][c] = True
    for c in range(cols):
        for r in [0, rows - 1]:
            if I[r][c] == 0 and not visited[r][c]:
                q.append((r, c))
                visited[r][c] = True
    # Directions for 4-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # BFS to mark reachable 0's
    while q:
        r, c = q.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nr, nc))
    # Copy I and set unvisited 0's to 4
    output = [row[:] for row in I]
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 0 and not visited[r][c]:
                output[r][c] = 4
    return output

def solve_4(I):
    # Replace 1 with 2
    modified = [[2 if cell == 1 else cell for cell in row] for row in I]
    # Append the first len(I)//2 rows
    h = len(I)
    output = modified + modified[:h//2]
    return output

def solve_5(I):
    rows = len(I)
    cols = len(I[0])
    # Find blocks
    blocks = []
    for color in range(1, 10):
        positions = []
        for r in range(rows):
            for c in range(cols):
                if I[r][c] == color:
                    positions.append((r, c))
        if not positions:
            continue
        rs, cs = zip(*positions)
        min_r = min(rs)
        max_r = max(rs)
        min_c = min(cs)
        max_c = max(cs)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        blocks.append((min_c, color, height, width))
    # Sort by min_c
    blocks.sort()
    # Create output
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    current_right = -1
    current_bottom = -1
    for _, color, h, w in blocks:
        if current_right == -1:
            for cc in range(w):
                for rr in range(h):
                    if rr < rows and cc < cols:
                        output[rr][cc] = color
            current_right = w - 1
            current_bottom = h - 1
        else:
            attach_col = current_right
            # Change the bottom cell
            if current_bottom < rows:
                output[current_bottom][attach_col] = color
            # Add h-1 below
            for i in range(1, h):
                new_r = current_bottom + i
                if new_r < rows:
                    output[new_r][attach_col] = color
            # Update current_bottom
            current_bottom += (h - 1)
            # Add new columns
            start_row = current_bottom - h + 1
            for j in range(1, w):
                new_col = current_right + j
                if new_col >= cols:
                    break
                for rr in range(start_row, current_bottom + 1):
                    if 0 <= rr < rows:
                        output[rr][new_col] = color
            # Update current_right
            current_right += (w - 1)
    return output

def solve_6(I):
    # Extract left 3x3 (columns 0-2)
    left = [row[0:3] for row in I]
    # Extract right 3x3 (columns 4-6)
    right = [row[4:7] for row in I]
    # Initialize 3x3 output with 0s
    out = [[0] * 3 for _ in range(3)]
    # Set 2 where both left and right have 1
    for i in range(3):
        for j in range(3):
            if left[i][j] == 1 and right[i][j] == 1:
                out[i][j] = 2
    return out

from collections import defaultdict

def solve_7(I):
    # Collect colors for each residue modulo 3
    diag_colors = defaultdict(list)
    for r in range(len(I)):
        for c in range(len(I[0])):
            if I[r][c] != 0:
                k = r + c
                m = k % 3
                diag_colors[m].append(I[r][c])
    
    # Determine the unique color for each residue
    color_map = {}
    for m in diag_colors:
        vals = set(diag_colors[m])
        if len(vals) != 1:
            raise ValueError("Inconsistent colors for residue")
        color_map[m] = list(vals)[0]
    
    # Create the output I
    rows = len(I)
    cols = len(I[0])
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            m = (r + c) % 3
            if m in color_map:
                output[r][c] = color_map[m]
            else:
                # If a residue is missing, set to 0 (though not needed in examples)
                output[r][c] = 0
    
    return output

def solve_8(I):
    I = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    red_pos = []
    purple_pos = set()
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 2:
                red_pos.append((r, c))
            elif I[r][c] == 8:
                purple_pos.add((r, c))
    if not red_pos:
        return I

    min_r = min(r for r, c in red_pos)
    max_r = max(r for r, c in red_pos)
    min_c = min(c for r, c in red_pos)
    max_c = max(c for r, c in red_pos)

    min_pr = min(r for r, c in purple_pos)
    max_pr = max(r for r, c in purple_pos)
    min_pc = min(c for r, c in purple_pos)
    max_pc = max(c for r, c in purple_pos)

    overlap_rows = max(min_r, min_pr) <= min(max_r, max_pr)
    overlap_cols = max(min_c, min_pc) <= min(max_c, max_pc)

    if overlap_rows and not overlap_cols:
        move_horiz = True
    elif overlap_cols and not overlap_rows:
        move_horiz = False
    else:
        return I  # No move if unclear

    if move_horiz:
        if min_c > max_pc:
            dir_val = -1
        else:
            dir_val = 1
        dr = 0
        dc = dir_val
    else:
        if min_r > max_pr:
            dir_val = -1
        else:
            dir_val = 1
        dr = dir_val
        dc = 0

    max_s = 0
    for s in range(1, rows + cols):
        overlap = False
        new_pos = []
        for r, c in red_pos:
            nr = r + s * dr
            nc = c + s * dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or (nr, nc) in purple_pos:
                overlap = True
                break
            new_pos.append((nr, nc))
        if overlap:
            break
        max_s = s

    if max_s == 0:
        return I

    # Clear original
    for r, c in red_pos:
        I[r][c] = 0

    # Set new
    for nr, nc in new_pos:
        I[nr][nc] = 2

    return I

def solve_9(I):
    n = len(I)
    # Find the color
    color = 0
    for row in I:
        for c in row:
            if c != 0:
                color = c
                break
        if color != 0:
            break
    # Create inverse I
    inverse = [[color if I[i][j] == 0 else 0 for j in range(n)] for i in range(n)]
    # Create output I
    out_n = 3 * n
    output = [[0] * out_n for _ in range(out_n)]
    # Place inverse in activated blocks
    for i in range(n):
        for j in range(n):
            if I[i][j] != 0:
                for di in range(n):
                    for dj in range(n):
                        output[3 * i + di][3 * j + dj] = inverse[di][dj]
    return output

from collections import defaultdict

def solve_10(I):
    height = len(I)
    width = len(I[0])
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    pos = defaultdict(list)
    for r in range(height):
        for c in range(width):
            color = I[r][c]
            if color > 0:
                pos[color].append((r, c))
    
    horizontals = []
    verticals = []
    for color, positions in pos.items():
        if len(positions) == 2:
            (r1, c1), (r2, c2) = positions
            if r1 == r2:
                min_c = min(c1, c2)
                max_c = max(c1, c2)
                horizontals.append((r1, min_c, max_c, color))
            elif c1 == c2:
                min_r = min(r1, r2)
                max_r = max(r1, r2)
                verticals.append((c1, min_r, max_r, color))
    
    # Fill horizontals first
    for row, min_c, max_c, color in horizontals:
        for c in range(min_c, max_c + 1):
            output[row][c] = color
    
    # Then fill verticals, overwriting
    for col, min_r, max_r, color in verticals:
        for r in range(min_r, max_r + 1):
            output[r][col] = color
    
    return output

def solve_11(I):
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    
    # Find bars: list of (col, height)
    bars = []
    for c in range(cols):
        height = 0
        for r in range(rows - 1, -1, -1):
            if I[r][c] == 5:
                height += 1
            else:
                break  # Stop if gap found (assuming contiguous from bottom)
        if height > 0:
            bars.append((c, height))
    
    # Sort bars by height ascending
    bars.sort(key=lambda x: x[1])
    
    # Assign colors: 4 for shortest, down to 1 for tallest
    colors = [4, 3, 2, 1]
    col_to_color = {bars[i][0]: colors[i] for i in range(len(bars))}
    
    # Apply colors to output
    for c, height in bars:
        color = col_to_color[c]
        count = 0
        for r in range(rows - 1, -1, -1):
            if output[r][c] == 5:
                output[r][c] = color
                count += 1
            if count == height:
                break
    
    return output

def solve_12(I):
    height = len(I)
    width = len(I[0])
    output = [row[:] for row in I]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(height):
        for c in range(width):
            if I[r][c] == 0:
                continue
            neighbors = []
            all_valid = True
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    neighbors.append(I[nr][nc])
                else:
                    all_valid = False
                    break
            if not all_valid:
                continue
            colors = set(neighbors)
            if len(colors) == 1:
                B = list(colors)[0]
                if B > 0 and B != I[r][c]:
                    A = I[r][c]
                    for drr in range(-2, 3):
                        for dcc in range(-2, 3):
                            if max(abs(drr), abs(dcc)) > 2:
                                continue
                            if drr != 0 and dcc != 0 and abs(drr) != abs(dcc):
                                continue
                            nr = r + drr
                            nc = c + dcc
                            if 0 <= nr < height and 0 <= nc < width:
                                if abs(drr) == abs(dcc):
                                    output[nr][nc] = A
                                else:
                                    output[nr][nc] = B
    return output

def solve_13(I):
    height = len(I)
    width = len(I[0]) if height > 0 else 0
    
    # Find seeds
    seeds = []
    for r in range(height):
        for c in range(width):
            if I[r][c] != 0:
                seeds.append((r, c, I[r][c]))
    
    if height > width:
        # Vertical mode
        seeds.sort(key=lambda x: x[0])
        r1, _, A = seeds[0]
        r2, _, B = seeds[1]
        d = r2 - r1
        current = r1
        step = 0
        while current < height:
            color = A if step % 2 == 0 else B
            for cc in range(width):
                I[current][cc] = color
            current += d
            step += 1
    else:
        # Horizontal mode
        seeds.sort(key=lambda x: x[1])
        _, c1, A = seeds[0]
        _, c2, B = seeds[1]
        d = c2 - c1
        current = c1
        step = 0
        while current < width:
            color = A if step % 2 == 0 else B
            for rr in range(height):
                I[rr][current] = color
            current += d
            step += 1
    
    return I

def solve_14(I):
    n = len(I)
    ps = []
    for i in range(n):
        if I[i][i] == 1:
            ps.append(i)
    if len(ps) < 2:
        return I
    ps.sort()
    d = ps[1] - ps[0]
    for j in range(1, len(ps)):
        if ps[j] - ps[j - 1] != d:
            return I  # Not arithmetic
    current = ps[-1] + d
    while current < n:
        if I[current][current] == 0:
            I[current][current] = 2
        current += d
    return I

import numpy as np

def solve_15(I):
    I = np.array(I)
    rows, cols = I.shape

    # Find horizontal row h and color C
    h = None
    C = None
    for i in range(rows):
        if np.all(I[i] == I[i, 0]) and I[i, 0] != 0:
            h = i
            C = I[i, 0]
            break

    # Find vertical column v with color C
    v = None
    for j in range(cols):
        if np.all(I[:, j] == C):
            v = j
            break

    # Function to get shape positions and bounding box in a quadrant
    def get_shape(minrow, maxrow, mincol, maxcol):
        pos = []
        for r in range(minrow, maxrow + 1):
            for c in range(mincol, maxcol + 1):
                if I[r, c] != 0 and I[r, c] != C:
                    pos.append((r, c, I[r, c]))
        if not pos:
            return None
        min_r = min(p[0] for p in pos)
        max_r = max(p[0] for p in pos)
        min_c = min(p[1] for p in pos)
        max_c = max(p[1] for p in pos)
        return pos, min_r, max_r, min_c, max_c

    # Get shapes for each quadrant
    ul = get_shape(0, h - 1, 0, v - 1)
    ur = get_shape(0, h - 1, v + 1, cols - 1)
    ll = get_shape(h + 1, rows - 1, 0, v - 1)
    lr = get_shape(h + 1, rows - 1, v + 1, cols - 1)

    # Compute s (max side length across all shapes)
    all_sides = []
    for q in [ul, ur, ll, lr]:
        if q:
            _, mr, Mr, mc, Mc = q
            all_sides.append(Mr - mr + 1)
            all_sides.append(Mc - mc + 1)
    s = max(all_sides) if all_sides else 0

    # Create output I
    out = np.zeros((2 * s, 2 * s), dtype=int)

    # Function to place shape in output
    def place(pos, min_r, min_c, out_min_r, out_min_c):
        for r, c, col in pos:
            rel_r = r - min_r
            rel_c = c - min_c
            if rel_r < s and rel_c < s:  # In case of padding if sizes differ
                out[out_min_r + rel_r, out_min_c + rel_c] = col

    # Place shapes
    if ul:
        place(ul[0], ul[1], ul[3], 0, 0)
    if ur:
        place(ur[0], ur[1], ur[3], 0, s)
    if ll:
        place(ll[0], ll[1], ll[3], s, 0)
    if lr:
        place(lr[0], lr[1], lr[3], s, s)

    return out.tolist()

def solve_16(I):
    if not I or len(I) < 2 or len(I[0]) < 2:
        return I
    
    output = [row[:] for row in I]
    
    A = I[0][0]
    B = I[0][1]
    C = I[1][0]
    D = I[1][1]
    
    swap = {A: B, B: A, C: D, D: C}
    
    for r in range(len(I)):
        for c in range(len(I[0])):
            # Skip key positions
            if (r == 0 and c in (0, 1)) or (r == 1 and c in (0, 1)):
                continue
            if I[r][c] != 0:
                output[r][c] = swap.get(I[r][c], I[r][c])
    
    return output

def solve_17(I):
    row0 = I[0]
    row1 = I[1]
    row2 = I[2]

    def construct(r):
        rev = r[::-1]
        return rev + r

    line0 = construct(row2)  # for output rows 0 and 5
    line1 = construct(row1)  # for output rows 1 and 4
    line2 = construct(row0)  # for output rows 2 and 3

    output = [
        line0,
        line1,
        line2,
        line2,
        line1,
        line0
    ]
    return output

def solve_18(I):
    n = len(I)
    new_grid = [row[:] for row in I]
    directions_ortho = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    directions_diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for i in range(n):
        for j in range(n):
            color = I[i][j]  # Use original to avoid mid-iteration changes
            if color == 1:
                for di, dj in directions_ortho:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        new_grid[ni][nj] = 7
            elif color == 2:
                for di, dj in directions_diag:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        new_grid[ni][nj] = 4
    return new_grid

def solve_19(I):
    mapping = {1: 5, 5: 1, 2: 6, 6: 2, 3: 4, 4: 3, 8: 9, 9: 8, 0: 0, 7: 7}
    new_grid = [[mapping.get(cell, cell) for cell in row] for row in I]
    return new_grid

from collections import defaultdict, deque

def solve_20(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    blues = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 1:
                blues.append((r, c))
    col_rows = defaultdict(list)
    for r, c in blues:
        col_rows[c].append(r)
    row_cols = defaultdict(list)
    for r, c in blues:
        row_cols[r].append(c)
    line_pos = set()
    for c, rs in col_rows.items():
        if len(rs) >= 2:
            min_r = min(rs)
            max_r = max(rs)
            for rr in range(min_r, max_r + 1):
                line_pos.add((rr, c))
    for r, cs in row_cols.items():
        if len(cs) >= 2:
            min_c = min(cs)
            max_c = max(cs)
            for cc in range(min_c, max_c + 1):
                line_pos.add((r, cc))
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 2 and not visited[r][c]:
                component = set()
                q = deque([(r, c)])
                visited[r][c] = True
                component.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 2 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            component.add((nr, nc))
                components.append(component)
    output = [row[:] for row in I]
    for pos in line_pos:
        r, c = pos
        if I[r][c] == 0:
            output[r][c] = 1
    for comp in components:
        if comp & line_pos:
            for pos in comp:
                r, c = pos
                output[r][c] = 1
    return output

def solve_21(I):
    height = len(I)
    width = len(I[0])
    output = [[0] * width for _ in range(height)]
    
    # Collect seeds: (row, col, color)
    seeds = []
    for r in range(height):
        for c in range(width):
            if I[r][c] != 0:
                seeds.append((r, c, I[r][c]))
    
    # Sort by row
    seeds.sort(key=lambda x: x[0])
    n = len(seeds)
    if n == 0:
        return output
    
    # Compute ends
    ends = []
    for i in range(n - 1):
        ri = seeds[i][0]
        rj = seeds[i + 1][0]
        end = (ri + rj) // 2
        ends.append(end)
    ends.append(height - 1)
    
    # Compute starts
    starts = [0]
    for i in range(n - 1):
        next_start = ends[i] + 1
        starts.append(next_start)
    
    # For each bar
    for k in range(n):
        start_r = starts[k]
        end_r = ends[k]
        color = seeds[k][2]
        seed_r = seeds[k][0]
        
        # Set sides
        for r in range(start_r, end_r + 1):
            output[r][0] = color
            output[r][width - 1] = color
        
        # Set full at seed_r
        for c in range(width):
            output[seed_r][c] = color
        
        # If touches top
        if start_r == 0:
            for c in range(width):
                output[0][c] = color
        
        # If touches bottom
        if end_r == height - 1:
            for c in range(width):
                output[height - 1][c] = color
    
    return output

def solve_22(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    
    # Find min and max row and col with non-zero
    min_r, max_r, min_c, max_c = rows, 0, cols, 0
    has_non_zero = False
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0:
                has_non_zero = True
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if not has_non_zero:
        return [row[:] for row in I]
    
    center_r = (min_r + max_r) // 2
    center_c = (min_c + max_c) // 2
    
    output = [row[:] for row in I]
    
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0:
                color = I[r][c]
                # Vertical reflection
                rv = r
                cv = center_c + (center_c - c)
                if 0 <= cv < cols and output[rv][cv] == 0:
                    output[rv][cv] = color
                # Horizontal reflection
                rh = center_r + (center_r - r)
                ch = c
                if 0 <= rh < rows and output[rh][ch] == 0:
                    output[rh][ch] = color
                # Both (point reflection)
                rb = center_r + (center_r - r)
                cb = center_c + (center_c - c)
                if 0 <= rb < rows and 0 <= cb < cols and output[rb][cb] == 0:
                    output[rb][cb] = color
    
    return output

import numpy as np

def solve_23(I):
    I = np.array(I)
    rows, cols = I.shape
    colors = np.unique(I)
    if len(colors) != 2:
        raise ValueError("Expected exactly two colors")
    c1, c2 = colors

    def count_full(c):
        full_rows = sum(1 for r in range(rows) if np.all(I[r, :] == c))
        full_cols = sum(1 for cc in range(cols) if np.all(I[:, cc] == c))
        return full_rows, full_cols

    fr1, fc1 = count_full(c1)
    fr2, fc2 = count_full(c2)

    if fr1 + fc1 > 0 and fr2 + fc2 == 0:
        line_c = c1
        bg = c2
        num_h = fr1
        num_v = fc1
    elif fr2 + fc2 > 0 and fr1 + fc1 == 0:
        line_c = c2
        bg = c1
        num_h = fr2
        num_v = fc2
    else:
        raise ValueError("Cannot determine line color")

    out_rows = num_h + 1
    out_cols = num_v + 1
    return [[bg] * out_cols for _ in range(out_rows)]

def solve_24(I):
    I = [row[:] for row in I]  # copy
    rows = len(I)
    cols = len(I[0])
    colored = []
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != 0:
                colored.append((i, j, I[i][j]))
    if not colored:
        return I
    from collections import defaultdict
    row_count = defaultdict(int)
    row_cols = defaultdict(list)
    for i, j, _ in colored:
        row_count[i] += 1
        row_cols[i].append(j)
    max_count = max(row_count.values())
    if max_count < 2:
        return I  # No transformation if no row with multiple
    k = max(row_count, key=row_count.get)
    cols_list = sorted(row_cols[k])
    a = cols_list[0]
    b = cols_list[-1]
    c = (a + b) // 2
    d = (b - a) // 2
    candidates = []
    r1 = k - d
    if 0 <= r1 < rows:
        candidates.append(r1)
    r2 = k + d
    if 0 <= r2 < rows:
        candidates.append(r2)
    center_r = None
    center_c = c
    for cand_r in candidates:
        valid = True
        for ri, ci, _ in colored:
            if abs(ri - cand_r) != abs(ci - c):
                valid = False
                break
        if valid:
            center_r = cand_r
            break
    if center_r is None:
        return I  # No valid center
    # Set grey
    if I[center_r][center_c] == 0:
        I[center_r][center_c] = 5
    # Add adjacents
    for ri, ci, col in colored:
        dr = ri - center_r
        dc = ci - center_c
        s_dr = 1 if dr > 0 else -1 if dr < 0 else 0
        s_dc = 1 if dc > 0 else -1 if dc < 0 else 0
        if s_dr == 0 or s_dc == 0:
            continue  # Skip if not diagonal
        add_r = center_r + s_dr
        add_c = center_c + s_dc
        if 0 <= add_r < rows and 0 <= add_c < cols:
            if I[add_r][add_c] == 0:
                I[add_r][add_c] = col
    return I

def solve_25(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    
    # Find h: number of consecutive 5's in column 0 from top
    h = 0
    for r in range(rows):
        if I[r][0] == 5:
            h += 1
        else:
            break
    
    # Find the last row with any non-zero cell
    initial_last = 0
    for r in range(rows):
        if any(I[r][c] != 0 for c in range(cols)):
            initial_last = r
    initial_height = initial_last + 1
    
    # Calculate remaining rows and number of full repetitions
    remaining = rows - initial_height
    reps = remaining // h
    
    # Perform the repetitions
    for i in range(reps):
        for j in range(h):
            target_r = initial_height + i * h + j
            for c in range(cols):
                if c == 0:
                    I[target_r][c] = 0
                else:
                    I[target_r][c] = I[j][c]
    
    return I

def solve_26(I):
    rows = len(I)
    cols = len(I[0]) if rows > 0 else 0

    # Find blue (1) positions
    blue_pos = [(i, j) for i in range(rows) for j in range(cols) if I[i][j] == 1]
    if not blue_pos:
        return []

    min_r = min(r for r, _ in blue_pos)
    max_r = max(r for r, _ in blue_pos)
    min_c = min(c for _, c in blue_pos)
    max_c = max(c for _, c in blue_pos)
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    # Find single cells (color != 0 and != 1)
    singles = [(i, j, I[i][j]) for i in range(rows) for j in range(cols) if I[i][j] != 0 and I[i][j] != 1]

    # Determine direction
    all_cols = {j for _, j, _ in singles}
    all_rows = {i for i, _, _ in singles}
    if len(all_cols) == 1:
        direction = 'vertical'
        singles.sort(key=lambda x: x[0])  # sort by row
    elif len(all_rows) == 1:
        direction = 'horizontal'
        singles.sort(key=lambda x: x[1])  # sort by col
    else:
        raise ValueError("Singles not aligned vertically or horizontally")

    num = len(singles)

    if direction == 'vertical':
        out_h = h * num
        out_w = w
        output = [[0] * out_w for _ in range(out_h)]
        for idx, (_, _, color) in enumerate(singles):
            for i in range(h):
                for j in range(w):
                    if I[min_r + i][min_c + j] == 1:
                        output[idx * h + i][j] = color
    else:  # horizontal
        out_h = h
        out_w = w * num
        output = [[0] * out_w for _ in range(out_h)]
        for idx, (_, _, color) in enumerate(singles):
            for i in range(h):
                for j in range(w):
                    if I[min_r + i][min_c + j] == 1:
                        output[i][idx * w + j] = color

    return output

import numpy as np

def solve_27(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    output = [row[:] for row in I]
    visited = [[False for _ in range(width)] for _ in range(height)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def dfs(r, c, color, component):
        stack = [(r, c)]
        visited[r][c] = True
        component.append((r, c))
        
        while stack:
            cr, cc = stack.pop()
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and not visited[nr][nc] and I[nr][nc] == color:
                    visited[nr][nc] = True
                    component.append((nr, nc))
                    stack.append((nr, nc))
    
    for r in range(height):
        for c in range(width):
            if I[r][c] != 0 and not visited[r][c]:
                component = []
                dfs(r, c, I[r][c], component)
                if len(component) <= 2:
                    for cr, cc in component:
                        output[cr][cc] = 3
    
    return output

def solve_28(I):
    if not I or not I[0]:
        return I
    height = len(I)
    width = len(I[0])
    output = [row[:] for row in I]

    # Detect border
    direction = None
    border_pos = None
    if all(I[r][0] == 5 for r in range(height)):
        direction = 'left'
        border_pos = 0
    elif all(I[r][width - 1] == 5 for r in range(height)):
        direction = 'right'
        border_pos = width - 1
    elif all(I[0][c] == 5 for c in range(width)):
        direction = 'up'
        border_pos = 0
    elif all(I[height - 1][c] == 5 for c in range(width)):
        direction = 'down'
        border_pos = height - 1

    if direction is None:
        return output  # No border, no change

    # Find bars
    bars = []
    if direction in ['left', 'right']:
        # Vertical bars
        for c in range(width):
            r = 0
            while r < height:
                if I[r][c] == 0 or I[r][c] == 5:
                    r += 1
                    continue
                color = I[r][c]
                r_start = r
                r += 1
                while r < height and I[r][c] == color:
                    r += 1
                r_end = r - 1
                bars.append({'pos': c, 'start': r_start, 'end': r_end, 'color': color})
    else:
        # Horizontal bars
        for r in range(height):
            c = 0
            while c < width:
                if I[r][c] == 0 or I[r][c] == 5:
                    c += 1
                    continue
                color = I[r][c]
                c_start = c
                c += 1
                while c < width and I[r][c] == color:
                    c += 1
                c_end = c - 1
                bars.append({'pos': r, 'start': c_start, 'end': c_end, 'color': color})

    # Compute distances
    for bar in bars:
        if direction == 'right':
            bar['dist'] = border_pos - bar['pos']
        elif direction == 'left':
            bar['dist'] = bar['pos'] - border_pos
        elif direction == 'down':
            bar['dist'] = border_pos - bar['pos']
        elif direction == 'up':
            bar['dist'] = bar['pos'] - border_pos

    # Sort by decreasing distance
    bars.sort(key=lambda b: -b['dist'])

    # Paint in order
    for bar in bars:
        color = bar['color']
        if direction in ['left', 'right']:
            if direction == 'right':
                col_start = bar['pos']
                col_end = border_pos - 1
            else:
                col_start = border_pos + 1
                col_end = bar['pos']
            for cc in range(col_start, col_end + 1):
                for rr in range(bar['start'], bar['end'] + 1):
                    output[rr][cc] = color
        else:
            if direction == 'down':
                row_start = bar['pos']
                row_end = border_pos - 1
            else:
                row_start = border_pos + 1
                row_end = bar['pos']
            for rr in range(row_start, row_end + 1):
                for cc in range(bar['start'], bar['end'] + 1):
                    output[rr][cc] = color

    return output

def solve_29(I):
    rows = len(I)
    cols = len(I[0])

    # Find grays
    grays = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 5:
                grays.append((r, c))

    # Find colorful components
    visited = [[False] * cols for _ in range(rows)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connected
    for r in range(rows):
        for c in range(cols):
            color = I[r][c]
            if color != 0 and color != 5 and not visited[r][c]:
                component = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append((color, component))

    # Output I
    output = [[0 for _ in range(3)] for _ in range(3)]

    # 8 directions for adjacency
    adj_dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for color, comp in components:
        attached = set()
        for cr, cc in comp:
            for dr, dc in adj_dirs:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 5:
                    attached.add((nr, nc))
        if len(attached) != 1:
            continue  # In case, but examples have exactly 1
        gr, gc = list(attached)[0]
        for pr, pc in comp:
            dr = pr - gr
            dc = pc - gc
            or_ = 1 + dr
            oc = 1 + dc
            if 0 <= or_ < 3 and 0 <= oc < 3:
                output[or_][oc] = I[pr][pc]

    # Set center to 5
    output[1][1] = 5

    return output

def solve_30(I):
    I = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    
    # Find min_col and max_col
    min_col = cols
    max_col = -1
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 5:
                min_col = min(min_col, j)
                max_col = max(max_col, j)
    if max_col < 0:
        return I
    
    left_ext = min_col
    right_ext = cols - 1 - max_col
    
    arm_cols = set()
    is_grey = [False] * rows
    for r in range(rows):
        has_grey = any(I[r][j] == 5 for j in range(cols))
        is_grey[r] = has_grey
        if has_grey:
            for j in range(min_col, max_col + 1):
                if I[r][j] == 0:
                    I[r][j] = 2
                    arm_cols.add(j)
    
    # Fill empty bands
    r = 0
    while r < rows:
        if is_grey[r]:
            while r < rows and is_grey[r]:
                r += 1
            continue
        empty_start = r
        while r < rows and not is_grey[r]:
            r += 1
        h = r - empty_start
        is_top = (empty_start == 0)
        is_bottom = (r == rows)
        is_internal = not (is_top or is_bottom)
        if is_internal:
            for rr in range(empty_start, empty_start + h):
                for j in range(left_ext):
                    I[rr][j] = 1
                for j in range(min_col, max_col + 1):
                    I[rr][j] = 2
                for j in range(max_col + 1, max_col + 1 + right_ext):
                    I[rr][j] = 1
        else:
            for rr in range(empty_start, empty_start + h):
                for j in arm_cols:
                    I[rr][j] = 1
    
    return I

def solve_31(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    
    # Find seed positions (where color is 1)
    seeds = []
    for r in range(height):
        for c in range(width):
            if I[r][c] == 1:
                seeds.append((r, c))
    
    # Create output I as copy
    output = [row[:] for row in I]
    
    # Set the crosses for each seed
    for r, c in seeds:
        # Set the entire row to 1, center to 2
        for j in range(width):
            output[r][j] = 1
        output[r][c] = 2
        
        # Set the entire column to 1 (skip center as it's already 2)
        for i in range(height):
            if i != r:
                output[i][c] = 1
    
    # Set the green diagonals for each seed
    for r, c in seeds:
        for dr in [-1, 1]:
            for dc in [-1, 1]:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    output[nr][nc] = 3
    
    return output

def solve_32(I):
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 7 and not visited[r][c]:
                component = []
                stack = [(r, c)]
                visited[r][c] = True
                color = I[r][c]
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))

                # Check if all have exactly 2 neighbors
                is_cycle = True
                for cr, cc in component:
                    neigh_count = 0
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == color:
                            neigh_count += 1
                    if neigh_count != 2:
                        is_cycle = False
                        break

                if is_cycle:
                    # Process bends
                    for cr, cc in component:
                        neigh_dirs = []
                        for i, (dr, dc) in enumerate(directions):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == color:
                                neigh_dirs.append(i)
                        # Should be exactly 2
                        if len(neigh_dirs) == 2:
                            i1, i2 = sorted(neigh_dirs)
                            # Not opposite
                            if not ((i1 == 0 and i2 == 1) or (i1 == 2 and i2 == 3)):
                                # Bend, find missing
                                all_dirs = {0, 1, 2, 3}
                                missing = all_dirs - set(neigh_dirs)
                                for mi in missing:
                                    dr, dc = directions[mi]
                                    nr, nc = cr + dr, cc + dc
                                    if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 7:
                                        output[nr][nc] = 2

    return output

import numpy as np

def solve_33(I):
    n = 3
    out = np.zeros((3 * n, 3 * n), dtype=int)
    grid_np = np.array(I)
    
    # Check for monochromatic rows
    mono_row = -1
    for i in range(n):
        if np.all(grid_np[i, :] == grid_np[i, 0]):
            mono_row = i
            break
    
    if mono_row != -1:
        # Horizontal tiling in super row mono_row
        for ii in range(n):
            for jj in range(3 * n):
                out[3 * mono_row + ii, jj] = grid_np[ii, jj % n]
    else:
        # Check for monochromatic columns
        mono_col = -1
        for j in range(n):
            if np.all(grid_np[:, j] == grid_np[0, j]):
                mono_col = j
                break
        if mono_col != -1:
            # Vertical tiling in super column mono_col
            for jj in range(n):
                for ii in range(3 * n):
                    out[ii, 3 * mono_col + jj] = grid_np[ii % n, jj]
    
    return out.tolist()

def solve_34(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    output = [[0 for _ in range(width)] for _ in range(height)]
    vertical_color = 2
    vertical_columns = set()
    horizontal_rows = {}
    
    for r in range(height):
        for c in range(width):
            color = I[r][c]
            if color != 0:
                if color == vertical_color:
                    vertical_columns.add(c)
                else:
                    horizontal_rows[r] = color  # Assumes no conflict in row
    
    # Place verticals
    for c in vertical_columns:
        for r in range(height):
            output[r][c] = vertical_color
    
    # Place horizontals, overriding
    for r, color in horizontal_rows.items():
        for c in range(width):
            output[r][c] = color
    
    return output

import numpy as np

def solve_35(I):
    if not I or not I[0]:
        return I
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    bottom_row = rows - 1
    for c in range(cols):
        if I[bottom_row][c] == 5:
            # Collect colored positions
            colored = []
            for r in range(rows):
                if I[r][c] != 0:
                    colored.append((r, I[r][c]))
            if not colored:
                continue
            # Fill segments
            prev_r = -1
            for r, color in colored:
                start = prev_r + 1
                end = r
                for rr in range(start, end + 1):
                    output[rr][c] = color
                prev_r = r
    return output

import numpy as np
from collections import deque

def solve_36(I):
    I = np.array(I)
    rows, cols = I.shape
    visited = np.zeros_like(I, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    output = I.copy()

    for i in range(rows):
        for j in range(cols):
            if I[i, j] == 8 and not visited[i, j]:
                component = []
                queue = deque([(i, j)])
                visited[i, j] = True
                while queue:
                    x, y = queue.popleft()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and I[nx, ny] == 8 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            queue.append((nx, ny))

                if not component:
                    continue

                min_r = min(r for r, c in component)
                max_r = max(r for r, c in component)
                min_c = min(c for r, c in component)
                max_c = max(c for r, c in component)
                h = max_r - min_r + 1
                w = max_c - min_c + 1

                if w > h:
                    axis = 'vertical'
                    center = (min_c + max_c) / 2.0
                else:
                    axis = 'horizontal'
                    center = (min_r + max_r) / 2.0

                for rr in range(min_r, max_r + 1):
                    for cc in range(min_c, max_c + 1):
                        if I[rr, cc] == 2:
                            if axis == 'vertical':
                                cc2 = round(2 * center - cc)
                                rr2 = rr
                                if 0 <= cc2 < cols and I[rr2, cc2] == 0:
                                    output[rr2, cc2] = 2
                            else:
                                rr2 = round(2 * center - rr)
                                cc2 = cc
                                if 0 <= rr2 < rows and I[rr2, cc2] == 0:
                                    output[rr2, cc2] = 2

    return output.tolist()

def solve_37(I):
    out = []
    for r in range(5):
        left = [1 if I[r][c] == 7 else 0 for c in range(6)]
        right = [1 if I[r][c] == 7 else 0 for c in range(7, 13)]
        row = [1 if left[c] or right[c] else 0 for c in range(6)]
        out.append(row)
    return out

from collections import deque

def solve_38(I):
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def bfs(start_r, start_c):
        component = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == 2:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return component

    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 2 and not visited[r][c]:
                comp = bfs(r, c)
                components.append(comp)

    comps_sorted = sorted(components, key=lambda comp: min(r for r, c in comp))

    top_comps = comps_sorted[:2]
    bottom_comps = comps_sorted[2:]

    def get_min_c(comp):
        return min(c for r, c in comp)

    top_left_comp = sorted(top_comps, key=get_min_c)[0]
    top_right_comp = sorted(top_comps, key=get_min_c)[1]
    bottom_left_comp = sorted(bottom_comps, key=get_min_c)[0]
    bottom_right_comp = sorted(bottom_comps, key=get_min_c)[1]

    def get_pattern(comp):
        min_r = min(r for r, c in comp)
        min_c = min(c for r, c in comp)
        pat = [[0] * 3 for _ in range(3)]
        for r, c in comp:
            pat[r - min_r][c - min_c] = 2
        return pat

    top_left = get_pattern(top_left_comp)
    top_right = get_pattern(top_right_comp)
    bottom_left = get_pattern(bottom_left_comp)
    bottom_right = get_pattern(bottom_right_comp)

    top_rows = []
    for i in range(3):
        top_rows.append(top_left[i] + [0] + top_right[i])

    bottom_rows = []
    for i in range(3):
        bottom_rows.append(bottom_left[i] + [0] + bottom_right[i])

    output = top_rows + [[0] * 7] + bottom_rows
    return output

import numpy as np
from collections import defaultdict

def solve_39(I):
    I = np.array(grid_lst)
    rows, cols = I.shape

    # Find purple bounds
    purple_rows = np.any(I == 8, axis=1)
    min_r = np.where(purple_rows)[0][0]
    max_r = np.where(purple_rows)[0][-1]
    purple_cols = np.any(I == 8, axis=0)
    min_c = np.where(purple_cols)[0][0]
    max_c = np.where(purple_cols)[0][-1]
    center_c = (min_c + max_c) / 2

    # Find blocks
    visited = np.zeros_like(I, dtype=bool)
    blocks = []
    for r in range(rows):
        for c in range(cols):
            color = I[r, c]
            if color != 0 and color != 8 and not visited[r, c]:
                component = []
                stack = [(r, c)]
                visited[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    component.append((rr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and I[nr, nc] == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                rs = [rr for rr, _ in component]
                cs = [cc for _, cc in component]
                block_min_r = min(rs)
                block_min_c = min(cs)
                block_center_c = (min(cs) + max(cs)) / 2
                blocks.append({'color': color, 'min_r': block_min_r, 'min_c': block_min_c, 'center_c': block_center_c})

    # Group by min_r
    groups = defaultdict(list)
    for b in blocks:
        groups[b['min_r']].append(b)

    # Sort group rows
    group_rows = sorted(groups.keys())

    # Assume exactly two groups
    top_min_r = group_rows[0]
    bottom_min_r = group_rows[1]

    top_blocks = sorted(groups[top_min_r], key=lambda b: b['min_c'])
    bottom_blocks = sorted(groups[bottom_min_r], key=lambda b: b['min_c'])

    # Fill top row
    top_row = [0, 0]
    if len(top_blocks) == 2:
        top_row[0] = top_blocks[0]['color']
        top_row[1] = top_blocks[1]['color']
    elif len(top_blocks) == 1:
        b = top_blocks[0]
        if b['center_c'] <= center_c:
            top_row[0] = b['color']
        else:
            top_row[1] = b['color']

    # Fill bottom row
    bottom_row = [0, 0]
    if len(bottom_blocks) == 2:
        bottom_row[0] = bottom_blocks[0]['color']
        bottom_row[1] = bottom_blocks[1]['color']
    elif len(bottom_blocks) == 1:
        b = bottom_blocks[0]
        if b['center_c'] <= center_c:
            bottom_row[0] = b['color']
        else:
            bottom_row[1] = b['color']

    return [top_row, bottom_row]

import numpy as np

def solve_40(I):
    if not I or not I[0]:
        return I
    
    I = np.array(I)
    rows, cols = I.shape
    
    # Find horizontal bars
    h_bars = {}  # color: list of row indices
    has_h = False
    for i in range(rows):
        if np.all(I[i] == I[i,0]) and I[i,0] > 0:
            c = I[i,0]
            h_bars.setdefault(c, []).append(i)
            has_h = True
    
    # Find vertical bars
    v_bars = {}  # color: list of col indices
    has_v = False
    for j in range(cols):
        if np.all(I[:,j] == I[0,j]) and I[0,j] > 0:
            c = I[0,j]
            v_bars.setdefault(c, []).append(j)
            has_v = True
    
    # Determine orientation
    if has_h:
        orientation = 'h'
        color_to_bars = h_bars
    elif has_v:
        orientation = 'v'
        color_to_bars = v_bars
    else:
        # No bars, remove all non-zero
        I[I > 0] = 0
        return I.tolist()
    
    # Output I
    output = I.copy()
    
    # Process each cell
    for i in range(rows):
        for j in range(cols):
            val = I[i, j]
            if val == 0:
                continue
            # Check if part of bar
            is_bar = False
            if orientation == 'h':
                if i in color_to_bars.get(val, []):
                    is_bar = True
            else:
                if j in color_to_bars.get(val, []):
                    is_bar = True
            if is_bar:
                continue
            # Movable
            if val not in color_to_bars or not color_to_bars[val]:
                output[i, j] = 0
                continue
            bars = color_to_bars[val]
            # Find nearest
            min_dist = float('inf')
            nearest = None
            current = i if orientation == 'h' else j
            for b in bars:
                dist = abs(current - b)
                if dist < min_dist:
                    min_dist = dist
                    nearest = b
            # Attach
            if current < nearest:
                attach = nearest - 1
            else:
                attach = nearest + 1
            if orientation == 'h':
                attach_i = attach
                attach_j = j
            else:
                attach_i = i
                attach_j = attach
            # Set
            output[attach_i, attach_j] = val
            output[i, j] = 0
    
    return output.tolist()

import numpy as np

def solve_41(I):
    I = np.array(grid_lst)
    rows, cols = I.shape
    candidates = set()
    # Check rows
    for i in range(rows):
        val = I[i, 0]
        if val != 0 and np.all(I[i, :] == val):
            candidates.add(val)
    # Check columns
    for j in range(cols):
        val = I[0, j]
        if val != 0 and np.all(I[:, j] == val):
            candidates.add(val)
    # Assuming exactly one candidate as per examples
    assert len(candidates) == 1
    c = next(iter(candidates))
    return [[c]]

def solve_42(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    # Assuming fixed size 5x7, but to be general, compute cols
    cols = len(I[0])
    sub_cols = 3  # fixed based on examples
    
    # Extract left and right
    left = [row[0:sub_cols] for row in I]
    right = [row[cols - sub_cols:cols] for row in I]
    
    # Create output
    output = [[0 for _ in range(sub_cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(sub_cols):
            if left[r][c] == 0 and right[r][c] == 0:
                output[r][c] = 8
    
    return output

def solve_43(I):
    height = len(I)
    width = len(I[0])
    output = [[0] * width for _ in range(height)]
    
    # Collect seeds: (row, col, color)
    seeds = []
    for r in range(height):
        for c in range(width):
            if I[r][c] != 0:
                seeds.append((r, c, I[r][c]))
    
    # Sort by row
    seeds.sort(key=lambda x: x[0])
    n = len(seeds)
    if n == 0:
        return output
    
    # Compute ends
    ends = []
    for i in range(n - 1):
        ri = seeds[i][0]
        rj = seeds[i + 1][0]
        end = (ri + rj) // 2
        ends.append(end)
    ends.append(height - 1)
    
    # Compute starts
    starts = [0]
    for i in range(n - 1):
        next_start = ends[i] + 1
        starts.append(next_start)
    
    # For each band
    for k in range(n):
        start_r = starts[k]
        end_r = ends[k]
        color = seeds[k][2]
        seed_r = seeds[k][0]
        
        # Set sides
        for r in range(start_r, end_r + 1):
            output[r][0] = color
            output[r][width - 1] = color
        
        # Set full at seed_r
        for c in range(width):
            output[seed_r][c] = color
        
        # If touches top
        if start_r == 0:
            for c in range(width):
                output[0][c] = color
        
        # If touches bottom
        if end_r == height - 1:
            for c in range(width):
                output[height - 1][c] = color
    
    return output

from collections import defaultdict

def solve_44(I):
    if not I or not I[0]:
        return I

    height = len(I)
    width = len(I[0])

    # Copy I
    output = [row[:] for row in I]

    # Set all seeds to 0
    for r in range(height):
        for c in range(width):
            if output[r][c] != 0 and output[r][c] != 5:
                output[r][c] = 0

    # Group seeds by color
    groups = defaultdict(list)
    for r in range(height):
        for c in range(width):
            val = I[r][c]
            if val != 0 and val != 5:
                groups[val].append((r, c))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for color, poss in groups.items():
        entry_points = set()
        for r, c in poss:
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and I[nr][nc] == 5:
                    entry_points.add((nr, nc))

        if not entry_points:
            continue

        min_r = min(x[0] for x in entry_points)
        max_r = max(x[0] for x in entry_points)
        min_c = min(x[1] for x in entry_points)
        max_c = max(x[1] for x in entry_points)

        for fr in range(min_r, max_r + 1):
            for fc in range(min_c, max_c + 1):
                output[fr][fc] = color

    return output

import numpy as np

def solve_45(I):
    I = np.array(grid_lst)
    rows, cols = I.shape
    
    for c in range(1, 10):
        r_idx, c_idx = np.where(I == c)
        if len(r_idx) == 0:
            continue
        min_r = r_idx.min()
        max_r = r_idx.max()
        min_c = c_idx.min()
        max_c = c_idx.max()
        
        if max_r - min_r < 2 or max_c - min_c < 2:
            continue
        
        # Check top and bottom borders
        if not np.all(I[min_r, min_c:max_c+1] == c):
            continue
        if not np.all(I[max_r, min_c:max_c+1] == c):
            continue
        # Check left and right borders
        if not np.all(I[min_r:max_r+1, min_c] == c):
            continue
        if not np.all(I[min_r:max_r+1, max_c] == c):
            continue
        # Check no c in internal
        if np.any(I[min_r+1:max_r, min_c+1:max_c] == c):
            continue
        
        # If all checks pass, extract internal
        return I[min_r+1:max_r, min_c+1:max_c].tolist()
    
    # If no frame found, return empty (though not expected)
    return []

def solve_46(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    min_r = rows
    max_r = -1
    min_c = cols
    max_c = -1
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if max_r < 0:
        return []
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    new_grid = [[0] * width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            new_grid[i][j] = I[min_r + i][min_c + j]
    return new_grid

import numpy as np

def solve_47(I):
    if not I:
        return I
    height = len(I)
    width = len(I[0])
    output = [row[:] for row in I]

    # Find the bounding box of the non-zero cells
    non_zero_positions = [(r, c) for r in range(height) for c in range(width) if I[r][c] != 0]
    if not non_zero_positions:
        return output
    min_row = min(r for r, c in non_zero_positions)
    max_row = max(r for r, c in non_zero_positions)
    min_col = min(c for r, c in non_zero_positions)
    max_col = max(c for r, c in non_zero_positions)

    # Assuming it's 3x3
    if max_row - min_row != 2 or max_col - min_col != 2:
        return output  # Not matching expected structure

    central_row = min_row + 1
    upper_row = min_row
    lower_row = min_row + 2

    # Extend central row horizontally
    left_color = I[central_row][min_col]
    right_color = I[central_row][min_col + 2]
    for c in range(min_col):
        output[central_row][c] = left_color
    for c in range(min_col + 3, width):
        output[central_row][c] = right_color

    # Extend upper row upwards
    u_l_color = I[upper_row][min_col]
    u_m_color = I[upper_row][min_col + 1]
    u_r_color = I[upper_row][min_col + 2]
    d = 1
    while True:
        row = upper_row - d
        if row < 0:
            break
        # Left up-left
        col = min_col - d
        if 0 <= col < width:
            output[row][col] = u_l_color
        # Middle up
        col = min_col + 1
        if 0 <= col < width:
            output[row][col] = u_m_color
        # Right up-right
        col = (min_col + 2) + d
        if 0 <= col < width:
            output[row][col] = u_r_color
        d += 1

    # Extend lower row downwards
    l_l_color = I[lower_row][min_col]
    l_m_color = I[lower_row][min_col + 1]
    l_r_color = I[lower_row][min_col + 2]
    d = 1
    while True:
        row = lower_row + d
        if row >= height:
            break
        # Left down-left
        col = min_col - d
        if 0 <= col < width:
            output[row][col] = l_l_color
        # Middle down
        col = min_col + 1
        if 0 <= col < width:
            output[row][col] = l_m_color
        # Right down-right
        col = (min_col + 2) + d
        if 0 <= col < width:
            output[row][col] = l_r_color
        d += 1

    return output

def solve_48(I):
    height = len(I)
    if height == 0:
        return I
    width = len(I[0])
    output = [[0] * width for _ in range(height)]
    for c in range(width):
        col_list = [I[r][c] for r in range(height) if I[r][c] != 0]
        k = len(col_list)
        for i in range(k):
            output[height - k + i][c] = col_list[i]
    return output

def solve_49(I):
    if not I or not I[0]:
        return I
    output = [row[:] for row in I]
    special_row = 1
    special_col = 1
    c = I[special_row][special_col]
    if c == 0:
        return output
    for i in range(len(I)):
        for j in range(len(I[0])):
            if I[i][j] == c and (i != special_row or j != special_col):
                output[i][j] = 0
    return output

import numpy as np

def solve_50(I):
    I = np.array(grid_lst)
    rows, cols = I.shape

    # Find the bounding box of the purple (8) rectangle
    purple_positions = np.argwhere(I == 8)
    if purple_positions.size == 0:
        return I.tolist()  # No purple, no changes
    min_row, min_col = np.min(purple_positions, axis=0)
    max_row, max_col = np.max(purple_positions, axis=0)

    # Iterate over all cells
    for row in range(rows):
        for col in range(cols):
            color = I[row, col]
            if color != 0 and color != 8:
                # Above or below
                if min_col <= col <= max_col:
                    if row < min_row:
                        I[min_row, col] = color
                    elif row > max_row:
                        I[max_row, col] = color
                # Left or right
                if min_row <= row <= max_row:
                    if col < min_col:
                        I[row, min_col] = color
                    elif col > max_col:
                        I[row, max_col] = color

    return I.tolist()

def solve_51(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != 0 and not visited[i][j]:
                color = I[i][j]
                comp = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append((len(comp), comp, color))
    if not components:
        return []
    components.sort(key=lambda x: x[0], reverse=True)
    _, max_comp, max_color = components[0]
    if not max_comp:
        return []
    min_r = min(r for r, c in max_comp)
    max_r = max(r for r, c in max_comp)
    min_c = min(c for r, c in max_comp)
    max_c = max(c for r, c in max_comp)
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    new_grid = [[0] * width for _ in range(height)]
    for r, c in max_comp:
        new_grid[r - min_r][c - min_c] = max_color
    return new_grid

from collections import defaultdict

def solve_52(I):
    n = len(I)
    new_grid = [row[:] for row in I]
    color_pos = defaultdict(list)
    for i in range(n):
        for j in range(n):
            c = I[i][j]
            if c != 0:
                color_pos[c].append((i, j))
    for c, pos in color_pos.items():
        if len(pos) < 2:
            continue
        diffs = set(i - j for i, j in pos)
        sums = set(i + j for i, j in pos)
        rows = [i for i, j in pos]
        min_r = min(rows)
        max_r = max(rows)
        if len(diffs) == 1:
            d = list(diffs)[0]
            for r in range(min_r, max_r + 1):
                jj = r - d
                if 0 <= jj < n:
                    new_grid[r][jj] = c
        elif len(sums) == 1:
            s = list(sums)[0]
            for r in range(min_r, max_r + 1):
                jj = s - r
                if 0 <= jj < n:
                    new_grid[r][jj] = c
    return new_grid

def solve_53(I):
    if not I or not I[0]:
        return []
    height = len(I)
    width = len(I[0])
    
    non_zero_positions = [(r, c) for r in range(height) for c in range(width) if I[r][c] != 0]
    
    if not non_zero_positions:
        return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # Assuming all black returns all black 3x3, though not in examples
    
    min_r = min(r for r, c in non_zero_positions)
    min_c = min(c for r, c in non_zero_positions)
    
    output = [[0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if min_r + i < height and min_c + j < width:
                output[i][j] = I[min_r + i][min_c + j]
    
    return output

import numpy as np

def solve_54(I):
    g = np.array(I)
    out = np.zeros((6, 6), dtype=int)
    for i in range(3):
        for j in range(3):
            if g[i, j] == 5:
                out[2*i:2*i+2, 2*j:2*j+2] = [[1, 2], [2, 1]]
    return out.tolist()

def solve_55(I):
    if not I or not I[0]:
        return []
    
    rows = len(I)
    cols = len(I[0])
    bg = I[0][0]
    visited = [[False] * cols for _ in range(rows)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != bg and not visited[i][j]:
                color = I[i][j]
                stack = [(i, j)]
                visited[i][j] = True
                pos = [(i, j)]
                min_r, max_r = i, i
                min_c, max_c = j, j
                while stack:
                    r, c = stack.pop()
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                            pos.append((nr, nc))
                            min_r = min(min_r, nr)
                            max_r = max(max_r, nr)
                            min_c = min(min_c, nc)
                            max_c = max(max_c, nc)
                area = len(pos)
                h = max_r - min_r + 1 if pos else 0
                w = max_c - min_c + 1 if pos else 0
                components.append((area, h, w, color))
    
    components.sort(key=lambda x: x[0])
    
    if not components:
        return []
    
    max_h = max(comp[1] for comp in components)
    max_w = max(comp[2] for comp in components)
    output = [[0 for _ in range(max_w)] for _ in range(max_h)]
    
    current_h = 0
    current_w = 0
    for _, comp_h, comp_w, color in components:
        new_h = max(current_h, comp_h)
        new_w = max(current_w, comp_w)
        # Fill added rows
        for r in range(current_h, new_h):
            for c in range(current_w):
                output[r][c] = color
        # Fill added columns
        for r in range(new_h):
            for c in range(current_w, new_w):
                output[r][c] = color
        current_h = new_h
        current_w = new_w
    
    return output

def solve_56(I):
    # Extract block colors
    colorL = I[0][0]
    colorM = I[0][4]
    colorR = I[0][8]
    
    # Extract representatives
    repL = I[4][1]
    repM = I[4][5]
    repR = I[4][9]
    
    # Mapping
    mapping = {repL: colorL, repM: colorM, repR: colorR}
    
    # Initialize output 13x11 with 7
    output = [[7] * 11 for _ in range(13)]
    
    # Place blocks for each representative in rows 7-19
    for in_r in range(7, 20):
        for c in range(11):
            col = I[in_r][c]
            if col != 7 and col in mapping:
                block_col = mapping[col]
                out_r = in_r - 7
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr = out_r + dr
                        nc = c + dc
                        if 0 <= nr < 13 and 0 <= nc < 11:
                            output[nr][nc] = block_col
    
    return output

def solve_57(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    
    # Find seeds where color is 2
    seeds = [(i, j) for i in range(height) for j in range(width) if I[i][j] == 2]
    
    if not seeds:
        return [row[:] for row in I]
    
    # Collect unique rows and columns
    horiz_rows = set(r for r, c in seeds)
    vert_cols = set(c for r, c in seeds)
    
    # Create output I initialized to 0
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    # Draw horizontal red lines
    for r in horiz_rows:
        for j in range(width):
            output[r][j] = 2
    
    # Draw vertical red lines
    for c in vert_cols:
        for i in range(height):
            output[i][c] = 2
    
    # Fill interior with blue (1) if possible
    if len(horiz_rows) >= 2 and len(vert_cols) >= 2:
        min_r = min(horiz_rows)
        max_r = max(horiz_rows)
        min_c = min(vert_cols)
        max_c = max(vert_cols)
        for i in range(min_r + 1, max_r):
            for j in range(min_c + 1, max_c):
                output[i][j] = 1
    
    return output

def solve_58(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    
    # Check for horizontal mode
    top_color = I[0][0] if all(c == I[0][0] and c != 0 for c in I[0]) else None
    bottom_color = I[-1][0] if all(c == I[-1][0] and c != 0 for c in I[-1]) else None
    horizontal_mode = top_color is not None and bottom_color is not None
    
    # Check for vertical mode
    left_color = I[0][0] if all(I[r][0] == I[0][0] and I[r][0] != 0 for r in range(rows)) else None
    right_color = I[0][-1] if all(I[r][-1] == I[0][-1] and I[r][-1] != 0 for r in range(rows)) else None
    vertical_mode = left_color is not None and right_color is not None
    
    # Assume only one mode is true, as per examples
    if horizontal_mode:
        for r in range(rows):
            for c in range(cols):
                if I[r][c] == 3:
                    dist_top = r
                    dist_bottom = rows - 1 - r
                    if dist_top < dist_bottom:
                        output[r][c] = top_color
                    else:
                        output[r][c] = bottom_color
    elif vertical_mode:
        for r in range(rows):
            for c in range(cols):
                if I[r][c] == 3:
                    dist_left = c
                    dist_right = cols - 1 - c
                    if dist_left < dist_right:
                        output[r][c] = left_color
                    else:
                        output[r][c] = right_color
    
    return output

import copy
from collections import defaultdict

def solve_59(I):
    out = copy.deepcopy(I)
    rows = len(I)
    cols = len(I[0])
    for r in range(rows):
        pos = defaultdict(list)
        for c in range(cols):
            colr = I[r][c]
            if colr != 0:
                pos[colr].append(c)
        for colr, clist in pos.items():
            if clist:
                minc = min(clist)
                maxc = max(clist)
                for cc in range(minc, maxc + 1):
                    out[r][cc] = colr
    return out

from collections import defaultdict

def solve_60(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 7 and not visited[r][c]:
                color = I[r][c]
                component = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append((color, component))
    color_to_comps = defaultdict(list)
    for colr, comp in components:
        color_to_comps[colr].append(comp)
    output = [row[:] for row in I]
    for colr, comps in color_to_comps.items():
        if len(comps) <= 1:
            continue
        for comp in comps:
            for r, c in comp:
                output[r][c] = 7
        for comp in comps:
            if not comp:
                continue
            min_r = min(rr for rr, cc in comp)
            max_r = max(rr for rr, cc in comp)
            min_c = min(cc for rr, cc in comp)
            max_c = max(cc for rr, cc in comp)
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            dr = 0
            dc = 0
            if min_r == 0:
                dr += height
            if max_r == rows - 1:
                dr -= height
            if min_c == 0:
                dc += width
            if max_c == cols - 1:
                dc -= width
            for rr, cc in comp:
                new_r = rr + dr
                new_c = cc + dc
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    output[new_r][new_c] = colr
    return output

import numpy as np
from collections import deque

def solve_61(I):
    I = np.array(I)
    rows, cols = I.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components = []

    def get_neighbors(r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    # Find all connected components of non-7 cells
    for i in range(rows):
        for j in range(cols):
            if I[i, j] != 7 and not visited[i, j]:
                color = I[i, j]
                component = []
                queue = deque([(i, j)])
                visited[i, j] = True
                while queue:
                    r, c = queue.popleft()
                    component.append((r, c))
                    for nr, nc in get_neighbors(r, c):
                        if not visited[nr, nc] and I[nr, nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                components.append((color, component))

    # Find the plus component
    plus_comp = None
    insert_color = None
    for color, comp in components:
        if len(comp) == 5:
            # Check degrees
            cell_set = set(comp)
            degrees = {}
            for r, c in comp:
                deg = sum((nr, nc) in cell_set for nr, nc in get_neighbors(r, c))
                degrees[(r, c)] = deg
            num_deg4 = sum(1 for d in degrees.values() if d == 4)
            num_deg1 = sum(1 for d in degrees.values() if d == 1)
            if num_deg4 == 1 and num_deg1 == 4:
                plus_comp = comp
                insert_color = color
                break

    if insert_color is None:
        return I.tolist()  # No plus, no change

    # Find target_color (assuming one other color)
    target_colors = set(color for color, _ in components if color != insert_color)
    if len(target_colors) != 1:
        target_color = None  # But in puzzle it's 1
    else:
        target_color = list(target_colors)[0]

    # Remove plus if insert_color < target_color
    if target_color is not None and insert_color < target_color:
        for r, c in plus_comp:
            I[r, c] = 7

    # Change centers of odd-sized target components
    for color, comp in components:
        if color != insert_color and len(comp) % 2 == 1:
            rs = [r for r, c in comp]
            cs = [c for r, c in comp]
            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs), max(cs)
            center_r = min_r + (max_r - min_r) // 2
            center_c = min_c + (max_c - min_c) // 2
            I[center_r, center_c] = insert_color

    return I.tolist()

def solve_62(I):
    new_grid = [row[:] for row in I]
    cols = len(I[0])
    rows = len(I)
    
    # Find pattern columns from row 0
    pattern_cols = [c for c in range(cols) if I[0][c] == 5]
    
    # Find target rows where column 9 == 5 and r != 0
    target_rows = [r for r in range(1, rows) if I[r][9] == 5]
    
    # Set 2 in target rows at pattern columns
    for r in target_rows:
        for c in pattern_cols:
            new_grid[r][c] = 2
    
    return new_grid

def solve_63(I):
    rows = len(I)
    if rows == 0:
        return I
    cols = len(I[0])
    output = [row[:] for row in I]
    for r in range(rows):
        left = output[r][0]
        right = output[r][cols - 1]
        if left != 0 and right != 0 and left == right:
            middle_zero = all(output[r][c] == 0 for c in range(1, cols - 1))
            if middle_zero:
                for c in range(cols):
                    output[r][c] = left
    return output

def solve_64(I):
    if not I or not I[0]:
        return I
    
    h = len(I)
    w = len(I[0])
    output = [row[:] for row in I]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def get_chain(sr, sc):
        chain = [(sr, sc)]
        cr, cc = sr, sc
        while True:
            next_pos = None
            count = 0
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and I[nr][nc] != 7 and (nr, nc) not in chain:
                    next_pos = (nr, nc)
                    count += 1
            if count == 0:
                return chain
            if count > 1:
                return []  # Assume no branches
            chain.append(next_pos)
            cr, cc = next_pos
    
    for r in range(h):
        for c in range(w):
            if I[r][c] == 5:
                chain = get_chain(r, c)
                if len(chain) <= 1:
                    continue
                black_idx = -1
                for i, (rr, cc) in enumerate(chain):
                    if I[rr][cc] == 0:
                        black_idx = i
                        break
                if black_idx < 0 or black_idx + 1 >= len(chain):
                    continue
                prev_r, prev_c = chain[black_idx - 1]
                black_r, black_c = chain[black_idx]
                dr = black_r - prev_r
                dc = black_c - prev_c
                perp1_dr, perp1_dc = dc, -dr
                perp2_dr, perp2_dc = -dc, dr
                perps = [(perp1_dr, perp1_dc), (perp2_dr, perp2_dc)]
                max_space = -1
                best_perp = None
                for pdr, pdc in perps:
                    if pdr == 0 and pdc == 0:
                        continue
                    if pdr == 0:
                        if pdc > 0:
                            space = w - black_c - 1
                        else:
                            space = black_c
                    else:
                        if pdr > 0:
                            space = h - black_r - 1
                        else:
                            space = black_r
                    if space > max_space:
                        max_space = space
                        best_perp = (pdr, pdc)
                if best_perp is None or max_space < len(chain) - black_idx - 1:
                    continue
                tail_positions = chain[black_idx + 1:]
                cr, cc = black_r + best_perp[0], black_c + best_perp[1]
                for i, (tr, tc) in enumerate(tail_positions):
                    output[cr][cc] = I[tr][tc]
                    cr += best_perp[0]
                    cc += best_perp[1]
                for tr, tc in tail_positions:
                    output[tr][tc] = 7
    
    return output

import numpy as np

def solve_65(I):
    g = np.array(I)
    purple_pos = np.argwhere(g == 8)
    orange_pos = np.argwhere(g == 7)
    pr, pc = purple_pos[0]
    or_, oc = orange_pos[0]
    rows, cols = g.shape
    out = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            on_purple = (r == pr) or (c == pc)
            on_orange = (r == or_) or (c == oc)
            if on_purple and on_orange:
                out[r, c] = 2
            elif on_purple:
                out[r, c] = 8
            elif on_orange:
                out[r, c] = 7
    return out.tolist()

import numpy as np
from typing import List

def solve_66(I):
    if not I or not I[0]:
        return [[0]]
    
    grid_np = np.array(I)
    rows, cols = grid_np.shape
    
    def get_components(target_color: int) -> List[List[tuple[int, int]]]:
        visited = np.zeros((rows, cols), dtype=bool)
        components = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for r in range(rows):
            for c in range(cols):
                if grid_np[r, c] == target_color and not visited[r, c]:
                    component = []
                    stack = [(r, c)]
                    visited[r, c] = True
                    while stack:
                        cr, cc = stack.pop()
                        component.append((cr, cc))
                        for dr, dc in directions:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid_np[nr, nc] == target_color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                    components.append(component)
        return components
    
    red_components = get_components(2)
    if len(red_components) != 2:
        return [[0]]  # Assuming always 2, but handle otherwise
    
    red1 = set(red_components[0])
    red2 = set(red_components[1])
    
    purple_components = get_components(8)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def touches(comp: List[tuple[int, int]], red_set: set[tuple[int, int]]) -> bool:
        for r, c in comp:
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in red_set:
                    return True
        return False
    
    for purp_comp in purple_components:
        if touches(purp_comp, red1) and touches(purp_comp, red2):
            return [[8]]
    
    return [[0]]

def solve_67(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    candidates = []

    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and not visited[r][c]:
                color = I[r][c]
                stack = [(r, c)]
                visited[r][c] = True
                component_size = 1
                min_r, max_r = r, r
                min_c, max_c = c, c

                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                            component_size += 1
                            min_r = min(min_r, nr)
                            max_r = max(max_r, nr)
                            min_c = min(min_c, nc)
                            max_c = max(max_c, nc)

                h = max_r - min_r + 1
                w = max_c - min_c + 1
                bb_area = h * w
                if component_size == bb_area:
                    candidates.append((bb_area, h, w, color))

    if not candidates:
        return []

    min_area = min(c[0] for c in candidates)
    # Assume unique min area as per examples
    for area, h, w, color in candidates:
        if area == min_area:
            return [[color] * w for _ in range(h)]

    return []

import numpy as np

def solve_68(I):
    if not I or not I[0]:
        return I
    I = np.array(I)
    rows, cols = I.shape
    output = I.copy()

    for r1 in range(rows):
        for r2 in range(r1, rows):
            if r2 - r1 + 1 < 5:
                continue
            for c1 in range(cols):
                for c2 in range(c1, cols):
                    if c2 - c1 + 1 < 5:
                        continue
                    # Check if uniform
                    color = I[r1, c1]
                    if color == 0:
                        continue
                    uniform = np.all(I[r1:r2+1, c1:c2+1] == color)
                    if not uniform:
                        continue
                    # Check maximality
                    can_up = r1 > 0 and np.all(I[r1-1, c1:c2+1] == color)
                    can_down = r2 < rows - 1 and np.all(I[r2+1, c1:c2+1] == color)
                    can_left = c1 > 0 and np.all(I[r1:r2+1, c1-1] == color)
                    can_right = c2 < cols - 1 and np.all(I[r1:r2+1, c2+1] == color)
                    if not (can_up or can_down or can_left or can_right):
                        # Maximal and large enough, set to 4
                        output[r1:r2+1, c1:c2+1] = 4
    return output.tolist()

def solve_69(I):
    I = [row[:] for row in I]
    rows = len(I)
    if rows == 0:
        return I
    cols = len(I[0])

    # Horizontal fills
    for r in range(rows):
        purple_cols = [c for c in range(cols) if I[r][c] == 8]
        if len(purple_cols) >= 2:
            min_c = min(purple_cols)
            max_c = max(purple_cols)
            for c in range(min_c + 1, max_c):
                I[r][c] = 3

    # Vertical fills
    for c in range(cols):
        purple_rows = [r for r in range(rows) if I[r][c] == 8]
        if len(purple_rows) >= 2:
            min_r = min(purple_rows)
            max_r = max(purple_rows)
            for rr in range(min_r + 1, max_r):
                I[rr][c] = 3

    return I

def solve_70(I):
    if not I or not I[0]:
        return []

    rows = len(I)
    cols = len(I[0])

    # Extract sections
    left = [row[0:5] for row in I]
    center = [row[6:11] for row in I]
    right = [row[12:17] for row in I]

    # Function to get top and bottom for a section
    def get_top_bottom(section):
        min_r = rows
        max_r = -1
        for r in range(rows):
            if any(cell != 7 for cell in section[r]):
                min_r = min(min_r, r)
                max_r = max(max_r, r)
        if max_r == -1:
            return None
        return min_r, max_r

    # Initialize output
    output = [[7 for _ in range(5)] for _ in range(5)]

    # List of sections in stacking order: left, center, right
    sections = [left, center, right]

    current_bottom = 4
    for sec in sections:
        res = get_top_bottom(sec)
        if res is None:
            continue
        top, bottom = res
        shift = current_bottom - bottom
        # Place
        for local_r in range(rows):
            out_r = local_r + shift
            if 0 <= out_r < rows:
                for c in range(5):
                    val = sec[local_r][c]
                    if val != 7:
                        output[out_r][c] = val
        # Update current_top
        current_top = top + shift
        # Next bottom
        current_bottom = current_top - 1

    return output

from collections import Counter
import numpy as np

def solve_71(I):
    if not I or not I[0]:
        return I
    I = np.array(I)
    rows, cols = I.shape
    positions = [(r, c) for r in range(rows) for c in range(cols) if I[r, c] != 0]
    n = len(positions)
    if n == 0:
        return I.tolist()
    
    # Compute sums for centroid
    sum_r = sum(r for r, c in positions)
    sum_c = sum(c for r, c in positions)
    
    # Find intruder color (appears once)
    color_count = Counter(I[r, c] for r, c in positions)
    intruder_color = next(col for col, cnt in color_count.items() if cnt == 1)
    
    # Find intruder position
    ir, ic = next((r, c) for r, c in positions if I[r, c] == intruder_color)
    
    # Deltas (using integer comparison)
    delta_r_num = sum_r - ir * n
    delta_c_num = sum_c - ic * n
    abs_dr = abs(delta_r_num)
    abs_dc = abs(delta_c_num)
    
    if abs_dr > abs_dc:
        is_vertical = True
        step_r = -1 if delta_r_num < 0 else 1 if delta_r_num > 0 else 0
        step_c = 0
    else:
        is_vertical = False
        step_c = -1 if delta_c_num < 0 else 1 if delta_c_num > 0 else 0
        step_r = 0
    
    if step_r == 0 and step_c == 0:
        return I.tolist()  # No change if no direction
    
    # Find min and max in the line
    if is_vertical:
        line_pos = [r for r in range(rows) if I[r, ic] != 0]
        min_p = min(line_pos)
        max_p = max(line_pos)
        far_end = min_p if step_r < 0 else max_p
        current = far_end + step_r
        while 0 <= current < rows:
            if I[current, ic] != 0:
                break  # Stop if non-empty, but in examples empty
            I[current, ic] = intruder_color
            current += step_r
    else:
        line_pos = [c for c in range(cols) if I[ir, c] != 0]
        min_p = min(line_pos)
        max_p = max(line_pos)
        far_end = min_p if step_c < 0 else max_p
        current = far_end + step_c
        while 0 <= current < cols:
            if I[ir, current] != 0:
                break
            I[ir, current] = intruder_color
            current += step_c
    
    return I.tolist()

def solve_72(I):
    result = []
    for row in I:
        if all(cell == row[0] for cell in row):
            result.append([5] * len(row))
        else:
            result.append([0] * len(row))
    return result

from collections import Counter

def solve_73(I):
    if not I or not I[0]:
        return []
    
    rows = len(I)
    cols = len(I[0])
    background = I[0][0]
    visited = [[False] * cols for _ in range(rows)]
    components = []
    
    def dfs(x, y, comp):
        stack = [(x, y)]
        visited[x][y] = True
        comp.append((x, y))
        while stack:
            cx, cy = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and I[nx][ny] != background:
                    visited[nx][ny] = True
                    stack.append((nx, ny))
                    comp.append((nx, ny))
    
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != background and not visited[i][j]:
                comp = []
                dfs(i, j, comp)
                components.append(comp)
    
    squares = []
    for comp in components:
        if not comp:
            continue
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        if h == w and len(comp) == h * w:
            squares.append((min_r, min_c, h))
    
    if not squares:
        return []
    
    n = squares[0][2]
    # Get frame_color from most common in first square
    first_min_r, first_min_c, _ = squares[0]
    colors = [I[first_min_r + i][first_min_c + j] for i in range(n) for j in range(n)]
    frame_color = Counter(colors).most_common(1)[0][0]
    
    output = [[frame_color for _ in range(n)] for _ in range(n)]
    
    for min_r, min_c, _ in squares:
        for i in range(n):
            for j in range(n):
                color = I[min_r + i][min_c + j]
                if color != frame_color:
                    output[i][j] = color
    
    return output

def solve_74(I):
    if not I or not I[0]:
        return I
    output = [row[:] for row in I]
    height = len(output)
    width = len(output[0])
    for c in range(width):
        for r in range(height - 2, -1, -1):
            if output[r][c] != 0 and output[r + 1][c] == 0:
                output[r + 1][c] = output[r][c]
                output[r][c] = 0
    return output

def solve_75(I):
    n = len(I)
    bars = []
    for c in range(n):
        if I[n-1][c] == 7:
            continue
        color = I[n-1][c]
        h = 1
        r = n - 2
        while r >= 0 and I[r][c] == color:
            h += 1
            r -= 1
        bars.append((c, color, h))
    colors = [b[1] for b in bars]
    heights = [b[2] for b in bars]
    if not bars:
        return [row[:] for row in I]
    new_colors = colors[-1:] + colors[:-1]
    new_heights = heights[1:] + heights[:1]
    new_grid = [row[:] for row in I]
    # Clear old bars
    for c, _, h in bars:
        for rr in range(n - h, n):
            new_grid[rr][c] = 7
    # Place new bars
    for i in range(len(bars)):
        c = bars[i][0]
        color = new_colors[i]
        h = new_heights[i]
        for rr in range(n - h, n):
            new_grid[rr][c] = color
    return new_grid

from collections import Counter

def solve_76(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    output = [row[:] for row in I]
    
    # Find purple length N (consecutive 8's in row 0 from left)
    N = 0
    for c in range(width):
        if I[0][c] == 8:
            N += 1
        else:
            break
    
    # Find gray_row: the row where all cells are 5
    gray_row = None
    for r in range(height):
        if all(cell == 5 for cell in I[r]):
            gray_row = r
            break
    if gray_row is None:
        return output  # No gray bar, no change
    
    # Assume bottom_row is gray_row + 2
    bottom_row = gray_row + 2
    if bottom_row >= height:
        return output
    
    # Count frequencies in bottom_row
    freq = Counter()
    for c in range(width):
        color = I[bottom_row][c]
        if color != 0:
            freq[color] += 1
    
    # For each color C with freq[C] == N
    for C in freq:
        if freq[C] == N:
            # Find columns where bottom has C
            cols = [c for c in range(width) if I[bottom_row][c] == C]
            # For each such col, add pillar of height N upwards from gray_row - 1
            for col in cols:
                for h in range(N):
                    grow_r = gray_row - 1 - h
                    if grow_r >= 0:
                        output[grow_r][col] = C
    
    return output

def solve_77(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    
    # Find horizontal bar rows
    h_rows = []
    for r in range(rows):
        if all(I[r][c] == 8 for c in range(cols)):
            h_rows.append(r)
    
    # Find vertical line columns
    vert_cols = []
    for c in range(cols):
        if all(I[r][c] == 8 for r in range(rows)):
            vert_cols.append(c)
    
    if not vert_cols or not h_rows:
        return output  # No changes if no verticals or horizontals
    
    left_vert = min(vert_cols)
    right_vert = max(vert_cols)
    
    # Top section
    top_end = h_rows[0] - 1
    if top_end >= 0:
        for r in range(0, top_end + 1):
            for c in range(left_vert + 1, right_vert):
                output[r][c] = 2
    
    # Middle sections
    for i in range(len(h_rows) - 1):
        start = h_rows[i] + 1
        end = h_rows[i + 1] - 1
        if start <= end:
            for r in range(start, end + 1):
                # Left
                for c in range(0, left_vert):
                    output[r][c] = 4
                # Center
                for c in range(left_vert + 1, right_vert):
                    output[r][c] = 6
                # Right
                for c in range(right_vert + 1, cols):
                    output[r][c] = 3
    
    # Bottom section
    bot_start = h_rows[-1] + 1
    if bot_start < rows:
        for r in range(bot_start, rows):
            for c in range(left_vert + 1, right_vert):
                output[r][c] = 1
    
    return output

from collections import defaultdict

def solve_78(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    count = defaultdict(int)
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != 0 and not visited[i][j]:
                color = I[i][j]
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    r, c = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                count[color] += 1
    items = [(cnt, colr) for colr, cnt in count.items() if cnt > 0]
    if not items:
        return []
    items.sort(key=lambda x: (-x[0], x[1]))
    num_rows = len(items)
    max_col = max(cnt for cnt, _ in items)
    output = [[0] * max_col for _ in range(num_rows)]
    for i, (cnt, colr) in enumerate(items):
        start = max_col - cnt
        for j in range(cnt):
            output[i][start + j] = colr
    return output

def solve_79(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Find bar rows
    bar_rows = [i for i in range(rows) if any(c > 0 and c != 5 for c in I[i])]
    
    if not bar_rows:
        return I  # No transformation if no bar
    
    min_bar = min(bar_rows)
    max_bar = max(bar_rows)
    
    # Uniform row is the last one
    uniform_row = max_bar
    non_zeros = {c for c in I[uniform_row] if c != 0}
    if len(non_zeros) != 1:
        raise ValueError("Uniform row has multiple non-zero colors")
    C = list(non_zeros)[0]
    
    # Mixed row
    mixed_row = min_bar
    js = [j for j in range(cols) if I[mixed_row][j] != 0]
    if not js:
        raise ValueError("No colors in mixed row")
    min_j = min(js)
    max_j = max(js)
    if len(js) != max_j - min_j + 1:
        raise ValueError("Colors in mixed row are not contiguous")
    S = [I[mixed_row][j] for j in range(min_j, max_j + 1)]
    
    # Find grey bounding box
    grey_pos = [(i, j) for i in range(rows) for j in range(cols) if I[i][j] == 5]
    if not grey_pos:
        return I  # No transformation if no grey
    
    min_r = min(i for i, j in grey_pos)
    max_r = max(i for i, j in grey_pos)
    min_c = min(j for i, j in grey_pos)
    max_c = max(j for i, j in grey_pos)
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    if len(S) != width:
        raise ValueError("Length of S does not match width")
    
    # Create output
    output = [[0] * width for _ in range(height)]
    for sub_i in range(height):
        for sub_j in range(width):
            actual_i = min_r + sub_i
            actual_j = min_c + sub_j
            if I[actual_i][actual_j] == 5:
                output[sub_i][sub_j] = S[sub_j]
            else:
                output[sub_i][sub_j] = C
    
    return output

import copy
from collections import Counter

def solve_80(I):
    out = copy.deepcopy(I)
    rows = len(I)
    cols = len(I[0])
    
    # Find the gray row (all 5s)
    gray_row = -1
    for r in range(rows):
        if all(cell == 5 for cell in I[r]):
            gray_row = r
            break
    
    if gray_row == -1:
        return out  # No gray row, no change (though assumes there is one)
    
    # Collect all colors in colored rows (0 to gray_row-1)
    color_counts = Counter()
    for r in range(gray_row):
        for c in range(cols):
            colr = I[r][c]
            if colr != 0:  # Though in colored rows, all non-zero
                color_counts[colr] += 1
    
    if not color_counts:
        return out
    
    # Find the color with max frequency
    max_color = max(color_counts, key=color_counts.get)
    
    # Bottom row
    bottom_row = rows - 1
    
    # Center column
    center_col = (cols - 1) // 2
    
    # Set the cell
    out[bottom_row][center_col] = max_color
    
    return out

def solve_81(I):
    if not I or not I[0]:
        return I
    
    # Extract the four 4x4 sections
    purple = [row[0:4] for row in I]
    gray = [row[5:9] for row in I]
    brown = [row[10:14] for row in I]
    yellow = [row[15:19] for row in I]
    
    # Create output 4x4 I
    output = [[0 for _ in range(4)] for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            if brown[i][j] != 0:
                output[i][j] = 9
            elif yellow[i][j] != 0:
                output[i][j] = 4
            elif purple[i][j] != 0:
                output[i][j] = 8
            elif gray[i][j] != 0:
                output[i][j] = 5
            # else remains 0
    
    return output

def solve_82(I):
    positions = [(i, j) for i in range(len(I)) for j in range(len(I[0])) if I[i][j] != 0]
    if not positions:
        return [[0] * 6 for _ in range(3)]  # Default, though not needed for puzzle
    min_r = min(i for i, j in positions)
    max_r = max(i for i, j in positions)
    min_c = min(j for i, j in positions)
    max_c = max(j for i, j in positions)
    pat = [[I[i][j] for j in range(min_c, max_c + 1)] for i in range(min_r, max_r + 1)]
    output = [row + row for row in pat]
    return output

import numpy as np

def solve_83(I):
    I = np.array(grid_lst)
    colors = np.unique(I)
    color_list = [c for c in colors if c != 0 and c != 5]
    assert len(color_list) == 1
    color = color_list[0]
    
    row_ranges = [(0, 2), (4, 6), (8, 10)]
    col_ranges = [(0, 2), (4, 6), (8, 10)]
    
    counts = np.zeros((3, 3), dtype=int)
    for i, (r_start, r_end) in enumerate(row_ranges):
        for j, (c_start, c_end) in enumerate(col_ranges):
            subgrid = I[r_start:r_end + 1, c_start:c_end + 1]
            counts[i, j] = np.sum(subgrid == color)
    
    max_count = np.max(counts)
    to_fill = np.argwhere(counts == max_count)
    
    output = I.copy()
    
    # Remove color cells not in to_fill regions
    for r in range(output.shape[0]):
        for c in range(output.shape[1]):
            if output[r, c] == color:
                in_fill = False
                for fi, fj in to_fill:
                    rs, re = row_ranges[fi]
                    cs, ce = col_ranges[fj]
                    if rs <= r <= re and cs <= c <= ce:
                        in_fill = True
                        break
                if not in_fill:
                    output[r, c] = 0
    
    # Fill the to_fill regions
    for fi, fj in to_fill:
        rs, re = row_ranges[fi]
        cs, ce = col_ranges[fj]
        for r in range(rs, re + 1):
            for c in range(cs, ce + 1):
                if output[r, c] == 0:
                    output[r, c] = color
    
    return output.tolist()

def solve_84(I):
    if not I:
        return I
    rows = len(I)
    cols = len(I[0])
    
    # Find heads in row 0
    head_col = {}
    for c in range(cols):
        color = I[0][c]
        if color != 0:
            head_col[color] = c
    
    # Find tails
    tail = {}
    for r in range(1, rows):
        for c in range(cols):
            color = I[r][c]
            if color != 0:
                tail[color] = (r, c)
    
    # Create output I
    output = [row[:] for row in I]
    
    # Fill for each color
    for color, h_col in head_col.items():
        if color in tail:
            t_r, t_c = tail[color]
            # Vertical fill
            for rr in range(t_r + 1):
                output[rr][h_col] = color
            # Horizontal fill
            min_c = min(h_col, t_c)
            max_c = max(h_col, t_c)
            for cc in range(min_c, max_c + 1):
                output[t_r][cc] = color
        else:
            # Vertical to bottom
            for rr in range(rows):
                output[rr][h_col] = color
    
    return output

def solve_85(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    output = [row[:] for row in I]
    
    mid = width // 2  # 5 for width 11
    left_len = mid  # 5
    right_start = mid + 1
    
    for r in range(height):
        if output[r][0] != 0 and output[r][width - 1] != 0 and all(output[r][j] == 0 for j in range(1, width - 1)):
            left_col = output[r][0]
            right_col = output[r][width - 1]
            for j in range(left_len):
                output[r][j] = left_col
            output[r][mid] = 5
            for j in range(right_start, width):
                output[r][j] = right_col
    
    return output

def solve_86(I):
    if not I or not I[0]:
        return I

    height = len(I)
    width = len(I[0])
    output = [row[:] for row in I]
    visited = [[False] * width for _ in range(height)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []

    def dfs(r, c, color):
        stack = [(r, c)]
        component = [(r, c)]
        visited[r][c] = True
        while stack:
            cr, cc = stack.pop()
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and not visited[nr][nc] and I[nr][nc] == color:
                    visited[nr][nc] = True
                    component.append((nr, nc))
                    stack.append((nr, nc))
        return component

    for r in range(height):
        for c in range(width):
            if I[r][c] != 0 and not visited[r][c]:
                comp = dfs(r, c, I[r][c])
                components.append((I[r][c], comp))

    from collections import defaultdict
    shape_to_comps = defaultdict(list)
    for color, pos_list in components:
        if not pos_list:
            continue
        min_r = min(rr for rr, cc in pos_list)
        min_c = min(cc for rr, cc in pos_list)
        sig = frozenset((rr - min_r, cc - min_c) for rr, cc in pos_list)
        shape_to_comps[sig].append((color, pos_list))

    for sig, complist in shape_to_comps.items():
        if len(complist) == 2:
            c1, pos1 = complist[0]
            c2, pos2 = complist[1]
            if c1 == 1 and c2 != 1:
                for rr, cc in pos1:
                    output[rr][cc] = c2
            elif c2 == 1 and c1 != 1:
                for rr, cc in pos2:
                    output[rr][cc] = c1

    return output

def solve_87(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    
    # Find mirror row or column
    mirror_row = -1
    for i in range(rows):
        if all(I[i][j] == 1 for j in range(cols)):
            mirror_row = i
            break
    
    mirror_col = -1
    for j in range(cols):
        if all(I[i][j] == 1 for i in range(rows)):
            mirror_col = j
            break
    
    # Collect original shape positions and colors
    original_shape = []
    color_set = set()
    for r in range(rows):
        for c in range(cols):
            val = I[r][c]
            if val != 0 and val != 1:
                original_shape.append((r, c, val))
                color_set.add(val)
    
    # Assume exactly two colors
    if len(color_set) != 2:
        return I  # Or handle differently, but per puzzle it's 2
    c1, c2 = list(color_set)
    
    def swap(col):
        if col == c1:
            return c2
        if col == c2:
            return c1
        return col
    
    # Create output I
    output = [row[:] for row in I]
    
    # Add reflections
    if mirror_col != -1:
        for r, c, val in original_shape:
            mir_c = 2 * mirror_col - c
            if 0 <= mir_c < cols:
                output[r][mir_c] = val
    elif mirror_row != -1:
        for r, c, val in original_shape:
            mir_r = 2 * mirror_row - r
            if 0 <= mir_r < rows:
                output[mir_r][c] = val
    
    # Swap colors in original positions
    for r, c, val in original_shape:
        output[r][c] = swap(val)
    
    return output

import numpy as np
from collections import Counter
from typing import List

def solve_88(I):
    if not I or not I[0]:
        return I

    height = len(I)
    width = len(I[0])

    # Find background color: most common
    flat = [cell for row in I for cell in row]
    counter = Counter(flat)
    background = counter.most_common(1)[0][0]

    # Find all connected components of non-background
    visited = [[False] * width for _ in range(height)]
    components = []
    for i in range(height):
        for j in range(width):
            if I[i][j] != background and not visited[i][j]:
                component = []
                stack = [(i, j)]
                color = I[i][j]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < height and 0 <= ny < width and not visited[nx][ny] and I[nx][ny] == color:
                            visited[nx][ny] = True
                            stack.append((nx, ny))
                components.append(component)

    if not components:
        return I

    # Largest component
    central_comp = max(components, key=len)
    central_color = I[central_comp[0][0]][central_comp[0][1]]

    # Bounding box of central
    rs = [r for r, c in central_comp]
    cs = [c for r, c in central_comp]
    min_r = min(rs)
    max_r = max(rs)
    min_c = min(cs)
    max_c = max(cs)

    # Make a copy of I to modify
    output = [row[:] for row in I]

    # Treat each non-background, non-central_color cell as seed
    for i in range(height):
        for j in range(width):
            color = I[i][j]
            if color != background and color != central_color:
                S = color
                r, c = i, j
                # Vertical growth
                if min_c <= c <= max_c:
                    if r < min_r:
                        for rr in range(r, min_r):
                            output[rr][c] = S
                    elif r > max_r:
                        for rr in range(max_r + 1, r + 1):
                            output[rr][c] = S
                # Horizontal growth
                if min_r <= r <= max_r:
                    if c < min_c:
                        for cc in range(c, min_c):
                            output[r][cc] = S
                    elif c > max_c:
                        for cc in range(max_c + 1, c + 1):
                            output[r][cc] = S

    return output

import numpy as np

def solve_89(I):
    I = np.array(grid_lst)
    n = I.shape[0]
    center = n // 2
    background = I[0, 0]

    # Find outliers
    outliers = []
    for r in range(n):
        for c in range(n):
            if r != center and c != center and I[r, c] != background:
                outliers.append((r, c))

    if not outliers:
        # No outliers, but examples always have at least one
        raise ValueError("No outliers found")

    # Determine quadrant based on first outlier
    r, c = outliers[0]
    if r < center:
        row_start = 0
        row_end = center
        if c < center:
            col_start = 0
            col_end = center
        else:
            col_start = center + 1
            col_end = n
    else:
        row_start = center + 1
        row_end = n
        if c < center:
            col_start = 0
            col_end = center
        else:
            col_start = center + 1
            col_end = n

    # Extract subgrid
    subgrid = I[row_start:row_end, col_start:col_end]
    return subgrid.tolist()

import numpy as np

def solve_90(I):
    h = len(I)
    return [row[:h] for row in I]

from collections import deque, defaultdict

def solve_91(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 8 and (r, c) not in visited:
                component = set()
                q = deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    component.add((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] != 8 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                components.append(component)
    # Assume exactly two components
    small_local = None
    large_local = None
    dot_s_r, dot_s_c = None, None
    dot_l_r, dot_l_c = None, None
    small_h, small_w = None, None
    large_h, large_w = None, None
    for comp in components:
        min_r = min(rr for rr, _ in comp)
        max_r = max(rr for rr, _ in comp)
        min_c = min(cc for _, cc in comp)
        max_c = max(cc for _, cc in comp)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        local = [[I[min_r + i][min_c + j] for j in range(w)] for i in range(h)]
        num_col = sum(1 for roww in local for cell in roww if cell != 0)
        if num_col == 1:
            small_local = local
            small_h = h
            small_w = w
            for ii in range(h):
                for jj in range(w):
                    if local[ii][jj] != 0:
                        dot_s_r = ii
                        dot_s_c = jj
                        break
                else:
                    continue
                break
        else:
            large_local = local
            large_h = h
            large_w = w
            freq = defaultdict(int)
            for ii in range(h):
                for jj in range(w):
                    if local[ii][jj] != 0:
                        freq[local[ii][jj]] += 1
            once_colors = [col for col, cnt in freq.items() if cnt == 1]
            dot_color = once_colors[0]  # Assume exactly one
            for ii in range(h):
                for jj in range(w):
                    if local[ii][jj] == dot_color:
                        dot_l_r = ii
                        dot_l_c = jj
                        break
                else:
                    continue
                break
    offset_r = dot_s_r - dot_l_r
    offset_c = dot_s_c - dot_l_c
    output = [[0] * small_w for _ in range(small_h)]
    for i in range(small_h):
        for j in range(small_w):
            i_l = i - offset_r
            j_l = j - offset_c
            if 0 <= i_l < large_h and 0 <= j_l < large_w:
                output[i][j] = large_local[i_l][j_l]
    return output

def solve_92(I):
    if not I or not I[0]:
        return I
    h = len(I)
    w = len(I[0])
    out_h = 3 * h
    out_w = 3 * w
    out = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for r in range(out_h):
        for c in range(out_w):
            out[r][c] = I[r % h][c % w]
    for r in range(out_h):
        for c in range(out_w):
            src_color = I[(r + 1) % h][(c + 1) % w]
            if src_color != 0 and out[r][c] == 0:
                out[r][c] = 2
    return out

def solve_93(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    rows_with_zero = set()
    cols_with_zero = set()
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 0:
                rows_with_zero.add(r)
                cols_with_zero.add(c)
    output = [row[:] for row in I]
    for r in range(rows):
        for c in range(cols):
            if r in rows_with_zero or c in cols_with_zero:
                output[r][c] = I[r][c] if I[r][c] == 2 else 0
    return output

from collections import defaultdict

def solve_94(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Collect positions for each color
    color_positions = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            color = I[r][c]
            if color != 0:
                color_positions[color].append((r, c))
    
    # Find the color with exactly one position
    unique_color = None
    unique_pos = None
    for color, positions in color_positions.items():
        if len(positions) == 1:
            unique_color = color
            unique_pos = positions[0]
            break  # Assume only one such color
    
    if unique_color is None:
        return I  # No change if none found, but per examples, there is one
    
    r, c = unique_pos
    
    # Create output I all 0's
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Set the 3x3 centered at (r, c)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if dr == 0 and dc == 0:
                    output[nr][nc] = unique_color
                else:
                    output[nr][nc] = 2
    
    return output

def solve_95(I):
    h = len(I) // 2
    w = len(I[0]) if I else 0
    output = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            top = I[r][c] != 0
            bottom = I[r + h][c] != 0
            if top + bottom == 1:
                output[r][c] = 6
    return output

from collections import deque
import copy

def solve_96(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    patterns = []
    templates = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def bfs(start_r, start_c, is_purple):
        q = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        component = [(start_r, start_c)]
        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                    if (is_purple and I[nr][nc] == 8) or (not is_purple and I[nr][nc] != 0 and I[nr][nc] != 8):
                        visited[nr][nc] = True
                        q.append((nr, nc))
                        component.append((nr, nc))
        return component

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c]:
                if I[r][c] == 8:
                    comp = bfs(r, c, True)
                    templates.append(comp)
                elif I[r][c] != 0:
                    comp = bfs(r, c, False)
                    patterns.append(comp)

    pattern_rels = {}
    for i, pat in enumerate(patterns):
        if not pat:
            continue
        min_r = min(rr for rr, cc in pat)
        min_c = min(cc for rr, cc in pat)
        rel = frozenset((rr - min_r, cc - min_c) for rr, cc in pat)
        pattern_rels[i] = (rel, min_r, min_c)

    output = copy.deepcopy(I)
    for temp in templates:
        if not temp:
            continue
        min_r = min(rr for rr, cc in temp)
        min_c = min(cc for rr, cc in temp)
        rel = frozenset((rr - min_r, cc - min_c) for rr, cc in temp)
        for pat_i, (pat_rel, pat_min_r, pat_min_c) in pattern_rels.items():
            if rel == pat_rel:
                for dr, dc in rel:
                    color = I[pat_min_r + dr][pat_min_c + dc]
                    output[min_r + dr][min_c + dc] = color
                break

    for pat in patterns:
        for r, c in pat:
            output[r][c] = 0

    return output

def solve_97(I):
    if not I:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [[5 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        c = 0
        while c < cols:
            if I[r][c] != 0:
                color = I[r][c]
                start = c
                while c < cols and I[r][c] == color:
                    c += 1
                end = c - 1
                new_start = start - 1 if start > 0 else 0
                new_end = end - 1
                if new_end >= new_start:
                    for cc in range(new_start, new_end + 1):
                        output[r][cc] = color
            else:
                c += 1
    return output

import numpy as np
from collections import defaultdict

def solve_98(I):
    I = np.array(I)
    rows, cols = I.shape

    # Find vertical_col: the column with the most 1s
    col_counts = defaultdict(int)
    for c in range(cols):
        for r in range(rows):
            if I[r, c] == 1:
                col_counts[c] += 1
    vertical_col = max(col_counts, key=col_counts.get)

    # Find bars: list of (row, color)
    bars = []
    for r in range(rows):
        if I[r, vertical_col] == 1:
            # Find candidate color (use first non-vertical col)
            cand_col = 0 if vertical_col != 0 else 1
            c = I[r, cand_col]
            is_bar = True
            for cc in range(cols):
                if cc == vertical_col:
                    if I[r, cc] != 1:
                        is_bar = False
                        break
                else:
                    if I[r, cc] != c:
                        is_bar = False
                        break
            if is_bar:
                bars.append((r, c))
    bars.sort()  # sort by row

    # Find separator rows
    separator_rows = set()
    for i in range(len(bars) - 1):
        r1, c1 = bars[i]
        r2, c2 = bars[i + 1]
        if c1 != c2 and (r1 + r2) % 2 == 0:
            mid = (r1 + r2) // 2
            if r1 < mid < r2:
                separator_rows.add(mid)

    # Original rows
    original_rows = set(r for r, c in bars)

    # Create output
    output = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        if r in original_rows:
            output[r] = 1
            output[r, vertical_col] = 8
        elif r in separator_rows:
            output[r] = 1
        else:
            # Find nearest bar
            min_dist = float('inf')
            nearest_color = 0
            for bar_r, bar_c in bars:
                dist = abs(r - bar_r)
                if dist < min_dist:
                    min_dist = dist
                    nearest_color = bar_c
                elif dist == min_dist:
                    # If tie, colors should be the same, but set to this if different (shouldn't happen)
                    nearest_color = bar_c
            output[r] = nearest_color
            output[r, vertical_col] = 1

    return output.tolist()

def solve_99(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [[1 if not (r % 2 == 1 and c % 2 == 1) else 0 for c in range(cols)] for r in range(rows)]
    return output

def solve_100(I):
    if not I or not I[0]:
        return I
    height = len(I)
    width = len(I[0])
    
    # Find non-zero colors
    colors = set()
    for r in range(height):
        for c in range(width):
            if I[r][c] != 0:
                colors.add(I[r][c])
    
    if len(colors) != 2:
        return I  # Assuming always two, but safety
    
    # Count occurrences
    count = {col: 0 for col in colors}
    for r in range(height):
        for c in range(width):
            if I[r][c] in count:
                count[I[r][c]] += 1
    
    # Determine main and mino
    main = max(count, key=count.get)
    mino = min(count, key=count.get)
    
    # Find min_c and max_c for main
    min_c = width
    max_c = -1
    for r in range(height):
        for c in range(width):
            if I[r][c] == main:
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    
    if max_c < min_c:
        return I  # No main cells
    
    center = (min_c + max_c) / 2.0
    
    # Create output
    output = [row[:] for row in I]
    
    # Transform minority cells
    for r in range(height):
        for c in range(width):
            if I[r][c] == mino:
                c_prime = int(2 * center - c)
                if 0 <= c_prime < width:
                    output[r][c] = I[r][c_prime]
                else:
                    output[r][c] = 0
    
    return output

import copy

def solve_101(I):
    if not I or not I[0]:
        return I
    I = copy.deepcopy(I)
    rows = len(I)
    cols = len(I[0])

    # Find key cells: color != 0 and != 5
    key_cells = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and I[r][c] != 5:
                key_cells.append((r, c))

    if not key_cells:
        return I

    # Find bounding box
    min_r = min(r for r, c in key_cells)
    max_r = max(r for r, c in key_cells)
    min_c = min(c for r, c in key_cells)
    max_c = max(c for r, c in key_cells)

    K = max_r - min_r + 1
    M = max_c - min_c + 1

    # Extract key
    key = [[0] * M for _ in range(K)]
    for r, c in key_cells:
        key[r - min_r][c - min_c] = I[r][c]

    # Find connected components of 5
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 5 and not visited[r][c]:
                comp = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 5 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append(comp)

    # Create block info: (min_r, min_c, comp)
    block_info = []
    for comp in components:
        if comp:
            b_min_r = min(rr for rr, cc in comp)
            b_min_c = min(cc for rr, cc in comp)
            block_info.append((b_min_r, b_min_c, comp))

    # Sort by min_r asc, then min_c asc
    block_info.sort(key=lambda x: (x[0], x[1]))

    # Assign colors
    for i in range(K):
        for j in range(M):
            idx = i * M + j
            if idx >= len(block_info):
                continue  # Safety, though should match
            _, _, positions = block_info[idx]
            color = key[i][j]
            for rr, cc in positions:
                I[rr][cc] = color

    return I

import numpy as np

def solve_102(I):
    I = np.array(grid_lst)
    height, width = I.shape
    middle = height // 2
    upper = I[0:middle, :]
    lower = I[middle + 1:, :]
    out = np.zeros((middle, width), dtype=int)
    diff = upper != lower
    out[diff] = 3
    return out.tolist()

def solve_103(I):
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 7 and not visited[r][c]:
                color = I[r][c]
                component = []
                min_r, max_r = r, r
                min_c, max_c = c, c
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append({
                    'color': color,
                    'min_r': min_r,
                    'max_r': max_r,
                    'min_c': min_c,
                    'max_c': max_c
                })

    from collections import defaultdict
    color_to_comps = defaultdict(list)
    for comp in components:
        color_to_comps[comp['color']].append(comp)

    for color, comps in color_to_comps.items():
        if len(comps) == 2:
            comps.sort(key=lambda x: x['min_r'])
            upper = comps[0]
            lower = comps[1]
            row_diff = lower['min_r'] - upper['max_r']
            if row_diff <= 0:
                continue

            options = [
                ('br_tl', upper['max_c'], lower['min_c']),
                ('bl_tr', upper['min_c'], lower['max_c'])
            ]
            for _, s_c, e_c in options:
                if abs(s_c - e_c) == row_diff:
                    start_r = upper['max_r']
                    start_c = s_c
                    end_c = e_c
                    dy = 1 if end_c > start_c else -1
                    for i in range(1, row_diff):
                        nr = start_r + i
                        nc = start_c + i * dy
                        output[nr][nc] = color
                    break

    return output

def solve_104(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    directions = {
        1: (0, 1),  # right
        2: (0, -1), # left
        7: (-1, 0), # up
        9: (1, 0)   # down
    }
    # Collect dots
    dots = []
    for r in range(rows):
        for c in range(cols):
            color = I[r][c]
            if color in directions:
                dots.append((r, c, color))
    # Initialize output with all 8
    output = [[8 for _ in range(cols)] for _ in range(rows)]
    # Move each dot
    for r, c, color in dots:
        dr, dc = directions[color]
        if dr != 0:
            dist = r if dr < 0 else rows - 1 - r
        else:
            dist = c if dc < 0 else cols - 1 - c
        steps = min(2, dist // 2)
        new_r = r + steps * dr
        new_c = c + steps * dc
        output[new_r][new_c] = color
    return output

def solve_105(I):
    rows = len(I)
    out = [[0] * 4 for _ in range(rows)]
    for r in range(rows):
        for c in range(4):
            left = 1 if I[r][c] == 8 else 0
            right = 1 if I[r][c + 5] == 5 else 0
            xor = left ^ right
            out[r][c] = 2 if xor == 1 else 0
    return out

def solve_106(I):
    out = [row[:] for row in I]
    for j in range(len(I[0])):
        if I[2][j] == 1:
            out[4][j] = 1
            out[2][j] = 0
    return out

def solve_107(I):
    template = [row[0:3] for row in I[0:3]]
    height = len(I)
    width = len(I[0])
    output = [row[:] for row in I]
    for i in range(height):
        for j in range(width):
            if I[i][j] == 1:
                for di in range(3):
                    for dj in range(3):
                        ni = i - 1 + di
                        nj = j - 1 + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            output[ni][nj] = template[di][dj]
    return output

import copy

def solve_108(I):
    new_grid = copy.deepcopy(I)
    rows = len(I)
    cols = len(I[0])
    purple_sum = 0
    red_sum = 0
    max_j = -1
    for j in range(cols):
        if I[rows - 1][j] == 7:
            continue
        color = I[rows - 1][j]
        if color not in (2, 8):
            continue
        h = 1
        for r in range(rows - 2, -1, -1):
            if I[r][j] == color:
                h += 1
            else:
                break
        if color == 8:
            purple_sum += h
        elif color == 2:
            red_sum += h
        max_j = max(max_j, j)
    grey_h = purple_sum - red_sum
    if grey_h <= 0 or max_j == -1:
        return new_grid
    new_j = max_j + 2
    if new_j >= cols:
        return new_grid
    for i in range(grey_h):
        r = rows - 1 - i
        new_grid[r][new_j] = 5
    return new_grid

def solve_109(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    # Clear all 2 to 0
    for r in range(rows):
        for c in range(cols):
            if output[r][c] == 2:
                output[r][c] = 0
    # For each column
    for c in range(cols):
        # Count h and find blues
        h = 0
        blues = []
        for r in range(rows):
            if I[r][c] == 2:
                h += 1
            if I[r][c] == 1:
                blues.append(r)
        if h > 0 and blues:
            k = max(blues)
            start = k + 1
            for i in range(h):
                nr = start + i
                if nr < rows:
                    output[nr][c] = 2
    return output

def solve_110(I):
    n = len(I)
    out = [[0] * (2 * n) for _ in range(2 * n)]

    # copy input to top-left
    for i in range(n):
        for j in range(n):
            out[i][j] = I[i][j]

    # determine colors
    A = I[0][0]
    B = I[n // 2][n // 2]
    C = I[0][1]
    cycle = [A, B, C]

    # fill top-right vertical bars
    for m in range(n):
        col = n + m
        color_idx = m % 3
        color = cycle[color_idx]
        for i in range(n):
            out[i][col] = color

    # fill bottom rows
    for k in range(n):
        row = n + k
        color_idx = k % 3
        main_color = cycle[color_idx]
        main_len = n + k
        # fill main
        for j in range(main_len):
            out[row][j] = main_color
        remaining = 2 * n - main_len
        if remaining == 0:
            continue
        # next color idx
        next_idx = (color_idx + 1) % 3
        curr_color = cycle[next_idx]
        if remaining == 1:
            out[row][main_len] = curr_color
            continue
        # fill 2 of curr
        for jj in range(2):
            out[row][main_len + jj] = curr_color
        pos = main_len + 2
        remaining -= 2
        while remaining > 0:
            next_idx = (next_idx + 1) % 3
            curr_color = cycle[next_idx]
            out[row][pos] = curr_color
            pos += 1
            remaining -= 1

    return out

import numpy as np
from collections import Counter

def solve_111(I):
    I = np.array(I)
    h, w = I.shape
    pattern_counts = Counter()
    
    for i in range(h - 2):
        for j in range(w - 2):
            sub = I[i:i+3, j:j+3]
            if np.any(sub != 0):
                key = tuple(tuple(row) for row in sub)
                pattern_counts[key] += 1
    
    if not pattern_counts:
        return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    max_count = max(pattern_counts.values())
    for pat, count in pattern_counts.items():
        if count == max_count:
            return [list(row) for row in pat]
    
    return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

from collections import deque

def solve_112(I):
    if not I or not I[0]:
        return I

    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    visited = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 8 and (i, j) not in visited:
                component = []
                queue = deque([(i, j)])
                visited.add((i, j))
                while queue:
                    r, c = queue.popleft()
                    component.append((r, c))
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 8 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))

                if len(component) == 3:
                    min_r = min(r for r, c in component)
                    max_r = max(r for r, c in component)
                    min_c = min(c for r, c in component)
                    max_c = max(c for r, c in component)
                    if max_r - min_r == 1 and max_c - min_c == 1:
                        comp_set = set(component)
                        for fr in range(min_r, min_r + 2):
                            for fc in range(min_c, min_c + 2):
                                if (fr, fc) not in comp_set:
                                    output[fr][fc] = 1
                                    break

    return output

def solve_113(I):
    output = [row[:] for row in I]
    height = len(I)
    if height == 0:
        return output
    width = len(I[0])
    for c in range(width):
        k = I[0][c]
        if k != 0:
            for r in range(height):
                if r % 2 == 0:
                    output[r][c] = k
                else:
                    if c - 1 >= 0:
                        output[r][c - 1] = k
                    if c + 1 < width:
                        output[r][c + 1] = k
    return output

import numpy as np
from collections import deque

def solve_114(I):
    if not I or not I[0]:
        return I
    
    I = np.array(I)
    rows, cols = I.shape
    output = I.copy()
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(rows):
        for j in range(cols):
            if I[i, j] != 0 and I[i, j] != 8 and not visited[i, j]:
                color = I[i, j]
                queue = deque([(i, j)])
                visited[i, j] = True
                while queue:
                    x, y = queue.popleft()
                    if I[x, y] == 0:
                        output[x, y] = color
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and (I[nx, ny] == 0 or I[nx, ny] == color):
                            visited[nx, ny] = True
                            queue.append((nx, ny))
    
    return output.tolist()

def solve_115(I):
    top = []
    for row in I:
        left = row
        right = row[::-1]
        top.append(left + right)
    bottom = top[::-1]
    output = top + bottom
    return output

def solve_116(I):
    if not I or not I[0]:
        return []
    h = len(I)
    w = len(I[0])
    count = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if I[r][c] == 3 and I[r][c + 1] == 3 and I[r + 1][c] == 3 and I[r + 1][c + 1] == 3:
                count += 1
    out = [[0 for _ in range(3)] for _ in range(3)]
    diag = [(0, 0), (1, 1), (2, 2)]
    for i in range(min(count, 3)):
        rr, cc = diag[i]
        out[rr][cc] = 1
    return out

def solve_117(I):
    I = [row[:] for row in I]  # Copy the I
    N = len(I)
    for r in range(N - 1):
        I[r][N - 1 - r] = 2
    for c in range(1, N):
        I[N - 1][c] = 4
    return I

def solve_118(I):
    rows = len(I)
    if rows < 3:
        return I
    cols = len(I[0])
    new_grid = [row[:] for row in I]
    
    for i in range(1, rows - 1):
        if I[i - 1] == I[i] == I[i + 1]:
            row = new_grid[i]
            j = 0
            while j < cols:
                if row[j] == 0:
                    j += 1
                    continue
                c = row[j]
                start = j
                while j < cols and row[j] == c:
                    j += 1
                # Modify the segment
                for k in range(start, j):
                    if (k - start) % 2 == 1:
                        new_grid[i][k] = 0
    
    return new_grid

from collections import deque

def solve_119(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def bfs(start_r, start_c):
        component = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] != 0:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return component

    output = [row[:] for row in I]
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and not visited[r][c]:
                comp = bfs(r, c)
                color_count = {}
                for pr, pc in comp:
                    colr = I[pr][pc]
                    color_count[colr] = color_count.get(colr, 0) + 1
                if len(color_count) != 2:
                    continue
                items = list(color_count.items())
                if items[0][1] > items[1][1]:
                    O, I = items[0][0], items[1][0]
                else:
                    O, I = items[1][0], items[0][0]
                rs = [pr for pr, _ in comp]
                cs = [pc for _, pc in comp]
                min_r = min(rs)
                max_r = max(rs)
                min_c = min(cs)
                max_c = max(cs)
                inner_pos = [(pr, pc) for pr, pc in comp if I[pr][pc] == I]
                if not inner_pos:
                    continue
                rs_i = [pr for pr, _ in inner_pos]
                cs_i = [pc for _, pc in inner_pos]
                min_r_i = min(rs_i)
                max_r_i = max(rs_i)
                min_c_i = min(cs_i)
                max_c_i = max(cs_i)
                h = max_r_i - min_r_i + 1
                w = max_c_i - min_c_i + 1
                for pr in range(min_r, max_r + 1):
                    for pc in range(min_c, max_c + 1):
                        if I[pr][pc] == O:
                            output[pr][pc] = I
                        elif I[pr][pc] == I:
                            output[pr][pc] = O
                for k in range(1, h + 1):
                    nr = min_r - k
                    if 0 <= nr < rows:
                        for pc in range(min_c, max_c + 1):
                            output[nr][pc] = O
                for k in range(1, h + 1):
                    nr = max_r + k
                    if 0 <= nr < rows:
                        for pc in range(min_c, max_c + 1):
                            output[nr][pc] = O
                for k in range(1, w + 1):
                    nc = min_c - k
                    if 0 <= nc < cols:
                        for pr in range(min_r, max_r + 1):
                            output[pr][nc] = O
                for k in range(1, w + 1):
                    nc = max_c + k
                    if 0 <= nc < cols:
                        for pr in range(min_r, max_r + 1):
                            output[pr][nc] = O
    return output

def solve_120(I):
    return [row[::-1] for row in I[::-1]]

def solve_121(I):
    if not I or not I[0]:
        return I
    
    h = len(I)
    w = len(I[0])
    out_w = w + h - 1
    output = [[0] * out_w for _ in range(h)]
    
    for r in range(h):
        left_zeros = h - 1 - r
        for c in range(w):
            output[r][left_zeros + c] = I[r][c]
    
    return output

import math

def solve_122(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    for c in range(width):
        seq = []
        for r in range(height):
            if I[r][c] != 0:
                seq.append(I[r][c])
        
        n = len(seq)
        if n == 0:
            continue
        
        h = int(math.sqrt(n))
        if h * h != n:
            raise ValueError("Number of cells is not a perfect square")
        
        idx = 0
        for k in range(1, h + 1):
            py_r = height - h + (k - 1)
            w = 2 * k - 1
            left = c - (w // 2)
            for j in range(w):
                col_pos = left + j
                if 0 <= col_pos < width:
                    output[py_r][col_pos] = seq[idx]
                idx += 1
    
    return output

from collections import defaultdict

def solve_123(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    pos = defaultdict(list)
    colors = set()
    for r in range(rows):
        for c in range(cols):
            val = I[r][c]
            if val != 0:
                pos[val].append((r, c))
                colors.add(val)
    if len(colors) != 2:
        return []  # Assume exactly two non-zero colors
    colors = list(colors)
    chosen_M = None
    chosen_min_r = None
    chosen_min_c = None
    chosen_max_r = None
    chosen_max_c = None
    chosen_T = None
    for M in colors:
        T = colors[0] if M == colors[1] else colors[1]
        if not pos[M]:
            continue
        rs = [p[0] for p in pos[M]]
        cs = [p[1] for p in pos[M]]
        min_r = min(rs)
        max_r = max(rs)
        min_c = min(cs)
        max_c = max(cs)
        if max_r == min_r or max_c == min_c:
            continue
        fill_rs = set(range(min_r + 1, max_r))
        fill_cs = set(range(min_c + 1, max_c))
        all_inside = all(r in fill_rs and c in fill_cs for r, c in pos[T])
        if all_inside:
            chosen_M = M
            chosen_T = T
            chosen_min_r = min_r
            chosen_max_r = max_r
            chosen_min_c = min_c
            chosen_max_c = max_c
            break
    if chosen_M is None:
        return []
    height = chosen_max_r - chosen_min_r - 1
    width = chosen_max_c - chosen_min_c - 1
    out = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        in_r = chosen_min_r + 1 + i
        for j in range(width):
            in_c = chosen_min_c + 1 + j
            if I[in_r][in_c] == chosen_T:
                out[i][j] = chosen_M
    return out

def solve_124(I):
    if not I or not I[0]:
        return []
    
    height = len(I)
    width = len(I[0])
    visited = [[False] * width for _ in range(height)]
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in range(height):
        for c in range(width):
            if I[r][c] > 0 and not visited[r][c]:
                color = I[r][c]
                size = 0
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    size += 1
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < height and 0 <= nc < width and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append((size, color))
    
    if not components:
        return []
    
    components.sort(key=lambda x: -x[0])
    max_n = components[0][0]
    parity = max_n % 2
    output = [[0] * max_n for _ in range(max_n)]
    
    for L, c in components:
        eff_s = L if L % 2 == parity else L - 1
        start = (max_n - eff_s) // 2
        for i in range(start, start + eff_s):
            for j in range(start, start + eff_s):
                output[i][j] = c
    
    return output

def solve_125(I):
    output = [row[:] for row in I]
    rows = len(output)
    cols = len(output[0])
    greens = []
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 3:
                greens.append((i, j))
    for r, c in greens:
        # Row r-2: 5 from c-2 to c+2
        if 0 <= r - 2 < rows:
            for j in range(max(0, c - 2), min(cols, c + 3)):
                output[r - 2][j] = 5
        # Row r-1: 2 at c-2, 5 at c, 2 at c+2
        if 0 <= r - 1 < rows:
            if 0 <= c - 2 < cols:
                output[r - 1][c - 2] = 2
            if 0 <= c < cols:
                output[r - 1][c] = 5
            if 0 <= c + 2 < cols:
                output[r - 1][c + 2] = 2
        # Row r: 2 at c-2, 3 at c (already), 2 at c+2
        if 0 <= r < rows:
            if 0 <= c - 2 < cols:
                output[r][c - 2] = 2
            if 0 <= c + 2 < cols:
                output[r][c + 2] = 2
        # Row r+1: 2 at c-2, 2 at c+2
        if 0 <= r + 1 < rows:
            if 0 <= c - 2 < cols:
                output[r + 1][c - 2] = 2
            if 0 <= c + 2 < cols:
                output[r + 1][c + 2] = 2
        # Row r+2: all 2, except 8 from c-2 to c+2
        if 0 <= r + 2 < rows:
            for j in range(cols):
                output[r + 2][j] = 2
            for j in range(max(0, c - 2), min(cols, c + 3)):
                output[r + 2][j] = 8
    return output

def solve_126(I):
    if not I or not I[0]:
        return I

    h = len(I)
    w = len(I[0])

    # Find bounding box of 5
    positions_5 = [(r, c) for r in range(h) for c in range(w) if I[r][c] == 5]
    if not positions_5:
        return [row[:] for row in I]

    min_r = min(r for r, c in positions_5)
    max_r = max(r for r, c in positions_5)
    min_c = min(c for r, c in positions_5)
    max_c = max(c for r, c in positions_5)

    bar_h = max_r - min_r + 1
    bar_w = max_c - min_c + 1

    output = [row[:] for row in I]

    is_vertical = bar_h > bar_w

    if is_vertical:
        for r in range(h):
            left_count = sum(1 for c in range(min_c) if I[r][c] != 0 and I[r][c] != 5)
            right_count = sum(1 for c in range(max_c + 1, w) if I[r][c] != 0 and I[r][c] != 5)

            new_min_c = min_c - left_count
            for c in range(new_min_c, min_c):
                if 0 <= c < w:
                    output[r][c] = 5

            new_max_c = max_c + right_count
            for c in range(max_c + 1, new_max_c + 1):
                if 0 <= c < w:
                    output[r][c] = 5
    else:
        for c in range(w):
            above_count = sum(1 for r in range(min_r) if I[r][c] != 0 and I[r][c] != 5)
            below_count = sum(1 for r in range(max_r + 1, h) if I[r][c] != 0 and I[r][c] != 5)

            new_min_r = min_r - above_count
            for r in range(new_min_r, min_r):
                if 0 <= r < h:
                    output[r][c] = 5

            new_max_r = max_r + below_count
            for r in range(max_r + 1, new_max_r + 1):
                if 0 <= r < h:
                    output[r][c] = 5

    # Remove non-absorbed colored cells
    for r in range(h):
        for c in range(w):
            if I[r][c] != 0 and I[r][c] != 5 and output[r][c] != 5:
                output[r][c] = 0

    return output

from collections import Counter
import numpy as np

def solve_127(I):
    if not grid_lst or not grid_lst[0]:
        return []
    I = grid_lst
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and not visited[r][c]:
                comp = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    color = I[cr][cc]
                    comp.append((cr, cc, color))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] != 0:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append(comp)

    if not components:
        return []

    sizes = [len(comp) for comp in components]
    max_size = max(sizes)
    base_idx = sizes.index(max_size)
    base_cells = components[base_idx]
    small_comps = [components[i] for i in range(len(components)) if i != base_idx]

    color_counts = Counter([color for _, _, color in base_cells])
    main_color = color_counts.most_common(1)[0][0]

    patterns = {}
    for comp in small_comps:
        colors = [color for _, _, color in comp]
        color_set = set(colors)
        non_two = color_set - {2}
        if len(non_two) == 1:
            C = list(non_two)[0]
            if colors.count(C) == 1:
                for pr, pc, pcolor in comp:
                    if pcolor == C:
                        core_r, core_c = pr, pc
                        break
                patterns[C] = comp

    base_min_r = min(rr for rr, _, _ in base_cells)
    base_max_r = max(rr for rr, _, _ in base_cells)
    base_min_c = min(cc for _, cc, _ in base_cells)
    base_max_c = max(cc for _, cc, _ in base_cells)
    h = base_max_r - base_min_r + 1
    w = base_max_c - base_min_c + 1
    output = [[main_color] * w for _ in range(h)]

    anchors = [(rr, cc, color) for rr, cc, color in base_cells if color != main_color]

    for ar, ac, a_color in anchors:
        if a_color in patterns:
            comp = patterns[a_color]
            core_r, core_c = None, None
            for pr, pc, pcolor in comp:
                if pcolor == a_color:
                    core_r, core_c = pr, pc
                    break
            for pr, pc, pcolor in comp:
                delta_r = pr - core_r
                delta_c = pc - core_c
                tr = ar + delta_r
                tc = ac + delta_c
                if base_min_r <= tr <= base_max_r and base_min_c <= tc <= base_max_c:
                    out_i = tr - base_min_r
                    out_j = tc - base_min_c
                    output[out_i][out_j] = pcolor

    return output

import numpy as np

def solve_128(I):
    if not I:
        return I
    g = np.array(I)
    rows, cols = g.shape
    visited = np.zeros((rows, cols), dtype=bool)

    def find_component(i, j):
        component = []
        stack = [(i, j)]
        visited[i, j] = True
        while stack:
            x, y = stack.pop()
            component.append((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and g[nx, ny] == 1:
                    visited[nx, ny] = True
                    stack.append((nx, ny))
        return component

    components = []
    for i in range(rows):
        for j in range(cols):
            if g[i, j] == 1 and not visited[i, j]:
                comp = find_component(i, j)
                components.append(comp)

    output = g.copy()
    for comp in components:
        if not comp:
            continue
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        min_r = min(rs)
        max_r = max(rs)
        min_c = min(cs)
        max_c = max(cs)
        center_r = (min_r + max_r) // 2
        center_c = (min_c + max_c) // 2
        # Horizontal line
        for c in range(cols):
            if output[center_r, c] == 8:
                output[center_r, c] = 6
        # Vertical line
        for r in range(rows):
            if output[r, center_c] == 8:
                output[r, center_c] = 6

    return output.tolist()

def solve_129(I):
    if not I or not I[0]:
        return I
    height = len(I)
    width = len(I[0])
    # Find the color C
    C = 0
    for row in I:
        for cell in row:
            if cell != 0:
                C = cell
                break
        if C != 0:
            break
    if C == 0:
        return [list(row) for row in I]
    # Find max_r
    max_r = 0
    for r in range(height):
        for j in range(width):
            if I[r][j] == C:
                max_r = max(max_r, r)
    # Create output
    output = [[0 for _ in range(width)] for _ in range(height)]
    for r in range(height):
        shift = max_r - r
        for j in range(width):
            if I[r][j] == C:
                new_j = j - shift
                if 0 <= new_j < width:
                    output[r][new_j] = C
    return output

def solve_130(I):
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    centers = []
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 5:
                centers.append((i, j))
    for r, c in centers:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                    output[nr][nc] = 1
    return output

import numpy as np

def solve_131(I):
    I = np.array(I)
    rows, cols = I.shape
    visited = np.zeros((rows, cols), bool)
    for i in range(rows):
        for j in range(cols):
            if I[i, j] != 0 and not visited[i, j]:
                color = I[i, j]
                stack = [(i, j)]
                visited[i, j] = True
                component = []
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and I[nx, ny] == color:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
                rs = [p[0] for p in component]
                cs = [p[1] for p in component]
                min_r, max_r = min(rs), max(rs)
                min_c, max_c = min(cs), max(cs)
                for ii in range(min_r + 1, max_r):
                    for jj in range(min_c + 1, max_c):
                        I[ii, jj] = 0
    return I.tolist()

from collections import deque

def solve_132(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    bg = I[0][0]
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != bg and not visited[r][c]:
                color = I[r][c]
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and I[nx][ny] == color:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                min_r = min(rx for rx, _ in comp)
                components.append((min_r, comp, color))
    components.sort(key=lambda x: x[0])
    for i, (_, comp, color) in enumerate(components):
        shift = -1 if i % 2 == 0 else 1
        # clear
        for r, c in comp:
            I[r][c] = bg
        # set new
        for r, c in comp:
            new_c = c + shift
            if 0 <= new_c < cols:
                I[r][new_c] = color
    return I

def solve_133(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(r, c, component):
        stack = [(r, c)]
        visited[r][c] = True
        component.append((r, c))
        while stack:
            cr, cc = stack.pop()
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == 1:
                    visited[nr][nc] = True
                    component.append((nr, nc))
                    stack.append((nr, nc))

    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 1 and not visited[r][c]:
                comp = []
                dfs(r, c, comp)
                components.append(comp)

    output = [row[:] for row in I]
    for comp in components:
        if not comp:
            continue
        rs = [pos[0] for pos in comp]
        cs = [pos[1] for pos in comp]
        min_r = min(rs)
        max_r = max(rs)
        min_c = min(cs)
        max_c = max(cs)

        # Find seed color C
        seeds = []
        for rr in range(min_r, max_r + 1):
            for cc in range(min_c, max_c + 1):
                if I[rr][cc] != 0 and I[rr][cc] != 1:
                    seeds.append(I[rr][cc])
        if not seeds:
            continue
        C = seeds[0]

        # Fill above row if exists
        if min_r > 0:
            above = min_r - 1
            for cc in range(min_c, max_c + 1):
                if output[above][cc] == 0:
                    output[above][cc] = C

        # Fill gaps in the box
        for rr in range(min_r, max_r + 1):
            for cc in range(min_c, max_c + 1):
                if output[rr][cc] == 0:
                    output[rr][cc] = C

    return output

import numpy as np
from collections import deque

def solve_134(I):
    if not I or not I[0]:
        return I
    g = np.array(I)
    rows, cols = g.shape
    visited = np.zeros_like(g, dtype=bool)

    def bfs(start_r, start_c, mark=False):
        queue = deque([(start_r, start_c)])
        visited[start_r, start_c] = True
        component = [] if mark else [(start_r, start_c)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and g[nr, nc] == 0:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
                    if not mark:
                        component.append((nr, nc))
        return component

    # Mark outside 0s
    for r in range(rows):
        for c in [0, cols - 1]:
            if g[r, c] == 0 and not visited[r, c]:
                bfs(r, c, mark=True)
    for c in range(cols):
        for r in [0, rows - 1]:
            if g[r, c] == 0 and not visited[r, c]:
                bfs(r, c, mark=True)

    # Find and fill enclosed square holes
    for r in range(rows):
        for c in range(cols):
            if g[r, c] == 0 and not visited[r, c]:
                component = bfs(r, c, mark=False)
                if component:
                    rs = [p[0] for p in component]
                    cs = [p[1] for p in component]
                    minr, maxr = min(rs), max(rs)
                    minc, maxc = min(cs), max(cs)
                    h = maxr - minr + 1
                    w = maxc - minc + 1
                    if h == w and len(component) == h * w:
                        for pr, pc in component:
                            g[pr, pc] = 2

    return g.tolist()

def solve_135(I):
    if not I or not I[0]:
        return [[7]]  # Assume 7 if empty, though not in examples
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    sizes = []
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 2 and not visited[i][j]:
                size = 0
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    r, c = stack.pop()
                    size += 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == 2:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                sizes.append(size)
    num_components = len(sizes)
    if num_components == 0:
        return [[7]]  # Assume 7 if no reds, though not in examples
    elif num_components == 1:
        return [[7]]
    else:
        if all(s == sizes[0] for s in sizes):
            return [[1]]
        else:
            return [[7]]

def solve_136(I):
    rows = len(I)
    cols = len(I[0])
    red_r = red_c = -1
    greens = []
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 2:
                red_r, red_c = i, j
            elif I[i][j] == 3:
                greens.append((i, j))
    num_greens = len(greens)
    N = num_greens + 1
    total = 2 * N + 1
    if num_greens == 0:
        return [[0] * total for _ in range(total)]  # Arbitrary, not in examples
    min_r = min(r for r, _ in greens)
    max_r = max(r for r, _ in greens)
    min_c = min(c for _, c in greens)
    max_c = max(c for _, c in greens)
    dist_up = red_r - min_r
    dist_left = red_c - min_c
    dist_down = max_r - red_r
    dist_right = max_c - red_c
    start_r1 = 0 if dist_up > 0 else 1
    start_c1 = 0 if dist_left > 0 else 1
    start_r2 = start_r1 + N
    start_c2 = start_c1 + N
    output = [[0] * total for _ in range(total)]
    for i in range(start_r1, start_r1 + N):
        for j in range(start_c1, start_c1 + N):
            output[i][j] = 3
    for i in range(start_r2, start_r2 + N):
        for j in range(start_c2, start_c2 + N):
            output[i][j] = 3
    return output

def solve_137(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(r, c, component):
        stack = [(r, c)]
        visited[r][c] = True
        component.append((r, c))
        while stack:
            cr, cc = stack.pop()
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] != 0:
                    visited[nr][nc] = True
                    component.append((nr, nc))
                    stack.append((nr, nc))

    output = [row[:] for row in I]
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and not visited[r][c]:
                component = []
                dfs(r, c, component)
                colors = set()
                for pr, pc in component:
                    colors.add(I[pr][pc])
                if len(colors) == 2:
                    a, b = list(colors)
                    for pr, pc in component:
                        if output[pr][pc] == a:
                            output[pr][pc] = b
                        else:
                            output[pr][pc] = a
    return output

def solve_138(I):
    height = len(I)
    width = len(I[0])
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    # Find seeds and map rows and columns to colors
    row_to_color = {}
    col_to_color = {}
    for r in range(height):
        for c in range(width):
            color = I[r][c]
            if color != 0:
                row_to_color[r] = color
                col_to_color[c] = color
    
    # Set vertical lines
    for c, color in col_to_color.items():
        for r in range(height):
            output[r][c] = color
    
    # Set horizontal lines with overrides for conflicts
    for r, color in row_to_color.items():
        for c in range(width):
            existing = output[r][c]
            if existing != 0 and existing != color:
                output[r][c] = 2
            else:
                output[r][c] = color
    
    return output

def solve_139(I):
    if not I:
        return []
    h = len(I)
    w = len(I[0])
    # Compute transpose
    transpose = [[I[j][i] for j in range(h)] for i in range(w)]
    # Horizontal flip of transpose (reverse each row)
    right = [row[::-1] for row in transpose]
    # Build top half
    top = [I[i] + right[i] for i in range(h)]
    # Compute bottom half as 180 rotation of top
    bottom = top[::-1]
    bottom = [row[::-1] for row in bottom]
    # Combine top and bottom
    return top + bottom

def solve_140(I):
    odd_indices = [1, 3, 5, 7, 9]
    n = len(odd_indices)
    out_size = 4 * n
    output = [[0 for _ in range(out_size)] for _ in range(out_size)]
    for i in range(n):
        for j in range(n):
            color = I[odd_indices[i]][odd_indices[j]]
            for dr in range(4):
                for dc in range(4):
                    output[4 * i + dr][4 * j + dc] = color
    return output

import numpy as np

def solve_141(I):
    I = np.array(I)
    rows, cols = I.shape
    output = np.full((rows, cols), 7, dtype=int)
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(rows):
        for j in range(cols):
            if I[i, j] != 7 and not visited[i, j]:
                component = []
                stack = [(i, j)]
                visited[i, j] = True
                colors = []
                eights = []
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    if I[x, y] != 8:
                        colors.append(I[x, y])
                    else:
                        eights.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and I[nx, ny] != 7 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
                if len(set(colors)) != 1 or len(eights) != 1:
                    continue
                C = colors[0]
                er, ec = eights[0]
                min_r = min(r for r, c in component)
                max_r = max(r for r, c in component)
                min_c = min(c for r, c in component)
                max_c = max(c for r, c in component)
                if len(component) != (max_r - min_r + 1) * (max_c - min_c + 1):
                    continue
                dy = 0
                if er == min_r:
                    dy = -1
                elif er == max_r:
                    dy = 1
                dx = 0
                if ec == min_c:
                    dx = -1
                elif ec == max_c:
                    dx = 1
                new_min_r = min_r + dy
                new_max_r = max_r + dy
                new_min_c = min_c + dx
                new_max_c = max_c + dx
                if new_min_r < 0 or new_max_r >= rows or new_min_c < 0 or new_max_c >= cols:
                    continue
                for nr in range(new_min_r, new_max_r + 1):
                    for nc in range(new_min_c, new_max_c + 1):
                        output[nr, nc] = C
    return output.tolist()

def solve_142(I):
    n = len(I)
    m = n // 2
    C = I[m][0]
    out_n = n - 1
    output = [[0 for _ in range(out_n)] for _ in range(out_n)]
    
    # Fill top-left from input, replacing non-zero with C
    for r in range(m):
        for c in range(m):
            if I[r][c] != 0:
                output[r][c] = C
    
    # Mirror horizontally to top-right
    for r in range(m):
        for c in range(m):
            output[r][out_n - 1 - c] = output[r][c]
    
    # Mirror vertically to bottom
    for r in range(m):
        for c in range(out_n):
            output[out_n - 1 - r][c] = output[r][c]
    
    return output

def solve_143(I):
    if not I:
        return []
    n = len(I)
    # Assume square I
    assert all(len(row) == n for row in I)
    
    # Find the non-zero color C
    colors = set()
    for row in I:
        for val in row:
            if val != 0:
                colors.add(val)
    assert len(colors) == 1
    c = next(iter(colors))
    
    # Create inverted I
    inverted = []
    for row in I:
        new_row = [c if x == 0 else 0 for x in row]
        inverted.append(new_row)
    
    # Tile horizontally twice
    horiz_tiled = []
    for row in inverted:
        horiz_tiled.append(row + row)
    
    # Tile vertically twice
    output = horiz_tiled + horiz_tiled
    
    return output

def solve_144(I):
    # Find purple positions and count yellows
    pos = []
    num_yellow = 0
    rows = len(I)
    cols = len(I[0])
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 8:
                pos.append((r, c))
            elif I[r][c] == 4:
                num_yellow += 1

    if not pos:
        return [[], [], []]  # Empty case, but assume not

    min_r = min(r for r, c in pos)
    max_r = max(r for r, c in pos)
    min_c = min(c for r, c in pos)
    max_c = max(c for r, c in pos)

    bb_h = max_r - min_r + 1
    bb_w = max_c - min_c + 1

    pat_h = 3
    pat_w = 3
    pattern = [[0] * pat_w for _ in range(pat_h)]

    start_r = pat_h - bb_h
    start_c = 0
    for i in range(bb_h):
        for j in range(bb_w):
            val = I[min_r + i][min_c + j]
            pattern[start_r + i][start_c + j] = val if val == 8 else 0

    # Replicate
    out_w = pat_w * num_yellow
    output = [[0] * out_w for _ in range(pat_h)]
    for k in range(num_yellow):
        for r in range(pat_h):
            for j in range(pat_w):
                output[r][k * pat_w + j] = pattern[r][j]

    return output

def solve_145(I):
    h = len(I)
    w = len(I[0])
    
    # Find grey position
    grey_r, grey_c = None, None
    for r in range(h):
        for c in range(w):
            if I[r][c] == 5:
                grey_r, grey_c = r, c
                break
        if grey_r is not None:
            break
    
    # Find color C
    C = None
    for r in range(h):
        for c in range(w):
            if I[r][c] not in (0, 5):
                if C is None:
                    C = I[r][c]
                elif C != I[r][c]:
                    # Assume single color, but handle if not
                    pass
    
    # Find starting cells adjacent to grey
    starts = set()
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr = grey_r + dr
            nc = grey_c + dc
            if 0 <= nr < h and 0 <= nc < w and I[nr][nc] == C:
                starts.add((nr, nc))
    
    # Flood fill to get component
    component = set()
    stack = list(starts)
    while stack:
        r, c = stack.pop()
        if (r, c) in component:
            continue
        component.add((r, c))
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and I[nr][nc] == C and (nr, nc) not in component:
                    stack.append((nr, nc))
    
    # Normalize to 3x3
    if not component:
        return [[0] * 3 for _ in range(3)]
    
    min_r = min(r for r, c in component)
    min_c = min(c for r, c in component)
    
    out = [[0] * 3 for _ in range(3)]
    for r, c in component:
        or_ = r - min_r
        oc = c - min_c
        out[or_][oc] = C
    
    return out

from collections import Counter

def solve_146(I):
    # Assume I is 3x3
    colors = [I[i][j] for i in range(3) for j in range(3)]
    freq = Counter(colors)
    if not freq:
        return [[0] * 9 for _ in range(9)]
    min_count = min(freq.values())
    candidates = [col for col, cnt in freq.items() if cnt == min_count]
    c = min(candidates)  # Smallest color if tie
    positions = [(i, j) for i in range(3) for j in range(3) if I[i][j] == c]
    large = [[0] * 9 for _ in range(9)]
    for pi, pj in positions:
        for di in range(3):
            for dj in range(3):
                large[3 * pi + di][3 * pj + dj] = I[di][dj]
    return large

def solve_147(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    
    # Find green positions to compute center
    green_positions = []
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 3:
                green_positions.append((i, j))
    
    if not green_positions:
        return I  # No green, no transformation
    
    min_r = min(p[0] for p in green_positions)
    max_r = max(p[0] for p in green_positions)
    min_c = min(p[1] for p in green_positions)
    max_c = max(p[1] for p in green_positions)
    
    # Assume 2x2
    cr = (min_r + max_r) / 2.0
    cc = (min_c + max_c) / 2.0
    
    # Collect red positions
    reds = [(i, j) for i in range(rows) for j in range(cols) if I[i][j] == 2]
    
    # Copy I
    output = [row[:] for row in I]
    
    # For each red, add rotated versions
    for r, c in reds:
        dx = r - cr
        dy = c - cc
        
        # 90 CW
        new_dx = dy
        new_dy = -dx
        nr = round(cr + new_dx)
        nc = round(cc + new_dy)
        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 0:
            output[nr][nc] = 2
        
        # 180
        new_dx = -dx
        new_dy = -dy
        nr = round(cr + new_dx)
        nc = round(cc + new_dy)
        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 0:
            output[nr][nc] = 2
        
        # 270 CW
        new_dx = -dy
        new_dy = dx
        nr = round(cr + new_dx)
        nc = round(cc + new_dy)
        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 0:
            output[nr][nc] = 2
    
    return output

def solve_148(I):
    if not I or not I[0]:
        return I
    output = [row[:] for row in I]
    pairs = [{1, 8}, {4, 7}]
    for r in range(len(I)):
        colored = [(c, I[r][c]) for c in range(len(I[0])) if I[r][c] != 0]
        if len(colored) == 2:
            left_pos, left_col = colored[0]
            right_pos, right_col = colored[1]
            if set([left_col, right_col]) in pairs:
                target = left_pos + 1
                if target != right_pos:
                    output[r][target] = right_col
                    output[r][right_pos] = 0
    return output

def solve_149(I):
    if not I:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    
    # Find height h of top uniform non-zero rows
    h = 0
    for r in range(rows):
        color = I[r][0]
        if color == 0:
            break
        uniform = all(c == color for c in I[r])
        if not uniform:
            break
        h += 1
    
    # Copy top h rows reversed to bottom h rows
    for i in range(h):
        output[rows - h + i] = I[h - 1 - i]
    
    return output

def solve_150(I):
    h = len(I)
    if h == 0:
        return [[0, 0]]
    w = len(I[0])
    out = [[0] * (w + 2) for _ in range(h + 2)]
    
    # Top row
    for j in range(w):
        out[0][j + 1] = I[0][j]
    
    # Bottom row
    for j in range(w):
        out[h + 1][j + 1] = I[h - 1][j]
    
    # Inner rows
    for i in range(h):
        # Prepend
        out[i + 1][0] = I[i][0]
        # Middle
        for j in range(w):
            out[i + 1][j + 1] = I[i][j]
        # Append
        out[i + 1][w + 1] = I[i][w - 1]
    
    return out

def solve_151(I):
    if not I:
        return I
    height = len(I)
    width = len(I[0])
    # Find the special cell
    r_spec, c_spec, C = -1, -1, -1
    for r in range(height):
        for c in range(width):
            if I[r][c] != 8:
                r_spec = r
                c_spec = c
                C = I[r][c]
                break  # Assume only one
        if r_spec != -1:
            break
    if r_spec == -1:
        return [row[:] for row in I]
    # Rows
    dist_top = r_spec
    dist_bottom = height - 1 - r_spec
    min_dist_row = min(dist_top, dist_bottom)
    if dist_top < dist_bottom or dist_top == dist_bottom:
        # Expand up
        min_r = r_spec - min_dist_row
        max_r = r_spec
    else:
        # Expand down
        min_r = r_spec
        max_r = r_spec + min_dist_row
    # Columns
    dist_left = c_spec
    dist_right = width - 1 - c_spec
    min_dist_col = min(dist_left, dist_right)
    if dist_left < dist_right or dist_left == dist_right:
        # Expand left
        min_c = c_spec - min_dist_col
        max_c = c_spec
    else:
        # Expand right
        min_c = c_spec
        max_c = c_spec + min_dist_col
    # Create output
    output = [row[:] for row in I]
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            output[r][c] = C
    return output

import numpy as np
from itertools import groupby

def solve_152(I):
    if not grid_lst or not grid_lst[0]:
        return []
    I = np.array(grid_lst)
    left_column = I[:, 0]
    top_row = I[0, :]
    is_vertical_layers = np.all(left_column == left_column[0])
    sequence = top_row if is_vertical_layers else left_column
    colors = [key for key, _ in groupby(sequence)]
    if is_vertical_layers:
        return [colors]
    else:
        return [[color] for color in colors]

def solve_153(I):
    flipped = I[::-1]
    return flipped + I

def solve_154(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    
    # Find the cross center and color B
    cr, cc, B = None, None, None
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            val = I[r][c]
            if val != 0:
                if (I[r-1][c-1] == val and
                    I[r-1][c+1] == val and
                    I[r+1][c-1] == val and
                    I[r+1][c+1] == val):
                    # Collect positions
                    positions = [(r, c), (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
                    all_pos = [(i, j) for i in range(rows) for j in range(cols) if I[i][j] == val]
                    if set(all_pos) == set(positions):
                        cr, cc, B = r, c, val
                        break
        if cr is not None:
            break
    
    if cr is None:
        return I  # No transformation if no cross found
    
    # Find A
    A = None
    for row in I:
        for val in row:
            if val != 0 and val != B:
                A = val
                break
        if A is not None:
            break
    
    if A is None:
        return I
    
    # Copy I
    output = [row[:] for row in I]
    
    # Reflection functions
    refl_h = lambda r, c: (2 * cr - r, c)
    refl_v = lambda r, c: (r, 2 * cc - c)
    refl_both = lambda r, c: (2 * cr - r, 2 * cc - c)
    reflections = [refl_h, refl_v, refl_both]
    
    # Add reflections
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == A:
                for refl in reflections:
                    nr, nc = refl(i, j)
                    if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                        output[nr][nc] = A
    
    return output

def solve_155(I):
    if not I or not I[0]:
        return I
    
    n = len(I)
    output = [row[:] for row in I]
    
    row_used = [[False] * (n + 1) for _ in range(n)]
    col_used = [[False] * (n + 1) for _ in range(n)]
    empties = []
    
    for i in range(n):
        for j in range(n):
            c = output[i][j]
            if c == 0:
                empties.append((i, j))
            else:
                row_used[i][c] = True
                col_used[j][c] = True
    
    def backtrack(idx):
        if idx == len(empties):
            return True
        r, c = empties[idx]
        for num in range(1, n + 1):
            if not row_used[r][num] and not col_used[c][num]:
                output[r][c] = num
                row_used[r][num] = True
                col_used[c][num] = True
                if backtrack(idx + 1):
                    return True
                output[r][c] = 0
                row_used[r][num] = False
                col_used[c][num] = False
        return False
    
    backtrack(0)
    return output

from collections import deque

def solve_156(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    bg = 7
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != bg and not visited[r][c]:
                color = I[r][c]
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and I[nx][ny] == color:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                has_top = any(rx == 0 for rx, _ in comp)
                has_bottom = any(rx == rows - 1 for rx, _ in comp)
                has_left = any(cy == 0 for _, cy in comp)
                has_right = any(cy == cols - 1 for _, cy in comp)
                if not (has_top and has_bottom and has_left and has_right):
                    components.append((len(comp), comp, color))
    components.sort(key=lambda x: x[0])
    output = [[bg for _ in range(cols)] for _ in range(rows)]
    current_left = 0
    for size, comp, color in components:
        if not comp:
            continue
        max_r = max(rx for rx, _ in comp)
        min_c = min(cy for _, cy in comp)
        shift_row = (rows - 1) - max_r
        shift_col = current_left - min_c
        for rx, cy in comp:
            new_r = rx + shift_row
            new_c = cy + shift_col
            output[new_r][new_c] = color
        current_left = max(cy + shift_col for _, cy in comp) + 1
    return output

def solve_157(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Find defect color C (assuming one color != 0 and != 1)
    C = None
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and I[r][c] != 1:
                C = I[r][c]
                break
        if C is not None:
            break
    
    if C is None:
        return [row[:] for row in I]  # No defect, return copy
    
    # Find defect positions
    defect_positions = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == C:
                defect_positions.append((r, c))
    
    # Find defect rows and cols
    rs = [p[0] for p in defect_positions]
    cs = [p[1] for p in defect_positions]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    defect_rows = set(range(min_r, max_r + 1))
    defect_cols = set(range(min_c, max_c + 1))
    
    # Create output
    output = [row[:] for row in I]
    
    # Apply transformations
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 1:
                if r in defect_rows:
                    output[r][c] = C
                elif c in defect_cols:
                    output[r][c] = C
    
    return output

def solve_158(I):
    output = []
    for row in I:
        pos1 = [c for c, val in enumerate(row) if val == 1]
        pos8 = [c for c, val in enumerate(row) if val == 8]
        if len(pos1) == 1 and len(pos8) == 1:
            c1 = pos1[0]
            c8 = pos8[0]
            if c1 < c8:
                extracted = row[c1 + 1 : c8]
                output.append(extracted)
    return output

import numpy as np

def solve_159(I):
    I = np.array(grid_lst)
    height, width = I.shape
    middle = height // 2
    upper = I[0:middle, :]
    lower = I[middle + 1:, :]
    out = np.zeros((middle, width), dtype=int)
    diff = upper != lower
    out[diff] = 3
    return out.tolist()

def solve_160(I):
    if not I or not I[0]:
        return []

    rows = len(I)
    cols = len(I[0])

    purples = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 8:
                purples.append((r, c))

    if not purples:
        return I

    purples.sort(key=lambda x: x[0])

    if len(purples) < 2:
        return I  # Assuming at least 2 based on examples

    dr = purples[1][0] - purples[0][0]
    dc = purples[1][1] - purples[0][1]

    def extend(start_r, start_c, dir_dr, dir_dc):
        current_r = start_r
        current_c = start_c
        while True:
            next_r = current_r + dir_dr
            next_c = current_c + dir_dc
            if not (0 <= next_r < rows and 0 <= next_c < cols):
                break
            if I[next_r][next_c] != 0:
                vert_r = current_r + dir_dr
                vert_c = current_c
                horiz_r = current_r
                horiz_c = current_c + dir_dc
                vert_red = (0 <= vert_r < rows and 0 <= vert_c < cols) and I[vert_r][vert_c] == 2
                horiz_red = (0 <= horiz_r < rows and 0 <= horiz_c < cols) and I[horiz_r][horiz_c] == 2
                if vert_red and horiz_red:
                    dir_dr = -dir_dr
                    dir_dc = -dir_dc
                elif vert_red:
                    dir_dr = -dir_dr
                elif horiz_red:
                    dir_dc = -dir_dc
                else:
                    break
                next_r = current_r + dir_dr
                next_c = current_c + dir_dc
                if not (0 <= next_r < rows and 0 <= next_c < cols):
                    break
                if I[next_r][next_c] != 0:
                    break
            if I[next_r][next_c] == 0:
                I[next_r][next_c] = 3
            else:
                break
            current_r = next_r
            current_c = next_c

    # Forward extension
    extend(purples[-1][0], purples[-1][1], dr, dc)

    # Backward extension
    back_dr = -dr
    back_dc = -dc
    extend(purples[0][0], purples[0][1], back_dr, back_dc)

    return I

import numpy as np

def solve_161(I):
    if not I or not I[0]:
        return I
    h = len(I)
    w = len(I[0])
    # Assume bottom row is uniform
    b = I[h-1][0]
    # Collect s for row 0
    s = []
    for col in range(w):
        if I[0][col] != b:
            s.append(I[0][col])
        else:
            break
    l = len(s)
    # Find minimal p
    p = None
    for pp in range(1, l + 1):
        unit = s[0:pp]
        if all(s[i] == unit[i % pp] for i in range(l)):
            p = pp
            break
    c = s[0:p]
    # For row 0, d_even = 0
    d_even = 0
    # Collect s_odd for row 1 (if h > 1)
    if h > 1:
        s_odd = []
        for col in range(w):
            if I[1][col] != b:
                s_odd.append(I[1][col])
            else:
                break
        l_odd = len(s_odd)
        # Find d_odd
        d_odd = None
        for d in range(p):
            if all(s_odd[i] == c[(i + d) % p] for i in range(l_odd)):
                d_odd = d
                break
    else:
        d_odd = (d_even + 1) % p  # If only one row, but unlikely
    # Now create output
    output = [[0] * w for _ in range(h)]
    for r in range(h):
        if r % 2 == 0:
            start_d = (d_even + 1) % p
        else:
            start_d = (d_odd + 1) % p
        for col in range(w):
            output[r][col] = c[(col + start_d) % p]
    return output

def solve_162(I):
    if not I or not I[0]:
        return I
    
    h = len(I)
    w = len(I[0])
    
    def find_components():
        visited = set()
        components = []
        for i in range(h):
            for j in range(w):
                if I[i][j] == 6 and (i, j) not in visited:
                    comp = []
                    stack = [(i, j)]
                    visited.add((i, j))
                    while stack:
                        r, c = stack.pop()
                        comp.append((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and I[nr][nc] == 6 and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                stack.append((nr, nc))
                    components.append(comp)
        return components
    
    components = find_components()
    
    seeds = []
    for i in range(h):
        for j in range(w):
            if I[i][j] != 0 and I[i][j] != 6:
                seeds.append((i, j, I[i][j]))
    
    comp_to_info = {}
    for sr, sc, col in seeds:
        min_dist = float('inf')
        assigned_comp = None
        for comp in components:
            this_min = min(abs(r - sr) + abs(c - sc) for r, c in comp)
            if this_min < min_dist:
                min_dist = this_min
                assigned_comp = comp
        comp_to_info[id(assigned_comp)] = (col, sr, sc)
    
    vectors_dx = []
    vectors_dy = []
    for comp in components:
        cid = id(comp)
        color, sr, sc = comp_to_info[cid]
        n = len(comp)
        center_r = sum(r for r, c in comp) / n
        center_c = sum(c for r, c in comp) / n
        dx = sc - center_c
        dy = sr - center_r
        vectors_dx.append(dx)
        vectors_dy.append(dy)
    
    avg_abs_dx = sum(abs(x) for x in vectors_dx) / len(vectors_dx)
    avg_abs_dy = sum(abs(y) for y in vectors_dy) / len(vectors_dy)
    
    if avg_abs_dy > avg_abs_dx:
        stacking = 'horizontal'
        sort_key = lambda c: sum(cc for rr, cc in c) / len(c)
    else:
        stacking = 'vertical'
        sort_key = lambda c: sum(rr for rr, cc in c) / len(c)
    
    sorted_comps = sorted(components, key=sort_key)
    
    small_grids = []
    for comp in sorted_comps:
        cid = id(comp)
        color = comp_to_info[cid][0]
        minr = min(r for r, c in comp)
        minc = min(c for r, c in comp)
        maxr = max(r for r, c in comp)
        maxc = max(c for r, c in comp)
        sh = maxr - minr + 1
        sw = maxc - minc + 1
        sg = [[0] * sw for _ in range(sh)]
        for r, c in comp:
            sg[r - minr][c - minc] = color
        small_grids.append(sg)
    
    if stacking == 'horizontal':
        if not small_grids:
            return []
        out_h = max(len(sg) for sg in small_grids)
        out_w = sum(len(sg[0]) for sg in small_grids)
        out = [[0] * out_w for _ in range(out_h)]
        cur = 0
        for sg in small_grids:
            sh = len(sg)
            sw = len(sg[0])
            for i in range(sh):
                for j in range(sw):
                    out[i][cur + j] = sg[i][j]
            cur += sw
    else:
        if not small_grids:
            return []
        out_w = max(len(sg[0]) for sg in small_grids)
        out_h = sum(len(sg) for sg in small_grids)
        out = [[0] * out_w for _ in range(out_h)]
        cur = 0
        for sg in small_grids:
            sh = len(sg)
            sw = len(sg[0])
            for i in range(sh):
                for j in range(sw):
                    out[cur + i][j] = sg[i][j]
            cur += sh
    
    return out

import copy

def solve_163(I):
    I = copy.deepcopy(I)
    rows = len(I)
    diag = rows - 1

    # Find current red positions on the diagonal
    reds = [(r, c) for r in range(rows) for c in range(rows) if I[r][c] == 2 and r + c == diag]
    if not reds:
        return I

    # Sort by row
    reds.sort()
    s = reds[0][0]
    l = len(reds)

    # Set old reds to 7
    for r, c in reds:
        I[r][c] = 7

    # Compute new start and length
    new_l = l + 1
    new_s = s - new_l

    # Place new reds
    for i in range(new_l):
        r = new_s + i
        c = diag - r
        if 0 <= r < rows and 0 <= c < rows:
            I[r][c] = 2

    return I

from collections import deque

def solve_164(I):
    if not I or not I[0]:
        return I
    
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    visited = set()
    
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and (r, c) not in visited:
                color = I[r][c]
                q = deque([(r, c)])
                visited.add((r, c))
                min_r, max_r = r, r
                min_c, max_c = c, c
                
                while q:
                    cr, cc = q.popleft()
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)
                    
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                if h >= 3 and w >= 3:
                    for ir in range(min_r + 1, max_r):
                        for ic in range(min_c + 1, max_c):
                            output[ir][ic] = 8
    
    return output

import numpy as np

def solve_165(I):
    I = np.array(grid_lst)
    rows, cols = I.shape

    # Find purple position
    purple_positions = list(zip(*np.where(I == 8)))
    pr, pc = purple_positions[0]

    # Directions for 4-connectivity
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Find adjacent colors
    adj_colors = set()
    for dr, dc in directions:
        nr, nc = pr + dr, pc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            col = I[nr, nc]
            if col != 0 and col != 8:
                adj_colors.add(col)
    color = next(iter(adj_colors))

    # Flood fill to find component
    visited = set()
    stack = [(pr, pc)]
    component = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        if I[r, c] != color and I[r, c] != 8:
            continue
        component.add((r, c))
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                stack.append((nr, nc))

    # Find bounds
    min_r = min(r for r, c in component)
    max_r = max(r for r, c in component)
    min_c = min(c for r, c in component)
    max_c = max(c for r, c in component)

    # Create output 3x3
    output = [[0 for _ in range(3)] for _ in range(3)]
    for r, c in component:
        rel_r = r - min_r
        rel_c = c - min_c
        output[rel_r][rel_c] = color

    return output

import numpy as np

def solve_166(I):
    I = np.array(I)
    rows, cols = I.shape

    # Find red positions
    red_pos = np.argwhere(I == 2)
    if len(red_pos) == 0:
        return I.tolist()
    min_r = np.min(red_pos[:,0])
    max_r = np.max(red_pos[:,0])
    min_c = np.min(red_pos[:,1])
    max_c = np.max(red_pos[:,1])

    center_r = min_r + 1
    center_c = min_c + 1

    # Find green positions
    green_pos = np.argwhere(I == 3)
    green_rs = green_pos[:,0]
    green_cs = green_pos[:,1]

    if np.all(green_cs == green_cs[0]):
        is_vertical = True
        common_col = green_cs[0]
        sorted_pos = sorted(green_rs)
        current_val = center_r
        new_center_c = common_col
    elif np.all(green_rs == green_rs[0]):
        is_vertical = False
        common_row = green_rs[0]
        sorted_pos = sorted(green_cs)
        current_val = center_c
        new_center_r = common_row
    else:
        raise ValueError("Greens not aligned")

    current_idx = sorted_pos.index(current_val)
    next_idx = current_idx + 1
    if next_idx >= len(sorted_pos):
        return I.tolist()

    new_val = sorted_pos[next_idx]
    if is_vertical:
        new_center_r = new_val
    else:
        new_center_c = new_val

    # Remove all 2's
    I[I == 2] = 0

    # Place new reds around new center
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr = new_center_r + dr
            nc = new_center_c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                I[nr, nc] = 2

    return I.tolist()

from collections import deque

def solve_167(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def fill_rect(minr, maxr, minc, maxc, color):
        h = maxr - minr + 1
        w = maxc - minc + 1
        if h < 3 or w < 3:
            return
        for r in range(minr + 1, maxr):
            for c in range(minc + 1, maxc):
                I[r][c] = color
        next_color = 5 - color
        fill_rect(minr + 1, maxr - 1, minc + 1, maxc - 1, next_color)

    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 1 and not visited[i][j]:
                minr = maxr = i
                minc = maxc = j
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    r, c = q.popleft()
                    minr = min(minr, r)
                    maxr = max(maxr, r)
                    minc = min(minc, c)
                    maxc = max(maxc, c)
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 1 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                fill_rect(minr, maxr, minc, maxc, 2)
    return I

def solve_168(I):
    if not I:
        return I
    rows = len(I)
    cols = len(I[0])
    # Find gray column g and height h
    g = -1
    for c in range(cols):
        if I[0][c] == 5:
            g = c
            break
    if g == -1:
        return I
    h = 0
    for r in range(rows):
        if I[r][g] == 5:
            h += 1
        else:
            break
    w_left_max = g
    w_right_max = g - 1
    # Left groups
    left_groups = []
    if w_left_max > 0:
        rows_left_max = h + 2
        left_groups.append((w_left_max, rows_left_max))
        for ww in range(w_left_max - 1, 0, -1):
            left_groups.append((ww, 2))
        # Intended s
        intended = sum(nr for _, nr in left_groups)
        if intended > rows:
            excess = intended - rows
            last_idx = len(left_groups) - 1
            old_nr = left_groups[last_idx][1]
            new_nr = max(0, old_nr - excess)
            left_groups[last_idx] = (left_groups[last_idx][0], new_nr)
    # Right groups
    right_groups = []
    if w_right_max > 0:
        rows_right_max = (h - 2) - 2 * (w_right_max - 1)
        if rows_right_max > 0:
            right_groups.append((w_right_max, rows_right_max))
            for ww in range(w_right_max - 1, 0, -1):
                right_groups.append((ww, 2))
    # Output
    output = [row[:] for row in I]
    # Place left
    r = 0
    for w, nr in left_groups:
        for _ in range(nr):
            if r >= rows:
                break
            for c in range(w):
                output[r][c] = 8
            r += 1
    # Place right
    r = 0
    for w, nr in right_groups:
        for _ in range(nr):
            if r >= rows:
                break
            for dc in range(1, w + 1):
                c = g + dc
                if c < cols:
                    output[r][c] = 6
            r += 1
    return output

import numpy as np

def solve_169(I):
    I = np.array(I)
    rows, cols = I.shape

    def get_components(c):
        visited = np.zeros((rows, cols), bool)
        components = []
        for i in range(rows):
            for j in range(cols):
                if I[i, j] == c and not visited[i, j]:
                    component = []
                    queue = [(i, j)]
                    visited[i, j] = True
                    while queue:
                        x, y = queue.pop(0)
                        component.append((x, y))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < rows and 0 <= ny < cols and I[nx, ny] == c and not visited[nx, ny]:
                                visited[nx, ny] = True
                                queue.append((nx, ny))
                    components.append(component)
        return components

    all_colors = set(np.unique(I))
    for c in all_colors:
        if c == 8:
            continue
        components = get_components(c)
        for comp in components:
            row_cols = {}
            for x, y in comp:
                if x not in row_cols:
                    row_cols[x] = []
                row_cols[x].append(y)
            max_run = 0
            for r in row_cols:
                clist = sorted(set(row_cols[r]))
                if not clist:
                    continue
                current_len = 1
                max_in_row = 1
                for k in range(1, len(clist)):
                    if clist[k] == clist[k - 1] + 1:
                        current_len += 1
                        max_in_row = max(max_in_row, current_len)
                    else:
                        current_len = 1
                max_run = max(max_run, max_in_row)
            if max_run < 3:
                for x, y in comp:
                    I[x, y] = 8
            else:
                for r in row_cols:
                    clist = sorted(set(row_cols[r]))
                    if not clist:
                        continue
                    runs = []
                    start = clist[0]
                    end = clist[0]
                    for k in range(1, len(clist)):
                        if clist[k] == end + 1:
                            end = clist[k]
                        else:
                            runs.append((start, end))
                            start = clist[k]
                            end = clist[k]
                    runs.append((start, end))
                    for st, en in runs:
                        length = en - st + 1
                        if length >= 3:
                            I[r, st] = 8
                            I[r, st + 1] = 8
    return I.tolist()

def solve_170(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    bg = 7
    color_positions = {}
    for r in range(rows):
        for c in range(cols):
            val = I[r][c]
            if val != bg:
                if val not in color_positions:
                    color_positions[val] = []
                color_positions[val].append((r, c))
    to_fill = []
    for colr, pos in color_positions.items():
        if not pos:
            continue
        rs = [p[0] for p in pos]
        cs = [p[1] for p in pos]
        min_r = min(rs)
        max_r = max(rs)
        min_c = min(cs)
        max_c = max(cs)
        area = (max_r - min_r + 1) * (max_c - min_c + 1)
        count = len(pos)
        if count < area:
            to_fill.append((colr, min_r, max_r, min_c, max_c))
    to_fill.sort(key=lambda x: -x[0])
    output = [row[:] for row in I]
    for colr, min_r, max_r, min_c, max_c in to_fill:
        for r in range(min_r, max_r + 1):
            for cc in range(min_c, max_c + 1):
                output[r][cc] = colr
    return output

import numpy as np

def solve_171(I):
    I = np.array(I)
    rows, cols = I.shape
    visited = np.zeros((rows, cols), dtype=bool)
    
    def dfs(r, c, component):
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols or visited[cr, cc] or I[cr, cc] == 0:
                continue
            visited[cr, cc] = True
            component.append((cr, cc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((cr + dr, cc + dc))
    
    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r, c] != 0 and not visited[r, c]:
                component = []
                dfs(r, c, component)
                components.append(component)
    
    for component in components:
        non_four_colors = set(I[r, c] for r, c in component if I[r, c] != 4)
        if len(non_four_colors) != 1:
            continue  # Assuming always one color, as per examples
        color = next(iter(non_four_colors))
        k = sum(1 for r, c in component if I[r, c] == color)
        
        rs, cs = zip(*component)
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        
        exp_min_r = max(0, min_r - k)
        exp_max_r = min(rows - 1, max_r + k)
        exp_min_c = max(0, min_c - k)
        exp_max_c = min(cols - 1, max_c + k)
        
        for er in range(exp_min_r, exp_max_r + 1):
            for ec in range(exp_min_c, exp_max_c + 1):
                if I[er, ec] == 0:
                    I[er, ec] = color
    
    return I.tolist()

import numpy as np
from collections import deque

def solve_172(I):
    I = np.array(I)
    rows, cols = I.shape

    # Find mirror
    mirror_type = None
    mirror_pos = None
    mirror_color = None
    for c in range(cols):
        col_values = I[:, c]
        if np.all(col_values == col_values[0]) and col_values[0] != 0:
            mirror_type = 'vertical'
            mirror_pos = c
            mirror_color = col_values[0]
            break
    if mirror_type is None:
        for r in range(rows):
            row_values = I[r, :]
            if np.all(row_values == row_values[0]) and row_values[0] != 0:
                mirror_type = 'horizontal'
                mirror_pos = r
                mirror_color = row_values[0]
                break

    # Find background: most frequent color
    counts = np.bincount(I.flatten())
    background = np.argmax(counts)
    if background == mirror_color:
        counts[mirror_color] = 0
        background = np.argmax(counts)

    # Find connected components
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(rows):
        for j in range(cols):
            if not visited[i, j] and I[i, j] != background and I[i, j] != mirror_color and I[i, j] != 0:
                comp_color = I[i, j]
                component = []
                queue = deque([(i, j)])
                visited[i, j] = True
                while queue:
                    x, y = queue.popleft()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and I[nx, ny] == comp_color:
                            visited[nx, ny] = True
                            queue.append((nx, ny))
                if len(component) >= 2:
                    components.append((component, comp_color))

    output = I.copy()

    # Reflect components
    for comp, color in components:
        if mirror_type == 'vertical':
            m = mirror_pos
            for r, c in comp:
                new_c = 2 * m - c
                if 0 <= new_c < cols:
                    output[r, new_c] = color
        else:  # horizontal
            m = mirror_pos
            for r, c in comp:
                new_r = 2 * m - r
                if 0 <= new_r < rows:
                    output[new_r, c] = color

    # Fill the gaps
    for comp, _ in components:
        if mirror_type == 'vertical':
            row_to_cols = {}
            for r, c in comp:
                if r not in row_to_cols:
                    row_to_cols[r] = set()
                row_to_cols[r].add(c)
            for r, o_cols in row_to_cols.items():
                ref_cols = {2 * mirror_pos - cc for cc in o_cols}
                left_cols = {cc for cc in o_cols.union(ref_cols) if cc < mirror_pos}
                right_cols = {cc for cc in o_cols.union(ref_cols) if cc > mirror_pos}
                if left_cols and right_cols:
                    left_max = max(left_cols)
                    right_min = min(right_cols)
                    for cc in range(left_max + 1, right_min):
                        output[r, cc] = mirror_color
        else:  # horizontal
            col_to_rows = {}
            for r, c in comp:
                if c not in col_to_rows:
                    col_to_rows[c] = set()
                col_to_rows[c].add(r)
            for c, o_rows in col_to_rows.items():
                ref_rows = {2 * mirror_pos - rr for rr in o_rows}
                upper_rows = {rr for rr in o_rows.union(ref_rows) if rr < mirror_pos}
                lower_rows = {rr for rr in o_rows.union(ref_rows) if rr > mirror_pos}
                if upper_rows and lower_rows:
                    upper_max = max(upper_rows)
                    lower_min = min(lower_rows)
                    for rr in range(upper_max + 1, lower_min):
                        output[rr, c] = mirror_color

    return output.tolist()

def solve_173(I):
    n = len(I)
    c = [I[k][0] for k in range(n)]
    
    m = n - 1
    while m >= 0 and c[m] == 0:
        m -= 1
    
    if m < 0:
        seq = []
    else:
        seq = c[:m + 1]
    
    len_seq = len(seq)
    if len_seq == 0:
        return [[0] * (2 * n) for _ in range(2 * n)]
    
    out_size = 2 * n
    num_layers = out_size
    full = num_layers // len_seq
    rem = num_layers % len_seq
    out_seq = seq * full + seq[:rem]
    
    out_grid = [[0] * out_size for _ in range(out_size)]
    for i in range(out_size):
        for j in range(out_size):
            out_grid[i][j] = out_seq[max(i, j)]
    
    return out_grid

def solve_174(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 6 and not visited[i][j]:
                stack = [(i, j)]
                visited[i][j] = True
                min_r, max_r = i, i
                min_c, max_c = j, j
                while stack:
                    r, c = stack.pop()
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 6 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                # Fill internal non-pink with 4
                for rr in range(min_r, max_r + 1):
                    for cc in range(min_c, max_c + 1):
                        if I[rr][cc] != 6:
                            output[rr][cc] = 4
                # Add top border
                if min_r > 0:
                    left = max(0, min_c - 1)
                    right = min(cols, max_c + 2)
                    for cc in range(left, right):
                        if output[min_r - 1][cc] == 8:
                            output[min_r - 1][cc] = 3
                # Add bottom border
                if max_r < rows - 1:
                    left = max(0, min_c - 1)
                    right = min(cols, max_c + 2)
                    for cc in range(left, right):
                        if output[max_r + 1][cc] == 8:
                            output[max_r + 1][cc] = 3
                # Add left border
                if min_c > 0:
                    for rr in range(min_r, max_r + 1):
                        if output[rr][min_c - 1] == 8:
                            output[rr][min_c - 1] = 3
                # Add right border
                if max_c < cols - 1:
                    for rr in range(min_r, max_r + 1):
                        if output[rr][max_c + 1] == 8:
                            output[rr][max_c + 1] = 3
    return output

def solve_175(I):
    if not I or not I[0]:
        return I
    height = len(I)
    width = len(I[0])
    output = [row[:] for row in I]
    hole_columns = set()
    for r in range(height):
        for c in range(width):
            if I[r][c] == 0:
                if r > 0 and I[r-1][c] > 0:
                    up_color = I[r-1][c]
                    if c > 0 and I[r][c-1] == up_color:
                        if c < width - 1 and I[r][c+1] == up_color:
                            hole_columns.add(c)
    bottom_r = height - 1
    for c in hole_columns:
        output[bottom_r][c] = 4
    return output

def solve_176(I):
    if not I:
        return []
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    # Find layers
    layers = []
    start = 0
    for r in range(rows):
        if all(x == 5 for x in I[r]):
            if start < r:
                layers.append((start, r))
            start = r + 1
    if start < rows:
        layers.append((start, rows))
    # Sections
    sections = [(0, 3, 1), (4, 7, 5), (8, 11, 9)]
    # Process each layer
    for layer_start, layer_end in layers:
        h = layer_end - layer_start
        if h == 0:
            continue
        signal_row = layer_start + h // 2
        for sec_start, sec_end, sig_col in sections:
            if sig_col >= cols:
                continue
            c = I[signal_row][sig_col]
            if c == 0:
                continue
            new_c = c + 5
            for rr in range(layer_start, layer_end):
                for cc in range(sec_start, sec_end):
                    output[rr][cc] = new_c
    return output

import numpy as np

def solve_177(I):
    I = np.array(I)
    rows, cols = I.shape
    output = I.copy()
    visited = np.zeros((rows, cols), bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(r, c, component):
        stack = [(r, c)]
        visited[r, c] = True
        component.append((r, c))
        while stack:
            cr, cc = stack.pop()
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and I[nr, nc] == 1:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
                    component.append((nr, nc))

    components = []
    for i in range(rows):
        for j in range(cols):
            if I[i, j] == 1 and not visited[i, j]:
                comp = []
                dfs(i, j, comp)
                components.append(comp)

    for comp in components:
        if not comp:
            continue
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        min_r = min(rs)
        max_r = max(rs)
        min_c = min(cs)
        max_c = max(cs)
        # Fill all 0 in the box to 8
        for ii in range(min_r, max_r + 1):
            for jj in range(min_c, max_c + 1):
                if I[ii, jj] == 0:
                    output[ii, jj] = 8
        # Now find gaps and extend
        for ii in range(min_r, max_r + 1):
            for jj in range(min_c, max_c + 1):
                if I[ii, jj] == 0 and (ii == min_r or ii == max_r or jj == min_c or jj == max_c):
                    # Extend based on sides
                    if jj == min_c:  # left
                        for cc in range(0, min_c):
                            output[ii, cc] = 8
                    if jj == max_c:  # right
                        for cc in range(max_c + 1, cols):
                            output[ii, cc] = 8
                    if ii == min_r:  # top
                        for rr in range(0, min_r):
                            output[rr, jj] = 8
                    if ii == max_r:  # bottom
                        for rr in range(max_r + 1, rows):
                            output[rr, jj] = 8

    return output.tolist()

import numpy as np

def solve_178(I):
    I = np.array(I)
    output = np.zeros_like(I)
    colors = np.unique(I[I > 0])
    for color in colors:
        positions = np.argwhere(I == color)
        if len(positions) == 0:
            continue
        min_r = positions[:, 0].min()
        max_r = positions[:, 0].max()
        height = max_r - min_r + 1
        shift = height
        for r, c in positions:
            new_r = r - shift
            if 0 <= new_r < I.shape[0]:
                output[new_r, c] = color
    return output.tolist()

from collections import Counter

def solve_179(I):
    if not I or not I[0]:
        return []
    
    height = len(I)
    width = len(I[0])
    
    # Collect all non-zero cells
    cells = [cell for row in I for cell in row if cell != 0]
    
    if not cells:
        return [[0] * width for _ in range(height)]  # If no non-zero, fill with 0, but not needed here
    
    # Find the most frequent non-zero color
    count = Counter(cells)
    most_common_color = count.most_common(1)[0][0]
    
    # Create output I filled with that color
    output = [[most_common_color] * width for _ in range(height)]
    
    return output

from collections import Counter

def solve_180(I):
    flat = [cell for row in I for cell in row]
    counts = Counter(flat)
    if 7 in counts:
        del counts[7]
    color_sizes = [(color, counts[color]) for color in counts]
    color_sizes.sort(key=lambda x: -x[1])  # descending size
    k = len(color_sizes)
    if k == 0:
        return []
    n = 2 * k - 1
    out = [[0] * n for _ in range(n)]
    current_size = n
    for color, _ in color_sizes:
        offset = (n - current_size) // 2
        for i in range(offset, offset + current_size):
            for j in range(offset, offset + current_size):
                out[i][j] = color
        current_size -= 2
    return out

import numpy as np

def solve_181(I):
    I = np.array(I)
    out = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            sub = I[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)]
            if np.all(sub != 0):
                colors = set(sub.flatten()) - {5}
                if len(colors) == 1:
                    out[i, j] = list(colors)[0]
    return out.tolist()

def solve_182(I):
    if not I or not I[0]:
        return []
    h = len(I)
    w = len(I[0])
    vertical = h > w
    red = 2
    green = 3
    purple = 8

    # Find bar position
    bar_pos = None
    if vertical:
        for r in range(h):
            if all(I[r][c] == red for c in range(w)):
                bar_pos = r
                break
    else:
        for c in range(w):
            if all(I[r][c] == red for r in range(h)):
                bar_pos = c
                break
    if bar_pos is None:
        return [row[:] for row in I]

    # Find green positions
    greens = [(r, c) for r in range(h) for c in range(w) if I[r][c] == green]
    if not greens:
        return [row[:] for row in I]

    # Determine shift
    shift = 0
    if vertical:
        rows = [pos[0] for pos in greens]
        min_r = min(rows)
        max_r = max(rows)
        if max_r < bar_pos:
            # move down
            shift = bar_pos - max_r - 1
        elif min_r > bar_pos:
            # move up
            shift = -(min_r - bar_pos - 1)
    else:
        cols = [pos[1] for pos in greens]
        min_c = min(cols)
        max_c = max(cols)
        if max_c < bar_pos:
            # move right
            shift = bar_pos - max_c - 1
        elif min_c > bar_pos:
            # move left
            shift = -(min_c - bar_pos - 1)

    # Copy I
    output = [row[:] for row in I]

    # Clear original
    for r, c in greens:
        output[r][c] = 0

    # Place moved
    if vertical:
        for r, c in greens:
            new_r = r + shift
            output[new_r][c] = green
    else:
        for r, c in greens:
            new_c = c + shift
            output[r][new_c] = green

    # Add purple bar
    if vertical:
        if shift > 0:
            purple_r = min_r + shift - 1
            for c in range(w):
                output[purple_r][c] = purple
        elif shift < 0:
            purple_r = max_r + shift + 1
            for c in range(w):
                output[purple_r][c] = purple
    else:
        if shift > 0:
            purple_c = min_c + shift - 1
            for r in range(h):
                output[r][purple_c] = purple
        elif shift < 0:
            purple_c = max_c + shift + 1
            for r in range(h):
                output[r][purple_c] = purple

    return output

from collections import defaultdict

def solve_183(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Collect positions for each color
    color_positions = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            color = I[r][c]
            if color != 0:
                color_positions[color].append((r, c))
    
    # Copy the I
    output = [row[:] for row in I]
    
    # For each color with exactly two positions, fill the rectangle
    for color, positions in color_positions.items():
        if len(positions) == 2:
            (r1, c1), (r2, c2) = positions
            min_r = min(r1, r2)
            max_r = max(r1, r2)
            min_c = min(c1, c2)
            max_c = max(c1, c2)
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    output[r][c] = color
    
    return output

import numpy as np

def solve_184(I):
    I = np.array(grid_lst)
    height, width = I.shape
    center_r = (height - 1) / 2.0
    center_c = (width - 1) / 2.0

    colors = np.unique(I[I != 0])

    layers = []
    for color in colors:
        positions = np.argwhere(I == color)
        min_r = positions[:, 0].min()
        max_r = positions[:, 0].max()
        min_c = positions[:, 1].min()
        max_c = positions[:, 1].max()

        h = max_r - min_r + 1
        w = max_c - min_c + 1
        if h != w:
            diff = abs(h - w)
            if h < w:
                old_center = (min_r + max_r) / 2.0
                add_min_center = old_center - diff / 2.0
                add_max_center = old_center + diff / 2.0
                dist_min = abs(add_min_center - center_r)
                dist_max = abs(add_max_center - center_r)
                if dist_min < dist_max:
                    min_r -= diff
                else:
                    max_r += diff
            else:
                old_center = (min_c + max_c) / 2.0
                add_min_center = old_center - diff / 2.0
                add_max_center = old_center + diff / 2.0
                dist_min = abs(add_min_center - center_c)
                dist_max = abs(add_max_center - center_c)
                if dist_min < dist_max:
                    min_c -= diff
                else:
                    max_c += diff

        side = max_r - min_r + 1
        layers.append({'color': color, 'min_r': min_r, 'max_r': max_r, 'min_c': min_c, 'max_c': max_c, 'side': side})

    # Sort by side ascending
    layers.sort(key=lambda x: x['side'])

    output = np.copy(I)

    for i, layer in enumerate(layers):
        if i == 0:
            # Fill entire box
            for r in range(layer['min_r'], layer['max_r'] + 1):
                for c in range(layer['min_c'], layer['max_c'] + 1):
                    if 0 <= r < height and 0 <= c < width and output[r, c] == 0:
                        output[r, c] = layer['color']
        else:
            prev = layers[i - 1]
            for r in range(layer['min_r'], layer['max_r'] + 1):
                for c in range(layer['min_c'], layer['max_c'] + 1):
                    if not (prev['min_r'] <= r <= prev['max_r'] and prev['min_c'] <= c <= prev['max_c']):
                        if 0 <= r < height and 0 <= c < width and output[r, c] == 0:
                            output[r, c] = layer['color']

    return output.tolist()

def solve_185(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Find unique columns with at least one 0
    col_set = set()
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 0:
                col_set.add(c)
    
    # Sort the columns
    sorted_cols = sorted(col_set)
    
    # Assign colors: column to color (1 + index)
    col_to_color = {col: idx + 1 for idx, col in enumerate(sorted_cols)}
    
    # Create output I
    output = [row[:] for row in I]
    
    # Replace 0s with the assigned color for their column
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 0:
                output[r][c] = col_to_color[c]
    
    return output

import numpy as np

def solve_186(I):
    g = np.array(I)
    nonzeros = np.argwhere(g != 0)
    c = len(nonzeros)
    n = int(np.sqrt(c))
    m = g.shape[0]
    k = m // n
    out = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            block = g[i * k:(i + 1) * k, j * k:(j + 1) * k]
            colors = block[block != 0]
            out[i, j] = colors[0]
    return out.tolist()

def solve_187(I):
    if not I or not I[0]:
        return []
    
    rows = len(I)
    cols = len(I[0])
    background = 8
    
    # Find all non-background cells
    component = [(r, c) for r in range(rows) for c in range(cols) if I[r][c] != background]
    
    if not component:
        return [row[:] for row in I]
    
    # Get neighbors function
    def get_neighbors(r, c):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        res = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] != background:
                res.append((nr, nc))
        return res
    
    # Compute degrees
    degrees = {p: len(get_neighbors(*p)) for p in component}
    
    # Find ends (degree 1)
    ends = [p for p in degrees if degrees[p] == 1]
    
    # Assume exactly two ends
    if len(ends) != 2:
        # If not a chain, return original (or handle differently, but per examples it's a chain)
        return [row[:] for row in I]
    
    # Choose start: min by row, then col
    start = min(ends, key=lambda p: (p[0], p[1]))
    
    # Traverse the path
    path = []
    current = start
    visited = set()
    while True:
        path.append(current)
        visited.add(current)
        neigh = [n for n in get_neighbors(*current) if n not in visited]
        if not neigh:
            break
        # Assume no branches
        if len(neigh) != 1:
            # If branch, return original (but per examples no)
            return [row[:] for row in I]
        current = neigh[0]
    
    # Check all visited
    if len(path) != len(component):
        return [row[:] for row in I]
    
    # Collect colors
    colors = [I[r][c] for r, c in path]
    
    # Reverse colors
    reversed_colors = colors[::-1]
    
    # Create output
    output = [row[:] for row in I]
    for i, (r, c) in enumerate(path):
        output[r][c] = reversed_colors[i]
    
    return output

import copy

def solve_188(I):
    out = copy.deepcopy(I)
    rows = len(I)
    cols = len(I[0])
    
    # Find palette position
    palette_r = -1
    palette_c = -1
    for r in range(rows - 1):
        for c in range(cols - 1):
            if all(I[r + i][c + j] > 0 and I[r + i][c + j] != 8 for i in [0, 1] for j in [0, 1]):
                palette_r = r
                palette_c = c
                break
        if palette_r != -1:
            break
    
    if palette_r == -1:
        return out
    
    # Get palette colors
    tl = I[palette_r][palette_c]
    tr = I[palette_r][palette_c + 1]
    bl = I[palette_r + 1][palette_c]
    br = I[palette_r + 1][palette_c + 1]
    
    # Determine canvas bounds
    frame_rows_start = palette_r
    frame_cols_start = palette_c
    
    if frame_rows_start == 0:
        canvas_row_start = 2
        canvas_row_end = rows - 1
    else:
        canvas_row_start = 0
        canvas_row_end = rows - 3
    
    if frame_cols_start == 0:
        canvas_col_start = 2
        canvas_col_end = cols - 1
    else:
        canvas_col_start = 0
        canvas_col_end = cols - 3
    
    # Canvas dimensions
    height = canvas_row_end - canvas_row_start + 1
    width = canvas_col_end - canvas_col_start + 1
    half_h = height // 2
    half_w = width // 2
    
    # Recolor non-zero cells in canvas
    for r in range(canvas_row_start, canvas_row_end + 1):
        for c in range(canvas_col_start, canvas_col_end + 1):
            if I[r][c] != 0:
                local_r = r - canvas_row_start
                local_c = c - canvas_col_start
                if local_r < half_h:
                    if local_c < half_w:
                        new_color = tl
                    else:
                        new_color = tr
                else:
                    if local_c < half_w:
                        new_color = bl
                    else:
                        new_color = br
                out[r][c] = new_color
    
    return out

def solve_189(I):
    rows = len(I)
    if rows == 0:
        return []
    cols = len(I[0])
    
    # Create horizontal flip
    flip = [row[::-1] for row in I]
    
    # Parts to concatenate: flip, original, flip, original
    parts = [flip, I, flip, I]
    
    # Build output
    out = []
    for r in range(rows):
        out_row = []
        for part in parts:
            out_row.extend(part[r])
        out.append(out_row)
    
    return out

import copy

def solve_190(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    palettes = []
    seeds = []
    for r in range(rows):
        c = 0
        while c < cols:
            if I[r][c] == 0:
                c += 1
                continue
            start = c
            seq = []
            while c < cols and I[r][c] != 0:
                seq.append(I[r][c])
                c += 1
            if len(seq) == 1:
                seeds.append((r, start, seq[0]))
            elif len(seq) >= 2:
                palettes.append(seq)
    output = copy.deepcopy(I)
    for r, c, color in seeds:
        for pal in palettes:
            if color in pal:
                i = pal.index(color)
                start_c = c - i
                for j in range(len(pal)):
                    place_c = start_c + j
                    if 0 <= place_c < cols:
                        output[r][place_c] = pal[j]
                break
    return output

from collections import defaultdict

def solve_191(I):
    h = len(I)
    w = len(I[0])
    groups = defaultdict(list)  # r -> list of c
    for r in range(h - 2):
        for c in range(w - 2):
            if (I[r][c] == 1 and I[r][c + 1] == 1 and I[r][c + 2] == 1 and
                I[r + 1][c] == 1 and I[r + 1][c + 1] == 0 and I[r + 1][c + 2] == 1 and
                I[r + 2][c] == 1 and I[r + 2][c + 1] == 1 and I[r + 2][c + 2] == 1):
                groups[r].append(c)
    if not groups:
        return I
    max_len = max(len(cs) for cs in groups.values())
    ref_r = next(r for r in groups if len(groups[r]) == max_len)
    ref_cs = set(groups[ref_r])
    output = [row[:] for row in I]
    for r in groups:
        if len(groups[r]) >= max_len:
            continue
        current_cs = set(groups[r])
        for c in ref_cs - current_cs:
            output[r][c] = 8
            output[r][c + 1] = 8
            output[r][c + 2] = 8
            output[r + 1][c] = 8
            # center remains 0
            output[r + 1][c + 2] = 8
            output[r + 2][c] = 8
            output[r + 2][c + 1] = 8
            output[r + 2][c + 2] = 8
    return output

import numpy as np

def solve_192(I):
    small = np.array(I)
    n = small.shape[0]  # Assuming square I, n=4
    large_size = n * n  # 16
    large = np.zeros((large_size, large_size), dtype=int)
    for i in range(n):
        for j in range(n):
            if small[i, j] != 0:
                large[i*n : (i+1)*n, j*n : (j+1)*n] = small
    return large.tolist()

def solve_193(I):
    return [row[-3:] for row in I[:3]]

def solve_194(I):
    if not I:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]

    # Find blue top-left
    min_r_blue = rows
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 1:
                min_r_blue = min(min_r_blue, r)
    if min_r_blue < rows:
        min_c_blue = cols
        for c in range(cols):
            if I[min_r_blue][c] == 1:
                min_c_blue = min(min_c_blue, c)
        # Extend up-left
        r = min_r_blue - 1
        c = min_c_blue - 1
        while r >= 0 and c >= 0:
            output[r][c] = 1
            r -= 1
            c -= 1

    # Find red bottom-right
    max_r_red = -1
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 2:
                max_r_red = max(max_r_red, r)
    if max_r_red >= 0:
        max_c_red = -1
        for c in range(cols):
            if I[max_r_red][c] == 2:
                max_c_red = max(max_c_red, c)
        # Extend down-right
        r = max_r_red + 1
        c = max_c_red + 1
        while r < rows and c < cols:
            output[r][c] = 2
            r += 1
            c += 1

    return output

def solve_195(I):
    if not I or not I[0]:
        return []
    height = len(I)
    width = len(I[0])
    output = [[0 for _ in range(4)] for _ in range(height)]
    for r in range(height):
        for c in range(4):
            if I[r][c] == 4 or (c + 5 < width and I[r][c + 5] == 4):
                output[r][c] = 8
    return output

import math

def solve_196(I):
    if not I or not I[0]:
        return []
    cols = len(I[0])
    # Find C and L from top row
    top = I[0]
    C = 0
    L = 0
    for i in range(cols):
        if top[i] != 0:
            if C == 0:
                C = top[i]
            if top[i] == C:
                L += 1
            else:
                break
    if L == 0:
        return [[0] * cols]
    # Generate groups
    groups = list(range(1, L + 1)) + list(range(1, L))[::-1]
    # Generate sequence
    sequence = []
    for k in groups:
        sequence.extend([C] * k)
        sequence.append(0)
    # Calculate output height
    seq_len = len(sequence)
    height = math.ceil(seq_len / cols)
    # Build output
    output = []
    for r in range(height):
        start = r * cols
        end = min(start + cols, seq_len)
        row = sequence[start:end] + [0] * (cols - (end - start))
        output.append(row)
    return output

import copy
from collections import defaultdict

def solve_197(I):
    if not I or not I[0]:
        return []

    rows = len(I)
    cols = len(I[0])

    # Find centers
    centers = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 0:
                has_up = (r > 0 and I[r-1][c] == 2)
                has_down = (r < rows-1 and I[r+1][c] == 2)
                has_left = (c > 0 and I[r][c-1] == 2)
                has_right = (c < cols-1 and I[r][c+1] == 2)
                if has_up and has_down and has_left and has_right:
                    centers.append((r, c))

    # Group for horizontal: row to list of c
    hor_groups = defaultdict(list)
    for r, c in centers:
        hor_groups[r].append(c)

    # Group for vertical: col to list of r
    ver_groups = defaultdict(list)
    for r, c in centers:
        ver_groups[c].append(r)

    # Copy I
    out = copy.deepcopy(I)

    # Horizontal fills
    for r, clist in hor_groups.items():
        if len(clist) >= 2:
            clist.sort()
            for i in range(len(clist) - 1):
                left_c = clist[i]
                right_c = clist[i + 1]
                start_c = left_c + 2
                end_c = right_c - 2
                for cc in range(start_c, end_c + 1):
                    if 0 <= cc < cols and out[r][cc] == 0:
                        out[r][cc] = 1

    # Vertical fills
    for c, rlist in ver_groups.items():
        if len(rlist) >= 2:
            rlist.sort()
            for i in range(len(rlist) - 1):
                upper_r = rlist[i]
                lower_r = rlist[i + 1]
                start_r = upper_r + 2
                end_r = lower_r - 2
                for rr in range(start_r, end_r + 1):
                    if 0 <= rr < rows and out[rr][c] == 0:
                        out[rr][c] = 1

    return out

import copy

def solve_198(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(r, c, component):
        stack = [(r, c)]
        visited[r][c] = True
        component.append((r, c))
        while stack:
            cr, cc = stack.pop()
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == 4:
                    visited[nr][nc] = True
                    stack.append((nr, nc))
                    component.append((nr, nc))

    output = copy.deepcopy(I)
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 4 and not visited[i][j]:
                component = []
                dfs(i, j, component)
                if not component:
                    continue
                min_r = min(p[0] for p in component)
                max_r = max(p[0] for p in component)
                min_c = min(p[1] for p in component)
                max_c = max(p[1] for p in component)
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                if h == w:
                    for rr in range(min_r, max_r + 1):
                        for cc in range(min_c, max_c + 1):
                            if output[rr][cc] == 0:
                                output[rr][cc] = 7
    return output

import numpy as np

def solve_199(I):
    g = np.array(I)
    rows, cols = g.shape
    out = np.zeros((2 * rows, 2 * cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if g[r, c] != 0:
                out[2 * r:2 * r + 2, 2 * c:2 * c + 2] = g[r, c]
    return out.tolist()

import copy

def solve_200(I):
    out = copy.deepcopy(I)
    rows = len(I)
    if rows == 0:
        return out
    cols = len(I[0])

    # Find horizontal_row
    horizontal_row = -1
    C = -1
    for r in range(rows):
        first = I[r][0]
        if all(I[r][c] == first for c in range(cols)) and first != 7 and first != 0:
            horizontal_row = r
            C = first
            break

    if horizontal_row == -1:
        return out

    # Find vertical_col
    vertical_col = -1
    for cc in range(cols):
        first = I[0][cc]
        if all(I[r][cc] == first for r in range(rows)) and first == C:
            vertical_col = cc
            break

    if vertical_col == -1:
        return out

    # Symmetrize vertical (left-right)
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 7 and I[r][c] != C:
                c_mirror = 2 * vertical_col - c
                if 0 <= c_mirror < cols:
                    out[r][c_mirror] = I[r][c]

    # Now, symmetrize horizontal (up-down) on the updated out
    grid2 = copy.deepcopy(out)
    for r in range(rows):
        for c in range(cols):
            if out[r][c] != 7 and out[r][c] != C:
                r_mirror = 2 * horizontal_row - r
                if 0 <= r_mirror < rows:
                    grid2[r_mirror][c] = out[r][c]

    return grid2

def solve_201(I):
    # Rotate 180 degrees by reversing rows and then reversing each row
    return [row[::-1] for row in I[::-1]]

def solve_202(I):
    rows = len(I)
    cols = len(I[0]) if rows > 0 else 0
    # Find the non-zero cell
    r, c, k = -1, -1, -1
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != 0:
                if r != -1:
                    raise ValueError("Multiple non-zero cells")
                r, c, k = i, j, I[i][j]
    if r == -1:
        raise ValueError("No non-zero cell")
    # Create output I
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    # Set center
    output[r][c] = k
    # Four directions
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in directions:
        # Compute max steps
        row_steps = r if dr == -1 else (rows - 1 - r) if dr == 1 else 0
        col_steps = c if dc == -1 else (cols - 1 - c) if dc == 1 else 0
        steps = min(row_steps, col_steps)
        for i in range(1, steps + 1):
            nr = r + i * dr
            nc = c + i * dc
            output[nr][nc] = k
    return output

def solve_203(I):
    if not I:
        return []
    h = len(I)
    w = len(I[0])
    # Find k: first column where all cells are 0
    k = w
    for j in range(w):
        if all(I[i][j] == 0 for i in range(h)):
            k = j
            break
    # Build output
    output = []
    for i in range(h):
        seq = I[i][:k]
        if k == 0:
            output.append([0] * w)
            continue
        last = seq[-1]
        num_repeats = (w - k) - (k - 1)
        append_part = seq[1:]
        row = seq + [last] * num_repeats + append_part
        output.append(row)
    return output

import numpy as np

def solve_204(I):
    g = np.array(I)
    right = np.fliplr(g)
    top = np.hstack((g, right))
    bottom = np.flipud(top)
    full = np.vstack((top, bottom))
    return full.tolist()

def solve_205(I):
    if not I:
        return []
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    visited = [[False] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 8 and not visited[r][c]:
                # Find rectangle bounds
                min_r = r
                min_c = c
                max_c = c
                while max_c + 1 < cols and I[r][max_c + 1] == 8:
                    max_c += 1
                max_r = r
                while max_r + 1 < rows and all(I[max_r + 1][cc] == 8 for cc in range(min_c, max_c + 1)):
                    max_r += 1

                # Compute h, w
                h = max_r - min_r + 1
                w = max_c - min_c + 1

                # Band heights
                base = h // 4
                extra = h % 4
                band_heights = [base] * 4
                if extra > 0:
                    band_heights[1] += 1
                    extra -= 1
                if extra > 0:
                    band_heights[2] += 1

                # Thicknesses
                inner = 2
                outer = w // 2
                middle_w = w - 2 * inner

                # Color the bands
                curr_r = min_r
                for b in range(4):
                    bh = band_heights[b]
                    if bh == 0:
                        continue
                    is_outer = b == 0 or b == 3
                    if b == 0:
                        left_col = 6
                        right_col = 1
                    elif b == 1:
                        left_col = 6
                        mid_col = 4
                        right_col = 1
                    elif b == 2:
                        left_col = 2
                        mid_col = 4
                        right_col = 3
                    else:
                        left_col = 2
                        right_col = 3

                    for lr in range(bh):
                        gr = curr_r + lr
                        if is_outer:
                            # Left
                            for lc in range(outer):
                                output[gr][min_c + lc] = left_col
                            # Right
                            start_right = min_c + w - outer
                            for lc in range(outer):
                                output[gr][start_right + lc] = right_col
                        else:
                            # Left
                            for lc in range(inner):
                                output[gr][min_c + lc] = left_col
                            # Middle
                            for lc in range(middle_w):
                                output[gr][min_c + inner + lc] = mid_col
                            # Right
                            start_right = min_c + w - inner
                            for lc in range(inner):
                                output[gr][start_right + lc] = right_col
                    curr_r += bh

                # Mark visited
                for vr in range(min_r, max_r + 1):
                    for vc in range(min_c, max_c + 1):
                        visited[vr][vc] = True

    return output

def solve_206(I):
    I = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])

    # Check for horizontal bars (top and bottom)
    top_color = I[0][0] if all(x == I[0][0] != 0 for x in I[0]) else None
    bottom_color = I[rows - 1][0] if all(x == I[rows - 1][0] != 0 for x in I[rows - 1]) else None
    if top_color is not None and bottom_color is not None and top_color != bottom_color:
        bar_top = 0
        bar_bottom = rows - 1
        space_start = 1
        space_end = rows - 2
        for c in range(cols):
            blue_rows = [r for r in range(space_start, space_end + 1) if I[r][c] == 1]
            if not blue_rows:
                continue
            topmost = min(blue_rows)
            dist_top = topmost - bar_top
            dist_bottom = bar_bottom - topmost
            if dist_top < dist_bottom:
                add_r = topmost - 1
                if space_start <= add_r <= space_end:
                    I[add_r][c] = top_color
            bottommost = max(blue_rows)
            dist_bottom = bar_bottom - bottommost
            dist_top = bottommost - bar_top
            if dist_bottom < dist_top:
                add_r = bottommost + 1
                if space_start <= add_r <= space_end:
                    I[add_r][c] = bottom_color
        return I

    # Check for vertical bars (left and right)
    left_color = I[0][0] if all(I[r][0] == I[0][0] != 0 for r in range(rows)) else None
    right_color = I[0][cols - 1] if all(I[r][cols - 1] == I[0][cols - 1] != 0 for r in range(rows)) else None
    if left_color is not None and right_color is not None and left_color != right_color:
        bar_left = 0
        bar_right = cols - 1
        space_start = 1
        space_end = cols - 2
        for r in range(rows):
            blue_cols = [c for c in range(space_start, space_end + 1) if I[r][c] == 1]
            if not blue_cols:
                continue
            leftmost = min(blue_cols)
            dist_left = leftmost - bar_left
            dist_right = bar_right - leftmost
            if dist_left < dist_right:
                add_c = leftmost - 1
                if space_start <= add_c <= space_end:
                    I[r][add_c] = left_color
            rightmost = max(blue_cols)
            dist_right = bar_right - rightmost
            dist_left = rightmost - bar_left
            if dist_right < dist_left:
                add_c = rightmost + 1
                if space_start <= add_c <= space_end:
                    I[r][add_c] = right_color
        return I

    # If no bars detected, return unchanged
    return I

import collections

def solve_207(I):
    if not I:
        return []
    rows = len(I)
    cols = len(I[0])
    petal_colors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 4:
                colors = []
                valid = True
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        valid = False
                        break
                    colors.append(I[nr][nc])
                if valid and all(color == colors[0] for color in colors):
                    petal_colors.append(colors[0])
    if not petal_colors:
        return [[0]]  # Default, though not needed in examples
    counter = collections.Counter(petal_colors)
    most_common = counter.most_common(1)[0][0]
    return [[most_common]]

def solve_208(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def dfs(r, c, component):
        stack = [(r, c)]
        visited[r][c] = True
        component.append((r, c))
        while stack:
            cr, cc = stack.pop()
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == 8:
                    visited[nr][nc] = True
                    stack.append((nr, nc))
                    component.append((nr, nc))

    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 8 and not visited[i][j]:
                component = []
                dfs(i, j, component)
                if component:
                    min_c = min(cc for _, cc in component)
                    max_c = max(cc for _, cc in component)
                    w = max_c - min_c + 1
                    # Clear original positions
                    for cr, cc in component:
                        output[cr][cc] = 0
                    # Set new positions
                    for cr, cc in component:
                        new_c = cc + w
                        if new_c < cols:
                            output[cr][new_c] = 8

    return output

def solve_209(I):
    blocks = [I[i:i+3] for i in range(0, 9, 3)]

    def is_symmetric(block):
        for r in range(3):
            for c in range(r + 1, 3):
                if block[r][c] != block[c][r]:
                    return False
        return True

    for block in blocks:
        if not is_symmetric(block):
            return block
    return []  # Fallback, assuming always one non-symmetric block

import numpy as np

def solve_210(I):
    I = np.array(grid_lst)
    height, width = I.shape
    background = I[0, 0]
    
    all_colors = set(I.flatten())
    trail_colors = [c for c in all_colors if c != background]
    
    min_rows = []
    for color in trail_colors:
        positions = np.argwhere(I == color)
        min_r = positions[:, 0].min()
        min_rows.append((min_r, color))
    
    min_rows.sort()
    colors = [c for _, c in min_rows]
    
    num_layers = len(colors)
    pad = 5 - num_layers
    
    output = []
    for _ in range(pad):
        output.append([background] * 3)
    for c in colors:
        output.append([c] * 3)
    
    return output

def solve_211(I):
    n = len(I)
    m = len(I[0])
    out = [[0 for _ in range(m)] for _ in range(n)]
    out[0][0] = I[1][1]
    out[0][m-1] = I[1][2]
    out[n-1][0] = I[2][1]
    out[n-1][m-1] = I[2][2]
    return out

def solve_212(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0]) // 2
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 0 and I[r][c + cols] == 0:
                output[r][c] = 5
    
    return output

import copy

def solve_213(I):
    if not I or not I[0]:
        return []

    height = len(I)
    width = len(I[0])
    visited = [[False] * width for _ in range(height)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    output = copy.deepcopy(I)

    for r in range(height):
        for c in range(width):
            if I[r][c] == 3 and not visited[r][c]:
                component = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < height and 0 <= nc < width and not visited[nr][nc] and I[nr][nc] == 3:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                if len(component) > 1:
                    for pr, pc in component:
                        output[pr][pc] = 8
    return output

from collections import deque

def solve_214(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    stems = []
    for c in range(cols):
        r = 0
        while r < rows:
            if I[r][c] == 2:
                start = r
                while r < rows and I[r][c] == 2:
                    r += 1
                end = r - 1
                stems.append((c, start, end - start + 1))
            else:
                r += 1
    # assume len(stems) == 2
    stem1, stem2 = stems[0], stems[1]
    def has_seeds(stem):
        c_st, sr, L = stem
        count = 0
        for rr in range(sr, sr + L):
            for cc in range(cols):
                if cc != c_st and I[rr][cc] == 8:
                    count += 1
        return count
    count1 = has_seeds(stem1)
    count2 = has_seeds(stem2)
    if count1 > 0:
        seeded = stem1
        noseed = stem2
    else:
        seeded = stem2
        noseed = stem1
    # process seeded
    cr, sr, L = seeded
    rel_pos = []
    for r in range(sr, sr + L):
        purple_cols = [cc for cc in range(cols) if I[r][cc] == 8]
        if purple_cols:
            cp = purple_cols[0]
            dirr = 1 if cp > cr else -1
            current = cr + dirr
            while current != cp:
                output[r][current] = 8
                current += dirr
            output[r][cp] = 4
            rel = r - sr + 1
            rel_pos.append(rel)
    # process noseed
    cn, sn, Ln = noseed
    dir_left = cn > cols // 2
    for rel in rel_pos:
        r = sn + rel - 1
        if dir_left:
            for cc in range(cn - 1, -1, -1):
                output[r][cc] = 8
        else:
            for cc in range(cn + 1, cols):
                output[r][cc] = 8
    return output

import numpy as np

def solve_215(I):
    I = np.array(grid_lst)
    height, width = I.shape
    
    # Find unique non-zero colors
    colors = np.unique(I[I > 0])
    
    # Collect shapes
    shapes = []
    for col in colors:
        rs, cs = np.where(I == col)
        if len(rs) == 0:
            continue
        min_r = rs.min()
        max_r = rs.max()
        min_c = cs.min()
        max_c = cs.max()
        shapes.append({'color': col, 'min_r': min_r, 'max_r': max_r, 'min_c': min_c, 'max_c': max_c})
    
    # Determine mode
    vertical = height > width
    
    # Sort shapes
    if vertical:
        shapes.sort(key=lambda s: s['min_r'])
    else:
        shapes.sort(key=lambda s: s['min_c'])
    
    if not shapes:
        return []
    
    # Assume all have same size
    H = shapes[0]['max_r'] - shapes[0]['min_r'] + 1
    W = shapes[0]['max_c'] - shapes[0]['min_c'] + 1
    num = len(shapes)
    
    if vertical:
        out_height = num * H
        out_width = W
    else:
        out_height = H
        out_width = num * W
    
    out = np.zeros((out_height, out_width), dtype=int)
    
    # Place shapes
    for i, s in enumerate(shapes):
        if vertical:
            start_r = i * H
            start_c = 0
        else:
            start_r = 0
            start_c = i * W
        for dr in range(H):
            for dc in range(W):
                in_r = s['min_r'] + dr
                in_c = s['min_c'] + dc
                out[start_r + dr, start_c + dc] = I[in_r, in_c]
    
    return out.tolist()

def solve_216(I):
    height = len(I)
    width = len(I[0])

    # Find divider rows and columns
    div_rows = [r for r in range(height) if all(x == 8 for x in I[r])]
    div_rows.sort()
    div_cols = [c for c in range(width) if all(I[r][c] == 8 for r in range(height))]
    div_cols.sort()

    # Region starts and ends
    row_starts = [0, div_rows[0] + 1, div_rows[1] + 1]
    row_ends = [div_rows[0], div_rows[1], height]
    col_starts = [0, div_cols[0] + 1, div_cols[1] + 1]
    col_ends = [div_cols[0], div_cols[1], width]

    output = [[0] * 3 for _ in range(3)]

    for i in range(3):
        for j in range(3):
            count = 0
            for r in range(row_starts[i], row_ends[i]):
                for c in range(col_starts[j], col_ends[j]):
                    if I[r][c] == 6:
                        count += 1
            if count == 2:
                output[i][j] = 1

    return output

def solve_217(I):
    return [row[::-1] for row in I]

def solve_218(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    
    # Count non-zeros per row
    row_nonzeros = [sum(1 for x in row if x != 0) for row in I]
    max_row_count = max(row_nonzeros)
    r = row_nonzeros.index(max_row_count)
    
    # Count non-zeros per column
    col_nonzeros = [0] * cols
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != 0:
                col_nonzeros[j] += 1
    max_col_count = max(col_nonzeros)
    c = col_nonzeros.index(max_col_count)
    
    # Copy I
    output = [row[:] for row in I]
    
    # Set yellow in rows r-1 and r+1
    for dr in [-1, 1]:
        nr = r + dr
        if 0 <= nr < rows:
            for dc in [-1, 0, 1]:
                nc = c + dc
                if 0 <= nc < cols:
                    output[nr][nc] = 4
    
    # Set yellow in row r, columns c-1 and c+1
    for dc in [-1, 1]:
        nc = c + dc
        if 0 <= nc < cols:
            output[r][nc] = 4
    
    return output

def solve_219(I):
    top = []
    for row in I:
        left = row
        right = row[::-1]
        top.append(left + right)
    bottom = top[::-1]
    output = top + bottom
    return output

from collections import defaultdict

def solve_220(I):
    new_grid = [row[:] for row in I]
    n = len(I)
    color_pos = defaultdict(list)
    for i in range(n):
        for j in range(n):
            c = I[i][j]
            if c != 0 and c != 6 and c != 7:
                color_pos[c].append((i, j))
    for c, pos in color_pos.items():
        if len(pos) < 3:
            continue
        row_count = defaultdict(int)
        col_count = defaultdict(int)
        for r, cc in pos:
            row_count[r] += 1
            col_count[cc] += 1
        cross_row = max(row_count, key=row_count.get)
        cross_col = max(col_count, key=col_count.get)
        # horizontal
        horz_cols = [cc for r, cc in pos if r == cross_row]
        min_j = min(horz_cols)
        max_j = max(horz_cols)
        left_len = cross_col - min_j
        right_len = max_j - cross_col
        if left_len != right_len:
            if left_len > right_len:
                new_grid[cross_row][0] = 0
                new_grid[cross_row][n-1] = c
            else:
                new_grid[cross_row][0] = c
                new_grid[cross_row][n-1] = 0
        # vertical
        vert_rows = [r for r, cc in pos if cc == cross_col]
        min_i = min(vert_rows)
        max_i = max(vert_rows)
        up_len = cross_row - min_i
        down_len = max_i - cross_row
        if up_len != down_len:
            if up_len > down_len:
                new_grid[0][cross_col] = 0
                new_grid[n-1][cross_col] = c
            else:
                new_grid[0][cross_col] = c
                new_grid[n-1][cross_col] = 0
    return new_grid

def solve_221(I):
    return I[::-1]

def solve_222(I):
    output = []
    for r in range(3):
        row = []
        for c in range(3):
            row.append(I[2 * r][2 * c])
        output.append(row)
    return output

def solve_223(I):
    n = len(I)
    c = I[0][0]  # Assuming all cells are c
    
    # Build sparse row
    unit_len = n + 1
    unit = [0] * n + [c]
    num_full_units = 15 // unit_len
    remainder = 15 % unit_len
    sparse_row = []
    for _ in range(num_full_units):
        sparse_row.extend(unit)
    sparse_row.extend([0] * remainder)
    
    # Build output rows
    period = n + 1
    num_full_periods = 15 // period
    rem_rows = 15 % period
    output = []
    for _ in range(num_full_periods):
        for _ in range(n):
            output.append(sparse_row[:])
        output.append([c] * 15)
    for _ in range(rem_rows):
        output.append(sparse_row[:])
    
    return output

def solve_224(I):
    output = [[0 for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            k = I[i + 10][j]
            if k != 0:
                output[i][j] = k
            else:
                b = I[i][j]
                if b != 0:
                    output[i][j] = b
                else:
                    p = I[i + 5][j]
                    if p != 0:
                        output[i][j] = p
                    else:
                        output[i][j] = 0
    return output

import numpy as np

def find_component(I, r, c, visited, height, width):
    color = I[r, c]
    component = []
    stack = [(r, c)]
    while stack:
        rr, cc = stack.pop()
        if visited[rr, cc]:
            continue
        visited[rr, cc] = True
        component.append((rr, cc))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and I[nr, nc] == color:
                stack.append((nr, nc))
    return component, color

def solve_225(I):
    I = np.array(I)
    height, width = I.shape
    base_pos = np.argwhere(I == 2)
    if len(base_pos) == 0:
        return I.tolist()
    rows_base = base_pos[:, 0]
    cols_base = base_pos[:, 1]
    if np.all(cols_base == cols_base[0]):
        is_vertical = True
        base_coord = cols_base[0]
        axis = 'col'
        if base_coord == 0:
            dir = -1
        elif base_coord == width - 1:
            dir = 1
        else:
            dir = 0  # Unknown, but assume edge
    elif np.all(rows_base == rows_base[0]):
        is_vertical = False
        base_coord = rows_base[0]
        axis = 'row'
        if base_coord == 0:
            dir = -1
        elif base_coord == height - 1:
            dir = 1
        else:
            dir = 0
    else:
        return I.tolist()  # Unknown

    visited = np.zeros_like(I, dtype=bool)
    shapes = []
    for r in range(height):
        for c in range(width):
            if I[r, c] > 0 and I[r, c] != 2 and not visited[r, c]:
                pos, color = find_component(I, r, c, visited, height, width)
                shapes.append({'color': color, 'pos': pos})

    for shape in shapes:
        pos = shape['pos']
        if axis == 'col':
            cols = [cc for _, cc in pos]
            min_c = min(cols)
            max_c = max(cols)
            if dir == 1:
                shape['dist'] = base_coord - max_c
            elif dir == -1:
                shape['dist'] = min_c - base_coord
            else:
                shape['dist'] = 0
            shape['tiebreaker'] = min([rr for rr, _ in pos])
        else:
            rows = [rr for rr, _ in pos]
            min_r = min(rows)
            max_r = max(rows)
            if dir == 1:
                shape['dist'] = base_coord - max_r
            elif dir == -1:
                shape['dist'] = min_r - base_coord
            else:
                shape['dist'] = 0
            shape['tiebreaker'] = min([cc for _, cc in pos])

    shapes.sort(key=lambda s: (s['dist'], s['tiebreaker']))

    output = np.zeros_like(I)
    for rr, cc in base_pos:
        output[rr, cc] = 2

    for shape in shapes:
        pos = shape['pos']
        color = shape['color']
        if axis == 'col':
            current_coords = [cc for _, cc in pos]
            min_current = min(current_coords)
            if dir == 1:
                max_delta = (width - 1) - max(current_coords)
                for delta in range(max_delta, -1, -1):
                    new_pos = [(rr, cc + delta) for rr, cc in pos]
                    if all(0 <= nc < width and output[nr, nc] == 0 for nr, nc in new_pos):
                        chosen_delta = delta
                        break
                else:
                    chosen_delta = 0
            elif dir == -1:
                min_delta = -min_current
                for delta in range(min_delta, 1):
                    new_pos = [(rr, cc + delta) for rr, cc in pos]
                    if all(0 <= nc < width and output[nr, nc] == 0 for nr, nc in new_pos):
                        chosen_delta = delta
                        break
                else:
                    chosen_delta = 0
            else:
                chosen_delta = 0
            new_pos = [(rr, cc + chosen_delta) for rr, cc in pos]
        else:
            current_coords = [rr for rr, _ in pos]
            min_current = min(current_coords)
            if dir == 1:
                max_delta = (height - 1) - max(current_coords)
                for delta in range(max_delta, -1, -1):
                    new_pos = [(rr + delta, cc) for rr, cc in pos]
                    if all(0 <= nr < height and output[nr, nc] == 0 for nr, nc in new_pos):
                        chosen_delta = delta
                        break
                else:
                    chosen_delta = 0
            elif dir == -1:
                min_delta = -min_current
                for delta in range(min_delta, 1):
                    new_pos = [(rr + delta, cc) for rr, cc in pos]
                    if all(0 <= nr < height and output[nr, nc] == 0 for nr, nc in new_pos):
                        chosen_delta = delta
                        break
                else:
                    chosen_delta = 0
            else:
                chosen_delta = 0
            new_pos = [(rr + chosen_delta, cc) for rr, cc in pos]
        for rr, cc in new_pos:
            output[rr, cc] = color

    return output.tolist()

import numpy as np

def solve_226(I):
    I = np.array(grid_lst)
    rows, cols = I.shape

    # Find bounding box for color 2 (red outline)
    where2 = np.where(I == 2)
    min_r2 = np.min(where2[0])
    max_r2 = np.max(where2[0])
    min_c2 = np.min(where2[1])
    max_c2 = np.max(where2[1])
    n = max_r2 - min_r2 + 1
    assert max_c2 - min_c2 + 1 == n

    # Find bounding box for other color (small pattern)
    where_other = np.where((I != 0) & (I != 2))
    min_ro = np.min(where_other[0])
    max_ro = np.max(where_other[0])
    min_co = np.min(where_other[1])
    max_co = np.max(where_other[1])
    m = max_ro - min_ro + 1
    assert m == 3
    assert max_co - min_co + 1 == 3

    # Extract small 3x3 pattern
    small = np.zeros((3, 3), dtype=int)
    for r in range(min_ro, max_ro + 1):
        for c in range(min_co, max_co + 1):
            small[r - min_ro, c - min_co] = I[r, c]

    # Calculate scale factor
    k = (n - 2) // 3
    assert k * 3 == n - 2

    # Create output I
    out = np.zeros((n, n), dtype=int)

    # Set border to 2
    out[0, :] = 2
    out[-1, :] = 2
    out[:, 0] = 2
    out[:, -1] = 2

    # Scale and place the small pattern in the interior
    for i in range(3):
        for j in range(3):
            color = small[i, j]
            for dr in range(k):
                for dc in range(k):
                    out_r = 1 + i * k + dr
                    out_c = 1 + j * k + dc
                    out[out_r, out_c] = color

    return out.tolist()

def solve_227(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]  # Copy the I

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            # Check if it's a 3x3 with perimeter 1 and center 0
            if (
                output[r-1][c-1] == 1 and output[r-1][c] == 1 and output[r-1][c+1] == 1 and
                output[r][c-1] == 1 and output[r][c] == 0 and output[r][c+1] == 1 and
                output[r+1][c-1] == 1 and output[r+1][c] == 1 and output[r+1][c+1] == 1
            ):
                # Set corners to 0
                output[r-1][c-1] = 0
                output[r-1][c+1] = 0
                output[r+1][c-1] = 0
                output[r+1][c+1] = 0
                # Set center and arms to 2
                output[r][c] = 2
                output[r-1][c] = 2
                output[r+1][c] = 2
                output[r][c-1] = 2
                output[r][c+1] = 2

    return output

def solve_228(I):
    if not I or not I[0]:
        return []
    height = len(I)
    width = len(I[0])
    output = [[0 for _ in range(width)] for _ in range(height)]
    # Fill horizontal lines
    for r in range(height):
        if I[r][0] != 0 and I[r][0] == I[r][width - 1]:
            color = I[r][0]
            for c in range(width):
                output[r][c] = color
    # Fill vertical lines
    for c in range(width):
        if I[0][c] != 0 and I[0][c] == I[height - 1][c]:
            color = I[0][c]
            for r in range(height):
                output[r][c] = color
    return output

def solve_229(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    
    # Find position of 4 (yellow)
    yellow_r, yellow_c = None, None
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 4:
                yellow_r, yellow_c = i, j
                break  # Assume unique
        if yellow_r is not None:
            break
    
    if yellow_r is None:
        return I  # No transformation if no yellow
    
    # Panel start indices
    start_rows = [0, 4, 8]
    start_cols = [0, 4, 8]
    
    # Source panel
    source_pr = yellow_r // 4
    source_pc = yellow_c // 4
    
    # Local position
    local_r = yellow_r - start_rows[source_pr]
    local_c = yellow_c - start_cols[source_pc]
    
    # Target panel
    target_pr = local_r
    target_pc = local_c
    
    # Start rows/cols for source and target
    source_start_r = start_rows[source_pr]
    source_start_c = start_cols[source_pc]
    target_start_r = start_rows[target_pr]
    target_start_c = start_cols[target_pc]
    
    # Create output
    output = [row[:] for row in I]
    
    # Clear all non-5 cells to 0
    for i in range(rows):
        for j in range(cols):
            if output[i][j] != 5:
                output[i][j] = 0
    
    # Copy source panel to target panel
    for dr in range(3):
        for dc in range(3):
            sr = source_start_r + dr
            sc = source_start_c + dc
            tr = target_start_r + dr
            tc = target_start_c + dc
            output[tr][tc] = I[sr][sc]
    
    return output

def solve_230(I):
    return [row + row[::-1] for row in I]

from collections import Counter

def solve_231(I):
    if not I or not I[0]:
        return []
    
    n = len(I)
    m = len(I[0])
    
    left_col = [row[0] for row in I]
    counts = Counter(left_col)
    majority = counts.most_common(1)[0][0]
    
    r = None
    for i in range(n):
        if I[i][0] != majority:
            r = i
            break
    
    if r is None:
        r = 0  # Default if no marker, though examples always have one
    
    new_grid = []
    for i in range(n):
        old_row_idx = (i - r + n) % n
        new_row = I[old_row_idx][1:]
        new_grid.append(new_row)
    
    return new_grid

def solve_232(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    min_col = float('inf')
    max_col = float('-inf')
    purple_rows = set()
    for r in range(rows):
        has_purple = False
        for c in range(cols):
            if I[r][c] == 8:
                has_purple = True
                min_col = min(min_col, c)
                max_col = max(max_col, c)
        if has_purple:
            purple_rows.add(r)
    if min_col == float('inf'):
        return I
    output = [row[:] for row in I]
    for r in purple_rows:
        for c in range(min_col, max_col + 1):
            if output[r][c] == 0:
                output[r][c] = 2
    return output

import math
from collections import defaultdict

def solve_233(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Find positions of grey (5)
    pos5 = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 5:
                pos5.append((r, c))
    
    n = len(pos5)
    if n == 0:
        return [row[:] for row in I]  # No change if no grey
    
    # Compute centroid
    sum_r = sum(r for r, c in pos5)
    sum_c = sum(c for r, c in pos5)
    cr = sum_r / n
    cc = sum_c / n
    
    # Collect positions for each color !=0, !=5
    color_pos = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            val = I[r][c]
            if val != 0 and val != 5:
                color_pos[val].append((r, c))
    
    if not color_pos:
        return [row[:] for row in I]  # No colors, no change
    
    # Compute min dist for each color
    min_dist = {}
    for col, poss in color_pos.items():
        mind = min(math.sqrt((r - cr)**2 + (c - cc)**2) for r, c in poss)
        min_dist[col] = mind
    
    # Choose color with smallest min_dist
    chosen = min(min_dist, key=lambda x: min_dist[x])
    
    # Create output: only the grey positions set to chosen, rest 0
    output = [[0] * cols for _ in range(rows)]
    for r, c in pos5:
        output[r][c] = chosen
    
    return output

def solve_234(I):
    if not I or not I[0]:
        return I
    
    # Check if uniform
    def is_uniform(g):
        color = g[0][0]
        for row in g:
            for cell in row:
                if cell != color:
                    return False
        return True
    
    # Check if symmetric (left-right mirror for 3x3)
    def is_symmetric(g):
        for r in range(3):
            if g[r][0] != g[r][2]:
                return False
        return True
    
    # Create output I
    output = [[0] * 3 for _ in range(3)]
    
    if is_uniform(I):
        output[0] = [5, 5, 5]
    else:
        if is_symmetric(I):
            # Main diagonal
            for i in range(3):
                output[i][i] = 5
        else:
            # Anti-diagonal
            for i in range(3):
                output[i][2 - i] = 5
    
    return output

from collections import deque
import copy

def solve_235(I):
    if not I or not I[0]:
        return I
    I = copy.deepcopy(I)  # Make a copy to avoid modifying original
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 5 and not visited[r][c]:
                # Start BFS for new component
                comp = []
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and I[nx][ny] == 5:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                # Recolor the component
                size = len(comp)
                new_color = 5 - size
                for x, y in comp:
                    I[x][y] = new_color
    return I

def solve_236(I):
    if not I or not I[0]:
        return []
    h = len(I)
    w = len(I[0])
    visited = [[False] * w for _ in range(h)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []

    def dfs(r, c):
        stack = [(r, c)]
        comp = []
        while stack:
            cr, cc = stack.pop()
            if visited[cr][cc]:
                continue
            visited[cr][cc] = True
            comp.append((cr, cc))
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and I[nr][nc] != 0:
                    stack.append((nr, nc))
        return comp

    for r in range(h):
        for c in range(w):
            if I[r][c] != 0 and not visited[r][c]:
                comp = dfs(r, c)
                components.append(comp)

    if len(components) != 2:
        return []  # Assume exactly two, but handle if not

    comp1, comp2 = components
    if len(comp1) < len(comp2):
        small_comp = comp1
        large_comp = comp2
    else:
        small_comp = comp2
        large_comp = comp1

    small_rs = [r for r, c in small_comp]
    small_cs = [c for r, c in small_comp]
    min_r_s = min(small_rs)
    max_r_s = max(small_rs)
    min_c_s = min(small_cs)
    max_c_s = max(small_cs)
    k = max_r_s - min_r_s + 1

    large_rs = [r for r, c in large_comp]
    large_cs = [c for r, c in large_comp]
    min_r_l = min(large_rs)
    max_r_l = max(large_rs)
    min_c_l = min(large_cs)
    max_c_l = max(large_cs)
    m = max_r_l - min_r_l + 1
    s = m // k

    large_color = I[large_comp[0][0]][large_comp[0][1]]

    output = [[0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            output[i][j] = I[min_r_s + i][min_c_s + j]

    for i in range(k):
        for j in range(k):
            full = True
            for di in range(s):
                if not full:
                    break
                for dj in range(s):
                    rr = min_r_l + i * s + di
                    cc = min_c_l + j * s + dj
                    if I[rr][cc] != large_color:
                        full = False
                        break
            if not full:
                output[i][j] = 0

    return output

def solve_237(I):
    if not I:
        return []
    h = len(I)
    if h == 0:
        return []
    w = len(I[0])
    out = [[8 for _ in range(w)] for _ in range(h)]
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            out[r][c] = 0
    return out

def solve_238(I):
    return I + list(reversed(I))

def solve_239(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]

    # Find purple bounds
    min_r, max_r, min_c, max_c = rows, -1, cols, -1
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 8:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    if min_r > max_r:
        return output  # No purple

    # Horizontal fills for each purple row
    for pr in range(min_r, max_r + 1):
        # Left
        for c in range(min_c - 1, -1, -1):
            if I[pr][c] == 1 or I[pr][c] == 8:
                break
            output[pr][c] = 4
        # Right
        for c in range(max_c + 1, cols):
            if I[pr][c] == 1 or I[pr][c] == 8:
                break
            output[pr][c] = 4

    # Vertical fills for each purple column
    for pc in range(min_c, max_c + 1):
        # Up
        for r in range(min_r - 1, -1, -1):
            if I[r][pc] == 1 or I[r][pc] == 8:
                break
            output[r][pc] = 4
        # Down
        for r in range(max_r + 1, rows):
            if I[r][pc] == 1 or I[r][pc] == 8:
                break
            output[r][pc] = 4

    return output

import numpy as np

def solve_240(I):
    I = np.array(grid_lst)
    rows, cols = I.shape
    
    # Find colors that form full rows
    row_line_colors = set()
    for r in range(rows):
        val = I[r, 0]
        if np.all(I[r, :] == val) and val != 0:
            row_line_colors.add(val)
    
    # Find colors that form full columns
    col_line_colors = set()
    for c in range(cols):
        val = I[0, c]
        if np.all(I[:, c] == val) and val != 0:
            col_line_colors.add(val)
    
    # Line color is the intersection (assuming one)
    line_colors = row_line_colors.intersection(col_line_colors)
    line_color = next(iter(line_colors))
    
    # Background is the other non-zero color (assuming two colors total)
    all_colors = set(I.flatten()) - {0}
    background = (all_colors - {line_color}).pop()
    
    # Find horizontal line positions
    h_lines = [r for r in range(rows) if np.all(I[r, :] == line_color)]
    h_lines.sort()
    
    # Find vertical line positions
    v_lines = [c for c in range(cols) if np.all(I[:, c] == line_color)]
    v_lines.sort()
    
    # Count positive row gaps
    num_row_gaps = 0
    prev = -1
    for hl in h_lines:
        start = prev + 1
        end = hl - 1
        if start <= end:
            num_row_gaps += 1
        prev = hl
    # After last
    start = prev + 1
    end = rows - 1
    if start <= end:
        num_row_gaps += 1
    
    # Count positive column gaps
    num_col_gaps = 0
    prev = -1
    for vl in v_lines:
        start = prev + 1
        end = vl - 1
        if start <= end:
            num_col_gaps += 1
        prev = vl
    # After last
    start = prev + 1
    end = cols - 1
    if start <= end:
        num_col_gaps += 1
    
    # Create output I
    output = [[background for _ in range(num_col_gaps)] for _ in range(num_row_gaps)]
    return output

def solve_241(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    mid_row = (rows - 1) // 2
    mid_col = (cols - 1) // 2
    for r in range(rows):
        for c in range(cols):
            k = I[r][c]
            if k == 0:
                continue
            # Vertical
            if r <= mid_row:
                v_len = r + 1
                v_start = r - (v_len - 1)
                v_end = r
            else:
                v_len = rows - r
                v_start = r
                v_end = r + (v_len - 1)
            # Horizontal
            if c <= mid_col:
                h_len = c + 1
                h_start = c - (h_len - 1)
                h_end = c
            else:
                h_len = cols - c
                h_start = c
                h_end = c + (h_len - 1)
            # Fill vertical
            for rr in range(v_start, v_end + 1):
                output[rr][c] = k
            # Fill horizontal
            for cc in range(h_start, h_end + 1):
                output[r][cc] = k
    return output

def solve_242(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    # Collect cycle of non-zero colors
    cycle = []
    for c in range(cols):
        if I[1][c] != 0:
            cycle.append(I[1][c])
    if not cycle:
        return I
    # Generate positions
    positions = []
    current = 0
    positions.append(current)
    delta = 1
    while True:
        current += delta
        if current >= cols:
            break
        positions.append(current)
        delta += 1
    # Create output I
    output = [row[:] for row in I]
    # Set colors cycling through cycle
    for i, pos in enumerate(positions):
        output[1][pos] = cycle[i % len(cycle)]
    return output

def solve_243(I):
    if not I or not I[0]:
        return []
    h = len(I)
    w = len(I[0])
    visited = set()
    components = []
    for i in range(h):
        for j in range(w):
            if I[i][j] != 0 and (i, j) not in visited:
                color = I[i][j]
                component = []
                stack = [(i, j)]
                visited.add((i, j))
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and I[nx][ny] == color and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            stack.append((nx, ny))
                if len(component) >= 4:
                    components.append((color, component))
    for color, comp in components:
        if not comp:
            continue
        min_r = min(p[0] for p in comp)
        max_r = max(p[0] for p in comp)
        min_c = min(p[1] for p in comp)
        max_c = max(p[1] for p in comp)
        bh = max_r - min_r + 1
        bw = max_c - min_c + 1
        small = [[0] * bw for _ in range(bh)]
        for r in range(bh):
            for c in range(bw):
                small[r][c] = I[min_r + r][min_c + c]
        mirrored = [row[::-1] for row in small]
        if mirrored == small:
            return small
    return []

import numpy as np

def solve_244(I):
    I = np.array(I)
    rows, cols = np.where(I != 0)
    if len(rows) == 0:
        return [[0] * 4 for _ in range(4)]
    min_r = rows.min()
    max_r = rows.max()
    min_c = cols.min()
    size = (max_r - min_r + 1) // 2
    extract_start_r = max_r - size + 1
    extract = I[extract_start_r : max_r + 1, min_c : min_c + size]
    flipped = extract[::-1, :]
    return flipped.tolist()

def solve_245(I):
    if not I or not I[0]:
        return []
    
    rows = len(I)
    cols = len(I[0])
    
    # Find min and max row and col with non-zero
    min_row = rows
    max_row = -1
    min_col = cols
    max_col = -1
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != 0:
                min_row = min(min_row, i)
                max_row = max(max_row, i)
                min_col = min(min_col, j)
                max_col = max(max_col, j)
    
    if min_row > max_row:
        return []  # all zero
    
    # Extract subgrid
    subgrid = []
    for i in range(min_row, max_row + 1):
        row = []
        for j in range(min_col, max_col + 1):
            row.append(I[i][j])
        subgrid.append(row)
    
    # Horizontal flip
    flipped = [row[::-1] for row in subgrid]
    
    return flipped

import numpy as np
from itertools import groupby

def solve_246(I):
    if not grid_lst or not grid_lst[0]:
        return []
    I = np.array(grid_lst)
    left_column = I[:, 0]
    top_row = I[0, :]
    is_vertical_layers = np.all(left_column == left_column[0])
    sequence = top_row if is_vertical_layers else left_column
    colors = [key for key, _ in groupby(sequence)]
    if is_vertical_layers:
        return [colors]
    else:
        return [[color] for color in colors]

def solve_247(I):
    # Transpose the I
    return [list(row) for row in zip(*I)]

import numpy as np

def solve_248(I):
    if not I or not I[0]:
        return I
    
    I = np.array(I)
    rows, cols = I.shape
    
    # Find wall
    wall_type = None
    wall_pos = None
    for r in range(rows):
        if np.all(I[r, :] == 0):
            wall_type = 'horizontal'
            wall_pos = r
            break
    if wall_type is None:
        for c in range(cols):
            if np.all(I[:, c] == 0):
                wall_type = 'vertical'
                wall_pos = c
                break
    if wall_type is None:
        return I.tolist()
    
    output = I.copy()
    
    if wall_type == 'vertical':
        # Assume left wall
        start = wall_pos + 1
        field_size = cols - start
        for line in range(rows):
            seq = list(output[line, start: start + field_size])
            if len(seq) < 2:
                continue
            if seq[0] == 8:
                if seq[1] == 8:
                    seq[0] = 7
                    seq[1] = 7
                    if len(seq) >= 7:
                        seq[5] = 0
                        seq[6] = 0
                else:
                    seq[1] = 8
            output[line, start: start + field_size] = seq
    else:  # horizontal
        if wall_pos == 0:  # top
            direction = 1
            start = wall_pos + 1
            field_size = rows - start
        else:  # bottom
            direction = -1
            start = wall_pos - 1
            field_size = start + 1
        for line in range(cols):
            if direction == 1:
                seq = list(output[start: start + field_size, line])
            else:
                seq = []
                curr = start
                for _ in range(field_size):
                    seq.append(output[curr, line])
                    curr -= 1
            if len(seq) < 2:
                continue
            if seq[0] == 8:
                if seq[1] == 8:
                    seq[0] = 7
                    seq[1] = 7
                    if len(seq) >= 7:
                        seq[5] = 0
                        seq[6] = 0
                else:
                    seq[1] = 8
            if direction == 1:
                output[start: start + field_size, line] = seq
            else:
                idx = 0
                curr = start
                for _ in range(field_size):
                    output[curr, line] = seq[idx]
                    idx += 1
                    curr -= 1
    
    return output.tolist()

def solve_249(I):
    N = len(I) // 2
    out = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            G = I[i][j + N]
            P = I[i + N][j]
            B = I[i + N][j + N]
            Y = I[i][j]
            if G != 0:
                out[i][j] = G
            elif P != 0:
                out[i][j] = P
            elif B != 0:
                out[i][j] = B
            elif Y != 0:
                out[i][j] = Y
            else:
                out[i][j] = 0
    return out

def solve_250(I):
    I = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])

    # Find red_row
    red_row = None
    for r in range(rows):
        if any(I[r][c] == 2 for c in range(cols)):
            red_row = r
            break  # Assume only one

    if red_row is None:
        return I

    # Find colored rows (non-zero, non-2)
    colored_rows = []
    for r in range(rows):
        if r != red_row and any(I[r][c] != 0 for c in range(cols)):
            colored_rows.append(r)

    if len(colored_rows) != 2:
        return I  # Assume exactly two

    top_row = min(colored_rows)
    bottom_row = max(colored_rows)

    # Get columns sets
    set_top = set(c for c in range(cols) if I[top_row][c] != 0)
    set_bottom = set(c for c in range(cols) if I[bottom_row][c] != 0)

    overlap = set_top & set_bottom
    if not overlap:
        return I

    min_col = min(overlap)
    max_col = max(overlap)

    len_top = len(set_top)
    len_bottom = len(set_bottom)

    if len_top == len_bottom:
        return I  # No fill if equal, though not in examples

    if len_top > len_bottom:
        height = red_row - top_row - 1
        start_r = top_row + 1
    else:
        height = bottom_row - red_row - 1
        start_r = red_row + 1

    # Fill
    for dr in range(height):
        r = start_r + dr
        for c in range(min_col, max_col + 1):
            I[r][c] = 4

    return I

def solve_251(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    # Corner colors
    tl = I[0][0]
    tr = I[0][cols - 1]
    bl = I[rows - 1][0]
    br = I[rows - 1][cols - 1]
    # Find bounding box of 8s
    min_r = rows
    max_r = -1
    min_c = cols
    max_c = -1
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 8:
                min_r = min(min_r, i)
                max_r = max(max_r, i)
                min_c = min(min_c, j)
                max_c = max(max_c, j)
    if max_r < 0:
        return []  # No 8s
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    quarter = h // 2
    # Create output
    output = [[0 for _ in range(w)] for _ in range(h)]
    for ri in range(h):
        for ci in range(w):
            actual_i = min_r + ri
            actual_j = min_c + ci
            if I[actual_i][actual_j] == 8:
                if ri < quarter:
                    if ci < quarter:
                        output[ri][ci] = tl
                    else:
                        output[ri][ci] = tr
                else:
                    if ci < quarter:
                        output[ri][ci] = bl
                    else:
                        output[ri][ci] = br
    return output

def solve_252(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    count = sum(sum(1 for cell in row if cell == 1) for row in I)
    output = [[0] * cols for _ in range(rows)]
    positions = [(0, 0), (0, 1), (0, 2), (1, 1)]
    for i in range(min(count, len(positions))):
        r, c = positions[i]
        output[r][c] = 2
    return output

def solve_253(I):
    if not I or not I[0]:
        return []
    n = len(I)
    out = [[0 for _ in range(2 * n)] for _ in range(2 * n)]

    # Top-left: copy I
    for i in range(n):
        for j in range(n):
            out[i][j] = I[i][j]

    # Bottom-left: 180 rotation
    for i in range(n):
        for j in range(n):
            out[n + i][j] = I[n - 1 - i][n - 1 - j]

    # Compute transpose
    trans = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            trans[i][j] = I[j][i]

    # Top-right: vertical flip of transpose
    for i in range(n):
        for j in range(n):
            out[i][n + j] = trans[n - 1 - i][j]

    # Bottom-right: horizontal flip of transpose
    for i in range(n):
        for j in range(n):
            out[n + i][n + j] = trans[i][n - 1 - j]

    return out

import numpy as np
from collections import deque

def solve_254(I):
    I = np.array(grid_lst)
    rows, cols = I.shape
    output = I.copy()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = np.zeros((rows, cols), dtype=bool)
    queue = deque()

    # Enqueue all border 0 cells
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and I[r, c] == 0:
                queue.append((r, c))
                visited[r, c] = True
                output[r, c] = 3

    # BFS flood fill with 3
    while queue:
        cr, cc = queue.popleft()
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and I[nr, nc] == 0:
                visited[nr, nc] = True
                output[nr, nc] = 3
                queue.append((nr, nc))

    # Set remaining 0's to 2
    output[output == 0] = 2

    return output.tolist()

import numpy as np

def solve_255(I):
    if not I:
        return []
    grid_np = np.array(I)
    h, w = grid_np.shape
    if w % 2 == 0:
        half_w = w // 2
        if np.array_equal(grid_np[:, :half_w], grid_np[:, half_w:]):
            return grid_np[:, :half_w].tolist()
    if h % 2 == 0:
        half_h = h // 2
        if np.array_equal(grid_np[:half_h, :], grid_np[half_h:, :]):
            return grid_np[:half_h, :].tolist()
    # Assume always one direction matches, as per examples
    return I

def solve_256(I):
    n = len(I)
    small_size = 2
    large_size = n - 1 - small_size
    scale = large_size // small_size

    # Find horizontal purple row h
    h = next(r for r in range(n) if all(I[r][c] == 8 for c in range(n)))

    # Find vertical purple column v
    v = next(c for c in range(n) if all(I[r][c] == 8 for r in range(n)))

    # Determine shape_rows
    if h == small_size:
        shape_rows = list(range(h + 1, h + 1 + large_size))
    else:
        shape_rows = list(range(h - large_size, h))

    # Determine shape_cols
    if v == small_size:
        shape_cols = list(range(v + 1, v + 1 + large_size))
    else:
        shape_cols = list(range(v - large_size, v))

    # Determine small_rows
    if h == small_size:
        small_rows = list(range(0, h))
    else:
        small_rows = list(range(h + 1, n))

    # Determine small_cols
    if v == small_size:
        small_cols = list(range(0, v))
    else:
        small_cols = list(range(v + 1, n))

    # Extract small_grid
    small_grid = [[I[r][c] for c in small_cols] for r in small_rows]

    # Create output
    output = [[0 for _ in range(large_size)] for _ in range(large_size)]
    for i in range(large_size):
        for j in range(large_size):
            r = shape_rows[i]
            c = shape_cols[j]
            if I[r][c] == 3:
                quad_i = i // scale
                quad_j = j // scale
                output[i][j] = small_grid[quad_i][quad_j]

    return output

from collections import deque

def solve_257(I):
    if not I or not I[0]:
        return I
    h = len(I)
    w = len(I[0])
    visited = [[False] * w for _ in range(h)]
    comps = []
    for i in range(h):
        for j in range(w):
            if I[i][j] != 0 and not visited[i][j]:
                color = I[i][j]
                cells = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                rep = min(cells)
                comps.append({'cells': cells, 'color': color, 'rep': rep})
    n = len(comps)
    enclose_matrix = [[False] * n for _ in range(n)]
    for ai in range(n):
        A = comps[ai]
        A_set = set(A['cells'])
        for bi in range(n):
            if ai == bi:
                continue
            B = comps[bi]
            rr, cc = B['rep']
            vis = [[False] * w for _ in range(h)]
            q = deque([(rr, cc)])
            vis[rr][cc] = True
            can_escape = False
            while q:
                r, c = q.popleft()
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    can_escape = True
                    break
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not vis[nr][nc] and (nr, nc) not in A_set:
                        vis[nr][nc] = True
                        q.append((nr, nc))
            if not can_escape:
                enclose_matrix[ai][bi] = True
    parent = [-1] * n
    for bi in range(n):
        enclosers = [ai for ai in range(n) if enclose_matrix[ai][bi]]
        direct = []
        for ai in enclosers:
            is_direct = True
            for ci in enclosers:
                if ci != ai and enclose_matrix[ai][ci] and enclose_matrix[ci][bi]:
                    is_direct = False
                    break
            if is_direct:
                direct.append(ai)
        if direct:
            parent[bi] = direct[0]
    def get_root_color(idx):
        if parent[idx] == -1:
            return comps[idx]['color']
        else:
            return get_root_color(parent[idx])
    for i in range(n):
        new_color = get_root_color(i)
        for r, c in comps[i]['cells']:
            I[r][c] = new_color
    return I

from collections import deque

def solve_258(I):
    if not I or not I[0]:
        return I
    
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    visited = set()
    
    dirs = [(dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if not (dr == 0 and dc == 0)]
    
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and (r, c) not in visited:
                color = I[r][c]
                component = set()
                q = deque([(r, c)])
                visited.add((r, c))
                component.add((r, c))
                
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                            component.add((nr, nc))
                
                # Build neighbors
                neighbors = {pos: [] for pos in component}
                for pos in component:
                    pr, pc = pos
                    for drr, dcc in dirs:
                        nr, nc = pr + drr, pc + dcc
                        if (nr, nc) in component:
                            neighbors[pos].append((nr, nc))
                
                # Find ends
                ends = [pos for pos in component if len(neighbors[pos]) == 1]
                
                # Extend from each end
                for end in ends:
                    neigh = neighbors[end][0]
                    dr = end[0] - neigh[0]
                    dc = end[1] - neigh[1]
                    current_r, current_c = end[0], end[1]
                    while True:
                        next_r = current_r + dr
                        next_c = current_c + dc
                        if not (0 <= next_r < rows and 0 <= next_c < cols):
                            break
                        output[next_r][next_c] = color
                        current_r = next_r
                        current_c = next_c
    
    return output

from collections import Counter

def solve_259(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Count non-zero colors
    color_count = Counter()
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0:
                color_count[I[r][c]] += 1
    
    if len(color_count) != 2:
        return I  # Assuming always 2 non-zero colors as per examples
    
    main, defect = color_count.most_common()[0][0], color_count.most_common()[1][0]
    
    # Directions for neighbors
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Copy I
    output = [row[:] for row in I]
    
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == defect:
                neigh_count = 0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == main:
                        neigh_count += 1
                if neigh_count >= 2:
                    output[r][c] = main
                else:
                    output[r][c] = 0
    
    return output

def solve_260(I):
    if len(I) < 3:
        return I  # Though not needed for given inputs, as safeguard
    first_row = I[0][:]
    second_row = I[1][:]
    B = I[2][0]  # Background color, assuming row 2 is uniform
    third_row = [x if x == B else 6 for x in first_row]
    return [first_row, second_row, third_row]

def solve_261(I):
    if not I or not I[0]:
        return []

    height = len(I)
    width = len(I[0])

    # Find background and special position
    bg = I[0][0]
    s_r = None
    s_c = None
    for i in range(height):
        for j in range(width):
            if I[i][j] != bg:
                if s_r is not None:
                    # Assume only one special cell
                    break
                s_r = i
                s_c = j

    if s_r is None:
        return I

    new_color = 1

    center = (width - 1) // 2
    reverse = (s_c == center)

    left_count = s_c
    right_count = width - s_c - 1

    # Copy I
    output = [row[:] for row in I]

    # Set vertical line
    for i in range(height):
        if i != s_r:
            output[i][s_c] = new_color

    # Set extensions
    top_row = 0
    bottom_row = height - 1

    if reverse:
        # Top: extend right
        for k in range(1, right_count + 1):
            if s_c + k < width:
                output[top_row][s_c + k] = new_color
        # Bottom: extend left
        for k in range(1, left_count + 1):
            if s_c - k >= 0:
                output[bottom_row][s_c - k] = new_color
    else:
        # Top: extend left
        for k in range(1, left_count + 1):
            if s_c - k >= 0:
                output[top_row][s_c - k] = new_color
        # Bottom: extend right
        for k in range(1, right_count + 1):
            if s_c + k < width:
                output[bottom_row][s_c + k] = new_color

    return output

def solve_262(I):
    if not I or not I[0]:
        return I
    output = [row[:] for row in I]
    rows = len(output)
    cols = len(output[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    changed = True
    while changed:
        changed = False
        to_remove = []
        for r in range(rows):
            for c in range(cols):
                if output[r][c] == 0:
                    continue
                color = output[r][c]
                neigh_count = 0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == color:
                        neigh_count += 1
                if neigh_count < 2:
                    to_remove.append((r, c))
        if to_remove:
            changed = True
            for r, c in to_remove:
                output[r][c] = 0
    return output

def solve_263(I):
    if not I:
        return []
    h = len(I)
    w = len(I[0])
    # Compute transpose
    transpose = [[I[j][i] for j in range(h)] for i in range(w)]
    # Horizontal flip of transpose (reverse each row)
    right = [row[::-1] for row in transpose]
    # Build top half
    top = [I[i] + right[i] for i in range(h)]
    # Compute bottom half as 180 rotation of top
    bottom = top[::-1]
    bottom = [row[::-1] for row in bottom]
    # Combine top and bottom
    return top + bottom

import copy
from collections import deque

def solve_264(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])

    # DFS to find connected components of color
    def dfs(r, c, color, visited, component):
        stack = [(r, c)]
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            component.append((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and I[nx][ny] == color and (nx, ny) not in visited:
                    stack.append((nx, ny))

    # Find all components of 1
    visited = set()
    components = []
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 1 and (i, j) not in visited:
                comp = []
                dfs(i, j, 1, visited, comp)
                components.append(comp)

    # Flood fill external 0's from borders
    visited_0 = set()
    queue = deque()
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and I[i][j] == 0:
                queue.append((i, j))
                visited_0.add((i, j))
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and I[nx][ny] == 0 and (nx, ny) not in visited_0:
                visited_0.add((nx, ny))
                queue.append((nx, ny))

    # Find hole cells: 0's not visited_0
    holes = []
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 0 and (i, j) not in visited_0:
                holes.append((i, j))

    # Map positions to component index
    component_dict = {pos: idx for idx, comp in enumerate(components) for pos in comp}

    # Mark components to change
    to_change = set()
    for h_r, h_c in holes:
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            n_r, n_c = h_r + dx, h_c + dy
            if 0 <= n_r < rows and 0 <= n_c < cols and I[n_r][n_c] == 1:
                comp_idx = component_dict[(n_r, n_c)]
                to_change.add(comp_idx)

    # Create output
    output = copy.deepcopy(I)
    for idx in to_change:
        for r, c in components[idx]:
            output[r][c] = 3

    return output

def solve_265(I):
    output = [row[:] for row in I]
    rows = len(I)
    if rows == 0:
        return output
    cols = len(I[0])
    # Find model row
    model_r = -1
    for r in range(rows):
        if all(I[r][c] != 0 for c in range(cols)):
            model_r = r
            break
    if model_r == -1:
        return output
    # Find colors in model
    colors = set(I[model_r][c] for c in range(cols))
    if len(colors) != 2:
        return output
    A = I[model_r][0]
    colors.remove(A)
    B = list(colors)[0]
    # Now, for each other row
    for r in range(rows):
        if r == model_r:
            continue
        # Find the length of partial
        k = 0
        for c in range(cols):
            if I[r][c] != 0:
                k = c + 1
            else:
                break
        if k == 0 or k == cols:
            continue
        # Check exactly two colors
        partial_colors = set(I[r][c] for c in range(k))
        if len(partial_colors) != 2:
            continue
        C = I[r][0]
        partial_colors.remove(C)
        D = list(partial_colors)[0]
        # Fill the entire row
        for c in range(cols):
            output[r][c] = C if I[model_r][c] == A else D
    return output

def solve_266(I):
    if not I:
        return []
    # Assume 5x1 I based on examples
    colors = [row[0] for row in I]
    # Swap first and second
    colors[0], colors[1] = colors[1], colors[0]
    # Swap fourth and fifth
    colors[3], colors[4] = colors[4], colors[3]
    return [[c] for c in colors]

import numpy as np

def solve_267(I):
    if not grid_lst or not grid_lst[0]:
        return []
    I = np.array(grid_lst)
    n = I.shape[0]
    # 180 degree rotation
    rotated = np.rot90(I, 2)
    # Create output I
    out = np.zeros((2 * n, 2 * n), dtype=int)
    # Top-left
    out[0:n, 0:n] = rotated
    # Top-right: horizontal flip of top-left
    out[0:n, n:2*n] = np.fliplr(out[0:n, 0:n])
    # Bottom-left: vertical flip of top-left
    out[n:2*n, 0:n] = np.flipud(out[0:n, 0:n])
    # Bottom-right: horizontal flip of bottom-left
    out[n:2*n, n:2*n] = np.fliplr(out[n:2*n, 0:n])
    return out.tolist()

def solve_268(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    # Find the colored cell
    r, c, color = None, None, None
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != 0:
                r, c, color = i, j, I[i][j]
                break
        if r is not None:
            break
    if r is None:
        return [row[:] for row in I]
    # Create output
    output = [row[:] for row in I]
    # Determine parity
    parity = c % 2
    # Fill with 4 in matching parity columns from row 0 to r
    for i in range(r + 1):
        for j in range(cols):
            if j % 2 == parity:
                output[i][j] = 4
    # Move the color down by 1 row
    if r + 1 < rows:
        output[r + 1][c] = color
    return output

def solve_269(I):
    h = len(I)
    w = len(I[0])
    
    # Find the seed position and color
    start_col = -1
    color = 0
    for r in range(h):
        for c in range(w):
            if I[r][c] != 0:
                start_col = c
                color = I[r][c]
                break
    
    # Create output I initialized to 0
    output = [[0] * w for _ in range(h)]
    
    # Fill the pattern starting from start_col
    for col_idx in range(start_col, w):
        offset = col_idx - start_col
        if offset % 2 == 0:
            # Full column with color
            for r in range(h):
                output[r][col_idx] = color
        else:
            # Special column
            special_index = (offset + 1) // 2
            if special_index % 2 == 1:
                output[0][col_idx] = 5
            else:
                output[h - 1][col_idx] = 5
    
    return output

import numpy as np

def solve_270(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]

    # Find red_row
    red_row = None
    for r in range(rows):
        if all(I[r][c] == 2 for c in range(cols)):
            red_row = r
            break
    if red_row is None:
        return output  # No red row, no change

    # Directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Visited
    visited = [[False] * cols for _ in range(rows)]

    # Find components of 1's
    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 1 and not visited[r][c]:
                component = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 1 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append(component)

    # Process each component
    for comp in components:
        if not comp:
            continue
        rs = [pos[0] for pos in comp]
        min_r = min(rs)
        max_r = max(rs)
        height = max_r - min_r + 1

        # Relative cells
        rel_cells = [[] for _ in range(height)]
        for pr, pc in comp:
            rel = pr - min_r
            rel_cells[rel].append(pc)

        # Widths: number of unique c's per rel
        widths = [len(set(rel_cells[rel])) for rel in range(height)]

        # Compute k
        k = 0
        rel = height - 1
        while rel >= 0 and widths[rel] == 1:
            k += 1
            rel -= 1

        # Clear original
        for pr, pc in comp:
            output[pr][pc] = 8

        if k == height:
            # Special: vertical in one col
            col = comp[0][1]  # all same
            d = (rows - 1) - max_r
            for pr, pc in comp:
                new_r = pr + d
                output[new_r][pc] = 1
            output[red_row][col] = 8
        else:
            # Normal
            stopping_rel = height - k - 1
            d = (red_row - 1) - (min_r + stopping_rel)
            for pr, pc in comp:
                new_r = pr + d
                if 0 <= new_r < rows:
                    output[new_r][pc] = 1

    return output

import copy
from collections import deque

def solve_271(I):
    if not I or not I[0]:
        return I
    h = len(I)
    w = len(I[0])
    out = copy.deepcopy(I)
    q = deque()
    # Collect all border cells that are 0
    for r in range(h):
        for c in range(w):
            if out[r][c] == 0 and (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                out[r][c] = 2
                q.append((r, c))
    # Directions for 4-connectivity
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # BFS to flood fill with 2
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0:
                out[nr][nc] = 2
                q.append((nr, nc))
    # Fill remaining 0s with 5
    for r in range(h):
        for c in range(w):
            if out[r][c] == 0:
                out[r][c] = 5
    return out

from collections import deque

def solve_272(I):
    if not I:
        return []
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    visited = [[False] * cols for _ in range(rows)]
    q = deque()
    # Add border 0's
    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1) and I[r][c] == 0 and not visited[r][c]:
                visited[r][c] = True
                q.append((r, c))
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Flood external 0's
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nr, nc))
    # Find and fill internal components
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 0 and not visited[r][c]:
                # New component
                component = []
                comp_q = deque()
                comp_q.append((r, c))
                visited[r][c] = True
                while comp_q:
                    cr, cc = comp_q.popleft()
                    component.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            comp_q.append((nr, nc))
                size = len(component)
                if size == 1:
                    fill_color = 5
                elif size == 2:
                    fill_color = 7
                else:
                    fill_color = 0  # Default, though not needed
                for pr, pc in component:
                    output[pr][pc] = fill_color
    return output

import numpy as np

def solve_273(I):
    if not I or not I[0]:
        return I
    I = np.array(I)
    rows, cols = I.shape

    # Check if horizontal bands
    is_horizontal = True
    for r in range(rows):
        non_zero_colors = set(I[r][I[r] != 0])
        if len(non_zero_colors) > 1:
            is_horizontal = False
            break

    if is_horizontal:
        # Process horizontal bands
        r = 0
        while r < rows:
            non_zero = I[r][I[r] != 0]
            if len(non_zero) == 0:
                r += 1
                continue
            color = np.unique(non_zero)[0]
            r_start = r
            while r < rows:
                curr_non_zero = I[r][I[r] != 0]
                if len(curr_non_zero) > 0 and np.all(curr_non_zero == color):
                    r += 1
                else:
                    break
            r_end = r - 1
            band_height = r_end - r_start + 1
            band_width = cols
            extend_vertical = band_height < band_width
            # Find and extend 0s
            for rr in range(r_start, r_end + 1):
                for c in range(cols):
                    if I[rr, c] == 0:
                        if extend_vertical:
                            I[r_start:r_end + 1, c] = 0
                        else:
                            I[rr, :] = 0
    else:
        # Process vertical bands
        c = 0
        while c < cols:
            non_zero = I[:, c][I[:, c] != 0]
            if len(non_zero) == 0:
                c += 1
                continue
            color = np.unique(non_zero)[0]
            c_start = c
            while c < cols:
                curr_non_zero = I[:, c][I[:, c] != 0]
                if len(curr_non_zero) > 0 and np.all(curr_non_zero == color):
                    c += 1
                else:
                    break
            c_end = c - 1
            band_width = c_end - c_start + 1
            band_height = rows
            extend_horizontal = band_width < band_height
            # Find and extend 0s
            for cc in range(c_start, c_end + 1):
                for rr in range(rows):
                    if I[rr, cc] == 0:
                        if extend_horizontal:
                            I[rr, c_start:c_end + 1] = 0
                        else:
                            I[:, cc] = 0

    return I.tolist()

import numpy as np

def solve_274(I):
    I = np.array(I)
    rows, cols = I.shape
    # Find grey row
    grey_row = next(i for i in range(rows) if np.all(I[i] == 5))
    # Find colors and their columns
    color_cols = {}
    for c in range(cols):
        for r in range(rows):
            if I[r, c] != 0 and I[r, c] != 5:
                color = I[r, c]
                if color not in color_cols:
                    color_cols[color] = c
                elif color_cols[color] != c:
                    raise ValueError("Multiple columns for color")
    colors = list(color_cols.keys())
    # Compute deltas
    heights = {}
    for color in colors:
        col = color_cols[color]
        # Above
        above_rows = [r for r in range(grey_row) if I[r, col] == color]
        h_above = (max(above_rows) - min(above_rows) + 1) if above_rows else 0
        # Below
        below_rows = [r for r in range(grey_row + 1, rows) if I[r, col] == color]
        h_below = (max(below_rows) - min(below_rows) + 1) if below_rows else 0
        delta = h_below - h_above
        heights[color] = delta
    # Find winner
    max_delta = max(heights.values())
    winners = [c for c, d in heights.items() if d == max_delta]
    win_color = winners[0]
    # Return 2x2 of win_color
    return [[win_color, win_color], [win_color, win_color]]

def solve_275(I):
    n = len(I)
    if n == 0:
        return I
    max_d = n // 2 - 1
    # Get input colors for each depth
    colors = [I[d][d] for d in range(max_d + 1)]
    # Create output I
    output = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            depth = min(i, j, n - 1 - i, n - 1 - j)
            output[i][j] = colors[max_d - depth]
    return output

def solve_276(I):
    h = len(I)
    w = len(I[0])
    half = h // 2
    out_h = h + 1
    out_w = w + 1
    output = [[0 for _ in range(out_w)] for _ in range(out_h)]
    
    # Place top half with 9 appended to each row
    for i in range(half):
        for j in range(w):
            output[i][j] = I[i][j]
        output[i][w] = 9
    
    # Separator row of 9's
    for j in range(out_w):
        output[half][j] = 9
    
    # Place bottom half with 9 prepended to each row
    for i in range(half):
        output[half + 1 + i][0] = 9
        for j in range(w):
            output[half + 1 + i][j + 1] = I[half + i][j]
    
    return output

import copy
from collections import deque

def solve_277(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])

    # DFS to find connected components
    def dfs(r, c, color, visited, component):
        stack = [(r, c)]
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            component.append((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and I[nx][ny] == color and (nx, ny) not in visited:
                    stack.append((nx, ny))

    # Flood fill external 0's from borders
    visited_0 = set()
    queue = deque()
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and I[i][j] == 0:
                queue.append((i, j))
                visited_0.add((i, j))
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and I[nx][ny] == 0 and (nx, ny) not in visited_0:
                visited_0.add((nx, ny))
                queue.append((nx, ny))

    # Create output
    output = copy.deepcopy(I)

    # Find and fill internal hole components
    visited_internal = set()
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 0 and (i, j) not in visited_0 and (i, j) not in visited_internal:
                comp = []
                dfs(i, j, 0, visited_internal, comp)
                size = len(comp)
                fill_color = 7 if size % 2 == 1 else 2
                for x, y in comp:
                    output[x][y] = fill_color

    return output

from collections import deque

def solve_278(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    max_size = 0
    best_comp = None
    B = None
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j]:
                color = I[i][j]
                comp = []
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    r, c = q.popleft()
                    comp.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                size = len(comp)
                if size > max_size:
                    max_size = size
                    best_comp = comp
                    B = color
    if best_comp is None:
        return []
    min_r = min(r for r, c in best_comp)
    max_r = max(r for r, c in best_comp)
    min_c = min(c for r, c in best_comp)
    max_c = max(c for r, c in best_comp)
    K = max_r - min_r + 1
    M = max_c - min_c + 1
    output = [[B] * M for _ in range(K)]
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if I[r][c] != B:
                F = I[r][c]
                sub_r = r - min_r
                sub_c = c - min_c
                for jj in range(M):
                    output[sub_r][jj] = F
                for ii in range(K):
                    output[ii][sub_c] = F
    return output

import math

def solve_279(I):
    if not I or not I[0]:
        return []

    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]

    # Compute core centroid
    sum_r, sum_c, count = 0, 0, 0
    for r in range(rows):
        for c in range(cols):
            if I[r][c] in (4, 7):
                sum_r += r
                sum_c += c
                count += 1
    if count == 0:
        return output
    core_r = sum_r / count
    core_c = sum_c / count

    # Find components
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and I[r][c] not in (0, 4, 7):
                color = I[r][c]
                pos = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    pos.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            stack.append((nr, nc))
                            visited[nr][nc] = True
                components.append((color, pos))

    # Compute angles and sort
    comp_list = []
    for old_color, pos in components:
        n = len(pos)
        avg_r = sum(rr for rr, cc in pos) / n
        avg_c = sum(cc for rr, cc in pos) / n
        delta_r = avg_r - core_r
        delta_c = avg_c - core_c
        angle = math.atan2(-delta_r, delta_c)
        comp_list.append((angle, old_color, pos))

    comp_list.sort(key=lambda x: x[0], reverse=True)

    n = len(comp_list)
    if n == 0:
        return output

    # Cycle colors
    for i in range(n):
        new_color = comp_list[(i - 1) % n][1]
        for r, c in comp_list[i][2]:
            output[r][c] = new_color

    return output

import numpy as np

def solve_280(I):
    I = np.array(I)
    rows, cols = I.shape
    visited = np.zeros_like(I, dtype=bool)
    spines = []
    for i in range(rows):
        for j in range(cols):
            if I[i, j] == 2 and not visited[i, j]:
                spine_rows = []
                col = j
                current_i = i
                # Collect upwards
                while current_i >= 0 and I[current_i, col] == 2 and not visited[current_i, col]:
                    visited[current_i, col] = True
                    spine_rows.append(current_i)
                    current_i -= 1
                # Reset to after the upwards
                current_i = i + 1
                # Collect downwards
                while current_i < rows and I[current_i, col] == 2 and not visited[current_i, col]:
                    visited[current_i, col] = True
                    spine_rows.append(current_i)
                    current_i += 1
                spine_rows = sorted(spine_rows)
                spines.append((col, min(spine_rows), max(spine_rows)))

    output = I.copy()
    for axis_col, min_r, max_r in spines:
        seed_color = None
        seed_side = None
        seed_count = 0
        seed_pos = None
        for r in range(min_r, max_r + 1):
            for j in range(cols):
                if j == axis_col:
                    continue
                c = I[r, j]
                if c != 0 and c != 2 and c != 4:
                    seed_color = c
                    seed_side = 'right' if j > axis_col else 'left'
                    seed_count += 1
                    seed_pos = (r, j)
        if seed_count != 1:
            continue
        body_side = 'left' if seed_side == 'right' else 'right'
        body_cond = lambda k: k < axis_col if body_side == 'left' else k > axis_col
        for r in range(min_r, max_r + 1):
            for j in range(cols):
                if I[r, j] == 4 and body_cond(j):
                    j_mirror = 2 * axis_col - j
                    if 0 <= j_mirror < cols and output[r, j_mirror] == 0:
                        output[r, j_mirror] = seed_color

    return output.tolist()

import numpy as np

def solve_281(I):
    I = np.array(I)
    # Find grey position
    grey_pos = np.argwhere(I == 5)
    grey_r, grey_c = grey_pos[0]
    # Find colorful positions
    color_pos = np.argwhere((I > 0) & (I != 5))
    min_r = np.min(color_pos[:, 0])
    max_r = np.max(color_pos[:, 0])
    min_c = np.min(color_pos[:, 1])
    max_c = np.max(color_pos[:, 1])
    # New min for copy
    new_min_r = grey_r - 1
    new_min_c = grey_c - 1
    # Copy each colorful cell
    for r, c in color_pos:
        rel_r = r - min_r
        rel_c = c - min_c
        new_r = new_min_r + rel_r
        new_c = new_min_c + rel_c
        I[new_r, new_c] = I[r, c]
    return I.tolist()

import numpy as np
from collections import Counter

def solve_282(I):
    g = np.array(I)
    tl = g[0:2, 0:2]
    tr = g[0:2, 3:5]
    bl = g[3:5, 0:2]
    br = g[3:5, 3:5]

    def to_tuple(arr):
        return tuple(tuple(row) for row in arr)

    quadrants = [to_tuple(tl), to_tuple(tr), to_tuple(bl), to_tuple(br)]
    counts = Counter(quadrants)
    for pat, cnt in counts.items():
        if cnt == 1:
            unique_pat = pat
            break
    return [list(row) for row in unique_pat]

import numpy as np

def solve_283(I):
    h = len(I)
    w = len(I[0])
    
    # Find sparse rows: rows with any 0
    sparse_rows = [i for i in range(h) if any(x == 0 for x in I[i])]
    
    # Group consecutive sparse rows into blocks
    blocks = []
    current = []
    for r in sparse_rows:
        if not current or r == current[-1] + 1:
            current.append(r)
        else:
            blocks.append(current)
            current = [r]
    if current:
        blocks.append(current)
    M = len(blocks)
    
    # Find runs of 0's in the first sparse row
    if not sparse_rows:
        return I  # No changes if no sparse rows
    sample_row = I[sparse_rows[0]]
    runs = []
    start = -1
    for c in range(w):
        if sample_row[c] == 0:
            if start == -1:
                start = c
        else:
            if start != -1:
                runs.append((start, c - 1))
                start = -1
    if start != -1:
        runs.append((start, w - 1))
    N = len(runs)
    
    # Now fill the I
    output = [row[:] for row in I]  # Copy I
    for bm in range(M):
        block_rows = blocks[bm]
        for bn in range(N):
            if bm == 0 or bm == M - 1 or bn == 0 or bn == N - 1:
                color = 2
            else:
                color = 3
            min_c, max_c = runs[bn]
            for r in block_rows:
                for c in range(min_c, max_c + 1):
                    output[r][c] = color
    
    return output

from collections import Counter

def solve_284(I):
    if not I or not I[0]:
        return []

    # Count frequencies
    flat = [cell for row in I for cell in row]
    counts = Counter(flat)

    # Main color: max count
    main_color = max(counts, key=counts.get)

    # Non-main perfect squares
    non_main_perfect = {}
    for colr, cnt in counts.items():
        if colr != main_color and cnt > 0:
            s = int(cnt ** 0.5)
            if s * s == cnt:
                non_main_perfect[colr] = s

    if not non_main_perfect:
        return I  # Or handle, but assume there are

    # Max side
    max_side = max(non_main_perfect.values())

    # Large color: assume unique, take the min color if multiple
    large_colors = [c for c, s in non_main_perfect.items() if s == max_side]
    large_color = min(large_colors)  # To handle potential multiple, pick smallest color

    # Small: others
    small_list = [(c, s) for c, s in non_main_perfect.items() if s < max_side]
    small_list.sort(key=lambda x: x[1])  # Sort by side ascending

    # Dimensions
    height = max_side
    left_width = sum(s + 1 for _, s in small_list)
    total_width = left_width + max_side

    # Create output filled with main
    output = [[main_color for _ in range(total_width)] for _ in range(height)]

    # Place small
    current_col = 0
    for c, s in small_list:
        place_start_col = current_col
        place_start_row = height - s
        for r in range(place_start_row, height):
            for cc in range(place_start_col, place_start_col + s):
                output[r][cc] = c
        current_col += s + 1

    # Place large
    large_start_col = left_width
    for r in range(height):
        for cc in range(large_start_col, large_start_col + max_side):
            output[r][cc] = large_color

    return output

def solve_285(I):
    return I + list(reversed(I))

def solve_286(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Find the gray row
    gray_row = None
    for r in range(rows):
        if all(cell == 5 for cell in I[r]):
            gray_row = r
            break
    if gray_row is None:
        return I  # No gray row, return unchanged (though examples have one)
    
    # Copy the I
    output = [row[:] for row in I]
    
    # Process each colored cell
    for r in range(rows):
        for c in range(cols):
            color = I[r][c]
            if color not in (1, 2):
                continue
            if r < gray_row:  # Above
                if color == 2:  # Extend down to gray_row-1
                    for rr in range(r, gray_row):
                        output[rr][c] = color
                else:  # color == 1, extend up to 0
                    for rr in range(0, r + 1):
                        output[rr][c] = color
            elif r > gray_row:  # Below
                if color == 2:  # Extend up to gray_row+1
                    for rr in range(gray_row + 1, r + 1):
                        output[rr][c] = color
                else:  # color == 1, extend down to rows-1
                    for rr in range(r, rows):
                        output[rr][c] = color
    
    return output

def solve_287(I):
    rows = len(I)
    cols = len(I[0])
    color_info = {}
    for r in range(rows):
        for c in range(cols):
            val = I[r][c]
            if val != 0 and val != 5:
                if val not in color_info:
                    color_info[val] = {'min_r': r, 'max_r': r, 'min_c': c, 'max_c': c}
                else:
                    info = color_info[val]
                    info['min_r'] = min(info['min_r'], r)
                    info['max_r'] = max(info['max_r'], r)
                    info['min_c'] = min(info['min_c'], c)
                    info['max_c'] = max(info['max_c'], c)
    unique_colors = list(color_info.keys())
    N = len(unique_colors)
    is_vertical = False
    for col in unique_colors:
        info = color_info[col]
        h = info['max_r'] - info['min_r'] + 1
        w = info['max_c'] - info['min_c'] + 1
        if h > w:
            is_vertical = True
            break
    if is_vertical:
        unique_colors.sort(key=lambda col: color_info[col]['min_c'])
    else:
        unique_colors.sort(key=lambda col: color_info[col]['min_r'])
    output = [[0 for _ in range(N)] for _ in range(N)]
    if is_vertical:
        for i, col in enumerate(unique_colors):
            for r in range(N):
                output[r][i] = col
    else:
        for i, col in enumerate(unique_colors):
            for c in range(N):
                output[i][c] = col
    return output

import numpy as np

def solve_288(I):
    input_grid = np.array(I)
    C = np.max(input_grid)  # The non-zero color
    output = np.zeros((9, 9), dtype=int)
    for r in range(9):
        d1 = r // 3
        d0 = r % 3
        for c in range(9):
            e1 = c // 3
            e0 = c % 3
            if input_grid[d1, e1] == 0 and input_grid[d0, e0] == 0:
                output[r, c] = C
    return output.tolist()

def solve_289(I):
    if not I or not I[0]:
        return I

    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []

    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 7 and not visited[r][c]:
                color = I[r][c]
                component = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append((color, component))

    # Clear all original positions
    for color, component in components:
        for r, c in component:
            output[r][c] = 7

    # Set the moved positions
    for color, component in components:
        size = len(component)
        for r, c in component:
            new_r = r + size
            if 0 <= new_r < rows:
                output[new_r][c] = color

    return output

def solve_290(I):
    output = [row[:] for row in I]

    def rotate_cw(block):
        new_block = [[0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                new_block[i][j] = block[2 - j][i]
        return new_block

    # Get initial block from columns 0-2
    block0 = [[I[r][c] for c in range(3)] for r in range(3)]

    # Rotate to fill columns 4-6
    block1 = rotate_cw(block0)
    for r in range(3):
        for j in range(3):
            output[r][4 + j] = block1[r][j]

    # Rotate block1 to fill columns 8-10
    block2 = rotate_cw(block1)
    for r in range(3):
        for j in range(3):
            output[r][8 + j] = block2[r][j]

    return output

import copy

def solve_291(I):
    # Find min_r
    height = len(I)
    if height == 0:
        return I
    width = len(I[0])
    
    min_r = height
    max_r = -1
    for r in range(height):
        for c in range(width):
            if I[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
    
    if min_r == height or max_r - min_r != 2:
        # Assume it's always 3 consecutive rows; handle if needed
        return I
    
    upper = I[min_r][:]
    core = I[min_r + 1][:]
    lower = I[min_r + 2][:]
    
    core_index = min_r + 1
    
    output = copy.deepcopy(I)
    
    for r in range(height):
        offset = (r - core_index) % 3
        if offset == 0:
            output[r] = core[:]
        elif offset == 1:
            output[r] = lower[:]
        else:  # offset == 2
            output[r] = upper[:]
    
    return output

def solve_292(I):
    if not I or not I[0]:
        return I

    height = len(I)
    width = len(I[0])
    visited = [[False] * width for _ in range(height)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []

    def dfs(r, c):
        stack = [(r, c)]
        comp = []
        visited[r][c] = True
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc, I[cr][cc]))
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and not visited[nr][nc] and I[nr][nc] != 0:
                    visited[nr][nc] = True
                    stack.append((nr, nc))
        return comp

    for r in range(height):
        for c in range(width):
            if I[r][c] != 0 and not visited[r][c]:
                comp = dfs(r, c)
                components.append(comp)

    shape_positions = []
    for comp in components:
        min_r = min(rr for rr, cc, col in comp)
        min_c = min(cc for rr, cc, col in comp)
        rel = [(rr - min_r, cc - min_c, col) for rr, cc, col in comp]
        shape_positions.append((min_r, min_c, rel))

    shape_positions.sort(key=lambda x: x[0])

    min_cs = [min_c for min_r, min_c, rel in shape_positions]
    min_cs_rev = min_cs[::-1]

    new_grid = [[0] * width for _ in range(height)]

    for i, (min_r, _, rel) in enumerate(shape_positions):
        new_min_c = min_cs_rev[i]
        for dr, dc, col in rel:
            new_r = min_r + dr
            new_c = new_min_c + dc
            new_grid[new_r][new_c] = col

    return new_grid

import numpy as np

def solve_293(I):
    I = grid_lst
    height = len(I)
    if height == 0:
        return []
    width = len(I[0])

    visited = [[False] * width for _ in range(height)]
    components = []

    for r in range(height):
        for c in range(width):
            if I[r][c] != 0 and not visited[r][c]:
                minr, maxr = r, r
                minc, maxc = c, c
                count2 = 1 if I[r][c] == 2 else 0
                stack = [(r, c)]
                visited[r][c] = True
                area = 1  # count number of cells in component for area

                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < height and 0 <= nc < width and I[nr][nc] != 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                            minr = min(minr, nr)
                            maxr = max(maxr, nr)
                            minc = min(minc, nc)
                            maxc = max(maxc, nc)
                            if I[nr][nc] == 2:
                                count2 += 1
                            area += 1

                components.append({
                    'minr': minr, 'maxr': maxr, 'minc': minc, 'maxc': maxc,
                    'count2': count2, 'area': area
                })

    if not components:
        return []

    max_count2 = max(c['count2'] for c in components)
    candidates = [c for c in components if c['count2'] == max_count2]

    if len(candidates) > 1:
        max_area = max(c['area'] for c in candidates)
        candidates = [c for c in candidates if c['area'] == max_area]

    if len(candidates) > 1:
        min_minr = min(c['minr'] for c in candidates)
        candidates = [c for c in candidates if c['minr'] == min_minr]

    if len(candidates) > 1:
        min_minc = min(c['minc'] for c in candidates)
        candidates = [c for c in candidates if c['minc'] == min_minc]

    selected = candidates[0]

    out = []
    for r in range(selected['minr'], selected['maxr'] + 1):
        row = []
        for c in range(selected['minc'], selected['maxc'] + 1):
            row.append(I[r][c])
        out.append(row)

    return out

import numpy as np

def solve_294(I):
    I = np.array(I)
    rows, cols = I.shape
    # Find unique non-zero color
    colors = np.unique(I[I != 0])
    if len(colors) == 0:
        return I.tolist()
    C = colors[0]
    # Find positions
    pos = np.argwhere(I == C)
    # Find min_r, min_c
    min_r = pos[:, 0].min()
    min_c = pos[:, 1].min()
    # Relative positions
    rel_pos = set((r - min_r, c - min_c) for r, c in pos)
    # Create output I
    output = np.zeros_like(I)
    # For each super sr=0..2, sc=0..2
    for sr in range(3):
        for sc in range(3):
            if (sr, sc) in rel_pos:
                # Place the small shape in this block
                block_r_start = sr * 3
                block_c_start = sc * 3
                for rel_r, rel_c in rel_pos:
                    global_r = block_r_start + rel_r
                    global_c = block_c_start + rel_c
                    if 0 <= global_r < rows and 0 <= global_c < cols:
                        output[global_r, global_c] = C
    return output.tolist()

import numpy as np

def solve_295(I):
    I = np.array(I)
    rows, cols = I.shape
    visited = np.zeros_like(I, dtype=bool)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]  # 8-connectivity

    def dfs(r, c, component):
        stack = [(r, c)]
        visited[r, c] = True
        while stack:
            cr, cc = stack.pop()
            component.append((cr, cc))
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and I[nr, nc] == 8:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r, c] == 8 and not visited[r, c]:
                component = []
                dfs(r, c, component)
                components.append(component)

    output = np.copy(I)
    for comp in components:
        if not comp:
            continue
        rs = [pos[0] for pos in comp]
        cs = [pos[1] for pos in comp]
        min_r = min(rs)
        max_r = max(rs)
        min_c = min(cs)
        max_c = max(cs)
        for rr in range(min_r, max_r + 1):
            for cc in range(min_c, max_c + 1):
                if output[rr, cc] == 0:
                    output[rr, cc] = 2

    return output.tolist()

import numpy as np

def solve_296(I):
    I = np.array(I)
    rows, cols = I.shape
    # Find the gray cell position
    gray_positions = np.argwhere(I == 5)
    center_r, center_c = gray_positions[0]
    # Create output I initialized to 0
    output = np.zeros((rows, cols), dtype=int)
    # For each non-zero cell, compute new position and set color
    for r in range(rows):
        for c in range(cols):
            color = I[r, c]
            if color != 0:
                new_r = 2 * center_r - r
                new_c = 2 * center_c - c
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    output[new_r, new_c] = color
    return output.tolist()

def solve_297(I):
    if not I or not I[0]:
        return []

    rows = len(I)
    cols = len(I[0])

    def find_rectangles():
        visited = [[False] * cols for _ in range(rows)]
        rects = []
        for r in range(rows):
            for c in range(cols):
                if I[r][c] != 0 and not visited[r][c]:
                    color = I[r][c]
                    # Extend right
                    c_right = c
                    while c_right + 1 < cols and I[r][c_right + 1] == color and not visited[r][c_right + 1]:
                        c_right += 1
                    # Extend down
                    r_bottom = r
                    while True:
                        next_r = r_bottom + 1
                        if next_r >= rows:
                            break
                        good = True
                        for cc in range(c, c_right + 1):
                            if I[next_r][cc] != color or visited[next_r][cc]:
                                good = False
                                break
                        if not good:
                            break
                        r_bottom = next_r
                    # Mark visited
                    for rr in range(r, r_bottom + 1):
                        for cc in range(c, c_right + 1):
                            visited[rr][cc] = True
                    # Add rect (min_r, max_r, min_c, max_c, color)
                    rects.append((r, r_bottom, c, c_right, color))
        return rects

    rects = find_rectangles()

    h_set = set()
    v_set = set()
    for min_r, max_r, min_c, max_c, _ in rects:
        h_set.add(min_r)
        h_set.add(max_r + 1)
        v_set.add(min_c)
        v_set.add(max_c + 1)

    h_lines = sorted(h_set)
    v_lines = sorted(v_set)

    num_log_rows = len(h_lines) - 1
    num_log_cols = len(v_lines) - 1

    if num_log_rows == 0 or num_log_cols == 0:
        return []

    output = [[0] * num_log_cols for _ in range(num_log_rows)]

    for i in range(num_log_rows):
        pixel_r = h_lines[i]
        for j in range(num_log_cols):
            pixel_c = v_lines[j]
            if 0 <= pixel_r < rows and 0 <= pixel_c < cols:
                output[i][j] = I[pixel_r][pixel_c]

    return output

def solve_298(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    for i in range(rows):
        for j in range(cols):
            color = I[i][j]
            if color != 0:
                if color % 2 == 0:
                    border = color // 2
                else:
                    border = color * 2
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if di == 0 and dj == 0:
                                output[ni][nj] = color
                            else:
                                output[ni][nj] = border
    return output

def solve_299(I):
    if not I or not I[0]:
        return []
    n = len(I)
    # Assume square
    assert len(I[0]) == n
    # Find c (non-zero value)
    c = max(max(row) for row in I)
    # Count k
    k = sum(1 for row in I for cell in row if cell != 0)
    b = n * n - k
    side = n * b
    output = [[0] * side for _ in range(side)]
    current_big = 0
    remaining = k
    while remaining > 0:
        r = min(remaining, b)
        for small_r in range(n):
            out_r = current_big * n + small_r
            out_c = 0
            for _ in range(r):
                for small_c in range(n):
                    output[out_r][out_c + small_c] = I[small_r][small_c]
                out_c += n
        current_big += 1
        remaining -= r
    return output

def solve_300(I):
    if not I or not I[0]:
        return I
    output = [row[:] for row in I]
    rows = len(output)
    cols = len(output[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    changed = True
    while changed:
        changed = False
        to_remove = []
        for r in range(rows):
            for c in range(cols):
                if output[r][c] == 0:
                    continue
                color = output[r][c]
                neigh_count = 0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == color:
                        neigh_count += 1
                if neigh_count < 2:
                    to_remove.append((r, c))
        if to_remove:
            changed = True
            for r, c in to_remove:
                output[r][c] = 0
    return output

def solve_301(I):
    n = len(I)
    out_n = 3 * n
    out = [[0 for _ in range(out_n)] for _ in range(out_n)]
    for i in range(n):
        for j in range(n):
            c = I[i][j]
            for x in range(3):
                for y in range(3):
                    out[i * 3 + x][j * 3 + y] = c
    return out

import numpy as np

def solve_302(I):
    I = np.array(grid_lst)
    rows, cols = I.shape

    # Find gray positions (5)
    gray_pos = np.argwhere(I == 5)
    if len(gray_pos) != 4:
        return I.tolist()  # Assume exactly 4

    # Find top: min row
    top_idx = np.argmin(gray_pos[:, 0])
    top_row, top_col = gray_pos[top_idx]

    # Bottom: max row
    bottom_idx = np.argmax(gray_pos[:, 0])
    bottom_row, bottom_col = gray_pos[bottom_idx]

    # Left: min col
    left_idx = np.argmin(gray_pos[:, 1])
    left_row, left_col = gray_pos[left_idx]

    # Right: max col
    right_idx = np.argmax(gray_pos[:, 1])
    right_row, right_col = gray_pos[right_idx]

    # Compute large bounds
    large_top = top_row + 1
    large_bottom = bottom_row - 1
    large_left = left_col + 1
    large_right = right_col - 1

    # Find the color: first non-zero non-5
    non_zero = np.argwhere((I != 0) & (I != 5))
    if non_zero.size == 0:
        return I.tolist()
    color = I[non_zero[0, 0], non_zero[0, 1]]

    # Draw top row
    for c in range(large_left, large_right + 1):
        if I[large_top, c] == 0:
            I[large_top, c] = color

    # Draw bottom row
    for c in range(large_left, large_right + 1):
        if I[large_bottom, c] == 0:
            I[large_bottom, c] = color

    # Draw sides
    for r in range(large_top + 1, large_bottom):
        if I[r, large_left] == 0:
            I[r, large_left] = color
        if I[r, large_right] == 0:
            I[r, large_right] = color

    return I.tolist()

import numpy as np

def solve_303(I):
    I = np.array(I)
    h, w = I.shape

    # Find full line rows and line_color
    line_rows = []
    for i in range(h):
        if np.all(I[i] == I[i][0]) and I[i][0] != 0:
            line_rows.append(i)
    line_color = I[line_rows[0]][0]

    # Spacing d
    line_rows = sorted(line_rows)
    d = line_rows[1] - line_rows[0]
    period = 2 * d

    # Find motif positions
    motif = []
    for i in range(h):
        for j in range(w):
            c = I[i, j]
            if c != 0 and c != line_color:
                motif.append((i, j, c))

    min_r = min(i for i, j, c in motif)
    min_c = min(j for i, j, c in motif)

    rel_motif = [(i - min_r, j - min_c, c) for i, j, c in motif]

    # Row starts
    row_mod = min_r % period
    row_starts = list(range(row_mod, h, period))

    # Col starts
    col_mod = min_c % period
    col_starts = list(range(col_mod, w, period))

    # Copy I
    new_grid = I.copy()

    # Place replications
    for sr in row_starts:
        for sc in col_starts:
            for dr, dc, c in rel_motif:
                nr = sr + dr
                nc = sc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    new_grid[nr, nc] = c

    return new_grid.tolist()

from collections import deque

def solve_304(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def bfs(start_r, start_c):
        color = I[start_r][start_c]
        if color == 7:
            return []
        component = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return component

    verticals = []
    horizontals = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 7 and not visited[r][c]:
                comp = bfs(r, c)
                if len(comp) < 2:
                    continue
                rs = [pr for pr, _ in comp]
                cs = [pc for _, pc in comp]
                uniq_rs = set(rs)
                uniq_cs = set(cs)
                color = I[rs[0]][cs[0]]
                if len(uniq_rs) == 1:  # horizontal
                    row = rs[0]
                    sorted_cs = sorted(uniq_cs)
                    if sorted_cs[-1] - sorted_cs[0] + 1 == len(sorted_cs):
                        horizontals.append((row, color))
                elif len(uniq_cs) == 1:  # vertical
                    col = cs[0]
                    sorted_rs = sorted(uniq_rs)
                    if sorted_rs[-1] - sorted_rs[0] + 1 == len(sorted_rs):
                        verticals.append((col, color))

    output = [[7 for _ in range(cols)] for _ in range(rows)]
    for col, color in verticals:
        for r in range(rows):
            output[r][col] = color
    for row, color in horizontals:
        for c in range(cols):
            output[row][c] = color
    return output

import numpy as np

def find_components(I, target):
    rows, cols = I.shape
    visited = np.zeros((rows, cols), bool)
    components = []
    for i in range(rows):
        for j in range(cols):
            if I[i, j] == target and not visited[i, j]:
                comp = []
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and I[nr, nc] == target:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                components.append(comp)
    return components

def find_colored_components(I):
    rows, cols = I.shape
    visited = np.zeros((rows, cols), bool)
    components = []
    for i in range(rows):
        for j in range(cols):
            if I[i, j] != 0 and not visited[i, j]:
                color = I[i, j]
                comp = []
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and I[nr, nc] == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                components.append((color, comp))
    return components

def normalize(pos):
    if not pos:
        return frozenset()
    min_r = min(r for r, _ in pos)
    min_c = min(c for _, c in pos)
    return frozenset((r - min_r, c - min_c) for r, c in pos)

def solve_305(I):
    I = np.array(grid_lst)
    rows, cols = I.shape
    half = cols // 2
    left = I[:, :half]
    right = I[:, half:]
    holes = find_components(left, 0)
    colored_comps = find_colored_components(right)
    output = left.copy()
    for hole_pos in holes:
        shape = normalize(hole_pos)
        for col, comp_pos in colored_comps:
            if normalize(comp_pos) == shape:
                for r, c in hole_pos:
                    output[r, c] = col
                break
    return output.tolist()

import copy

def solve_306(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Find non-zero positions
    non_zeros = [(r, c) for r in range(rows) for c in range(cols) if I[r][c] != 0]
    if not non_zeros:
        return [row[:] for row in I]
    
    rs = [r for r, c in non_zeros]
    cs = [c for r, c in non_zeros]
    min_r = min(rs)
    max_r = max(rs)
    min_c = min(cs)
    max_c = max(cs)
    
    orig_h = max_r - min_r + 1
    orig_w = max_c - min_c + 1
    
    # Available spaces
    avail_above = min_r
    avail_below = rows - 1 - max_r
    avail_left = min_c
    avail_right = cols - 1 - max_c
    
    # Extension sizes
    ext_h_above = min(avail_above, orig_h)
    ext_h_below = min(avail_below, orig_h)
    ext_w_left = min(avail_left, orig_w)
    ext_w_right = min(avail_right, orig_w)
    
    # Extension row/column ranges
    above_start_r = min_r - ext_h_above
    below_start_r = max_r + 1
    left_start_c = min_c - ext_w_left
    right_start_c = max_c + 1
    
    # Colors from original block
    top_left = I[min_r][min_c]
    top_right = I[min_r][max_c]
    bot_left = I[max_r][min_c]
    bot_right = I[max_r][max_c]
    
    # Copy I
    output = copy.deepcopy(I)
    
    # Fill above left
    for r in range(above_start_r, min_r):
        for c in range(left_start_c, min_c):
            output[r][c] = bot_right
    
    # Fill above right
    for r in range(above_start_r, min_r):
        for c in range(right_start_c, right_start_c + ext_w_right):
            output[r][c] = bot_left
    
    # Fill below left
    for r in range(below_start_r, below_start_r + ext_h_below):
        for c in range(left_start_c, min_c):
            output[r][c] = top_right
    
    # Fill below right
    for r in range(below_start_r, below_start_r + ext_h_below):
        for c in range(right_start_c, right_start_c + ext_w_right):
            output[r][c] = top_left
    
    return output

def solve_307(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    # Identify partial rows
    is_partial = [any(cell != 5 for cell in row) for row in I]
    # Find sections: list of (start, end) inclusive
    sections = []
    i = 0
    while i < rows:
        if is_partial[i]:
            start = i
            while i < rows and is_partial[i]:
                i += 1
            end = i - 1
            sections.append((start, end))
        else:
            i += 1
    s = len(sections)
    if s == 0:
        return I
    selected_sec = [0, (s - 1) // 2, s - 1]
    # Find gaps from first partial row
    if not sections:
        return I
    partial_row_idx = sections[0][0]
    partial_row = I[partial_row_idx]
    gaps = []
    j = 0
    while j < cols:
        if partial_row[j] == 0:
            start = j
            while j < cols and partial_row[j] == 0:
                j += 1
            end = j - 1
            gaps.append((start, end))
        else:
            j += 1
    g = len(gaps)
    if g == 0:
        return I
    selected_gap = [0, (g - 1) // 2, g - 1]
    # Copy I
    output = [row[:] for row in I]
    # Fill
    for k in range(3):
        sec_idx = selected_sec[k]
        gap_idx = selected_gap[k]
        color = k + 1
        start_r, end_r = sections[sec_idx]
        start_c, end_c = gaps[gap_idx]
        for r in range(start_r, end_r + 1):
            for c in range(start_c, end_c + 1):
                output[r][c] = color
    return output

import numpy as np

def solve_308(I):
    g = np.array(I)
    rows, cols = g.shape

    # Find frame bounding box
    frame_pos = np.argwhere(g == 5)
    min_r = frame_pos[:, 0].min()
    max_r = frame_pos[:, 0].max()
    min_c = frame_pos[:, 1].min()
    max_c = frame_pos[:, 1].max()

    # Interior bounds
    int_min_r = min_r + 1
    int_max_r = max_r - 1
    int_min_c = min_c + 1
    int_max_c = max_c - 1
    h = int_max_r - int_min_r + 1
    w = int_max_c - int_min_c + 1
    half_h = h // 2
    half_w = w // 2

    # Find colored cells
    colored = []
    for r in range(rows):
        for c in range(cols):
            colr = g[r, c]
            if colr != 0 and colr != 5:
                colored.append((r, c, colr))

    (r1, c1, color1), (r2, c2, color2) = colored

    if r1 == r2:
        r = r1
        # Sort by column
        if c1 > c2:
            c1, c2 = c2, c1
            color1, color2 = color2, color1
        left_color = color1
        right_color = color2
        if r < min_r:
            side = 'top'
        elif r > max_r:
            side = 'bottom'
        else:
            raise ValueError("Invalid side")
        if side == 'top':
            top_left_color = left_color
            top_right_color = right_color
            bot_left_color = right_color
            bot_right_color = left_color
        else:  # bottom
            bot_left_color = left_color
            bot_right_color = right_color
            top_left_color = right_color
            top_right_color = left_color
    elif c1 == c2:
        c = c1
        # Sort by row
        if r1 > r2:
            r1, r2 = r2, r1
            color1, color2 = color2, color1
        top_color = color1
        bot_color = color2
        if c < min_c:
            side = 'left'
        elif c > max_c:
            side = 'right'
        else:
            raise ValueError("Invalid side")
        if side == 'left':
            top_left_color = top_color
            bot_left_color = bot_color
            top_right_color = bot_color
            bot_right_color = top_color
        else:  # right
            top_right_color = top_color
            bot_right_color = bot_color
            top_left_color = bot_color
            bot_left_color = top_color
    else:
        raise ValueError("Colored cells not on same row or column")

    # Fill top half
    for rr in range(int_min_r, int_min_r + half_h):
        for cc in range(int_min_c, int_min_c + half_w):
            g[rr, cc] = top_left_color
        for cc in range(int_min_c + half_w, int_min_c + w):
            g[rr, cc] = top_right_color

    # Fill bottom half
    for rr in range(int_min_r + half_h, int_min_r + h):
        for cc in range(int_min_c, int_min_c + half_w):
            g[rr, cc] = bot_left_color
        for cc in range(int_min_c + half_w, int_min_c + w):
            g[rr, cc] = bot_right_color

    return g.tolist()

def solve_309(I):
    if not I or not I[0]:
        return []
    h = len(I) // 2
    w = len(I[0])
    output = [[0 for _ in range(w)] for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if I[i][j] == 0 and I[i + h][j] == 0:
                output[i][j] = 2
    return output

def solve_310(I):
    if not I or not I[0]:
        return I
    height = len(I)
    width = len(I[0])
    min_r = height
    max_r = -1
    min_c = width
    max_c = -1
    for r in range(height):
        for c in range(width):
            if I[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if min_r == height:
        return I  # no structure
    inner_top_r = min_r + 1
    inner_bottom_r = max_r - 1
    inner_left_c = min_c + 1
    inner_right_c = max_c - 1
    # Collect colors
    top_left = I[inner_top_r][inner_left_c]
    top_right = I[inner_top_r][inner_right_c]
    bottom_left = I[inner_bottom_r][inner_left_c]
    bottom_right = I[inner_bottom_r][inner_right_c]
    # Set inner to 0
    I[inner_top_r][inner_left_c] = 0
    I[inner_top_r][inner_right_c] = 0
    I[inner_bottom_r][inner_left_c] = 0
    I[inner_bottom_r][inner_right_c] = 0
    # Outer positions
    outer_top_r = min_r - 1
    outer_bottom_r = max_r + 1
    outer_left_c = min_c - 1
    outer_right_c = max_c + 1
    # Set outer colors (180 rotation)
    I[outer_top_r][outer_left_c] = bottom_right
    I[outer_top_r][outer_right_c] = bottom_left
    I[outer_bottom_r][outer_left_c] = top_right
    I[outer_bottom_r][outer_right_c] = top_left
    return I

def solve_311(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    count = [0] * 10
    for r in range(rows):
        for c in range(cols):
            color = I[r][c]
            count[color] += 1
    max_count = max(count)
    C = count.index(max_count)
    output = [row[:] for row in I]
    for r in range(rows):
        for c in range(cols):
            if output[r][c] != C:
                output[r][c] = 5
    return output

from collections import deque

def solve_312(I):
    if not I or not I[0]:
        return I
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 5 and not visited[r][c]:
                q = deque([(r, c)])
                visited[r][c] = True
                min_r, max_r = r, r
                min_c, max_c = c, c
                while q:
                    x, y = q.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and I[nx][ny] == 5:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                            min_r = min(min_r, nx)
                            max_r = max(max_r, nx)
                            min_c = min(min_c, ny)
                            max_c = max(max_c, ny)
                # Add above
                if min_r > 0:
                    add_r = min_r - 1
                    if min_c > 0:
                        output[add_r][min_c - 1] = 1
                    if max_c + 1 < cols:
                        output[add_r][max_c + 1] = 2
                # Add below
                if max_r + 1 < rows:
                    add_r = max_r + 1
                    if min_c > 0:
                        output[add_r][min_c - 1] = 3
                    if max_c + 1 < cols:
                        output[add_r][max_c + 1] = 4
    return output

def solve_313(I):
    if not I or not I[0]:
        return I

    rows = len(I)
    cols = len(I[0])

    # Find accent color: any color not 0 or 5
    colors = set()
    for row in I:
        for cell in row:
            if cell != 0 and cell != 5:
                colors.add(cell)

    if len(colors) != 1:
        return I  # Assume one accent color per examples

    accent = list(colors)[0]

    # Find horizontal rows: all cells ==5 or ==accent
    horiz_rows = []
    for r in range(rows):
        if all(cell == 5 or cell == accent for cell in I[r]):
            horiz_rows.append(r)

    # Find vertical columns: all cells ==5 or ==accent
    vert_cols = []
    for c in range(cols):
        if all(I[r][c] == 5 or I[r][c] == accent for r in range(rows)):
            vert_cols.append(c)

    # Create output
    output = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            is_horiz = r in horiz_rows
            is_vert = c in vert_cols
            if is_horiz and is_vert:
                output[r][c] = accent
            elif is_horiz or is_vert:
                output[r][c] = 5

    return output

def solve_314(I):
    if not I or not I[0]:
        return I
    h = sum(1 for row in I if any(x != 0 for x in row))
    out = []
    for row in I:
        l = len(row)
        shifted = row[h:l] + row[h:h + h]
        new_row = row + shifted
        out.append(new_row)
    return out

import numpy as np

def solve_315(I):
    I = np.array(grid_lst)
    n = I.shape[0]  # assume square
    large = np.zeros((n * n, n * n), dtype=int)
    for i in range(n):
        for j in range(n):
            c = I[i, j]
            for k in range(n):
                for l in range(n):
                    if I[k, l] == c:
                        large[n * i + k, n * j + l] = c
    return large.tolist()

def solve_316(I):
    output = [row[:] for row in I]
    for r in range(len(I)):
        non_zeros = [(c, I[r][c]) for c in range(len(I[r])) if I[r][c] != 0]
        if len(non_zeros) == 1:
            start_col, color = non_zeros[0]
            for cc in range(start_col, len(I[r])):
                if (cc - start_col) % 2 == 0:
                    output[r][cc] = color
                else:
                    output[r][cc] = 5
    return output

from collections import Counter

def solve_317(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Flatten the I to count frequencies
    flat = [cell for row in I for cell in row]
    color_count = Counter(flat)
    
    # Background is the most common color
    background = color_count.most_common(1)[0][0]
    
    # Create output I
    output = [row[:] for row in I]
    
    # Process each column
    for c in range(cols):
        s = I[0][c]
        if s == background:
            continue  # No seed in this column
        
        # Transform the column
        for r in range(rows):
            current = I[r][c]
            if current == background:
                output[r][c] = s
            elif current == s:
                output[r][c] = background
            else:
                output[r][c] = s
    
    return output

def solve_318(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]

    # Border color B
    B = I[0][0]

    # Find all columns that are entirely B
    full_B_cols = []
    for c in range(cols):
        if all(I[r][c] == B for r in range(rows)):
            full_B_cols.append(c)

    # Assuming three: left (0), sep, right (cols-1)
    if len(full_B_cols) != 3:
        return I  # Unexpected, return unchanged
    sep = full_B_cols[1]  # The middle one

    L_start = 1
    L_end = sep - 1
    R_start = sep + 1
    R_end = cols - 2
    background = 1  # Hardcoded as per examples

    # Inner rows
    for i in range(1, rows - 1):
        # Find segments in left
        j = L_start
        while j <= L_end:
            if I[i][j] == background:
                j += 1
                continue
            # Start of segment
            S = j
            C = I[i][j]
            while j <= L_end and I[i][j] == C:
                j += 1
            E = j - 1
            # Now process segment S to E, color C
            left_off = S - L_start
            right_off = L_end - E
            length = E - S + 1
            if left_off == 1 and right_off == 1:
                # Stretch
                start_r = R_start + 1
                end_r = R_end - 1
                for k in range(start_r, end_r + 1):
                    output[i][k] = C
            elif left_off == 1:
                # Left align
                start_r = R_start + 1
                end_r = start_r + length - 1
                if end_r <= R_end:
                    for k in range(start_r, end_r + 1):
                        output[i][k] = C
            elif right_off == 1:
                # Right align
                end_r = R_end - 1
                start_r = end_r - length + 1
                if start_r >= R_start:
                    for k in range(start_r, end_r + 1):
                        output[i][k] = C

    return output

def solve_319(I):
    I = [row[:] for row in grid_lst]  # Copy the I
    # Find the chain
    chain = []
    i = 0
    n = len(I)
    m = len(I[0]) if I else 0
    while i < min(n, m) and I[i][i] != 0 and I[i][i] != 1:
        chain.append(I[i][i])
        i += 1
    k = len(chain)
    if k == 0:
        return I
    # Frame top-left
    row_start = i
    col_start = i
    # Find row_end
    r = row_start
    while r < n and I[r][col_start] == 1:
        r += 1
    row_end = r - 1
    # Find col_end
    c = col_start
    while c < m and I[row_start][c] == 1:
        c += 1
    col_end = c - 1
    # Inner area
    i_row_start = row_start + 1
    i_row_end = row_end - 1
    i_col_start = col_start + 1
    i_col_end = col_end - 1
    # Inner dimensions
    h = i_row_end - i_row_start + 1 if i_row_end >= i_row_start else 0
    w = i_col_end - i_col_start + 1 if i_col_end >= i_col_start else 0
    # Fill the inner area
    for row in range(i_row_start, i_row_end + 1):
        for col in range(i_col_start, i_col_end + 1):
            local_r = row - i_row_start
            local_c = col - i_col_start
            dist_top = local_r
            dist_bot = (h - 1) - local_r
            dist_left = local_c
            dist_right = (w - 1) - local_c
            d = min(dist_top, dist_bot, dist_left, dist_right)
            idx = min(d, k - 1)
            I[row][col] = chain[idx]
    return I

def solve_320(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    background = I[0][cols - 1]
    min_col = cols
    max_col = -1
    for c in range(cols):
        for r in range(rows):
            if I[r][c] != background:
                min_col = min(min_col, c)
                max_col = max(max_col, c)
    if max_col < 0:
        return [row[:] for row in I]
    w = max_col - min_col + 1
    standard = None
    for r in range(rows):
        has_non = any(I[r][c] != background for c in range(min_col, max_col + 1))
        if has_non:
            standard = [I[r][c] for c in range(min_col, max_col + 1)]
            break
    if standard is None:
        return [row[:] for row in I]
    output = [row[:] for row in I]
    rev_standard = standard[::-1]
    for r in range(rows):
        current = [I[r][c] for c in range(min_col, max_col + 1)]
        if current == rev_standard:
            start_c = min_col
            segment_length = w + 1
            if start_c + segment_length > cols:
                continue  # Safeguard, though not needed in examples
            segment = [I[r][c] for c in range(start_c, start_c + segment_length)]
            new_segment = [segment[-1]] + segment[:-1]
            for i in range(segment_length):
                output[r][start_c + i] = new_segment[i]
    return output

import numpy as np

def solve_321(I):
    g = np.array(I)
    height, width = g.shape

    # Find uniform row
    horiz_cross = None
    cross_color = None
    for r in range(height):
        if np.all(g[r] == g[r, 0]):
            horiz_cross = r
            cross_color = g[r, 0]
            break  # Assuming unique

    # Find uniform column with cross_color
    vert_cross = None
    for c in range(width):
        if np.all(g[:, c] == cross_color):
            vert_cross = c
            break  # Assuming unique

    # Function to get most frequent color
    def most_frequent(subgrid):
        if subgrid.size == 0:
            return 0
        counts = np.bincount(subgrid.ravel(), minlength=10)
        return np.argmax(counts)

    # Quadrants
    tl = g[0:horiz_cross, 0:vert_cross]
    tr = g[0:horiz_cross, vert_cross + 1 : width]
    bl = g[horiz_cross + 1 : height, 0:vert_cross]
    br = g[horiz_cross + 1 : height, vert_cross + 1 : width]

    tl_color = most_frequent(tl)
    tr_color = most_frequent(tr)
    bl_color = most_frequent(bl)
    br_color = most_frequent(br)

    output = [
        [tl_color, cross_color, tr_color],
        [cross_color, cross_color, cross_color],
        [bl_color, cross_color, br_color],
    ]
    return output

def solve_322(I):
    top = I[0:4]
    bottom = I[5:9]
    out = [[0 for _ in range(4)] for _ in range(4)]
    for r in range(4):
        for c in range(4):
            top_true = top[r][c] == 1
            bottom_true = bottom[r][c] == 2
            if top_true != bottom_true:
                out[r][c] = 3
    return out

def solve_323(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    seeds = []
    for r in range(rows):
        colored = [(c, I[r][c]) for c in range(cols) if I[r][c] != 0]
        if colored:
            c, color = colored[0]
            seeds.append((r, c, color))
    seeds.sort(key=lambda x: x[0])
    for i in range(len(seeds)):
        seed_r, seed_c, color = seeds[i]
        next_r = seeds[i+1][0] if i+1 < len(seeds) else rows
        target_r = next_r - 1
        max_c = cols - 1
        for cc in range(seed_c, max_c + 1):
            output[seed_r][cc] = color
        for rr in range(seed_r + 1, target_r + 1):
            output[rr][max_c] = color
    return output

from collections import defaultdict

def solve_324(I):
    I = grid_lst
    height = len(I)
    if height == 0:
        return []
    width = len(I[0])

    visited = [[False] * width for _ in range(height)]
    components = []

    for r in range(height):
        for c in range(width):
            if I[r][c] != 0 and not visited[r][c]:
                minr, maxr = r, r
                minc, maxc = c, c
                colors = set([I[r][c]])
                stack = [(r, c)]
                visited[r][c] = True
                area = 1

                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < height and 0 <= nc < width and I[nr][nc] != 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                            minr = min(minr, nr)
                            maxr = max(maxr, nr)
                            minc = min(minc, nc)
                            maxc = max(maxc, nc)
                            colors.add(I[nr][nc])
                            area += 1

                components.append({
                    'minr': minr, 'maxr': maxr, 'minc': minc, 'maxc': maxc,
                    'num_distinct': len(colors), 'area': area
                })

    if not components:
        return []

    max_distinct = max(c['num_distinct'] for c in components)
    candidates = [c for c in components if c['num_distinct'] == max_distinct]

    if len(candidates) > 1:
        max_area = max(c['area'] for c in candidates)
        candidates = [c for c in candidates if c['area'] == max_area]

    if len(candidates) > 1:
        min_minr = min(c['minr'] for c in candidates)
        candidates = [c for c in candidates if c['minr'] == min_minr]

    if len(candidates) > 1:
        min_minc = min(c['minc'] for c in candidates)
        candidates = [c for c in candidates if c['minc'] == min_minc]

    selected = candidates[0]

    out = []
    for r in range(selected['minr'], selected['maxr'] + 1):
        row = []
        for c in range(selected['minc'], selected['maxc'] + 1):
            row.append(I[r][c])
        out.append(row)

    return out

from collections import deque

def solve_325(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and not visited[r][c]:
                color = I[r][c]
                size = 0
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    x, y = q.popleft()
                    size += 1
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and I[nx][ny] == color:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                components.append((size, color))
    if not components:
        return []
    components.sort(key=lambda x: -x[0])
    max_h = components[0][0]
    num_comp = len(components)
    output = [[0] * num_comp for _ in range(max_h)]
    for i, (size, color) in enumerate(components):
        for rr in range(size):
            output[rr][i] = color
    return output

def solve_326(I):
    if not grid_lst or not grid_lst[0]:
        return []
    I = [row[:] for row in grid_lst]
    height = len(I)
    width = len(I[0])
    # Collect bar colors
    colors = []
    for c in range(width):
        val = I[0][c]
        if val == 0 or val == 8:
            continue
        is_bar = all(I[r][c] == val for r in range(height))
        if is_bar:
            colors.append(val)
    # Find groups
    groups = []
    c = 0
    while c < width:
        positions = [r for r in range(height) if I[r][c] == 8]
        if not positions:
            c += 1
            continue
        min_r = min(positions)
        max_r = max(positions)
        group_cols = [c]
        c += 1
        while c < width:
            pos_next = [r for r in range(height) if I[r][c] == 8]
            if not pos_next:
                break
            min_next = min(pos_next)
            max_next = max(pos_next)
            if min_next != min_r or max_next != max_r:
                break
            group_cols.append(c)
            c += 1
        groups.append((min_r, max_r, group_cols))
    # Create output
    output = [[0 for _ in range(width)] for _ in range(height)]
    for i, (min_r, max_r, cols) in enumerate(groups):
        if i >= len(colors):
            break  # In case more groups than colors, but examples match
        color = colors[i]
        for col in cols:
            for r in range(height):
                if I[r][col] == 8:
                    output[r][col] = color
    return output

import numpy as np

def solve_327(I):
    if not I or not I[0]:
        return I
    I = np.array(I)
    rows, cols = I.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    for i in range(rows):
        for j in range(cols):
            if I[i, j] == 2 and not visited[i, j]:
                component = []
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and I[nx, ny] == 2 and not visited[nx, ny]:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
                components.append(component)

    cols_with_8 = [j for i in range(rows) for j in range(cols) if I[i, j] == 8]
    min_col_8 = min(cols_with_8) if cols_with_8 else 0

    cols_with_1 = [j for i in range(rows) for j in range(cols) if I[i, j] == 1]
    max_col_1 = max(cols_with_1) if cols_with_1 else cols - 1

    output = I.copy()
    for comp in components:
        neighbors = set()
        for x, y in comp:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and I[nx, ny] != 2:
                    neighbors.add(I[nx, ny])
        if not neighbors:
            continue
        C = next(iter(neighbors))  # Assume consistent
        if C not in (1, 8):
            continue
        min_c = min(y for x, y in comp)
        max_c = max(y for x, y in comp)
        if C == 1:
            delta = max_col_1 - max_c
        else:
            delta = min_col_8 - min_c
        # Set old positions to C
        for x, y in comp:
            output[x, y] = C
        # Set new positions to 2
        for x, y in comp:
            new_y = y + delta
            if 0 <= new_y < cols:
                output[x, new_y] = 2
    return output.tolist()

def solve_328(I):
    if not I or not I[0]:
        return I
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    for r in range(rows):
        # Find rightmost purple (8)
        rightmost_purple = -1
        for c in range(cols):
            if I[r][c] == 8:
                rightmost_purple = c
        target_start = 0 if rightmost_purple == -1 else rightmost_purple + 1
        
        # Find the green block: min and max c with 3
        green_cols = [c for c in range(cols) if I[r][c] == 3]
        if not green_cols:
            continue
        min_c = min(green_cols)
        max_c = max(green_cols)
        length = max_c - min_c + 1
        # Assume contiguous, as in examples
        
        # Clear original
        for c in range(min_c, max_c + 1):
            output[r][c] = 0
        
        # Set new positions, if within bounds
        if target_start + length > cols:
            # Doesn't happen in examples, but to handle, perhaps truncate or something, but assume fits
            pass
        for c in range(target_start, target_start + length):
            if c < cols:
                output[r][c] = 3
    
    return output

def solve_329(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    count = [0] * 10
    for row in I:
        for cell in row:
            count[cell] += 1
    color_freq = [(i, count[i]) for i in range(10)]
    sorted_colors = sorted(color_freq, key=lambda x: x[1], reverse=True)
    kept = {sorted_colors[0][0], sorted_colors[1][0]}
    output = [row[:] for row in I]
    for i in range(rows):
        for j in range(cols):
            if output[i][j] not in kept:
                output[i][j] = 7
    return output

def solve_330(I):
    if not I or not I[0]:
        return I
    h = len(I)
    w = len(I[0])
    center_r = h // 2
    center_c = w // 2
    max_d = max(center_r, center_c)
    corner_colors = [None] * (max_d + 1)
    side_colors = [None] * (max_d + 1)
    for r in range(h):
        for c in range(w):
            col = I[r][c]
            if col == 0:
                continue
            dr = abs(r - center_r)
            dc = abs(c - center_c)
            d = max(dr, dc)
            if dr == d and dc == d:
                if corner_colors[d] is None:
                    corner_colors[d] = col
            elif dr == d or dc == d:
                if side_colors[d] is None:
                    side_colors[d] = col
    out = [[0] * w for _ in range(h)]
    for r in range(1, h, 2):
        for c in range(1, w, 2):
            dr = abs(r - center_r)
            dc = abs(c - center_c)
            d = max(dr, dc)
            if d > max_d:
                continue
            if dr == d and dc == d:
                if corner_colors[d] is not None:
                    out[r][c] = corner_colors[d]
            elif (dr == d or dc == d) and side_colors[d] is not None:
                out[r][c] = side_colors[d]
    return out

def solve_331(I):
    if not I or not I[0]:
        return I
    return [list(row) for row in zip(*I)]

import numpy as np

def solve_332(I):
    I = np.array(grid_lst)
    h, w = I.shape
    # Find uniform columns
    uniform_cols = {}
    for c in range(w):
        col_vals = I[:, c]
        if np.all(col_vals == col_vals[0]):
            color = col_vals[0]
            uniform_cols.setdefault(color, []).append(c)
    # Find uniform rows
    uniform_rows = {}
    for r in range(h):
        row_vals = I[r, :]
        if np.all(row_vals == row_vals[0]):
            color = row_vals[0]
            uniform_rows.setdefault(color, []).append(r)
    # Find L: color in both uniform rows and columns
    possible_L = set(uniform_cols.keys()) & set(uniform_rows.keys())
    assert len(possible_L) == 1
    L = list(possible_L)[0]
    # Get sorted line positions
    vline_cols = sorted(uniform_cols[L])
    n = len(vline_cols) + 1
    k = (w - len(vline_cols)) // n
    hline_rows = sorted(uniform_rows[L])
    assert len(hline_rows) == n - 1
    assert (h - len(hline_rows)) // n == k
    # Start positions for big cells
    start_cols = [0] + [c + 1 for c in vline_cols]
    start_rows = [0] + [r + 1 for r in hline_rows]
    # Initialize output
    output = [[0 for _ in range(n)] for _ in range(n)]
    for big_r in range(n):
        for big_c in range(n):
            sr = start_rows[big_r]
            sc = start_cols[big_c]
            sub = I[sr:sr + k, sc:sc + k]
            vals = set(sub.flatten())
            if len(vals) == 1:
                col = list(vals)[0]
                if col != L and col != 0:
                    output[big_r][n - 1 - big_c] = col
    return output

import numpy as np

def solve_333(I):
    if not I or not I[0]:
        return I
    I = np.array(I)
    rows, cols = I.shape
    bg = I[0, 0]

    # Find pattern positions (non-bg, non-0)
    pattern_pos = np.argwhere((I != bg) & (I != 0))
    if len(pattern_pos) == 0:
        return I.tolist()
    min_r_p, min_c_p = np.min(pattern_pos, axis=0)
    max_r_p, max_c_p = np.max(pattern_pos, axis=0)

    # Find target positions (0's)
    target_pos = np.argwhere(I == 0)
    if len(target_pos) == 0:
        return I.tolist()
    min_r_t, min_c_t = np.min(target_pos, axis=0)
    max_r_t, max_c_t = np.max(target_pos, axis=0)

    # Extract pattern subgrid
    pattern = I[min_r_p:max_r_p+1, min_c_p:max_c_p+1]

    # Rotate 180: reverse rows, then reverse each row
    rotated = np.flipud(np.fliplr(pattern))

    # Place rotated pattern onto target area
    I[min_r_t:max_r_t+1, min_c_t:max_c_t+1] = rotated

    return I.tolist()

def solve_334(I):
    height = len(I)
    width = len(I[0])

    # Find pink_start
    pink_start = None
    for r in range(height):
        if all(cell == 6 for cell in I[r]):
            pink_start = r
            break

    # Find structure cells
    structure_cells = [(rr, cc) for rr in range(pink_start) for cc in range(width) if I[rr][cc] == 5]

    rows = [rr for rr, cc in structure_cells]
    min_r = min(rows)
    max_r = max(rows)

    cols = [cc for rr, cc in structure_cells]
    min_c = min(cols)
    max_c = max(cols)
    w = max_c - min_c + 1
    original_start = min_c

    # Find top_h
    top_h = 0
    for r in range(min_r, max_r + 1):
        if all(I[r][c] == 5 for c in range(original_start, original_start + w)):
            top_h += 1
        else:
            break

    pillar_h = (max_r - min_r + 1) - top_h

    # Determine new_start and shift_dir
    if original_start == 0:
        new_start = width - w
        shift_dir = -1
    else:
        new_start = 0
        shift_dir = 1

    # Create output
    output = [row[:] for row in I]

    # Clear old structure
    for rr, cc in structure_cells:
        output[rr][cc] = 1

    # Place new structure
    for r in range(min_r, min_r + top_h):
        for c in range(new_start, new_start + w):
            output[r][c] = 5

    for i in range(pillar_h):
        r = min_r + top_h + i
        shift = shift_dir * i
        base = new_start + shift
        for rel in [0, 2, 4]:
            cc = base + rel
            if 0 <= cc < width:
                output[r][cc] = 5

    # Place brown
    if pillar_h > 0:
        lowest_i = pillar_h - 1
        lowest_base = new_start + shift_dir * lowest_i
        if shift_dir > 0:
            lowest_left = lowest_base + 0
            brown_start = lowest_left + 1
            brown_end = width - 1
        else:
            lowest_right = lowest_base + 4
            brown_start = 0
            brown_end = lowest_right - 1
        for c in range(max(0, brown_start), min(width - 1, brown_end) + 1):
            output[pink_start][c] = 9

    return output

import copy

def solve_335(I):
    output = copy.deepcopy(I)
    rows = len(I)
    if rows == 0:
        return output
    cols = len(I[0])

    # Find all candidate positions from vertical patterns
    candidates = set()
    for c in range(cols):
        for k in range(rows - 6):
            if (I[k][c] == 1 and I[k+1][c] == 1 and
                I[k+2][c] == 8 and I[k+3][c] == 8 and I[k+4][c] == 8 and
                I[k+5][c] == 1 and I[k+6][c] == 1 and
                (k == 0 or I[k-1][c] != 1) and
                (k+7 >= rows or I[k+7][c] != 1)):
                r = k + 3
                candidates.add((r, c))

    # For each candidate, check horizontal pattern
    for r, c in candidates:
        m = c - 3
        if m >= 0 and m + 6 < cols:
            if (I[r][m] == 1 and I[r][m+1] == 1 and
                I[r][m+2] == 8 and I[r][m+3] == 8 and I[r][m+4] == 8 and
                I[r][m+5] == 1 and I[r][m+6] == 1 and
                (m == 0 or I[r][m-1] != 1) and
                (m+7 >= cols or I[r][m+7] != 1)):
                output[r][c] = 4

    return output

def solve_336(I):
    I = [row[:] for row in I]  # copy
    rows = len(I)
    cols = len(I[0])

    # Find red (2)
    red_i, red_j = None, None
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 2:
                red_i, red_j = i, j
                break
        if red_i is not None:
            break

    if red_i is None:
        return I  # No red, no change

    # Find adjacent pink (6), 8-connected
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni = red_i + di
            nj = red_j + dj
            if 0 <= ni < rows and 0 <= nj < cols and I[ni][nj] == 6:
                neighbors.append((di, dj))

    # Assume exactly one
    if len(neighbors) != 1:
        return I  # If not, no change or handle differently, but per examples =1

    dr, dc = neighbors[0]
    ext_dr = -dr
    ext_dc = -dc

    # Move in extension direction
    cur_i = red_i + ext_dr
    cur_j = red_j + ext_dc
    while 0 <= cur_i < rows and 0 <= cur_j < cols and I[cur_i][cur_j] == 7:
        cur_i += ext_dr
        cur_j += ext_dc

    # Change the first non-7 to 7
    if 0 <= cur_i < rows and 0 <= cur_j < cols and I[cur_i][cur_j] != 7:
        I[cur_i][cur_j] = 7

    return I

def solve_337(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    
    # Find green bounds
    min_row_green = height
    max_row_green = -1
    min_col_green = width
    max_col_green = -1
    for r in range(height):
        for c in range(width):
            if I[r][c] == 3:
                min_row_green = min(min_row_green, r)
                max_row_green = max(max_row_green, r)
                min_col_green = min(min_col_green, c)
                max_col_green = max(max_col_green, c)
    
    # Find red bounds
    min_row_red = height
    max_row_red = -1
    min_col_red = width
    max_col_red = -1
    for r in range(height):
        for c in range(width):
            if I[r][c] == 2:
                min_row_red = min(min_row_red, r)
                max_row_red = max(max_row_red, r)
                min_col_red = min(min_col_red, c)
                max_col_red = max(max_col_red, c)
    
    # Compute shifts
    down_shift = (min_row_green + 1) - min_row_red
    right_shift = (min_col_green + 1) - min_col_red
    
    # Create output I
    output = [row[:] for row in I]
    
    # Move reds
    for r in range(height):
        for c in range(width):
            if I[r][c] == 2:
                output[r][c] = 0
                new_r = r + down_shift
                new_c = c + right_shift
                if 0 <= new_r < height and 0 <= new_c < width:
                    output[new_r][new_c] = 2
    
    return output

def solve_338(I):
    rows = len(I)
    cols = len(I[0])
    # Find positions of 2 and 3
    pos2 = pos3 = None
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 2:
                pos2 = (r, c)
            elif I[r][c] == 3:
                pos3 = (r, c)
    r2, c2 = pos2
    r3, c3 = pos3
    output = [row[:] for row in I]
    # Horizontal at r2
    c_min = min(c2, c3)
    c_max = max(c2, c3)
    for c in range(c_min, c_max + 1):
        if (r2, c) != pos2 and (r2, c) != pos3:
            output[r2][c] = 8
    # Vertical at c3
    r_min = min(r2, r3)
    r_max = max(r2, r3)
    for r in range(r_min, r_max + 1):
        if (r, c3) != pos2 and (r, c3) != pos3:
            output[r][c3] = 8
    return output

from collections import deque

def solve_339(I):
    if not I or not I[0]:
        return []
    
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and not visited[r][c]:
                color = I[r][c]
                size = 0
                min_c = cols
                queue = deque([(r, c)])
                visited[r][c] = True
                
                while queue:
                    cr, cc = queue.popleft()
                    size += 1
                    min_c = min(min_c, cc)
                    
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                
                components.append((size, min_c, color))
    
    if not components:
        return []
    
    max_size = max(size for size, _, _ in components)
    max_components = [(min_c, color) for size, min_c, color in components if size == max_size]
    max_components.sort()  # sort by min_c
    
    colors = [color for _, color in max_components]
    height = max_size
    width = len(colors)
    
    output = [[colors[j] for j in range(width)] for _ in range(height)]
    return output

import copy

def solve_340(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    
    # Copy the I
    output = copy.deepcopy(I)
    
    # Find horizontal groups in bottom row
    bottom_row = I[height - 1]
    color_to_horiz = {}
    i = 0
    while i < width:
        c = bottom_row[i]
        if c == 0:
            i += 1
            continue
        start = i
        while i < width and bottom_row[i] == c:
            i += 1
        length = i - start
        color_to_horiz[c] = (start, length)
    
    # Find vertical groups in right column, excluding bottom row
    right_col = [I[r][width - 1] for r in range(height - 1)]
    vertical_groups = []
    j = 0
    while j < height - 1:
        c = right_col[j]
        if c == 0:
            j += 1
            continue
        start = j
        while j < height - 1 and right_col[j] == c:
            j += 1
        group_height = j - start
        vertical_groups.append((c, start, group_height))
    
    # Apply transformations
    for c, start_r, h in vertical_groups:
        if c in color_to_horiz:
            start_c, len_c = color_to_horiz[c]
            for rr in range(start_r, start_r + h):
                for cc in range(start_c, start_c + len_c):
                    output[rr][cc] = c
    
    return output

def solve_341(I):
    return [row + row for row in I]

def solve_342(I):
    import sys
    rows = len(I)
    cols = len(I[0])
    
    # Find bounding box of red (2)
    min_r = sys.maxsize
    max_r = -sys.maxsize - 1
    min_c = sys.maxsize
    max_c = -sys.maxsize - 1
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 2:
                min_r = min(min_r, i)
                max_r = max(max_r, i)
                min_c = min(min_c, j)
                max_c = max(max_c, j)
    
    if min_r == sys.maxsize:
        return I  # No red, no change
    
    # Collect grey positions
    greys = []
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 5:
                greys.append((i, j))
    
    # Create new I
    new_grid = [row[:] for row in I]
    
    for r, c in greys:
        # Compute target_r
        if r < min_r:
            target_r = min_r - 1
        elif r > max_r:
            target_r = max_r + 1
        else:
            target_r = r
        
        # Compute target_c
        if c < min_c:
            target_c = min_c - 1
        elif c > max_c:
            target_c = max_c + 1
        else:
            target_c = c
        
        # Place at target
        new_grid[target_r][target_c] = 5
        
        # Remove from original if moved
        if target_r != r or target_c != c:
            new_grid[r][c] = 0
    
    return new_grid

from collections import deque

def solve_343(I):
    if not I or not I[0]:
        return I
    h = len(I)
    w = len(I[0])
    output = [row[:] for row in I]
    visited = [[False] * w for _ in range(h)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(h):
        for j in range(w):
            if I[i][j] == 0 and not visited[i][j]:
                component = []
                touches_boundary = False
                q = deque()
                q.append((i, j))
                visited[i][j] = True
                component.append((i, j))
                if i == 0 or i == h - 1 or j == 0 or j == w - 1:
                    touches_boundary = True
                while q:
                    r, c = q.popleft()
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and I[nr][nc] == 0:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                            component.append((nr, nc))
                            if nr == 0 or nr == h - 1 or nc == 0 or nc == w - 1:
                                touches_boundary = True
                if not touches_boundary:
                    for pr, pc in component:
                        output[pr][pc] = 1
    return output

import copy

def solve_344(I):
    I = copy.deepcopy(I)
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(r, c, component):
        stack = [(r, c)]
        visited[r][c] = True
        component.append((r, c))
        while stack:
            cr, cc = stack.pop()
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] != 8:
                    visited[nr][nc] = True
                    stack.append((nr, nc))
                    component.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 8 and not visited[r][c]:
                component = []
                dfs(r, c, component)
                if not component:
                    continue
                big_min_r = min(rr for rr, cc in component)
                big_max_r = max(rr for rr, cc in component)
                big_min_c = min(cc for rr, cc in component)
                big_max_c = max(cc for rr, cc in component)
                big_h = big_max_r - big_min_r + 1
                big_w = big_max_c - big_min_c + 1
                colored = [(rr, cc) for rr, cc in component if I[rr][cc] != 0]
                if not colored:
                    continue
                pat_min_r = min(rr for rr, cc in colored)
                pat_max_r = max(rr for rr, cc in colored)
                pat_min_c = min(cc for rr, cc in colored)
                pat_max_c = max(cc for rr, cc in colored)
                pat_h = pat_max_r - pat_min_r + 1
                pat_w = pat_max_c - pat_min_c + 1
                pattern = [[I[pat_min_r + i][pat_min_c + j] for j in range(pat_w)] for i in range(pat_h)]
                for i in range(big_h):
                    for j in range(big_w):
                        rr = big_min_r + i
                        cc = big_min_c + j
                        I[rr][cc] = pattern[i % pat_h][j % pat_w]
    return I

import numpy as np

def solve_345(I):
    small = np.array(I)
    flat = small.flatten()
    unique = set(flat)
    k = len(unique)
    n = small.shape[0]
    large = np.zeros((n * k, n * k), dtype=int)
    for i in range(k):
        for j in range(k):
            large[i * n : (i + 1) * n, j * n : (j + 1) * n] = small
    return large.tolist()

def solve_346(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    visited = set()
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and (r, c) not in visited:
                # Check if start: no predecessor
                pred_r, pred_c = r - 1, c - 1
                if 0 <= pred_r < rows and 0 <= pred_c < cols and I[pred_r][pred_c] != 0:
                    continue  # Has predecessor, skip
                # Start a chain
                index = 0
                curr_r, curr_c = r, c
                while True:
                    visited.add((curr_r, curr_c))
                    if index % 2 == 1:
                        output[curr_r][curr_c] = 4
                    # Next cell
                    next_r, next_c = curr_r + 1, curr_c + 1
                    if 0 <= next_r < rows and 0 <= next_c < cols and I[next_r][next_c] != 0:
                        curr_r, curr_c = next_r, next_c
                        index += 1
                    else:
                        break
    return output

from collections import deque

def solve_347(I):
    if not I or not I[0]:
        return []
    
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    corner_colors = {'tl': None, 'tr': None, 'bl': None, 'br': None}
    shapes = {
        'tl': {(0,0), (0,1), (1,0)},
        'tr': {(0,0), (0,1), (1,1)},
        'bl': {(0,0), (1,0), (1,1)},
        'br': {(0,1), (1,0), (1,1)}
    }
    
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and not visited[r][c]:
                color = I[r][c]
                positions = []
                queue = deque([(r, c)])
                visited[r][c] = True
                
                while queue:
                    cr, cc = queue.popleft()
                    positions.append((cr, cc))
                    
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                
                if len(positions) == 3:
                    min_r = min(pr for pr, _ in positions)
                    min_c = min(pc for _, pc in positions)
                    rel_pos = {(pr - min_r, pc - min_c) for pr, pc in positions}
                    
                    for corner, shape in shapes.items():
                        if rel_pos == shape:
                            corner_colors[corner] = color
                            break
    
    output = [[0] * 4 for _ in range(4)]
    
    # Fill TL
    cc = corner_colors['tl']
    if cc is not None:
        output[0][0] = cc
        output[0][1] = cc
        output[1][0] = cc
    
    # Fill TR
    cc = corner_colors['tr']
    if cc is not None:
        output[0][2] = cc
        output[0][3] = cc
        output[1][3] = cc
    
    # Fill BL
    cc = corner_colors['bl']
    if cc is not None:
        output[2][0] = cc
        output[3][0] = cc
        output[3][1] = cc
    
    # Fill BR
    cc = corner_colors['br']
    if cc is not None:
        output[2][3] = cc
        output[3][2] = cc
        output[3][3] = cc
    
    return output

def solve_348(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    bars = []
    for c in range(cols):
        r = 0
        while r < rows:
            if I[r][c] == 5:
                start = r
                while r < rows and I[r][c] == 5:
                    r += 1
                end = r - 1
                height = end - start + 1
                bars.append((height, c, start, end))
            else:
                r += 1
    if not bars:
        return output
    max_h = max(b[0] for b in bars)
    min_h = min(b[0] for b in bars)
    for h, c, start, end in bars:
        if h == max_h:
            color = 1
        elif h == min_h:
            color = 2
        else:
            continue
        for rr in range(start, end + 1):
            output[rr][c] = color
    return output

def solve_349(I):
    if not I:
        return I
    rows = len(I)
    cols = len(I[0])
    
    # Find red_row and L
    red_row = None
    L = 0
    for row in range(rows):
        if I[row] and I[row][0] == 2:
            l = 0
            while l < cols and I[row][l] == 2:
                l += 1
            L = l
            red_row = row
            break
    
    if red_row is None:
        return [row[:] for row in I]
    
    top_length = red_row + L
    output = [row[:] for row in I]
    
    for k in range(rows):
        length = top_length - k
        if length <= 0:
            continue
        color = 3 if k < red_row else 2 if k == red_row else 1
        for c in range(min(length, cols)):
            output[k][c] = color
    
    return output

def solve_350(I):
    if not I or not I[0]:
        return []

    rows = len(I)
    cols = len(I[0])
    hollows = []

    for i in range(rows - 3):
        for j in range(cols - 3):
            # Extract the 4x4 block
            block = [I[i + x][j + y] for x in range(4) for y in range(4)]
            # Border positions (all except inner 2x2)
            border = [block[x * 4 + y] for x in range(4) for y in range(4)
                      if not (1 <= x <= 2 and 1 <= y <= 2)]
            inner = [block[x * 4 + y] for x in range(1, 3) for y in range(1, 3)]

            border_colors = set(border)
            if len(border_colors) == 1 and 0 not in border_colors and all(v == 0 for v in inner):
                c = list(border_colors)[0]
                hollows.append({'min_r': i, 'min_c': j, 'color': c})

    if not hollows:
        return []

    # Compute overall bounding box spans
    min_r_all = min(h['min_r'] for h in hollows)
    max_r_all = max(h['min_r'] + 3 for h in hollows)
    min_c_all = min(h['min_c'] for h in hollows)
    max_c_all = max(h['min_c'] + 3 for h in hollows)
    height = max_r_all - min_r_all + 1
    width = max_c_all - min_c_all + 1

    vertical = height > width

    if vertical:
        hollows.sort(key=lambda h: (h['min_r'], h['min_c']))
    else:
        hollows.sort(key=lambda h: (h['min_c'], h['min_r']))

    def get_shape(color):
        return [
            [color, color, color, color],
            [color, 0, 0, color],
            [color, 0, 0, color],
            [color, color, color, color]
        ]

    if vertical:
        out = []
        for h in hollows:
            out.extend(get_shape(h['color']))
        return out
    else:
        num = len(hollows)
        out = [[0] * (4 * num) for _ in range(4)]
        for idx, h in enumerate(hollows):
            sh = get_shape(h['color'])
            for r in range(4):
                out[r][idx * 4:idx * 4 + 4] = sh[r]
        return out

import numpy as np

def most_frequent_non_zero(subgrid):
    if subgrid.size == 0:
        return 0
    counts = np.bincount(subgrid.ravel(), minlength=10)
    counts[0] = 0
    if np.max(counts) == 0:
        return 0
    return np.argmax(counts)

def solve_351(I):
    g = np.array(I)
    height, width = g.shape

    # Find uniform row
    horiz_cross = None
    cross_color = None
    for r in range(height):
        if np.all(g[r] == g[r, 0]):
            horiz_cross = r
            cross_color = g[r, 0]
            break

    # Find uniform column with cross_color
    vert_cross = None
    for c in range(width):
        if np.all(g[:, c] == cross_color):
            vert_cross = c
            break

    # Quadrants
    tl = g[0:horiz_cross, 0:vert_cross]
    tr = g[0:horiz_cross, vert_cross + 1 : width]
    bl = g[horiz_cross + 1 : height, 0:vert_cross]
    br = g[horiz_cross + 1 : height, vert_cross + 1 : width]

    tl_color = most_frequent_non_zero(tl)
    tr_color = most_frequent_non_zero(tr)
    bl_color = most_frequent_non_zero(bl)
    br_color = most_frequent_non_zero(br)

    size = horiz_cross  # assuming square quadrants
    output = np.zeros((size, size), dtype=int)

    for i in range(size):
        for j in range(size):
            if tl[i, j] == tl_color:
                output[i, j] = tl_color
            elif tr[i, j] == tr_color:
                output[i, j] = tr_color
            elif bl[i, j] == bl_color:
                output[i, j] = bl_color
            elif br[i, j] == br_color:
                output[i, j] = br_color
            else:
                output[i, j] = 0

    return output.tolist()

def solve_352(I):
    height = len(I)
    if height == 0:
        return I
    width = len(I[0])
    output = [row[:] for row in I]
    for r in range(height):
        for c in range(1, width - 1):
            if output[r][c] == 0 and output[r][c - 1] == 1 and output[r][c + 1] == 1:
                output[r][c] = 2
    return output

def solve_353(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    background = I[0][0]
    min_r = rows
    max_r = -1
    min_c = cols
    max_c = -1
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != background:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if max_r < 0:
        return []
    out_rows = max_r - min_r + 1
    out_cols = max_c - min_c + 1
    out = [[0 for _ in range(out_cols)] for _ in range(out_rows)]
    for i in range(out_rows):
        for j in range(out_cols):
            orig_r = min_r + i
            orig_c = min_c + j
            val = I[orig_r][orig_c]
            out[i][j] = 0 if val == background else val
    return out

import math

def solve_354(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    positions = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 8:
                positions.append((r, c))
    if not positions:
        return [row[:] for row in I]
    sum_r = sum(r for r, c in positions)
    count = len(positions)
    avg_r = sum_r / count
    center_r = (rows - 1) / 2
    delta = center_r - avg_r
    s_low = math.floor(delta)
    s_high = math.ceil(delta)
    dist_low = abs(s_low - delta)
    dist_high = abs(s_high - delta)
    candidates = []
    if dist_low <= dist_high + 1e-9:
        candidates.append(s_low)
    if dist_high <= dist_low + 1e-9:
        candidates.append(s_high)
    s = min(candidates, key=lambda x: abs(x))
    out = [[0] * cols for _ in range(rows)]
    for r, c in positions:
        new_r = r + s
        if 0 <= new_r < rows:
            out[new_r][c] = 2
    return out

def solve_355(I):
    color_map = {0: 2, 1: 4, 2: 3}
    output = []
    for row in I:
        for col, val in enumerate(row):
            if val == 5:
                color = color_map[col]
                output.append([color, color, color])
                break
    return output

def solve_356(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    if rows == 3 and cols % 3 == 0:
        is_horizontal = True
        num_patterns = cols // 3
    elif cols == 3 and rows % 3 == 0:
        is_horizontal = False
        num_patterns = rows // 3
    else:
        # Invalid, but per problem, assume valid
        return I
    
    max_count = -1
    best_sub = None
    
    for i in range(num_patterns):
        if is_horizontal:
            sub = [row[i*3 : (i+1)*3] for row in I]
        else:
            sub = [row[:] for row in I[i*3 : (i+1)*3]]  # Copy to avoid reference issues, though not necessary here
        
        count = sum(1 for row in sub for cell in row if cell != 0)
        
        if count > max_count:
            max_count = count
            best_sub = sub
    
    return best_sub

import numpy as np

def solve_357(I):
    I = np.array(I)
    h, w = I.shape
    output = I.copy()
    for top in range(h):
        for bot in range(top + 1, h):
            for left in range(w):
                for right in range(left + 1, w):
                    subgrid = I[top:bot + 1, left:right + 1]
                    if np.all(subgrid == 0):
                        can_extend_up = top > 0 and np.all(I[top - 1, left:right + 1] == 0)
                        can_extend_down = bot < h - 1 and np.all(I[bot + 1, left:right + 1] == 0)
                        can_extend_left = left > 0 and np.all(I[top:bot + 1, left - 1] == 0)
                        can_extend_right = right < w - 1 and np.all(I[top:bot + 1, right + 1] == 0)
                        if not (can_extend_up or can_extend_down or can_extend_left or can_extend_right):
                            output[top:bot + 1, left:right + 1] = 2
    return output.tolist()

def solve_358(I):
    # Copy the I
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    mid = rows // 2  # Middle row, for general case, but here 1
    
    # Find position of 2
    r, c = None, None
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 2:
                r, c = i, j
                break
        if r is not None:
            break
    
    if r is None:
        return output  # No 2, return as is
    
    # Remove the 2
    output[r][c] = 0
    
    # Compute left and right
    left = c - 1
    right = c + 1
    
    if r == mid:
        top = 0
        bot = rows - 1
        if left >= 0:
            output[top][left] = 3
            output[bot][left] = 8
        if right < cols:
            output[top][right] = 6
            output[bot][right] = 7
    elif r == 0:
        if left >= 0:
            output[mid][left] = 8
        if right < cols:
            output[mid][right] = 7
    elif r == rows - 1:
        if left >= 0:
            output[mid][left] = 3
        if right < cols:
            output[mid][right] = 6
    
    return output

def solve_359(I):
    if not I or not I[0]:
        return I
    
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    
    for r in range(rows):
        pos = [c for c in range(cols) if I[r][c] != 0]
        if len(pos) < 2:
            continue
        pos.sort()
        for i in range(len(pos) - 1):
            for c in range(pos[i] + 1, pos[i + 1]):
                output[r][c] = 2
    
    return output

import numpy as np

def solve_360(I):
    I = np.array(I)
    rows, cols = I.shape
    max_zeros = -1
    candidates = []

    # Backslash diagonals (\): constant col - row = k
    for k in range(-rows + 1, cols):
        diag = []
        for r in range(rows):
            c = r + k
            if 0 <= c < cols:
                diag.append((r, c))
        if diag:
            count = sum(1 for r, c in diag if I[r, c] == 0)
            length = len(diag)
            if count > max_zeros:
                max_zeros = count
                candidates = [diag]
            elif count == max_zeros:
                candidates.append(diag)

    # Forward slash diagonals (/): constant row + col = s
    for s in range(rows + cols - 1):
        diag = []
        for r in range(rows):
            c = s - r
            if 0 <= c < cols:
                diag.append((r, c))
        if diag:
            count = sum(1 for r, c in diag if I[r, c] == 0)
            length = len(diag)
            if count > max_zeros:
                max_zeros = count
                candidates = [diag]
            elif count == max_zeros:
                candidates.append(diag)

    # Filter by max length
    if candidates:
        max_len = max(len(d) for d in candidates)
        candidates = [d for d in candidates if len(d) == max_len]

    # Assume unique after filters; pick the first if multiple
    if not candidates:
        return I.tolist()
    chosen_diag = candidates[0]

    # Create output
    output = I.copy()
    for r, c in chosen_diag:
        if I[r, c] == 0:
            output[r, c] = 8

    return output.tolist()

def solve_361(I):
    if not I or not I[0]:
        return I

    height = len(I)
    width = len(I[0])
    visited = [[False] * width for _ in range(height)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []

    for r in range(height):
        for c in range(width):
            if I[r][c] != 0 and not visited[r][c]:
                color = I[r][c]
                positions = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    positions.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < height and 0 <= nc < width and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append((color, positions))

    if len(components) != 2:
        return I  # Assuming always 2, but safeguard

    # Sort by size
    components.sort(key=lambda x: len(x[1]))

    small_color, small_pos = components[0]
    large_color, large_pos = components[1]

    # Create output I
    out_grid = [row[:] for row in I]

    # Recolor large component
    for pr, pc in large_pos:
        out_grid[pr][pc] = small_color

    # Remove small component
    for pr, pc in small_pos:
        out_grid[pr][pc] = 0

    return out_grid

def solve_362(I):
    rows = len(I)
    cols = len(I[0])
    r, c, C = None, None, None
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != 7:
                r, c, C = i, j, I[i][j]
                break
        if r is not None:
            break
    base = [5, 2, 8, 9, 6, 1, 3, 4, 0]
    k = base.index(C)
    used = base[k:] + base[:k]
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            dist = abs(i - r) + abs(j - c)
            output[i][j] = used[dist % 9]
    return output

def solve_363(I):
    # Count non-zero cells
    K = sum(1 for row in I for cell in row if cell != 0)
    
    if K == 0:
        return []
    
    size = 3 * K
    output = [[0 for _ in range(size)] for _ in range(size)]
    
    for i in range(3):
        for j in range(3):
            color = I[i][j]
            for di in range(K):
                for dj in range(K):
                    output[i * K + di][j * K + dj] = color
    
    return output

import numpy as np

def solve_364(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    background = 7  # Based on examples
    
    # Map from k to color
    k_to_color = {}
    
    for r in range(height):
        for c in range(width):
            if I[r][c] != background:
                k = min(r, height - 1 - r, c, width - 1 - c)
                k_to_color[k] = I[r][c]
    
    # Create output
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    for r in range(height):
        for c in range(width):
            k = min(r, height - 1 - r, c, width - 1 - c)
            output[r][c] = k_to_color[k]
    
    return output

import math

def solve_365(I):
    if not I or not I[0]:
        return []

    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]

    # Compute core centroid of all inner cells (color != 0,1,2)
    sum_r, sum_c, count = 0, 0, 0
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and I[r][c] != 1 and I[r][c] != 2:
                sum_r += r
                sum_c += c
                count += 1
    if count == 0:
        return output
    core_r = sum_r / count
    core_c = sum_c / count

    # Find components
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and I[r][c] != 0 and I[r][c] != 1 and I[r][c] != 2:
                color = I[r][c]
                pos = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    pos.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            stack.append((nr, nc))
                            visited[nr][nc] = True
                components.append((color, pos))

    # Compute angles and sort descending
    comp_list = []
    for old_color, pos in components:
        n = len(pos)
        avg_r = sum(rr for rr, cc in pos) / n
        avg_c = sum(cc for rr, cc in pos) / n
        delta_r = avg_r - core_r
        delta_c = avg_c - core_c
        angle = math.atan2(-delta_r, delta_c)
        comp_list.append((angle, old_color, pos))

    comp_list.sort(key=lambda x: x[0], reverse=True)

    n = len(comp_list)
    if n == 0:
        return output

    # Cycle colors: each gets next in clockwise order
    for i in range(n):
        new_color = comp_list[(i + 1) % n][1]
        for r, c in comp_list[i][2]:
            output[r][c] = new_color

    return output

from collections import deque

def solve_366(I):
    I = [row[:] for row in I]
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def is_straight(component):
        if not component:
            return False
        rs = [r for r, c in component]
        cs = [c for r, c in component]
        return len(set(rs)) == 1 or len(set(cs)) == 1

    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 6 and (i, j) not in visited:
                component = []
                queue = deque([(i, j)])
                visited.add((i, j))
                while queue:
                    r, c = queue.popleft()
                    component.append((r, c))
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 6 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                size = len(component)
                straight = is_straight(component)
                if size == 2:
                    color = 9
                elif size == 3:
                    color = 2 if straight else 4
                elif size == 4:
                    color = 8
                elif size == 5:
                    color = 3
                elif size == 6:
                    color = 5
                else:
                    color = 6  # fallback, though not needed
                for r, c in component:
                    I[r][c] = color
    return I

def solve_367(I):
    if not I or not I[0]:
        return I
    n = len(I)
    out = [[0 for _ in range(n * n)] for _ in range(n * n)]
    for i in range(n):
        for j in range(n):
            if I[i][j] == 5:
                for di in range(n):
                    for dj in range(n):
                        out[i * n + di][j * n + dj] = I[di][dj]
    return out

from collections import defaultdict

def solve_368(I):
    rows = len(I)
    cols = len(I[0])
    color_pos = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            col = I[r][c]
            if col != 0:
                color_pos[col].append((r, c))
    centers = []
    for col, poss in color_pos.items():
        if len(poss) == 1:
            r, c = poss[0]
            centers.append((r, c, col))
    output = [row[:] for row in I]
    for cen_r, cen_c, d in centers:
        aligned = []
        # same row, left
        for cc in range(cen_c):
            if I[cen_r][cc] != 0:
                aligned.append((cen_r, cc, I[cen_r][cc]))
        # same row, right
        for cc in range(cen_c + 1, cols):
            if I[cen_r][cc] != 0:
                aligned.append((cen_r, cc, I[cen_r][cc]))
        # same col, up
        for rr in range(cen_r):
            if I[rr][cen_c] != 0:
                aligned.append((rr, cen_c, I[rr][cen_c]))
        # same col, down
        for rr in range(cen_r + 1, rows):
            if I[rr][cen_c] != 0:
                aligned.append((rr, cen_c, I[rr][cen_c]))
        if not aligned:
            continue
        colors = set(a[2] for a in aligned)
        if len(colors) != 1:
            continue
        C = list(colors)[0]
        if C == d:
            continue
        for ar, ac, _ in aligned:
            if ar == cen_r:
                if ac < cen_c:
                    new_r, new_c = cen_r, cen_c - 1
                else:
                    new_r, new_c = cen_r, cen_c + 1
            else:
                if ar < cen_r:
                    new_r, new_c = cen_r - 1, cen_c
                else:
                    new_r, new_c = cen_r + 1, cen_c
            output[new_r][new_c] = C
            output[ar][ac] = 0
    return output

import numpy as np

def solve_369(I):
    I = np.array(I)
    n = I.shape[0]  # Assuming square I, here 9
    max_ones = -1
    best_sub = None
    for i in range(n - 2):
        for j in range(n - 2):
            sub = I[i:i+3, j:j+3]
            if np.all(sub != 0):
                ones = np.sum(sub == 1)
                if ones > max_ones:
                    max_ones = ones
                    best_sub = sub.copy()
    return best_sub.tolist()

def solve_370(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    output = [row[:] for row in I]  # copy the I

    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 2 and not visited[i][j]:
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                component.append((i, j))
                size = 1
                while stack:
                    r, c = stack.pop()
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == 2:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                            component.append((nr, nc))
                            size += 1
                if size >= 4:
                    for pr, pc in component:
                        output[pr][pc] = 6
    return output

import copy

def solve_371(I):
    if not I or not I[0]:
        return I
    
    output = copy.deepcopy(I)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows = len(I)
    cols = len(I[0])
    
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 2:
                count = 0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 2:
                        count += 1
                if count == 0:
                    output[r][c] = 1
    
    return output

import collections

def solve_372(I):
    # Define block positions: (row_start, row_end, col_start, col_end)
    blocks = [
        (1, 3, 1, 2),  # top left
        (1, 3, 4, 5),  # top middle
        (1, 3, 7, 8),  # top right
        (5, 7, 1, 2),  # bottom left
        (5, 7, 4, 5),  # bottom middle
        (5, 7, 7, 8),  # bottom right
    ]
    
    modes = []
    for rs, re, cs, ce in blocks:
        vals = []
        for i in range(rs, re + 1):
            for j in range(cs, ce + 1):
                vals.append(I[i][j])
        counter = collections.Counter(vals)
        mode = counter.most_common(1)[0][0]
        modes.append(mode)
    
    # Build output I
    output = [
        [0, 0, 0, 0, 0],
        [0, modes[0], modes[1], modes[2], 0],
        [0, modes[3], modes[4], modes[5], 0],
        [0, 0, 0, 0, 0]
    ]
    return output

def solve_373(I):
    if not I or not I[0]:
        return I
    height = len(I)
    width = len(I[0])
    code_row = -1
    for r in range(height):
        if any(cell == 7 for cell in I[r]):
            code_row = r
            break
    if code_row == -1:
        return I
    signals = [c for c in range(width) if I[code_row][c] == 7]
    current_color = 6
    current_row = code_row + 2
    while current_row < height and signals:
        new_signals = []
        for i in range(len(signals) - 1):
            c1 = signals[i]
            c2 = signals[i + 1]
            if c2 - c1 == 2:
                mid = c1 + 1
                I[current_row][mid] = current_color
                new_signals.append(mid)
        signals = new_signals
        current_color = 13 - current_color
        current_row += 2
    return I

import numpy as np
from collections import defaultdict

def solve_374(I):
    I = np.array(grid_lst)
    rows, cols = I.shape
    groups = defaultdict(list)
    for r in range(rows):
        pos = np.where(I[r] == 4)[0]
        if len(pos) == 2:
            left, right = sorted(pos)
            groups[(left, right)].append(r)
    for (left, right), row_list in groups.items():
        sorted_rows = sorted(row_list)
        for i in range(len(sorted_rows) - 1):
            r1 = sorted_rows[i]
            r2 = sorted_rows[i + 1]
            for rr in range(r1 + 1, r2):
                for cc in range(left + 1, right):
                    I[rr, cc] = 2
    return I.tolist()

def solve_375(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    base_row = rows - 1
    # Find left_arm and right_arm
    left_arm = None
    right_arm = None
    for c in range(cols):
        if I[base_row][c] == 5:
            if left_arm is None:
                left_arm = c
            right_arm = c
    # Find top_arm
    top_arm = rows
    for r in range(rows):
        if I[r][left_arm] == 5 or I[r][right_arm] == 5:
            top_arm = r
            break
    # Find purple_start
    purple_start = rows
    for r in range(top_arm, rows):
        inner_filled = False
        for c in range(left_arm + 1, right_arm):
            if I[r][c] != 0:
                inner_filled = True
                break
        if inner_filled:
            purple_start = r
            break
    # Compute empty_height
    empty_height = purple_start - top_arm
    # Get fill_color
    fill_color = 0
    if purple_start < rows:
        for c in range(left_arm + 1, right_arm):
            if I[purple_start][c] != 0:
                fill_color = I[purple_start][c]
                break
    # Create output 3x3
    out = [[0 for _ in range(3)] for _ in range(3)]
    # Directions for rows: 1 for L->R, -1 for R->L
    directions = [1, -1, 1]
    count = 0
    for row in range(3):
        if count >= empty_height:
            break
        dir = directions[row]
        if dir == 1:
            col_range = range(0, 3)
        else:
            col_range = range(2, -1, -1)
        for col in col_range:
            if count < empty_height:
                out[row][col] = fill_color
                count += 1
            else:
                break
    return out

def solve_376(I):
    new_grid = [[2 if cell == 6 else cell for cell in row] for row in I]
    return new_grid

import numpy as np

def solve_377(I):
    I = np.array(I)
    rows, cols = I.shape
    output = I.copy()
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(r, c):
        component = []
        stack = [(r, c)]
        visited[r, c] = True
        while stack:
            cr, cc = stack.pop()
            component.append((cr, cc))
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and I[nr, nc] == 2:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        return component

    for i in range(rows):
        for j in range(cols):
            if I[i, j] == 2 and not visited[i, j]:
                comp = dfs(i, j)
                if len(comp) >= 2:
                    rs = [p[0] for p in comp]
                    cs = [p[1] for p in comp]
                    min_r = min(rs)
                    max_r = max(rs)
                    min_c = min(cs)
                    max_c = max(cs)
                    exp_min_r = max(0, min_r - 1)
                    exp_max_r = min(rows - 1, max_r + 1)
                    exp_min_c = max(0, min_c - 1)
                    exp_max_c = min(cols - 1, max_c + 1)
                    for ii in range(exp_min_r, exp_max_r + 1):
                        for jj in range(exp_min_c, exp_max_c + 1):
                            if I[ii, jj] == 0:
                                output[ii, jj] = 3
    return output.tolist()

import numpy as np
from collections import deque

def solve_378(I):
    if not I or not I[0]:
        return I
    
    I = np.array(I)
    h, w = I.shape
    
    # Find connected components of 1's
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for i in range(h):
        for j in range(w):
            if I[i, j] == 1 and not visited[i, j]:
                component = []
                q = deque([(i, j)])
                visited[i, j] = True
                while q:
                    r, c = q.popleft()
                    component.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and I[nr, nc] == 1 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                components.append(component)
    
    # Flood fill reachable 9's from borders
    temp = I.copy()
    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    # Add border cells if 9
    for i in range(h):
        if temp[i, 0] == 9:
            q.append((i, 0))
            visited[i, 0] = True
        if temp[i, w - 1] == 9:
            q.append((i, w - 1))
            visited[i, w - 1] = True
    for j in range(w):
        if temp[0, j] == 9:
            q.append((0, j))
            visited[0, j] = True
        if temp[h - 1, j] == 9:
            q.append((h - 1, j))
            visited[h - 1, j] = True
    
    while q:
        r, c = q.popleft()
        temp[r, c] = -1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and temp[nr, nc] == 9 and not visited[nr, nc]:
                visited[nr, nc] = True
                q.append((nr, nc))
    
    # Find hole cells (remaining 9's)
    holes = [(i, j) for i in range(h) for j in range(w) if temp[i, j] == 9]
    
    # Map positions to component indices
    comp_index = {}
    for idx, comp in enumerate(components):
        for pos in comp:
            comp_index[pos] = idx
    
    # Collect components to change
    to_change = set()
    for r, c in holes:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and I[nr, nc] == 1:
                pos = (nr, nc)
                if pos in comp_index:
                    to_change.add(comp_index[pos])
    
    # Create output
    output = I.copy()
    for idx in to_change:
        for r, c in components[idx]:
            output[r, c] = 8
    
    return output.tolist()

import numpy as np

def solve_379(I):
    I = np.array(I)
    rows, cols = I.shape
    output = I.copy()
    visited = np.zeros_like(I, dtype=bool)
    
    for i in range(rows):
        for j in range(cols):
            color = I[i, j]
            if not visited[i, j] and color != 0 and color != 7 and color != 8:
                component = []
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    x, y = stack.pop()
                    component.append((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and I[nx, ny] == color:
                            visited[nx, ny] = True
                            stack.append((nx, ny))
                size = len(component)
                # Clear original positions
                for x, y in component:
                    output[x, y] = 7
                # Place at new positions
                for x, y in component:
                    nx = x - size
                    output[nx, y] = color
    return output.tolist()

import numpy as np

def solve_380(I):
    if not I or not I[0]:
        return I
    I = np.array(I)
    rows, cols = I.shape
    
    # Find original bounding box excluding 8
    shape_cells = np.argwhere((I != 0) & (I != 8))
    if len(shape_cells) == 0:
        return I.tolist()
    min_r = np.min(shape_cells[:, 0])
    max_r = np.max(shape_cells[:, 0])
    min_c = np.min(shape_cells[:, 1])
    max_c = np.max(shape_cells[:, 1])
    
    # Find purple position
    purple = np.argwhere(I == 8)
    pr, pc = purple[0]
    
    # New bounding box
    new_min_r = min(min_r, pr)
    new_max_r = max(max_r, pr)
    new_min_c = min(min_c, pc)
    new_max_c = max(max_c, pc)
    
    # Determine colors
    border_color = I[min_r, min_c]
    interior_color = I[min_r + 1, min_c + 1] if min_r + 1 <= max_r and min_c + 1 <= max_c else border_color
    
    # Create output
    output = np.zeros((rows, cols), dtype=int)
    
    # Fill with interior color
    for r in range(new_min_r, new_max_r + 1):
        for c in range(new_min_c, new_max_c + 1):
            output[r, c] = interior_color
    
    # Set perimeter to border color
    # Top
    for c in range(new_min_c, new_max_c + 1):
        output[new_min_r, c] = border_color
    # Bottom
    for c in range(new_min_c, new_max_c + 1):
        output[new_max_r, c] = border_color
    # Left
    for r in range(new_min_r + 1, new_max_r):
        output[r, new_min_c] = border_color
    # Right
    for r in range(new_min_r + 1, new_max_r):
        output[r, new_max_c] = border_color
    
    return output.tolist()

def solve_381(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    output = [[0 for _ in range(width)] for _ in range(height)]
    
    pattern = [
        (-1, -1, 5), (-1, 0, 1), (-1, 1, 5),
        (0, -1, 1), (0, 0, 0), (0, 1, 1),
        (1, -1, 5), (1, 0, 1), (1, 1, 5)
    ]
    
    for r in range(height):
        for c in range(width):
            if I[r][c] != 0:
                for dr, dc, val in pattern:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        output[nr][nc] = val
    
    return output

from collections import deque
import copy

def solve_382(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = copy.deepcopy(I)
    visited = set()

    # DFS to find connected components of color 5
    def dfs(r, c, component):
        stack = [(r, c)]
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            component.append((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and I[nx][ny] == 5 and (nx, ny) not in visited:
                    stack.append((nx, ny))

    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 5 and (i, j) not in visited:
                component = []
                dfs(i, j, component)
                if not component:
                    continue
                # Find bounding box
                min_r = min(x for x, y in component)
                max_r = max(x for x, y in component)
                min_c = min(y for x, y in component)
                max_c = max(y for x, y in component)
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                # Color each cell in component based on position in bounding box
                for x, y in component:
                    local_r = x - min_r
                    local_c = y - min_c
                    is_corner = (local_r == 0 or local_r == h - 1) and (local_c == 0 or local_c == w - 1)
                    is_border = (local_r == 0 or local_r == h - 1 or local_c == 0 or local_c == w - 1)
                    if is_corner:
                        output[x][y] = 1
                    elif is_border:
                        output[x][y] = 4
                    else:
                        output[x][y] = 2

    return output

def solve_383(I):
    if not I or not I[0]:
        return I

    height = len(I)
    width = len(I[0])
    output = [row[:] for row in I]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def flood_fill(start_r, start_c, col):
        if output[start_r][start_c] != 6:
            return
        stack = [(start_r, start_c)]
        while stack:
            cr, cc = stack.pop()
            if output[cr][cc] == 6:
                output[cr][cc] = col
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        stack.append((nr, nc))

    # Find sources
    sources = []
    for r in range(height):
        for c in range(width):
            if output[r][c] != 7 and output[r][c] != 6:
                sources.append((r, c, output[r][c]))

    # Paint for each source
    for r, c, col in sources:
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and output[nr][nc] == 6:
                flood_fill(nr, nc, col)

    # Set sources to 7
    for r, c, col in sources:
        output[r][c] = 7

    return output

from collections import deque

def solve_384(I):
    if not I:
        return []
    n = len(I)
    if n == 0:
        return []
    # Assume square
    assert all(len(row) == n for row in I)

    # Find components
    visited = [[False] * n for _ in range(n)]
    corner_colors = {1: 8, 2: 8, 3: 8, 4: 8}  # Default to 8 if missing
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(n):
        for j in range(n):
            if I[i][j] != 8 and not visited[i][j]:
                # Flood fill to find component
                color = I[i][j]
                component = []
                queue = deque([(i, j)])
                visited[i][j] = True
                while queue:
                    x, y = queue.popleft()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < n and 0 <= ny < n and not visited[nx][ny] and I[nx][ny] == color:
                            visited[nx][ny] = True
                            queue.append((nx, ny))

                # Determine orientation
                min_r = min(r for r, c in component)
                min_c = min(c for r, c in component)
                rel = sorted([(r - min_r, c - min_c) for r, c in component])
                types = {
                    tuple(sorted([(0,0),(0,1),(1,0)])): 1,
                    tuple(sorted([(0,0),(1,0),(1,1)])): 2,
                    tuple(sorted([(0,1),(1,0),(1,1)])): 3,
                    tuple(sorted([(0,0),(0,1),(1,1)])): 4
                }
                shape_tuple = tuple(rel)
                if shape_tuple in types:
                    typ = types[shape_tuple]
                    corner_colors[typ] = color

    # Create output I all 8
    output = [[8] * n for _ in range(n)]

    # k = (n - 1) // 2
    k = (n - 1) // 2

    # Top-left: type 1
    color = corner_colors[1]
    # Top row, left k cells
    for c in range(k):
        output[0][c] = color
    # Left column, rows 1 to k-1
    for r in range(1, k):
        output[r][0] = color

    # Top-right: type 4
    color = corner_colors[4]
    # Top row, right k cells
    for c in range(n - k, n):
        output[0][c] = color
    # Right column, rows 1 to k-1
    for r in range(1, k):
        output[r][n - 1] = color

    # Bottom-left: type 2
    color = corner_colors[2]
    # Bottom row, left k cells
    for c in range(k):
        output[n - 1][c] = color
    # Left column, rows n-k to n-2
    for r in range(n - k, n - 1):
        output[r][0] = color

    # Bottom-right: type 3
    color = corner_colors[3]
    # Bottom row, right k cells
    for c in range(n - k, n):
        output[n - 1][c] = color
    # Right column, rows n-k to n-2
    for r in range(n - k, n - 1):
        output[r][n - 1] = color

    return output

import numpy as np
from collections import deque

def solve_385(I):
    if not I or not I[0]:
        return I
    
    I = np.array(I)
    rows, cols = I.shape
    output = I.copy()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = np.zeros_like(I, dtype=bool)
    
    for r in range(rows):
        for c in range(cols):
            if I[r, c] != 8 and not visited[r, c]:
                # Find component
                component = []
                colors = set()
                queue = deque([(r, c)])
                visited[r, c] = True
                component.append((r, c))
                if I[r, c] != 0:
                    colors.add(I[r, c])
                
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr, nc] != 8 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                            component.append((nr, nc))
                            if I[nr, nc] != 0:
                                colors.add(I[nr, nc])
                
                # If exactly two colors, color the component
                if len(colors) == 2:
                    color_list = list(colors)
                    color_map = {color_list[0]: color_list[1], color_list[1]: color_list[0]}
                    
                    # Assigned colors
                    color_assigned = np.zeros_like(I)
                    q = deque()
                    for pr, pc in component:
                        if I[pr, pc] != 0:
                            color_assigned[pr, pc] = I[pr, pc]
                            q.append((pr, pc))
                    
                    # BFS to color
                    while q:
                        cr, cc = q.popleft()
                        current_color = color_assigned[cr, cc]
                        expected_color = color_map[current_color]
                        for dr, dc in directions:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < rows and 0 <= nc < cols and I[nr, nc] != 8:
                                if color_assigned[nr, nc] == 0:
                                    color_assigned[nr, nc] = expected_color
                                    q.append((nr, nc))
                                elif color_assigned[nr, nc] != expected_color:
                                    # Assume no conflict
                                    pass
                    
                    # Apply to output
                    for pr, pc in component:
                        if I[pr, pc] == 0:
                            output[pr, pc] = color_assigned[pr, pc]
    
    return output.tolist()

def solve_386(I):
    n = len(I)
    output = [row[:] for row in I]
    for r in range(n):
        for c in range(n):
            if I[r][c] == 4:
                pr = n - 1 - r
                pc = n - 1 - c
                output[r][c] = I[pr][pc]
    return output

import numpy as np

def solve_387(I):
    I = np.array(grid_lst)
    rows, cols = I.shape

    # Find bottom_row
    bottom_row = max(i for i in range(rows) for j in range(cols) if I[i, j] != 0)

    # Find L and R in bottom_row
    non_zero_cols = [j for j in range(cols) if I[bottom_row, j] != 0]
    L = min(non_zero_cols)
    R = max(non_zero_cols)

    # Check contiguous
    if any(I[bottom_row, j] == 0 for j in range(L, R + 1)):
        return grid_lst  # Return original if not contiguous

    # Find outer color
    outer = I[bottom_row, L]

    # Find H from left
    H = 0
    while H <= (R - L + 1) // 2 and I[bottom_row, L + H] == outer:
        H += 1

    # Check right
    h_right = 0
    while h_right < H and I[bottom_row, R - h_right] == outer:
        h_right += 1
    if h_right != H:
        return grid_lst  # Return original if asymmetric

    # Find inner
    if H == 0:
        inner = outer
    else:
        inner = I[bottom_row, L + H]
        if any(I[bottom_row, j] != inner for j in range(L + H, R - H + 1)):
            return grid_lst  # Return original if not uniform

    # Upper row
    upper_row = bottom_row - 1
    if upper_row < 0:
        return grid_lst

    # Check upper row
    for j in range(cols):
        if L + H <= j <= R - H:
            if I[upper_row, j] != outer:
                return grid_lst
        else:
            if I[upper_row, j] != 0:
                return grid_lst

    # Add H rows above upper_row
    for level in range(1, H + 1):
        add_row = upper_row - level
        if add_row < 0:
            continue
        offset = H - level
        left_pos = L + offset
        right_pos = R - offset
        if 0 <= left_pos < cols and 0 <= right_pos < cols:
            I[add_row, left_pos] = inner
            I[add_row, right_pos] = inner

    return I.tolist()

def solve_388(I):
    # Find distinct non-zero colors
    colors = set()
    for row in I:
        for cell in row:
            if cell != 0:
                colors.add(cell)
    k = len(colors)
    
    # Create output I of size 3k x 3k initialized to 0
    size = 3 * k
    output = [[0 for _ in range(size)] for _ in range(size)]
    
    # Fill blocks
    for r in range(3):
        for c in range(3):
            color = I[r][c]
            if color != 0:
                for dr in range(k):
                    for dc in range(k):
                        output[r * k + dr][c * k + dc] = color
    
    return output

def solve_389(I):
    if not I or not I[0]:
        return []
    h = len(I)
    w = len(I[0])
    non_zeros = [(r, c) for r in range(h) for c in range(w) if I[r][c] != 0]
    if not non_zeros:
        return []
    min_r = min(r for r, c in non_zeros)
    max_r = max(r for r, c in non_zeros)
    min_c = min(c for r, c in non_zeros)
    max_c = max(c for r, c in non_zeros)
    N = max_r - min_r + 1
    if N != max_c - min_c + 1:
        raise ValueError("Shape is not square")
    colors = set(I[r][c] for r, c in non_zeros)
    if len(colors) != 2:
        raise ValueError("Expected exactly two non-zero colors")
    A = I[min_r][min_c]
    B = next(c for c in colors if c != A)
    inner_pos = [(r, c) for r, c in non_zeros if I[r][c] == B]
    if not inner_pos:
        raise ValueError("No inner color found")
    min_r_b = min(r for r, c in inner_pos)
    max_r_b = max(r for r, c in inner_pos)
    min_c_b = min(c for r, c in inner_pos)
    max_c_b = max(c for r, c in inner_pos)
    M = max_r_b - min_r_b + 1
    if M != max_c_b - min_c_b + 1:
        raise ValueError("Inner shape is not square")
    off = (N - M) // 2
    if min_r_b - min_r != off or min_c_b - min_c != off:
        raise ValueError("Inner shape is not concentric")
    output = [[B for _ in range(N)] for _ in range(N)]
    for i in range(M):
        for j in range(M):
            output[off + i][off + j] = A
    return output

from collections import deque

def solve_390(I):
    if not I or not I[0]:
        return []
    
    h = len(I)
    w = len(I[0])
    
    visited = [[False] * w for _ in range(h)]
    
    q = deque()
    
    # Enqueue all boundary 0s
    for i in range(h):
        for j in range(w):
            if (i == 0 or i == h - 1 or j == 0 or j == w - 1) and I[i][j] == 0:
                q.append((i, j))
                visited[i][j] = True
    
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Flood fill to mark reachable 0s
    while q:
        x, y = q.popleft()
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and I[nx][ny] == 0:
                visited[nx][ny] = True
                q.append((nx, ny))
    
    # Collect colors neighboring the holes
    hole_neighbors = set()
    for i in range(h):
        for j in range(w):
            if I[i][j] == 0 and not visited[i][j]:
                for dx, dy in dirs:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < h and 0 <= ny < w and I[nx][ny] != 0:
                        hole_neighbors.add(I[nx][ny])
    
    # Assume exactly one color
    color = list(hole_neighbors)[0]
    return [[color]]

def solve_391(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    for r in range(rows):
        for c in range(cols):
            if output[r][c] == 4 and c % 3 == 0:
                output[r][c] = 6
    return output

import numpy as np

def solve_392(I):
    if not I or not I[0]:
        return []
    I = np.array(I)
    height, width = I.shape
    
    # Find vertical columns: every row non-zero
    vertical_columns = [c for c in range(width) if np.all(I[:, c] != 0)]
    
    # Find horizontal rows: every column non-zero
    horizontal_rows = [r for r in range(height) if np.all(I[r, :] != 0)]
    
    if not vertical_columns or not horizontal_rows:
        return I.tolist()
    
    # Find V_color: from a non-horizontal row in a vertical column
    non_h_row = next(r for r in range(height) if r not in horizontal_rows)
    v_color = I[non_h_row, vertical_columns[0]]
    
    # Find H_color: from a non-vertical column in a horizontal row
    non_v_col = next(c for c in range(width) if c not in vertical_columns)
    h_color = I[horizontal_rows[0], non_v_col]
    
    # Find current intersection color (assume uniform)
    inter_r, inter_c = horizontal_rows[0], vertical_columns[0]
    current = I[inter_r, inter_c]
    
    # Determine new color
    new_color = h_color if current == v_color else v_color
    
    # Set intersection to new_color
    for r in horizontal_rows:
        for c in vertical_columns:
            I[r, c] = new_color
    
    return I.tolist()

from collections import deque

def solve_393(I):
    if not I or not I[0]:
        return []
    
    h = len(I)
    w = len(I[0])
    
    output = [row[:] for row in I]
    visited = [[False] * w for _ in range(h)]
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(h):
        for j in range(w):
            if I[i][j] > 0 and not visited[i][j]:
                color = I[i][j]
                component = []
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and I[nx][ny] == color:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                
                if not component:
                    continue
                
                min_r = min(pos[0] for pos in component)
                max_r = max(pos[0] for pos in component)
                min_c = min(pos[1] for pos in component)
                max_c = max(pos[1] for pos in component)
                
                expected_size = (max_r - min_r + 1) * (max_c - min_c + 1)
                if len(component) != expected_size:
                    continue  # Not a solid rectangle
                
                hh = max_r - min_r + 1
                ww = max_c - min_c + 1
                
                for lr in range(1, hh - 1):
                    global_r = min_r + lr
                    if lr % 2 == 1:  # Type A
                        start, step = 1, 2
                    else:  # Type B
                        start, step = 2, 2
                    for lc in range(start, ww - 1, step):
                        global_c = min_c + lc
                        output[global_r][global_c] = 0
    
    return output

import numpy as np

def solve_394(I):
    I = np.array(I)
    rows, cols = I.shape
    
    # Find the purple bar column c, r_start, h
    c = -1
    r_start = -1
    max_h = 0
    for col in range(cols):
        for row in range(rows):
            if I[row, col] == 8:
                rs = row
                hh = 1
                for r in range(row + 1, rows):
                    if I[r, col] == 8:
                        hh += 1
                    else:
                        break
                if hh > max_h:
                    max_h = hh
                    c = col
                    r_start = rs
    
    h = max_h
    original = I[0].tolist()
    
    # Create output I filled with 5s
    output = np.full((rows, cols), 5, dtype=int)
    
    # Build the pyramid
    for k in range(h):
        py_row = r_start + k
        ideal_w = 1 + 2 * k
        w = min(ideal_w, cols)
        half = (w - 1) // 2
        start_idx = c - half
        end_idx = c + half
        
        # Extract and modify segment
        segment = []
        for idx in range(start_idx, end_idx + 1):
            if 0 <= idx < cols:
                val = original[idx]
                if val == 5:
                    val = 8
                segment.append(val)
        
        # Place segment in output
        start_col = c - half
        for j in range(len(segment)):
            output[py_row, start_col + j] = segment[j]
    
    return output.tolist()

def solve_395(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    visited = [[False] * cols for _ in range(rows)]

    def dfs(r, c, component):
        stack = [(r, c)]
        visited[r][c] = True
        component.append((r, c))
        while stack:
            cr, cc = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == 5:
                    visited[nr][nc] = True
                    stack.append((nr, nc))
                    component.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 5 and not visited[r][c]:
                component = []
                dfs(r, c, component)
                if component:
                    rs = [p[0] for p in component]
                    cs = [p[1] for p in component]
                    min_r = min(rs)
                    max_r = max(rs)
                    min_c = min(cs)
                    max_c = max(cs)
                    h = max_r - min_r + 1
                    w = max_c - min_c + 1
                    inner_h = max(0, h - 2)
                    inner_w = max(0, w - 2)
                    if inner_h > 0 and inner_w > 0:
                        r_off = (h - inner_h) // 2
                        c_off = (w - inner_w) // 2
                        start_r = min_r + r_off
                        end_r = start_r + inner_h - 1
                        start_c = min_c + c_off
                        end_c = start_c + inner_w - 1
                        for ir in range(start_r, end_r + 1):
                            for ic in range(start_c, end_c + 1):
                                output[ir][ic] = 2
    return output

def solve_396(I):
    if not I or not I[0]:
        return []
    
    h = len(I)
    w = len(I[0])
    
    # Find gray column
    gray_col = -1
    for c in range(w):
        if all(I[r][c] == 5 for r in range(h)):
            gray_col = c
            break
    if gray_col == -1:
        return I  # Assume there is one; fallback to return I
    
    left_w = gray_col
    right_start = gray_col + 1
    
    # Left zeros positions
    left_zeros = set()
    for r in range(h):
        for c in range(left_w):
            if I[r][c] == 0:
                left_zeros.add((r, c))
    
    # Right non-zero positions and color
    right_pos = set()
    colors = set()
    for r in range(h):
        for c in range(right_start, w):
            val = I[r][c]
            if val != 0:
                colors.add(val)
                rel_c = c - right_start
                right_pos.add((r, rel_c))
    
    color = None
    if len(colors) == 1:
        color = next(iter(colors))
    
    # Create output as copy of left
    output = [row[:left_w] for row in I]
    
    # If positions match and color exists, fill
    if right_pos == left_zeros and color is not None:
        for r, c in left_zeros:
            output[r][c] = color
    
    return output

import numpy as np

def solve_397(I):
    I = np.array(grid_lst)
    flip = np.fliplr(I)
    result = np.hstack([I, flip, I, flip, I])
    return result.tolist()

def get_line_points(r0, c0, r1, c1):
    points = []
    dx = abs(c1 - c0)
    dy = abs(r1 - r0)
    r = r0
    c = c0
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    if dx > dy:
        err = dx // 2
        for _ in range(dx + 1):
            points.append((r, c))
            err -= dy
            if err < 0:
                r += sr
                err += dx
            c += sc
    else:
        err = dy // 2
        for _ in range(dy + 1):
            points.append((r, c))
            err -= dx
            if err < 0:
                c += sc
                err += dy
            r += sr
    return points

def solve_398(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    red_positions = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 2:
                red_positions.append((r, c))
    if len(red_positions) != 2:
        return [row[:] for row in I]
    (r1, c1), (r2, c2) = red_positions
    points = get_line_points(r1, c1, r2, c2)
    output = [row[:] for row in I]
    for r, c in points:
        if output[r][c] == 0:
            output[r][c] = 2
        elif output[r][c] == 1:
            output[r][c] = 3
    return output

from collections import Counter

def solve_399(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    
    # Collect frequencies
    counter = Counter()
    for row in I:
        for cell in row:
            counter[cell] += 1
    
    # Sort colors by decreasing frequency, then by increasing color number if tie
    color_list = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    
    # Positions in order: left to right, bottom to top within column
    positions = [(c, r) for c in range(width) for r in range(height - 1, -1, -1)]
    
    # Create output I
    output = [[0] * width for _ in range(height)]
    
    # Fill the output
    idx = 0
    for colr, count in color_list:
        for _ in range(count):
            c, r = positions[idx]
            output[r][c] = colr
            idx += 1
    
    return output

def solve_400(I):
    if not I or not I[0]:
        return I
    height = len(I)
    width = len(I[0])
    colors = I[0][:]
    output = [row[:] for row in I]
    for r in range(2, height):
        color = colors[(r - 2) % len(colors)]
        for c in range(width):
            output[r][c] = color
    return output

def solve_401(I):
    if not I or not I[0]:
        return I
    n = len(I)
    assert len(I[0]) == n
    max_l = n // 2 - 1
    ordered_colors = []
    seen = set()
    for l in range(max_l, -1, -1):
        color = I[l][l]
        if color not in seen:
            ordered_colors.append(color)
            seen.add(color)
    num = len(ordered_colors)
    color_map = {}
    for i in range(num):
        color_map[ordered_colors[i]] = ordered_colors[(i + 1) % num]
    output = [[color_map.get(cell, cell) for cell in row] for row in I]
    return output

import numpy as np

def solve_402(I):
    I = [row[:] for row in I]  # copy
    rows = len(I)
    cols = len(I[0])

    # Find purple_col: the column with 8's
    purple_col = None
    for c in range(cols):
        if any(I[r][c] == 8 for r in range(rows)):
            purple_col = c
            break  # assuming unique

    # Find red_row: the row with 2's
    red_row = None
    for r in range(rows):
        if any(I[r][c] == 2 for c in range(cols)):
            red_row = r
            break  # assuming unique

    # Fill purple col where 0 to 8
    for r in range(rows):
        if I[r][purple_col] == 0:
            I[r][purple_col] = 8

    # Fill red row where 0 to 2
    for c in range(cols):
        if I[red_row][c] == 0:
            I[red_row][c] = 2

    # Set intersection to 4
    I[red_row][purple_col] = 4

    return I

def solve_403(I):
    # Extract top-right 2x2 subgrid (rows 0-1, columns 3-4)
    subgrid = [row[3:5] for row in I[0:2]]
    # Rotate 180 degrees: reverse rows, then reverse each row
    rotated = [row[::-1] for row in subgrid[::-1]]
    return rotated

def solve_404(I):
    if not I:
        return []
    height = len(I)
    width = len(I[0])
    bars = []
    for r in range(height - 1):
        row = I[r]
        non_zeros = [c for c in row if c != 0]
        if non_zeros:
            color = non_zeros[0]
            length = len(non_zeros)
            if not all(c == color for c in non_zeros):
                raise ValueError("Mixed colors in bar")
            bars.append((length, color))
    bars.sort(key=lambda x: x[0])
    output = [[0 for _ in range(width)] for _ in range(height)]
    output[height - 1] = I[height - 1][:]
    num_bars = len(bars)
    if num_bars == 0:
        return output
    start_row = height - num_bars - 1
    for i in range(num_bars):
        len_bar, colr = bars[i]
        row = start_row + i
        for j in range(len_bar):
            c = width - 1 - j
            output[row][c] = colr
    return output

import copy

def solve_405(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    reds = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 2:
                reds.append((r, c))
    
    if not reds:
        return I
    
    red_rs = [r for r, c in reds]
    red_cs = [c for r, c in reds]
    
    unique_rows = len(set(red_rs))
    unique_cols = len(set(red_cs))
    
    median_r = sorted(red_rs)[len(red_rs) // 2]
    median_c = sorted(red_cs)[len(red_cs) // 2]
    
    min_r = min(red_rs)
    max_r = max(red_rs)
    min_c = min(red_cs)
    max_c = max(red_cs)
    
    output = [row[:] for row in I]
    
    if unique_rows > unique_cols:
        # Main vertical at median_c from min_r to max_r
        for rr in range(min_r, max_r + 1):
            if output[rr][median_c] == 0:
                output[rr][median_c] = 3
        
        # Connect horizontally for off-main seeds
        for r, c in reds:
            if c != median_c:
                start_c = min(c, median_c)
                end_c = max(c, median_c)
                for cc in range(start_c, end_c + 1):
                    if output[r][cc] == 0:
                        output[r][cc] = 3
    else:
        # Main horizontal at median_r from min_c to max_c
        for cc in range(min_c, max_c + 1):
            if output[median_r][cc] == 0:
                output[median_r][cc] = 3
        
        # Connect vertically for off-main seeds
        for r, c in reds:
            if r != median_r:
                start_r = min(r, median_r)
                end_r = max(r, median_r)
                for rr in range(start_r, end_r + 1):
                    if output[rr][c] == 0:
                        output[rr][c] = 3
    
    return output

import numpy as np

def solve_406(I):
    I = np.array(I)
    rows, cols = I.shape

    # Find grey position
    grey_pos = np.argwhere(I == 5)
    grey_r, grey_c = grey_pos[0]

    # Collect original red positions
    red_positions = [tuple(pos) for pos in np.argwhere(I == 2)]

    # Set original reds to 3
    for r, c in red_positions:
        I[r, c] = 3

    # Set rotated positions to 2
    for r, c in red_positions:
        dr = r - grey_r
        dc = c - grey_c
        new_dr = dc
        new_dc = -dr
        new_r = grey_r + new_dr
        new_c = grey_c + new_dc
        if 0 <= new_r < rows and 0 <= new_c < cols:
            I[new_r, new_c] = 2

    return I.tolist()

from collections import deque
import math

def solve_407(I):
    if not I:
        return []
    I = [row[:] for row in I]  # copy the I
    rows = len(I)
    cols = len(I[0])

    # Mark background 0's (reachable from border) as -1
    visited = [[False] * cols for _ in range(rows)]
    q = deque()
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and I[i][j] == 0:
                q.append((i, j))
                visited[i][j] = True
                I[i][j] = -1

    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 0 and not visited[nr][nc]:
                visited[nr][nc] = True
                I[nr][nc] = -1
                q.append((nr, nc))

    # Reset visited for hole detection
    visited = [[False] * cols for _ in range(rows)]

    # Find and fill holes
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 0 and not visited[i][j]:
                # New hole component
                component = []
                size = 0
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    size += 1
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))

                # Determine color based on size
                if size == 1:
                    color = 6
                elif size == 4:
                    color = 7
                elif size == 9:
                    color = 8
                else:
                    # Assume no other sizes based on examples
                    continue

                # Fill the hole
                for cr, cc in component:
                    I[cr][cc] = color

    # Restore background -1 to 0
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == -1:
                I[i][j] = 0

    return I

def solve_408(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Find all-0 rows
    zero_rows = [r for r in range(rows) if all(cell == 0 for cell in I[r])]
    
    # Find all-0 columns
    zero_cols = [c for c in range(cols) if all(I[r][c] == 0 for r in range(rows))]
    
    # Copy I
    output = [row[:] for row in I]
    
    # Set all cells in zero_rows to 2
    for r in zero_rows:
        for c in range(cols):
            output[r][c] = 2
    
    # Set all cells in zero_cols to 2
    for c in zero_cols:
        for r in range(rows):
            output[r][c] = 2
    
    return output

import numpy as np

def solve_409(I):
    g = np.array(I)
    unique, counts = np.unique(g, return_counts=True)
    max_count = np.max(counts)
    color = unique[np.argmax(counts)]  # Assuming unique max
    positions = np.argwhere(g == color)
    out = np.zeros((9, 9), dtype=int)
    for r, c in positions:
        out[3*r:3*r+3, 3*c:3*c+3] = g
    return out.tolist()

def solve_410(I):
    if len(I) != 3 or any(len(row) != 3 for row in I):
        raise ValueError("Input must be 3x3 I")
    
    row_indices = [2, 1, 0, 0, 1, 2, 2, 1, 0]
    output = []
    
    for i in range(9):
        idx = row_indices[i]
        middle = I[idx][:]
        left = middle[::-1]
        right = left[:]
        output_row = left + middle + right
        output.append(output_row)
    
    return output

def solve_411(I):
    if not I or not I[0]:
        return I
    n = len(I)
    m = len(I[0])
    out = [[0] * (2 * m) for _ in range(2 * n)]
    for i in range(n):
        for j in range(m):
            val = I[i][j]
            for di in range(2):
                for dj in range(2):
                    out[2 * i + di][2 * j + dj] = val
    return out

from collections import defaultdict

def solve_412(I):
    if not I or not I[0]:
        return []
    h = len(I)
    w = len(I[0])
    pos = defaultdict(list)
    for i in range(h):
        for j in range(w):
            c = I[i][j]
            if c != 0:
                pos[c].append((i, j))
    components = []
    for c, lst in pos.items():
        if not lst:
            continue
        min_i = min(p[0] for p in lst)
        max_i = max(p[0] for p in lst)
        min_j = min(p[1] for p in lst)
        max_j = max(p[1] for p in lst)
        bh = max_i - min_i + 1
        bw = max_j - min_j + 1
        bs = max(bh, bw)
        components.append((bs, c))
    if not components:
        return [[0]]
    components.sort(reverse=True)
    n = components[0][0]
    out = [[0] * n for _ in range(n)]
    for s, c in components:
        offset = (n - s) // 2
        sr = offset
        er = sr + s - 1
        sc = offset
        ec = sc + s - 1
        # top
        for j in range(sc, ec + 1):
            out[sr][j] = c
        # bottom
        for j in range(sc, ec + 1):
            out[er][j] = c
        # left
        for i in range(sr + 1, er):
            out[i][sc] = c
        # right
        for i in range(sr + 1, er):
            out[i][ec] = c
    return out

def solve_413(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    
    for r in range(rows):
        color = 0
        for c in range(cols):
            if I[r][c] != 0 and I[r][c] != 5:
                color = I[r][c]
                break
        if color != 0:
            for c in range(cols):
                if I[r][c] == 5:
                    output[r][c] = color
    
    return output

import numpy as np

def solve_414(I):
    rows = len(I)
    cols = len(I[0])
    n = rows - 2

    # Find separator columns from top row
    seps = [c for c in range(cols) if I[0][c] == 5]

    # Starts and ends for each shape
    starts = [0] + [seps[i] + 1 for i in range(len(seps))]
    ends = [seps[0] - 1 if seps else cols - 1] + [seps[i + 1] - 1 for i in range(len(seps) - 1)] + [cols - 1]

    # Function to extract presence matrix for a shape
    def get_presence(shape_idx):
        s = starts[shape_idx]
        e = ends[shape_idx]
        presence = [[0] * n for _ in range(n)]
        for r in range(n):
            for cc in range(n):
                global_c = s + 1 + cc
                if global_c <= e:
                    val = I[r + 1][global_c]
                    presence[r][cc] = 1 if val != 0 else 0
                # else remains 0
        return presence

    # Function to get color for a shape
    def get_color(shape_idx):
        s = starts[shape_idx]
        # Find a non-zero cell, starting from s+1
        for r in range(1, rows - 1):
            for cc in range(1, min(n + 1, ends[shape_idx] - s + 1)):
                val = I[r][s + cc]
                if val != 0:
                    return val
        return 0  # Should not happen

    A = get_presence(0)
    B = get_presence(1)
    fg = get_color(2)
    bg = get_color(3)

    A_np = np.array(A)
    B_np = np.array(B)
    kron = np.kron(B_np, A_np)
    output_np = np.where(kron == 1, fg, bg)
    return output_np.tolist()

from collections import deque

def solve_415(I):
    if not I or not I[0]:
        return I
    
    # Find noise color
    colors = set()
    for row in I:
        for cell in row:
            colors.add(cell)
    colors.discard(0)
    colors.discard(1)
    if len(colors) != 1:
        raise ValueError("Expected exactly one noise color")
    noise_color = colors.pop()
    
    height = len(I)
    width = len(I[0])
    visited = set()
    queue = deque()
    
    # Add all border non-1 cells
    for r in range(height):
        for c in [0, width - 1]:
            if I[r][c] != 1:
                if (r, c) not in visited:
                    visited.add((r, c))
                    queue.append((r, c))
    for c in range(width):
        for r in [0, height - 1]:
            if I[r][c] != 1:
                if (r, c) not in visited:
                    visited.add((r, c))
                    queue.append((r, c))
    
    # Flood fill
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < height and 0 <= ny < width and (nx, ny) not in visited and I[nx][ny] != 1:
                visited.add((nx, ny))
                queue.append((nx, ny))
    
    # Count inside noise cells
    count = 0
    for r in range(height):
        for c in range(width):
            if I[r][c] == noise_color and (r, c) not in visited:
                count += 1
    
    # Create 3x3 output
    output = [[0] * 3 for _ in range(3)]
    idx = 0
    for r in range(3):
        for c in range(3):
            if idx < count:
                output[r][c] = noise_color
                idx += 1
            else:
                break
    
    return output

from collections import defaultdict, Counter

def solve_416(I):
    if not I or not I[0]:
        return []

    rows = len(I)
    cols = len(I[0])

    # Count frequencies to find background
    all_colors = [I[i][j] for i in range(rows) for j in range(cols)]
    count = Counter(all_colors)
    bg = count.most_common(1)[0][0]

    # Collect positions per color
    pos = defaultdict(list)
    for i in range(rows):
        for j in range(cols):
            c = I[i][j]
            pos[c].append((i, j))

    # Find S
    max_side = 0
    non_bg_colors = [c for c in pos if c != bg and pos[c]]
    for c in non_bg_colors:
        positions = pos[c]
        min_r = min(p[0] for p in positions)
        max_r = max(p[0] for p in positions)
        min_c = min(p[1] for p in positions)
        max_c = max(p[1] for p in positions)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        max_side = max(max_side, max(h, w))

    S = max_side
    if S == 0:
        return []

    # Initialize output
    output = [[bg for _ in range(S)] for _ in range(S)]

    # Place each color's positions
    for c in non_bg_colors:
        positions = pos[c]
        min_r = min(p[0] for p in positions)
        max_r = max(p[0] for p in positions)
        min_c = min(p[1] for p in positions)
        max_c = max(p[1] for p in positions)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        offset_r = (S - h) // 2
        offset_c = (S - w) // 2
        for r, col in positions:
            local_r = r - min_r
            local_c = col - min_c
            out_r = local_r + offset_r
            out_c = local_c + offset_c
            output[out_r][out_c] = c

    return output

import numpy as np

def solve_417(I):
    I = np.array(I)
    I[I == 7] = 5
    return I.tolist()

import numpy as np

def solve_418(I):
    I = np.array(I)
    rows, cols = I.shape
    output = I.copy()
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(r, c):
        component = []
        stack = [(r, c)]
        visited[r, c] = True
        while stack:
            cr, cc = stack.pop()
            component.append((cr, cc))
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and I[nr, nc] == 2:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        return component

    for i in range(rows):
        for j in range(cols):
            if I[i, j] == 2 and not visited[i, j]:
                comp = dfs(i, j)
                if not comp:
                    continue
                rs = [p[0] for p in comp]
                cs = [p[1] for p in comp]
                unique_rs = set(rs)
                unique_cs = set(cs)
                if len(unique_rs) == 1:  # horizontal
                    row_c = rs[0]
                    min_c = min(cs)
                    max_c = max(cs)
                    L = len(comp)
                    if max_c - min_c + 1 != L:
                        continue
                    for rr in range(rows):
                        d = abs(rr - row_c)
                        start_c = min_c + d
                        end_c = max_c - d
                        if start_c <= end_c:
                            for cc in range(start_c, end_c + 1):
                                if output[rr, cc] == 0:
                                    output[rr, cc] = 8
                elif len(unique_cs) == 1:  # vertical
                    col_c = cs[0]
                    min_r = min(rs)
                    max_r = max(rs)
                    L = len(comp)
                    if max_r - min_r + 1 != L:
                        continue
                    for cc in range(cols):
                        d = abs(cc - col_c)
                        start_r = min_r + d
                        end_r = max_r - d
                        if start_r <= end_r:
                            for rr in range(start_r, end_r + 1):
                                if output[rr, cc] == 0:
                                    output[rr, cc] = 8
    return output.tolist()

def solve_419(I):
    h = len(I)
    if h == 0:
        return []
    w = len(I[0])
    # Create horizontally flipped I
    flipped = [[I[r][w - 1 - c] for c in range(w)] for r in range(h)]
    # Concatenate original and flipped
    output = [I[r] + flipped[r] for r in range(h)]
    return output

def solve_420(I):
    out = [row[:] for row in I]
    for r in range(len(I)):
        color = I[r][0]
        if color != 0:
            for c in range(len(I[r])):
                if I[r][c] == 5:
                    out[r][c] = color
    return out

def solve_421(I):
    main = [I[i][i] for i in range(5)]
    anti = [I[i][4 - i] for i in range(5)]
    out = [[0] * 3 for _ in range(3)]
    out[0][0] = main[0]
    out[0][1] = main[1]
    out[0][2] = anti[0]
    out[1][0] = anti[1]
    out[1][1] = main[2]
    out[1][2] = anti[3]
    out[2][0] = anti[4]
    out[2][1] = main[3]
    out[2][2] = main[4]
    return out

def solve_422(I):
    if not I or not I[0]:
        return I
    n = len(I)
    C = I[n-1][n-1]
    base_size = 0
    for r in range(n-1, -1, -1):
        if all(I[r][c] == C for c in range(n)):
            base_size += 1
        else:
            break
    main_size = n - base_size
    # Assume verification passed as per examples

    even_color = [I[0][j] for j in range(main_size)]
    odd_color = [I[1][j] for j in range(main_size)]

    # Find minimal p
    p = 1
    for possible_p in range(1, main_size // 2 + 1):
        is_period = True
        for j in range(main_size - possible_p):
            if even_color[j] != even_color[j + possible_p] or odd_color[j] != odd_color[j + possible_p]:
                is_period = False
                break
        if is_period:
            p = possible_p
            break

    # New colors for period
    new_even = [odd_color[k] for k in range(p)]
    new_odd = [even_color[k] for k in range(p)]

    # Create output
    output = [[0] * n for _ in range(n)]
    for i in range(n):
        is_even = (i % 2 == 0)
        for j in range(n):
            k = j % p
            output[i][j] = new_even[k] if is_even else new_odd[k]

    return output

import copy
from collections import Counter

def solve_423(I):
    output = copy.deepcopy(I)
    pillar_cols = [0, 1, 3, 4, 6, 7]
    
    # Vertical propagation
    for c in pillar_cols:
        # Top rows to middle top
        if I[0][c] == I[6][c] and I[0][c] != 1 and I[0][c] != 0:
            output[3][c] = I[0][c]
        # Bottom rows to middle bottom
        if I[1][c] == I[7][c] and I[1][c] != 1 and I[1][c] != 0:
            output[4][c] = I[1][c]
    
    # Horizontal propagation
    left_group = [0, 3, 6]
    right_group = [1, 4, 7]
    for r in range(8):
        # Left group
        colors = [output[r][c] for c in left_group]
        count = Counter(colors)
        if len(count) == 2 and 1 in count and count[1] == 1:
            candidates = [k for k in count if k != 1]
            if len(candidates) == 1:
                C = candidates[0]
                if count[C] == 2:
                    for c in left_group:
                        if output[r][c] == 1:
                            output[r][c] = C
                            break
        # Right group
        colors = [output[r][c] for c in right_group]
        count = Counter(colors)
        if len(count) == 2 and 1 in count and count[1] == 1:
            candidates = [k for k in count if k != 1]
            if len(candidates) == 1:
                C = candidates[0]
                if count[C] == 2:
                    for c in right_group:
                        if output[r][c] == 1:
                            output[r][c] = C
                            break
    
    return output

def solve_424(I):
    n = len(I)
    block = []
    for row in I:
        repeated_row = row * n
        block.append(repeated_row)
    output = block * n
    return output

import numpy as np

def solve_425(I):
    small = np.array(I)
    large = np.zeros((9, 9), dtype=int)
    for i in range(3):
        for j in range(3):
            if small[i, j] == 2:
                large[3*i:3*i+3, 3*j:3*j+3] = small
    return large.tolist()

from collections import deque
from typing import List

def solve_426(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    count = {}
    
    def bfs(r: int, c: int, color: int):
        q = deque([(r, c)])
        visited[r][c] = True
        while q:
            x, y = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and I[nx][ny] == color:
                    visited[nx][ny] = True
                    q.append((nx, ny))
    
    for r in range(rows):
        for c in range(cols):
            if I[r][c] > 0 and not visited[r][c]:
                color = I[r][c]
                bfs(r, c, color)
                if color not in count:
                    count[color] = 0
                count[color] += 1
    
    odd_color = None
    for color, cnt in count.items():
        if cnt % 2 == 1:
            odd_color = color
            break  # Assume only one, as per examples
    
    if odd_color is None:
        return []  # Should not happen based on examples
    
    positions = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == odd_color:
                positions.append((r, c))
    
    if not positions:
        return []
    
    min_r = min(r for r, _ in positions)
    max_r = max(r for r, _ in positions)
    min_c = min(c for _, c in positions)
    max_c = max(c for _, c in positions)
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    output = [[0] * width for _ in range(height)]
    for i in range(height):
        for j in range(width):
            output[i][j] = I[min_r + i][min_c + j]
    
    return output

def solve_427(I):
    if not I or not I[0]:
        return []
    
    rows = len(I)
    cols = len(I[0])
    
    positions = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0:
                positions.append((c, I[r][c]))
    
    sorted_positions = sorted(positions)
    colors = [col for _, col in sorted_positions]
    
    while len(colors) < 9:
        colors.append(0)
    
    output = []
    for i in range(3):
        row = colors[i*3:(i+1)*3]
        if i % 2 == 1:
            row = row[::-1]
        output.append(row)
    
    return output

def solve_428(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 5:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            output[nr][nc] = 1
    return output

import numpy as np

def solve_429(I):
    I = np.array(grid_lst)
    height, width = I.shape
    middle = height // 2
    upper = I[0:middle, :]
    lower = I[middle + 1:, :]
    out = np.zeros((middle, width), dtype=int)
    diff = upper != lower
    out[diff] = 3
    return out.tolist()

def solve_430(I):
    if not I or not I[0]:
        return []

    rows = len(I)
    cols = len(I[0])

    # Get vertical columns and colors from row 0
    vert_cols = []
    vert_colors = []
    for c in range(cols):
        if I[0][c] != 0:
            vert_cols.append(c)
            vert_colors.append(I[0][c])

    N = len(vert_cols)
    if N == 0:
        return []

    W = 2 * N + 1

    # Find bar rows and colors
    bars = []
    vert_set = set(vert_cols)
    for r in range(rows):
        non_vert_colors = [I[r][c] for c in range(cols) if c not in vert_set]
        if non_vert_colors:
            s = set(non_vert_colors)
            if len(s) == 1 and list(s)[0] != 0:
                bar_color = list(s)[0]
                bars.append((r, bar_color))

    B = len(bars)
    H = 2 * B + 1

    output = [[0 for _ in range(W)] for _ in range(H)]

    # Set empty rows
    for i in range(H):
        if i % 2 == 0:
            for j in range(1, W, 2):
                m = (j - 1) // 2
                output[i][j] = vert_colors[m]

    # Set bar rows
    for k in range(B):
        out_row = 2 * k + 1
        r, bar_color = bars[k]
        # Even columns: bar_color
        for j in range(0, W, 2):
            output[out_row][j] = bar_color
        # Odd columns: input value at intersection
        for j in range(1, W, 2):
            m = (j - 1) // 2
            vc = vert_cols[m]
            output[out_row][j] = I[r][vc]

    return output

def solve_431(I):
    if not I or not I[0]:
        return []
    
    height = len(I)
    # Extract subgrids
    yellow = [row[0:4] for row in I]
    brown = [row[5:9] for row in I]
    blue = [row[10:14] for row in I]
    
    output = [[0] * 4 for _ in range(height)]
    
    for i in range(height):
        for j in range(4):
            if yellow[i][j] != 0:
                output[i][j] = yellow[i][j]
            elif brown[i][j] != 0:
                output[i][j] = brown[i][j]
            elif blue[i][j] != 0:
                output[i][j] = blue[i][j]
            else:
                output[i][j] = 0
    
    return output

def solve_432(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    for c in range(cols):
        current = 0
        for r in range(rows):
            if I[r][c] != 0:
                output[r][c] = I[r][c]
                current = I[r][c]
            else:
                if current != 0:
                    output[r][c] = current
                else:
                    output[r][c] = 0
    return output

def solve_433(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Find the position of the purple cell (8)
    r, c = -1, -1
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 8:
                r, c = i, j
                break
        if r != -1:
            break
    
    if r == -1:
        return I  # No purple cell, return unchanged
    
    output = [row[:] for row in I]
    
    # Upper part
    if r > 0:
        current_col = c
        current_row = r - 1
        level = 1
        while current_row >= 0:
            if level % 2 == 1:
                # Odd level: single cell
                output[current_row][current_col] = 5
            else:
                # Even level: three cells to the right if possible
                if current_col + 2 < cols:
                    for cc in range(current_col, current_col + 3):
                        output[current_row][cc] = 5
                    current_col += 2
                else:
                    output[current_row][current_col] = 5
                    break
            current_row -= 1
            level += 1
    
    # Lower part
    if r < rows - 1:
        current_col = c
        current_row = r + 1
        level = 1
        while current_row < rows:
            if level % 2 == 1:
                # Odd level: single cell
                output[current_row][current_col] = 5
            else:
                # Even level: three cells to the left if possible
                if current_col - 2 >= 0:
                    for cc in range(current_col - 2, current_col + 1):
                        output[current_row][cc] = 5
                    current_col -= 2
                else:
                    output[current_row][current_col] = 5
                    break
            current_row += 1
            level += 1
    
    return output

import numpy as np

def solve_434(I):
    if not I or not I[0]:
        return []
    grid_np = np.array(I)
    rows, cols = grid_np.shape
    visited = np.zeros((rows, cols), dtype=bool)

    def dfs(x, y):
        stack = [(x, y)]
        visited[x, y] = True
        while stack:
            cx, cy = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and grid_np[nx, ny] == 8:
                    visited[nx, ny] = True
                    stack.append((nx, ny))

    count = 0
    for i in range(rows):
        for j in range(cols):
            if grid_np[i, j] == 8 and not visited[i, j]:
                dfs(i, j)
                count += 1

    n = count
    output = [[8 if i == j else 0 for j in range(n)] for i in range(n)]
    return output

def solve_435(I):
    n = len(I)
    m = 2 * n
    output = [[0] * m for _ in range(m)]
    for r in range(n):
        for c in range(n):
            if I[r][c] != 0:
                k = I[r][c]
                d = c - r
                for i in range(r, m):
                    j = i + d
                    if 0 <= j < m:
                        output[i][j] = k
    return output

def solve_436(I):
    h = len(I) // 2
    w = len(I[0]) if I else 0
    output = []
    for i in range(h):
        row = []
        for j in range(w):
            top_val = I[i][j]
            bot_val = I[i + h][j]
            if top_val == 3 or bot_val == 5:
                row.append(4)
            else:
                row.append(0)
        output.append(row)
    return output

import numpy as np

def solve_437(I):
    if not I or not I[0]:
        return I
    I = np.array(I)
    rows, cols = I.shape
    # Count non-zero in each column
    non_zero_counts = np.sum(I != 0, axis=0)
    # Find max count
    max_count = np.max(non_zero_counts)
    # Find rightmost column with max count
    selected_col = np.where(non_zero_counts == max_count)[0][-1]
    # Create output: keep selected column, set others to 0
    output = np.zeros_like(I)
    output[:, selected_col] = I[:, selected_col]
    return output.tolist()

def solve_438(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Find connected components
    visited = set()
    components = []
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != 0 and (i, j) not in visited:
                comp_cells = []
                stack = [(i, j)]
                visited.add((i, j))
                minr, maxr, minc, maxc = i, i, j, j
                while stack:
                    r, c = stack.pop()
                    color = I[r][c]
                    comp_cells.append((r, c, color))
                    minr = min(minr, r)
                    maxr = max(maxr, r)
                    minc = min(minc, c)
                    maxc = max(maxc, c)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] != 0 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
                components.append({
                    'cells': comp_cells,
                    'min_r': minr,
                    'max_r': maxr,
                    'min_c': minc,
                    'max_c': maxc
                })
    
    n = len(components)
    if n == 0:
        return I
    
    # Build graph for groups based on row overlap
    graph = [[] for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if max(components[a]['min_r'], components[b]['min_r']) <= min(components[a]['max_r'], components[b]['max_r']):
                graph[a].append(b)
                graph[b].append(a)
    
    # Find connected components (groups)
    group_visited = [False] * n
    groups = []
    for start in range(n):
        if not group_visited[start]:
            group = []
            stack = [start]
            group_visited[start] = True
            while stack:
                cur = stack.pop()
                group.append(cur)
                for nei in graph[cur]:
                    if not group_visited[nei]:
                        group_visited[nei] = True
                        stack.append(nei)
            groups.append(group)
    
    # Initialize output I to all 0
    output_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Process each group
    for group in groups:
        # Sort shapes by max_c descending
        sorted_shapes = sorted(group, key=lambda idx: components[idx]['max_c'], reverse=True)
        
        for shape_idx in sorted_shapes:
            comp = components[shape_idx]
            orig_right = comp['max_c']
            # Get unique rows for this shape
            shape_rows = set(r for r, c, _ in comp['cells'])
            max_right = 14
            for row in shape_rows:
                left_obstacle = cols
                for cc in range(cols):
                    if output_grid[row][cc] != 0:
                        left_obstacle = cc
                        break
                max_right_row = left_obstacle - 1 if left_obstacle < cols else 14
                max_right = min(max_right, max_right_row)
            s = max_right - orig_right
            if s < 0:
                s = 0
            # Place cells
            for r, c, color in comp['cells']:
                new_c = c + s
                output_grid[r][new_c] = color
    
    return output_grid

import numpy as np

def solve_439(I):
    I = np.array(grid_lst)
    if not I.size:
        return grid_lst

    height, width = I.shape
    visited = [[False] * width for _ in range(height)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components = []

    for r in range(height):
        for c in range(width):
            if I[r, c] != 0 and not visited[r][c]:
                color = I[r, c]
                positions = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    positions.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < height and 0 <= nc < width and not visited[nr][nc] and I[nr, nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append((color, positions))

    # Create output I
    out_grid = I.copy()

    # Recolor each component
    for _, positions in components:
        size = len(positions)
        new_color = 2 if size == 6 else 1
        for pr, pc in positions:
            out_grid[pr, pc] = new_color

    return out_grid.tolist()

import copy

def solve_440(I):
    output = copy.deepcopy(I)
    if not I or not I[0]:
        return output
    rows = len(I)
    cols = len(I[0])
    
    # Toggle horizontal
    for i in range(rows):
        if I[i][0] == 4 and I[i][cols-1] == 4:
            for j in range(1, cols-1):
                c = output[i][j]
                if c == 0:
                    output[i][j] = 8
                elif c == 8:
                    output[i][j] = 0
                elif c == 6:
                    output[i][j] = 7
                elif c == 7:
                    output[i][j] = 6
    
    # Toggle vertical
    for j in range(cols):
        if I[0][j] == 4 and I[rows-1][j] == 4:
            for i in range(1, rows-1):
                c = output[i][j]
                if c == 0:
                    output[i][j] = 8
                elif c == 8:
                    output[i][j] = 0
                elif c == 6:
                    output[i][j] = 7
                elif c == 7:
                    output[i][j] = 6
    
    return output

import numpy as np

def solve_441(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Find bounding box
    min_r, max_r, min_c, max_c = rows, -1, cols, -1
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    
    if min_r == rows:
        return I  # no shape
    
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    
    # Output I
    output = [[0] * cols for _ in range(rows)]
    
    # Horizontal row
    six_positions = []
    i = 0
    while True:
        start_c = min_c + i * (w + 1)
        if start_c >= cols:
            break
        color = 6 if i % 3 == 2 else 7
        if color == 6:
            six_positions.append(start_c)
        for rel_r in range(h):
            abs_r = min_r + rel_r
            if abs_r >= rows:
                break
            for rel_c in range(w):
                abs_c = start_c + rel_c
                if abs_c >= cols:
                    break
                orig_val = I[min_r + rel_r][min_c + rel_c]
                if orig_val != 0:
                    output[abs_r][abs_c] = color
        i += 1
    
    # Vertical stacks for 6 positions
    for start_c in six_positions:
        j = 1
        while True:
            start_r = min_r + j * (h + 1)
            if start_r >= rows:
                break
            color = 6
            for rel_r in range(h):
                abs_r = start_r + rel_r
                if abs_r >= rows:
                    break
                for rel_c in range(w):
                    abs_c = start_c + rel_c
                    if abs_c >= cols:
                        break
                    orig_val = I[min_r + rel_r][min_c + rel_c]
                    if orig_val != 0:
                        output[abs_r][abs_c] = color
            j += 1
    
    return output

def solve_442(I):
    if not I or not I[0]:
        return I
    h = len(I)
    w = len(I[0])
    output = [row[:] for row in I]
    blues = [(r, c) for r in range(h) for c in range(w) if I[r][c] == 1]
    directions = [(-1, 0, 2), (1, 0, 8), (0, -1, 7), (0, 1, 6)]
    for r, c in blues:
        for dr, dc, color in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                output[nr][nc] = color
    return output

import numpy as np

def solve_443(I):
    I = np.array(I)
    height, width = I.shape
    output = I.copy()
    parity = width % 2
    for col in range(width):
        if (col % 2) != parity:
            for row in range(height):
                if I[row, col] == 5:
                    output[row, col] = 3
                    break
    return output.tolist()

def solve_444(I):
    if not I or not I[0]:
        return I
    
    h = len(I)
    w = len(I[0])
    output = [row[:] for row in I]
    core_color = 3
    
    # Collect core positions per row and column
    row_core = [[] for _ in range(h)]
    col_core = [[] for _ in range(w)]
    for r in range(h):
        for c in range(w):
            if I[r][c] == core_color:
                row_core[r].append(c)
                col_core[c].append(r)
    
    # Process each peripheral cell
    for r in range(h):
        for c in range(w):
            colr = I[r][c]
            if colr > 0 and colr != core_color:
                if row_core[r]:  # Horizontal extension
                    min_col = min(row_core[r])
                    max_col = max(row_core[r])
                    if c < min_col:
                        for cc in range(c + 1, min_col):
                            output[r][cc] = colr
                    elif c > max_col:
                        for cc in range(max_col + 1, c):
                            output[r][cc] = colr
                elif col_core[c]:  # Vertical extension
                    min_row = min(col_core[c])
                    max_row = max(col_core[c])
                    if r < min_row:
                        for rr in range(r + 1, min_row):
                            output[rr][c] = colr
                    elif r > max_row:
                        for rr in range(max_row + 1, r):
                            output[rr][c] = colr
    
    return output

def solve_445(I):
    c = 0
    for row in I:
        for cell in row:
            if cell != 0:
                c = cell
                break
        if c != 0:
            break
    if c == 1:
        return [[0, 5, 0], [5, 5, 5], [0, 5, 0]]
    elif c == 2:
        return [[5, 5, 5], [0, 5, 0], [0, 5, 0]]
    elif c == 3:
        return [[0, 0, 5], [0, 0, 5], [5, 5, 5]]
    else:
        return []  # Should not occur based on examples

from collections import defaultdict

def solve_446(I):
    left = [row[0:10] for row in I]
    right = [row[11:21] for row in I]
    rows = len(I)
    cols = 10
    left_pos = defaultdict(set)
    right_pos = defaultdict(set)
    for i in range(rows):
        for j in range(cols):
            c = left[i][j]
            if c != 0:
                left_pos[c].add((i, j))
            c = right[i][j]
            if c != 0:
                right_pos[c].add((i, j))
    output = [[0] * cols for _ in range(rows)]
    all_colors = set(left_pos.keys()) | set(right_pos.keys())
    for color in all_colors:
        inter = left_pos[color] & right_pos[color]
        for pos in inter:
            ii, jj = pos
            output[ii][jj] = color
        l_only = left_pos[color] - inter
        r_only = right_pos[color] - inter
        if l_only and r_only:
            l_pos = list(l_only)[0]
            r_pos = list(r_only)[0]
            output[l_pos[0]][l_pos[1]] = 2
            output[r_pos[0]][r_pos[1]] = 1
    return output

def solve_447(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    # Find the special cell
    special_r = special_c = special_color = None
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != 0 and I[i][j] != 5:
                special_r = i
                special_c = j
                special_color = I[i][j]
                break  # Assume only one
        if special_r is not None:
            break
    if special_color is None:
        return [row[:] for row in I]
    parity_row = special_r % 2
    parity_col = special_c % 2
    output = [row[:] for row in I]
    for i in range(rows):
        if i % 2 == parity_row:
            for j in range(cols):
                if j % 2 == parity_col and output[i][j] == 0:
                    output[i][j] = special_color
    return output

def solve_448(I):
    if not I or not I[0]:
        return I
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    r_r, c_r = None, None
    r_p, c_p = None, None
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 2:
                r_r, c_r = i, j
            elif I[i][j] == 8:
                r_p, c_p = i, j
    if r_r is None or r_p is None:
        return output  # No change if missing
    # Horizontal at r_r, cols min(c_p, c_r) to max(c_p, c_r)
    min_c = min(c_p, c_r)
    max_c = max(c_p, c_r)
    for j in range(min_c, max_c + 1):
        output[r_r][j] = 4
    # Vertical at c_p, rows min(r_p, r_r) to max(r_p, r_r)
    min_r = min(r_p, r_r)
    max_r = max(r_p, r_r)
    for i in range(min_r, max_r + 1):
        output[i][c_p] = 4
    # Restore originals
    output[r_r][c_r] = 2
    output[r_p][c_p] = 8
    return output

def solve_449(I):
    # Find distinct non-zero colors
    colors = set()
    for row in I:
        for cell in row:
            if cell != 0:
                colors.add(cell)
    k = len(colors)
    
    # Create output I of size 3k x 3k initialized to 0
    size = 3 * k
    output = [[0 for _ in range(size)] for _ in range(size)]
    
    # Fill blocks
    for r in range(3):
        for c in range(3):
            color = I[r][c]
            if color != 0:
                for dr in range(k):
                    for dc in range(k):
                        output[r * k + dr][c * k + dc] = color
    
    return output

import numpy as np
from collections import deque

def solve_450(I):
    if not grid_lst or not grid_lst[0]:
        return []
    I = np.array(grid_lst)
    h, w = I.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for i in range(h):
        for j in range(w):
            if I[i, j] > 0 and I[i, j] != 5 and not visited[i, j]:
                c = I[i, j]
                component_cells = []
                q = deque([(i, j)])
                visited[i, j] = True
                while q:
                    x, y = q.popleft()
                    component_cells.append((x, y))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny] and I[nx, ny] == c:
                            visited[nx, ny] = True
                            q.append((nx, ny))
                minr = min(xx for xx, _ in component_cells)
                maxr = max(xx for xx, _ in component_cells)
                minc = min(yy for _, yy in component_cells)
                maxc = max(yy for _, yy in component_cells)
                count = 0
                for ii in range(minr, maxr + 1):
                    for jj in range(minc, maxc + 1):
                        if I[ii, jj] == 5:
                            count += 1
                components.append((count, c))
    components.sort(key=lambda x: x[0])
    if not components:
        return []
    M = max(cnt for cnt, _ in components)
    K = len(components)
    out = [[0] * M for _ in range(K)]
    for idx, (cnt, col) in enumerate(components):
        for j in range(cnt):
            out[idx][j] = col
    return out

import numpy as np

def solve_451(I):
    I = np.array(grid_lst)
    rows, cols = I.shape

    # Find bounding box
    grey_positions = np.argwhere(I == 5)
    min_r = np.min(grey_positions[:, 0])
    max_r = np.max(grey_positions[:, 0])
    min_c = np.min(grey_positions[:, 1])
    max_c = np.max(grey_positions[:, 1])

    # Find the gap and the side
    gap_r, gap_c, side = None, None, None

    # Check top
    top_row = I[min_r, min_c:max_c+1]
    if np.any(top_row != 5):
        missing_cs = np.where(top_row != 5)[0] + min_c
        if len(missing_cs) == 1:
            gap_r, gap_c, side = min_r, missing_cs[0], 'top'

    # Check bottom
    bottom_row = I[max_r, min_c:max_c+1]
    if np.any(bottom_row != 5):
        missing_cs = np.where(bottom_row != 5)[0] + min_c
        if len(missing_cs) == 1:
            gap_r, gap_c, side = max_r, missing_cs[0], 'bottom'

    # Check left
    left_col = I[min_r:max_r+1, min_c]
    if np.any(left_col != 5):
        missing_rs = np.where(left_col != 5)[0] + min_r
        if len(missing_rs) == 1:
            gap_r, gap_c, side = missing_rs[0], min_c, 'left'

    # Check right
    right_col = I[min_r:max_r+1, max_c]
    if np.any(right_col != 5):
        missing_rs = np.where(right_col != 5)[0] + min_r
        if len(missing_rs) == 1:
            gap_r, gap_c, side = missing_rs[0], max_c, 'right'

    # Assume exactly one gap found
    assert side is not None

    # Fill interior
    for r in range(min_r + 1, max_r):
        for c in range(min_c + 1, max_c):
            I[r, c] = 8

    # Fill gap
    I[gap_r, gap_c] = 8

    # Extend outwards
    if side == 'top':
        for r in range(gap_r - 1, -1, -1):
            I[r, gap_c] = 8
    elif side == 'bottom':
        for r in range(gap_r + 1, rows):
            I[r, gap_c] = 8
    elif side == 'left':
        for c in range(gap_c - 1, -1, -1):
            I[gap_r, c] = 8
    elif side == 'right':
        for c in range(gap_c + 1, cols):
            I[gap_r, c] = 8

    return I.tolist()

def solve_452(I):
    output = [row[:] for row in I]
    for i in range(len(output)):
        for j in range(len(output[0])):
            if output[i][j] == 5:
                output[i][j] = 8
            elif output[i][j] == 8:
                output[i][j] = 5
    return output

from collections import deque
import copy

def solve_453(I):
    if not I or not I[0]:
        return []
    
    h = len(I)
    w = len(I[0])
    
    output = copy.deepcopy(I)
    visited = [[False] * w for _ in range(h)]
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(h):
        for j in range(w):
            if I[i][j] == 2 and not visited[i][j]:
                component = []
                q = deque([(i, j)])
                visited[i][j] = True
                while q:
                    x, y = q.popleft()
                    component.append((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx][ny] and I[nx][ny] == 2:
                            visited[nx][ny] = True
                            q.append((nx, ny))
                
                if not component:
                    continue
                
                min_r = min(pos[0] for pos in component)
                max_r = max(pos[0] for pos in component)
                min_c = min(pos[1] for pos in component)
                max_c = max(pos[1] for pos in component)
                
                expected = (max_r - min_r + 1) * (max_c - min_c + 1)
                
                # Always set component to 0
                for x, y in component:
                    output[x][y] = 0
                
                # Check if hollow
                if len(component) < expected:
                    inner_min_r = min_r + 1
                    inner_max_r = max_r - 1
                    inner_min_c = min_c + 1
                    inner_max_c = max_c - 1
                    if inner_min_r <= inner_max_r and inner_min_c <= inner_max_c:
                        all_zero = True
                        for r in range(inner_min_r, inner_max_r + 1):
                            for c in range(inner_min_c, inner_max_c + 1):
                                if I[r][c] != 0:
                                    all_zero = False
                                    break
                            if not all_zero:
                                break
                        if all_zero:
                            inner_area = (inner_max_r - inner_min_r + 1) * (inner_max_c - inner_min_c + 1)
                            if expected - len(component) == inner_area:
                                for r in range(inner_min_r, inner_max_r + 1):
                                    for c in range(inner_min_c, inner_max_c + 1):
                                        output[r][c] = 3
    
    return output

def solve_454(I):
    flat = [cell for row in I for cell in row if cell != 0]
    if not flat:
        return []
    c = flat[0]  # All non-zero are same
    n = len(flat)
    return [[c] * n]

def solve_455(I):
    output = [row[:] for row in I]
    h = len(I)
    w = len(I[0])
    top_color = I[0][1]
    bottom_color = I[h-1][1]
    left_color = I[1][0]
    right_color = I[1][w-1]
    inner_top_row = 1
    inner_bottom_row = h - 2
    inner_left_col = 1
    inner_right_col = w - 2
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            s = I[r][c]
            if s == 0:
                continue
            output[r][c] = 0
            if s == top_color:
                output[inner_top_row][c] = s
            elif s == bottom_color:
                output[inner_bottom_row][c] = s
            elif s == left_color:
                output[r][inner_left_col] = s
            elif s == right_color:
                output[r][inner_right_col] = s
    return output

def solve_456(I):
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def dfs(r, c, color, comp):
        stack = [(r, c)]
        while stack:
            x, y = stack.pop()
            if visited[x][y]:
                continue
            visited[x][y] = True
            comp.append((x, y))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and I[nx][ny] == color:
                    stack.append((nx, ny))

    for i in range(rows):
        for j in range(cols):
            if I[i][j] != 0 and not visited[i][j]:
                comp = []
                dfs(i, j, I[i][j], comp)
                components.append(comp)

    if len(components) != 2:
        return I

    comp1, comp2 = components

    def get_bbox(comp):
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        return min(rs), max(rs), min(cs), max(cs)

    minr1, maxr1, minc1, maxc1 = get_bbox(comp1)
    minr2, maxr2, minc2, maxc2 = get_bbox(comp2)

    row_ol_start = max(minr1, minr2)
    row_ol_end = min(maxr1, maxr2)
    row_ol_len = max(0, row_ol_end - row_ol_start + 1)

    col_ol_start = max(minc1, minc2)
    col_ol_end = min(maxc1, maxc2)
    col_ol_len = max(0, col_ol_end - col_ol_start + 1)

    output = [row[:] for row in I]

    if row_ol_len > 0 and col_ol_len == 0:
        # horizontal separation
        if minc1 < minc2:
            left_minr, left_maxr, left_minc, left_maxc = minr1, maxr1, minc1, maxc1
            right_minr, right_maxr, right_minc, right_maxc = minr2, maxr2, minc2, maxc2
        else:
            left_minr, left_maxr, left_minc, left_maxc = minr2, maxr2, minc2, maxc2
            right_minr, right_maxr, right_minc, right_maxc = minr1, maxr1, minc1, maxc1
        gap_start_c = left_maxc + 1
        gap_end_c = right_minc - 1
        ol_start_r = max(left_minr, right_minr)
        ol_end_r = min(left_maxr, right_maxr)
        purple_start_r = ol_start_r + 1
        purple_end_r = ol_end_r - 1
        if purple_start_r <= purple_end_r:
            for r in range(purple_start_r, purple_end_r + 1):
                for c in range(gap_start_c, gap_end_c + 1):
                    output[r][c] = 8

    elif col_ol_len > 0 and row_ol_len == 0:
        # vertical separation
        if minr1 < minr2:
            top_minr, top_maxr, top_minc, top_maxc = minr1, maxr1, minc1, maxc1
            bot_minr, bot_maxr, bot_minc, bot_maxc = minr2, maxr2, minc2, maxc2
        else:
            top_minr, top_maxr, top_minc, top_maxc = minr2, maxr2, minc2, maxc2
            bot_minr, bot_maxr, bot_minc, bot_maxc = minr1, maxr1, minc1, maxc1
        gap_start_r = top_maxr + 1
        gap_end_r = bot_minr - 1
        ol_start_c = max(top_minc, bot_minc)
        ol_end_c = min(top_maxc, bot_maxc)
        purple_start_c = ol_start_c + 1
        purple_end_c = ol_end_c - 1
        if purple_start_c <= purple_end_c:
            for r in range(gap_start_r, gap_end_r + 1):
                for c in range(purple_start_c, purple_end_c + 1):
                    output[r][c] = 8

    return output

def solve_457(I):
    rows = len(I)
    cols = len(I[0])
    # Find purple bbox
    min_r, max_r, min_c, max_c = rows, -1, cols, -1
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 8:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    # Collect colors for each quadrant
    tl, tr, bl, br = 0, 0, 0, 0
    for r in range(rows):
        for c in range(cols):
            color = I[r][c]
            if color != 0 and color != 8:
                if r < min_r and c < min_c:
                    tl = color
                elif r < min_r and c > max_c:
                    tr = color
                elif r > max_r and c < min_c:
                    bl = color
                elif r > max_r and c > max_c:
                    br = color
    # Create new I all 0
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    # Set the positions
    new_grid[min_r][min_c] = tl
    new_grid[min_r][max_c] = tr
    new_grid[max_r][min_c] = bl
    new_grid[max_r][max_c] = br
    return new_grid

def solve_458(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 3:
                for di, dj in directions:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and I[ni][nj] == 2:
                        output[i][j] = 8
                        output[ni][nj] = 0
    return output

def solve_459(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if I[r][c] == 0:
                continue
            up = I[r - 1][c]
            down = I[r + 1][c]
            left = I[r][c - 1]
            right = I[r][c + 1]
            if up == down == left == right != 0 and up != I[r][c]:
                return [[I[r][c]]]
    return []

import copy

def solve_460(I):
    output = copy.deepcopy(I)
    rows = len(I)
    cols = len(I[0]) if rows > 0 else 0
    visited = set()
    components = []

    def dfs(r, c, comp):
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            comp.append((cr, cc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] != 0 and (nr, nc) not in visited:
                    stack.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and (r, c) not in visited:
                comp = []
                dfs(r, c, comp)
                components.append(comp)

    if len(components) != 2:
        return output  # Assume exactly two, but return unchanged if not

    bbs = []
    for comp in components:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        min_r = min(rs)
        max_r = max(rs)
        min_c = min(cs)
        max_c = max(cs)
        bbs.append((min_r, max_r, min_c, max_c))

    bb1, bb2 = bbs
    min_r1, max_r1, min_c1, max_c1 = bb1
    min_r2, max_r2, min_c2, max_c2 = bb2

    row_overlap = max(min_r1, min_r2) <= min(max_r1, max_r2)
    col_overlap = max(min_c1, min_c2) <= min(max_c1, max_c2)

    if row_overlap and not col_overlap:
        # Horizontal separation
        if min_c1 < min_c2:
            left_max_c = max_c1
            right_min_c = min_c2
        else:
            left_max_c = max_c2
            right_min_c = min_c1
        start_c = left_max_c + 1
        end_c = right_min_c - 1
        for c in range(start_c, end_c + 1):
            for r in range(rows):
                output[r][c] = 3
    elif col_overlap and not row_overlap:
        # Vertical separation
        if min_r1 < min_r2:
            upper_max_r = max_r1
            lower_min_r = min_r2
        else:
            upper_max_r = max_r2
            lower_min_r = min_r1
        start_r = upper_max_r + 1
        end_r = lower_min_r - 1
        for r in range(start_r, end_r + 1):
            for c in range(cols):
                output[r][c] = 3

    return output

def solve_461(I):
    if not I or len(I) != 3 or len(I[0]) != 6:
        return []  # Invalid input, but assuming always 3x6 as per examples

    output = [[0 for _ in range(3)] for _ in range(3)]

    for i in range(3):
        for j in range(3):
            left_has_yellow = I[i][j] == 4
            right_has_green = I[i][j + 3] == 3
            if left_has_yellow or right_has_green:
                output[i][j] = 6

    return output

def solve_462(I):
    if not I or not I[0]:
        return I

    rows = len(I)
    cols = len(I[0])

    # Find central column and height h
    center = -1
    h = 0
    for c in range(cols):
        if I[0][c] == 7:
            this_h = 0
            for r in range(rows):
                if I[r][c] == 7:
                    this_h += 1
                else:
                    break
            center = c
            h = this_h
            break

    if center == -1:
        return [row[:] for row in I]

    # Initialize output with all 0s
    output = [[0 for _ in range(cols)] for _ in range(rows)]

    # Fill the pattern
    for r in range(h):
        max_level = h - 1 - r
        left_extend = min(center, max_level)
        right_extend = min(cols - 1 - center, max_level)
        leftmost = center - left_extend
        rightmost = center + right_extend
        for c in range(leftmost, rightmost + 1):
            level = abs(c - center)
            output[r][c] = 7 if level % 2 == 0 else 8

    return output

def solve_463(I):
    if not I or not I[0]:
        return I
    
    height = len(I)
    width = len(I[0])
    
    # Copy the I to modify
    new_grid = [row[:] for row in I]
    
    # Process horizontal fills for each row
    for r in range(height):
        blues = [c for c in range(width) if I[r][c] == 1]
        if len(blues) < 2:
            continue
        blues.sort()
        for i in range(len(blues) - 1):
            start = blues[i] + 1
            end = blues[i + 1]
            for c in range(start, end):
                new_grid[r][c] = 8
    
    # Process vertical fills for each column
    for c in range(width):
        blues = [r for r in range(height) if I[r][c] == 1]
        if len(blues) < 2:
            continue
        blues.sort()
        for i in range(len(blues) - 1):
            start = blues[i] + 1
            end = blues[i + 1]
            for r in range(start, end):
                new_grid[r][c] = 8
    
    return new_grid

def solve_464(I):
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 2:
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and output[nr][nc] == 0:
                            output[nr][nc] = 1
    return output

def solve_465(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    pos3 = None
    pos4 = None
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 3:
                pos3 = (r, c)
            elif I[r][c] == 4:
                pos4 = (r, c)
    if pos3 is None or pos4 is None:
        return I
    r3, c3 = pos3
    r4, c4 = pos4
    dr = r4 - r3
    dc = c4 - c3
    move_r = 0
    if dr > 0:
        move_r = 1
    elif dr < 0:
        move_r = -1
    move_c = 0
    if dc > 0:
        move_c = 1
    elif dc < 0:
        move_c = -1
    new_r = r3 + move_r
    new_c = c3 + move_c
    output = [row[:] for row in I]
    output[r3][c3] = 0
    output[new_r][new_c] = 3
    return output

def solve_466(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    # Find C and N
    N = 0
    C = -1
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 8:
                if C == -1:
                    C = I[r][c]
                N += 1
    # Middle row
    mid = rows // 2
    # Left padding
    left = (cols - N) // 2
    # Create output
    output = [[8 for _ in range(cols)] for _ in range(rows)]
    # Set the line
    for i in range(N):
        output[mid][left + i] = C
    return output

from collections import deque

def solve_467(I):
    if not I or not I[0]:
        return []
    
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    
    sources = []
    for c in range(cols):
        if I[0][c] != 0:
            sources.append((c, I[0][c]))
    
    def flood_fill(out, start_r, start_c, new_color):
        if out[start_r][start_c] != 5:
            return
        q = deque([(start_r, start_c)])
        while q:
            x, y = q.popleft()
            if out[x][y] == 5:
                out[x][y] = new_color
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and out[nx][ny] == 5:
                        q.append((nx, ny))
    
    for c, color in sources:
        hit_r = None
        for r in range(1, rows):
            if I[r][c] == 5:
                hit_r = r
                break
        if hit_r is not None:
            flood_fill(output, hit_r, c, color)
    
    return output

from collections import Counter

def solve_468(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    all_cells = [cell for row in I for cell in row]
    freq = Counter(all_cells)
    if not freq:
        return []
    dot_color = min(freq, key=freq.get)
    dot_pos = [(i, j) for i in range(rows) for j in range(cols) if I[i][j] == dot_color]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    region_counts = Counter()
    for r, c in dot_pos:
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append(I[nr][nc])
        if neighbors:
            neigh_color = neighbors[0]
            if all(x == neigh_color for x in neighbors) and neigh_color != dot_color:
                region_counts[neigh_color] += 1
    if not region_counts:
        return []
    winner = max(region_counts, key=region_counts.get)
    return [[winner]]

def solve_469(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    # Row fills
    for r in range(rows):
        purple_cols = [c for c in range(cols) if I[r][c] == 8]
        if len(purple_cols) >= 2:
            min_c = min(purple_cols)
            max_c = max(purple_cols)
            for c in range(min_c, max_c + 1):
                output[r][c] = 8
    # Column fills
    for c in range(cols):
        purple_rows = [r for r in range(rows) if I[r][c] == 8]
        if len(purple_rows) >= 2:
            min_r = min(purple_rows)
            max_r = max(purple_rows)
            for rr in range(min_r, max_r + 1):
                output[rr][c] = 8
    return output

def solve_470(I):
    if not I or not I[0]:
        return I
    h = len(I)
    w = len(I[0])
    output = [row[:] for row in I]
    visited = [[False] * w for _ in range(h)]
    floating_cells = []
    color = None
    for i in range(h):
        for j in range(w):
            if I[i][j] != 7 and I[i][j] != 4 and not visited[i][j]:
                color = I[i][j]
                stack = [(i, j)]
                component = []
                visited[i][j] = True
                while stack:
                    r, c = stack.pop()
                    component.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                floating_cells = component  # assume only one
    if not floating_cells:
        return output
    min_r = min(r for r, c in floating_cells)
    max_r = max(r for r, c in floating_cells)
    height = max_r - min_r + 1
    new_min_r = h - height
    row_shift = new_min_r - min_r
    min_c = min(c for r, c in floating_cells)
    max_c = max(c for r, c in floating_cells)
    new_min_c = (w - 1) - max_c
    col_shift = new_min_c - min_c
    for r, c in floating_cells:
        output[r][c] = 7
        new_r = r + row_shift
        new_c = c + col_shift
        output[new_r][new_c] = I[r][c]
    return output

import copy

def solve_471(I):
    if not I or not I[0]:
        return []
    
    rows = len(I)
    cols = len(I[0])
    output = copy.deepcopy(I)
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    
    def dfs(r, c, component):
        stack = [(r, c)]
        visited[r][c] = True
        component.append((r, c))
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        while stack:
            cr, cc = stack.pop()
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == 0:
                    visited[nr][nc] = True
                    component.append((nr, nc))
                    stack.append((nr, nc))
    
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 0 and not visited[i][j]:
                component = []
                dfs(i, j, component)
                if len(component) >= 2:
                    for r, c in component:
                        output[r][c] = 8
    
    return output

def solve_472(I):
    output = [[2 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            if I[i][j] == 0 and I[i][j + 4] == 0:
                output[i][j] = 0
    return output

def solve_473(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [[8 for _ in range(cols)] for _ in range(rows)]
    max_c = cols - 1
    col = 0
    velocity = 1
    for r in range(rows - 1, -1, -1):
        output[r][col] = 1
        next_col = col + velocity
        if next_col < 0:
            next_col = -next_col
            velocity = -velocity
        elif next_col > max_c:
            next_col = 2 * max_c - next_col
            velocity = -velocity
        col = next_col
    return output

def solve_474(I):
    if not I or not I[0]:
        return []

    rows = len(I)
    cols = len(I[0])

    def find_rectangles():
        visited = [[False] * cols for _ in range(rows)]
        rects = []
        for r in range(rows):
            for c in range(cols):
                if I[r][c] != 0 and not visited[r][c]:
                    color = I[r][c]
                    # Extend right
                    c_right = c
                    while c_right + 1 < cols and I[r][c_right + 1] == color and not visited[r][c_right + 1]:
                        c_right += 1
                    # Extend down
                    r_bottom = r
                    while True:
                        next_r = r_bottom + 1
                        if next_r >= rows:
                            break
                        good = True
                        for cc in range(c, c_right + 1):
                            if I[next_r][cc] != color or visited[next_r][cc]:
                                good = False
                                break
                        if not good:
                            break
                        r_bottom = next_r
                    # Mark visited
                    for rr in range(r, r_bottom + 1):
                        for cc in range(c, c_right + 1):
                            visited[rr][cc] = True
                    # Add rect (min_r, max_r, min_c, max_c, color)
                    rects.append((r, r_bottom, c, c_right, color))
        return rects

    rects = find_rectangles()

    h_set = set()
    v_set = set()
    for min_r, max_r, min_c, max_c, _ in rects:
        h_set.add(min_r)
        h_set.add(max_r + 1)
        v_set.add(min_c)
        v_set.add(max_c + 1)

    h_lines = sorted(h_set)
    v_lines = sorted(v_set)

    num_log_rows = len(h_lines) - 1
    num_log_cols = len(v_lines) - 1

    if num_log_rows == 0 or num_log_cols == 0:
        return []

    output = [[0] * num_log_cols for _ in range(num_log_rows)]

    for i in range(num_log_rows):
        pixel_r = h_lines[i]
        for j in range(num_log_cols):
            pixel_c = v_lines[j]
            if 0 <= pixel_r < rows and 0 <= pixel_c < cols:
                output[i][j] = I[pixel_r][pixel_c]

    return output

def solve_475(I):
    rows = len(I)
    cols = len(I[0])
    structures = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            if I[i][j] == 2 and I[i][j + 1] == 2 and I[i + 1][j] == 2 and I[i + 1][j + 1] == 2:
                structures.append((i, j))
    pulls = []
    for min_r, min_c in structures:
        max_r = min_r + 1
        max_c = min_c + 1
        # Up
        candidates = [(r, c) for r in range(min_r) for c in range(min_c, max_c + 1) if I[r][c] == 1]
        if candidates:
            maxrow = max(r for r, c in candidates)
            cands = [(r, c) for r, c in candidates if r == maxrow]
            chosen = min(cands, key=lambda x: x[1])
            pulls.append((min_r - 1, chosen[1], chosen[0], chosen[1]))
        # Down
        candidates = [(r, c) for r in range(max_r + 1, rows) for c in range(min_c, max_c + 1) if I[r][c] == 1]
        if candidates:
            minrow = min(r for r, c in candidates)
            cands = [(r, c) for r, c in candidates if r == minrow]
            chosen = min(cands, key=lambda x: x[1])
            pulls.append((max_r + 1, chosen[1], chosen[0], chosen[1]))
        # Left
        candidates = [(r, c) for c in range(min_c) for r in range(min_r, max_r + 1) if I[r][c] == 1]
        if candidates:
            maxcol = max(c for r, c in candidates)
            cands = [(r, c) for r, c in candidates if c == maxcol]
            chosen = min(cands, key=lambda x: x[0])
            pulls.append((chosen[0], min_c - 1, chosen[0], chosen[1]))
        # Right
        candidates = [(r, c) for c in range(max_c + 1, cols) for r in range(min_r, max_r + 1) if I[r][c] == 1]
        if candidates:
            mincol = min(c for r, c in candidates)
            cands = [(r, c) for r, c in candidates if c == mincol]
            chosen = min(cands, key=lambda x: x[0])
            pulls.append((chosen[0], max_c + 1, chosen[0], chosen[1]))
    output = [row[:] for row in I]
    for nr, nc, orr, oc in pulls:
        output[nr][nc] = 1
        output[orr][oc] = 0
    return output

def solve_476(I):
    height = len(I)
    if height == 0:
        return I
    width = len(I[0])
    visited = [[False] * width for _ in range(height)]
    components = []
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]  # 8-way

    for r in range(height):
        for c in range(width):
            if I[r][c] != 0 and not visited[r][c]:
                color = I[r][c]
                component_cells = []
                stack = [(r, c)]
                visited[r][c] = True
                min_r = r
                max_r = r
                while stack:
                    cr, cc = stack.pop()
                    component_cells.append((cr, cc))
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < height and 0 <= nc < width and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                shape = []
                for cr, cc in component_cells:
                    offset = cr - min_r
                    shape.append((offset, cc))
                comp = {'min_r': min_r, 'height': max_r - min_r + 1, 'shape': shape, 'color': color}
                components.append(comp)

    if not components:
        return [row[:] for row in I]

    components.sort(key=lambda x: x['min_r'])
    components = components[::-1]
    overall_start = min(comp['min_r'] for comp in components)
    output = [[0] * width for _ in range(height)]
    current_r = overall_start
    for comp in components:
        for offset, cc in comp['shape']:
            output[current_r + offset][cc] = comp['color']
        current_r += comp['height']
    return output

from collections import defaultdict

def solve_477(I):
    if not I or not I[0]:
        return I
    height = len(I)
    width = len(I[0])
    
    # Find color counts
    color_count = defaultdict(int)
    positions = {}
    for r in range(height):
        for c in range(width):
            col = I[r][c]
            if col != 0:
                color_count[col] += 1
                positions[col] = (r, c)
    
    # Find unique color (appears once)
    unique_colors = [col for col, cnt in color_count.items() if cnt == 1]
    if len(unique_colors) != 1:
        return I  # Safety, but assuming always one
    center_col = unique_colors[0]
    center_r, center_c = positions[center_col]
    
    # Extract vertical sequence
    vert_rows = sorted([rr for rr in range(height) if I[rr][center_c] != 0])
    S_vert = [I[rr][center_c] for rr in vert_rows]
    L_v = len(S_vert)
    
    # Find matching shift for vertical
    vert_shifted = None
    for k in range(L_v):
        shifted = S_vert[k:] + S_vert[:k]
        match = True
        for i, rr in enumerate(vert_rows):
            if shifted[rr % L_v] != S_vert[i]:
                match = False
                break
        if match:
            vert_shifted = shifted
            break
    
    # Extract horizontal sequence
    hor_cols = sorted([cc for cc in range(width) if I[center_r][cc] != 0])
    S_hor = [I[center_r][cc] for cc in hor_cols]
    L_h = len(S_hor)
    
    # Find matching shift for horizontal
    hor_shifted = None
    for k in range(L_h):
        shifted = S_hor[k:] + S_hor[:k]
        match = True
        for i, cc in enumerate(hor_cols):
            if shifted[cc % L_h] != S_hor[i]:
                match = False
                break
        if match:
            hor_shifted = shifted
            break
    
    # Create output I
    output = [[0] * width for _ in range(height)]
    
    # Fill vertical (center column)
    if vert_shifted:
        for r in range(height):
            output[r][center_c] = vert_shifted[r % L_v]
    
    # Fill horizontal (center row)
    if hor_shifted:
        for c in range(width):
            output[center_r][c] = hor_shifted[c % L_h]
    
    return output

import collections

def solve_478(I):
    if not I or not I[0]:
        return I

    rows = len(I)
    cols = len(I[0])

    def get_majority(vals):
        if not vals:
            return 0
        counts = collections.Counter(vals)
        max_count = max(counts.values())
        candidates = [k for k in counts if counts[k] == max_count]
        return min(candidates)

    # Per column majority
    col_maj = [[0] * cols for _ in range(rows)]
    for c in range(cols):
        vals = [I[r][c] for r in range(rows)]
        maj = get_majority(vals)
        for r in range(rows):
            col_maj[r][c] = maj

    diff_col = sum(1 for r in range(rows) for c in range(cols) if I[r][c] != col_maj[r][c])

    # Per row majority
    row_maj = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        vals = [I[r][c] for c in range(cols)]
        maj = get_majority(vals)
        for c in range(cols):
            row_maj[r][c] = maj

    diff_row = sum(1 for r in range(rows) for c in range(cols) if I[r][c] != row_maj[r][c])

    if diff_col <= diff_row:
        return col_maj
    else:
        return row_maj

def solve_479(I):
    if not I:
        return []
    rows = len(I)
    output = [[0 for _ in range(4)] for _ in range(rows)]
    for i in range(rows):
        for j in range(4):
            if I[i][j] == 0 and I[i][j + 4] == 0:
                output[i][j] = 4
    return output

def solve_480(I):
    if not I or not I[0]:
        return I

    h = len(I)
    w = len(I[0])

    # Find all colored positions
    positions = [(r, c) for r in range(h) for c in range(w) if I[r][c] != 0]

    if not positions:
        return [row[:] for row in I]

    # Find min and max row and col
    min_r = min(r for r, c in positions)
    max_r = max(r for r, c in positions)
    min_c = min(c for r, c in positions)
    max_c = max(c for r, c in positions)

    # Find center (position of 2)
    centers = [(r, c) for r, c in positions if I[r][c] == 2]
    assert len(centers) == 1, "Exactly one center (color 2) expected"
    center_r, center_c = centers[0]

    # Find top color (unique colored cell in min_r)
    top_cs = [c for r, c in positions if r == min_r]
    assert len(top_cs) == 1, "Unique top seed expected"
    top_color = I[min_r][top_cs[0]]

    # Find bottom color
    bottom_cs = [c for r, c in positions if r == max_r]
    assert len(bottom_cs) == 1
    bottom_color = I[max_r][bottom_cs[0]]

    # Find left color
    left_rs = [r for r, c in positions if c == min_c]
    assert len(left_rs) == 1
    left_color = I[left_rs[0]][min_c]

    # Find right color
    right_rs = [r for r, c in positions if c == max_c]
    assert len(right_rs) == 1
    right_color = I[right_rs[0]][max_c]

    # Create output I
    output = [[0] * w for _ in range(h)]

    # Draw top border
    for c in range(min_c, max_c + 1):
        output[min_r][c] = top_color

    # Draw bottom border
    for c in range(min_c, max_c + 1):
        output[max_r][c] = bottom_color

    # Draw left border
    for r in range(min_r + 1, max_r):
        output[r][min_c] = left_color

    # Draw right border
    for r in range(min_r + 1, max_r):
        output[r][max_c] = right_color

    # Draw vertical arm
    for r in range(min_r + 1, max_r):
        output[r][center_c] = 5

    # Draw horizontal arm
    for c in range(min_c + 1, max_c):
        output[center_r][c] = 5

    # Set center to 2
    output[center_r][center_c] = 2

    return output

import numpy as np

def solve_481(I):
    if not I or not I[0]:
        return I
    I = np.array(I)
    rows, cols = I.shape
    visited = np.zeros((rows, cols), dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(r, c, color):
        component = []
        stack = [(r, c)]
        visited[r, c] = True
        while stack:
            cr, cc = stack.pop()
            component.append((cr, cc))
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and I[nr, nc] == color:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        return component

    components = []
    for i in range(rows):
        for j in range(cols):
            if I[i, j] != 0 and not visited[i, j]:
                color = I[i, j]
                comp = dfs(i, j, color)
                components.append((color, comp))

    # Find purple component
    purple_comp = None
    for color, comp in components:
        if color == 8:
            purple_comp = comp
            break

    if purple_comp is None:
        return I.tolist()

    min_row_p = min(r for r, c in purple_comp)

    output = I.copy()

    for color, comp in components:
        if color == 8:
            continue
        min_row_s = min(r for r, c in comp)
        delta = min_row_p - min_row_s
        # Clear old positions
        for r, c in comp:
            output[r, c] = 0
        # Set new positions
        for r, c in comp:
            nr = r + delta
            if 0 <= nr < rows:
                output[nr, c] = color

    return output.tolist()

import copy

def solve_482(I):
    if not I or not I[0]:
        return []
    
    h = len(I)
    w = len(I[0])
    output = copy.deepcopy(I)
    
    # Compute top_r for each column
    top_r = [h] * w  # Initialize to h (beyond I) in case no 6
    for c in range(w):
        for r in range(h):
            if I[r][c] == 6:
                top_r[c] = r
                break
    
    # Find minimal top_r (tallest peak), rightmost if tie
    min_top = min(top_r)
    peak_col = max(c for c in range(w) if top_r[c] == min_top)
    
    # Find maximal top_r (deepest valley), leftmost if tie
    max_top = max(top_r)
    valley_col = min(c for c in range(w) if top_r[c] == max_top)
    
    # Extend peak with yellow (4) from row 1 to top_r[peak_col]-1
    for r in range(1, top_r[peak_col]):
        output[r][peak_col] = 4
    
    # Extend valley with brown (9) from row 1 to top_r[valley_col]-1
    for r in range(1, top_r[valley_col]):
        output[r][valley_col] = 9
    
    return output

def solve_483(I):
    if not I or not I[0]:
        return I
    
    h = len(I)
    w = len(I[0])
    
    # Find C: the color of the cross (assuming consistent non-0, non-5 color)
    colors = set()
    for row in I:
        for cell in row:
            if cell != 0 and cell != 5:
                colors.add(cell)
    if len(colors) != 1:
        raise ValueError("Unexpected number of colors")
    C = list(colors)[0]
    
    # Count N: number of gray cells (5)
    N = sum(1 for row in I for cell in row if cell == 5)
    
    # Find original horizontal row: row with max count of C
    row_counts = [sum(1 for cell in row if cell == C) for row in I]
    original_row = row_counts.index(max(row_counts))
    
    # Find original vertical column: column with max count of C
    col_counts = [sum(1 for r in range(h) if I[r][c] == C) for c in range(w)]
    original_col = col_counts.index(max(col_counts))
    
    # Compute new positions
    new_row = original_row + N
    new_col = original_col - N
    
    # Create output I
    output = [[0] * w for _ in range(h)]
    
    # Set horizontal bar
    for c in range(w):
        output[new_row][c] = C
    
    # Set vertical bar
    for r in range(h):
        output[r][new_col] = C
    
    return output

def solve_484(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    
    # Find purple (8) and red (2) positions
    pr, pc = None, None
    rr, rc = None, None
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 8:
                pr, pc = r, c
            elif I[r][c] == 2:
                rr, rc = r, c
    
    if pr is None or rr is None:
        return I  # No change if missing
    
    # Determine slide direction opposite to red
    dir_sign = 1 if rc < pc else -1
    
    # Find slide column
    step = 1
    new_c = None
    while True:
        candidate_c = pc + step * dir_sign
        if candidate_c < 0 or candidate_c >= cols:
            break  # Should not happen based on examples
        if pr + 1 < rows and I[pr + 1][candidate_c] == 7:
            new_c = candidate_c
            break
        step += 1
    
    if new_c is None:
        return I  # No valid position, though unlikely
    
    # Fall down in new_c
    r = pr + 1
    while r < rows and I[r][new_c] == 7:
        r += 1
    new_pr = r - 1
    
    # Create output
    output = [row[:] for row in I]
    
    # Move red to old purple position
    output[pr][pc] = 2
    
    # Set old red to 7
    output[rr][rc] = 7
    
    # Set new purple position
    output[new_pr][new_c] = 8
    
    return output

def solve_485(I):
    height = len(I)
    if height == 0:
        return []
    width = len(I[0])

    visited = [[False] * width for _ in range(height)]
    components = []

    for r in range(height):
        for c in range(width):
            if I[r][c] != 0 and not visited[r][c]:
                minr, maxr = r, r
                minc, maxc = c, c
                count2 = 1 if I[r][c] == 2 else 0
                stack = [(r, c)]
                visited[r][c] = True
                area = 1  # count number of cells in component for area

                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < height and 0 <= nc < width and I[nr][nc] != 0 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                            minr = min(minr, nr)
                            maxr = max(maxr, nr)
                            minc = min(minc, nc)
                            maxc = max(maxc, nc)
                            if I[nr][nc] == 2:
                                count2 += 1
                            area += 1

                components.append({
                    'minr': minr, 'maxr': maxr, 'minc': minc, 'maxc': maxc,
                    'count2': count2, 'area': area
                })

    if not components:
        return []

    max_count2 = max(c['count2'] for c in components)
    candidates = [c for c in components if c['count2'] == max_count2]

    if len(candidates) > 1:
        max_area = max(c['area'] for c in candidates)
        candidates = [c for c in candidates if c['area'] == max_area]

    if len(candidates) > 1:
        min_minr = min(c['minr'] for c in candidates)
        candidates = [c for c in candidates if c['minr'] == min_minr]

    if len(candidates) > 1:
        min_minc = min(c['minc'] for c in candidates)
        candidates = [c for c in candidates if c['minc'] == min_minc]

    selected = candidates[0]

    out = []
    for r in range(selected['minr'], selected['maxr'] + 1):
        row = []
        for c in range(selected['minc'], selected['maxc'] + 1):
            row.append(I[r][c])
        out.append(row)

    return out

def solve_486(I):
    if not I:
        return []
    h = len(I)
    w = len(I[0])
    # Find background color (first non-0)
    background = None
    for row in I:
        for cell in row:
            if cell != 0:
                background = cell
                break
        if background is not None:
            break
    # Assume square I, k=3
    k = 3
    super_size = h // k
    # Initialize output
    output = [[background for _ in range(k)] for _ in range(k)]
    # Set 0 in super positions where input has 0
    for r in range(h):
        for c in range(w):
            if I[r][c] == 0:
                sr = r // super_size
                sc = c // super_size
                output[sr][sc] = 0
    return output

def solve_487(I):
    # Expand each row horizontally
    expanded_rows = []
    for row in I:
        a, b, c = row
        expanded = [a, a, b, c, c]
        expanded_rows.append(expanded)
    
    # Create output by duplicating rows vertically
    output = []
    output.append(expanded_rows[0])
    output.append(expanded_rows[0])
    output.append(expanded_rows[1])
    output.append(expanded_rows[2])
    output.append(expanded_rows[2])
    
    return output

from collections import deque
import copy

def solve_488(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    visited = set()
    queue = deque()

    # Add all 0 cells on top and bottom rows to starting points
    for col in range(cols):
        if I[0][col] == 0:
            queue.append((0, col))
            visited.add((0, col))
        if I[rows - 1][col] == 0:
            queue.append((rows - 1, col))
            visited.add((rows - 1, col))

    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # BFS to mark all reachable 0 cells
    while queue:
        row, col = queue.popleft()
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))

    # Create output I and change unvisited 0's to 4
    output = copy.deepcopy(I)
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 0 and (r, c) not in visited:
                output[r][c] = 4

    return output

def solve_489(I):
    if not I or not I[0]:
        return I
    height = len(I)
    width = len(I[0])
    
    # Find horizontal separators: rows that are all 0
    h_seps = [r for r in range(height) if all(I[r][c] == 0 for c in range(width))]
    
    # Find vertical separators: columns that are all 0
    v_seps = [c for c in range(width) if all(I[r][c] == 0 for r in range(height))]
    
    # Tile row starts and ends
    row_starts = [0] + [s + 1 for s in h_seps]
    row_ends = [s - 1 for s in h_seps] + [height - 1]
    tile_rows = [(start, end) for start, end in zip(row_starts, row_ends) if start <= end]
    
    # Tile column starts and ends
    col_starts = [0] + [s + 1 for s in v_seps]
    col_ends = [s - 1 for s in v_seps] + [width - 1]
    tile_cols = [(start, end) for start, end in zip(col_starts, col_ends) if start <= end]
    
    # Find the pattern tile
    pattern = None
    pattern_tile_r = None
    pattern_tile_c = None
    for tr in tile_rows:
        for tc in tile_cols:
            subgrid = [[I[r][c] for c in range(tc[0], tc[1] + 1)] for r in range(tr[0], tr[1] + 1)]
            has_special = any(val != 0 and val != 7 for row in subgrid for val in row)
            if has_special:
                pattern = subgrid
                pattern_tile_r = tr
                pattern_tile_c = tc
                break
        if pattern:
            break
    
    if not pattern:
        return I  # No pattern found, return unchanged
    
    # Copy pattern to all tiles
    for tr in tile_rows:
        for tc in tile_cols:
            for i, r in enumerate(range(tr[0], tr[1] + 1)):
                for j, c in enumerate(range(tc[0], tc[1] + 1)):
                    I[r][c] = pattern[i][j]
    
    return I

import numpy as np

def solve_490(I):
    if not I:
        return I
    rows = len(I)
    cols = len(I[0]) if rows else 0
    color_ranges = {}
    for r in range(rows):
        for c in range(cols):
            val = I[r][c]
            if val != 0:
                if val not in color_ranges:
                    color_ranges[val] = [r, r]
                else:
                    color_ranges[val][0] = min(color_ranges[val][0], r)
                    color_ranges[val][1] = max(color_ranges[val][1], r)
    output = [row[:] for row in I]
    for colr, (min_r, max_r) in color_ranges.items():
        block_rows = [I[r][:] for r in range(min_r, max_r + 1)]
        block_rows.reverse()
        for i, r in enumerate(range(min_r, max_r + 1)):
            output[r] = block_rows[i]
    return output

def solve_491(I):
    if not I:
        return I
    rows = len(I)
    cols = len(I[0]) if rows else 0
    positions = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 8:
                positions.append((r, c))
    if not positions:
        return [row[:] for row in I]
    rs = [p[0] for p in positions]
    cs = [p[1] for p in positions]
    min_r = min(rs)
    max_r = max(rs)
    min_c = min(cs)
    max_c = max(cs)
    output = [row[:] for row in I]
    # Top border
    for c in range(min_c, max_c + 1):
        if output[min_r][c] == 0:
            output[min_r][c] = 1
    # Bottom border
    for c in range(min_c, max_c + 1):
        if output[max_r][c] == 0:
            output[max_r][c] = 1
    # Left border
    for r in range(min_r, max_r + 1):
        if output[r][min_c] == 0:
            output[r][min_c] = 1
    # Right border
    for r in range(min_r, max_r + 1):
        if output[r][max_c] == 0:
            output[r][max_c] = 1
    return output

def solve_492(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])

    # Find pattern cells: non-0, non-5
    pattern_cells = [(r, c) for r in range(rows) for c in range(cols) if I[r][c] != 0 and I[r][c] != 5]
    if not pattern_cells:
        return I

    min_r_p = min(r for r, c in pattern_cells)
    min_c_p = min(c for r, c in pattern_cells)
    max_r_p = max(r for r, c in pattern_cells)
    max_c_p = max(c for r, c in pattern_cells)
    height_p = max_r_p - min_r_p + 1
    width_p = max_c_p - min_c_p + 1

    # Find grey connected components (4-connected)
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 5 and not visited[r][c]:
                component = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    rr, cc = stack.pop()
                    component.append((rr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == 5:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append(component)

    # Copy I
    output = [row[:] for row in I]

    # For each component, copy pattern aligning top-left of bounding boxes
    for comp in components:
        if not comp:
            continue
        min_r_t = min(r for r, c in comp)
        min_c_t = min(c for r, c in comp)
        # Assuming same height and width as pattern
        for rel_r in range(height_p):
            for rel_c in range(width_p):
                color = I[min_r_p + rel_r][min_c_p + rel_c]
                tr = min_r_t + rel_r
                tc = min_c_t + rel_c
                if 0 <= tr < rows and 0 <= tc < cols:
                    output[tr][tc] = color

    return output

def solve_493(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Find outer bounding box of 2's
    outer_min_r, outer_max_r = rows, -1
    outer_min_c, outer_max_c = cols, -1
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 2:
                outer_min_r = min(outer_min_r, r)
                outer_max_r = max(outer_max_r, r)
                outer_min_c = min(outer_min_c, c)
                outer_max_c = max(outer_max_c, c)
    
    if outer_min_r > outer_max_r:
        return []  # No frame, but assume there is
    
    outer_h = outer_max_r - outer_min_r + 1
    outer_w = outer_max_c - outer_min_c + 1
    inner_h = outer_h - 2
    inner_w = outer_w - 2
    
    # Find minimal bounding box of inner colors (!=0 and !=2)
    colored_min_r, colored_max_r = rows, -1
    colored_min_c, colored_max_c = cols, -1
    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and I[r][c] != 2:
                colored_min_r = min(colored_min_r, r)
                colored_max_r = max(colored_max_r, r)
                colored_min_c = min(colored_min_c, c)
                colored_max_c = max(colored_max_c, c)
    
    if colored_min_r > colored_max_r:
        # No colors, but assume there are
        pass
    
    colored_h = colored_max_r - colored_min_r + 1
    colored_w = colored_max_c - colored_min_c + 1
    
    scale_h = inner_h // colored_h
    scale_w = inner_w // colored_w
    
    # Extract colored subgrid
    colored_grid = []
    for i in range(colored_h):
        row = [I[colored_min_r + i][colored_min_c + j] for j in range(colored_w)]
        colored_grid.append(row)
    
    # Scale it
    scaled_inner = []
    for i in range(colored_h):
        orig_row = colored_grid[i]
        scaled_row = []
        for val in orig_row:
            scaled_row.extend([val] * scale_w)
        for _ in range(scale_h):
            scaled_inner.append(scaled_row)
    
    # Create output
    output = [[0] * outer_w for _ in range(outer_h)]
    
    # Set border to 2
    for c in range(outer_w):
        output[0][c] = 2
        output[outer_h - 1][c] = 2
    for r in range(outer_h):
        output[r][0] = 2
        output[r][outer_w - 1] = 2
    
    # Place scaled inner
    for ri in range(inner_h):
        for ci in range(inner_w):
            output[1 + ri][1 + ci] = scaled_inner[ri][ci]
    
    return output

import numpy as np

def solve_494(I):
    I = np.array(I)
    rows, cols = I.shape
    
    # Find height h of grey (5) in column 0
    grey_col = 0
    h = 0
    for r in range(rows):
        if I[r, grey_col] == 5:
            h += 1
        else:
            break
    
    # Find start_col of right uniform bars
    start_col = cols
    for c in range(cols - 1, -1, -1):
        color = I[0, c]
        if color == 0:
            break
        is_uniform = all(I[r, c] == color for r in range(rows))
        if not is_uniform:
            break
        start_col = c
    
    # Collect colors sequence
    colors = [I[0, c] for c in range(start_col, cols)]
    k = len(colors)
    if k == 0:
        return I.tolist()
    
    new_col = start_col - 1
    output = I.copy()
    
    # Clear original right bars
    for c in range(start_col, cols):
        for r in range(rows):
            output[r, c] = 0
    
    # Place new bar
    for r in range(rows):
        seg = r // h
        idx = seg % k
        output[r, new_col] = colors[idx]
    
    return output.tolist()

import numpy as np

def solve_495(I):
    if not I or not I[0]:
        return I
    
    I = np.array(I)
    rows, cols = I.shape
    
    # Find min_r and max_r
    non_zero_rows = np.where(np.any(I != 0, axis=1))[0]
    if len(non_zero_rows) == 0:
        return I.tolist()
    min_r = np.min(non_zero_rows)
    max_r = np.max(non_zero_rows)
    
    H = max_r - min_r + 1
    split = H // 2
    bottom_start = min_r + (H - split)
    
    # Copy I
    output = I.copy()
    
    # Change 1 to 2 in bottom rows
    for r in range(bottom_start, max_r + 1):
        for c in range(cols):
            if output[r, c] == 1:
                output[r, c] = 2
    
    return output.tolist()

def solve_496(I):
    starts = [0, 6, 12, 18, 24]
    output = [[0 for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            colors = set()
            for k in range(5):
                for l in range(5):
                    r = starts[k] + i
                    c = starts[l] + j
                    colors.add(I[r][c])
            if len(colors) == 1:
                output[i][j] = next(iter(colors))
            else:
                output[i][j] = 1
    return output

def solve_497(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    output = [row[:] for row in I]
    visited = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 0 and (r, c) not in visited:
                component = []
                stack = [(r, c)]
                visited.add((r, c))
                while stack:
                    cr, cc = stack.pop()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] == 0 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
                size = len(component)
                color = 4 - size
                for pr, pc in component:
                    output[pr][pc] = color
    return output

def solve_498(I):
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 0:
                is_interior = True
                for di, dj in directions:
                    ni = i + di
                    nj = j + dj
                    if not (0 <= ni < rows and 0 <= nj < cols) or I[ni][nj] != 0:
                        is_interior = False
                        break
                if is_interior:
                    output[i][j] = 8
    return output

def solve_499(I):
    rows = len(I)
    cols = len(I[0]) if rows > 0 else 0
    positions = []
    for r in range(rows):
        for c in range(cols):
            if I[r][c] == 1:
                positions.append((r, c))
    if len(positions) != 2:
        return I
    (r1, c1), (r2, c2) = positions
    center_r = (r1 + r2) // 2
    center_c = (c1 + c2) // 2
    output = [row[:] for row in I]
    # Set the plus shape if within bounds
    if 0 <= center_r < rows and 0 <= center_c - 1 < cols:
        output[center_r][center_c - 1] = 3
    if 0 <= center_r < rows and 0 <= center_c < cols:
        output[center_r][center_c] = 3
    if 0 <= center_r < rows and 0 <= center_c + 1 < cols:
        output[center_r][center_c + 1] = 3
    if 0 <= center_r - 1 < rows and 0 <= center_c < cols:
        output[center_r - 1][center_c] = 3
    if 0 <= center_r + 1 < rows and 0 <= center_c < cols:
        output[center_r + 1][center_c] = 3
    return output

def solve_500(I):
    height = len(I)
    width = len(I[0])
    
    # Find the gray row
    gray_row = None
    for r in range(height):
        if all(c == 5 for c in I[r]):
            gray_row = r
            break
    
    # Assume upper and lower have same height
    upper_height = gray_row
    lower_start = gray_row + 1
    out_height = upper_height
    
    out = [[0 for _ in range(width)] for _ in range(out_height)]
    
    for i in range(out_height):
        upper_row = I[i]
        lower_row = I[lower_start + i]
        for j in range(width):
            if upper_row[j] != 0:
                out[i][j] = upper_row[j]
            elif lower_row[j] != 0:
                out[i][j] = lower_row[j]
    
    return out

def solve_501(I):
    out = [[0 for _ in range(4)] for _ in range(5)]
    for i in range(5):
        for j in range(4):
            val_tl = I[i][j]
            val_tr = I[i][5 + j]
            val_bl = I[6 + i][j]
            val_br = I[6 + i][5 + j]
            if val_br != 0:
                out[i][j] = val_br
            elif val_tl != 0:
                out[i][j] = val_tl
            elif val_tr != 0:
                out[i][j] = val_tr
            elif val_bl != 0:
                out[i][j] = val_bl
            else:
                out[i][j] = 0
    return out

import numpy as np

def solve_502(I):
    if not I or not I[0]:
        return I
    
    I = np.array(I)
    rows, cols = I.shape
    visited = np.zeros((rows, cols), bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def dfs(r, c):
        stack = [(r, c)]
        component = []
        while stack:
            cr, cc = stack.pop()
            if visited[cr, cc]:
                continue
            visited[cr, cc] = True
            component.append((cr, cc))
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and I[nr, nc] == 5:
                    stack.append((nr, nc))
        return component
    
    gray_components = []
    for r in range(rows):
        for c in range(cols):
            if I[r, c] == 5 and not visited[r, c]:
                comp = dfs(r, c)
                gray_components.append(comp)
    
    output = I.copy()
    for comp in gray_components:
        rs = [pos[0] for pos in comp]
        cs = [pos[1] for pos in comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        n = max_r - min_r + 1
        h = n // 2
        mid_r = min_r + h
        mid_c = min_c + h
        
        tl = I[min_r - 1, min_c - 1]
        tr = I[min_r - 1, max_c + 1]
        bl = I[max_r + 1, min_c - 1]
        br = I[max_r + 1, max_c + 1]
        
        output[min_r - 1, min_c - 1] = 0
        output[min_r - 1, max_c + 1] = 0
        output[max_r + 1, min_c - 1] = 0
        output[max_r + 1, max_c + 1] = 0
        
        for r in range(min_r, mid_r):
            for c in range(min_c, mid_c):
                output[r, c] = tl
            for c in range(mid_c, max_c + 1):
                output[r, c] = tr
        
        for r in range(mid_r, max_r + 1):
            for c in range(min_c, mid_c):
                output[r, c] = bl
            for c in range(mid_c, max_c + 1):
                output[r, c] = br
    
    return output.tolist()

def solve_503(I):
    if not I or len(I) != 2:
        return I
    cols = len(I[0])
    color_top = I[0][0]
    color_bottom = I[1][0]
    output = [[0] * cols for _ in range(2)]
    for c in range(cols):
        if c % 2 == 0:
            output[0][c] = color_top
            output[1][c] = color_bottom
        else:
            output[0][c] = color_bottom
            output[1][c] = color_top
    return output

def solve_504(I):
    rows = len(I)
    cols = len(I[0]) if rows > 0 else 0
    visited = set()
    components = []

    def dfs(r, c):
        comp = []
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            comp.append((cr, cc))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and I[nr][nc] != 0 and (nr, nc) not in visited:
                    stack.append((nr, nc))
        return comp

    for r in range(rows):
        for c in range(cols):
            if I[r][c] != 0 and (r, c) not in visited:
                comp = dfs(r, c)
                components.append(comp)

    if not components:
        return []

    large_comp = max(components, key=len)

    repl = {}
    for comp in components:
        if comp == large_comp:
            continue
        if len(comp) != 2:
            continue
        pos1, pos2 = comp
        if pos1[0] != pos2[0]:
            continue
        if abs(pos1[1] - pos2[1]) != 1:
            continue
        if pos1[1] < pos2[1]:
            left = pos1
            right = pos2
        else:
            left = pos2
            right = pos1
        A = I[left[0]][left[1]]
        B = I[right[0]][right[1]]
        repl[B] = A

    rs = [p[0] for p in large_comp]
    cs = [p[1] for p in large_comp]
    min_r = min(rs)
    max_r = max(rs)
    min_c = min(cs)
    max_c = max(cs)
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    output = [[0] * width for _ in range(height)]

    for r, c in large_comp:
        out_r = r - min_r
        out_c = c - min_c
        color = I[r][c]
        output[out_r][out_c] = repl.get(color, color)

    return output

from collections import defaultdict
import copy

def solve_505(I):
    if not I or not I[0]:
        return I
    height = len(I)
    width = len(I[0])
    claims = defaultdict(list)
    
    # Find all 3x3 monochromatic squares
    for r in range(height - 2):
        for c in range(width - 2):
            # Collect the 9 cells
            cells = [I[r + i][c + j] for i in range(3) for j in range(3)]
            unique_colors = set(cells)
            if len(unique_colors) == 1:
                color = list(unique_colors)[0]
                if color > 0:
                    r_center = r + 1
                    c_center = c + 1
                    # Claim horizontal line
                    for k in range(width):
                        claims[(r_center, k)].append(color)
                    # Claim vertical line
                    for k in range(height):
                        claims[(k, c_center)].append(color)
    
    # Create output as copy of input
    output = copy.deepcopy(I)
    
    # Resolve claims
    for (r, c), ls in claims.items():
        s = set(ls)
        if len(s) == 1:
            output[r][c] = list(s)[0]
        elif len(s) > 1:
            output[r][c] = 0
    
    return output

def solve_506(I):
    if not I:
        return I
    rows = len(I)
    cols = len(I[0])

    # Find horiz_rows: rows where all cells == 3
    horiz_rows = []
    for r in range(rows):
        if all(I[r][c] == 3 for c in range(cols)):
            horiz_rows.append(r)

    if not horiz_rows:
        return [row[:] for row in I]

    # Find vert_cols: columns where there exists r not in horiz_rows with I[r][c] == 3
    vert_cols_set = set()
    for c in range(cols):
        for r in range(rows):
            if I[r][c] == 3 and r not in horiz_rows:
                vert_cols_set.add(c)
                break
    vert_cols = sorted(list(vert_cols_set))

    if not vert_cols:
        return [row[:] for row in I]

    # Build regions: list of (start_col, end_col, type) where type='left', 'right', 'center'
    regions = []
    # leftmost
    if vert_cols[0] > 0:
        regions.append((0, vert_cols[0] - 1, 'left'))
    # centers
    for i in range(len(vert_cols) - 1):
        s = vert_cols[i] + 1
        e = vert_cols[i + 1] - 1
        if s <= e:
            regions.append((s, e, 'center'))
    # rightmost
    if vert_cols[-1] < cols - 1:
        s = vert_cols[-1] + 1
        e = cols - 1
        if s <= e:
            regions.append((s, e, 'right'))

    # Make a copy
    output = [row[:] for row in I]

    # Fill top arm
    top_start = 0
    top_end = horiz_rows[0] - 1
    if top_start <= top_end:
        for reg_start, reg_end, reg_type in regions:
            if reg_type == 'left':
                color = 2
            elif reg_type == 'right':
                color = 4
            else:
                continue
            for r in range(top_start, top_end + 1):
                for c in range(reg_start, reg_end + 1):
                    if output[r][c] == 0:
                        output[r][c] = color

    # Fill bottom arm
    bot_start = horiz_rows[-1] + 1
    bot_end = rows - 1
    if bot_start <= bot_end:
        for reg_start, reg_end, reg_type in regions:
            if reg_type == 'left':
                color = 1
            elif reg_type == 'right':
                color = 8
            else:
                continue
            for r in range(bot_start, bot_end + 1):
                for c in range(reg_start, reg_end + 1):
                    if output[r][c] == 0:
                        output[r][c] = color

    # Fill middle sections
    for i in range(len(horiz_rows) - 1):
        mid_start = horiz_rows[i] + 1
        mid_end = horiz_rows[i + 1] - 1
        if mid_start <= mid_end:
            color = 7
            for reg_start, reg_end, reg_type in regions:
                if reg_type == 'center':
                    for r in range(mid_start, mid_end + 1):
                        for c in range(reg_start, reg_end + 1):
                            if output[r][c] == 0:
                                output[r][c] = color

    return output

def solve_507(I):
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    visited = [[False] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    components = []
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 5 and not visited[i][j]:
                component = []
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    r, c = stack.pop()
                    component.append((r, c))
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == 5:
                            visited[nr][nc] = True
                            stack.append((nr, nc))
                components.append(component)

    size_to_comp = {}
    for comp in components:
        size = len(comp)
        if size not in size_to_comp:
            size_to_comp[size] = []
        size_to_comp[size].append(comp)

    sizes = sorted(size_to_comp.keys(), reverse=True)
    color_map = {sizes[0]: 1, sizes[1]: 4, sizes[2]: 2}

    for size, color in color_map.items():
        for comp in size_to_comp[size]:
            for r, c in comp:
                output[r][c] = color

    return output

def solve_508(I):
    if not I or not I[0]:
        return I
    
    h = len(I)
    w = len(I[0])
    
    # Find the center (position of 0)
    ci, cj = None, None
    for i in range(h):
        for j in range(w):
            if I[i][j] == 0:
                ci, cj = i, j
                break
        if ci is not None:
            break
    
    # Create output I
    output = [row[:] for row in I]
    
    # Set the diagonals to 0
    for i in range(h):
        diff = i - ci
        j1 = cj + diff
        j2 = cj - diff
        if 0 <= j1 < w:
            output[i][j1] = 0
        if 0 <= j2 < w:
            output[i][j2] = 0
    
    return output

def solve_509(I):
    N = len(I) // 2
    out = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            G = I[i][j + N]
            P = I[i + N][j]
            B = I[i + N][j + N]
            Y = I[i][j]
            if G != 0:
                out[i][j] = G
            elif P != 0:
                out[i][j] = P
            elif B != 0:
                out[i][j] = B
            elif Y != 0:
                out[i][j] = Y
            else:
                out[i][j] = 0
    return out

def solve_510(I):
    if not I:
        return []
    h = len(I)
    def zigzag(start, max_idx):
        seq = list(range(start, max_idx + 1))
        seq.extend(range(max_idx - 1, -1, -1))
        return seq
    indices = zigzag(0, h - 1) + zigzag(1, h - 1)
    output = [I[i][:] for i in indices]
    return output

def solve_511(I):
    if not I or not I[0]:
        return []

    rows = len(I)
    cols = len(I[0])

    def find_rectangles():
        visited = [[False] * cols for _ in range(rows)]
        rects = []
        for r in range(rows):
            for c in range(cols):
                if I[r][c] != 0 and not visited[r][c]:
                    color = I[r][c]
                    # Extend right
                    c_right = c
                    while c_right + 1 < cols and I[r][c_right + 1] == color and not visited[r][c_right + 1]:
                        c_right += 1
                    # Extend down
                    r_bottom = r
                    while True:
                        next_r = r_bottom + 1
                        if next_r >= rows:
                            break
                        good = True
                        for cc in range(c, c_right + 1):
                            if I[next_r][cc] != color or visited[next_r][cc]:
                                good = False
                                break
                        if not good:
                            break
                        r_bottom = next_r
                    # Mark visited
                    for rr in range(r, r_bottom + 1):
                        for cc in range(c, c_right + 1):
                            visited[rr][cc] = True
                    # Add rect (min_r, max_r, min_c, max_c, color)
                    rects.append((r, r_bottom, c, c_right, color))
        return rects

    rects = find_rectangles()

    h_set = set()
    v_set = set()
    for min_r, max_r, min_c, max_c, _ in rects:
        h_set.add(min_r)
        h_set.add(max_r + 1)
        v_set.add(min_c)
        v_set.add(max_c + 1)

    h_lines = sorted(h_set)
    v_lines = sorted(v_set)

    num_log_rows = len(h_lines) - 1
    num_log_cols = len(v_lines) - 1

    if num_log_rows == 0 or num_log_cols == 0:
        return []

    output = [[0] * num_log_cols for _ in range(num_log_rows)]

    for i in range(num_log_rows):
        pixel_r = h_lines[i]
        for j in range(num_log_cols):
            pixel_c = v_lines[j]
            if 0 <= pixel_r < rows and 0 <= pixel_c < cols:
                output[i][j] = I[pixel_r][pixel_c]

    return output

def solve_512(I):
    if not I or not I[0]:
        return I
    n = len(I)
    new_grid = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            new_grid[n - 1 - c][r] = I[r][c]
    return new_grid

import numpy as np

def solve_513(I):
    g = np.array(I)
    # Extract left 3x3: rows 1-3, cols 1-3
    left = g[1:4, 1:4]
    # Extract right 3x3: rows 1-3, cols 5-7
    right = g[1:4, 5:8]
    
    # Find extras in left: positions where 5 and not in center column (col 1)
    extras = []
    for r in range(3):
        for c in range(3):
            if left[r, c] == 5 and c != 1:
                extras.append((r, c))
    
    # Sort by row, then col
    extras.sort()
    
    # Compute deltas
    p1, p2 = extras
    delta_r = p2[0] - p1[0]
    delta_c = p2[1] - p1[1]
    
    # Determine color
    if delta_r == 0:
        color = 1
    elif delta_r * delta_c > 0:
        color = 2
    else:
        color = 3
    
    # Create output: replace 5 in right with color, 0 remains 0
    out = np.zeros((3, 3), dtype=int)
    out[right == 5] = color
    
    return out.tolist()

def solve_514(I):
    if not I or not I[0]:
        return []
    n = len(I)
    out = [[0 for _ in range(2 * n)] for _ in range(2 * n)]

    # Top-left: copy I
    for i in range(n):
        for j in range(n):
            out[i][j] = I[i][j]

    # Bottom-left: 180 rotation
    for i in range(n):
        for j in range(n):
            out[n + i][j] = I[n - 1 - i][n - 1 - j]

    # Compute transpose
    trans = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            trans[i][j] = I[j][i]

    # Top-right: vertical flip of transpose
    for i in range(n):
        for j in range(n):
            out[i][n + j] = trans[n - 1 - i][j]

    # Bottom-right: horizontal flip of transpose
    for i in range(n):
        for j in range(n):
            out[n + i][n + j] = trans[i][n - 1 - j]

    return out

import collections

def solve_515(I):
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])
    bg = I[7][0]
    kept = set()
    seed_rows = {}
    for r in [1, 3, 5]:
        color = I[r][0]
        if color != 0:
            kept.add(color)
            seed_rows[color] = r
    visited = [[False] * cols for _ in range(rows)]
    counts = {}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(7, rows):
        for c in range(cols):
            if I[r][c] != bg and not visited[r][c]:
                color = I[r][c]
                component = []
                queue = collections.deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if 7 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                if color not in kept:
                    for pr, pc in component:
                        output[pr][pc] = bg
                else:
                    counts[color] = counts.get(color, 0) + 1
    for color, srow in seed_rows.items():
        count = counts.get(color, 0)
        if count == 0:
            output[srow][0] = 0
        else:
            for j in range(count):
                output[srow][j] = color
    return output

import numpy as np

def solve_516(I):
    I = np.array(I)
    rows, cols = I.shape
    # Find full yellow rows
    yellow_rows = [r for r in range(rows) if np.all(I[r] == 4)]
    # Find full yellow columns
    yellow_cols = [c for c in range(cols) if np.all(I[:, c] == 4)]
    num_yr = len(yellow_rows)
    num_yc = len(yellow_cols)
    output = I.copy()
    if num_yr > num_yc:
        # Row sections
        dividers = sorted(yellow_rows)
        section_starts = [0] + [d + 1 for d in dividers]
        section_ends = dividers + [rows]
        for start, end in zip(section_starts, section_ends):
            if start >= end:
                continue
            subgrid = I[start:end, :]
            colors = set(subgrid.flatten())
            candidates = colors - {0, 1, 4}
            if len(candidates) == 1:
                C = candidates.pop()
                for r in range(start, end):
                    for c in range(cols):
                        if I[r, c] == 1:
                            output[r, c] = C
    elif num_yc > num_yr:
        # Column sections
        dividers = sorted(yellow_cols)
        section_starts = [0] + [d + 1 for d in dividers]
        section_ends = dividers + [cols]
        for start, end in zip(section_starts, section_ends):
            if start >= end:
                continue
            subgrid = I[:, start:end]
            colors = set(subgrid.flatten())
            candidates = colors - {0, 1, 4}
            if len(candidates) == 1:
                C = candidates.pop()
                for c in range(start, end):
                    for r in range(rows):
                        if I[r, c] == 1:
                            output[r, c] = C
    return output.tolist()

import numpy as np
from collections import Counter

def solve_517(I):
    small = np.array(I)
    n, m = small.shape
    # Find mode for each column
    modes = []
    for c in range(m):
        col = small[:, c]
        count = Counter(col)
        mode = max(count, key=count.get)
        modes.append(mode)
    # Group consecutive columns with same mode
    strips = []
    i = 0
    while i < m:
        j = i
        current_mode = modes[i]
        while j < m and modes[j] == current_mode:
            j += 1
        strips.append((i, j, current_mode))
        i = j
    # Get dominant colors
    doms = [s[2] for s in strips]
    # Check if strictly increasing
    increasing = all(doms[k] < doms[k + 1] for k in range(len(doms) - 1))
    # Determine new order
    if increasing:
        new_order = list(range(len(strips) - 1, -1, -1))
    else:
        new_order = list(range(1, len(strips))) + [0]
    # Build new I
    new_grid = np.zeros((n, m), dtype=int)
    current_col = 0
    for idx in new_order:
        start, end, _ = strips[idx]
        width = end - start
        new_grid[:, current_col : current_col + width] = small[:, start : end]
        current_col += width
    return new_grid.tolist()

def solve_518(I):
    positions = []
    for r in range(len(I)):
        for c in range(len(I[0])):
            if I[r][c] == 4:
                positions.append((r, c))
    if not positions:
        return []
    min_r = min(p[0] for p in positions)
    max_r = max(p[0] for p in positions)
    min_c = min(p[1] for p in positions)
    max_c = max(p[1] for p in positions)
    out_h = 2 * (max_r - min_r + 1)
    out_w = 2 * (max_c - min_c + 1)
    output = [[0] * out_w for _ in range(out_h)]
    for r, c in positions:
        nr = 2 * (r - min_r)
        nc = 2 * (c - min_c)
        output[nr][nc] = 4
        output[nr][nc + 1] = 4
        output[nr + 1][nc] = 4
        output[nr + 1][nc + 1] = 4
    return output

def solve_519(I):
    if not I:
        return I
    rows = len(I)
    output = [row[:] for row in I]
    half = rows // 2
    for r in range(half):
        output[r] = I[rows - 1 - r][:]
    return output

import numpy as np

def solve_520(I):
    height = len(I)
    width = len(I[0])
    out_width = 3  # Fixed based on examples
    
    output = [[0 for _ in range(out_width)] for _ in range(height)]
    
    for i in range(height):
        for j in range(out_width):
            left_val = I[i][j]
            right_val = I[i][j + 4]
            if left_val == 0 and right_val == 0:
                output[i][j] = 3
    
    return output

def solve_521(I):
    if not I or not I[0]:
        return I
    rows = len(I)
    cols = len(I[0])
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for r in range(rows):
        for c in range(cols):
            color = I[r][c]
            if color in counts:
                counts[color] += 1
    output = [[0] * cols for _ in range(rows)]
    for color in range(1, 5):
        height = counts[color]
        if height > 0:
            start_row = rows - height
            col = color - 1
            for r in range(start_row, rows):
                output[r][col] = color
    return output

import numpy as np

def solve_522(I):
    I = np.array(I)
    height, width = I.shape
    shape_size = 3
    max_row = height - shape_size
    max_col = width - shape_size

    # Find the ring position and color
    nonzero = np.argwhere(I != 0)
    colors = np.unique(I[nonzero[:, 0], nonzero[:, 1]])
    if len(colors) != 1:
        raise ValueError("Expected exactly one non-zero color")
    C = colors[0]
    min_r = np.min(nonzero[:, 0])
    min_c = np.min(nonzero[:, 1])
    max_r = np.max(nonzero[:, 0])
    max_c = np.max(nonzero[:, 1])
    if max_r - min_r != shape_size - 1 or max_c - min_c != shape_size - 1:
        raise ValueError("Shape is not 3x3")

    # Direction map
    direction_map = {
        6: 'up',
        4: 'down',
        8: 'right',
        3: 'left'
    }
    if C not in direction_map:
        raise ValueError("Unknown color")
    dir = direction_map[C]

    # Compute new position
    if dir == 'up':
        new_min_r = 0
        new_min_c = min_c
    elif dir == 'down':
        new_min_r = max_row
        new_min_c = min_c
    elif dir == 'right':
        new_min_r = min_r
        new_min_c = max_col
    elif dir == 'left':
        new_min_r = min_r
        new_min_c = 0

    # Create output by copying I and moving the shape
    output = I.copy()
    # Clear old position (border only)
    for dr in range(shape_size):
        for dc in range(shape_size):
            if not (dr == 1 and dc == 1):
                output[min_r + dr, min_c + dc] = 0
    # Set new position (border only, center remains 0)
    for dr in range(shape_size):
        for dc in range(shape_size):
            if not (dr == 1 and dc == 1):
                output[new_min_r + dr, new_min_c + dc] = C

    return output.tolist()

import numpy as np

def solve_523(I):
    if not I or not I[0]:
        return I
    
    input_grid = np.array(I)
    rows, cols = input_grid.shape
    
    all_zero_cols = [c for c in range(cols) if np.all(input_grid[:, c] == 0)]
    
    base = np.copy(input_grid)
    for r in range(rows):
        for c in range(cols):
            if base[r, c] == 0:
                if c in all_zero_cols:
                    continue
                else:
                    base[r, c] = 8
    
    output = np.tile(base, (2, 2))
    return output.tolist()

def solve_524(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    # Find green (3)
    green = None
    for i in range(rows):
        for j in range(cols):
            if I[i][j] == 3:
                green = (i, j)
                break
        if green:
            break
    
    if not green:
        return I  # No green, no change (though examples have one)
    
    gr, gc = green
    
    # Collect purple positions
    purples = [(i, j) for i in range(rows) for j in range(cols) if I[i][j] == 8]
    
    # Collect red positions
    reds = [(i, j) for i in range(rows) for j in range(cols) if I[i][j] == 2]
    
    # Create output I
    output = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # For each red, compute vector, shift all purples
    for rr, rc in reds:
        vr = rr - gr
        vc = rc - gc
        for pr, pc in purples:
            nr = pr + vr
            nc = pc + vc
            if 0 <= nr < rows and 0 <= nc < cols:
                output[nr][nc] = 8
    
    return output

def solve_525(I):
    if not I or not I[0]:
        return I
    
    # Find the foreground color C
    C = None
    for row in I:
        for cell in row:
            if cell != 0 and cell != 5:
                C = cell
                break
        if C is not None:
            break
    
    if C is None:
        return I  # No transformation if no C found
    
    # Create output I
    output = []
    for row in I:
        new_row = []
        for cell in row:
            if cell == 5:
                new_row.append(C)
            else:
                new_row.append(0)
        output.append(new_row)
    
    return output

def solve_526(I):
    if not I or not I[0]:
        return I
    height = len(I)
    width = len(I[0])
    # Find purple positions
    purple_pos = [(r, c) for r in range(height) for c in range(width) if I[r][c] == 8]
    if not purple_pos:
        return [row[:] for row in I]
    rows_with_8 = set(r for r, c in purple_pos)
    cols_with_8 = set(c for r, c in purple_pos)
    is_horizontal = len(rows_with_8) == 1
    is_vertical = len(cols_with_8) == 1
    if not (is_horizontal or is_vertical):
        return [row[:] for row in I]  # Not handling non-straight
    # Find C
    colors = set()
    for r in range(height):
        for c in range(width):
            if I[r][c] != 0 and I[r][c] != 8:
                colors.add(I[r][c])
    if len(colors) != 1:
        return [row[:] for row in I]  # Not handling multiple colors
    C = colors.pop()
    # Create output
    output = [row[:] for row in I]
    # Remove all C
    for r in range(height):
        for c in range(width):
            if output[r][c] == C:
                output[r][c] = 0
    if is_horizontal:
        purple_row = next(iter(rows_with_8))
        for j in range(width):
            if I[purple_row][j] == 8:
                # Above
                has_above = any(I[r][j] == C for r in range(purple_row))
                if has_above and purple_row - 1 >= 0:
                    output[purple_row - 1][j] = C
                # Below
                has_below = any(I[r][j] == C for r in range(purple_row + 1, height))
                if has_below and purple_row + 1 < height:
                    output[purple_row + 1][j] = C
    elif is_vertical:
        purple_col = next(iter(cols_with_8))
        for i in range(height):
            if I[i][purple_col] == 8:
                # Left
                has_left = any(I[i][c] == C for c in range(purple_col))
                if has_left and purple_col - 1 >= 0:
                    output[i][purple_col - 1] = C
                # Right
                has_right = any(I[i][c] == C for c in range(purple_col + 1, width))
                if has_right and purple_col + 1 < width:
                    output[i][purple_col + 1] = C
    return output

import numpy as np

def solve_527(I):
    g = np.array(I)
    rows, cols = g.shape

    # Find vertical mirrors: columns with max consec 2 >=3
    vert_mirrors = []
    for c in range(cols):
        max_consec = 0
        current = 0
        for r in range(rows):
            if g[r, c] == 2:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0
        if max_consec >= 3:
            vert_mirrors.append(c)

    # Find horizontal mirrors: rows with max consec 2 >=3
    horiz_mirrors = []
    for r in range(rows):
        max_consec = 0
        current = 0
        for c in range(cols):
            if g[r, c] == 2:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0
        if max_consec >= 3:
            horiz_mirrors.append(r)

    # Collect greys
    greys = [(r, c) for r in range(rows) for c in range(cols) if g[r, c] == 5]

    # Output copy
    output = g.copy()

    for r, c in greys:
        min_dist = float('inf')
        nearest_m = None
        is_vert = None
        # Check vertical
        for m in vert_mirrors:
            dist = abs(c - m)
            if dist < min_dist:
                min_dist = dist
                nearest_m = m
                is_vert = True
        # Check horizontal
        for m in horiz_mirrors:
            dist = abs(r - m)
            if dist < min_dist:
                min_dist = dist
                nearest_m = m
                is_vert = False
        if nearest_m is not None and min_dist > 0:
            if is_vert:
                new_c = 2 * nearest_m - c
                new_r = r
            else:
                new_r = 2 * nearest_m - r
                new_c = c
            if 0 <= new_r < rows and 0 <= new_c < cols:
                output[new_r, new_c] = 5
        # Remove original
        output[r, c] = 0

    return output.tolist()

import numpy as np
from collections import Counter

def solve_528(I):
    if not I:
        return []
    I = np.array(I)
    unique, counts = np.unique(I, return_counts=True)
    cell_count = dict(zip(unique, counts))
    if 0 in cell_count:
        del cell_count[0]
    bar_count = {c: cnt // 2 for c, cnt in cell_count.items()}
    if not bar_count:
        return []
    dominant = max(bar_count, key=bar_count.get)
    minorities = {c: cnt for c, cnt in bar_count.items() if c != dominant}
    sorted_min = sorted(minorities, key=minorities.get, reverse=True)
    return [[c] for c in sorted_min]

from collections import Counter

def solve_529(I):
    flat = [cell for row in I for cell in row if cell != 0]
    counts = Counter(flat)
    colors = sorted(counts, key=lambda c: counts[c], reverse=True)
    return [[c] for c in colors]

def solve_530(I):
    n = len(I)
    if n == 0:
        return []
    # Find color c and positions
    positions = []
    c = None
    for i in range(n):
        for j in range(n):
            if I[i][j] != 0:
                if c is None:
                    c = I[i][j]
                positions.append((i, j))
    # Group by column
    from collections import defaultdict
    col_rows = defaultdict(list)
    for r, col in positions:
        col_rows[col].append(r)
    # Get motifs
    motifs = []
    for col, rows in col_rows.items():
        rows.sort()
        if len(rows) != 2 or rows[1] != rows[0] + 2:
            continue  # Assume valid
        start_r = rows[0]
        motifs.append((start_r, col))
    # Sort by start_r
    motifs.sort()
    k = len(motifs)
    if k == 0:
        return [[0] * (2 * n) for _ in range(2 * n)]
    d = n // k
    # Base
    s0, c0 = motifs[0]
    # Output
    out_n = 2 * n
    output = [[0] * out_n for _ in range(out_n)]
    num_motifs_out = 2 * k
    for j in range(num_motifs_out):
        sj = s0 + j * d
        cj = c0 + j * d
        if sj + 2 < out_n and cj < out_n:
            output[sj][cj] = c
            output[sj + 2][cj] = c
            for col in range(out_n):
                output[sj + 1][col] = 3
    return output

def solve_531(I):
    if not I or not I[0]:
        return []
    rows = len(I)
    cols = len(I[0])
    # Find positions of 5
    five_pos = [(i, j) for i in range(rows) for j in range(cols) if I[i][j] == 5]
    if not five_pos:
        return [row[:] for row in I]  # Assume always have 5's, but return copy if not
    row_set = set(p[0] for p in five_pos)
    col_set = set(p[1] for p in five_pos)
    halve_height = len(col_set) == 1
    halve_width = len(row_set) == 1
    # Create copy and replace 5 with 8
    new_grid = [row[:] for row in I]
    for i, j in five_pos:
        new_grid[i][j] = 8
    # Take subgrid
    if halve_height:
        out_rows = rows // 2
        return [new_grid[i] for i in range(out_rows)]
    elif halve_width:
        out_cols = cols // 2
        return [[new_grid[i][j] for j in range(out_cols)] for i in range(rows)]
    else:
        return new_grid  # Fallback, assume always one or the other

def solve_532(I):
    output = [row[:] for row in I]
    rows = len(I)
    cols = len(I[0])

    # Find positions where != 8
    positions = [(r, c) for r in range(rows) for c in range(cols) if I[r][c] != 8]
    if not positions:
        return output

    min_r = min(r for r, c in positions)
    max_r = max(r for r, c in positions)
    height = max_r - min_r + 1
    half = height // 2
    upper_min = min_r
    upper_max = min_r + half - 1
    lower_min = min_r + half
    lower_max = max_r

    # Count non-8 in upper_max row for shift
    count_row = upper_max
    shift = sum(1 for c in range(cols) if I[count_row][c] != 8)

    # Find unique colors
    colors = set(I[r][c] for r, c in positions)
    c1, c2 = list(colors)
    swap = {c1: c2, c2: c1}

    # Apply swap to lower half
    for r in range(lower_min, lower_max + 1):
        for c in range(cols):
            if I[r][c] != 8:
                output[r][c] = swap[I[r][c]]

    # Clear original upper half positions
    for r in range(upper_min, upper_max + 1):
        for c in range(cols):
            if I[r][c] != 8:
                output[r][c] = 8

    # Move swapped upper half right by shift
    for r in range(upper_min, upper_max + 1):
        for c in range(cols):
            if I[r][c] != 8:
                new_c = c + shift
                if 0 <= new_c < cols:
                    output[r][new_c] = swap[I[r][c]]

    return output

def solve_533(I):
    if not I or not I[0]:
        return I
    
    h = len(I)
    w = len(I[0])
    
    # Find the color C of the single non-zero cell
    C = None
    for r in range(h):
        for c in range(w):
            if I[r][c] != 0:
                if C is not None:
                    raise ValueError("Multiple non-zero cells found")
                C = I[r][c]
    
    if C is None:
        raise ValueError("No non-zero cell found")
    
    # Create new I of zeros
    new_grid = [[0 for _ in range(w)] for _ in range(h)]
    
    # Set top and bottom rows to C
    for c in range(w):
        new_grid[0][c] = C
        new_grid[h-1][c] = C
    
    # Set left and right columns to C (excluding top and bottom already set)
    for r in range(1, h-1):
        new_grid[r][0] = C
        new_grid[r][w-1] = C
    
    return new_grid

from collections import deque

def solve_534(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    background = 0
    visited = [[False] * cols for _ in range(rows)]
    grid_copy = [row[:] for row in I]
    
    def bfs(start_r, start_c):
        comp = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        comp.append((start_r, start_c))
        while queue:
            cr, cc = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and I[nr][nc] != background:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
                    comp.append((nr, nc))
        return comp
    
    for i in range(rows):
        for j in range(cols):
            if I[i][j] != background and not visited[i][j]:
                comp = bfs(i, j)
                if not comp:
                    continue
                rs = [r for r, c in comp]
                cs = [c for r, c in comp]
                min_r, max_r = min(rs), max(rs)
                min_c, max_c = min(cs), max(cs)
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                if h == w and len(comp) == h * w:
                    colors = set(I[r][c] for r, c in comp)
                    k = len(colors)
                    for offset in range(1, k + 1):
                        new_row = max_r + offset
                        if new_row >= rows:
                            break
                        for new_col in range(min_c, max_c + 1):
                            grid_copy[new_row][new_col] = 3
    
    return grid_copy

import copy

def solve_535(I):
    output = copy.deepcopy(I)
    height = len(I)
    width = len(I[0])
    
    # Top-left corner
    if I[0][0] != 7:
        c = I[0][0]
        output[0][0] = 7
        output[1][1] = c
        output[1][2] = c
        output[2][1] = c
        output[2][2] = c
    
    # Top-right corner
    if I[0][width-1] != 7:
        c = I[0][width-1]
        output[0][width-1] = 7
        output[1][5] = c
        output[1][6] = c
        output[2][5] = c
        output[2][6] = c
    
    # Bottom-left corner
    if I[height-1][0] != 7:
        c = I[height-1][0]
        output[height-1][0] = 7
        output[4][2] = c
        output[5][2] = c
        output[6][3] = c
    
    # Bottom-right corner
    if I[height-1][width-1] != 7:
        c = I[height-1][width-1]
        output[height-1][width-1] = 7
        output[4][5] = c
        output[5][5] = c
        output[6][4] = c
    
    return output

def solve_536(I):
    if not I or not I[0]:
        return []
    
    input_row = I[0]
    n = len(input_row)
    k = sum(1 for x in input_row if x != 0)
    if k == 0:
        return [[0]]
    s = n * k
    
    current = input_row + [0] * (s - n)
    output = [None] * s
    output[s-1] = current[:]
    
    for r in range(s-2, -1, -1):
        current = [0] + current[:-1]
        output[r] = current[:]
    
    return output

def solve_537(I):
    if not I or not I[0]:
        return I
    
    rows = len(I)
    cols = len(I[0])
    
    output = [row[:] for row in I]
    
    yellows = [(r, c) for r in range(rows) for c in range(cols) if I[r][c] == 4]
    
    for yr, yc in yellows:
        r_border = min(yr, rows - 1 - yr, yc, cols - 1 - yc)
        
        max_r = 0
        for candidate_r in range(0, r_border + 1):
            ok = True
            for i in range(-candidate_r, candidate_r + 1):
                for j in range(-candidate_r, candidate_r + 1):
                    pr = yr + i
                    pc = yc + j
                    if I[pr][pc] == 5:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
            max_r = candidate_r
        
        # Fill the square
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                pr = yr + i
                pc = yc + j
                if 0 <= pr < rows and 0 <= pc < cols and I[pr][pc] == 0:
                    output[pr][pc] = 2
    
    return output

