#!/usr/bin/env python3
# %%
"""
Convert a txt-file containing a Python program with '\n' sequences
into a proper .py file with real newlines and correct indentation.

Example input content (inside input.txt):

"def solve(I):\n    # Get grid dimensions using DSL\n    h = height(I)\n    w = width(I)\n    if h == 0 or w == 0:\n        return I\n    \n    # Create mutable grid from input\n    grid = [list(row) for row in I]\n    ...\n    return O"

Usage:
    python txt_to_py.py input.txt output.py
"""
from pathlib import Path
import re

# === Define your input string here ===
input_txt = (
    "def solve(I):\\n"
    "    # Get grid dimensions using DSL\\n"
    "    h = get_height(I)\\n"
    "    w = get_width(I)\\n"
    "    if h == 0 or w == 0:\\n"
    "        return I\\n"
    "    \\n"
    "    # Create mutable grid from input\\n"
    "    grid = [list(row) for row in I]\\n"
    "    \\n"
    "    # Simulate top-down pouring: loop over rows\\n"
    "    for r in range(h - 1):\\n"
    "        # Get initial sources: columns with 6 in current row\\n"
    "        sources = [j for j in range(w) if grid[r][j] == 6]\\n"
    "        \\n"
    "        # Get hit columns: where below is 2\\n"
    "        hit_cols = [j for j in sources if grid[r + 1][j] == 2]\\n"
    "        \\n"
    "        # For each hit, compute horizontal red component in row r+1 and spread in row r\\n"
    "        for hit in hit_cols:\\n"
    "            left = hit\\n"
    "            while left > 0 and grid[r + 1][left - 1] == 2:\\n"
    "                left -= 1\\n"
    "            right = hit\\n"
    "            while right < w - 1 and grid[r + 1][right + 1] == 2:\\n"
    "                right += 1\\n"
    "            spread_l = max(0, left - 1)\\n"
    "            spread_r = min(w - 1, right + 1)\\n"
    "            for j in range(spread_l, spread_r + 1):\\n"
    "                if grid[r][j] == 7:\\n"
    "                    grid[r][j] = 6\\n"
    "        \\n"
    "        # After all spreads, fill downward from all current 6s in row r where below is 7\\n"
    "        for j in range(w):\\n"
    "            if grid[r][j] == 6 and grid[r + 1][j] == 7:\\n"
    "                grid[r + 1][j] = 6\\n"
    "    \\n"
    "    # Convert back to immutable tuple of tuples\\n"
    "    O = tuple(tuple(row) for row in grid)\\n"
    "    return O"
)

# === Define output file name ===
output_file = Path("output.py")


def decode_escapes(s: str) -> str:
    """Turn literal escape sequences like '\\n' into real newlines."""
    # Replace common escaped sequences
    s = s.replace(r"\n", "\n")
    s = s.replace(r"\t", "\t")
    s = s.replace(r"\r", "\r")

    # Handle double-escaped versions (\\n)
    s = s.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")

    return s


def extract_string_payload(raw: str) -> str:
    """If the text is quoted (e.g. JSON-like), extract the string inside quotes."""
    matches = list(re.finditer(r'"(.*)"', raw, flags=re.DOTALL))
    if matches:
        return max(matches, key=lambda m: len(m.group(1))).group(1)
    return raw


def convert_string_to_py(input_txt: str, output_path: Path):
    # Clean up any surrounding quotes and decode newlines
    payload = extract_string_payload(input_txt)
    code = decode_escapes(payload).lstrip("\n").rstrip("\n")
    output_path.write_text(code, encoding="utf-8")
    print(f"✅ Python program written to {output_path.resolve()}")


if __name__ == "__main__":
    input_txt = "def solve(I):\n    # Get grid dimensions and background color\n    h, w = get_shape(I)\n    bg = most_common_color(I)\n    \n    # Find all single-color objects ignoring background\n    all_objs = as_objects(I, T, F, T)\n    \n    # Filter to get large 5x5 blocks (size 25)\n    large_objs_list = []\n    for obj in all_objs:\n        if size(obj) == 25:\n            large_objs_list.append(obj)\n    \n    # For each large block, compute its C, min_r, min_c and store\n    block_infos = []\n    for obj in large_objs_list:\n        C = get_color(obj)\n        inds = toindices(obj)\n        rs = [pos[0] for pos in inds]\n        cs = [pos[1] for pos in inds]\n        min_r = min(rs)\n        min_c = min(cs)\n        block_infos.append((C, min_r, min_c, obj))  # include obj for later\n    \n    # Create output grid as full canvas of background\n    O = create_grid(bg, (h, w))\n    \n    # Process each block: fill 5x5 with C, then carve based on small C cells\n    for C, min_r, min_c, obj in block_infos:\n        # Fill the 5x5 bbox with C\n        bbox_inds = frozenset((r, c) for r in range(min_r, min_r + 5) for c in range(min_c, min_c + 5))\n        O = fill(O, C, bbox_inds)\n        \n        # Find small C indices (all C minus this large block)\n        all_C_inds = ofcolor(I, C)\n        large_inds = toindices(obj)\n        small_inds = difference(all_C_inds, large_inds)\n        \n        # If no small cells, no carving\n        if len(small_inds) == 0:\n            continue\n        \n        # Compute bounding box for small cells\n        small_rs = [pos[0] for pos in small_inds]\n        small_cs = [pos[1] for pos in small_inds]\n        s_min_r = min(small_rs)\n        s_min_c = min(small_cs)\n        \n        # Relative pattern positions (where C is filled in 3x3)\n        pattern_rel = frozenset((r - s_min_r, c - s_min_c) for r, c in small_inds)\n        \n        # Compute absolute carve positions in block interior\n        carve_inds = frozenset((min_r + 1 + pr, min_c + 1 + pc) for pr, pc in pattern_rel)\n        \n        # Carve by filling with bg\n        O = fill(O, bg, carve_inds)\n    \n    # If no blocks, return the bg canvas\n    if len(block_infos) == 0:\n        return O\n    \n    # Compute overall min/max for blocks' bboxes\n    all_min_r = min(min_r for C, min_r, min_c, obj in block_infos)\n    all_max_r = max(min_r + 4 for C, min_r, min_c, obj in block_infos)\n    all_min_c = min(min_c for C, min_r, min_c, obj in block_infos)\n    all_max_c = max(min_c + 4 for C, min_r, min_c, obj in block_infos)\n    \n    # Compute crop start with 1-cell border (safe >=0)\n    crop_r = max(0, all_min_r - 1)\n    crop_c = max(0, all_min_c - 1)\n    \n    # Compute crop end (safe <= h-1, w-1)\n    end_r = min(h - 1, all_max_r + 1)\n    end_c = min(w - 1, all_max_c + 1)\n    \n    # Compute dimensions\n    crop_h = end_r - crop_r + 1\n    crop_w = end_c - crop_c + 1\n    \n    # Crop to the tight enclosing grid with borders\n    O = crop(O, (crop_r, crop_c), (crop_h, crop_w))\n    \n    return O"
    output_file = Path("output.py")
    convert_string_to_py(input_txt, output_file)

# %%
