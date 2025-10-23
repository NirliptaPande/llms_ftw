from typing import Tuple, List

Grid = Tuple[Tuple[int, ...], ...]


def render_grid_ascii(grid: Grid, cell_width: int = 1) -> str:
    """
    Render grid as ASCII art for VLM prompts
    
    Args:
        grid: Grid tuple (rows of color values)
        cell_width: Width of each cell in characters
        
    Returns:
        ASCII string representation
    """
    if not grid or not grid[0]:
        return ""
    
    lines = []
    for row in grid:
        row_str = " ".join(str(cell).rjust(cell_width) for cell in row)
        lines.append(row_str)
    
    return "\n".join(lines)


def render_grid_with_border(grid: Grid, title: str = None) -> str:
    """
    Render grid with optional title and border
    
    Args:
        grid: Grid tuple
        title: Optional title to display above grid
        
    Returns:
        Bordered ASCII representation
    """
    if not grid or not grid[0]:
        return ""
    
    ascii_grid = render_grid_ascii(grid)
    lines = ascii_grid.split("\n")
    width = len(lines[0]) if lines else 0
    
    border = "=" * width
    result = []
    
    if title:
        result.append(title)
        result.append(border)
    
    result.extend(lines)
    
    return "\n".join(result)


def render_side_by_side(grid_a: Grid, grid_b: Grid, 
                        label_a: str = "Input", 
                        label_b: str = "Output",
                        spacing: int = 4) -> str:
    """
    Render two grids side by side for comparison
    
    Args:
        grid_a: First grid
        grid_b: Second grid
        label_a: Label for first grid
        label_b: Label for second grid
        spacing: Number of spaces between grids
        
    Returns:
        Side-by-side ASCII representation
    """
    lines_a = render_grid_ascii(grid_a).split("\n")
    lines_b = render_grid_ascii(grid_b).split("\n")
    
    max_lines = max(len(lines_a), len(lines_b))
    width_a = len(lines_a[0]) if lines_a else 0
    
    # Pad shorter grid with empty lines
    while len(lines_a) < max_lines:
        lines_a.append(" " * width_a)
    while len(lines_b) < max_lines:
        lines_b.append("")
    
    spacer = " " * spacing
    result = [f"{label_a}{spacer}{label_b}"]
    result.append("-" * (width_a + spacing + len(lines_b[0])))
    
    for line_a, line_b in zip(lines_a, lines_b):
        result.append(f"{line_a}{spacer}{line_b}")
    
    return "\n".join(result)


def render_object_overlay(grid: Grid, object_cells: List[Tuple[int, int]], 
                          marker: str = "*") -> str:
    """
    Render grid with specific cells marked (for debugging)
    
    Args:
        grid: Grid tuple
        object_cells: List of (row, col) positions to mark
        marker: Character to use for marking
        
    Returns:
        ASCII grid with marked cells
    """
    if not grid or not grid[0]:
        return ""
    
    # Convert to mutable structure
    marked_grid = [list(row) for row in grid]
    
    # Mark cells
    for i, j in object_cells:
        if 0 <= i < len(marked_grid) and 0 <= j < len(marked_grid[0]):
            marked_grid[i][j] = marker
    
    # Convert back and render
    marked_tuple = tuple(tuple(row) for row in marked_grid)
    return render_grid_ascii(marked_tuple)


def format_dimensions(grid: Grid) -> str:
    """Get human-readable dimensions"""
    if not grid:
        return "0x0"
    return f"{len(grid)}x{len(grid[0])}"