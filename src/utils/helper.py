import re
from typing import List, Dict, Set

def extract_functions(pattern_text: str) -> Set[str]:
    """
    Extract DSL function names from Phase 1 pattern description.
    
    Args:
        pattern_text: Output from Phase 1 (contains OPS section)
    
    Returns:
        Set of DSL function names found
    """
    # All DSL function names
    DSL_FUNCTIONS = {
    'add', 'adjacent', 'apply_each_function', 'apply_function_on_cartesian_product',
    'argmax', 'argmin', 'as_generic_tuple', 'as_indices', 'as_object', 'as_objects',
    'as_tuple', 'bordering', 'bottom_half', 'bounding_box_delta', 'bounding_box_indices',
    'box', 'cartesian_product', 'cellwise', 'center', 'centerofmass', 'chain',
    'color_at_location', 'color_count', 'color_filter', 'combine_two_function_results',
    'compose', 'condition_if_else', 'contains', 'corner_indices', 'counterdiagonal_mirror',
    'count_colors', 'create_grid', 'crement', 'crop', 'decrement', 'diagonal_mirror',
    'diagonal_neighbors', 'difference', 'direct_neighbors', 'divide', 'double',
    'downscale', 'equals', 'erase_patch', 'extract_first_matching', 'fill',
    'fill_background', 'fix_first_argument', 'fix_last_argument', 'flatten',
    'get_color', 'get_first', 'get_height', 'get_last', 'get_other', 'get_shape',
    'get_width', 'greater_than', 'halve', 'horizontal_concat', 'horizontal_line',
    'horizontal_matching', 'horizontal_mirror', 'horizontal_periodicity',
    'horizontal_split', 'horizontal_upscale', 'identity', 'inbox', 'increment',
    'initset', 'insert', 'intersection', 'interval', 'is_equal', 'is_even',
    'is_horizontal_line', 'is_portrait', 'is_positive', 'is_square', 'is_vertical_line',
    'keep_if_condition', 'keep_if_condition_and_flatten', 'least_common',
    'least_common_color', 'left_half', 'leftmost', 'line_between', 'logical_and',
    'logical_not', 'logical_or', 'lower_left_corner', 'lower_right_corner',
    'lowermost', 'make_cell', 'manhattan_distance', 'maximum', 'minimum',
    'most_common', 'most_common_color', 'move_object', 'move_until_touching',
    'multiply', 'negate', 'neighbors', 'occurrences', 'of_color', 'outbox',
    'paint_onto_grid', 'paint_onto_grid_background', 'pairwise', 'palette',
    'partition', 'partition_only_foreground', 'position', 'power', 'recolor',
    'remove', 'remove_duplicates', 'remove_solid_color_strips_from_grid', 'repeat',
    'replace', 'right_half', 'rightmost', 'rot180', 'rot270', 'rot90',
    'shift_by_vector', 'shift_to_origin', 'shoot', 'sign', 'size', 'size_filter',
    'smallest_subgrid_containing', 'solid_color_strips_in_grid', 'sort', 'subtract',
    'switch', 'to_horizontal_vec', 'to_indices', 'to_object', 'to_tuple',
    'to_vertical_vec', 'top_half', 'transform', 'transform_and_flatten',
    'transform_both', 'transform_both_and_flatten', 'trim_border', 'union',
    'upper_left_corner', 'upper_right_corner', 'uppermost', 'upscale', 'valmax',
    'valmin', 'vertical_concat', 'vertical_line', 'vertical_matching',
    'vertical_mirror', 'vertical_periodicity', 'vertical_split', 'vertical_upscale'
    }
    
    
    # Find which functions appear in the text
    found = set()
    for func in DSL_FUNCTIONS:
        # Use word boundary to avoid partial matches
        if re.search(rf'\b{func}\b', pattern_text):
            found.add(func)
    
    return found


class ProgramLibrary:
    def __init__(self):
        self.programs = []
    
    def add(self, task_id: str, pattern: str, code: str):
        """Store successful solution"""
        keywords = extract_functions(pattern)
        
        self.programs.append({
            'task_id': task_id,
            'pattern': pattern,
            'code': code,
            'keywords': keywords
        })
    
    def find_similar(self, query_keywords: Set[str], top_k: int = 5) -> List[Dict]:
        """
        Find programs with similar keyword overlap.
        
        Args:
            query_keywords: Set of DSL function names
            top_k: Number of results to return
        
        Returns:
            List of dicts with keys: program, similarity, shared_functions
        """
        if not query_keywords:
            return []
        
        results = []
        
        for prog in self.programs:
            prog_keywords = prog['keywords']
            
            # Calculate Jaccard similarity
            intersection = query_keywords & prog_keywords
            union = query_keywords | prog_keywords
            
            if len(union) == 0:
                continue
            
            similarity = len(intersection) / len(union)
            
            if similarity > 0:
                results.append({
                    'program': prog,
                    'similarity': similarity,
                    'shared_functions': intersection
                })
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    

def format_similar_programs(similar_programs: List[Dict]) -> str:
    """
    Format library matches for insertion into Phase 2 prompt.
    
    Args:
        similar_programs: Output from library.find_similar()
    
    Returns:
        Formatted string to insert into prompt
    """
    if not similar_programs:
        return ""
    
    output = []
    
    for i, entry in enumerate(similar_programs, 1):
        prog = entry['program']
        similarity = entry['similarity']
        shared = entry['shared_functions']
        
        output.append(f"""```python
        # Similar program {i} (similarity: {similarity:.2f})
        # Shared functions: {{{', '.join(sorted(shared))}}}
        {prog['code']}```""")
    
    return "\n".join(output)

def format_examples(train_examples: List) -> str:
    """
    Format training examples for Phase 1 prompt.
    
    Args:
        train_examples: List of (input, output) grid pairs
    
    Returns:
        Formatted string
    """
    output = []
    
    for i, (inp, out) in enumerate(train_examples, 1):
        output.append(f"""Example {i}:
        Input: {inp}
        Output: {out}
        """)
    
    return "\n".join(output)