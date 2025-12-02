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
        'identity', 'add', 'subtract', 'multiply', 'divide', 'double', 'halve', 'is_even', 'negate', 'is_positive',
        'logical_not', 'logical_and', 'logical_or', 'is_equal', 'greater_than', 'contains', 'union', 'intersection',
        'difference', 'remove_duplicates', 'sort', 'repeat', 'size', 'maximum', 'minimum', 'valmax', 'valmin',
        'argmax', 'argmin', 'most_common', 'least_common', 'increment', 'decrement', 'crement', 'sign',
        'to_vertical_vec', 'to_horizontal_vec', 'initset', 'insert', 'remove', 'get_first', 'get_other', 'get_last',
        'flatten', 'keep_if_condition', 'keep_if_condition_and_flatten', 'extract_first_matching', 'interval',
        'to_tuple', 'as_generic_tuple', 'as_tuple', 'make_cell', 'cartesian_product', 'pairwise', 'condition_if_else',
        'compose', 'chain', 'equals', 'fix_last_argument', 'fix_first_argument', 'power', 'combine_two_function_results',
        'transform', 'apply_each_function', 'transform_and_flatten', 'transform_both', 'transform_both_and_flatten',
        'apply_function_on_cartesian_product', 'most_common_color', 'least_common_color', 'get_height', 'get_width',
        'get_shape', 'is_portrait', 'color_count', 'color_filter', 'as_indices', 'size_filter', 'of_color',
        'upper_left_corner', 'upper_right_corner', 'lower_left_corner', 'lower_right_corner', 'crop', 'to_indices',
        'recolor', 'shift_by_vector', 'shift_to_origin', 'direct_neighbors', 'diagonal_neighbors', 'neighbors',
        'as_objects', 'partition', 'partition_only_foreground', 'uppermost', 'lowermost', 'leftmost', 'rightmost',
        'is_square', 'is_vertical_line', 'is_horizontal_line', 'horizontal_matching', 'vertical_matching',
        'manhattan_distance', 'adjacent', 'bordering', 'centerofmass', 'palette', 'count_colors', 'get_color',
        'to_object', 'as_object', 'rot90', 'rot180', 'rot270', 'horizontal_mirror', 'vertical_mirror',
        'diagonal_mirror', 'counterdiagonal_mirror', 'fill', 'fill_background', 'paint_onto_grid',
        'paint_onto_grid_background', 'horizontal_upscale', 'vertical_upscale', 'upscale', 'downscale',
        'horizontal_concat', 'vertical_concat', 'smallest_subgrid_containing', 'horizontal_split', 'vertical_split',
        'cellwise', 'replace', 'switch', 'center', 'position', 'color_at_location', 'create_grid', 'corner_indices',
        'line_between', 'erase_patch', 'trim_border', 'move_object', 'top_half', 'bottom_half', 'left_half',
        'right_half', 'vertical_line', 'horizontal_line', 'bounding_box_indices', 'bounding_box_delta',
        'move_until_touching', 'inbox', 'outbox', 'box', 'shoot', 'occurrences', 'solid_color_strips_in_grid',
        'remove_solid_color_strips_from_grid', 'horizontal_periodicity', 'vertical_periodicity'
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