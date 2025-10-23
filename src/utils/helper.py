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
        # Transforms
        'hmirror', 'vmirror', 'dmirror', 'cmirror',
        'rot90', 'rot180', 'rot270',
        
        # Compose
        'vconcat', 'hconcat', 'crop', 'upscale', 'downscale',
        
        # Objects
        'objects', 'colorfilter', 'sizefilter', 'argmax', 'argmin',
        
        # Modify
        'fill', 'paint', 'replace', 'shift',
        
        # Query
        'size', 'height', 'width', 'shape', 'palette', 'mostcolor',
        'ulcorner', 'urcorner', 'toindices',
        
        # Create
        'canvas', 'full',
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