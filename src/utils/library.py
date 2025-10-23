"""
Program library for storing and retrieving solved ARC tasks
"""

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
    # All DSL function names (comprehensive list from dsl.py)
    DSL_FUNCTIONS = {
        # Functional programming
        'identity', 'compose', 'chain', 'fork', 'apply', 'mapply',
        'lbind', 'rbind', 'matcher', 'extract',
        
        # Transforms
        'hmirror', 'vmirror', 'dmirror', 'cmirror',
        'rot90', 'rot180', 'rot270',
        
        # Compose
        'vconcat', 'hconcat', 'crop', 'upscale', 'downscale',
        'hsplit', 'vsplit', 'tophalf', 'bottomhalf', 'lefthalf', 'righthalf',
        
        # Objects
        'objects', 'colorfilter', 'sizefilter', 'ofcolor',
        'toobject', 'normalize', 'toindices', 'asindices',
        
        # Modify
        'fill', 'paint', 'replace', 'switch', 'shift', 'move', 'cover',
        
        # Query
        'size', 'height', 'width', 'shape', 'palette', 'mostcolor', 'leastcolor',
        'ulcorner', 'urcorner', 'llcorner', 'lrcorner', 'center', 'corners',
        'color', 'index', 'occurrences',
        
        # Spatial
        'box', 'inbox', 'outbox', 'backdrop', 'delta',
        'vfrontier', 'hfrontier', 'shoot', 'connect', 'position',
        'gravitate', 'compress', 'frontiers',
        
        # Set operations
        'combine', 'intersection', 'difference', 'merge', 'dedupe',
        'sfilter', 'mfilter', 'contained', 'initset',
        
        # Aggregation
        'argmax', 'argmin', 'valmax', 'valmin', 'maximum', 'minimum',
        'mostcommon', 'leastcommon', 'order', 'repeat',
        
        # Arithmetic
        'add', 'subtract', 'multiply', 'divide', 'invert',
        'double', 'halve', 'increment', 'decrement', 'sign',
        'toivec', 'tojvec',
        
        # Boolean
        'equality', 'both', 'either', 'flip', 'positive', 'even', 'greater',
        
        # Create
        'canvas', 'astuple', 'trim',
        
        # Advanced
        'cellwise', 'hperiod', 'vperiod',
    }
    
    # Find which functions appear in the text
    found = set()
    for func in DSL_FUNCTIONS:
        # Use word boundary to avoid partial matches
        if re.search(rf'\b{func}\b', pattern_text):
            found.add(func)
    
    return found


class ProgramLibrary:
    """Storage and retrieval of solved ARC programs"""
    
    def __init__(self):
        self.programs = []
    
    def save(self, filepath: str):
        """
        Save library to disk (JSON format).
        
        Args:
            filepath: Path to save library (e.g., 'library.json')
        """
        import json
        
        # Convert programs to JSON-serializable format
        data = []
        for prog in self.programs:
            data.append({
                'task_id': prog['task_id'],
                'pattern': prog['pattern'],
                'code': prog['code'],
                'keywords': list(prog['keywords'])  # Convert set to list
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """
        Load library from disk.
        
        Args:
            filepath: Path to library file
        """
        import json
        import os
        
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for prog_data in data:
            self.programs.append({
                'task_id': prog_data['task_id'],
                'pattern': prog_data['pattern'],
                'code': prog_data['code'],
                'keywords': set(prog_data['keywords'])  # Convert list back to set
            })
    
    def add(self, task_id: str, pattern: str, code: str):
        """
        Store successful solution.
        
        Args:
            task_id: Unique identifier for the task
            pattern: Phase 1 pattern analysis output
            code: Python code with solve(I) function
        """
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
    
    def get(self, task_id: str) -> Dict:
        """Get program by task ID"""
        for prog in self.programs:
            if prog['task_id'] == task_id:
                return prog
        return None
    
    def __len__(self):
        """Number of programs in library"""
        return len(self.programs)


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
{prog['code']}
```""")
    
    return "\n".join(output)


# Example usage
if __name__ == "__main__":
    library = ProgramLibrary()
    
    # Add some programs
    library.add(
        'test_1',
        'x1 = hmirror(I); O = vconcat(I, x1)',
        'def solve(I):\n    x1 = hmirror(I)\n    return vconcat(I, x1)'
    )
    
    library.add(
        'test_2',
        'x1 = objects(I, T, F, T); x2 = argmax(x1, size); O = fill(I, ZERO, x2)',
        'def solve(I):\n    x1 = objects(I, T, F, T)\n    x2 = argmax(x1, size)\n    return fill(I, ZERO, x2)'
    )
    
    # Search
    results = library.find_similar({'hmirror', 'vconcat'}, top_k=2)
    
    print(f"Found {len(results)} similar programs:")
    for r in results:
        print(f"  Task: {r['program']['task_id']}")
        print(f"  Similarity: {r['similarity']:.2f}")
        print(f"  Shared: {r['shared_functions']}")