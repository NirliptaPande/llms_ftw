"""
Utility to load ARC tasks from JSON files
"""

import json
from typing import Dict, List, Tuple


def load_task_from_json(filepath: str) -> Dict:
    """
    Load an ARC task from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Dict with 'train' and 'test' keys
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert lists to tuples (required by DSL)
    task = {}
    
    if 'train' in data:
        task['train'] = [
            {
                'input': list_to_tuple_grid(example['input']),
                'output': list_to_tuple_grid(example['output'])
            }
            for example in data['train']
        ]
    
    if 'test' in data:
        task['test'] = [
            {
                'input': list_to_tuple_grid(example['input']),
                'output': list_to_tuple_grid(example['output']) if 'output' in example else None
            }
            for example in data['test']
        ]
    
    return task


def list_to_tuple_grid(grid: List[List[int]]) -> Tuple[Tuple[int]]:
    """Convert nested list to nested tuple"""
    return tuple(tuple(row) for row in grid)


def tuple_to_list_grid(grid: Tuple[Tuple[int]]) -> List[List[int]]:
    """Convert nested tuple to nested list"""
    return [list(row) for row in grid]


def save_task_to_json(task: Dict, filepath: str):
    """
    Save a task to JSON file.
    
    Args:
        task: Task dict with 'train' and 'test' keys
        filepath: Output path
    """
    # Convert tuples back to lists
    data = {}
    
    if 'train' in task:
        data['train'] = [
            {
                'input': tuple_to_list_grid(example['input']),
                'output': tuple_to_list_grid(example['output'])
            }
            for example in task['train']
        ]
    
    if 'test' in task:
        data['test'] = [
            {
                'input': tuple_to_list_grid(example['input']),
                'output': tuple_to_list_grid(example['output']) if example['output'] else []
            }
            for example in task['test']
        ]
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_tasks_from_directory(directory: str) -> Dict[str, Dict]:
    """
    Load all tasks from a directory of JSON files.
    
    Args:
        directory: Path to directory with JSON task files
    
    Returns:
        Dict mapping task_id -> task data
    """
    import os
    
    tasks = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            task_id = filename.replace('.json', '')
            filepath = os.path.join(directory, filename)
            
            try:
                task = load_task_from_json(filepath)
                tasks[task_id] = task
            except Exception as e:
                print(f"⚠️  Failed to load {filename}: {e}")
    
    return tasks


# Example usage
if __name__ == "__main__":
    # Create example task
    example_task = {
        'train': [
            {
                'input': ((0, 1), (2, 3)),
                'output': ((3, 2), (1, 0))
            }
        ],
        'test': [
            {
                'input': ((4, 5), (6, 7)),
                'output': None
            }
        ]
    }
    
    # Save to JSON
    save_task_to_json(example_task, 'example_task.json')
    print("✅ Saved example_task.json")
    
    # Load it back
    loaded = load_task_from_json('example_task.json')
    print("✅ Loaded task:")
    print(f"   Train examples: {len(loaded['train'])}")
    print(f"   Test examples: {len(loaded['test'])}")