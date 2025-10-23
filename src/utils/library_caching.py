"""
Utility to pre-build and inspect the program library from solvers.py

Usage:
    python library_caching.py              # Build library from solvers.py
    python library_caching.py --inspect    # Show what's in the library
    python library_caching.py --rebuild    # Force rebuild even if cache exists
"""

import sys
import os
from library import ProgramLibrary, extract_functions

def load_dsl():
    """Load DSL into namespace"""
    print("Loading DSL...")
    dsl_globals = {}
    try:
        with open('./src/utils/dsl.py', 'r') as f:
            dsl_code = f.read()
        exec(dsl_code, dsl_globals)
        count = len([k for k in dsl_globals.keys() if not k.startswith('_')])
        print(f"✅ Loaded DSL with {count} functions\n")
        return dsl_globals
    except Exception as e:
        print(f"❌ Failed to load DSL: {e}")
        sys.exit(1)


def load_solvers(solvers_path: str, library: ProgramLibrary, dsl_globals: dict):
    """Load all solve_* functions from solvers.py"""
    import re
    
    try:
        with open(solvers_path, 'r') as f:
            solvers_code = f.read()
        
        # Execute to get all solve functions
        namespace = dsl_globals.copy()
        exec(solvers_code, namespace)
        
        # Find all solve_* functions
        count = 0
        for name, obj in namespace.items():
            if name.startswith('solve_') and callable(obj):
                task_id = name.replace('solve_', '')
                
                # Extract just this function's code
                pattern = rf'def {name}\(I\):.*?(?=\ndef |\Z)'
                match = re.search(pattern, solvers_code, re.DOTALL)
                
                if match:
                    func_code = match.group(0).strip()
                    # Normalize to 'def solve(I):'
                    func_code = func_code.replace(f'def {name}(', 'def solve(')
                    
                    # Extract keywords
                    keywords = extract_functions(func_code)
                    
                    # Add to library
                    library.add(task_id, func_code, func_code)
                    
                    count += 1
                    print(f"  ✓ {task_id}: {len(keywords)} keywords - {', '.join(sorted(list(keywords)[:5]))}{'...' if len(keywords) > 5 else ''}")
        
        print(f"\n✅ Loaded {count} solutions from {solvers_path}")
        return count
        
    except FileNotFoundError:
        print(f"❌ Solvers file not found: {solvers_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading solvers: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def build_library(force_rebuild=False):
    """Build library from solvers.py and save to cache"""
    cache_file = 'library_cache.json'
    
    # Check if cache exists
    if os.path.exists(cache_file) and not force_rebuild:
        print(f"⚠️  Library cache already exists: {cache_file}")
        print("   Use --rebuild to force rebuild\n")
        return
    
    print("=" * 80)
    print("BUILDING LIBRARY FROM SOLVERS.PY")
    print("=" * 80)
    
    # Load DSL
    dsl_globals = load_dsl()
    
    # Initialize library
    library = ProgramLibrary()
    
    # Load all solvers
    print("Loading solvers from solvers.py...\n")
    count = load_solvers('./src/utils/solvers.py', library, dsl_globals)
    
    # Save to cache
    print(f"\nSaving library to {cache_file}...")
    library.save(cache_file)
    print(f"✅ Library cache saved!")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Programs in library: {count}")
    print(f"Cache file: {cache_file}")
    print(f"Cache size: {os.path.getsize(cache_file) / 1024:.1f} KB")
    print("\nLibrary ready for use! main.py will load it automatically.")


def inspect_library():
    """Show what's in the library cache"""
    cache_file = 'library_cache.json'
    
    if not os.path.exists(cache_file):
        print(f"❌ No library cache found: {cache_file}")
        print("   Run: python build_library.py")
        return
    
    print("=" * 80)
    print("LIBRARY INSPECTION")
    print("=" * 80)
    
    # Load library
    library = ProgramLibrary()
    library.load(cache_file)
    
    print(f"\n📚 Library contains {len(library)} programs\n")
    
    # Show each program
    for i, prog in enumerate(library.programs[:20], 1):  # Show first 20
        task_id = prog['task_id']
        keywords = prog['keywords']
        code_lines = prog['code'].count('\n') + 1
        
        print(f"{i:2d}. Task: {task_id}")
        print(f"    Keywords ({len(keywords)}): {', '.join(sorted(list(keywords)[:8]))}{'...' if len(keywords) > 8 else ''}")
        print(f"    Code: {code_lines} lines")
        print()
    
    if len(library) > 20:
        print(f"... and {len(library) - 20} more programs")
    
    # Keyword statistics
    print("\n" + "=" * 80)
    print("KEYWORD STATISTICS")
    print("=" * 80)
    
    keyword_counts = {}
    for prog in library.programs:
        for kw in prog['keywords']:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
    
    # Top 20 keywords
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 most used DSL functions:")
    for i, (kw, count) in enumerate(sorted_keywords[:20], 1):
        bar = "█" * min(40, count * 40 // sorted_keywords[0][1])
        print(f"{i:2d}. {kw:20s} {count:3d} {bar}")
    
    print(f"\nTotal unique keywords: {len(keyword_counts)}")


def main():
    """Main entry point"""
    import sys
    
    # Parse arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg in ['--inspect', '-i']:
            inspect_library()
        elif arg in ['--rebuild', '-r']:
            build_library(force_rebuild=True)
        elif arg in ['--help', '-h']:
            print(__doc__)
        else:
            print(f"Unknown argument: {arg}")
            print(__doc__)
    else:
        # Default: build library
        build_library(force_rebuild=False)


if __name__ == "__main__":
    main()