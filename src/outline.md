### Overview

I have these two repositiories, one where you generate a natural language description and then given the natural language description, you generate the output programs, 
in the second one, you just generate programs and add them to the library etc etc
I want to combine? them? Idk man, 
Have a library of functions, ehh, I am not getting into that mess for now, maybe, just use what EPang has used, I want to do what Eric did, and combine the idea of objectness and actions? basically start w/ a DSL

I think, I would like to start off w/ the library of actions and whatever the hell, give the LLM and option to solve the program by starting it one at a time, and conditioning the responses on the previous 'trace' because there is information flow b/w the 2-10 examples, and then solve the response. Ask it for this series of transformations and that's it tbh
cool, lets do this then

In the prompting session, should I give it the code for objects first then ask it to extract actions, or how do I do this? I think I can give it the code for objects and ask it to extract objects, and also give it a basic understanding of wha tthe transformations are like and ask it for a series of trasformations in this this space or ask it for transformations in natural language and then transform it accordingly.

## TODO

 So I think I will start simple, but yeah, I will start w/ the first instance of a task, ask for an output transformation, then conditioned on this, ask for a tranformation for the next two, and so on, and in the last one, ask for the program for all of them , does that make sense?

 Okay, outlined the process,
 The process is:

    Phase 1: Sequential example analysis (DSL-like notation)
    Library Search: Find similar programs by keywords (free)
    Phase 2a: Generate initial program from Phase 1 + library examples
    Test initial program
    Phase 2b: If not perfect, evolve with mutations
    Early stopping strategies throughout

```python
def solve_task(task, library, max_iter=10):
    # Phase 1: Sequential analysis
    pattern = call_llm(PHASE1_PROMPT, task, "sonnet-3.5", temp=0)
    # ~150 tokens
    
    # Library search (free)
    keywords = extract_functions(pattern)
    similar = library.find_similar(keywords, top_k=5)
    
    # Phase 2a: Initial generation
    program = call_llm(PHASE2_PROMPT, pattern, similar, "haiku", temp=0)
    score = test_program(program, task)
    # ~200 tokens
    
    if score == 1.0:  # Early stop ✓
        library.add(task.id, pattern, program)
        return program, score
    
    # Phase 2b: Evolution
    best_program = program
    best_score = score
    no_improvement = 0
    
    for i in range(max_iter):
        old_score = best_score
        
        # Generate mutations
        mutations = generate_mutations(best_program)
        
        for mutant in mutations:
            try:
                s = test_program(mutant, task)
                if s > best_score:
                    best_program = mutant
                    best_score = s
                    no_improvement = 0
                
                if s == 1.0:  # Early stop ✓
                    library.add(task.id, pattern, best_program)
                    return best_program, s
            except:
                pass
        
        # Early stop checks
        if best_score == old_score:
            no_improvement += 1
        
        if no_improvement >= 3:  # Early stop ✓
            break
        
        if best_score > 0.8 and i > 5:  # Early stop ✓
            break
    
    # ~1250 tokens for evolution
    
    if best_score == 1.0:
        library.add(task.id, pattern, best_program)
    
    return best_program, best_score
```

Total cost per task:
    - Best case (perfect first try): ~350 tokens
    - Typical case (needs evolution): ~1600 tokens
    - Worst case (max iterations): ~3000 tokens
Another thing, not all functions are included, like ad, subtract etc etc, this is done. 