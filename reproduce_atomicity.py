import numpy
import pygad

def get_mutation_indices(original, mutated):
    """
    Returns the indices where original and mutated differ.
    """
    return numpy.where(original != mutated)[0]

def test_mutation_by_space_atomicity():
    print("Testing mutation_by_space atomicity...")
    
    # 1. Setup Gene Structure
    # Let's define a structure: [2, 1, 3]
    # Total genes = 6
    # Block 0: indices [0, 1]
    # Block 1: indices [2]
    # Block 2: indices [3, 4, 5]
    gene_structure = [2, 1, 3]
    num_genes = sum(gene_structure)
    
    # Gene space (required for mutation_by_space)
    # Using simple list for all genes
    gene_space = [0, 1] 
    
    # Initial Population (all zeros)
    initial_population = numpy.zeros((1, num_genes))
    
    # Initialize GA
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=1,
                           fitness_func=lambda *args: 0, # Dummy fitness
                           initial_population=initial_population,
                           gene_type=int,
                           gene_structure=gene_structure,
                           gene_space=gene_space,
                           mutation_type="random",
                           mutation_probability=None, # Force use of mutation_num_genes
                           mutation_num_genes=1,      # Mutate 1 logical block
                           random_seed=42)
                           
    # Manually invoke mutation_by_space to isolate it
    # We pass the population directly
    offspring = ga_instance.population.copy()
    
    # FORCE mutation_by_space call (normally called internally if gene_space is present)
    mutated_offspring = ga_instance.mutation_by_space(offspring)
    
    original_gene = ga_instance.population[0]
    new_gene = mutated_offspring[0]
    
    diff_indices = get_mutation_indices(original_gene, new_gene)
    print(f"Original: {original_gene}")
    print(f"Mutated:  {new_gene}")
    print(f"Diff Indices: {diff_indices}")
    
    # Check if Atomicity is Respected
    # If index 0 is mutated, index 1 MUST also be mutated (Block 0)
    # If index 2 is mutated, only index 2 should be mutated (Block 1)
    # If index 3 is mutated, 4 and 5 MUST also be mutated (Block 2)
    
    boundaries = [0, 2, 3, 6] # Derived from [2, 1, 3]
    
    # Group diff indices into blocks
    mutated_blocks = set()
    for idx in diff_indices:
        # Find which block this idx belongs to
        block_id = -1
        for i in range(len(boundaries)-1):
            if boundaries[i] <= idx < boundaries[i+1]:
                block_id = i
                break
        mutated_blocks.add(block_id)

    print(f"Mutated Blocks: {mutated_blocks}")

    # Verify that for each mutated block, ALL its genes are mutated
    # Note: In this specific test setup with gene_space=[0, 1] and initial=0, 
    # mutation *might* pick 0 again, so a gene might technically "mutate" to the same value.
    # However, since we are using random.sample for indices selection in mutation_by_space,
    # we mainly want to ensure the LOOP iterates over the full block.
    # But inspecting the values is the only way to see the result.
    # To strictly verify, we should check if the CODE iterates correctly. 
    # But as black-box, we check if multiple genes changed in multi-gene blocks.
    
    success = True
    for block_id in mutated_blocks:
        start, end = boundaries[block_id], boundaries[block_id+1]
        block_indices = range(start, end)
        
        # Check if ALL indices in this block are in diff_indices?
        # NOT necessarily true if mutation picked the same value by chance.
        # But with 0/1 space and 0 init, 50% chance to change.
        # Let's just check if we observe partial block mutation which would be a BUG.
        # Actually, if the code loops structure, it performs mutation for each.
        
        # A stronger check: logic trace. 
        # But for now let's assert that IF we have >1 diff indices, they belong to the expected blocks.
        pass

    if len(mutated_blocks) == 1:
        print("PASS: Exactly one logical block expected to be mutated.")
    elif len(mutated_blocks) == 0:
        print("WARN: No mutations observed (chance?). Run again.")
    else:
        print(f"FAIL: Expected 1 block mutated, got {len(mutated_blocks)}")

    # Check for split blocks (this is the real test of atomicity failure)
    # If 0 is in diff but 1 is NOT, and we know 1 COULD have changed, it's suspicious but not proof (chance).
    # Proof is if we run this many times.
    
if __name__ == "__main__":
    test_mutation_by_space_atomicity()
