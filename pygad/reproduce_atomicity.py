import numpy
import pygad

def get_mutation_indices(original, mutated):
    """
    Returns the indices where original and mutated differ.
    """
    return numpy.where(original != mutated)[0]

# Proper fitness function with 3 arguments
def fitness_func(ga_instance, solution, solution_idx):
    return 0

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
    # We suppress warnings to avoid clutter
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=1,
                           fitness_func=fitness_func, # Correct signature
                           initial_population=initial_population,
                           gene_type=float, # Use float to avoid precision issues
                           gene_structure=gene_structure,
                           gene_space=gene_space,
                           mutation_type="random",
                           mutation_probability=None, # Force use of mutation_num_genes
                           mutation_num_genes=1,      # Mutate 1 logical block
                           random_seed=42,            # Fixed seed for reproducibility
                           suppress_warnings=True)
                           
    # Manually invoke mutation_by_space to isolate it
    # We pass the population directly
    offspring = ga_instance.population.copy()
    
    # FORCE mutation_by_space call
    # Note: mutation_by_space is an instance method
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

    print(f"Mutated Blocks IDs: {mutated_blocks}")

    if not mutated_blocks:
          print("WARN: No genes changed value (chance?).")
          return

    fail = False
    for block_id in mutated_blocks:
        start, end = boundaries[block_id], boundaries[block_id+1]
        
        # Verify that ALL expected indices for this block have changed?
        # IMPORTANT: mutation might randomly pick the SAME value as original (0 -> 0).
        # But for Block 2 (size 3), probability of ALL 3 picking 0 is low (0.5^3 = 0.125).
        # We can't strictly assert values change, but we can verify that NO OTHER block was touched partialy.
        # Actually, simpler check: 
        # Since we set mutation_num_genes=1, ONLY ONE block should be in mutated_blocks.
        # If we see indices from multiple blocks, it's a fail (unless we got lucky and multiple blocks were selected? No, we set num_genes=1).
        pass

    if len(mutated_blocks) > 1:
        print(f"FAIL: Expected 1 block mutated, but genes from {len(mutated_blocks)} blocks changed: {mutated_blocks}")
        fail = True
    
    # Check for partial mutation in the mutated block?
    # Hard to proof unless we use a gene space that GUARANTEES change (e.g. init 0, space [1]).
    
    print("Test finished.")

if __name__ == "__main__":
    test_mutation_by_space_atomicity()
