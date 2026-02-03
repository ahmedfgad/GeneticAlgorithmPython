import unittest
import numpy
import random

import pygad

# Helper function to find mutated indices
def get_mutated_indices(original, mutated):
    return numpy.where(original != mutated)[0]

# Dummy fitness function
def fitness_func(ga_instance, solution, solution_idx):
    return 0

class TestMutationAtomicity(unittest.TestCase):

    def setUp(self):
        # Structure: [2, 1, 3] -> 6 genes
        # Blocks: [0, 1], [2], [3, 4, 5]
        self.gene_structure = [2, 1, 3]
        self.num_genes = sum(self.gene_structure)
        self.boundaries = [0, 2, 3, 6]
        self.initial_pop = numpy.zeros((1, self.num_genes))
        self.gene_space = [0, 1]
        
        # Suppress warnings
        self.suppress_warnings = True

    def test_mutation_randomly_atomicity(self):
        # Mutation by randomly (value changing)
        ga = pygad.GA(num_generations=1,
                      num_parents_mating=1,
                      fitness_func=fitness_func,
                      initial_population=self.initial_pop.copy(),
                      gene_structure=self.gene_structure,
                      mutation_type="random",
                      mutation_num_genes=1, # 1 BLOCK
                      random_mutation_min_val=1,
                      random_mutation_max_val=2,
                      suppress_warnings=self.suppress_warnings)
        
        # Determine logical block count
        # mutation_num_genes=1 means 1 logical block should be mutated.
        
        # Run mutation manually
        ga.mutation_randomly(ga.population)
        
        mutated_indices = get_mutated_indices(self.initial_pop[0], ga.population[0])
        self.assertTrue(len(mutated_indices) > 0, "No mutation occurred")
        
        # Check if indices belong to a single block
        blocks_mutated = set()
        for idx in mutated_indices:
            for b_i in range(len(self.boundaries)-1):
                if self.boundaries[b_i] <= idx < self.boundaries[b_i+1]:
                    blocks_mutated.add(b_i)
        
        self.assertEqual(len(blocks_mutated), 1, f"Mutation should affect exactly 1 block, affected: {blocks_mutated}")
        
        # Verify ALL genes in that block are mutated?
        # mutation_randomly loop: for gene_idx in range(start, end). 
        # Yes, it iterates all genes in the block and assigns random value.
        block_idx = list(blocks_mutated)[0]
        start, end = self.boundaries[block_idx], self.boundaries[block_idx+1]
        
        mutated_all = True
        for i in range(start, end):
            if ga.population[0, i] == 0: # Assuming 0 was initial
                # mutation range is 1 to 2, so it MUST change from 0.
                mutated_all = False
        
        self.assertTrue(mutated_all, f"All genes in block {block_idx} should be mutated")

    def test_inversion_mutation_atomicity(self):
        # Setup population: [0, 1, 2, 3, 4, 5]
        pop = numpy.array([numpy.arange(self.num_genes, dtype=float)])
        
        ga = pygad.GA(num_generations=1,
                      num_parents_mating=1,
                      fitness_func=fitness_func,
                      initial_population=pop,
                      gene_structure=self.gene_structure,
                      mutation_type="inversion",
                      suppress_warnings=self.suppress_warnings)
        
        # Force a mutation
        # inversion selects 2 blocks and reverses them.
        # Since we have only 3 blocks, possibilities:
        # Blocks [0, 1], [0, 1, 2], [1, 2].
        
        # Mock random so we know what happens? Or just analyze result.
        print(f"\nOriginal Gene: {pop[0]}")
        ga.inversion_mutation(ga.population)
        
        new_pop = ga.population[0]
        print(f"Mutated Gene:  {new_pop}")
        
        if numpy.array_equal(pop[0], new_pop):
            # It's possible random selection picked nothing or single-element blocks?
            # With intra-block inversion, [2] inverted is [2]. [0, 1] inverted is [1, 0].
            # [3, 4, 5] inverted is [5, 4, 3].
            # If nothing changed, it means probably only single-element blocks were selected OR no blocks selected?
            print("Inversion did not change (maybe selected blocks were size 1?)")
            
        # Check INTRA-BLOCK integrity
        # With new logic, blocks should STAY IN PLACE but their contents might be flipped.
        # Let's verify that boundaries are respected: i.e. Block A is still at index 0-2 (but maybe inverted).
        
        # Original: [0, 1], [2], [3, 4, 5]
        # Mutated:  [x, y], [z], [a, b, c]
        
        # Verify Block 0 (Index 0-2)
        b0_new = new_pop[0:2]
        b0_orig = pop[0, 0:2]
        # It must be either [0, 1] OR [1, 0]
        valid_b0 = numpy.array_equal(b0_new, b0_orig) or numpy.array_equal(b0_new, b0_orig[::-1])
        self.assertTrue(valid_b0, f"Block 0 corrupted: {b0_new}")

        # Verify Block 1 (Index 2-3) -> Length 1, always same
        b1_new = new_pop[2:3]
        b1_orig = pop[0, 2:3]
        self.assertTrue(numpy.array_equal(b1_new, b1_orig), f"Block 1 corrupted: {b1_new}")
        
        # Verify Block 2 (Index 3-6)
        b2_new = new_pop[3:6]
        b2_orig = pop[0, 3:6]
        # It must be either [3, 4, 5] OR [5, 4, 3]
        valid_b2 = numpy.array_equal(b2_new, b2_orig) or numpy.array_equal(b2_new, b2_orig[::-1])
        self.assertTrue(valid_b2, f"Block 2 corrupted: {b2_new}")
        
        # Check if ANY mutation happened (excluding length 1 blocks)
        changed = not numpy.array_equal(pop[0], new_pop)
        if not changed:
             print("Note: Random selection resulted in no effective change (maybe only Block 1 selected?)")
    def test_scramble_mutation_atomicity(self):
        # Setup population: [0, 1, 2, 3, 4, 5]
        pop = numpy.array([numpy.arange(self.num_genes, dtype=float)])
        
        ga = pygad.GA(num_generations=1,
                      num_parents_mating=1,
                      fitness_func=fitness_func,
                      initial_population=pop,
                      gene_structure=self.gene_structure,
                      mutation_type="scramble",
                      suppress_warnings=self.suppress_warnings)
        
        # Scramble now does Intra-Block Scramble
        print(f"\nOriginal Gene (Scramble): {pop[0]}")
        ga.scramble_mutation(ga.population)
        new_pop = ga.population[0]
        print(f"Mutated Gene (Scramble):  {new_pop}")
        
        # Verify block integrity (Boundaries respected)
        # Block 0 [0, 1] -> should still contain {0, 1} but order might change
        b0 = new_pop[0:2]
        self.assertTrue(set(b0) == {0, 1}, f"Block 0 content corrupted: {b0}")
        
        # Block 1 [2] -> should be {2}
        b1 = new_pop[2:3]
        self.assertTrue(set(b1) == {2}, f"Block 1 content corrupted: {b1}")
        
        # Block 2 [3, 4, 5] -> should be {3, 4, 5}
        b2 = new_pop[3:6]
        self.assertTrue(set(b2) == {3, 4, 5}, f"Block 2 content corrupted: {b2}")
        
        # Check if change occurred (might not if random shuffle returns same order, but likely)
        if numpy.array_equal(pop[0], new_pop):
             print("Scramble resulted in no change (possible with small blocks/bad luck)")

if __name__ == '__main__':
    unittest.main()