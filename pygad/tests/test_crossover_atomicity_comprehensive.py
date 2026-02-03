import unittest
import numpy
import pygad

def fitness_func(ga_instance, solution, solution_idx):
    return numpy.sum(solution)

class TestCrossoverAtomicity(unittest.TestCase):
    def setUp(self):
        # Define a gene structure: [2, 1, 3]
        # Logical Block 0: indices 0, 1 (length 2)
        # Logical Block 1: index 2 (length 1)
        # Logical Block 2: indices 3, 4, 5 (length 3)
        # Total genes: 6
        # Valid boundaries: 0, 2, 3, 6
        self.gene_structure = [2, 1, 3]
        self.num_genes = sum(self.gene_structure)
        self.suppress_warnings = True

    def test_single_point_crossover_atomicity(self):
        """
        Verify that single point crossover only cuts at logical boundaries.
        """
        # Parents:
        # P1: [0, 0, 0, 0, 0, 0]
        # P2: [1, 1, 1, 1, 1, 1]
        parent1 = numpy.zeros(self.num_genes)
        parent2 = numpy.ones(self.num_genes)
        parents = numpy.array([parent1, parent2])
        
        ga = pygad.GA(num_generations=1,
                      num_parents_mating=2,
                      fitness_func=fitness_func,
                      initial_population=parents.copy(), # Not used for crossover call directly but needed for init
                      gene_structure=self.gene_structure,
                      crossover_type="single_point",
                      crossover_probability=1.0, # Force crossover
                      suppress_warnings=self.suppress_warnings)
        
        # We need to simulate the crossover call manually or inspect offspring
        # pygad.GA.crossover method takes parents and offspring_size
        offspring_size = (100, self.num_genes) # Generate many offspring to check all possible cuts
        
        # We must access the crossover method. It is part of the GA instance mixed in? 
        # Actually it's in ga.crossover(...) but that's the high level loop.
        # The specific methods are methods of the class if mixed in.
        # Let's call the specific crossover function if valid, or run a generation.
        # Easier to call method directly: ga.single_point_crossover(parents, offspring_size)
        
        offspring = ga.single_point_crossover(parents, offspring_size)
        
        # Verify EACH offspring
        # A valid offspring must look like: [0...0 1...1] or [1...1 0...0] (if parents order mixed)
        # The transition point MUST be at index 2 or 3 (0 and 6 are trivial).
        
        valid_boundaries = {0, 2, 3, 6}
        
        for child in offspring:
            # Find transition points
            # Diff array: non-zero where value changes
            diffs = numpy.diff(child)
            changes = numpy.where(diffs != 0)[0]
            
            # changes gives index i where child[i] != child[i+1].
            # So the cut is after i. The boundary is i+1.
            # Example: [0, 0, 1, 1, 1, 1] -> diff at index 1 (value 0!=1). Cut at 2.
            
            cut_indices = changes + 1
            
            for cut in cut_indices:
                self.assertIn(cut, valid_boundaries, f"Invalid cut at index {cut} for child {child}")

    def test_two_points_crossover_atomicity(self):
        """
        Verify that two points crossover only cuts at logical boundaries.
        """
        parent1 = numpy.zeros(self.num_genes)
        parent2 = numpy.ones(self.num_genes)
        parents = numpy.array([parent1, parent2])
        
        ga = pygad.GA(num_generations=1,
                      num_parents_mating=2,
                      fitness_func=fitness_func,
                      initial_population=parents.copy(),
                      gene_structure=self.gene_structure,
                      crossover_type="two_points",
                      crossover_probability=1.0,
                      suppress_warnings=self.suppress_warnings)
        
        offspring_size = (100, self.num_genes)
        offspring = ga.two_points_crossover(parents, offspring_size)
        
        valid_boundaries = {0, 2, 3, 6}
        
        for child in offspring:
            diffs = numpy.diff(child)
            changes = numpy.where(diffs != 0)[0]
            cut_indices = changes + 1
            
            for cut in cut_indices:
                self.assertIn(cut, valid_boundaries, f"Invalid cut at index {cut} for child {child}")

    def test_uniform_crossover_atomicity(self):
        """
        Verify that uniform crossover respects block integrity (all genes in a block are from same parent).
        """
        # Parents with distinct values to identify inheritance
        # Block 0 (0-2): [0, 0] vs [10, 10]
        # Block 1 (2-3): [1, 1] vs [11, 11]
        # Block 2 (3-6): [2, 2, 2] vs [12, 12, 12]
        
        # P1: [0, 0, 1, 2, 2, 2]
        # P2: [10, 10, 11, 12, 12, 12]
        
        # To make it easier:
        # P1: [0, 0, 0, 0, 0, 0]
        # P2: [1, 1, 1, 1, 1, 1]
        # Is enough to check consistency.
        
        parent1 = numpy.zeros(self.num_genes)
        parent2 = numpy.ones(self.num_genes)
        parents = numpy.array([parent1, parent2])
        
        ga = pygad.GA(num_generations=1,
                      num_parents_mating=2,
                      fitness_func=fitness_func,
                      initial_population=parents.copy(),
                      gene_structure=self.gene_structure,
                      crossover_type="uniform",
                      crossover_probability=1.0,
                      suppress_warnings=self.suppress_warnings)
        
        offspring_size = (100, self.num_genes)
        offspring = ga.uniform_crossover(parents, offspring_size)
        
        for child in offspring:
            # Check Block 0: Indices 0, 1 must be EQUAL
            self.assertEqual(child[0], child[1], f"Block 0 split! {child}")
            
            # Check Block 1: Index 2 (len 1, always valid)
            pass
            
            # Check Block 2: Indices 3, 4, 5 must be EQUAL
            self.assertEqual(child[3], child[4], f"Block 2 split! {child}")
            self.assertEqual(child[4], child[5], f"Block 2 split! {child}")

    def test_scattered_crossover_atomicity(self):
        """
        Verify that scattered crossover (same logic as uniform usually) respects block integrity.
        """
        parent1 = numpy.zeros(self.num_genes)
        parent2 = numpy.ones(self.num_genes)
        parents = numpy.array([parent1, parent2])
        
        ga = pygad.GA(num_generations=1,
                      num_parents_mating=2,
                      fitness_func=fitness_func,
                      initial_population=parents.copy(),
                      gene_structure=self.gene_structure,
                      crossover_type="scattered",
                      crossover_probability=1.0,
                      suppress_warnings=self.suppress_warnings)
        
        offspring_size = (100, self.num_genes)
        offspring = ga.scattered_crossover(parents, offspring_size)
        
        for child in offspring:
            # Check Block 0: Indices 0, 1 must be EQUAL
            self.assertEqual(child[0], child[1], f"Block 0 split! {child}")
            # Check Block 2: Indices 3, 4, 5 must be EQUAL
            self.assertEqual(child[3], child[4], f"Block 2 split! {child}")
            self.assertEqual(child[4], child[5], f"Block 2 split! {child}")

if __name__ == '__main__':
    unittest.main()
