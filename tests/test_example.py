import pygad
import numpy as np

def fitness_func(ga_instance, x, x_idx):
    rng_noise =  np.random.default_rng(678910)
    dummy_fit = rng_noise.random()*100
    x = np.sort(x)
    return dummy_fit


gene_space = np.arange(1,41,1)

ga_instance = pygad.GA(num_generations = 20,
                           num_parents_mating = 40,
                           sol_per_pop = 50,
                           num_genes = 6,
                           init_range_low = gene_space[0],
                           init_range_high = gene_space[-1],
                           gene_space = gene_space,
                           gene_type = int,
                           keep_elitism = 2,
                           mutation_probability = 0.025,
                           fitness_func = fitness_func,
                           save_solutions = True,
                           allow_duplicate_genes = True,
                           suppress_warnings=True,
                           random_seed=12345)
ga_instance.run()

trial = ga_instance.solutions
trial = np.sort(trial)

unique_genes = []
for i_genes in range(trial.shape[0]):
    unique_genes.append(np.unique(trial[i_genes,:]))

for unique_gene in unique_genes:
    assert len(unique_gene) == len(trial[0])
