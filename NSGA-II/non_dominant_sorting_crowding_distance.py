import numpy

population_fitness = numpy.array([[20, 2.2],
                                  [60, 4.4],
                                  [65, 3.5],
                                  [15, 4.4],
                                  [55, 4.5],
                                  [50, 1.8],
                                  [80, 4.0],
                                  [25, 4.6]])

def get_non_dominated_set(curr_solutions):
    """
    Get the set of non-dominated solutions from the current set of solutions.

    Parameters
    ----------
    curr_solutions : TYPE
        The set of solutions to find its non-dominated set.

    Returns
    -------
    dominated_set : TYPE
        A set of the dominated solutions.
    non_dominated_set : TYPE
        A set of the non-dominated set.

    """
    # List of the members of the current dominated pareto front/set.
    dominated_set = []
    # List of the non-members of the current dominated pareto front/set.
    non_dominated_set = []
    for idx1, sol1 in enumerate(curr_solutions):
        # Flag indicates whether the solution is a member of the current dominated set.
        is_dominated = True
        for idx2, sol2 in enumerate(curr_solutions):
            if idx1 == idx2:
                continue
            # Zipping the 2 solutions so the corresponding genes are in the same list.
            # The returned array is of size (N, 2) where N is the number of genes.
            two_solutions = numpy.array(list(zip(sol1[1], sol2[1])))

            #TODO Consider repacing < by > for maximization problems.
            # Checking for if any solution dominates the current solution by applying the 2 conditions.
            # le_eq (less than or equal): All elements must be True.
            # le (less than): Only 1 element must be True.
            le_eq = two_solutions[:, 1] <= two_solutions[:, 0]
            le = two_solutions[:, 1] < two_solutions[:, 0]

            # If the 2 conditions hold, then a solution dominates the current solution.
            # The current solution is not considered a member of the dominated set.
            if le_eq.all() and le.any():
                # Set the is_dominated flag to False to NOT insert the current solution in the current dominated set.
                # Instead, insert it into the non-dominated set.
                is_dominated = False
                non_dominated_set.append(sol1)
                break
            else:
                # Reaching here means the solution does not dominate the current solution.
                pass

        # If the flag is True, then no solution dominates the current solution.
        if is_dominated:
            dominated_set.append(sol1)

    # Return the dominated and non-dominated sets.
    return dominated_set, non_dominated_set

def non_dominated_sorting(population_fitness):
    """
    Apply the non-dominant sorting over the population_fitness to create the pareto fronts based on non-dominaned sorting of the solutions.

    Parameters
    ----------
    population_fitness : TYPE
        An array of the population fitness across all objective function.

    Returns
    -------
    pareto_fronts : TYPE
        An array of the pareto fronts.

    """
    # A list of all non-dominated sets.
    pareto_fronts = []

    # The remaining set to be explored for non-dominance.
    # Initially it is set to the entire population.
    # The solutions of each non-dominated set are removed after each iteration.
    remaining_set = population_fitness.copy()

    # Zipping the solution index with the solution's fitness.
    # This helps to easily identify the index of each solution.
    # Each element has:
        # 1) The index of the solution.
        # 2) An array of the fitness values of this solution across all objectives.
    # remaining_set = numpy.array(list(zip(range(0, population_fitness.shape[0]), non_dominated_set)))
    remaining_set = list(zip(range(0, population_fitness.shape[0]), remaining_set))

    # A list mapping the index of each pareto front to the set of solutions in this front.
    solutions_fronts_indices = [-1]*len(remaining_set)
    solutions_fronts_indices = numpy.array(solutions_fronts_indices)

    # Index of the current pareto front.
    front_index = -1
    while len(remaining_set) > 0:
        front_index += 1

        # Get the current non-dominated set of solutions.
        pareto_front, remaining_set = get_non_dominated_set(curr_solutions=remaining_set)
        pareto_front = numpy.array(pareto_front, dtype=object)
        pareto_fronts.append(pareto_front)

        solutions_indices = pareto_front[:, 0].astype(int)
        solutions_fronts_indices[solutions_indices] = front_index

    return pareto_fronts, solutions_fronts_indices

def crowding_distance(pareto_front):
    """
    Calculate the crowding dstance for all solutions in the current pareto front.

    Parameters
    ----------
    pareto_front : TYPE
        The set of solutions in the current pareto front.

    Returns
    -------
    obj_crowding_dist_list : TYPE
        A nested list of the values for all objectives alongside their crowding distance.
    crowding_dist_sum : TYPE
        A list of the sum of crowding distances across all objectives for each solution.
    crowding_dist_front_sorted_indices : TYPE
        The indices of the solutions (relative to the current front) sorted by the crowding distance.
    crowding_dist_pop_sorted_indices : TYPE
        The indices of the solutions (relative to the population) sorted by the crowding distance.
    """

    # Each solution in the pareto front has 2 elements:
        # 1) The index of the solution in the population.
        # 2) A list of the fitness values for all objectives of the solution.
    # Before proceeding, remove the indices from each solution in the pareto front.
    pareto_front_no_indices = numpy.array([pareto_front[:, 1][idx] for idx in range(pareto_front.shape[0])])

    # If there is only 1 solution, then return empty arrays for the crowding distance.
    if pareto_front_no_indices.shape[0] == 1:
        # There is only 1 index.
        return numpy.array([]), numpy.array([]), numpy.array([0]), pareto_front[:, 0].astype(int)

    # An empty list holding info about the objectives of each solution. The info includes the objective value and crowding distance.
    obj_crowding_dist_list = []
    # Loop through the objectives to calculate the crowding distance of each solution across all objectives.
    for obj_idx in range(pareto_front_no_indices.shape[1]):
        obj = pareto_front_no_indices[:, obj_idx]
        # This variable has a nested list where each child list zip the following together:
            # 1) The index of the objective value.
            # 2) The objective value.
            # 3) Initialize the crowding distance by zero.
        obj = list(zip(range(len(obj)), obj, [0]*len(obj)))
        obj = [list(element) for element in obj]
        # This variable is the sorted version where sorting is done by the objective value (second element).
        # Note that the first element is still the original objective index before sorting.
        obj_sorted = sorted(obj, key=lambda x: x[1])
    
        # Get the minimum and maximum values for the current objective.
        obj_min_val = min(population_fitness[:, obj_idx])
        obj_max_val = max(population_fitness[:, obj_idx])
        denominator = obj_max_val - obj_min_val
        # To avoid division by zero, set the denominator to a tiny value.
        if denominator == 0:
            denominator = 0.0000001

        # Set the crowding distance to the first and last solutions (after being sorted) to infinity.
        inf_val = float('inf')
        # crowding_distance[0] = inf_val
        obj_sorted[0][2] = inf_val
        # crowding_distance[-1] = inf_val
        obj_sorted[-1][2] = inf_val

        # If there are only 2 solutions in the current pareto front, then do not proceed.
        # The crowding distance for such 2 solutions is infinity.
        if len(obj_sorted) <= 2:
            break
    
        for idx in range(1, len(obj_sorted)-1):
            # Calculate the crowding distance.
            crowding_dist = obj_sorted[idx+1][1] - obj_sorted[idx-1][1]
            crowding_dist = crowding_dist / denominator
            # Insert the crowding distance back into the list to override the initial zero.
            obj_sorted[idx][2] = crowding_dist
    
        # Sort the objective by the original index at index 0 of the each child list.
        obj_sorted = sorted(obj_sorted, key=lambda x: x[0])
        obj_crowding_dist_list.append(obj_sorted)

    obj_crowding_dist_list = numpy.array(obj_crowding_dist_list)
    crowding_dist = numpy.array([obj_crowding_dist_list[idx, :, 2] for idx in range(len(obj_crowding_dist_list))])
    crowding_dist_sum = numpy.sum(crowding_dist, axis=0)

    # An array of the sum of crowding distances across all objectives. 
    # Each row has 2 elements:
        # 1) The index of the solution.
        # 2) The sum of all crowding distances for all objective of the solution.
    crowding_dist_sum = numpy.array(list(zip(obj_crowding_dist_list[0, :, 0], crowding_dist_sum)))
    crowding_dist_sum = sorted(crowding_dist_sum, key=lambda x: x[1], reverse=True)

    # The sorted solutions' indices by the crowding distance.
    crowding_dist_front_sorted_indices = numpy.array(crowding_dist_sum)[:, 0]
    crowding_dist_front_sorted_indices = crowding_dist_front_sorted_indices.astype(int)
    # Note that such indices are relative to the front, NOT the population.
    # It is mandatory to map such front indices to population indices before using them to refer to the population.
    crowding_dist_pop_sorted_indices = pareto_front[:, 0]
    crowding_dist_pop_sorted_indices = crowding_dist_pop_sorted_indices[crowding_dist_front_sorted_indices]
    crowding_dist_pop_sorted_indices = crowding_dist_pop_sorted_indices.astype(int)

    return obj_crowding_dist_list, crowding_dist_sum, crowding_dist_front_sorted_indices, crowding_dist_pop_sorted_indices

def tournament_selection_nsga2(self,
                               pareto_fronts,
                               solutions_fronts_indices, 
                               num_parents):

    """
    Select the parents using the tournament selection technique for NSGA-II. 
    The traditional tournament selection uses the fitness values. But the tournament selection for NSGA-II uses non-dominated sorting and crowding distance.
    Using non-dominated sorting, the solutions are distributed across pareto fronts. The fronts are given the indices 0, 1, 2, ..., N where N is the number of pareto fronts. The lower the index of the pareto front, the better its solutions.
    To select the parents solutions, 2 solutions are selected randomly. If the 2 solutions are in different pareto fronts, then the solution comming from a pareto front with lower index is selected.
    If 2 solutions are in the same pareto front, then crowding distance is calculated. The solution with the higher crowding distance is selected.
    If the 2 solutions are in the same pareto front and have the same crowding distance, then a solution is randomly selected.
    Later, the selected parents will mate to produce the offspring.

    It accepts 2 parameters:
        -pareto_fronts: A nested array of all the pareto fronts. Each front has its solutions.
        -solutions_fronts_indices: A list of the pareto front index of each solution in the current population.
        -num_parents: The number of parents to be selected.

    It returns an array of the selected parents alongside their indices in the population.
    """

    if self.gene_type_single == True:
        parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
    else:
        parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)

    # The indices of the selected parents.
    parents_indices = []

    # Randomly generate pairs of indices to apply for NSGA-II tournament selection for selecting the parents solutions.
    rand_indices = numpy.random.randint(low=0.0, 
                                        high=len(solutions_fronts_indices), 
                                        size=(num_parents, self.K_tournament))
    # rand_indices[0, 0] = 5
    # rand_indices[0, 1] = 3
    # rand_indices[1, 0] = 1
    # rand_indices[1, 1] = 6

    for parent_num in range(num_parents):
        # Return the indices of the current 2 solutions.
        current_indices = rand_indices[parent_num]
        # Return the front index of the 2 solutions.
        parent_fronts_indices = solutions_fronts_indices[current_indices]

        if parent_fronts_indices[0] < parent_fronts_indices[1]:
            # If the first solution is in a lower pareto front than the second, then select it.
            selected_parent_idx = current_indices[0]
        elif parent_fronts_indices[0] > parent_fronts_indices[1]:
            # If the second solution is in a lower pareto front than the first, then select it.
            selected_parent_idx = current_indices[1]
        else:
            # The 2 solutions are in the same pareto front.
            # The selection is made using the crowding distance.

            # A list holding the crowding distance of the current 2 solutions. It is initialized to -1.
            solutions_crowding_distance = [-1, -1]

            # Fetch the current pareto front.
            pareto_front = pareto_fronts[parent_fronts_indices[0]] # Index 1 can also be used.

            # If there is only 1 solution in the pareto front, just return it without calculating the crowding distance (it is useless).
            if pareto_front.shape[0] == 1:
                selected_parent_idx = current_indices[0] # Index 1 can also be used.
            else:
                # Reaching here means the pareto front has more than 1 solution.

                # Calculate the crowding distance of the solutions of the pareto front.
                obj_crowding_distance_list, crowding_distance_sum, crowding_dist_front_sorted_indices, crowding_dist_pop_sorted_indices = crowding_distance(pareto_front.copy())

                # This list has the sorted front-based indices for the solutions in the current pareto front.
                crowding_dist_front_sorted_indices = list(crowding_dist_front_sorted_indices)
                # This list has the sorted population-based indices for the solutions in the current pareto front.
                crowding_dist_pop_sorted_indices = list(crowding_dist_pop_sorted_indices)

                # Return the indices of the solutions from the pareto front.
                solution1_idx = crowding_dist_pop_sorted_indices.index(current_indices[0])
                solution2_idx = crowding_dist_pop_sorted_indices.index(current_indices[1])
    
                # Fetch the crowding distance using the indices.
                solutions_crowding_distance[0] = crowding_distance_sum[solution1_idx][1]
                solutions_crowding_distance[1] = crowding_distance_sum[solution2_idx][1]
    
                # # Instead of using the crowding distance, we can select the solution that comes first in the list.
                # # Its limitation is that it is biased towards the low indexed solution if the 2 solutions have the same crowding distance.
                # if solution1_idx < solution2_idx:
                #     # Select the first solution if it has higher crowding distance.
                #     selected_parent_idx = current_indices[0]
                # else:
                #     # Select the second solution if it has higher crowding distance.
                #     selected_parent_idx = current_indices[1]
    
                if solutions_crowding_distance[0] > solutions_crowding_distance[1]:
                    # Select the first solution if it has higher crowding distance.
                    selected_parent_idx = current_indices[0]
                elif solutions_crowding_distance[1] > solutions_crowding_distance[0]:
                    # Select the second solution if it has higher crowding distance.
                    selected_parent_idx = current_indices[1]
                else:
                    # If the crowding distance is equal, select the parent randomly.
                    rand_num = numpy.random.uniform()
                    if rand_num < 0.5:
                        # If the random number is < 0.5, then select the first solution.
                        selected_parent_idx = current_indices[0]
                    else:
                        # If the random number is >= 0.5, then select the second solution.
                        selected_parent_idx = current_indices[1]

        # Insert the selected parent index.
        parents_indices.append(selected_parent_idx)
        # Insert the selected parent.
        parents[parent_num, :] = self.population[selected_parent_idx, :].copy()

    # Make sure the parents indices is returned as a NumPy array.
    return parents, numpy.array(parents_indices)

def nsga2_selection(self,
                    pareto_fronts,
                    solutions_fronts_indices, 
                    num_parents):

    """
    Select the parents using the Non-Dominated Sorting Genetic Algorithm II (NSGA-II). 
    The selection is done using non-dominated sorting and crowding distance.
    Using non-dominated sorting, the solutions are distributed across pareto fronts. The fronts are given the indices 0, 1, 2, ..., N where N is the number of pareto fronts. The lower the index of the pareto front, the better its solutions.
    The parents are selected from the lower pareto fronts and moving up until selecting the number of desired parents. 
    A solution from a pareto front X cannot be taken as a parent until all solutions in pareto front Y is selected given that Y < X.
    For a pareto front X, if only a subset of its solutions is needed, then the corwding distance is used to determine which solutions to be selected from the front. The solution with the higher crowding distance is selected.
    If the 2 solutions are in the same pareto front and have the same crowding distance, then a solution is randomly selected.
    Later, the selected parents will mate to produce the offspring.

    It accepts 2 parameters:
        -pareto_fronts: A nested array of all the pareto fronts. Each front has its solutions.
        -solutions_fronts_indices: A list of the pareto front index of each solution in the current population.
        -num_parents: The number of parents to be selected.

    It returns an array of the selected parents alongside their indices in the population.
    """

    if self.gene_type_single == True:
        parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
    else:
        parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)

    # The indices of the selected parents.
    parents_indices = []

    # The number of remaining parents to be selected.
    num_remaining_parents = num_parents

    # A loop variable holding the index of the current pareto front.
    pareto_front_idx = 0
    while num_remaining_parents != 0 and pareto_front_idx < len(pareto_fronts):
        # Return the current pareto front.
        current_pareto_front = pareto_fronts[pareto_front_idx]
        # Check if the entire front fits into the parents array.
        # If so, then insert all the solutions in the current front into the parents array.
        if num_remaining_parents >= len(current_pareto_front):
            for sol_idx in range(len(current_pareto_front)):
                selected_solution_idx = current_pareto_front[sol_idx, 0]
                # Insert the parent into the parents array.
                parents[sol_idx, :] = self.population[selected_solution_idx, :].copy()
                # Insert the index of the selected parent.
                parents_indices.append(selected_solution_idx)

            # Decrement the number of remaining parents by the length of the pareto front.
            num_remaining_parents -= len(current_pareto_front)
        else:
            # If only a subset of the front is needed, then use the crowding distance to sort the solutions and select only the number needed.

            # Calculate the crowding distance of the solutions of the pareto front.
            obj_crowding_distance_list, crowding_distance_sum, crowding_dist_front_sorted_indices, crowding_dist_pop_sorted_indices = crowding_distance(current_pareto_front.copy())

            for selected_solution_idx in crowding_dist_pop_sorted_indices[0:num_remaining_parents]:
                # Insert the parent into the parents array.
                parents[sol_idx, :] = self.population[selected_solution_idx, :].copy()
                # Insert the index of the selected parent.
                parents_indices.append(selected_solution_idx)

            # Decrement the number of remaining parents by the number of selected parents.
            num_remaining_parents -= num_remaining_parents

        # Increase the pareto front index to take parents from the next front.
        pareto_front_idx += 1

    # Make sure the parents indices is returned as a NumPy array.
    return parents, numpy.array(parents_indices)


pareto_fronts, solutions_fronts_indices = non_dominated_sorting(population_fitness)
# # print('\nsolutions_fronts_indices\n', solutions_fronts_indices)
# for i, s in enumerate(pareto_fronts):
#     # print(f'Dominated Pareto Front Set {i+1}:\n{s}')
#     print(f'Dominated Pareto Front Indices {i+1}:\n{s[:, 0]}')
# print("\n\n\n--------------------")

class Object(object):
    pass

obj = Object()
obj.population = numpy.random.rand(8, 4)
obj.gene_type_single = True
obj.gene_type = [float, 0]
obj.K_tournament = 2

parents, parents_indices = tournament_selection_nsga2(self=obj, 
                                                      pareto_fronts=pareto_fronts,
                                                      solutions_fronts_indices=solutions_fronts_indices,
                                                      num_parents=40)
print(f'Tournament Parent Selection for NSGA-II - Indices: \n{parents_indices}')

parents, parents_indices = nsga2_selection(self=obj,
                                           pareto_fronts=pareto_fronts,
                                           solutions_fronts_indices=solutions_fronts_indices, 
                                           num_parents=40)
print(f'NSGA-II Parent Selection - Indices: \n{parents_indices}')

# for idx in range(len(pareto_fronts)):
#     # Fetch the current pareto front.
#     pareto_front = pareto_fronts[idx]
#     obj_crowding_distance_list, crowding_distance_sum, crowding_dist_front_sorted_indices, crowding_dist_pop_sorted_indices = crowding_distance(pareto_front.copy())
#     print('Front IDX', crowding_dist_front_sorted_indices)
#     print('POP IDX  ', crowding_dist_pop_sorted_indices)
#     print(f'Sorted Sum of Crowd Dists\n{crowding_distance_sum}')

# # Fetch the current pareto front.
# pareto_front = pareto_fronts[0]
# obj_crowding_distance_list, crowding_distance_sum, crowding_dist_front_sorted_indices, crowding_dist_pop_sorted_indices = crowding_distance(pareto_front.copy())
# print('\n', crowding_dist_pop_sorted_indices)
# print(f'Sorted Sum of Crowd Dists\n{crowding_distance_sum}')
