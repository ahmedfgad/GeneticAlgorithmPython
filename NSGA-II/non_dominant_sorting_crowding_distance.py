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
    Apply the non-dominant sorting over the population_fitness to create sets of non-dominaned solutions.

    Parameters
    ----------
    population_fitness : TYPE
        An array of the population fitness across all objective function.

    Returns
    -------
    non_dominated_sets : TYPE
        An array of the non-dominated sets.

    """
    # A list of all non-dominated sets.
    non_dominated_sets = []

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
    while len(remaining_set) > 0:
        # Get the current non-dominated set of solutions.
        d1, remaining_set = get_non_dominated_set(curr_solutions=remaining_set)
        non_dominated_sets.append(numpy.array(d1, dtype=object))
    return non_dominated_sets

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
    """

    # Each solution in the pareto front has 2 elements:
        # 1) The index of the solution in the population.
        # 2) A list of the fitness values for all objectives of the solution.
    # Before proceeding, remove the indices from each solution in the pareto front.
    pareto_front = numpy.array([pareto_front[idx] for idx in range(pareto_front.shape[0])])

    # If there is only 1 solution, then return empty arrays for the crowding distance.
    if pareto_front.shape[0] == 1:
        return numpy.array([]), numpy.array([])

    # An empty list holding info about the objectives of each solution. The info includes the objective value and crowding distance.
    obj_crowding_dist_list = []
    # Loop through the objectives to calculate the crowding distance of each solution across all objectives.
    for obj_idx in range(pareto_front.shape[1]):
        obj = pareto_front[:, obj_idx]
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

    return obj_crowding_dist_list, crowding_dist_sum

non_dominated_sets = non_dominated_sorting(population_fitness)

# for i, s in enumerate(non_dominated_sets):
#     print(f'dominated Pareto Front Set {i+1}:\n{s}')
# print("\n\n\n--------------------")

# Fetch the current pareto front.
pareto_front = non_dominated_sets[1][:, 1]
obj_crowding_distance_list, crowding_distance_sum = crowding_distance(pareto_front)

print(obj_crowding_distance_list)
print(f'\nSorted Sum of Crowd Dists\n{crowding_distance_sum}')
