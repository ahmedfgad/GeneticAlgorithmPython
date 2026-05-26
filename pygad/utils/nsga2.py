import numpy
import pygad

class NSGA2:

    def __init__(self):
        pass

    def get_non_dominated_set(self, curr_solutions):
        """
        Split the input solutions into a non-dominated set and a
        dominated set. The non-dominated set is the next Pareto front.

        Parameters
        ----------
        curr_solutions : list
            A list of (index, fitness_vector) pairs to be partitioned.
            ``index`` is the position of the solution in the original
            population and ``fitness_vector`` is its objective values.

        Returns
        -------
        dominated_set : list
            The (index, fitness_vector) pairs that are dominated by at
            least one other solution in ``curr_solutions``.
        non_dominated_set : list
            The (index, fitness_vector) pairs that no other solution in
            ``curr_solutions`` dominates. These form the current Pareto
            front.
        """
        # List of the members of the current dominated pareto front/set.
        dominated_set = []
        # List of the non-members of the current dominated pareto front/set.
        # The non-dominated set is the pareto front set.
        non_dominated_set = []
        for idx1, sol1 in enumerate(curr_solutions):
            # Flag indicates whether the solution is a member of the current dominated set.
            is_not_dominated = True
            for idx2, sol2 in enumerate(curr_solutions):
                if idx1 == idx2:
                    continue

                # Zipping the 2 solutions so the corresponding genes are in the same list.
                # The returned array is of size (N, 2) where N is the number of genes.
                two_solutions = numpy.array(list(zip(sol1[1], sol2[1])))

                # Use < for minimization problems and > for maximization problems.
                # Checking if any solution dominates the current solution by applying the 2 conditions.
                # gr_eq (greater than or equal): All elements must be True.
                # gr (greater than): Only 1 element must be True.
                gr_eq = two_solutions[:, 1] >= two_solutions[:, 0]
                gr = two_solutions[:, 1] > two_solutions[:, 0]

                # If the 2 conditions hold, then a solution (sol2) dominates the current solution (sol1).
                # The current solution (sol1) is not considered a member of the non-dominated set.
                if gr_eq.all() and gr.any():
                    # Set the is_not_dominated flag to False because another solution dominates the current solution (sol1)
                    is_not_dominated = False
                    # DO NOT insert the current solution in the current non-dominated set.
                    # Instead, insert it into the dominated set.
                    dominated_set.append(sol1)
                    break
                else:
                    # Reaching here means the solution does not dominate the current solution.
                    pass
    
            # If the flag is True, then no solution dominates the current solution.
            # Insert the current solution (sol1) into the non-dominated set.
            if is_not_dominated:
                non_dominated_set.append(sol1)
    
        # Return the dominated and non-dominated sets.
        return dominated_set, non_dominated_set

    def non_dominated_sorting(self, fitness):
        """
        Sort the population into Pareto fronts using non-dominated
        sorting. Front 0 contains the solutions no other solution
        dominates; front 1 contains those that are only dominated by
        front 0; and so on.

        Only works for multi-objective problems.

        Parameters
        ----------
        fitness : numpy.ndarray
            A 2D array of fitness values, one row per solution and one
            column per objective.

        Returns
        -------
        pareto_fronts : list
            A list of Pareto fronts. Each front is a numpy array whose
            rows are (population_index, fitness_vector) pairs.
        solutions_fronts_indices : numpy.ndarray
            A 1D integer array of length ``len(fitness)``. Entry ``i``
            is the index of the Pareto front the i-th solution belongs
            to.

        Raises
        ------
        TypeError
            If the fitness rows are scalar (single-objective problem)
            or of an unsupported type.
        """

        # Verify that the problem is multi-objective optimization as non-dominated sorting is only applied to multi-objective problems.
        if type(fitness[0]) in [list, tuple, numpy.ndarray]:
            pass
        elif type(fitness[0]) in self.supported_int_float_types:
            raise TypeError('Non-dominated sorting is only applied when optimizing multi-objective problems.\n\nBut a single-objective optimization problem found as the fitness function returns a single numeric value.\n\nTo use multi-objective optimization, consider returning an iterable of any of these data types:\n1)list\n2)tuple\n3)numpy.ndarray')
        else:
            raise TypeError(f'Non-dominated sorting is only applied when optimizing multi-objective problems. \n\nTo use multi-objective optimization, consider returning an iterable of any of these data types:\n1)list\n2)tuple\n3)numpy.ndarray\n\nBut the data type {type(fitness[0])} found.')

        # A list of all non-dominated sets.
        pareto_fronts = []
    
        # The remaining set to be explored for non-dominance.
        # Initially it is set to the entire population.
        # The solutions of each non-dominated set are removed after each iteration.
        remaining_set = fitness.copy()
    
        # Zipping the solution index with the solution's fitness.
        # This helps to easily identify the index of each solution.
        # Each element has:
            # 1) The index of the solution.
            # 2) An array of the fitness values of this solution across all objectives.
        remaining_set = list(zip(range(0, fitness.shape[0]), remaining_set))
    
        # A list mapping the index of each pareto front to the set of solutions in this front.
        solutions_fronts_indices = [-1]*len(remaining_set)
        solutions_fronts_indices = numpy.array(solutions_fronts_indices)
    
        # Index of the current pareto front.
        front_index = -1
        while len(remaining_set) > 0:
            front_index += 1

            # Get the current non-dominated set of solutions.
            remaining_set, pareto_front = self.get_non_dominated_set(curr_solutions=remaining_set)
            pareto_front = numpy.array(pareto_front, dtype=object)
            pareto_fronts.append(pareto_front)
    
            solutions_indices = pareto_front[:, 0].astype(int)
            solutions_fronts_indices[solutions_indices] = front_index

        return pareto_fronts, solutions_fronts_indices

    def crowding_distance(self, pareto_front, fitness):
        """
        Calculate the crowding distance for every solution in the given
        Pareto front. The crowding distance measures how isolated each
        solution is from its neighbours along every objective. Boundary
        solutions get a crowding distance of infinity.

        Parameters
        ----------
        pareto_front : numpy.ndarray
            The Pareto front returned by ``non_dominated_sorting``. Each
            row is a (population_index, fitness_vector) pair.
        fitness : numpy.ndarray
            Fitness of the entire population. Used to compute the per-
            objective range that normalises the crowding distance.

        Returns
        -------
        obj_crowding_dist_list : numpy.ndarray
            A nested array with the per-objective sorted lists used
            internally. Each entry is ``[front_index, objective_value,
            crowding_distance]``.
        crowding_dist_sum : list
            A list of ``[front_index, sum_of_crowding_distances]`` pairs
            sorted by the sum in descending order (highest first).
        crowding_dist_front_sorted_indices : numpy.ndarray
            Indices of the solutions inside ``pareto_front`` sorted by
            crowding distance (best first).
        crowding_dist_pop_sorted_indices : numpy.ndarray
            The same ordering but mapped back to the population indices,
            so the caller can use them directly against
            ``self.population``.
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
            # This variable has a nested list where each child list zips the following together:
                # 1) The index of the objective value.
                # 2) The objective value.
                # 3) Initialize the crowding distance by zero.
            obj = list(zip(range(len(obj)), obj, [0]*len(obj)))
            obj = [list(element) for element in obj]
            # This variable is the sorted version where sorting is done by the objective value (second element).
            # Note that the first element is still the original objective index before sorting.
            obj_sorted = sorted(obj, key=lambda x: x[1])
        
            # Get the minimum and maximum values for the current objective.
            obj_min_val = min(fitness[:, obj_idx])
            obj_max_val = max(fitness[:, obj_idx])
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
                obj_crowding_dist_list.append(obj_sorted)
                break
    
            for idx in range(1, len(obj_sorted)-1):
                # Calculate the crowding distance.
                crowding_dist = obj_sorted[idx+1][1] - obj_sorted[idx-1][1]
                crowding_dist = crowding_dist / denominator
                # Insert the crowding distance back into the list to override the initial zero.
                obj_sorted[idx][2] = crowding_dist
        
            # Sort the objective by the original index at index 0 of each child list.
            obj_sorted = sorted(obj_sorted, key=lambda x: x[0])
            obj_crowding_dist_list.append(obj_sorted)
    
        obj_crowding_dist_list = numpy.array(obj_crowding_dist_list)
        crowding_dist = numpy.array([obj_crowding_dist_list[idx, :, 2] for idx in range(len(obj_crowding_dist_list))])
        crowding_dist_sum = numpy.sum(crowding_dist, axis=0)
    
        # An array of the sum of crowding distances across all objectives. 
        # Each row has 2 elements:
            # 1) The index of the solution.
            # 2) The sum of all crowding distances for all objectives of the solution.
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

    def sort_solutions_nsga2(self,
                             fitness,
                             find_best_solution=False):
        """
        Sort the solutions by fitness and return their population
        indices in best-to-worst order.

        For single-objective problems the sort is a plain descending
        sort on the fitness value. For multi-objective problems the
        sort uses non-dominated sorting and then crowding distance
        inside each Pareto front; solutions in front X always come
        before solutions in front X+1.

        Parameters
        ----------
        fitness : numpy.ndarray
            Fitness of the entire population.
        find_best_solution : bool
            If True, the method is being called only to identify the
            best solution and ``self.pareto_fronts`` is left untouched.
            If False (the default), the method is being called as part
            of the GA lifecycle and ``self.pareto_fronts`` is updated to
            reflect the latest fronts.

        Returns
        -------
        solutions_sorted : list
            Population indices sorted from best to worst.

        Raises
        ------
        TypeError
            If a fitness row is neither a scalar nor a list / tuple /
            numpy array.
        """
        if type(fitness[0]) in [list, tuple, numpy.ndarray]:
            # Multi-objective optimization problem.
            solutions_sorted = []
            # Split the solutions into pareto fronts using non-dominated sorting.
            pareto_fronts, solutions_fronts_indices = self.non_dominated_sorting(fitness)
            if find_best_solution:
                # Do not edit the pareto_fronts instance attribute when just getting the best solution.
                pass
            else:
                # The method is called within the regular GA lifecycle.
                # We have to edit the pareto_fronts to be assigned the latest pareto front.
                self.pareto_fronts = pareto_fronts.copy()
            for pareto_front in pareto_fronts:
                # Sort the solutions in the front using crowded distance.
                _, _, _, crowding_dist_pop_sorted_indices = self.crowding_distance(pareto_front=pareto_front.copy(),
                                                                                   fitness=fitness)
                crowding_dist_pop_sorted_indices = list(crowding_dist_pop_sorted_indices)
                # Append the sorted solutions into the list.
                solutions_sorted.extend(crowding_dist_pop_sorted_indices)
        elif type(fitness[0]) in pygad.GA.supported_int_float_types:
            # Single-objective optimization problem.
            solutions_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
            # Reverse the sorted solutions so that the best solution comes first.
            solutions_sorted.reverse()
        else:
            raise TypeError(f'Each element in the fitness array must be of a number of an iterable (list, tuple, numpy.ndarray). But the type {type(fitness[0])} found')

        return solutions_sorted
