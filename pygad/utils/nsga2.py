import numpy
import pygad


class NSGA2:

    def __init__(self):
        pass

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
            objective range that normalizes the crowding distance.

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
            obj_sorted[0][2] = inf_val
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
