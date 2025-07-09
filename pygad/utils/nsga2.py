import numpy
import pygad

class NSGA2:

    def __init__(self):
        pass

    def get_non_dominated_set(self, curr_solutions):
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
        Apply non-dominant sorting over the fitness to create the pareto fronts based on non-dominated sorting of the solutions.
    
        Parameters
        ----------
        fitness : TYPE
            An array of the population fitness across all objective function.
    
        Returns
        -------
        pareto_fronts : TYPE
            An array of the pareto fronts.

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
        Calculate the crowding distance for all solutions in the current pareto front.
    
        Parameters
        ----------
        pareto_front : TYPE
            The set of solutions in the current pareto front.
        fitness : TYPE
            The fitness of the current population.
    
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

    def sort_solutions_nsga2(self,
                             fitness,
                             find_best_solution=False):
        """
        Sort the solutions based on the fitness.
        The sorting procedure differs based on whether the problem is single-objective or multi-objective optimization.
        If it is multi-objective, then non-dominated sorting and crowding distance are applied.
        At first, non-dominated sorting is applied to classify the solutions into pareto fronts.
        Then the solutions inside each front are sorted using crowded distance.
        The solutions inside pareto front X always come before those in front X+1.
    
        Parameters
        ----------
        fitness: The fitness of the entire population.
        find_best_solution: Whether the method is called only to find the best solution or as part of the PyGAD lifecycle. This is to decide whether the pareto_fronts instance attribute is edited or not.

        Returns
        -------
        solutions_sorted : TYPE
            The indices of the sorted solutions.
    
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
