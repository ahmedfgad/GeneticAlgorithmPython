"""
Shared building blocks for the NSGA family of selection methods.

The methods here are used by both NSGA-II and NSGA-III. They live in
this module so neither algorithm has to depend on the other.
"""

import operator

import numpy


class NSGA:

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
        def _fitness_rows(solutions):
            # Return the fitness vectors as plain Python lists when they
            # are all 1D numeric numpy arrays of the same dtype and length.
            # Otherwise return None so the generic loop below is used.
            if type(solutions) is not list or len(solutions) < 2:
                return None
            head = solutions[0]
            first = head[1] if type(head) is tuple and len(head) == 2 else None
            if (type(first) is not numpy.ndarray or first.ndim != 1
                    or first.size == 0 or first.dtype.kind not in "biuf"):
                return None
            rows = []
            for sol in solutions:
                if type(sol) is not tuple or len(sol) != 2:
                    return None
                vector = sol[1]
                if (type(vector) is not numpy.ndarray
                        or vector.dtype != first.dtype
                        or vector.shape != first.shape):
                    return None
                rows.append(vector.tolist())
            return rows

        dominated_set = []
        non_dominated_set = []

        rows = _fitness_rows(curr_solutions)
        if rows is not None:
            # The vectors are tiny, so plain Python comparisons are cheaper
            # than numpy calls. PyGAD maximizes, so "other" dominates "row"
            # when all values are >= and the two rows are not equal.
            for sol, row in zip(curr_solutions, rows):
                for other in rows:
                    if other != row and all(map(operator.ge, other, row)):
                        dominated_set.append(sol)
                        break
                else:
                    non_dominated_set.append(sol)
            return dominated_set, non_dominated_set

        for idx1, sol1 in enumerate(curr_solutions):
            is_not_dominated = True
            for idx2, sol2 in enumerate(curr_solutions):
                if idx1 == idx2:
                    continue
                two_solutions = numpy.array(list(zip(sol1[1], sol2[1])))
                # PyGAD maximizes, so domination uses >= and >.
                greater_or_equal = two_solutions[:, 1] >= two_solutions[:, 0]
                strictly_greater = two_solutions[:, 1] > two_solutions[:, 0]
                if greater_or_equal.all() and strictly_greater.any():
                    is_not_dominated = False
                    dominated_set.append(sol1)
                    break
            if is_not_dominated:
                non_dominated_set.append(sol1)
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
        if type(fitness[0]) in [list, tuple, numpy.ndarray]:
            pass
        elif type(fitness[0]) in self.supported_int_float_types:
            raise TypeError(
                "Non-dominated sorting is only applied when optimizing "
                "multi-objective problems.\n\n"
                "But a single-objective optimization problem found as the "
                "fitness function returns a single numeric value.\n\n"
                "To use multi-objective optimization, consider returning an "
                "iterable of any of these data types:\n"
                "1)list\n2)tuple\n3)numpy.ndarray")
        else:
            raise TypeError(
                f"Non-dominated sorting is only applied when optimizing "
                f"multi-objective problems. \n\nTo use multi-objective "
                f"optimization, consider returning an iterable of any of "
                f"these data types:\n1)list\n2)tuple\n3)numpy.ndarray\n\n"
                f"But the data type {type(fitness[0])} found.")

        pareto_fronts = []

        remaining_set = fitness.copy()
        # Pair every solution with its population index so we can keep
        # track of who is where as we peel off fronts.
        remaining_set = list(zip(range(0, fitness.shape[0]), remaining_set))

        solutions_fronts_indices = [-1] * len(remaining_set)
        solutions_fronts_indices = numpy.array(solutions_fronts_indices)

        front_index = -1
        while len(remaining_set) > 0:
            front_index += 1
            remaining_set, pareto_front = self.get_non_dominated_set(
                curr_solutions=remaining_set)
            pareto_front = numpy.array(pareto_front, dtype=object)
            pareto_fronts.append(pareto_front)
            solutions_indices = pareto_front[:, 0].astype(int)
            solutions_fronts_indices[solutions_indices] = front_index

        return pareto_fronts, solutions_fronts_indices
