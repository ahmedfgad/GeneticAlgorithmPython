import numpy

population = numpy.array([[20, 2.2],
                          [60, 4.4],
                          [65, 3.5],
                          [15, 4.4],
                          [55, 4.5],
                          [50, 1.8],
                          [80, 4.0],
                          [25, 4.6]])

def non_dominant_sorting(curr_set):
    # List of the members of the current dominant front/set.
    dominant_set = []
    # List of the non-members of the current dominant front/set.
    non_dominant_set = []
    for idx1, sol1 in enumerate(curr_set):
        # Flag indicates whether the solution is a member of the current dominant set.
        is_dominant = True
        for idx2, sol2 in enumerate(curr_set):
            if idx1 == idx2:
                continue
            # Zipping the 2 solutions so the corresponding genes are in the same list.
            # The returned array is of size (N, 2) where N is the number of genes.
            b = numpy.array(list(zip(sol1, sol2)))
    
            #TODO Consider repacing < by > for maximization problems.
            # Checking for if any solution dominates the current solution by applying the 2 conditions.
            # le_eq: All elements must be True.
            # le: Only 1 element must be True.
            le_eq = b[:, 1] <= b[:, 0]
            le = b[:, 1] < b[:, 0]

            # If the 2 conditions hold, then a solution dominates the current solution.
            # The current solution is not considered a member of the dominant set.
            if le_eq.all() and le.any():
                # print(f"{sol2} dominates {sol1}")
                # Set the is_dominant flag to False to not insert the current solution in the current dominant set.
                # Instead, insert it into the non-dominant set.
                is_dominant = False
                non_dominant_set.append(sol1)
                break
            else:
                # Reaching here means the solution does not dominant the current solution.
                # print(f"{sol2} does not dominate {sol1}")
                pass

        # If the flag is True, then no solution dominates the current solution.
        if is_dominant:
            dominant_set.append(sol1)

    # Return the dominant and non-dominant sets.
    return dominant_set, non_dominant_set

dominant_set = []
non_dominant_set = population.copy()
while len(non_dominant_set) > 0:
    d1, non_dominant_set = non_dominant_sorting(non_dominant_set)
    dominant_set.append(d1)

for i, s in enumerate(dominant_set):
    print(f'Dominant Front Set {i+1}:\n{s}')
