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
    dominant_set.append(numpy.array(d1))

for i, s in enumerate(dominant_set):
    print(f'Dominant Front Set {i+1}:\n{s}')

print("\n\n\n--------------------")
def crowding_distance(front):
    # An empty list holding info about the objectives of each solution. The info includes the objective value and crowding distance.
    obj_crowding_dist_list = []
    # Loop through the objectives to calculate the crowding distance of each solution across all objectives.
    for obj_idx in range(front.shape[1]):
        obj = front[:, obj_idx]
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
        obj_min_val = min(population[:, obj_idx])
        obj_max_val = max(population[:, obj_idx])
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

        # If there are only 2 solutions in the current front, then do not proceed.
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

    return obj_crowding_dist_list, crowding_dist_sum

# Fetch the current front.
front = dominant_set[1]
obj_crowding_distance_list, crowding_distance_sum = crowding_distance(front)

print(obj_crowding_distance_list)
print(f'\nSum of Crowd Dists\n{crowding_distance_sum}')
