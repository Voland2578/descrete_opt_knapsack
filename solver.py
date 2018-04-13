#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import pdb

Item = namedtuple("Item", ['index', 'value', 'weight'])


# takes items in order until knapsack is completely full
def dumb_greedy(items, capacity):
    optimal = 0

    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0] * len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return (taken, value, optimal)

def __sort_items(itemsList, fun):
    return sorted(itemsList, key=fun)

# Assumes that the items is sorted by value/weight
def __max_value(items, start_idx, capacity):
    remaining_capacity = capacity
    max_value = 0
    for x in items[start_idx:]:
        if x.weight <= remaining_capacity:
            remaining_capacity = remaining_capacity - x.weight
            max_value += x.value
        else:
            max_value += x.value * (remaining_capacity / x.weight)
            break
    return max_value
    
def __attribute_function ( items, attribute, default_if_not_chosen=None):
    def r(item):
        value = getattr(items[item[0]], attribute)
        if default_if_not_chosen is not None:
            is_chosen = item[1]
            return default_if_not_chosen if is_chosen == 0 else value
        return value
    return r

def transform_sorted_array(sorted_taken, sorted_items):
    result = [0] * len(sorted_items)
    true_indices = [x for x in map(__attribute_function(sorted_items,'index'),enumerate(sorted_taken) )]
    for (idx, element) in enumerate(true_indices):
        result[element] = sorted_taken[idx]
    return result
    

def greedy_relaxation (items, capacity):
    taken = [0] * len(items)
    sorted_by_weight = __sort_items( items, lambda x: -x.value / x.weight)
    (value, taken_array) =  greedy_relaxation_step (sorted_by_weight, capacity, taken, 0, taken, 0, 0)
    tranformed_to_original_order = transform_sorted_array(taken_array, sorted_by_weight)
    return (tranformed_to_original_order,value, 0)

def printCurrentState(items, taken_array, full_capacity):
    filled_capacity = sum( map (__attribute_function(items,'weight'),enumerate(taken_array)))
    filled_value = sum( map (__attribute_function(items,'value'),enumerate(taken_array)))
    output = "{} Filled Capacity: {}, Filled Value: {}".format(taken_array, filled_capacity, filled_value)
    return output
    

def greedy_relaxation_step(items, capacity, taken_array, current_value, best_array, best_value, item_idx):

    # reached the end of processing
    if item_idx == len(items):
        return best_value, best_array

    # what happens if we don't take the item
    not_taken_max_value = current_value + __max_value(items, item_idx+1, capacity)
    # item data
    item_weight, item_value = items[item_idx].weight, items[item_idx].value
    
    # examine the item only if it feasible to take it
    taken_max_value = -1
    if (item_weight <= capacity):
        taken_max_value = current_value + __max_value(items, item_idx, capacity)
    
    # we should take the element only if the max theoretical value is better than currently observed best_array
    if taken_max_value > best_value:
        element_taken = taken_array.copy()
        element_taken[item_idx] = 1
        if current_value + item_value > best_value:
            best_value, best_array = current_value + item_value, element_taken
        # percolate data back update
        (best_value, best_array) =  greedy_relaxation_step(items, capacity - item_weight,
                                                                                element_taken,
                                                                                current_value + item_value,
                                                                                best_array, best_value,
                                                                                item_idx + 1)
    if not_taken_max_value > best_value:
        not_taken = taken_array.copy()
        not_taken[item_idx] = 0
        best_value, best_array =  greedy_relaxation_step(items, capacity, not_taken,
                                                                                current_value ,
                                                                                best_array, best_value,
                                                                                item_idx + 1)
    
    return best_value, best_array

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    # use dumb greedy algorithm
    # (taken, value, optimal) = dumb_greedy(items, capacity)
    ( taken,value,optimal) = greedy_relaxation (items, capacity)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
