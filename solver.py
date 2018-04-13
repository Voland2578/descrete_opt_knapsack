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


def exhaustive_greedy_no_relaxation(items, capacity):
    taken = [0] * len(items)
    (taken_array, value, capacity) =  exhaustive_greedy_step(items, capacity, taken, 0, 0)
    return (taken_array,value, 0)

# remaining capacity to fill
# items - all items array
# taken - currently taken array
# idx - we need idx of the item currently being explored
# return ( new taken array and filled capacity)
def exhaustive_greedy_step(items, capacity, taken, current_value, item_idx):
    # need to break
    # if we do not take the value
    not_taken = taken.copy()
    not_taken[item_idx] = 0
    (processed_not_taken, not_taken_value, not_taken_capacity) = exhaustive_greedy_step(items, capacity, not_taken,
                                                                                        current_value, item_idx + 1)

    # item information
    (item_weight, item_value) = (items[item_idx].weight, items[item_idx].value)
    if item_weight > capacity:
        return processed_not_taken, not_taken_value, not_taken_capacity
    else:
        element_taken = taken.copy()
        element_taken[item_idx] = 1
        (processed_taken, taken_value, taken_capacity) = exhaustive_greedy_step(items, capacity - item_weight,
                                                                                element_taken,
                                                                                current_value + item_value,
                                                                                item_idx + 1)

        if taken_value > not_taken_value:
            return processed_taken, taken_value, taken_capacity
        return processed_not_taken, not_taken_value, not_taken_capacity


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
    (taken, value, optimal) = exhaustive_greedy_no_relaxation(items, capacity)

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
