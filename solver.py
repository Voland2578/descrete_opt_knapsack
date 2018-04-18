#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from utils import *
import pdb
import bisect
Item = namedtuple("Item", ['index', 'value', 'weight'])
BFS = namedtuple("BFS",['path', 'value', 'remaining_capacity','best_possible'])

def updateBFS(bqueue, keys, item, max_best):
    insert_idx = bisect.bisect_left(keys, max_best)
    keys.insert(insert_idx, max_best)
    bqueue.insert(insert_idx, item)


def best_first_search(items, capacity):
    sorted_by_weight = sort_items( items, lambda x: -x.value / x.weight)
    (value, taken_array) =  __best_first_search (sorted_by_weight, capacity)
    # transform back to the original indices
    tranformed_to_original_order = transform_sorted_array(taken_array, sorted_by_weight)
    return (tranformed_to_original_order,value, 0)

def __best_first_search(items, capacity):

    # create starting tuple
    best_value = -34
    best_solution = None
    bqueue = [ BFS( [],0, capacity,0)];  keys = [0]
    qLen = 1

    while len(bqueue) > 0:
        n = bqueue.pop()
        keys.pop()
        #print(n)
        node_best_possible = n.best_possible
        # what if you have already reached the last element. No more expansion
        if len(n.path) == len(items):
            continue

        if node_best_possible <= best_value:
            continue
        # index to expand is the search level to expand
        # thus, if the path of the current node is [1,1,0,0], index to expand would be 4
        index_to_expand,path, remaining_capacity,p_value = len(n.path), n.path, n.remaining_capacity,n.value

        # expand means you will trying to put 0 or 1 at that index
        item = items[index_to_expand]

        exp_max = -1
        if item.weight <= remaining_capacity:
            chosen_item_value = p_value+item.value
            # maximum value if the item has been chosen
            exp_max = chosen_item_value + max_value(items, index_to_expand + 1, remaining_capacity-item.weight)
            # if choosing the item has potential to beat the best value, let's explore
            if exp_max > best_value:
                next_path = path.copy()
                next_path = next_path + [1]
                # if the chosen node value is higher than known best, remember the bests
                if chosen_item_value > best_value:
                    best_value = chosen_item_value
                    best_solution = next_path
                # create expanded object
                take_obj = BFS ( next_path, chosen_item_value, remaining_capacity-item.weight,exp_max)
                updateBFS(bqueue,keys, take_obj, exp_max)

        # not exp
        nexp_max = p_value + max_value(items, index_to_expand+1, capacity)
        if nexp_max > best_value:
            next_path = path.copy();
            next_path = next_path + [0]
            not_take_obj= BFS(next_path, p_value, remaining_capacity,nexp_max)
            updateBFS(bqueue, keys, not_take_obj, nexp_max)


        if len(bqueue) > 1 and (best_value > bqueue[0].best_possible):
            #idx = bisect.bisect_left(keys, best_value)
            #print(bqueue)
            #print("Dropping {} from queue best{}. qL: {}. idx: {}".format(idx, best_value, len(bqueue), idx))

            #keys = keys[idx:]
            #bqueue = bqueue[idx:]
            #print(bqueue)
            pass

    best_solution = best_solution + [0] * ( len(items) - len(best_solution))
    #pdb.set_trace()

    return (best_value,best_solution)
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



def greedy_relaxation (items, capacity):
    taken = [0] * len(items)
    # presort by v/w
    sorted_by_weight = sort_items( items, lambda x: -x.value / x.weight)
    (value, taken_array) =  greedy_relaxation_step (sorted_by_weight, capacity, taken, 0, taken, 0, 0)
    # transform back to the original indices
    tranformed_to_original_order = transform_sorted_array(taken_array, sorted_by_weight)
    return (tranformed_to_original_order,value, 0)


def greedy_relaxation_step(items, capacity, taken_array, current_value, best_array, best_value, item_idx):

    # reached the end of processing
    if item_idx == len(items):
        return best_value, best_array

    # what happens if we don't take the item
    not_taken_max_value = current_value + max_value(items, item_idx+1, capacity)
    # item data
    item_weight, item_value = items[item_idx].weight, items[item_idx].value
    
    # examine the item only if it feasible to take it
    taken_max_value = -1
    if item_weight <= capacity:
        taken_max_value = current_value + max_value(items, item_idx, capacity)
    
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

def dynamic_programming_non_recursive(items, capacity):
    # initialize array of elements by capacity+1. to include 0 capacity
    result = [[0] * (capacity+1) for i in range(0,len(items))]
    # https://stackoverflow.com/questions/17636567/python-initialize-multi-dimensional-list read for further understanding of copy by value

    for num_items in range(1, len(items)+1):
        item_idx = num_items - 1
        weight, value = items[item_idx].weight, items[num_items-1].value
        for c in range(0, capacity+1):
            take_item = -1
            # choose not to take the item
            not_take_item = result[item_idx-1][c]

            # choose to take the item
            if weight<=c:
                # value of the item and the best value of capacity-weight for 1 fever item
                take_item = value + result[item_idx-1][c-weight]

            result[item_idx][c] = max(take_item, not_take_item)

    #board=[[result[y][x] for y in range(0, len(items))] for x in range(0, capacity + 1)]
    #print(*board, sep='\n')
    taken_array, value = parse_dyn_board(result,items, capacity,dyn_double_array_board_parse)
    return taken_array, value, 1


def dyn_recursive(items, capacity):
    memory = {}

    result = __dyn_recursive_inner(items, capacity, memory, len(items)-1)
    #pdb.set_trace()
    taken_array, value = parse_dyn_board (memory, items, capacity, dyn_memoise_board_parse)
    #pdb.set_trace()

    return taken_array,value,1

def __dyn_recursive_inner(items, capacity, memory, idx):
    if idx == 0:
        return 0
    key = "{}_{}".format(idx, capacity)
    val =  memory.get(key,-1)
    if val != -1:
        return val
    else:
        item = items[idx]
        not_taken = __dyn_recursive_inner(items, capacity, memory, idx-1)
        if item.weight <= capacity:
            taken = item.value + __dyn_recursive_inner(items, capacity - item.weight, memory, idx-1 )
            val = max(taken, not_taken)
        else:
            val = not_taken

        memory[key] = val
        return val


def solve_it(file_location, input_data, algorithm=None):
    # Modify this code to run your optimization algorithm
    print (algorithm)
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

    if algorithm is not None:
        taken, value, optimal = algorithm(items, capacity)
    else:
        # use dumb greedy algorithm
        # (taken, value, optimal) = dumb_greedy(items, capacity)
        #( taken,value,optimal) = greedy_relaxation (items, capacity)

        #4 has huge memory requirements as capacity is gigantic

        #(taken,value, optimal) = dynamic_programming_non_recursive(items, capacity)
        (taken,value, optimal) = best_first_search(items, capacity)
        #(taken, value, optimal) = dyn_recursive(items, capacity)

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
        print(solve_it(file_location, input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
