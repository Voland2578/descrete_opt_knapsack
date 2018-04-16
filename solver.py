#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from utils import *
import pdb
import queue
from queue import PriorityQueue

Item = namedtuple("Item", ['index', 'value', 'weight'])
BFS = namedtuple("BFS",['path', 'value', 'remaining_capacity','best_possible'])

def best_first_search(items, capacity):
    # create starting tuple
    myq = queue.PriorityQueue()
    best_value = -34
    root = BFS( [],0, capacity,0)
    myq.put((0,root))
    while not myq.empty():
        best_possible, n = myq.get()
        print("Got: {} -> {}".format(n, best_possible))

        # what if you have already reached the last element. No more expansion
        if len(n.path) == len(items):
            continue

        if -best_possible <= best_value:
            continue

        index_to_expand,path, remaining_capacity,p_value = len(n.path), n.path, n.remaining_capacity,n.value
        # expand the item
        #print("Expanded idx: {} path: {} remaining: {}, current_val: {}".format(index_to_expand,path, remaining_capacity,p_value ))
        item = items[index_to_expand]

        exp_max = -1
        if item.weight <= remaining_capacity:
            exp = path.copy(); exp = exp + [1]
            item = items[index_to_expand]
            exp_max = p_value+item.value + max_value(items, index_to_expand + 1, remaining_capacity-item.weight)
            print("MAX: {}, IDX {}".format(exp_max,index_to_expand))
            # need parent path, parent value, parent_rem_capacity
            if exp_max > best_value:
                cur_value = p_value + item.value
                print("CUR: {} Best {}".format(cur_value, best_value))
                if cur_value > best_value:
                    best_value = cur_value
            take_obj = BFS ( exp, p_value + item.value, remaining_capacity-item.weight,-exp_max)
            print ("Adding+: {}".format(take_obj))
            myq.put( ( -exp_max, take_obj) )

        # not exp
        nexp = path.copy(); nexp = nexp + [0]
        nexp_max = max_value(items, index_to_expand+1, capacity)
        #print("Expanding idx {}. MaxT: {}. MaxNT: {}".format(index_to_expand, exp_max, -nexp_max),-nexp_max)

        if nexp_max > best_value:
            not_take_obj= BFS(nexp, p_value, remaining_capacity,-nexp_max)
            print ("Adding- {}".format(not_take_obj))
            myq.put( (-nexp_max, not_take_obj))

        print("Reval best: {}".format(best_value))
    '''
    # remove val. Deq
    idx, val = searchMinNumber(myq.queue, 10)
    myq.queue = myq.queue[idx + 1:]

    print(myq.queue)
    while not myq.empty():
        h = myq.get()
        print(h)
    '''
    pdb.set_trace()
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


    taken_array, value = parse_dyn_board(result,items, capacity)
    return taken_array, value, 1





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
    #( taken,value,optimal) = greedy_relaxation (items, capacity)

    #4 has huge memory requirements as capacity is gigantic

    #(taken,value, optimal) = dynamic_programming_non_recursive(items, capacity)
    (taken,value, optimal) = best_first_search(items, capacity)

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
