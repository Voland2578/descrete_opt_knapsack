import sys
# return the index and the value
# of the maximum number < value passed in
# if nothing is found, return -1
def searchMinNumber(l, val):
    start_idx, end_idx = 0, len(l) - 1
    curr_idx, curr_val = -1, -1

    while True:
        # check for the end of the list
        if val > l[end_idx]:
            return end_idx, l[end_idx]

        length = end_idx - start_idx + 1
        # list of 1
        if start_idx == end_idx:
            if val > l[start_idx]:
                return start_idx, l[start_idx]
            else:
                return curr_idx, curr_val

        split_in_half = start_idx + math.floor(length / 2)
        # print("st: {}, end: {}, split_idx: {}, split_val: {} curr_idx: {} curr_v: {}".format(start_idx, end_idx, split_in_half, l[split_in_half], curr_idx, curr_val))

        # value higher than largest midway through the list
        # advance forward
        if val > l[split_in_half]:
            curr_idx, curr_val = split_in_half, l[split_in_half]
            start_idx = split_in_half + 1
        # value smaller than the largest one. Move back
        else:
            end_idx = split_in_half - 1



def parse_dyn_board(board, items, capacity, get_value_func):
    # unwrap the board
    row, col = len(items) - 1, capacity
    # maximum value
    max_value = get_value_func(board, row, col)
    taken_array = []
    while len(taken_array) < len(items):
        # if the adjacent value ( same capacity, fewer items) is the same
        # it means this item was not selected
        if get_value_func(board,row,col) > get_value_func(board,row - 1,col):
            # item was taken
            taken_array = [1] + taken_array
            row, col = row - 1, col - items[row].weight
        else:
            # location of previous entry if the item has been taken
            taken_array = [0] + taken_array
            row, col = row - 1, col
    return taken_array, max_value

def dyn_double_array_board_parse(board, row, col):
    return board[row][col]
def dyn_memoise_board_parse(board, row, col):
    return board.get( "{}_{}".format(row, col), sys.maxsize)



# Print current state
def printCurrentState(items, taken_array, full_capacity):
    filled_capacity = sum( map (attribute_function(items,'weight'),enumerate(taken_array)))
    filled_value = sum( map (attribute_function(items,'value'),enumerate(taken_array)))
    output = "{} Filled Capacity: {}, Filled Value: {}".format(taken_array, filled_capacity, filled_value)
    return output

# sort items by sort function
def sort_items(itemsList, fun):
    return sorted(itemsList, key=fun)

# Assumes that the items is sorted by value/weight
def max_value(items, start_idx, capacity):
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


# Picks an attribute from the Item object
# if default_if_not_chosen is provided, the function will check is the item was selected
#    * if it is selected, the value is returned, default otherwise
def attribute_function(items, attribute, default_if_not_chosen=None):
    def r(item):
        value = getattr(items[item[0]], attribute)
        if default_if_not_chosen is not None:
            is_chosen = item[1]
            return default_if_not_chosen if is_chosen == 0 else value
        return value

    return r


# input:  Takes list of items and correspond taken array
# output: taken array that is alligned with the original item indices.
# Specifically:
#    For each item, checks its original index in the Item object and populates a corresponding entry in the output array
def transform_sorted_array(sorted_taken, sorted_items):
    result = [0] * len(sorted_items)
    true_indices = [x for x in map(attribute_function(sorted_items, 'index'), enumerate(sorted_taken))]
    for (idx, element) in enumerate(true_indices):
        result[element] = sorted_taken[idx]
    return result
