import solver
import submit
def solve_it(input_file, input_data):
    algo = None
    if "ks_400" in input_file:
        algo = solver.best_first_search
    elif "ks_1000" in input_file:
        algo = solver.best_first_search
    else:
        algo = solver.dyn_recursive
    return solver.solve_it(input_file,input_data, algo)
