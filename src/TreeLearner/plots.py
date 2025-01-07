import matplotlib.pyplot as plt

'''I want to make few plots here:
1. Y-axis: valid_error, dist_error, X-axis: range(len(tree_sequence)), a simple line graph'''
def prepare_plots(tree_sequence, iteration, path, topk):
    valid_error = []
    dist_error = []
    for tree, error_bounds in tree_sequence.items():
        valid_error.append(error_bounds['valid_error'])
        dist_error.append(error_bounds['dist_error'])
    plt.plot(range(len(tree_sequence)), valid_error, label='valid_error')
    plt.plot(range(len(tree_sequence)), dist_error, label='dist_error')
    plt.xlabel('Chain of trees')
    plt.ylabel('Error')
    plt.legend()
    plt.title(f'Error vs Chain of trees')
    plt.savefig(f'{path}/plots/ex8_{iteration}_{topk}.png')
    plt.clf()
    plt.close()