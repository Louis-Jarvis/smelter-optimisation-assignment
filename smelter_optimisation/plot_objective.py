import matplotlib.pyplot as plt


def plot_objective(solution):
    f, ax = plt.subplots()

    fx = solution.fx_hist
    x = [i for i in range(len(fx))]
    ax.plot(x, fx)

    ax.set(xlabel="Function Evaluations", ylabel="Objective Function Value f(x)")
