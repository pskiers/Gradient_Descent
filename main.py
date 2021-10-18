from descent import gradient_descent
from math import cos, pi, sin
from plot import plot_history
import numpy


# hardcoded function f(x)
def f(x):
    return (x[0] * x[0]) + (x[1] * x[1])


# hardcoded gradient of function f(x)
def gradient_f(x):
    return numpy.array([2 * x[0], 2 * x[1]])


# hardcoded function g(x)
def g(x):
    return x[0] * x[0] - 10 * cos(2 * pi * x[0]) + 10


# hardcoded gradient of function g(x)
def gradient_g(x):
    return numpy.array([2 * x[0] + 20 * pi * sin(2 * pi * x[0])])


def main():
    starts = {}
    x1 = float(input('Enter starting x1 coordinates for f: '))
    x2 = float(input('Enter starting x2 coordinates for f: '))
    starts[f] = [x1, x2]
    starts[g] = [float(input('Enter starting point for g: '))]
    steps = {}
    steps[f] = float(input('Enter step for f: '))
    steps[g] = float(input('Enter step for g: '))
    for func in FUNC_N_GRAD.keys():
        minimum, history = gradient_descent(x=starts[func],
                                            function=func,
                                            gradient=FUNC_N_GRAD[func],
                                            epsilon=EPSILONS[func],
                                            step=steps[func],
                                            step_change=STEP_CHANGES[func])
        print("Function minimum in: ", minimum, '\n')
        plot_history(history, func)


FUNC_N_GRAD = {f: gradient_f, g: gradient_g}
EPSILONS = {f: 0.001, g: 0.001}
STEP_CHANGES = {f: 1.5, g: 1.5}


if __name__ == "__main__":
    main()
