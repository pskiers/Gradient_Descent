from descent import gradient_descent
from math import cos, pi, sin
import numpy


# hardcoded function f(x)
def f(x):
    return (x[0] * x[0]) + (x[1] * x[1])


# hardcoded gradient of function f(x)
def gradient_f(x):
    numpy.array()
    return [2 * x[0], 2 * x[1]]


# hardcoded function g(x)
def g(x):
    return x * x - 10 * cos(2 * pi * x) + 10


# hardcoded gradient of function g(x)
def gradient_g(x):
    return 2 * x + 20 * pi * sin(2 * pi * x)


def main():
    pass


FUNC_N_GRAD = {f: gradient_f, g: gradient_f}
EPSILONS = {f: 0.001, g: 0.001}
STEPS = {f: 1, g: 1}
STEP_CHANGES = {f: 1.5, g: 1.5}
STARTS = {f: [5, 5], g: 65}

if __name__ == "__main__":
    for func in FUNC_N_GRAD.keys():
        minimum = gradient_descent(x=STARTS[func],
                                   function=func,
                                   gradient=FUNC_N_GRAD[func],
                                   epsilon=EPSILONS[func],
                                   step=STEPS[func],
                                   step_change=STEP_CHANGES[f])
        print("Function minimum in: ", minimum, '\n')
