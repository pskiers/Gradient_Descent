from descent import gradient_descent
from math import cos, pi, sin
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
    pass


FUNC_N_GRAD = {f: gradient_f, g: gradient_g}
EPSILONS = {f: 0.001, g: 0.001}
STEPS = {f: 1, g: 0.01}
STEP_CHANGES = {f: 1.5, g: 1.5}
STARTS = {f: [5, 5], g: [0.51]}

if __name__ == "__main__":
    for func in FUNC_N_GRAD.keys():
        minimum = gradient_descent(x=STARTS[func],
                                   function=func,
                                   gradient=FUNC_N_GRAD[func],
                                   epsilon=EPSILONS[func],
                                   step=STEPS[func],
                                   step_change=STEP_CHANGES[func])
        print("Function minimum in: ", minimum, '\n')
