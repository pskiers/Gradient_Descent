def get_next_x(x, gradient, step):
    return x - gradient(x) * step


def gradient_descent(x, function, gradient, epsilon, step, step_change):
    history = [x]
    while True:
        next_x = get_next_x(x=x, gradient=gradient, step=step)
        if (abs(function(x) - function(next_x))) <= epsilon:
            ok = True
            for i in gradient(x):
                if abs(i) >= epsilon:
                    ok = False
                    break
            if ok:
                return x, history
        if function(x) <= function(next_x):
            step = step / step_change
            continue
        x = next_x
        history.append(x)
