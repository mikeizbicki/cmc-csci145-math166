import math
import numpy as np

def sign(a):
    '''
    Convert a boolean value into +/- 1.

    >>> sign(12.5)
    1
    >>> sign(-12.5)
    -1
    '''
    if a >= 0:
        return 1
    else:
        return -1


def set_exp(xs, p):
    '''
    Compute the "set exponential" function.

    For efficiency, this function is a generator.
    This means that large sets will never be explicitly stored,
    and this function will always use O(1) memory.

    The doctests below first convert the generator into a list for visualization.

    >>> list(set_exp([-1, +1], 0))
    []
    >>> list(set_exp([-1, +1], 1))
    [[-1], [1]]
    >>> list(set_exp([-1, +1], 2))
    [[-1, -1], [1, -1], [-1, 1], [1, 1]]
    >>> list(set_exp([-1, +1], 3))
    [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]]

    Observe that the length grows exponentially with the power.

    >>> len(list(set_exp([-1, +1], 4)))
    16
    >>> len(list(set_exp([-1, +1], 5)))
    32
    >>> len(list(set_exp([-1, +1], 6)))
    64
    >>> len(list(set_exp([-1, +1], 7)))
    128
    >>> len(list(set_exp([-1, +1], 8)))
    256
    '''
    assert(len(xs) > 0)
    assert(p >= 0)
    assert(type(p) == int)
    if p == 1:
        for x in xs:
            yield [x]
    elif p > 1:
        for x in xs:
            for ys in set_exp(xs, p - 1):
                yield ys + [x]


from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=50000, test_size=10000, random_state=0
    )
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

X_train = (X_train / 128) - 1
X_test = (X_test / 128) - 1

y_train = np.where(y_train == 1, 1, -1)
y_test = np.where(y_test == 1, 1, -1)

print(f"X_train.shape={X_train.shape}")
print(f"y_train.shape={y_train.shape}")

d0 = X_train.shape[1]
d = 10
print(f"d={d}")

H_binary = (lambda x: i for i in [-1, 1])
H_axis = (lambda x: sign(x[i]) for i in range(d))
H_axis2 = (lambda x: sigma * sign(x[i]) for i in range(d) for sigma in [1, -1])
H_multiaxis2 = (lambda x: sign(sum([sigma[j] * sign(x[j]) for j in range(d)])) for sigma in set_exp([-1, 1], d))
H_multiaxis3 = (lambda x: sign(sum([sigma[j] * sign(x[j]) for j in range(d)])) for sigma in set_exp([-1, 0, 1], d))


class TEA:
    def __init__(self, H):
        self.H = H
        self.g = None

    def fit(self, X, y):
        g_score = -math.inf
        g = None
        for h in self.H:
            h_score = sum([h(x) == y for x, y in zip(X.to_numpy(), y.to_numpy())])
            if h_score > g_score:
                g = h
                g_score = h_score
        self.g = g

    def predict(self, x):
        assert(self.g is not None)
        return self.g(x)

