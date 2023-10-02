import numpy as np
from LogisticRegression import LogisticRegression as Lr

X = np.array([
    (2, 1),
    (1, 3),
    (0, 4),
])
y = np.array([
    1,
    0,
    0,
])

lr = Lr(lr=0.2, n_iters=20)
lr.fit(X,y,print_output=True)
y_pred = lr.predict(X)
print('y_pred on X =', y_pred)
print('y_actual on X =', y)
