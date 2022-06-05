import numpy as np
import cvxpy as cp
import time
from itertools import product


def sample_vars() -> tuple:
    h1 = np.random.normal(loc=6000, scale=100 ** 0.5)
    while h1 < 5700 or h1 > 6300:
        h1 = np.random.normal(loc=6000, scale=100 ** 0.5)
    h2 = np.random.normal(loc=4000, scale=50 ** 0.5)
    while h2 < 3850 or h2 > 4150:
        h2 = np.random.normal(loc=4000, scale=50 ** 0.5)
    t1 = [np.random.uniform(3.5, 4.5), np.random.uniform(8, 10), np.random.uniform(6, 8), np.random.uniform(9, 11)]
    t2 = [np.random.uniform(0.8, 1.2), np.random.uniform(0.8, 1.2), np.random.uniform(2.5, 3.5),
          np.random.uniform(36, 44)]
    t1 = np.array(t1)
    t2 = np.array(t2)
    h = np.array([h1, h2])
    return np.concatenate([t1, t2, h])


def get_b_and_B():
    b = np.array([0.5, 1, 1, 1, 0.2, 0.2, 0.5, 4, 300, 150])
    B = np.diag(b)
    return b, B


def mean_of_vars() -> tuple:
    c = np.array([12, 20, 18, 40])
    q = np.array([2, 2])
    h = np.array([6000, 4000])
    t = np.array([
        [4, 9, 7, 10],
        [1, 1, 3, 40]
    ])
    return c, q, h, t


def solve_non_robust(c, q, h, t):
    x = cp.Variable(4)
    v = cp.Variable(2)
    t1, t2 = t[0, :], t[1, :]
    h1, h2 = h[0], h[1]
    constraints = [t1 @ x <= h1 + v[0],
                   t2 @ x <= h2 + v[1],
                   x >= 0,
                   v >= 0,
                   v <= 600]
    objective_f = c @ x - q @ v
    obj = cp.Maximize(objective_f)
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return prob.value, x.value, v.value


def solve_robust(r=1000):
    c = np.array([12, 20, 18, 40])
    q = np.array([2, 2])
    t1_max = np.array([4.5, 10, 8, 11])
    t2_max = np.array([1.2, 1.2, 3.5, 44])
    h1_min, h2_min = 5700, 3850
    b, B = get_b_and_B()
    B_inv = np.diag(1 / b)
    xi_0 = np.array([4, 9, 7, 10, 1, 1, 3, 40, 6000, 4000])

    x = cp.Variable(4)
    y1 = cp.Variable(31)
    y2 = cp.Variable(31)
    inside_max = [0, q[0] * (t1_max @ x - h1_min), q[1] * (t2_max @ x - h2_min),
                  q[0] * (t1_max @ x - h1_min) + q[1] * (t2_max @ x - h2_min)]
    obj_func = cp.max(cp.hstack(inside_max)) - c @ x

    G = np.vstack([np.eye(10), -np.eye(10), B_inv, np.zeros(shape=[1, 10])])
    h = np.hstack([-xi_0 + b, xi_0 + b, -B_inv @ xi_0, r])
    a1 = np.array([1] * 4 + [0] * 4 + [1, 0])
    A1 = np.diag(a1)
    A2 = np.diag(np.ones(10) - a1)
    temp_matrix = np.vstack([np.eye(4), np.eye(4), np.zeros(shape=[2, 4])])
    x_tilde = temp_matrix @ x - np.hstack([np.zeros(8), np.ones(2)])

    constraints = [x >= 0,
                   h @ y1 >= -600, h @ y2 >= -600,
                   G.T @ y1 == A1.T @ x_tilde,
                   G.T @ y2 == A2.T @ x_tilde,
                   y1[:20] <= 0, y2[:20] <= 0,
                   cp.norm2(y1[20:30]) <= -y1[30], cp.norm2(y2[20:30]) <= -y2[30]
                   ]
    obj = cp.Minimize(obj_func)
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return -prob.value, x.value


def actual_value_for_x(x, c, q, t, h):
    t1, t2 = t[0, :], t[1, :]
    h1, h2 = h[0], h[1]
    inside_max = [0, q[0] * (t1 @ x - h1), q[1] * (t2 @ x - h2),
                  q[0] * (t1 @ x - h1) + q[1] * (t2 @ x - h2)]
    obj_func = np.max(np.hstack(inside_max)) - c @ x
    return -obj_func


def matrix_inner_product(A, B):
    return cp.trace(A @ B)


def solve_1(mu, Sigma, r):
    L = 4
    beta = cp.Variable(10)
    Lambda = cp.Variable((10, 10))
    gamma = cp.Variable(1)
    y1 = cp.Variable(31)
    y2 = cp.Variable(31)
    phi = [cp.Variable((10, 1)) for l in range(L)]
    varphi = cp.Variable((10, L))
    tau = cp.Variable(L)
    x = cp.Variable(4)
    theta = [cp.Variable((1, 1)) for l in range(L)]
    x_tilde = np.eye(5, 4) @ x + np.array([0] * 4 + [1])
    q1, q2 = 2, 2

    b, B = get_b_and_B()
    c = np.array([12, 20, 18, 40])
    A1 = np.diag([1] * 4 + [0] * 4 + [1, 0])
    A2 = np.eye(10) - A1
    G = np.vstack([np.eye(10), -np.eye(10), np.diag(1 / b), np.zeros([1, 10])])
    x_hat = np.vstack([np.eye(4), np.eye(4), np.zeros([2, 4])]) @ x - np.hstack([np.zeros(8), np.ones(2)])
    h = np.concatenate([-mu + b, mu + b, np.diag(1 / b) @ mu, np.array([r])])

    constraints = [x >= 0,
                   h @ y1 >= -600, h @ y2 >= -600,
                   G.T @ y1 == A1.T @ x_hat,
                   G.T @ y2 == A2.T @ x_hat,
                   y1[:20] <= 0, y2[:20] <= 0,
                   cp.norm2(y1[20:30]) <= -y1[30], cp.norm2(y2[20:30]) <= -y2[30]
                   ]
    A1_ = q1 * np.block([[np.eye(4), np.zeros([4, 6])], [np.zeros(8), -1, 0]])
    A2_ = q2 * np.block([[np.zeros([4, 4]), np.eye(4), np.zeros([4, 2])], [np.zeros(9), -1]])
    A3_ = A1_ + A2_
    A4_ = np.zeros([5, 10])
    A = [A1_, A2_, A3_, A4_]
    for l in range(L):
        consts = [r * tau[l] - 2 * phi[l].T @ mu + theta[l] <= gamma,  # TODO: MAYBE ADD a0 @ x_hat ? ?????
                  2 * cp.reshape(phi[l], shape=(10,)) + 2 * varphi[:, l] - beta + A[l].T @ x_tilde + c @ x == 0,
                  # TODO: c@x???
                  cp.bmat([[Lambda, phi[l]], [phi[l].T, theta[l]]]) >= 0,
                  cp.norm2(varphi[:, l]) <= tau[l]
                  ]
        constraints += consts

    objective = cp.Minimize(mu @ beta + matrix_inner_product(Lambda, Sigma) + gamma - c @ x)
    prob = cp.Problem(objective=objective, constraints=constraints)
    prob.solve()
    return prob.value, x.value


def solve_2(mu, t, r, g_vecs):
    L = 4
    x = cp.Variable(4)
    beta = cp.Variable(10)
    lambda_vec = cp.Variable(2 ** 10 - 1)
    gamma = cp.Variable(1)
    y1 = cp.Variable(31)
    y2 = cp.Variable(31)
    theta_1 = [cp.Variable(10) for l in range(L)]
    theta_2 = [cp.Variable(10) for l in range(L)]
    x_tilde = np.eye(5, 4) @ x + np.array([0] * 4 + [1])
    q1, q2 = 2, 2

    mathcal_G = np.array(g_vecs).T
    b, B = get_b_and_B()
    c = np.array([12, 20, 18, 40])
    A1 = np.diag([1] * 4 + [0] * 4 + [1, 0])
    A2 = np.eye(10) - A1
    G = np.vstack([np.eye(10), -np.eye(10), np.diag(1 / b), np.zeros([1, 10])])
    x_hat = np.vstack([np.eye(4), np.eye(4), np.zeros([2, 4])]) @ x - np.hstack([np.zeros(8), np.ones(2)])
    h = np.concatenate([-mu + b, mu + b, np.diag(1 / b) @ mu, np.array([r])])

    constraints = [x >= 0,
                   h @ y1 >= -600, h @ y2 >= -600,
                   G.T @ y1 == A1.T @ x_hat,
                   G.T @ y2 == A2.T @ x_hat,
                   y1[:20] <= 0, y2[:20] <= 0,
                   cp.norm2(y1[20:30]) <= -y1[30], cp.norm2(y2[20:30]) <= -y2[30],
                   lambda_vec >= 0
                   ]
    A1_ = q1 * np.block([[np.eye(4), np.zeros([4, 6])], [np.zeros(8), -1, 0]])
    A2_ = q2 * np.block([[np.zeros([4, 4]), np.eye(4), np.zeros([4, 2])], [np.zeros(9), -1]])
    A3_ = A1_ + A2_
    A4_ = np.zeros([5, 10])
    A = [A1_, A2_, A3_, A4_]
    for l in range(L):
        consts = [(mathcal_G @ lambda_vec - theta_1[l] + theta_2[l]) @ mu - (theta_1[l] + theta_2[l]) @ b <= gamma,
                  A[l].T @ x_tilde - beta - mathcal_G @ lambda_vec + theta_1[l] - theta_2[l] == 0,
                  theta_1[l] <= 0,
                  theta_2[l] <= 0
                  ]
        constraints += consts

    objective = cp.Minimize(mu @ beta + lambda_vec @ t + gamma - c @ x)
    prob = cp.Problem(objective=objective, constraints=constraints)
    prob.solve()
    return prob.value, x.value


if __name__ == '__main__':
    gamma1, gamma2, r = 10 ** 9, 10, 10
    N_vals = [10, 100, 1000, 10000]

    c = np.array([12, 20, 18, 40])
    q = np.array([2, 2])
    temp = product(range(2), repeat=10)
    g = [np.array(t) for t in temp if sum(t) > 0]
    # g.discard(np.array([0]*10))

    test_set = [sample_vars() for i in range(10 ** 4)]

    for N in N_vals:
        train_set = []
        for i in range(N):
            train_set.append(sample_vars())
        train_set = np.array(train_set)
        mu = np.mean(train_set, axis=0)
        Sigma = np.cov(train_set, rowvar=False)
        t = [np.mean([(xi - mu) @ gi for xi in train_set]) for gi in g]
        # print(solve_1(mu, Sigma, r))
        print(solve_2(mu, t, r, g))
