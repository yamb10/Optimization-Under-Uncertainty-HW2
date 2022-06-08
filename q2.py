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
    """
    From HW 1
    :param r:
    :return:
    """
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
    """
    does not work
    :param mu: mean of xi
    :param Sigma: covariance of xi
    :param r: radius for U_constraints
    :return:
    """
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
    rho = 7550

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
        consts = [rho * tau[l] - 2 * phi[l].T @ mu + theta[l] <= gamma,  # TODO: MAYBE ADD a0 @ x_hat ? ?????
                  2 * cp.reshape(phi[l], shape=(10,)) + 2 * varphi[:, l] - beta + A[l].T @ x_tilde == 0,
                  # TODO: c@x???
                  cp.bmat([[Lambda, phi[l]], [phi[l].T, theta[l]]]) >> 0,
                  cp.norm(varphi[:, l]) <= tau[l]
                  ]
        constraints += consts

    objective = cp.Minimize(mu @ beta + matrix_inner_product(Lambda, Sigma) + gamma - c @ x)
    prob = cp.Problem(objective=objective, constraints=constraints)
    prob.solve()
    return prob.value, x.value


def get_Al():
    q1 = q2 = 2
    A1_ = q1 * np.block([[np.eye(4), np.zeros([4, 6])], [np.zeros(8), -1, 0]])
    A2_ = q2 * np.block([[np.zeros([4, 4]), np.eye(4), np.zeros([4, 2])], [np.zeros(9), -1]])
    A3_ = A1_ + A2_
    A4_ = np.zeros([5, 10])
    A = [A1_, A2_, A3_, A4_]
    return A


def solve_2(mu, t, r, g_vecs):
    """"
    Old version, does not work
    """
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


def solve_4(N, r, epsilon, train=None):
    """
    surprisingly, works
    :param N: Train set size
    :param r: radius for U_constraints
    :param epsilon: size of ball around data points
    :param train: a train set to use. If None then samples a new train set of size N
    :return:
    """
    xi_0 = np.array([4, 9, 7, 10, 1, 1, 3, 40, 6000, 400])
    G = np.vstack([np.identity(10), -np.identity(10)])
    b, B = get_b_and_B()
    h = np.hstack([xi_0 - b, -xi_0 - b])
    c = np.array([12, 20, 18, 40])
    F = -np.identity(20)
    if train is None:
        train = [sample_vars() for _ in range(N)]
    A_l = get_Al()
    lamda = cp.Variable(1)
    t = [cp.Variable(1) for _ in range(N)]
    x = cp.Variable(4)
    phi = [[cp.Variable(20) for _ in range(4)] for _ in range(N)]
    A1 = np.diag([1] * 4 + [0] * 4 + [1, 0])
    A2 = np.eye(10) - A1
    h_2 = np.hstack([-xi_0 + b, xi_0 + b, -np.diag(1 / b) @ xi_0, r])
    G_2 = np.vstack([np.eye(10), -np.eye(10), np.diag(1 / b), np.zeros((1, 10))])
    x_hat = np.hstack([np.eye(4), np.eye(4), np.zeros([4, 2])]).T @ x - np.hstack([np.zeros(8), np.ones(2)])
    y1 = cp.Variable(31)
    y2 = cp.Variable(31)

    constraints = [x >= 0,
                   h_2 @ y1 >= -600, h_2 @ y2 >= -600,
                   G_2.T @ y1 == A1.T @ x_hat,
                   G_2.T @ y2 == A2.T @ x_hat,
                   y1[:20] <= 0, y2[:20] <= 0,
                   cp.norm2(y1[20:30]) <= -y1[30], cp.norm2(y2[20:30]) <= -y2[30]
                   ]
    x_tilde = np.eye(5, 4) @ x + np.array([0] * 4 + [1])
    for i in range(N):
        t_i = t[i]
        xi_i = train[i]
        for l in range(4):
            print(l)
            phi_il = phi[i][l]
            """constraints += [(h-G@xi_i)@phi_il + (A_l[l]@xi_i).T@x_tilde <= t_i,
                            cp.norm(A_l[l].T@x_tilde - G.T@phi_il) <= lamda,
                            phi_il >= 0,
                            lamda >= 0]"""
            phi_il = phi[i][l]
            constraints += [(A_l[l] @ xi_i).T @ x_tilde <= t_i,
                            cp.norm(A_l[l].T @ x_tilde, 'inf') <= lamda,
                            lamda >= 0]
    objective = cp.Minimize(lamda * epsilon + 1 / N * cp.sum(t) - c @ x)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return prob.value, x.value


def ei(i, length) -> np.ndarray:
    """
    elementary vector e_i
    :param i: index
    :param length: num od rows
    :return: np.ndarray of zeros except of index i, which contains 1
    """
    arr = np.zeros(length)
    arr[i] = 1
    return arr


def solve_2_v2(mu, t, r, g_vecs):
    """
    newer version, does not work either
    :param mu:
    :param t:
    :param r: radius for U_constraints
    :param g_vecs: g_i s
    :return:
    """
    d = len(mu)
    m = len(t)
    L = 4
    x = cp.Variable(4)
    beta = cp.Variable(d + m)
    phi = [cp.Variable(3 * m + 2 * d) for l in range(L)]
    kappa = cp.Variable(1)
    lamda = cp.Variable(1)
    y1 = cp.Variable(3 * d + 1)
    y2 = cp.Variable(3 * d + 1)
    gamma = cp.Variable(1)
    x_tilde = np.eye(5, 4) @ x + np.array([0] * 4 + [1])

    b, B = get_b_and_B()
    c = np.array([12, 20, 18, 40])
    A1 = np.diag([1] * 4 + [0] * 4 + [1, 0])
    A2 = np.eye(10) - A1
    G = np.vstack([np.eye(10), -np.eye(10), np.diag(1 / b), np.zeros([1, 10])])
    x_hat = np.vstack([np.eye(4), np.eye(4), np.zeros([2, 4])]) @ x - np.hstack([np.zeros(8), np.ones(2)])
    h = np.concatenate([-mu + b, mu + b, np.diag(1 / b) @ mu, np.array([r])])

    C = np.vstack([np.eye(d), np.zeros([m, d])])
    D = np.vstack([np.zeros([d, m]), np.eye(m)])
    G_cone = np.vstack(
        [np.vstack([gi.T, np.zeros([2, d])]) for gi in g_vecs] + [np.vstack([ei(i, d).T, np.zeros(d)]) for
                                                                  i in range(d)])
    F_cone = 1 / 2 * np.vstack(
        [np.vstack([np.zeros([1, m]), ei(i, m), ei(i, m)]) for i in range(m)] + [np.zeros([2 * d, m])])
    h_cone = np.vstack(
        [np.vstack([mu @ gi, 1 / 2, -1 / 2]) for gi in g_vecs] + [np.vstack([mu[i], b[i]]) for i in range(d)])

    A = get_Al()

    constraints = [x >= 0,
                   h @ y1 >= -600, h @ y2 >= -600,
                   G.T @ y1 == A1.T @ x_hat,
                   G.T @ y2 == A2.T @ x_hat,
                   y1[:20] <= 0, y2[:20] <= 0,
                   cp.norm2(y1[20:30]) <= -y1[30], cp.norm2(y2[20:30]) <= -y2[30],
                   kappa >= 0,
                   lamda >= 0
                   ]

    def in_lorenz_k(vec):
        return cp.norm(vec[:-1]) <= vec[-1]

    for l in range(L):
        constraints_l = [
            lamda - kappa - h_cone.T @ phi[l] <= gamma,
            A[l].T @ x_tilde - C.T @ beta + G_cone.T @ phi[l] == 0,
            -D.T @ beta + F_cone.T @ phi[l] == 0,
        ]
        constraints_l += [in_lorenz_k(phi[l][3 * index: 3 * index + 3]) for index in range(m)] \
                         + [in_lorenz_k(phi[l][3 * m + 2 * index: 3 * m + 2 * index + 2]) for index in range(d)]
        constraints += constraints_l

    xi0_t = C @ mu + D @ t
    objective = cp.Minimize(beta @ xi0_t + kappa - lamda + gamma - c @ x)
    prob = cp.Problem(objective=objective, constraints=constraints)
    prob.solve()
    return prob.value, x.value


def solve_3(mu, Sigma, gamma1, gamma2):
    """
    does not work
    :param mu:
    :param Sigma:
    :param gamma1:
    :param gamma2:
    :return:
    """
    d = len(mu)
    L = 4
    x = cp.Variable(4)
    Lambda = cp.Variable([d, d])
    phi = cp.Variable([1, 1])
    delta = cp.Variable(1)
    v_M = [cp.Variable([d, 1]) for l in range(L)]
    v_N = [cp.Variable([d, 1]) for l in range(L)]
    d_M = [cp.Variable([1, 1]) for l in range(L)]
    C_N = [cp.Variable([d, d]) for l in range(L)]
    y1 = cp.Variable(3 * d + 1)
    y2 = cp.Variable(3 * d + 1)
    theta_1 = [cp.Variable(d) for l in range(L)]
    theta_2 = [cp.Variable(d) for l in range(L)]
    x_tilde = np.eye(5, 4) @ x + np.array([0] * 4 + [1])

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
                   ]
    A = get_Al()
    for l in range(L):
        first_mat = cp.bmat([[Lambda, v_M[l]], [v_M[l].T, d_M[l]]])
        secont_mat = cp.bmat([[C_N[l], v_N[l]], [v_N[l].T, phi]])
        l_constraints = [
            delta >= theta_1[l].T @ (mu + b) + theta_2[l].T @ (b - mu) - 2 * (v_M[l] + v_N[l]).T @ mu + d_M[l] \
            + cp.trace(C_N[l].T @ Sigma),
            x_tilde @ A[l] - theta_1[l] + theta_2[l] + 2 * cp.reshape(v_M[l] + v_N[l], [10, ]) == 0,
            first_mat >> 0,
            secont_mat >> 0
        ]
        constraints += l_constraints

    objective = cp.Minimize(delta + gamma1 * cp.trace(Lambda @ Sigma) + gamma2 * phi - c @ x)
    prob = cp.Problem(objective=objective, constraints=constraints)
    prob.solve()
    return prob.value, x.value


def solve_5(N, r, epsilon, train=None):
    """
    does not work
    :param N:
    :param r:
    :param epsilon:
    :param train: train set to use. If None then samples a new train set
    :return:
    """
    x = cp.Variable(4)
    c = np.array([12, 20, 18, 40])
    b, B = get_b_and_B()
    xi_0 = np.array([4, 9, 7, 10, 1, 1, 3, 40, 6000, 400])
    G = np.vstack([-np.eye(10), np.eye(10), np.diag(1 / b), np.zeros(10), np.eye(10), np.zeros(10)])
    if train is None:
        train = [sample_vars() for _ in range(N)]
    h = []
    for i in range(N):
        h_i = np.hstack([xi_0 - b, -xi_0 + b, -np.diag(1 / b) @ xi_0, r, -train[i].reshape(-1), epsilon])
        h.append(h_i)
    F = -np.eye(42)
    y = [[cp.Variable(42) for l in range(4)] for _ in range(N)]
    constraints = []
    gamma = [cp.Variable(1) for _ in range(N)]
    A_l = get_A_l()
    x_tilde = np.eye(5, 4) @ x + np.array([0] * 4 + [1])
    A1 = np.diag([1] * 4 + [0] * 4 + [1, 0])
    A2 = np.eye(10) - A1
    h_2 = np.hstack([-xi_0 + b, xi_0 + b, -np.diag(1 / b) @ xi_0, r])
    G_2 = np.vstack([np.eye(10), -np.eye(10), np.diag(1 / b), np.zeros((1, 10))])
    x_hat = np.hstack([np.eye(4), np.eye(4), np.zeros([4, 2])]).T @ x - np.hstack([np.zeros(8), np.ones(2)])
    y1 = cp.Variable(31)
    y2 = cp.Variable(31)
    constraints = [x >= 0,
                   h_2 @ y1 >= -600, h_2 @ y2 >= -600,
                   G_2.T @ y1 == A1.T @ x_hat,
                   G_2.T @ y2 == A2.T @ x_hat,
                   y1[:20] <= 0, y2[:20] <= 0,
                   cp.norm2(y1[20:30]) <= -y1[30], cp.norm2(y2[20:30]) <= -y2[30]
                   ]
    c_tilde = np.hstack([c, 0])
    for i in range(N):
        for l in range(4):
            constraints.append(h[i] @ y[i][l] >= -gamma[i] - x_tilde @ c_tilde)
            constraints.append(-G.T @ y[i][l] == -A_l[l].T @ x_tilde)
            constraints += [-y[i][l][:20] >= 0, cp.norm(-y[i][l][20:30]) <= -y[i][l][30],
                            cp.norm(-y[i][l][31:41]) <= -y[i][l][41]]
    objective = cp.Minimize(1 / N * cp.sum(gamma))
    prob = cp.Problem(objective, constraints)
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
        t = [np.mean([((xi - mu) @ gi) ** 2 for xi in train_set]) for gi in g]
        print(1)
        print(solve_1(mu, Sigma, r))
        print(2)
        print(solve_2_v2(mu, t, r, g))
        print(3)
        print(solve_3(mu, Sigma, gamma1, gamma2))
        print(4)
        print(solve_4(N, r, epsilon=1e-3, train=train_set))
        print(5)
        print(solve_5(N, r, epsilon=1e-3))
