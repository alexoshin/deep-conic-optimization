"""Robust Kalman filtering SOCPs."""

from jaxtyping import Float
import torch
from torch import Tensor
from ..cones import Cone, ZeroCone, NonnegativeOrthant, SecondOrderCone


def generate_robust_kalman_filter_socp(
    N: int = 10,
    n_x: int = 4,
    dt: float = 0.1,
    layout: str = "csr",
) -> tuple[
    Float[Tensor, "n n"],
    Float[Tensor, "n"],
    Float[Tensor, "m n"],
    Float[Tensor, "m"],
    list[Cone],
]:
    """Generates a random robust Kalman filtering SOCP."""

    if n_x <= 0 or n_x % 2 != 0:
        raise ValueError("n_x must be a positive even integer.")

    n_pos = n_x // 2
    n_w = n_pos
    n_y = n_pos

    tau = 2.0  # Scaling for Huber terms
    rho = 2.0  # Huber parameter

    # Define dynamics
    gamma = 0.05  # drag
    A = torch.eye(n_x)
    A[:n_pos, n_pos:] = (1.0 - 0.5 * gamma * dt) * dt * torch.eye(n_pos)
    A[n_pos:, n_pos:] = (1.0 - gamma * dt) * torch.eye(n_pos)

    B = torch.zeros(n_x, n_w)
    B[:n_pos, :] = 0.5 * dt**2 * torch.eye(n_pos)
    B[n_pos:, :] = dt * torch.eye(n_pos)

    C_mat = torch.zeros(n_y, n_x)
    C_mat[:, :n_pos] = torch.eye(n_pos)

    # Generate noise
    w_traj = torch.randn(N, n_w)
    v_traj = torch.randn(N, n_y)

    # Add outliers
    p = 0.2
    outlier_std = 20.0
    inds = torch.rand(N) < p
    v_traj[inds] = outlier_std * torch.randn(N, n_y)[inds]

    # Initial state
    x_init = torch.zeros(n_x)

    # Simulate the system to get the measurements
    y_traj = torch.zeros(N, n_y)
    x_traj = torch.zeros(N + 1, n_x)
    x_traj[0] = x_init
    for k in range(N):
        y_traj[k] = C_mat @ x_traj[k] + v_traj[k]
        x_traj[k + 1] = A @ x_traj[k] + B @ w_traj[k]

    # Define the optimization problem
    n = N * (n_x + n_w + n_y + 4) + n_x

    # Define indices for the variables
    var_inds = {}
    start = 0
    for k in range(N + 1):
        # x_k
        end = start + n_x
        var_inds[f"x_{k}"] = slice(start, end)
        start = end

    for k in range(N):
        # w_k
        end = start + n_w
        var_inds[f"w_{k}"] = slice(start, end)
        start = end

    for k in range(N):
        # v_k
        end = start + n_y
        var_inds[f"v_{k}"] = slice(start, end)
        start = end

    for k in range(N):
        # t_k
        var_inds[f"t_{k}"] = start
        start += 1

    for k in range(N):
        # z_k
        var_inds[f"z_{k}"] = start
        start += 1

    for k in range(N):
        # r_k
        var_inds[f"r_{k}"] = start
        start += 1

    for k in range(N):
        # s_k
        var_inds[f"s_{k}"] = start
        start += 1

    # Define the objective
    P = torch.zeros(n, n)
    q = torch.zeros(n)
    for k in range(N):
        w_k_inds = var_inds[f"w_{k}"]
        P[w_k_inds, w_k_inds] = torch.eye(n_w)

        z_k_ind = var_inds[f"z_{k}"]
        P[z_k_ind, z_k_ind] = tau

        r_k_ind = var_inds[f"r_{k}"]
        q[r_k_ind] = tau * rho

        s_k_ind = var_inds[f"s_{k}"]
        q[s_k_ind] = tau * rho

    # Define the equality constraints
    n_eq = N * (n_x + n_y + 1)
    A_eq = torch.zeros(n_eq, n)
    b_eq = torch.zeros(n_eq)

    start = 0
    for k in range(N):
        # Dynamics: x_{k+1} = A x_k + B w_k
        end = start + n_x
        A_eq[start:end, var_inds[f"x_{k + 1}"]] = -torch.eye(n_x)
        A_eq[start:end, var_inds[f"x_{k}"]] = A
        A_eq[start:end, var_inds[f"w_{k}"]] = B
        start = end

        # Measurement: y_k = C x_k + v_k
        end = start + n_y
        A_eq[start:end, var_inds[f"x_{k}"]] = C_mat
        A_eq[start:end, var_inds[f"v_{k}"]] = torch.eye(n_y)
        b_eq[start:end] = y_traj[k]
        start = end

        # Constraint for auxiliary variables: t_k - z_k = r_k - s_k
        end = start + 1
        A_eq[start:end, var_inds[f"t_{k}"]] = 1.0
        A_eq[start:end, var_inds[f"z_{k}"]] = -1.0
        A_eq[start:end, var_inds[f"r_{k}"]] = -1.0
        A_eq[start:end, var_inds[f"s_{k}"]] = 1.0
        start = end

    # Define the inequality constraints
    n_ineq = 2 * N
    G_ineq = torch.zeros(n_ineq, n)
    h_ineq = torch.zeros(n_ineq)

    start = 0
    for k in range(N):
        # r_k >= 0  => -r_k <= 0
        end = start + 1
        G_ineq[start:end, var_inds[f"r_{k}"]] = -1.0
        h_ineq[start:end] = 0.0
        start = end

        # s_k >= 0 => -s_k <= 0
        end = start + 1
        G_ineq[start:end, var_inds[f"s_{k}"]] = -1.0
        h_ineq[start:end] = 0.0
        start = end

    # Define the second-order cone constraints
    socc_As = []
    socc_bs = []
    socc_cones = []

    current_idx = n_eq + n_ineq

    for k in range(N):
        # ||v_k||_2 <= t_k
        # -t_k + s_t = 0
        # -v_k + s_vec = 0

        A_k = torch.zeros(n_y, n)
        A_k[:, var_inds[f"v_{k}"]] = torch.eye(n_y)
        c_k = torch.zeros(n)
        c_k[var_inds[f"t_{k}"]] = 1.0

        A_cone = torch.cat([-c_k.unsqueeze(0), -A_k], dim=0)
        b_cone = torch.zeros(n_y + 1)

        socc_As.append(A_cone)
        socc_bs.append(b_cone)

        cone_size = 1 + n_y
        socc_cones.append(SecondOrderCone(torch.arange(current_idx, current_idx + cone_size)))
        current_idx += cone_size

    if socc_As:
        A_socc = torch.cat(socc_As, dim=0)
        b_socc = torch.cat(socc_bs, dim=0)
    else:
        A_socc = torch.zeros(0, n)
        b_socc = torch.zeros(0)

    # Assemble final matrices
    A_cosmo = torch.cat([A_eq, G_ineq, A_socc], dim=0)
    b_cosmo = torch.cat([b_eq, h_ineq, b_socc], dim=0)

    if layout == "coo":
        P = P.to_sparse_coo()
        A_cosmo = A_cosmo.to_sparse_coo()
    elif layout == "csr":
        P = P.to_sparse_csr()
        A_cosmo = A_cosmo.to_sparse_csr()
    elif layout == "csc":
        P = P.to_sparse_csc()
        A_cosmo = A_cosmo.to_sparse_csc()
    else:  # layout == "dense"
        pass

    cones = []
    if n_eq > 0:
        cones.append(ZeroCone(torch.arange(n_eq)))
    if n_ineq > 0:
        cones.append(NonnegativeOrthant(torch.arange(n_eq, n_eq + n_ineq)))
    cones.extend(socc_cones)

    return P, q, A_cosmo, b_cosmo, cones
