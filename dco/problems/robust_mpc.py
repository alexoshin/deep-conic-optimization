"""Robust MPC SOCPs."""

from jaxtyping import Float
import torch
from torch import Tensor

from ..cones import Cone, NonnegativeOrthant, SecondOrderCone, ZeroCone, convert_socc_to_conic


def propagate_covariances(
    A_d: Float[Tensor, "n_x n_x"],
    Sigma_w: Float[Tensor, "n_x n_x"],
    N: int,
) -> Float[Tensor, "T_plus_1 n_x n_x"]:
    """Propagate additive disturbance covariances for a nominal linear system."""
    if N < 0:
        raise ValueError("N must be nonnegative.")
    Sigma = torch.zeros(N + 1, A_d.shape[0], A_d.shape[0], dtype=A_d.dtype, device=A_d.device)
    for t in range(N):
        Sigma[t + 1] = A_d @ Sigma[t] @ A_d.mT + Sigma_w
    return Sigma


def state_constraint_tightening(
    A_x: Float[Tensor, "m_x n_x"],
    covariances: Float[Tensor, "T_plus_1 n_x n_x"],
) -> Float[Tensor, "T_plus_1 m_x"]:
    """Compute sqrt(a_i^N Sigma_t a_i) for each state half-space row."""
    variances = torch.einsum("in,tnm,im->ti", A_x, covariances, A_x)
    return torch.sqrt(torch.clamp(variances, min=0.0))


def assemble_robust_mpc_socp(
    A_d: Float[Tensor, "n_x n_x"],
    B_d: Float[Tensor, "n_x n_u"],
    Q: Float[Tensor, "n_x n_x"],
    R: Float[Tensor, "n_u n_u"],
    Q_T: Float[Tensor, "n_x n_x"],
    A_x: Float[Tensor, "m_x n_x"],
    b_x: Float[Tensor, "m_x"],
    A_u: Float[Tensor, "m_u n_u"],
    b_u: Float[Tensor, "m_u"],
    Sigma_w: Float[Tensor, "n_x n_x"],
    x_init: Float[Tensor, "n_x"],
    x_target: Float[Tensor, "n_x"],
    rho: float | Float[Tensor, ""],
    rho_T: float | Float[Tensor, ""],
    kappa: float = 2.0,
    N: int = 20,
    layout: str = "csr",
) -> tuple[
    Float[Tensor, "n n"],
    Float[Tensor, "n"],
    Float[Tensor, "m n"],
    Float[Tensor, "m"],
    list[Cone],
]:
    """Assemble a robust linear MPC problem in conic form."""
    if N <= 0:
        raise ValueError("N must be positive.")

    n_x = A_d.shape[0]
    n_u = B_d.shape[1]
    m_x = A_x.shape[0]
    m_u = A_u.shape[0]

    tensor_kwargs = {"dtype": A_d.dtype, "device": A_d.device}
    n = (N + 1) * n_x + N * n_u + N + 1
    n_eq = (N + 1) * n_x
    n_ineq = (N + 1) * m_x + N * m_u
    n_soc = N * (n_u + 1) + (n_x + 1)
    m = n_eq + n_ineq + n_soc

    def x_slice(t: int) -> slice:
        start = t * n_x
        return slice(start, start + n_x)

    u_start = (N + 1) * n_x

    def u_slice(t: int) -> slice:
        start = u_start + t * n_u
        return slice(start, start + n_u)

    tau_start = u_start + N * n_u

    def tau_index(t: int) -> int:
        return tau_start + t

    tau_terminal_index = tau_start + N

    P = torch.zeros(n, n, **tensor_kwargs)
    q = torch.zeros(n, **tensor_kwargs)
    for t in range(N):
        P[x_slice(t), x_slice(t)] = 2.0 * Q
        P[u_slice(t), u_slice(t)] = 2.0 * R
        q[tau_index(t)] = torch.as_tensor(rho, **tensor_kwargs)
    P[x_slice(N), x_slice(N)] = 2.0 * Q_T
    q[tau_terminal_index] = torch.as_tensor(rho_T, **tensor_kwargs)

    A = torch.zeros(m, n, **tensor_kwargs)
    b = torch.zeros(m, **tensor_kwargs)

    row = 0
    for t in range(N):
        next_row = row + n_x
        A[row:next_row, x_slice(t + 1)] = torch.eye(n_x, **tensor_kwargs)
        A[row:next_row, x_slice(t)] = -A_d
        A[row:next_row, u_slice(t)] = -B_d
        row = next_row

    next_row = row + n_x
    A[row:next_row, x_slice(0)] = torch.eye(n_x, **tensor_kwargs)
    b[row:next_row] = x_init
    row = next_row

    covariances = propagate_covariances(A_d, Sigma_w, N)
    tightening = state_constraint_tightening(A_x, covariances)
    for t in range(N + 1):
        next_row = row + m_x
        A[row:next_row, x_slice(t)] = A_x
        b[row:next_row] = b_x - kappa * tightening[t]
        row = next_row

    for t in range(N):
        next_row = row + m_u
        A[row:next_row, u_slice(t)] = A_u
        b[row:next_row] = b_u
        row = next_row

    cones: list[Cone] = [
        ZeroCone(torch.arange(n_eq)),
        NonnegativeOrthant(torch.arange(n_eq, n_eq + n_ineq)),
    ]

    for t in range(N):
        F = torch.zeros(n_u, n, **tensor_kwargs)
        F[:, u_slice(t)] = torch.eye(n_u, **tensor_kwargs)
        c = torch.zeros(n, **tensor_kwargs)
        c[tau_index(t)] = 1.0
        A_cone, b_cone = convert_socc_to_conic(
            F, torch.zeros(n_u, **tensor_kwargs), c, torch.zeros((), **tensor_kwargs)
        )
        next_row = row + n_u + 1
        A[row:next_row] = A_cone
        b[row:next_row] = b_cone
        cones.append(SecondOrderCone(torch.arange(row, next_row)))
        row = next_row

    F = torch.zeros(n_x, n, **tensor_kwargs)
    F[:, x_slice(N)] = torch.eye(n_x, **tensor_kwargs)
    c = torch.zeros(n, **tensor_kwargs)
    c[tau_terminal_index] = 1.0
    A_cone, b_cone = convert_socc_to_conic(F, -x_target, c, torch.zeros((), **tensor_kwargs))
    next_row = row + n_x + 1
    A[row:next_row] = A_cone
    b[row:next_row] = b_cone
    cones.append(SecondOrderCone(torch.arange(row, next_row)))
    row = next_row

    if row != m:
        raise RuntimeError(f"assembled {row} rows, expected {m}.")

    if layout == "coo":
        P = P.to_sparse_coo()
        A = A.to_sparse_coo()
    elif layout == "csr":
        P = P.to_sparse_csr()
        A = A.to_sparse_csr()
    elif layout == "csc":
        P = P.to_sparse_csc()
        A = A.to_sparse_csc()
    else:  # layout == "dense"
        pass

    return P, q, A, b, cones


def generate_robust_mpc_socp(
    N: int = 20,
    n_masses: int = 3,
    dt: float = 0.2,
    k_spring: float = 1.0,
    c_damp: float = 0.3,
    kappa: float = 2.0,
    sigma: float = 0.01,
    u_max: float = 0.5,
    pos_bound: float = 5.0,
    vel_bound: float = 1.0,
    x0_pos_scale: float = 1.0,
    rho_range: tuple[float, float] = (0.01, 0.1),
    rho_T_range: tuple[float, float] = (1.0, 10.0),
    layout: str = "csr",
):
    """Robust MPC for a mass-spring-damper chain.

    The system consists of `n_masses` unit masses in series, with the left end anchored.
    The dynamics are thus fixed across instances, with only x_init and the cost weights varying.
    """

    Zero_masses = torch.zeros(n_masses, n_masses)
    I_masses = torch.eye(n_masses)

    # Construct the continuous-time dynamics A_c = [0 I; -K -C], B_c = [0; I]
    K = Zero_masses.clone()
    for i in range(n_masses):
        K[i, i] = 2.0 * k_spring
        if i > 0:
            K[i, i - 1] = -k_spring
        if i < n_masses - 1:
            K[i, i + 1] = -k_spring
    K[n_masses - 1, n_masses - 1] = k_spring  # free right end (one neighbor)

    C = c_damp * I_masses

    Ac = torch.cat([torch.cat([Zero_masses, I_masses], dim=1), torch.cat([-K, -C], dim=1)], dim=0)
    Bc = torch.cat([Zero_masses, I_masses], dim=0)

    n_x = 2 * n_masses
    n_u = n_masses

    # Exact zero-order-hold discretization using expm([Ac, Bc; 0, 0] * dt)
    aug = torch.zeros(n_x + n_u, n_x + n_u)
    aug[:n_x, :n_x] = Ac
    aug[:n_x, n_x:] = Bc

    Md = torch.matrix_exp(aug * dt)
    A_d = Md[:n_x, :n_x]
    B_d = Md[:n_x, n_x:]

    n_pos = n_x // 2
    Q = torch.eye(n_x)
    R = torch.eye(n_u)
    Q_T = torch.eye(n_x)

    A_x = torch.cat([torch.eye(n_x), -torch.eye(n_x)], dim=0)
    per_state = torch.cat([pos_bound * torch.ones(n_pos), vel_bound * torch.ones(n_x - n_pos)])
    b_x = torch.cat([per_state, per_state])  # upper then lower (symmetric box)

    A_u = torch.cat([torch.eye(n_u), -torch.eye(n_u)], dim=0)
    b_u = u_max * torch.ones(2 * n_u)

    Sigma_w = sigma**2 * torch.eye(n_x)

    x_init = torch.cat([x0_pos_scale * (-1.0 + 2.0 * torch.rand(n_masses)), -0.3 + 0.6 * torch.rand(n_masses)])
    x_target = torch.zeros(n_x)

    rho = rho_range[0] + torch.rand(()) * (rho_range[1] - rho_range[0])
    rho_T = rho_T_range[0] + torch.rand(()) * (rho_T_range[1] - rho_T_range[0])

    return assemble_robust_mpc_socp(
        A_d=A_d,
        B_d=B_d,
        Q=Q,
        R=R,
        Q_T=Q_T,
        A_x=A_x,
        b_x=b_x,
        A_u=A_u,
        b_u=b_u,
        Sigma_w=Sigma_w,
        x_init=x_init,
        x_target=x_target,
        rho=rho,
        rho_T=rho_T,
        kappa=kappa,
        N=N,
        layout=layout,
    )
