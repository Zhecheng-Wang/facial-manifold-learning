import numpy as np
from numpy.typing import NDArray
from typing import Callable, TypeAlias, Literal
from scipy.optimize import least_squares, minimize
import torch
from torch import Tensor

ND32: TypeAlias = torch.Tensor


ND32: TypeAlias = Tensor


import torch
from torch import Tensor
from typing import Callable, TypeAlias

ND32: TypeAlias = Tensor


class LeastSquaresSolverVanilla:
    """Least squares solver for blendshape fitting."""

    m_pred_fn: Callable[[ND32], ND32] = lambda x: x
    target: ND32
    eps: float = 1e-4  # finite‐difference step
    lr: float = 1e-2
    steps: int = 400

    def __init__(self, z_shape: int):
        self.target = torch.zeros([z_shape])
        self.z = torch.zeros([z_shape])

    def solve(
        self, m_pred_fn: Callable[[ND32], ND32], target: ND32, verbose: bool = False
    ) -> ND32:
        """Solve the least squares problem via gradient descent."""
        self.m_pred_fn = m_pred_fn
        self.target = target
        self.z = torch.zeros_like(self.z)

        for i in range(self.steps):
            # g = self.grad(self.z)
            g = self.grad_via_vjp(self.z)
            self.z = self.z - self.lr * g

            if verbose and (i % 10 == 0 or i == self.steps - 1):
                l = self.loss(self.z)
                print(f"[iter {i:03d}] loss={l:.6f}, ‖grad‖={g.norm():.6f}")

        return self.z

    def residual(self, z: ND32) -> ND32:
        return self.m_pred_fn(z) - self.target

    def loss(self, z: ND32) -> float:
        r = self.residual(z)
        return float((r**2).sum().item())

    def grad(self, z: ND32) -> ND32:
        """
        Exact least-squares gradient: ∇L = 2 J(z)^T r(z).
        """
        # J = self.jacobian(z)  # shape [n, m]
        r = self.residual(z).view(-1)  # shape [n]
        # grad is 2 * J^T @ r
        return 2.0 * J.t().matmul(r)  # shape [m]

    def grad_via_vjp(self, z: torch.Tensor) -> torch.Tensor:
        # Build a differentiable z
        z_req = z.clone().detach().requires_grad_(True)
        pred = self.m_pred_fn(z_req).view(-1)  # [P]
        r = pred - self.target.view(-1)  # [P]

        # Compute J^T r directly
        # torch.autograd.grad returns a tuple; we take [0]
        Jt_r = torch.autograd.grad(
            outputs=pred,
            inputs=z_req,
            grad_outputs=r,
            retain_graph=False,
            create_graph=False,
        )[
            0
        ]  # shape [W]

        # And the gradient of L=r^Tr is 2 J^T r
        return 2.0 * Jt_r

    # def jacobian_autograd(self, z: torch.Tensor) -> torch.Tensor:
    #     # This would take too long calculating the entire jacobian.
    #     # z: shape [W], requires_grad=False is fine
    #     # M(z) should return a Tensor of shape [N,3] (or any shape)
    #     # we’ll flatten the output to a vector of length P = N*3
    #     def M_flat(z_):
    #         return self.m_pred_fn(z_).view(-1)  # shape [P]
    #
    #     # returns a Tensor of shape [P, W]
    #     J = jacobian(M_flat, z)
    #     return J

    def jacobian(self, z: ND32) -> ND32:
        """
        Numerically approximate the Jacobian J_ij = ∂[M(z)]_i / ∂z_j
        using forward finite differences.
        Returns a tensor of shape [n, m], where n = # output dims, m = len(z).
        """
        z0 = z.clone().detach()
        V0 = self.m_pred_fn(z0).detach().view(-1)  # flatten to shape [n]
        m = z0.numel()
        n = V0.numel()

        J = torch.zeros(n, m, device=z.device)
        for j in range(m):
            zj = z0.clone().detach().view(-1)
            zj[j] += self.eps
            zj = zj.view_as(z)  # restore original shape
            V1 = self.m_pred_fn(zj).detach().view(-1)
            J[:, j] = (V1 - V0) / self.eps

        return J


class GradientSolverSciPy:
    z_shape: int

    def __init__(self, z_shape: int):
        self.z_shape = z_shape

    def solve(
        self, m_pred_fn: Callable[[ND32], ND32], target: ND32, verbose: bool = False
    ) -> np.ndarray:
        """
        Minimize L(z) = ||M(z) - target||^2 using scipy.optimize.minimize,
        supplying exact gradients via PyTorch autograd.
        """
        # Move target to a fixed torch tensor
        target_t = target.detach()

        def fun_and_grad(z_np: np.ndarray):
            # Convert to torch tensor, enable grad
            z_t = torch.from_numpy(z_np.astype(np.float32))
            z_t.requires_grad_(True)

            # Forward: predict mesh and compute loss
            pred = m_pred_fn(z_t)  # [N,3]
            r = pred - target_t  # [N,3]
            loss = (r.view(-1) ** 2).sum()  # scalar

            # Backward: get gradient dL/dz and populate z's gradients
            loss.backward()
            grad = z_t.grad  # [W]

            # Return scalar and numpy gradient
            return float(loss.item()), grad.cpu().numpy()

        # wrapper for scipy API
        def fun(z_np: np.ndarray) -> float:
            f, _ = fun_and_grad(z_np)
            return f

        def jac(z_np: np.ndarray) -> np.ndarray:
            _, g = fun_and_grad(z_np)
            return g

        # Initial guess
        z0 = np.zeros(self.z_shape, dtype=np.float32)

        res = minimize(
            fun=fun,
            x0=z0,
            jac=jac,
            bounds=[(-3.0, 3.0)] * self.z_shape,
            method="L-BFGS-B",  # supports bounds & uses gradient
            options={"ftol": 1e-6, "gtol": 1e-6, "maxiter": 1000, "disp": verbose},
        )

        return res.x


class LeastSquaresSolver:
    def __init__(self, z_shape: int, lr=1e-1, steps=100, opt_name: str = "sgd"):
        self.z_shape = z_shape
        self.lr = lr
        self.steps = steps
        self.opt_name = opt_name.lower()

    def solve(
        self, m_pred_fn: Callable[[Tensor], Tensor], target: Tensor, verbose=False
    ) -> torch.Tensor:
        # initialize z
        z = torch.nn.Parameter(
            torch.zeros(self.z_shape, dtype=torch.float32, device=target.device)
        )
        target_t = target.to(z.device)

        # pick optimizer (or None for pure GD)
        optimizer = None
        if self.opt_name == "sgd":
            optimizer = torch.optim.SGD([z], lr=self.lr, momentum=0.0)
        elif self.opt_name == "sgd_mom":
            optimizer = torch.optim.SGD([z], lr=self.lr, momentum=0.9)
        elif self.opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop([z], lr=self.lr, alpha=0.99)
        elif self.opt_name == "adagrad":
            optimizer = torch.optim.Adagrad([z], lr=self.lr)
        elif self.opt_name == "adamw":
            optimizer = torch.optim.AdamW([z], lr=self.lr, weight_decay=1e-4)
        elif self.opt_name == "lbfgs":
            optimizer = torch.optim.LBFGS([z], lr=self.lr, steps=20, history_size=10)
        elif self.opt_name == "gd" or self.opt_name == "vanilla":
            optimizer = None  # will do manual gradient descent
        else:  # default to Adam
            optimizer = torch.optim.Adam([z], lr=self.lr)

        for i in range(self.steps):
            if isinstance(optimizer, torch.optim.LBFGS):

                def closure():
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(m_pred_fn(z), target_t)
                    loss.backward()
                    return loss

                optimizer.step(closure)

            elif optimizer is not None:
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(m_pred_fn(z), target_t)
                loss.backward()
                optimizer.step()
            else:
                # pure gradient descent
                if z.grad is not None:
                    z.grad.zero_()

                loss = torch.nn.functional.mse_loss(m_pred_fn(z), target_t)
                loss.backward()

                with torch.no_grad():
                    z.data -= self.lr * z.grad

            if verbose and (i % 10 == 0 or i == self.steps - 1):
                z_cpu = z.detach().cpu()
                print(
                    f"[iter {i:03d}]  loss={loss:.12f}, "
                    f"max={z_cpu.max():.4f}, "
                    f"‖grad‖={z.grad.norm():.24f}"
                )

        return z.detach().cpu().numpy()
