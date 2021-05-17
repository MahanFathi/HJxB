import jax.numpy as jnp
from jax import jit, lax, vmap

from typing import (
    Callable,
    Tuple,
    Optional,
    Union,
)

from enum import Enum

# props: integrators are borrowed from https://github.com/nikihowe/myriad

class IntegrationOrder(Enum):
  CONSTANT="CONSTANT"
  LINEAR="LINEAR"
  QUADRATIC="QUADRATIC"

def integrate(
        dynamics_t: Callable[[jnp.ndarray, Union[float, jnp.ndarray]], jnp.ndarray],  # dynamics function
        x_0: jnp.ndarray,  # starting state
        interval_us: jnp.ndarray,  # controls
        h: float,  # step size
        N: int,  # steps
        ts: Optional[jnp.ndarray], # allow for optional time-dependent dynamics
        integration_order: IntegrationOrder, # allows user to choose interpolation for controls
) -> Tuple[jnp.ndarray, jnp.ndarray]:

  @jit
  def rk4_step(x, u1, u2, u3, ts=None):
      k1 = dynamics_t(x, u1)
      k2 = dynamics_t(x + h*k1/2, u2)
      k3 = dynamics_t(x + h*k2/2, u2)
      k4 = dynamics_t(x + h*k3, u3)
      return x + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

  @jit
  def heun_step(x, u1, u2, ts=None):
      k1 = dynamics_t(x, u1)
      k2 = dynamics_t(x + h*k1, u2)
      return x + (h/2) * (k1 + k2)

  @jit
  def euler_step(x, u, ts=None):
    return x + h*dynamics_t(x, u)

  def fn(carried_state, idx):
    if integration_order == IntegrationOrder.CONSTANT:
      if ts is not None:
          one_step_forward = euler_step(carried_state, interval_us[idx], *ts[idx:idx+2])
      else:
          one_step_forward = euler_step(carried_state, interval_us[idx])
    elif integration_order == IntegrationOrder.LINEAR:
      if ts is not None:
          one_step_forward = heun_step(carried_state, interval_us[idx], interval_us[idx+1] *ts[idx:idx+2])
      else:
          one_step_forward = heun_step(carried_state, interval_us[idx], interval_us[idx+1])
    elif integration_order == IntegrationOrder.QUADRATIC:
      if ts is not None:
          one_step_forward = rk4_step(carried_state, interval_us[2*idx], interval_us[2*idx+1], interval_us[2*idx+2], *ts[idx:idx+2])
      else:
          one_step_forward = rk4_step(carried_state, interval_us[2*idx], interval_us[2*idx+1], interval_us[2*idx+2])
    else:
        print("Please choose an integration order among: {CONSTANT, LINEAR, QUADRATIC}")
        raise KeyError

    return one_step_forward, one_step_forward # (carry, y)

  x_T, all_next_states = lax.scan(fn, x_0, jnp.arange(N))
  return x_T, jnp.concatenate((x_0[None], all_next_states))

# Used for the augmented state cost calculation
integrate_in_parallel = vmap(integrate, in_axes=(None, 0, 0, None, None, 0, None))#, static_argnums=(0, 5, 6)
