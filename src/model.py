import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import pdf
from jaxtyping import Array, ArrayLike, Float

if not jax.config.read("jax_enable_x64"):
    raise RuntimeError(
        "This project requires JAX float64. "
        "Set JAX_ENABLE_X64=1 before importing src modules."
    )

ACTIONS = jnp.array([-1.0, 0.0, 1.0])

MESH = 0.05
NOISE_L = -5.0
NOISE_R = 5.0

NOISE = jnp.arange(NOISE_L, NOISE_R + MESH, MESH)

weights = pdf(NOISE) * MESH
PROBABILITIES = weights / jnp.sum(weights)


def cost(
    x: Float[Array, "*x_shape"],
    a: Float[Array, "*a_shape"],
) -> Array:
    return x**2 + a**2


def terminal_cost(
    x: Float[ArrayLike, "*batch"],
) -> Float[Array, "*batch"]:
    x = jnp.asarray(x)
    return x**2


def transition(
    x: Float[Array, "*batch"],
    a: Float[Array, ""],
    noise: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    return x + a + noise
