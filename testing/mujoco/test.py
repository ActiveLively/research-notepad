from jax import numpy as jnp
from dynamics import Dynamics, unroll

if __name__ == '__main__':

    horizon = 1000

    path = 'franka_emika_panda/mjx_single_cube.xml'
    dyn = Dynamics(path)

    home = dyn.model.keyframe('pickup')

    ctrl = home.ctrl

    print(ctrl)
    print(ctrl.shape)

    xt = jnp.zeros((dyn.state_dim,))
    xt = xt.at[:dyn.nq].set(home.qpos)


    U = jnp.zeros((horizon, dyn.control_dim)) + ctrl

    X = unroll(dyn, xt, U)

    dyn.render(X, path = 'arm_still.mp4')
