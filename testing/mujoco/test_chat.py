import mujoco
import mujoco.viewer
import importlib.resources
import time

# Locate the bundled model
with importlib.resources.path("mujoco", "model/cartpole.xml") as model_path:
    model = mujoco.MjModel.from_xml_path(str(model_path))

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 5:
        mujoco.mj_step(model, data)
        viewer.sync()