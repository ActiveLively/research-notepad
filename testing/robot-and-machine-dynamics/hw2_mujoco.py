import mujoco
import mujoco.viewer
import numpy as np
import time

xml_path = "franka_emika_panda/panda.xml" 
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

sequence = [
    ("joint2", 0, 1.57),
    ("joint2", 1.57, 0.785),

    ("joint1", 0, 2.89), 
    ("joint1", 2.89, 0),

    ("joint3", -0.5, 0.5),
    ("joint3", 0.5, 0.785),

    ("joint5", 0, 2.89),
    ("joint5", 2.89, 0.785),

    ("joint4", -0.1, -3.0),
    ("joint4", -3.0, -0.1),

    ("joint6", 0, 3.75),
    ("joint6", 3.75, 0)
]

print("Starting MuJoCo Animation...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    
    while viewer.is_running():
        
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        print("Starting sequence...")
        
        reset_triggered = False
        
        for joint_name, start, end in sequence:
            if reset_triggered: break 
            
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            q_addr = model.jnt_qposadr[joint_id]
            steps = np.linspace(start, end, 100)
            
            for val in steps:
                if not viewer.is_running(): break
                
                data.time += 0.01 

                if data.time == 0:
                    print("Reset clicked! Restarting...")
                    reset_triggered = True
                    break 
                
                data.qpos[q_addr] = val
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.01)

        if not reset_triggered:
            time.sleep(1)