import rtde_receive
import rtde_control

ROBOT_IP = "192.168.56.101"

try:
    # Initialize the interfaces
    # The 'testing' environment you were using earlier is perfect for this
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)

    # Get the current TCP Pose (x, y, z, rx, ry, rz)
    actual_tcp_pose = rtde_r.getActualTCPPose()
    print(f"Connected! Current TCP Pose: {actual_tcp_pose}")

except Exception as e:
    print(f"Could not connect: {e}")