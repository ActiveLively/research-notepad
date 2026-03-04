import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# Import the standard trajectory and action messages
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class URTrajectoryClient(Node):
    def __init__(self):
        super().__init__('ur_trajectory_client')
        
        # Create an Action Client to talk to the UR scaled trajectory controller
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

    def send_goal(self):
        self.get_logger().info('Waiting for action server to come online...')
        self._action_client.wait_for_server()

        # 1. Initialize the Goal message
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()
        
        # 2. Define the exact joint names for the UR robot
        goal_msg.trajectory.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        # 3. Create a trajectory point (target state)
        point = JointTrajectoryPoint()
        
        # Define target joint angles in radians. 
        # This example moves the arm to a safe "up" position.
        point.positions = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0] 
        
        # Set how long the robot has to reach this point from the start
        point.time_from_start = Duration(sec=5, nanosec=0)

        # 4. Append the point to the trajectory
        goal_msg.trajectory.points.append(point)

        self.get_logger().info('Sending trajectory goal...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by the hardware controller.')
            return

        self.get_logger().info('Goal accepted! Robot should be moving.')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Action finished with error code: {result.error_code}')
        # Error code 0 means SUCCESS
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = URTrajectoryClient()
    action_client.send_goal()
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()