import copy
import math
import threading
import time

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import DriveOnHeading, NavigateToPose, Spin
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_R_JOINT_NAMES = [
    'arm_r_joint1',
    'arm_r_joint2',
    'arm_r_joint3',
    'arm_r_joint4',
    'arm_r_joint5',
    'arm_r_joint6',
    'arm_r_joint7',
]

HOME_ARM_POSITIONS = [0.0] * len(ARM_R_JOINT_NAMES)


def radians(values_degrees):
    return [math.radians(value) for value in values_degrees]


# These positions are the first safe poses already used by this project.
# The arm is driven directly through arm_r_controller, avoiding MoveItPy.
SIDE_ARM_POSITION = radians([0.0, -60.0, 0.0, -20.0, 0.0, 0.0, 0.0])
ELBOW_UP_POSITION = radians([0.0, -60.0, -90.0, -90.0, 0.0, 0.0, 0.0])
WRIST_WAVE_DEGREES = 15.0


class WaveInteraction(Node):
    STATE_IDLE = 'idle'
    STATE_SENDING_ORIGINAL = 'sending_original'
    STATE_NAVIGATING_ORIGINAL = 'navigating_original'
    STATE_CANCELLING_ORIGINAL = 'cancelling_original'
    STATE_SENDING_APPROACH = 'sending_approach'
    STATE_NAVIGATING_APPROACH = 'navigating_approach'
    STATE_ALIGNING = 'aligning_to_person'
    STATE_WAVING = 'waving'
    STATE_RETURNING_HOME = 'returning_home'
    STATE_RESUMING = 'resuming_original'
    STATE_SUCCEEDED = 'succeeded'
    STATE_CANCELLED = 'cancelled'
    STATE_ABORTED = 'aborted'
    STATE_REJECTED = 'rejected'
    STATE_FAILED = 'failed'

    GOAL_ORIGINAL = 'original'
    GOAL_APPROACH = 'approach'
    GOAL_RESUME = 'resume'

    def __init__(self):
        super().__init__('wave_interaction')

        self.callback_group = ReentrantCallbackGroup()
        self.state_lock = threading.RLock()

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('stand_off_distance', 1.0)
        self.declare_parameter('approach_skip_margin', 0.15)
        self.declare_parameter('person_target_max_age', 3.0)
        self.declare_parameter('use_fixed_person_fallback', True)
        self.declare_parameter('fixed_person_x', -1.465)
        self.declare_parameter('fixed_person_y', -0.050)
        self.declare_parameter('fixed_person_yaw', 3.123)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('drive_on_heading_action', '/drive_on_heading')
        self.declare_parameter('straight_approach_speed', 0.75)
        self.declare_parameter('straight_approach_timeout_margin', 25.0)
        self.declare_parameter('spin_action', '/spin')
        self.declare_parameter('face_person_timeout', 15.0)
        self.declare_parameter('face_person_yaw_tolerance', 0.06)
        self.declare_parameter('face_person_max_angular_speed', 0.80)
        self.declare_parameter('face_person_kp', 1.8)
        self.declare_parameter('camera_yaw_offset', 0.0)
        self.declare_parameter('base_linear_stop_threshold', 0.01)
        self.declare_parameter('base_angular_stop_threshold', 0.02)
        self.declare_parameter('base_stop_stable_duration', 0.50)
        self.declare_parameter('base_stop_timeout', 8.0)
        self.declare_parameter('arm_home_tolerance', 0.05)
        self.declare_parameter('arm_home_stable_duration', 0.30)
        self.declare_parameter('arm_home_timeout', 45.0)
        self.declare_parameter('arm_trajectory_timeout', 120.0)
        self.declare_parameter(
            'arm_controller_action',
            '/arm_r_controller/follow_joint_trajectory',
        )

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.stand_off_distance = float(
            self.get_parameter('stand_off_distance').value
        )
        self.approach_skip_margin = float(
            self.get_parameter('approach_skip_margin').value
        )
        self.person_target_max_age = float(
            self.get_parameter('person_target_max_age').value
        )
        self.use_fixed_person_fallback = bool(
            self.get_parameter('use_fixed_person_fallback').value
        )
        self.fixed_person_x = float(
            self.get_parameter('fixed_person_x').value
        )
        self.fixed_person_y = float(
            self.get_parameter('fixed_person_y').value
        )
        self.fixed_person_yaw = float(
            self.get_parameter('fixed_person_yaw').value
        )
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.drive_on_heading_action = self.get_parameter(
            'drive_on_heading_action'
        ).value
        self.straight_approach_speed = float(
            self.get_parameter('straight_approach_speed').value
        )
        self.straight_approach_timeout_margin = float(
            self.get_parameter('straight_approach_timeout_margin').value
        )
        self.spin_action = self.get_parameter('spin_action').value
        self.face_person_timeout = float(
            self.get_parameter('face_person_timeout').value
        )
        self.face_person_yaw_tolerance = float(
            self.get_parameter('face_person_yaw_tolerance').value
        )
        self.face_person_max_angular_speed = float(
            self.get_parameter('face_person_max_angular_speed').value
        )
        self.face_person_kp = float(
            self.get_parameter('face_person_kp').value
        )
        self.camera_yaw_offset = float(
            self.get_parameter('camera_yaw_offset').value
        )
        self.base_linear_stop_threshold = float(
            self.get_parameter('base_linear_stop_threshold').value
        )
        self.base_angular_stop_threshold = float(
            self.get_parameter('base_angular_stop_threshold').value
        )
        self.base_stop_stable_duration = float(
            self.get_parameter('base_stop_stable_duration').value
        )
        self.base_stop_timeout = float(
            self.get_parameter('base_stop_timeout').value
        )
        self.arm_home_tolerance = float(
            self.get_parameter('arm_home_tolerance').value
        )
        self.arm_home_stable_duration = float(
            self.get_parameter('arm_home_stable_duration').value
        )
        self.arm_home_timeout = float(
            self.get_parameter('arm_home_timeout').value
        )
        self.arm_trajectory_timeout = float(
            self.get_parameter('arm_trajectory_timeout').value
        )
        arm_controller_action = self.get_parameter(
            'arm_controller_action'
        ).value

        self.interaction_state = self.STATE_IDLE
        self.active_goal_handle = None
        self.active_goal_kind = None
        self.active_goal_sequence = None
        self.goal_sequence = 0

        self.original_goal_pose = None
        self.interaction_done_for_goal = False
        self.person_present = False
        self.latest_person_target = None
        self.latest_person_target_map = None
        self.latest_person_target_is_fixed = False
        self.interaction_person_target_map = None
        self.interaction_person_target_is_fixed = False
        self.person_target_received_at = None

        self.latest_linear_speed = None
        self.latest_angular_speed = None
        self.latest_joint_positions = {}

        self.worker_thread = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose',
            callback_group=self.callback_group,
        )
        self.spin_client = ActionClient(
            self,
            Spin,
            self.spin_action,
            callback_group=self.callback_group,
        )
        self.drive_on_heading_client = ActionClient(
            self,
            DriveOnHeading,
            self.drive_on_heading_action,
            callback_group=self.callback_group,
        )
        self.arm_trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            arm_controller_action,
            callback_group=self.callback_group,
        )
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            10,
        )
        self.interaction_active_pub = self.create_publisher(
            Bool,
            '/interaction_active',
            10,
        )

        self.create_subscription(
            PoseStamped,
            '/interaction_goal_pose',
            self._external_goal_cb,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Bool,
            '/person_detected',
            self._person_detected_cb,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            PointStamped,
            '/person_target',
            self._person_target_cb,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Bool,
            '/wave_command',
            self._wave_command_cb,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Odometry,
            '/odom',
            self._odom_cb,
            20,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_cb,
            20,
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            'Interaction ready: original goal -> cancel -> drive straight '
            f'to {self.stand_off_distance:.2f} m -> wave -> arm home -> '
            'resume original goal.'
        )

    # ------------------------------------------------------------------
    # Shared state
    # ------------------------------------------------------------------
    def _set_state(self, new_state):
        with self.state_lock:
            old_state = self.interaction_state
            self.interaction_state = new_state

        if old_state != new_state:
            self.get_logger().info(
                f'Interaction state: {old_state} -> {new_state}'
            )

    def _worker_is_running(self):
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def _start_worker(self, target, *args):
        with self.state_lock:
            previous_worker = (
                self.worker_thread
                if self._worker_is_running()
                else None
            )

            if previous_worker is None:
                self.worker_thread = threading.Thread(
                    target=target,
                    args=args,
                    daemon=True,
                )
            else:
                # A goal response may arrive just before the worker that sent
                # that goal exits. Queue the next phase instead of losing it.
                self.worker_thread = threading.Thread(
                    target=self._run_after_worker,
                    args=(previous_worker, target, args),
                    daemon=True,
                )

            self.worker_thread.start()
            return True

    @staticmethod
    def _run_after_worker(previous_worker, target, args):
        if previous_worker is not threading.current_thread():
            previous_worker.join()
        target(*args)

    # ------------------------------------------------------------------
    # Sensor callbacks and safety state
    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        linear = msg.twist.twist.linear
        angular = msg.twist.twist.angular

        with self.state_lock:
            self.latest_linear_speed = math.sqrt(
                linear.x * linear.x
                + linear.y * linear.y
                + linear.z * linear.z
            )
            self.latest_angular_speed = math.sqrt(
                angular.x * angular.x
                + angular.y * angular.y
                + angular.z * angular.z
            )

    def _joint_state_cb(self, msg: JointState):
        with self.state_lock:
            for name, position in zip(msg.name, msg.position):
                self.latest_joint_positions[name] = float(position)

    def _base_is_stopped_now(self):
        with self.state_lock:
            if (
                self.latest_linear_speed is None
                or self.latest_angular_speed is None
            ):
                return False

            return (
                self.latest_linear_speed
                <= self.base_linear_stop_threshold
                and self.latest_angular_speed
                <= self.base_angular_stop_threshold
            )

    def _wait_until_base_stopped(self):
        deadline = time.monotonic() + self.base_stop_timeout
        stable_since = None

        while rclpy.ok() and time.monotonic() < deadline:
            if self._base_is_stopped_now():
                if stable_since is None:
                    stable_since = time.monotonic()
                elif (
                    time.monotonic() - stable_since
                    >= self.base_stop_stable_duration
                ):
                    self.get_logger().info(
                        'Base is confirmed stationary'
                    )
                    return True
            else:
                stable_since = None

            time.sleep(0.05)

        self.get_logger().error(
            'Base did not become stationary before the safety timeout'
        )
        return False

    def _arm_is_home_now(self):
        with self.state_lock:
            if not all(
                name in self.latest_joint_positions
                for name in ARM_R_JOINT_NAMES
            ):
                return False

            return all(
                abs(self.latest_joint_positions[name])
                <= self.arm_home_tolerance
                for name in ARM_R_JOINT_NAMES
            )

    def _wait_until_arm_home(self):
        deadline = time.monotonic() + self.arm_home_timeout
        stable_since = None

        while rclpy.ok() and time.monotonic() < deadline:
            if self._arm_is_home_now():
                if stable_since is None:
                    stable_since = time.monotonic()
                elif (
                    time.monotonic() - stable_since
                    >= self.arm_home_stable_duration
                ):
                    self.get_logger().info(
                        'Right arm is confirmed at the zero/home position'
                    )
                    return True
            else:
                stable_since = None

            time.sleep(0.05)

        self.get_logger().error(
            'Right arm did not reach the zero/home position before timeout'
        )
        return False

    # ------------------------------------------------------------------
    # External Nav2 goal ownership
    # ------------------------------------------------------------------
    def _external_goal_cb(self, msg: PoseStamped):
        with self.state_lock:
            busy_states = {
                self.STATE_SENDING_ORIGINAL,
                self.STATE_NAVIGATING_ORIGINAL,
                self.STATE_CANCELLING_ORIGINAL,
                self.STATE_SENDING_APPROACH,
                self.STATE_NAVIGATING_APPROACH,
                self.STATE_ALIGNING,
                self.STATE_WAVING,
                self.STATE_RETURNING_HOME,
                self.STATE_RESUMING,
            }
            if self.interaction_state in busy_states:
                self.get_logger().warning(
                    'Ignoring a new external goal while interaction state is '
                    f'{self.interaction_state}'
                )
                return

            self.original_goal_pose = copy.deepcopy(msg)
            self.interaction_done_for_goal = False
            self.interaction_person_target_map = None

        self._send_nav_goal(
            copy.deepcopy(msg),
            self.GOAL_ORIGINAL,
        )

    def _send_nav_goal(self, pose: PoseStamped, goal_kind):
        if pose is None or not pose.header.frame_id:
            self.get_logger().error('Cannot send an invalid Nav2 goal')
            self._set_state(self.STATE_FAILED)
            return False

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                'NavigateToPose action server is unavailable'
            )
            self._set_state(self.STATE_FAILED)
            return False

        pose_to_send = copy.deepcopy(pose)
        pose_to_send.header.stamp = self.get_clock().now().to_msg()

        goal = NavigateToPose.Goal()
        goal.pose = pose_to_send

        with self.state_lock:
            self.goal_sequence += 1
            sequence = self.goal_sequence
            self.active_goal_sequence = sequence
            self.active_goal_kind = goal_kind
            self.active_goal_handle = None

            if goal_kind == self.GOAL_APPROACH:
                self.interaction_state = self.STATE_SENDING_APPROACH
            elif goal_kind == self.GOAL_RESUME:
                self.interaction_state = self.STATE_RESUMING
            else:
                self.interaction_state = self.STATE_SENDING_ORIGINAL

        labels = {
            self.GOAL_ORIGINAL: 'original navigation goal',
            self.GOAL_APPROACH: 'person approach goal',
            self.GOAL_RESUME: 'resumed original goal',
        }
        self.get_logger().info(f'Sending {labels[goal_kind]}')

        try:
            future = self.nav_client.send_goal_async(goal)
            future.add_done_callback(
                lambda completed: self._goal_response_cb(
                    completed,
                    sequence,
                    goal_kind,
                )
            )
        except Exception as error:
            self.get_logger().error(
                f'Failed to request Nav2 goal: {error}'
            )
            self._set_state(self.STATE_FAILED)
            return False

        return True

    def _goal_response_cb(self, future, sequence, goal_kind):
        with self.state_lock:
            if sequence != self.active_goal_sequence:
                return

        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(
                f'Failed to receive Nav2 goal response: {error}'
            )
            self._set_state(self.STATE_FAILED)
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f'Nav2 rejected the {goal_kind} goal')
            if goal_kind == self.GOAL_APPROACH:
                self._start_worker(self._safe_resume_without_wave)
            else:
                self._set_state(self.STATE_REJECTED)
            return

        with self.state_lock:
            if sequence != self.active_goal_sequence:
                return

            self.active_goal_handle = goal_handle
            if goal_kind == self.GOAL_APPROACH:
                self.interaction_state = self.STATE_NAVIGATING_APPROACH
            else:
                self.interaction_state = self.STATE_NAVIGATING_ORIGINAL

        self.get_logger().info(f'Nav2 accepted the {goal_kind} goal')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda completed: self._nav_result_cb(
                completed,
                sequence,
                goal_kind,
            )
        )

    def _nav_result_cb(self, future, sequence, goal_kind):
        with self.state_lock:
            if sequence != self.active_goal_sequence:
                return
            state_at_result = self.interaction_state

        try:
            wrapped_result = future.result()
            status = wrapped_result.status
        except Exception as error:
            self.get_logger().error(
                f'Could not retrieve Nav2 result: {error}'
            )
            self._set_state(self.STATE_FAILED)
            return

        with self.state_lock:
            if sequence != self.active_goal_sequence:
                return
            self.active_goal_handle = None

        if status == GoalStatus.STATUS_SUCCEEDED:
            if goal_kind == self.GOAL_APPROACH:
                self.get_logger().info(
                    'Robot reached the person approach position'
                )
                self._set_state(self.STATE_WAVING)
                self._start_worker(self._interaction_wave_worker)
            else:
                self.get_logger().info('Navigation goal succeeded')
                self._set_state(self.STATE_SUCCEEDED)
            return

        if status == GoalStatus.STATUS_CANCELED:
            if (
                goal_kind in {self.GOAL_ORIGINAL, self.GOAL_RESUME}
                and state_at_result == self.STATE_CANCELLING_ORIGINAL
            ):
                self.get_logger().info(
                    'Original goal is fully cancelled; calculating the '
                    f'{self.stand_off_distance:.2f} m person approach goal'
                )
                self._set_state(self.STATE_SENDING_APPROACH)
                self._start_worker(self._approach_person_worker)
            else:
                self.get_logger().warning(
                    f'{goal_kind} goal finished as cancelled'
                )
                self._set_state(self.STATE_CANCELLED)
            return

        if status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error(f'{goal_kind} goal was aborted')
            if goal_kind == self.GOAL_APPROACH:
                self._start_worker(self._safe_resume_without_wave)
            else:
                self._set_state(self.STATE_ABORTED)
            return

        self.get_logger().error(
            f'{goal_kind} goal ended with unexpected status {status}'
        )
        self._set_state(self.STATE_FAILED)

    # ------------------------------------------------------------------
    # Detection -> cancel -> approach
    # ------------------------------------------------------------------
    def _person_detected_cb(self, msg: Bool):
        detected = bool(msg.data)
        with self.state_lock:
            self.person_present = detected

            # The simulated standing person is static. If RGB detection is
            # valid but no depth-based /person_target has arrived, use the
            # configured map position so detection still triggers approach.
            if (
                detected
                and self.use_fixed_person_fallback
                and self.interaction_state == self.STATE_NAVIGATING_ORIGINAL
                and not self.interaction_done_for_goal
                and not self._target_is_fresh_locked()
            ):
                fallback = PointStamped()
                fallback.header.frame_id = self.map_frame
                fallback.header.stamp = self.get_clock().now().to_msg()
                fallback.point.x = self.fixed_person_x
                fallback.point.y = self.fixed_person_y
                fallback.point.z = 0.0
                self.latest_person_target_map = fallback
                self.latest_person_target_is_fixed = True
                self.person_target_received_at = time.monotonic()
                self.get_logger().warning(
                    'No fresh depth target; using configured standing-person '
                    f'position ({self.fixed_person_x:.3f}, '
                    f'{self.fixed_person_y:.3f}) in map'
                )

        self._try_start_interaction()

    def _person_target_cb(self, msg: PointStamped):
        # Convert the detected 3D point to map coordinates immediately.
        # Once interaction starts, this map point is frozen so losing the
        # camera detection while turning does not cancel the interaction.
        target_map = self._person_target_to_map(msg)
        if target_map is None:
            return

        with self.state_lock:
            self.latest_person_target = copy.deepcopy(msg)
            self.latest_person_target_map = target_map
            self.latest_person_target_is_fixed = False
            self.person_target_received_at = time.monotonic()

        self._try_start_interaction()

    def _person_target_to_map(self, msg: PointStamped):
        if not msg.header.frame_id:
            return None

        if msg.header.frame_id == self.map_frame:
            result = copy.deepcopy(msg)
            result.header.frame_id = self.map_frame
            result.header.stamp = self.get_clock().now().to_msg()
            return result

        try:
            source_to_map = self.tf_buffer.lookup_transform(
                self.map_frame,
                msg.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.30),
            )
        except TransformException as error:
            self.get_logger().warning(
                f'Cannot transform person target to map yet: {error}'
            )
            return None

        x, y, z = self._transform_point(
            msg.point.x,
            msg.point.y,
            msg.point.z,
            source_to_map,
        )

        result = PointStamped()
        result.header.frame_id = self.map_frame
        result.header.stamp = self.get_clock().now().to_msg()
        result.point.x = x
        result.point.y = y
        result.point.z = z
        return result

    def _target_is_fresh_locked(self):
        return (
            self.latest_person_target_map is not None
            and self.person_target_received_at is not None
            and (
                time.monotonic() - self.person_target_received_at
                <= self.person_target_max_age
            )
        )

    def _try_start_interaction(self):
        with self.state_lock:
            if not self.person_present:
                return
            if self.interaction_done_for_goal:
                return
            if self.interaction_state != self.STATE_NAVIGATING_ORIGINAL:
                return
            if self.active_goal_handle is None:
                return
            if not self._target_is_fresh_locked():
                return

            # Freeze the person's map position before cancelling Nav2.
            self.interaction_person_target_map = copy.deepcopy(
                self.latest_person_target_map
            )
            self.interaction_person_target_is_fixed = (
                self.latest_person_target_is_fixed
            )
            self.interaction_done_for_goal = True
            self.interaction_state = self.STATE_CANCELLING_ORIGINAL
            goal_handle = self.active_goal_handle
            sequence = self.active_goal_sequence

        self.get_logger().info(
            'Person detected and position latched; cancelling original goal'
        )
        self.interaction_active_pub.publish(Bool(data=True))

        try:
            cancel_future = goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(
                lambda completed: self._after_cancel_request(
                    completed,
                    sequence,
                )
            )
        except Exception as error:
            self.get_logger().error(
                f'Failed to request Nav2 cancellation: {error}'
            )
            with self.state_lock:
                if sequence == self.active_goal_sequence:
                    self.interaction_done_for_goal = False
                    self.interaction_person_target_map = None
                    self.interaction_person_target_is_fixed = False
                    self.interaction_state = self.STATE_NAVIGATING_ORIGINAL

    def _after_cancel_request(self, future, sequence):
        with self.state_lock:
            if sequence != self.active_goal_sequence:
                return

        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(
                f'Nav2 cancellation request failed: {error}'
            )
            with self.state_lock:
                if sequence == self.active_goal_sequence:
                    self.interaction_done_for_goal = False
                    self.interaction_state = self.STATE_NAVIGATING_ORIGINAL
            return

        if response is None or not response.goals_canceling:
            self.get_logger().warning(
                'Nav2 did not accept cancellation of the original goal'
            )
            with self.state_lock:
                if sequence == self.active_goal_sequence:
                    self.interaction_done_for_goal = False
                    self.interaction_state = self.STATE_NAVIGATING_ORIGINAL
            return

        self.get_logger().info(
            'Cancellation accepted; waiting for Nav2 terminal CANCELLED result'
        )

    def _approach_person_worker(self):
        """Drive straight ahead from the detection heading; never turn."""
        straight_distance = self._straight_approach_distance()

        if straight_distance is None:
            self.get_logger().error(
                'Could not calculate the straight person approach distance'
            )
            self._safe_resume_without_wave()
            return

        if straight_distance <= self.approach_skip_margin:
            self.get_logger().info(
                'Robot is already within the configured 1 m stand-off distance'
            )
            self._set_state(self.STATE_WAVING)
            self._interaction_wave_worker()
            return

        self._set_state(self.STATE_NAVIGATING_APPROACH)
        if not self._drive_straight(straight_distance):
            self.get_logger().error('Straight approach to person failed')
            self._safe_resume_without_wave()
            return

        self.get_logger().info(
            'Robot completed the straight approach without changing heading'
        )
        self._stop_base_command()
        self._set_state(self.STATE_WAVING)
        self._interaction_wave_worker()

    def _straight_approach_distance(self):
        with self.state_lock:
            person_target = copy.deepcopy(self.interaction_person_target_map)

        if person_target is None:
            return None

        try:
            base_to_map = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.50),
            )
        except TransformException as error:
            self.get_logger().warning(
                f'Cannot calculate straight approach distance: {error}'
            )
            return None

        robot_x = base_to_map.transform.translation.x
        robot_y = base_to_map.transform.translation.y
        person_distance = math.hypot(
            person_target.point.x - robot_x,
            person_target.point.y - robot_y,
        )

        if not math.isfinite(person_distance):
            return None

        straight_distance = max(
            0.0,
            person_distance - self.stand_off_distance,
        )
        self.get_logger().info(
            f'Person distance={person_distance:.2f} m; driving straight '
            f'{straight_distance:.2f} m at {self.straight_approach_speed:.2f} m/s '
            'with zero commanded rotation'
        )
        return straight_distance

    def _drive_straight(self, distance):
        if not self.drive_on_heading_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                'Nav2 DriveOnHeading action server is unavailable'
            )
            return False

        goal = DriveOnHeading.Goal()
        goal.target.x = float(distance)
        goal.target.y = 0.0
        goal.target.z = 0.0
        goal.speed = float(self.straight_approach_speed)

        allowance = max(
            10.0,
            distance / max(self.straight_approach_speed, 0.05)
            + self.straight_approach_timeout_margin,
        )
        whole = int(allowance)
        goal.time_allowance.sec = whole
        goal.time_allowance.nanosec = int((allowance - whole) * 1.0e9)

        send_future = self.drive_on_heading_client.send_goal_async(goal)
        if not self._wait_for_future(send_future, 5.0):
            self.get_logger().error('Timed out sending straight-drive goal')
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Nav2 rejected the straight-drive goal')
            return False

        result_future = goal_handle.get_result_async()
        if not self._wait_for_future(result_future, allowance + 5.0):
            self.get_logger().error('Straight-drive action timed out')
            cancel_future = goal_handle.cancel_goal_async()
            self._wait_for_future(cancel_future, 2.0)
            self._stop_base_command()
            return False

        wrapped_result = result_future.result()
        if wrapped_result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                'Straight-drive action failed with status '
                f'{wrapped_result.status}'
            )
            self._stop_base_command()
            return False

        return True

    def _build_approach_pose_with_retries(self):
        for _ in range(20):
            result = self._build_approach_pose()
            if result[0] is not None:
                return result
            time.sleep(0.10)
        return None, None

    def _build_approach_pose(self):
        with self.state_lock:
            # Use the person position frozen at first detection. Continuous
            # camera detection is not required during cancellation/approach.
            person_target = copy.deepcopy(
                self.interaction_person_target_map
            )
            target_is_fixed = self.interaction_person_target_is_fixed

        if person_target is None:
            return None, None

        try:
            base_to_map = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.50),
            )
        except TransformException as error:
            self.get_logger().warning(
                f'Waiting for robot base TF: {error}'
            )
            return None, None

        person_x = person_target.point.x
        person_y = person_target.point.y
        person_z = person_target.point.z

        robot_x = base_to_map.transform.translation.x
        robot_y = base_to_map.transform.translation.y

        delta_x = person_x - robot_x
        delta_y = person_y - robot_y
        distance = math.hypot(delta_x, delta_y)

        if not math.isfinite(distance) or distance < 0.05:
            return None, None

        if target_is_fixed:
            # The standing person's yaw is known. Approach one metre in front
            # of the person rather than stopping behind them.
            approach_x = (
                person_x
                + self.stand_off_distance * math.cos(self.fixed_person_yaw)
            )
            approach_y = (
                person_y
                + self.stand_off_distance * math.sin(self.fixed_person_yaw)
            )
        else:
            # For a live depth target, approach along the current robot-person
            # line while keeping the configured stand-off distance.
            approach_x = (
                person_x
                - self.stand_off_distance * delta_x / distance
            )
            approach_y = (
                person_y
                - self.stand_off_distance * delta_y / distance
            )

        approach_yaw = math.atan2(
            person_y - approach_y,
            person_x - approach_x,
        )
        distance_to_goal = math.hypot(
            approach_x - robot_x,
            approach_y - robot_y,
        )

        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = approach_x
        pose.pose.position.y = approach_y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(approach_yaw / 2.0)
        pose.pose.orientation.w = math.cos(approach_yaw / 2.0)

        approach_type = (
            'fixed front approach'
            if target_is_fixed
            else 'live-target approach'
        )
        self.get_logger().info(
            'Latched person target in map: '
            f'({person_x:.2f}, {person_y:.2f}, {person_z:.2f}); '
            f'{approach_type}; '
            f'approach goal=({approach_x:.2f}, {approach_y:.2f}); '
            f'robot-to-goal distance={distance_to_goal:.2f} m'
        )

        return pose, distance_to_goal

    @staticmethod
    def _transform_point(x, y, z, transform_stamped):
        transform = transform_stamped.transform
        q = transform.rotation

        # Quaternion rotation matrix.
        r00 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        r01 = 2.0 * (q.x * q.y - q.z * q.w)
        r02 = 2.0 * (q.x * q.z + q.y * q.w)
        r10 = 2.0 * (q.x * q.y + q.z * q.w)
        r11 = 1.0 - 2.0 * (q.x * q.x + q.z * q.z)
        r12 = 2.0 * (q.y * q.z - q.x * q.w)
        r20 = 2.0 * (q.x * q.z - q.y * q.w)
        r21 = 2.0 * (q.y * q.z + q.x * q.w)
        r22 = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)

        return (
            transform.translation.x + r00 * x + r01 * y + r02 * z,
            transform.translation.y + r10 * x + r11 * y + r12 * z,
            transform.translation.z + r20 * x + r21 * y + r22 * z,
        )

    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _yaw_from_quaternion(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _stop_base_command(self):
        self.cmd_vel_pub.publish(Twist())

    def _current_face_error(self, person_target):
        try:
            base_to_map = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.50),
            )
        except TransformException as error:
            self.get_logger().warning(
                f'Cannot calculate person-facing angle: {error}'
            )
            return None

        robot_x = base_to_map.transform.translation.x
        robot_y = base_to_map.transform.translation.y
        robot_yaw = self._yaw_from_quaternion(
            base_to_map.transform.rotation
        )
        desired_yaw = math.atan2(
            person_target.point.y - robot_y,
            person_target.point.x - robot_x,
        ) - self.camera_yaw_offset

        # Always return the shortest signed rotation in [-pi, pi].
        return self._normalize_angle(desired_yaw - robot_yaw)

    def _face_person(self):
        """Use Nav2 Spin so the base takes the shortest turn to the person."""
        with self.state_lock:
            person_target = copy.deepcopy(self.interaction_person_target_map)

        if person_target is None:
            self.get_logger().error('Cannot face person: no latched target')
            return False

        yaw_error = self._current_face_error(person_target)
        if yaw_error is None:
            return False

        if abs(yaw_error) <= self.face_person_yaw_tolerance:
            self.get_logger().info('Robot already faces the person')
            return True

        if not self.spin_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 Spin action server is unavailable')
            return False

        direction = 'left' if yaw_error > 0.0 else 'right'
        self.get_logger().info(
            f'Facing person with shortest {direction} turn: '
            f'{math.degrees(abs(yaw_error)):.1f} deg'
        )

        goal = Spin.Goal()
        goal.target_yaw = float(yaw_error)
        timeout_whole = int(self.face_person_timeout)
        goal.time_allowance.sec = timeout_whole
        goal.time_allowance.nanosec = int(
            (self.face_person_timeout - timeout_whole) * 1.0e9
        )

        send_future = self.spin_client.send_goal_async(goal)
        if not self._wait_for_future(send_future, 5.0):
            self.get_logger().error('Timed out sending shortest-turn goal')
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Nav2 rejected shortest-turn goal')
            return False

        result_future = goal_handle.get_result_async()
        if not self._wait_for_future(
            result_future,
            self.face_person_timeout + 5.0,
        ):
            self.get_logger().error('Shortest-turn action timed out')
            cancel_future = goal_handle.cancel_goal_async()
            self._wait_for_future(cancel_future, 2.0)
            return False

        wrapped_result = result_future.result()
        if wrapped_result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                'Shortest-turn action failed with status '
                f'{wrapped_result.status}'
            )
            return False

        time.sleep(0.20)
        final_error = self._current_face_error(person_target)
        if final_error is None:
            return False

        if abs(final_error) > self.face_person_yaw_tolerance:
            self.get_logger().warning(
                'Spin completed but final facing error is '
                f'{math.degrees(abs(final_error)):.1f} deg'
            )
            return False

        self.get_logger().info('Robot is facing the person; starting wave')
        return True

    def _face_person_then_wave_worker(self):
        # Deliberately preserve the heading from first detection.
        self.get_logger().info(
            'Skipping alignment spin; preserving detection heading'
        )
        self._stop_base_command()
        self._set_state(self.STATE_WAVING)
        self._interaction_wave_worker()

    # ------------------------------------------------------------------
    # Wave and safety sequencing
    # ------------------------------------------------------------------
    def _wave_command_cb(self, msg: Bool):
        if not msg.data:
            return

        with self.state_lock:
            if self._worker_is_running():
                return
            if self.interaction_state in {
                self.STATE_SENDING_ORIGINAL,
                self.STATE_NAVIGATING_ORIGINAL,
                self.STATE_CANCELLING_ORIGINAL,
                self.STATE_SENDING_APPROACH,
                self.STATE_NAVIGATING_APPROACH,
                self.STATE_ALIGNING,
                self.STATE_RESUMING,
            }:
                self.get_logger().warning(
                    'Standalone wave rejected while navigation is active'
                )
                return

        self._set_state(self.STATE_WAVING)
        self._start_worker(self._standalone_wave_worker)

    def _standalone_wave_worker(self):
        if not self._wait_until_base_stopped():
            self._set_state(self.STATE_FAILED)
            return

        success = self.do_wave()
        if success and self._wait_until_arm_home():
            self._set_state(self.STATE_IDLE)
            self.get_logger().info('Standalone wave completed safely')
        else:
            self._set_state(self.STATE_FAILED)

    def _interaction_wave_worker(self):
        if not self._wait_until_base_stopped():
            # The arm is still down, so resuming navigation is safe.
            self._safe_resume_without_wave()
            return

        wave_success = self.do_wave()
        home_verified = self._wait_until_arm_home()
        for attempt in range(1, 4):
            if home_verified:
                break
            self.get_logger().warning(
                f'Arm is not home; recovery attempt {attempt} of 3'
            )
            self._set_state(self.STATE_RETURNING_HOME)
            self._command_arm_home()
            home_verified = self._wait_until_arm_home()
        self._set_state(self.STATE_WAVING)
        base_still_stopped = self._wait_until_base_stopped()

        if not wave_success:
            self.get_logger().error('Wave sequence failed')
        if not home_verified:
            self.get_logger().error(
                'Navigation will not resume because the arm is not home'
            )
        if not base_still_stopped:
            self.get_logger().error(
                'Navigation will not resume because the base safety check failed'
            )

        if home_verified and base_still_stopped:
            if wave_success:
                self.get_logger().info(
                    'Wave completed, arm is at zero, and base is stationary; '
                    'resuming the original goal'
                )
            else:
                self.get_logger().warning(
                    'Wave controller did not report success, but the arm is home '
                    'and the base is stationary; resuming the original goal'
                )
            self._resume_original_goal()
        else:
            self._set_state(self.STATE_FAILED)

    def do_wave(self):
        """Execute one fast arm trajectory that ends with all joints at zero."""
        if not self._base_is_stopped_now():
            self.get_logger().error(
                'Refusing to raise the arm because the base is moving'
            )
            return False

        current_positions = self._current_arm_positions()
        if current_positions is None:
            self.get_logger().error(
                'Cannot wave because right-arm joint states are unavailable'
            )
            return False

        if not self.arm_trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                'Right-arm FollowJointTrajectory server is unavailable'
            )
            return False

        trajectory = JointTrajectory()
        trajectory.joint_names = list(ARM_R_JOINT_NAMES)

        # Current pose -> arm out -> elbow up -> two wrist cycles ->
        # all seven joints exactly zero. Total commanded duration: 3.20 seconds.
        wave_left = list(ELBOW_UP_POSITION)
        wave_right = list(ELBOW_UP_POSITION)
        wave_left[5] = math.radians(-WRIST_WAVE_DEGREES)
        wave_right[5] = math.radians(WRIST_WAVE_DEGREES)

        # Fast trajectory kept within the configured 5 rad/s velocity and
        # 5 rad/s^2 acceleration ceilings.
        sequence = [
            (0.05, current_positions),
            (0.55, SIDE_ARM_POSITION),
            (1.15, ELBOW_UP_POSITION),
            (1.40, wave_left),
            (1.65, wave_right),
            (1.90, wave_left),
            (2.15, wave_right),
            (2.40, ELBOW_UP_POSITION),
            (3.20, HOME_ARM_POSITIONS),
        ]

        for seconds, positions in sequence:
            trajectory.points.append(
                self._trajectory_point(positions, seconds)
            )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance.sec = 30

        self.get_logger().info(
            'Executing fast maximum-safe wave and returning the arm to zero'
        )

        send_future = self.arm_trajectory_client.send_goal_async(goal)
        if not self._wait_for_future(send_future, 3.0):
            self.get_logger().error('Timed out sending arm trajectory')
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Arm trajectory was rejected')
            return False

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + self.arm_trajectory_timeout
        base_moved = False

        while rclpy.ok() and time.monotonic() < deadline:
            if result_future.done():
                break

            if not self._base_is_stopped_now():
                base_moved = True
                self.get_logger().error(
                    'Base movement detected while arm was raised; '
                    'cancelling arm trajectory'
                )
                cancel_future = goal_handle.cancel_goal_async()
                self._wait_for_future(cancel_future, 1.0)
                break

            time.sleep(0.02)

        if base_moved:
            self._command_arm_home()
            return False

        if not result_future.done():
            self.get_logger().error('Arm trajectory timed out')
            cancel_future = goal_handle.cancel_goal_async()
            self._wait_for_future(cancel_future, 1.0)
            self._command_arm_home()
            return False

        wrapped_result = result_future.result()
        if wrapped_result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                f'Arm trajectory failed with status {wrapped_result.status}'
            )
            self._command_arm_home()
            return False

        controller_result = wrapped_result.result
        if (
            controller_result.error_code
            != FollowJointTrajectory.Result.SUCCESSFUL
        ):
            self.get_logger().error(
                'Arm controller returned error code '
                f'{controller_result.error_code}: '
                f'{controller_result.error_string}'
            )
            self._command_arm_home()
            return False

        self.get_logger().info(
            'Wave trajectory completed and commanded the arm to zero'
        )
        return True

    def _current_arm_positions(self):
        with self.state_lock:
            if not all(
                name in self.latest_joint_positions
                for name in ARM_R_JOINT_NAMES
            ):
                return None

            return [
                self.latest_joint_positions[name]
                for name in ARM_R_JOINT_NAMES
            ]

    @staticmethod
    def _trajectory_point(positions, seconds):
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]

        whole_seconds = int(seconds)
        point.time_from_start.sec = whole_seconds
        point.time_from_start.nanosec = int(
            (seconds - whole_seconds) * 1.0e9
        )
        return point

    def _command_arm_home(self):
        """Send a separate recovery trajectory to all-zero arm position."""
        current_positions = self._current_arm_positions()
        if current_positions is None:
            self.get_logger().error(
                'Cannot command arm home because joint states are unavailable'
            )
            return False

        if not self.arm_trajectory_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(
                'Cannot command arm home: controller is unavailable'
            )
            return False

        trajectory = JointTrajectory()
        trajectory.joint_names = list(ARM_R_JOINT_NAMES)
        trajectory.points = [
            self._trajectory_point(current_positions, 0.10),
            self._trajectory_point(HOME_ARM_POSITIONS, 1.60),
        ]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance.sec = 10

        send_future = self.arm_trajectory_client.send_goal_async(goal)
        if not self._wait_for_future(send_future, 3.0):
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        if not self._wait_for_future(result_future, 60.0):
            return False

        wrapped_result = result_future.result()
        return (
            wrapped_result.status == GoalStatus.STATUS_SUCCEEDED
            and wrapped_result.result.error_code
            == FollowJointTrajectory.Result.SUCCESSFUL
        )

    @staticmethod
    def _wait_for_future(future, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        while rclpy.ok() and time.monotonic() < deadline:
            if future.done():
                return True
            time.sleep(0.02)
        return future.done()

    # ------------------------------------------------------------------
    # Resume behavior
    # ------------------------------------------------------------------
    def _safe_resume_without_wave(self):
        self.get_logger().warning(
            'Approach/wave could not be completed; the arm is down, so the '
            'original goal will be resumed safely'
        )

        if not self._arm_is_home_now():
            self._set_state(self.STATE_RETURNING_HOME)
            self._command_arm_home()

        if not self._wait_until_arm_home():
            self._set_state(self.STATE_FAILED)
            return

        self._resume_original_goal()

    def _resume_original_goal(self):
        self.interaction_active_pub.publish(Bool(data=False))
        with self.state_lock:
            original_pose = copy.deepcopy(self.original_goal_pose)

        if original_pose is None:
            self.get_logger().error(
                'No saved original goal exists to resume'
            )
            self._set_state(self.STATE_FAILED)
            return

        if not self._send_nav_goal(original_pose, self.GOAL_RESUME):
            self._set_state(self.STATE_FAILED)


def main(args=None):
    rclpy.init(args=args)

    node = None
    executor = MultiThreadedExecutor(num_threads=4)

    try:
        node = WaveInteraction()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(
                f'Wave interaction node crashed: {error}'
            )
        else:
            print(f'Wave interaction node failed to start: {error}')
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
