import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool


class PersonDetectorNode(Node):
    """Detect the nearest visible person and publish a depth-based 3D target."""

    def __init__(self):
        super().__init__('person_detector_node')

        self.declare_parameter('camera_topic', '/zedm/image')
        self.declare_parameter('depth_topic', '/zedm/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/zedm/camera_info')
        self.declare_parameter(
            'optical_frame',
            'zedm_left_camera_optical_frame',
        )
        self.declare_parameter('weights_path', 'yolov4-tiny.weights')
        self.declare_parameter('config_path', 'yolov4-tiny.cfg')
        self.declare_parameter('names_path', 'coco.names')
        self.declare_parameter('confidence_threshold', 0.50)
        self.declare_parameter('nms_threshold', 0.40)
        self.declare_parameter('detection_hold_frames', 3)
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('process_every_n_frames', 2)
        self.declare_parameter('network_input_size', 320)
        self.declare_parameter('depth_patch_radius', 10)
        self.declare_parameter('minimum_depth', 0.40)
        self.declare_parameter('maximum_depth', 8.00)
        self.declare_parameter('maximum_depth_age', 0.50)

        self.camera_topic = self.get_parameter('camera_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter(
            'camera_info_topic'
        ).value
        self.optical_frame = self.get_parameter('optical_frame').value
        weights_path = self.get_parameter('weights_path').value
        config_path = self.get_parameter('config_path').value
        names_path = self.get_parameter('names_path').value

        self.confidence_threshold = float(
            self.get_parameter('confidence_threshold').value
        )
        self.nms_threshold = float(
            self.get_parameter('nms_threshold').value
        )
        self.hold_frames = max(
            1,
            int(self.get_parameter('detection_hold_frames').value),
        )
        self.publish_annotated = bool(
            self.get_parameter('publish_annotated').value
        )
        self.process_every_n_frames = max(
            1,
            int(self.get_parameter('process_every_n_frames').value),
        )
        self.network_input_size = int(
            self.get_parameter('network_input_size').value
        )
        self.depth_patch_radius = max(
            1,
            int(self.get_parameter('depth_patch_radius').value),
        )
        self.minimum_depth = float(
            self.get_parameter('minimum_depth').value
        )
        self.maximum_depth = float(
            self.get_parameter('maximum_depth').value
        )
        self.maximum_depth_age = float(
            self.get_parameter('maximum_depth_age').value
        )

        if self.network_input_size % 32 != 0:
            raise ValueError('network_input_size must be divisible by 32')

        with open(names_path, 'r', encoding='utf-8') as names_file:
            self.classes = [
                line.strip()
                for line in names_file.readlines()
                if line.strip()
            ]

        if 'person' not in self.classes:
            raise RuntimeError('The COCO names file does not contain person')
        self.person_class_id = self.classes.index('person')

        self.net = cv2.dnn.readNetFromDarknet(
            config_path,
            weights_path,
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.output_layers = self.net.getUnconnectedOutLayersNames()

        self.bridge = CvBridge()
        self.latest_depth = None
        self.latest_depth_encoding = None
        self.latest_depth_stamp = None
        self.latest_camera_info = None

        self.frame_counter = 0
        self.consecutive_detections = 0
        self.consecutive_misses = 0
        self.person_present = False
        self.interaction_active = False

        self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool,
            '/interaction_active',
            self.interaction_active_callback,
            10,
        )

        self.detection_pub = self.create_publisher(
            Bool,
            '/person_detected',
            10,
        )
        self.target_pub = self.create_publisher(
            PointStamped,
            '/person_target',
            10,
        )
        self.annotated_pub = None
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(
                Image,
                '/person_detection/annotated',
                10,
            )

        self.get_logger().info(
            'Person detector ready. '
            f'RGB={self.camera_topic}, depth={self.depth_topic}, '
            f'camera_info={self.camera_info_topic}'
        )

    @staticmethod
    def stamp_to_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9

    def interaction_active_callback(self, msg: Bool):
        self.interaction_active = bool(msg.data)
        if self.interaction_active:
            self.person_present = True
            self.consecutive_misses = 0

    def depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough',
            )
        except CvBridgeError as error:
            self.get_logger().warning(
                f'Could not convert depth image: {error}'
            )
            return

        if depth is None or depth.ndim != 2:
            return

        self.latest_depth = np.asarray(depth)
        self.latest_depth_encoding = msg.encoding
        self.latest_depth_stamp = msg.header.stamp

    def camera_info_callback(self, msg: CameraInfo):
        self.latest_camera_info = msg

    def image_callback(self, msg: Image):
        self.frame_counter += 1
        if self.frame_counter % self.process_every_n_frames != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )
        except CvBridgeError as error:
            self.get_logger().warning(
                f'Could not convert RGB image: {error}'
            )
            return

        image_height, image_width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(self.network_input_size, self.network_input_size),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                class_probability = float(scores[class_id])
                objectness = float(detection[4])
                confidence = objectness * class_probability

                if (
                    class_id != self.person_class_id
                    or confidence < self.confidence_threshold
                ):
                    continue

                center_x, center_y, width, height = (
                    detection[0:4]
                    * np.array([
                        image_width,
                        image_height,
                        image_width,
                        image_height,
                    ])
                ).astype(int)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([
                    x,
                    y,
                    int(width),
                    int(height),
                ])
                confidences.append(confidence)

        raw_indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold,
        )
        indices = np.asarray(raw_indices).reshape(-1).tolist()
        person_found_this_frame = len(indices) > 0

        if person_found_this_frame:
            self.consecutive_detections += 1
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1
            self.consecutive_detections = 0

        if (
            not self.person_present
            and self.consecutive_detections >= self.hold_frames
        ):
            self.person_present = True
            self.get_logger().info('Person detected')
        elif (
            self.person_present
            and not self.interaction_active
            and self.consecutive_misses >= self.hold_frames
        ):
            self.person_present = False
            self.get_logger().info('Person no longer detected')

        self.detection_pub.publish(Bool(data=self.person_present))

        selected_index = None
        selected_depth = None
        selected_center = None

        if indices:
            depth_candidates = []
            for index in indices:
                depth, center = self.get_person_depth(
                    msg,
                    boxes[index],
                    image_width,
                    image_height,
                )
                if depth is not None and center is not None:
                    depth_candidates.append((depth, index, center))

            if depth_candidates:
                # Interact with the nearest person that has a valid depth.
                selected_depth, selected_index, selected_center = min(
                    depth_candidates,
                    key=lambda candidate: candidate[0],
                )
            else:
                # Keep a visible selection for annotation, but do not publish
                # a navigation target until valid depth is available.
                selected_index = max(
                    indices,
                    key=lambda index: boxes[index][2] * boxes[index][3],
                )

        if (
            self.person_present
            and selected_index is not None
            and selected_depth is not None
            and selected_center is not None
        ):
            target = self.make_target_point(
                msg,
                selected_center,
                selected_depth,
                image_width,
                image_height,
            )
            if target is not None:
                self.target_pub.publish(target)

        if self.publish_annotated and self.annotated_pub is not None:
            annotated = frame.copy()
            for index in indices:
                x, y, width, height = boxes[index]
                color = (
                    (0, 255, 255)
                    if index == selected_index
                    else (0, 255, 0)
                )
                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + width, y + height),
                    color,
                    2,
                )
                label = f'person {confidences[index]:.2f}'
                if index == selected_index and selected_depth is not None:
                    label += f'  {selected_depth:.2f} m'
                cv2.putText(
                    annotated,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            annotated_msg = self.bridge.cv2_to_imgmsg(
                annotated,
                encoding='bgr8',
            )
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

    def get_person_depth(
        self,
        rgb_msg,
        box,
        rgb_width,
        rgb_height,
    ):
        if self.latest_depth is None or self.latest_depth_stamp is None:
            return None, None

        rgb_time = self.stamp_to_seconds(rgb_msg.header.stamp)
        depth_time = self.stamp_to_seconds(self.latest_depth_stamp)
        if (
            rgb_time > 0.0
            and depth_time > 0.0
            and abs(rgb_time - depth_time) > self.maximum_depth_age
        ):
            return None, None

        depth_height, depth_width = self.latest_depth.shape[:2]
        x, y, width, height = box

        rgb_u = int(np.clip(x + width / 2, 0, rgb_width - 1))
        rgb_v = int(np.clip(y + height / 2, 0, rgb_height - 1))

        depth_u = int(round(rgb_u * depth_width / rgb_width))
        depth_v = int(round(rgb_v * depth_height / rgb_height))
        depth_u = int(np.clip(depth_u, 0, depth_width - 1))
        depth_v = int(np.clip(depth_v, 0, depth_height - 1))

        radius = self.depth_patch_radius
        u_min = max(0, depth_u - radius)
        u_max = min(depth_width, depth_u + radius + 1)
        v_min = max(0, depth_v - radius)
        v_max = min(depth_height, depth_v + radius + 1)

        patch = self.latest_depth[v_min:v_max, u_min:u_max]
        if patch.size == 0:
            return None, None

        patch_meters = patch.astype(np.float32)
        encoding = (self.latest_depth_encoding or '').upper()
        if encoding in {'16UC1', 'MONO16'}:
            patch_meters *= 0.001

        valid = patch_meters[
            np.isfinite(patch_meters)
            & (patch_meters >= self.minimum_depth)
            & (patch_meters <= self.maximum_depth)
        ]
        if valid.size < 5:
            return None, None

        return float(np.median(valid)), (depth_u, depth_v)

    def make_target_point(
        self,
        rgb_msg,
        depth_center,
        depth_meters,
        rgb_width,
        rgb_height,
    ):
        depth_u, depth_v = depth_center
        depth_height, depth_width = self.latest_depth.shape[:2]

        camera_info = self.latest_camera_info
        if camera_info is not None and camera_info.k[0] > 0.0:
            info_width = camera_info.width or depth_width
            info_height = camera_info.height or depth_height
            scale_x = depth_width / float(info_width)
            scale_y = depth_height / float(info_height)
            fx = camera_info.k[0] * scale_x
            fy = camera_info.k[4] * scale_y
            cx = camera_info.k[2] * scale_x
            cy = camera_info.k[5] * scale_y
        else:
            # Fallback values match the simulated ZED RGB-D intrinsics.
            fx = 525.0 * depth_width / 640.0
            fy = 525.0 * depth_height / 360.0
            cx = 320.0 * depth_width / 640.0
            cy = 180.0 * depth_height / 360.0

        if fx <= 0.0 or fy <= 0.0:
            return None

        point = PointStamped()
        point.header.stamp = rgb_msg.header.stamp
        point.header.frame_id = self.optical_frame
        point.point.x = (float(depth_u) - cx) * depth_meters / fx
        point.point.y = (float(depth_v) - cy) * depth_meters / fy
        point.point.z = depth_meters

        if not all(
            math.isfinite(value)
            for value in (
                point.point.x,
                point.point.y,
                point.point.z,
            )
        ):
            return None

        return point


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

