import glob
import json
import os
import sys
from pathlib import Path

# add repository root to python path
from src.utils.detection import Detection
from src.utils.perspective import parse_perspective

sys.path.append(str(Path(__file__).parent.parent.parent))

import cv2
import argparse
import open3d as o3d
import numpy as np

from src.utils.vis_utils import VisualizationUtils
from src.utils.utils import id_to_class_name_mapping, class_name_to_id_mapping

np.set_printoptions(suppress=True)
from scipy.spatial.transform import Rotation as R


# Example:
# python visualize_image_with_lidar_labels_all_frames.py --input_folder_path_images images --input_folder_path_point_clouds point_clouds --input_folder_path_labels labels --camera_id s110_camera_basler_south1_8mm --lidar_id s110_lidar_ouster_south --use_detection_boxes_in_s110_base  --viz_mode [box2d,box3d,point_cloud,track_history] --output_folder_path_visualization visualization


def process_boxes(box_data, use_two_colors, input_type, camera_id, lidar_id, perspective, use_boxes_in_s110_base):
    """
    :param box_data:                JSON data of the boxes
    :param use_two_colors:          If True, use two colors for the boxes
    :param input_type:              Type of the input data (labels or detections)
    :param camera_id:               ID of the camera
    :param lidar_id:                ID of the lidar
    :param perspective:             Perspective object of the camera
    :param use_boxes_in_s110_base:  If True, use the boxes in the s110 base coordinate system (no transformation from lidar to base will be applied)
    :return:
    """
    if "openlabel" in box_data:
        detections = []
        num_detections_lidar = 0
        num_detections_camera = 0
        num_detections_fused = 0
        for frame_idx, frame_obj in box_data["openlabel"]["frames"].items():
            for box_idx, box in frame_obj["objects"].items():
                category = box["object_data"]["type"]
                if "cuboid" in box["object_data"]:
                    cuboid = box["object_data"]["cuboid"]["val"]
                    quaternion = [cuboid[3], cuboid[4], cuboid[5], cuboid[6]]
                    yaw = R.from_quat(quaternion).as_euler("xyz", degrees=False)[2]
                    pos_history = np.array([])
                    if "attributes" in box["object_data"]["cuboid"]:
                        attribute = VisualizationUtils.get_attribute_by_name(
                            box["object_data"]["cuboid"]["attributes"]["text"], "body_color"
                        )
                        if attribute is not None:
                            color = attribute["val"]
                        else:
                            color = ""
                        attribute = VisualizationUtils.get_attribute_by_name(
                            box["object_data"]["cuboid"]["attributes"]["num"], "num_points"
                        )
                        if attribute is not None:
                            num_points = int(attribute["val"])
                        else:
                            num_points = -1

                        attribute = VisualizationUtils.get_attribute_by_name(
                            box["object_data"]["cuboid"]["attributes"]["num"], "score"
                        )
                        if attribute is not None:
                            score = round(float(attribute["val"]), 2)
                        else:
                            score = -1

                        # extract sensor ID as attribute to each detected object
                        attribute = VisualizationUtils.get_attribute_by_name(
                            box["object_data"]["cuboid"]["attributes"]["text"], "sensor_id"
                        )
                        if attribute is not None:
                            sensor_id = attribute["val"]
                        else:
                            sensor_id = ""

                        # TODO: temp set sensor ID for mono3d detections
                        sensor_id = "s110_camera_basler_south2_8mm"

                        if "vec" in box["object_data"]["cuboid"]["attributes"]:
                            attribute = VisualizationUtils.get_attribute_by_name(
                                box["object_data"]["cuboid"]["attributes"]["vec"], "track_history"
                            )
                            if attribute is not None:
                                track_history_list = attribute["val"]
                                pos_history = np.array(track_history_list).reshape(-1, 3)
                            else:
                                pos_history = np.array([])
                    else:
                        color = ""
                        num_points = -1
                        score = -1
                        pos_history = np.array([])

                    # TODO: temp set sensor ID for mono3d detections
                    # sensor_id = "s110_camera_basler_south2_8mm"

                    if "lidar" in sensor_id and not "camera" in sensor_id:
                        num_detections_lidar += 1
                    elif "camera" in sensor_id and not "lidar" in sensor_id:
                        num_detections_camera += 1
                    elif "camera" in sensor_id and "lidar" in sensor_id:
                        num_detections_lidar += 1
                        num_detections_camera += 1
                        num_detections_fused += 1

                    detection = Detection(
                        location=np.array([[cuboid[0]], [cuboid[1]], [cuboid[2]]]),
                        dimensions=(cuboid[7], cuboid[8], cuboid[9]),
                        yaw=yaw,
                        category=category,
                        uuid=box_idx,
                        color=color,
                        score=score,
                        num_lidar_points=num_points,
                        pos_history=pos_history,
                    )
                    detections.append(detection)
                    if "box3d" in viz_mode:
                        if use_two_colors and input_type == "detections":
                            color_rgb = (245, 44, 71)  # red
                        elif use_two_colors and input_type == "labels":
                            color_rgb = (27, 250, 27)  # green
                        else:
                            if viz_color_mode == "by_category":
                                color_rgb = id_to_class_name_mapping[str(class_name_to_id_mapping[category])][
                                    "color_rgb"
                                ]
                            elif viz_color_mode == "by_sensor_type":
                                # TODO: parse sensor_id from attribute
                                # sensor_id = "s110_lidar_ouster_south"
                                if "lidar" in sensor_id and not "camera" in sensor_id:
                                    # set to light blue
                                    color_rgb = (52, 192, 235)
                                elif "camera" in sensor_id and not "lidar" in sensor_id:
                                    # set to light green
                                    color_rgb = (96, 255, 96)
                                elif "camera" in sensor_id and "lidar" in sensor_id:
                                    # set to red
                                    color_rgb = (255, 96, 96)
                                else:
                                    print(
                                        "Unknown sensor type. Please check sensor_id. Valid sensor types are: s110_lidar_ouster_south, s110_lidar_ouster_north, s110_camera_basler_south1_8mm, s110_camera_basler_south1_8mm. Exiting..."
                                    )
                                    sys.exit()
                            else:
                                print(
                                    "Unknown color mode. Valid color modes are: by_category, by_sensor_type. Exiting..."
                                )
                                sys.exit()
                        # swap channels because opencv uses bgr
                        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
                        utils.draw_3d_box(img, box, color_bgr, camera_id, lidar_id, use_boxes_in_s110_base, perspective)
                    if "track_history" in viz_mode:
                        # draw track history
                        location_current = detection.location.flatten()
                        if use_boxes_in_s110_base:
                            z_pos = 0
                        else:
                            z_pos = location_current[2] - detection.dimensions[2] / 2.0
                        track_history_positions = [
                            np.array(
                                [
                                    location_current[0],
                                    location_current[1],
                                    z_pos,
                                ]
                            )
                        ]
                        for position3d in detection.pos_history:
                            track_history_positions.append([position3d[0], position3d[1], position3d[2]])

                        projected_track_history = utils.project_3d_box_to_2d(
                            np.array(track_history_positions),
                            camera_id,
                            lidar_id,
                            use_boxes_in_s110_base=use_boxes_in_s110_base,
                        )
                        projected_track_history = np.array(projected_track_history).astype(int).T
                        cv2.polylines(img, [projected_track_history], False, (0, 0, 255), 1, cv2.LINE_AA)

            if num_detections_lidar > 0 or num_detections_camera > 0 or num_detections_fused > 0:
                if "statistics" in viz_mode:
                    utils.plot_statistics(img, num_detections_lidar, num_detections_camera, num_detections_fused)


def set_track_history(labels_json, label_file_paths, current_frame_idx, use_boxes_in_s110_base):
    for frame_idx, frame_obj in labels_json["openlabel"]["frames"].items():
        for uuid, box in frame_obj["objects"].items():
            if "cuboid" in box["object_data"]:
                track_history = []
                # iterate all previous frames and append historic position
                for idx in range(current_frame_idx - 1, -1, -1):
                    found_same_object = False
                    label_file_path_current = label_file_paths[idx]
                    labels_json_current = json.load(open(label_file_path_current))
                    for frame_idx_current, frame_obj_current in labels_json_current["openlabel"]["frames"].items():
                        for uuid_current, box_current in frame_obj_current["objects"].items():
                            if uuid_current == uuid:
                                found_same_object = True
                                if "cuboid" in box_current["object_data"]:
                                    cuboid_current = box_current["object_data"]["cuboid"]["val"]
                                    track_history.append(cuboid_current[0])
                                    track_history.append(cuboid_current[1])
                                    if use_boxes_in_s110_base:
                                        track_history.append(0)
                                    else:
                                        track_history.append(cuboid_current[2] - cuboid_current[9] / 2.0)
                    if not found_same_object:
                        break
                if "attributes" in box["object_data"]["cuboid"]:
                    track_history_attribute = None
                    if "vec" in box["object_data"]["cuboid"]["attributes"]:
                        track_history_attribute = VisualizationUtils.get_attribute_by_name(
                            box["object_data"]["cuboid"]["attributes"]["vec"], "track_history"
                        )
                    else:
                        box["object_data"]["cuboid"]["attributes"]["vec"] = []
                    if track_history_attribute is None:
                        # create track history attribute and add current location
                        track_history_attribute = {"name": "track_history", "val": track_history}
                        box["object_data"]["cuboid"]["attributes"]["vec"].append(track_history_attribute)
                    else:
                        # update track history attribute
                        track_history_attribute["val"] = track_history


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="VizLabel Argument Parser")
    argparser.add_argument(
        "--input_folder_path_images",
        default="images",
        help="Image folder path. Default: images",
    )
    argparser.add_argument(
        "--input_folder_path_point_clouds",
        default="",
        help="Point cloud folder path. Default: point_clouds",
    )
    argparser.add_argument(
        "--input_folder_path_labels",
        help="Input folder path to lidar labels in OpenLABEL format.",
    )
    argparser.add_argument(
        "--input_folder_path_detections",
        help="Input folder path to lidar detections in OpenLABEL format.",
    )
    argparser.add_argument(
        "--camera_id",
        default="",
        help="Camera id. Default: s110_camera_basler_south1_8mm",
    )
    argparser.add_argument(
        "--lidar_id",
        default="",
        help="Lidar id. Default: s110_lidar_ouster_south",
    )
    # add file path to calibration data
    argparser.add_argument(
        "--file_path_calibration_data",
        default="",
        help="File path to calibration data. Default: calib/calibration.json",
    )
    argparser.add_argument(
        "--use_label_boxes_in_s110_base",
        action="store_true",
        help="Assume that the label boxes are in s110 base frame. Set this to true, if the labels are in s110_base coordinate frame.",
    )
    argparser.add_argument(
        "--use_detection_boxes_in_s110_base",
        action="store_true",
        help="Assume that the detection boxes are in s110 base frame.",
    )
    argparser.add_argument(
        "--viz_mode",
        default="box3d_projected",
        help="Visualization mode. Available modes are: [box2d, box3d_projected, box3d, point_cloud, track_history]. Mode can be combined and separated by comma, e.g. box2d,box3d. Default: box3d",
    )
    argparser.add_argument(
        "--viz_color_mode",
        default="by_class",
        help="Visualization color mode. Available modes are: [by_class, by_physical_color]",
    )
    argparser.add_argument(
        "--output_folder_path_visualization",
        default="visualization",
        help="Output folder path to save visualization result to disk.",
    )
    args = argparser.parse_args()

    input_folder_path_images = args.input_folder_path_images
    input_folder_path_point_clouds = args.input_folder_path_point_clouds
    input_folder_path_labels = args.input_folder_path_labels
    input_folder_path_detections = args.input_folder_path_detections
    camera_id = args.camera_id
    lidar_id = args.lidar_id
    use_label_boxes_in_s110_base = bool(args.use_label_boxes_in_s110_base)
    use_detection_boxes_in_s110_base = bool(args.use_detection_boxes_in_s110_base)
    viz_mode = args.viz_mode
    viz_color_mode = args.viz_color_mode
    output_folder_path_visualization = args.output_folder_path_visualization

    if not os.path.exists(output_folder_path_visualization):
        Path(output_folder_path_visualization).mkdir(parents=True)

    utils = VisualizationUtils()
    camera_perspectives = {"s110_camera_basler_south1_8mm": None, "s110_camera_basler_south2_8mm": None}

    types = ("*.jpg", "*.png")  # the tuple of image file types
    input_image_file_paths = []
    for files in types:
        input_image_file_paths.extend(sorted(glob.glob(input_folder_path_images + "/" + files)))
    print("Found {} images in {}".format(len(input_image_file_paths), input_folder_path_images))

    if input_folder_path_point_clouds != "":
        point_cloud_file_paths = sorted(glob.glob(input_folder_path_point_clouds + "/*.pcd"))
        print("Found {} point cloud files.".format(len(point_cloud_file_paths)))
    else:
        point_cloud_file_paths = [""] * len(input_image_file_paths)

    if input_folder_path_labels is not None:
        label_file_paths = sorted(glob.glob(input_folder_path_labels + "/*.json"))
        print("Found {} label files.".format(len(label_file_paths)))
    else:
        label_file_paths = [""] * len(input_image_file_paths)

    if input_folder_path_detections is not None:
        detection_file_paths = sorted(glob.glob(input_folder_path_detections + "/*.json"))
        print("Found {} detection files.".format(len(detection_file_paths)))
    else:
        detection_file_paths = [""] * len(input_image_file_paths)

    parse_lidar_id = lidar_id == ""
    parse_camera_id = camera_id == ""

    current_frame_idx = 0
    for image_file_path, point_cloud_file_path, label_file_path, detection_file_path in zip(
        input_image_file_paths,
        point_cloud_file_paths,
        label_file_paths,
        detection_file_paths,
    ):
        print("image: ", image_file_path)
        # extract lidar id from label_file_path
        if parse_lidar_id:
            lidar_id = "_".join(Path(point_cloud_file_path).name.split(".")[0].split("_")[2:6])
        if parse_camera_id:
            camera_id = "_".join(Path(image_file_path).name.split(".")[0].split("_")[2:])

        if camera_perspectives[camera_id] is None:
            camera_perspectives[camera_id] = parse_perspective(args.file_path_calibration_data)
            camera_perspectives[camera_id].initialize_matrices()

        img = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        if "point_cloud" in viz_mode:
            if point_cloud_file_path != "":
                point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
                img = utils.project_point_cloud_to_image(img, point_cloud, lidar_id=lidar_id, camera_id=camera_id)
            else:
                print("No point cloud file found. Please set input_folder_path_point_clouds.")
                sys.exit(1)

        if label_file_path != "":
            labels_json = json.load(open(label_file_path))
        else:
            labels_json = []

        if detection_file_path != "":
            detections_json = json.load(open(detection_file_path))
        else:
            detections_json = []

        if "box2d" in viz_mode or "box3d" in viz_mode or "box3d_projected" in viz_mode or "track_history" in viz_mode:

            use_two_colors = label_file_path != "" and detection_file_path != ""
            if label_file_path != "":
                if "track_history" in viz_mode:
                    # set track history attribute
                    set_track_history(
                        labels_json,
                        label_file_paths,
                        current_frame_idx,
                        use_boxes_in_s110_base=use_detection_boxes_in_s110_base or use_label_boxes_in_s110_base,
                    )

                process_boxes(
                    labels_json,
                    use_two_colors,
                    input_type="labels",
                    camera_id=camera_id,
                    lidar_id=lidar_id,
                    perspective=camera_perspectives[camera_id],
                    use_boxes_in_s110_base=use_label_boxes_in_s110_base,
                )
            if detection_file_path != "":
                if "track_history" in viz_mode:
                    # set track history attribute
                    set_track_history(
                        detections_json,
                        detection_file_paths,
                        current_frame_idx,
                        use_boxes_in_s110_base=use_detection_boxes_in_s110_base or use_label_boxes_in_s110_base,
                    )
                process_boxes(
                    detections_json,
                    use_two_colors,
                    input_type="detections",
                    camera_id=camera_id,
                    lidar_id=lidar_id,
                    perspective=camera_perspectives[camera_id],
                    use_boxes_in_s110_base=use_detection_boxes_in_s110_base,
                )
        if output_folder_path_visualization:
            cv2.imwrite(
                os.path.join(output_folder_path_visualization, Path(image_file_path).name.replace("png", "jpg")), img
            )
        else:
            cv2.imshow("image", img)
            cv2.waitKey()
        current_frame_idx = current_frame_idx + 1
