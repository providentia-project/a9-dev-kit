import open3d as o3d
import numpy as np
import os

from pypcd import pypcd
import argparse


# Description: Remove noise (outliers) from point clouds
# Example: python a9-dataset-dev-kit/src/preprocessing/filter_noise_point_cloud.py --input_folder_path_point_clouds <INPUT_FOLDER_PATH_POINT_CLOUDS> \
#                                                                                  --output_folder_path_point_clouds <OUTPUT_FOLDER_PATH_POINT_CLOUDS> \
#                                                                                  --nb_points 1 \
#                                                                                  --radius 0.4


def write_point_cloud(point_cloud_array, output_file_path_merged_pcd):
    num_points = len(point_cloud_array[:, 0])
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        + "VERSION 0.7\n"
        + "FIELDS x y z intensity\n"
        + "SIZE 4 4 4 4\n"
        + "TYPE F F F F\n"
        + "COUNT 1 1 1 1\n"
        + "WIDTH "
        + str(num_points)
        + "\n"
        + "HEIGHT 1\n"
        + "VIEWPOINT 0 0 0 1 0 0 0\n"
        + "POINTS "
        + str(num_points)
        + "\n"
        + "DATA ascii\n"
    )
    with open(output_file_path_merged_pcd, "w") as writer:
        writer.write(header)
        np.savetxt(writer, point_cloud_array, delimiter=" ", fmt="%.4f %.4f %.4f %.4f")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--input_folder_path_point_clouds", default="point_clouds", type=str, help="Input folder path of point clouds"
    )
    argparser.add_argument(
        "--output_folder_path_point_clouds",
        default="output",
        type=str,
        help="Output folder path of filtered point clouds",
    )
    argparser.add_argument(
        "--nb_points",
        default="1",
        type=int,
        help="Pick the minimum amount of points that the sphere should contain. Higher nb_points removes more points.",
    )
    argparser.add_argument(
        "--radius",
        default="0.4",
        type=float,
        help="Defines the radius of the sphere that will be used for counting the neighbors.",
    )

    args = argparser.parse_args()
    input_folder_path_point_cloud = args.input_folder_path_point_clouds
    output_folder_path_point_cloud = args.output_folder_path_point_clouds
    nb_points = args.nb_points
    radius = args.radius

    if not os.path.exists(output_folder_path_point_cloud):
        os.mkdir(output_folder_path_point_cloud)

    for file_name in sorted(os.listdir(input_folder_path_point_cloud)):
        pcd = pypcd.PointCloud.from_path(os.path.join(args.input_folder_path_point_clouds, file_name))
        point_cloud_array = pcd.pc_data.view(np.float32).reshape(pcd.pc_data.shape + (-1,))

        xyz = point_cloud_array[:, :3]
        intensities = point_cloud_array[:, 3]
        max_intensity = np.max(intensities)
        intensities_norm = np.array(intensities / max_intensity)
        intensities_norm_two_col = np.c_[intensities_norm, intensities_norm]
        intensities_norm_three_col = np.c_[intensities_norm_two_col, intensities_norm]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(intensities_norm_three_col)

        print("num points before: ", str(len(pcd.points)))
        pcd_filtered, indices_keep = pcd.remove_radius_outlier(nb_points, radius)
        print("num points after: ", str(len(pcd.points)))
        print("removed ", str(len(pcd.points) - len(indices_keep)), " outliers.")

        points_array = np.asarray(pcd_filtered.points)
        intensity_array = np.asarray(pcd.colors)
        # filter intensity array
        intensity_array = intensity_array[indices_keep]
        point_cloud_array = np.c_[points_array, intensity_array[:, 0]]
        write_point_cloud(point_cloud_array, os.path.join(args.output_folder_path_point_clouds, file_name))
