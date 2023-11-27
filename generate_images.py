from src.utility.SetupUtility import SetupUtility
SetupUtility.setup([])

import numpy as np
import random
from math import radians
import argparse
import yaml
import os
import json
import mathutils
from src.utility.Initializer import Initializer
from src.utility.filter.Filter import Filter
from src.utility.loader.ObjectLoader import ObjectLoader
from src.utility.loader.CCMaterialLoader import CCMaterialLoader
from src.utility.MaterialUtility import Material
from src.utility.loader.SceneNetLoader import SceneNetLoader
from src.utility.object.FloorExtractor import FloorExtractor
from src.utility.object.ObjectPoseSampler import ObjectPoseSampler
from src.utility.CameraUtility import CameraUtility
from src.utility.lighting.SurfaceLighting import SurfaceLighting
from src.utility.BlenderUtility import get_all_blender_mesh_objects
from src.utility.MeshObjectUtility import MeshObject
from src.utility.RendererUtility import RendererUtility
from src.utility.SegMapRendererUtility import SegMapRendererUtility
from src.utility.CocoWriterUtility import CocoWriterUtility
from src.utility.MathUtility import MathUtility
from src.utility.LabelIdMapping import LabelIdMapping
from src.utility.object.PhysicsSimulation import PhysicsSimulation

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', nargs='?', type=str, default="synthetic_data_generation/config.yaml",
                        help="Path to the config file")
    parser.add_argument('object', nargs='?', type=str, default="synthetic_data_generation/object_files/object.obj",
                        help="Path to the object.obj file")
    parser.add_argument('obj_scenenet', nargs='?', type=str, default="synthetic_data_generation/object_files/scenenet.obj",
                        help="Path to the scenenet.obj file")
    parser.add_argument('texture', nargs='?', type=str, default="synthetic_data_generation/texture_library",
                        help="Path to the texture_library folder")
    parser.add_argument('cctexture', nargs='?', type=str, default="resources/cctextures",
                        help="Path to the cctextures folder")
    parser.add_argument('label_mapping', nargs='?', type=str, default="resources/id_mappings/nyu_idset.csv",
                        help="Path to the nyu_idset.csv file")
    parser.add_argument('unknown_texture_folder', nargs='?', type=str,
                        default="synthetic_data_generation/texture_library/unknown",
                        help="Path to the unknown texture folder folder")
    parser.add_argument('output_dir', nargs='?', type=str, default="synthetic_data_generation/output",
                        help="Path to where the final files will be saved ")

    return parser

def object_loader(args, cfg):
    """ Loads the objects into the scene and sets their physics attributes and category_ids.

    Args:
        args: argument parser
        cfg: config file

    """
    objects = ObjectLoader.load(args.object)
    sorted_objects = [object.get_name() for object in objects]
    sorted_objects.sort()
    # Set some category ids for loaded objects
    for j, obj in enumerate(objects):
        obj.set_cp("category_id", sorted_objects.index(obj.get_name()) + 1)
        obj.set_cp("is_cc_texture", True)
        if j == cfg["base_object"]["index"]:
            obj.set_cp("physics", False)
        else:
            obj.set_cp("physics", True)

# loads the background scene
def load_scenenet(args,cfg):
    """ Loads the background_scene_objects into the scene and sets their physics attributes and category_ids

    Args:
        args: argument parser
        cfg: config file

    """
    label_mapping = LabelIdMapping.from_csv(args.label_mapping)
    scenenet_objects = SceneNetLoader.load(args.obj_scenenet, args.texture, label_mapping,
                                           args.unknown_texture_folder)

    # Set some category ids for loaded objects
    for k, scene_obj in enumerate(scenenet_objects):
        scene_obj.set_cp("physics", False)
        scene_obj.set_cp("category_id", + cfg["num_objects"] + 2)

def scene_lighting():
    """ Fetches ceiling and sets the lighting"""

    selected_objects = MeshObject.convert_to_meshes(get_all_blender_mesh_objects())
    ceiling = Filter.by_attr(selected_objects, "name", ".*[c|C]eiling.*", regex=True)

    light_object_ceiling = ceiling
    SurfaceLighting.run(
        light_object_ceiling,
        emission_strength=np.random.uniform(4.0, 12.0),
        keep_using_base_color=False,
        emission_color=list(np.random.uniform([0.25, 0.25, 0.25, 0], [1, 1, 1, 0]))
    )

def object_sampling(cfg):
    """ Samples object's location in the scene.
    Args:
        cfg: config file

    """
    objects_to_sample = get_all_blender_mesh_objects()[(cfg["base_object"]["index"]) + 1:]
    random.shuffle(objects_to_sample)
    objects_to_check_collisions = get_all_blender_mesh_objects()[(cfg["base_object"]["index"]) + 1:]
    max_tries = cfg["max_tries"]

    def sample_pose(obj):
        # used to set random location of the objects
        pos_sampler = np.random.uniform(cfg["object_sampler"]["location_min"],
                                        cfg["object_sampler"]["location_max"])
        rot_sampler = cfg["object_sampler"]["rotation"]
        obj.set_location(pos_sampler)
        obj.set_rotation_euler(rot_sampler)

    ObjectPoseSampler.sample(
        objects_to_sample=MeshObject.convert_to_meshes(objects_to_sample),
        sample_pose_func=sample_pose,
        objects_to_check_collisions=MeshObject.convert_to_meshes(objects_to_check_collisions),
        max_tries=max_tries
    )

def add_rigidbody(cfg):
    """ Adds a rigid body element to all mesh objects and sets their physics attributes depending on their custom
    properties.

    Args:
        cfg: config file

    """
    collision_margin = cfg["rigidbody_parameters"]["collision_margin"]
    collision_mesh_source = cfg["rigidbody_parameters"]["collision_mesh_source"]
    mass_scaling = cfg["rigidbody_parameters"]["mass_scaling"]
    mass_factor = cfg["rigidbody_parameters"]["mass_factor"]
    collision_shape = cfg["rigidbody_parameters"]["collision_shape"]
    friction = cfg["rigidbody_parameters"]["friction"]
    angular_damping = cfg["rigidbody_parameters"]["angular_damping"]
    linear_damping = cfg["rigidbody_parameters"]["linear_damping"]

    # Temporary function which returns either the value set in the custom properties (if set) or the fallback value.
    def get_physics_attribute(obj, cp_name, default_value):
        if cp_name in obj:
            return obj[cp_name]
        else:
            return default_value

    # Go over all mesh objects and set their physics attributes based on the custom properties or (if not set) based
    # on the module config
    for obj in get_all_blender_mesh_objects():
        mesh_obj = MeshObject(obj)
        # Skip if the object has already an active rigid body component
        if mesh_obj.get_rigidbody() is None:
            if "physics" not in obj:
                raise Exception("The obj: '{}' has no physics attribute, each object needs one.".format(obj.name))

            # Collect physics attributes
            collision_shape = get_physics_attribute(obj, "physics_collision_shape", collision_shape)
            collision_margin = get_physics_attribute(obj, "physics_collision_margin", collision_margin)
            mass = get_physics_attribute(obj, "physics_mass", None if mass_scaling else 1)
            collision_mesh_source = get_physics_attribute(obj, "physics_collision_mesh_source",
                                                          collision_mesh_source)
            friction = get_physics_attribute(obj, "physics_friction", friction)
            angular_damping = get_physics_attribute(obj, "physics_angular_damping", angular_damping)
            linear_damping = get_physics_attribute(obj, "physics_linear_damping", linear_damping)

            # Set physics attributes
            mesh_obj.enable_rigidbody(
                active=obj["physics"],
                collision_shape="COMPOUND" if collision_shape == "CONVEX_DECOMPOSITION" else collision_shape,
                collision_margin=collision_margin,
                mass=mass,
                mass_factor=mass_factor,
                collision_mesh_source=collision_mesh_source,
                friction=friction,
                angular_damping=angular_damping,
                linear_damping=linear_damping
            )

def physics_simulation(cfg):
    """ Performs physics simulation in the scene.

    Args:
        cfg: config file

    """
    add_rigidbody(cfg)

    PhysicsSimulation.simulate_and_fix_final_poses(
        min_simulation_time=cfg["physics_simulation"]["min_simulation_time"],
        max_simulation_time=cfg["physics_simulation"]["max_simulation_time"],
        check_object_interval=cfg["physics_simulation"]["check_object_interval"],
        object_stopped_location_threshold=cfg["physics_simulation"]["object_stopped_location_threshold"],
        object_stopped_rotation_threshold=cfg["physics_simulation"]["object_stopped_rotation_threshold"],
        substeps_per_frame=cfg["physics_simulation"]["substeps_per_frame"],
        solver_iters=cfg["physics_simulation"]["solver_iters"]
    )

# function to get location and rotation angles of the objects
def object_positions(cfg, new_run_idx, new_run_id):
    all_objects = MeshObject.convert_to_meshes(get_all_blender_mesh_objects())
    obj_pos_rot = {new_run_idx: { "run_id": new_run_id, "name": [], "category_id": [], "6d_pose": []}}
    for obj_idx in range(len(all_objects)):
        obj_pos_rot[new_run_idx]["name"].append(all_objects[obj_idx].get_name())
        obj_pos_rot[new_run_idx]["category_id"].append(all_objects[obj_idx].get_cp("category_id"))
        obj_pos_rot[new_run_idx]["6d_pose"].append(list(all_objects[obj_idx].get_location())+list(all_objects[obj_idx].get_rotation()))
    return obj_pos_rot

# function to set different camera angles
def camera_positions(frame_number, cfg):
    """ sets different camera angles.

    Args:
        frame_number : camera frame
        cfg : config file

    """
    selected_objects = MeshObject.convert_to_meshes(get_all_blender_mesh_objects())
    poi = selected_objects[cfg["base_object"]["index"]].get_location()
    # Sample random camera location above objects
    location = np.random.uniform(cfg["camera_sampler"]["location_min"], cfg["camera_sampler"]["location_max"])

    # Compute rotation based on vector going from location towards poi
    rotation_matrix = CameraUtility.rotation_from_forward_vec(poi - location)
    # Add homog cam pose based on location an rotation
    cam2world_matrix = MathUtility.build_transformation_mat(location, rotation_matrix)
    CameraUtility.add_camera_pose(cam2world_matrix, frame=frame_number)

def add_texture(args, cfg):
    """ adds different textures to objects loaded.

    Args:
        args : argument parser
        cfg : config file

    """
    # Loads textures for objects
    textures = CCMaterialLoader.load(args.cctexture)
    selected_objects = MeshObject.convert_to_meshes(get_all_blender_mesh_objects())
    random.shuffle(textures)
    for s, objects in enumerate(selected_objects):
        objects.set_material(0, textures[s])
        Material.make_emissive(textures[s], cfg["emission_strength_objects"])

def main():
    parser = arg_parser()
    args = parser.parse_args()
    with open(args.config_file, "r") as yamlfile:
        cfg = yaml.load(yamlfile)

    for image in range(cfg["num_images"]):

        # Initialise the basic blender environment
        Initializer.init()
        # Loads the blocks obj file
        object_loader(args, cfg)
        # Adds textures to the objects
        add_texture(args, cfg)
        # Provides different location and rotation to the objects
        object_sampling(cfg)
        # Performs physics simulation in the scene
        physics_simulation(cfg)
        # Loads the background scene obj file
        load_scenenet(args, cfg)
        # Provides different lighting in the scene
        scene_lighting()
        # Samples different camera angles
        frame_number = 0
        camera_positions(frame_number, cfg)
        obj_pos_rot = object_positions(cfg, new_run_idx="run_index_0", new_run_id=0)


        # activate normal and distance rendering
        RendererUtility.enable_normals_output()

        # set the amount of samples, which should be used for the color rendering
        RendererUtility.set_samples(cfg["rendering_samples"])

        # render the whole pipeline
        data = RendererUtility.render()
        seg_data = SegMapRendererUtility.render(map_by=["instance", "class", "name"])

        # Write data to coco file
        CocoWriterUtility.write(args.output_dir,
                                instance_segmaps=seg_data["instance_segmaps"],
                                instance_attribute_maps=seg_data["instance_attribute_maps"],
                                colors=data["colors"],
                                color_file_format="JPEG")
        """
        root = args.output_dir
        pose_annotations = os.path.join(root, "coco_data/6d_pose_annotations.json")
        if os.path.exists(pose_annotations):
            with open(pose_annotations, 'r') as fp:
                existing_coco_annotations = json.load(fp)
                prev_run_idx = list(existing_coco_annotations.keys())[-1]
                prev_run_id = existing_coco_annotations[prev_run_idx]["run_id"]

                new_run_id = int(prev_run_id) + 1
                new_run_idx = prev_run_idx[:-1] + str(new_run_id)

                # fetching 6d pose
                obj_pos_rot = object_positions(cfg, new_run_idx, new_run_id)
                print(existing_coco_annotations)

                new_key_values_dict = {new_run_idx: obj_pos_rot[new_run_idx]}
                existing_coco_annotations.update(new_key_values_dict)

                with open(pose_annotations, 'w') as outfile:
                    outfile.write(json.dumps(existing_coco_annotations))
        else:
            # Writing to 6d_pose_annotations.json
            with open(os.path.join(root, "coco_data/6d_pose_annotations.json"), 'w') as outfile:
                # fetching 6d pose
                obj_pos_rot = object_positions(cfg, new_run_idx="run_index_0", new_run_id=0)
                outfile.write(json.dumps(obj_pos_rot))
        """




        Initializer.cleanup()

if __name__ == "__main__":
    main()