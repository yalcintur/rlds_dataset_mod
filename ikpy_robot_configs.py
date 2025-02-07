# ikpy_robot_configs.py

# This dictionary maps dataset names to the corresponding IK configuration.
# For each dataset you can specify:
#   - "urdf_path": The path to the robot’s URDF.
#   - "eef_field": The key within the observation dictionary where the EEF position is stored.
#                   (If not defined, your IK function may choose a default.)
#   - "eef_slice": A tuple (start, end) indicating the slice of the tensor (from the given field)
#                  that contains the EEF position.
#   - Optionally, other keys such as "eef_key" can be provided if the data is nested.

IKPY_ROBOT_CONFIGS = {
    "bridge_oxe": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",       # from trajectory["observation"]["state"]
         "eef_slice": (0, 6),
    },
    "bridge_orig": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "bridge_dataset": {  # alias for bridge_orig
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "ppgm": {
         "urdf_path": "path_to_urdf",
         "eef_field": "cartesian_position",  # from trajectory["observation"]["cartesian_position"]
         "eef_slice": (0, 6),
    },
    "ppgm_static": {
         "urdf_path": "path_to_urdf",
         "eef_field": "cartesian_position",
         "eef_slice": (0, 6),
    },
    "ppgm_wrist": {
         "urdf_path": "path_to_urdf",
         "eef_field": "cartesian_position",
         "eef_slice": (0, 6),
    },
    "fractal20220817_data": {
         "urdf_path": "path_to_urdf",
         # No explicit EEF extraction in the transform.
    },
    "kuka": {
         "urdf_path": "path_to_urdf",
         # Transform does not extract EEF position explicitly.
    },
    "taco_play": {
         "urdf_path": "path_to_urdf",
         "eef_field": "robot_obs",   # from trajectory["observation"]["robot_obs"]
         "eef_slice": (0, 6),
    },
    "jaco_play": {
         "urdf_path": "path_to_urdf",
         "eef_field": "end_effector_cartesian_pos",  # from trajectory["observation"]["end_effector_cartesian_pos"]
         "eef_slice": (0, 6),
    },
    "berkeley_cable_routing": {
         "urdf_path": "path_to_urdf",
    },
    "roboturk": {
         "urdf_path": "path_to_urdf",
    },
    "nyu_door_opening_surprising_effectiveness": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",  # from trajectory["observation"]["state"]
         "eef_slice": (0, 6),
    },
    "viola": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "berkeley_autolab_ur5": {
         "urdf_path": "path_to_urdf",
    },
    "toto": {
         "urdf_path": "path_to_urdf",
    },
    "language_table": {
         "urdf_path": "path_to_urdf",
    },
    "columbia_cairlab_pusht_real": {
         "urdf_path": "path_to_urdf",
    },
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "nyu_rot_dataset_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",  # from trajectory["observation"]["state"]
         "eef_slice": (0, 6),
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",  # (the transform concatenates parts of state)
         "eef_slice": (0, 6),
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (-6, None),  # last six elements
    },
    "maniskill_dataset_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 7),  # as defined in the transform
    },
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 7),
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "bc_z": {
         "urdf_path": "path_to_urdf",
    },
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "robo_net": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 4),  # the transform pads with zeros for indices 4-6
    },
    "berkeley_mvp_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "berkeley_rpt_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "kaist_nonprehensile_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "stanford_mask_vit_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "end_effector_pose",
         "eef_slice": (0, 4),  # zeros are added for the remaining dimensions
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "asu_table_top_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "ground_truth_states",
         "eef_key": "EE",  # use: trajectory["observation"]["ground_truth_states"]["EE"]
    },
    "stanford_robocook_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "imperialcollege_sawyer_wrist_cam": {
         "urdf_path": "path_to_urdf",
    },
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
         "urdf_path": "path_to_urdf",
    },
    "uiuc_d3field": {
         "urdf_path": "path_to_urdf",
    },
    "utaustin_mutex": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 8),
    },
    "berkeley_fanuc_manipulation": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "cmu_playing_with_food": {
         "urdf_path": "path_to_urdf",
    },
    "cmu_play_fusion": {
         "urdf_path": "path_to_urdf",
    },
    "cmu_stretch": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 3),
    },
    "berkeley_gnm_recon": {
         "urdf_path": "path_to_urdf",
    },
    "berkeley_gnm_cory_hall": {
         "urdf_path": "path_to_urdf",
    },
    "berkeley_gnm_sac_son": {
         "urdf_path": "path_to_urdf",
    },
    "droid": {
         "urdf_path": "path_to_urdf",
    },
    "fmb_dataset": {
         "urdf_path": "path_to_urdf",
    },
    "dobbe": {
         "urdf_path": "path_to_urdf",
    },
    "roboset": {
         "urdf_path": "path_to_urdf",
    },
    "rh20t": {
         "urdf_path": "path_to_urdf",
    },
    "tdroid_carrot_in_bowl": {
         "urdf_path": "path_to_urdf",
         "eef_field": "cartesian_position",
         "eef_slice": (0, 6),
    },
    "tdroid_pour_corn_in_pot": {
         "urdf_path": "path_to_urdf",
         "eef_field": "cartesian_position",
         "eef_slice": (0, 6),
    },
    "tdroid_flip_pot_upright": {
         "urdf_path": "path_to_urdf",
         "eef_field": "cartesian_position",
         "eef_slice": (0, 6),
    },
    "tdroid_move_object_onto_plate": {
         "urdf_path": "path_to_urdf",
         "eef_field": "cartesian_position",
         "eef_slice": (0, 6),
    },
    "tdroid_knock_object_over": {
         "urdf_path": "path_to_urdf",
         "eef_field": "cartesian_position",
         "eef_slice": (0, 6),
    },
    "tdroid_cover_object_with_towel": {
         "urdf_path": "path_to_urdf",
         "eef_field": "cartesian_position",
         "eef_slice": (0, 6),
    },
    "droid_wipe": {
         "urdf_path": "path_to_urdf",
    },
    "libero_spatial_no_noops": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "libero_object_no_noops": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "libero_goal_no_noops": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
    "libero_10_no_noops": {
         "urdf_path": "path_to_urdf",
         "eef_field": "state",
         "eef_slice": (0, 6),
    },
}
