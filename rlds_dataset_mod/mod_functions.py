from abc import ABC, abstractmethod

import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds


class TfdsModFunction(ABC):
    @classmethod
    @abstractmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        """
        Modifies the data builder feature dict to reflect feature changes of ModFunction.
        """
        ...

    @classmethod
    @abstractmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        Perform arbitrary modifications on the dataset that comply with the modified feature definition.
        """
        ...


def mod_obs_features(features, obs_feature_mod_function):
    """Utility function to only modify keys in observation dict."""
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            key: obs_feature_mod_function(
                                key, features["steps"]["observation"][key]
                            )
                            for key in features["steps"]["observation"].keys()
                        }
                    ),
                    **{
                        key: features["steps"][key]
                        for key in features["steps"].keys()
                        if key not in ("observation",)
                    },
                }
            ),
            **{key: features[key] for key in features.keys() if key not in ("steps",)},
        }
    )


class ResizeAndJpegEncode(TfdsModFunction):
    MAX_RES: int = 256

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        def downsize_and_jpeg(key, feat):
            """Downsizes image features, encodes as jpeg."""
            if len(feat.shape) >= 2 and feat.shape[0] >= 64 and feat.shape[1] >= 64:  # is image / depth feature
                should_jpeg_encode = (
                    isinstance(feat, tfds.features.Image) and "depth" not in key
                )
                if len(feat.shape) > 2:
                    new_shape = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES, feat.shape[2])
                else:
                    new_shape = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES)

                if isinstance(feat, tfds.features.Image):
                    return tfds.features.Image(
                        shape=new_shape,
                        dtype=feat.dtype,
                        encoding_format="jpeg" if should_jpeg_encode else "png",
                        doc=feat.doc,
                    )
                else:
                    return tfds.features.Tensor(
                        shape=new_shape,
                        dtype=feat.dtype,
                        doc=feat.doc,
                    )

            return feat

        return mod_obs_features(features, downsize_and_jpeg)

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def resize_image_fn(step):
            # resize images
            for key in step["observation"]:
                if len(step["observation"][key].shape) >= 2 and (
                    step["observation"][key].shape[0] >= 64
                    or step["observation"][key].shape[1] >= 64
                ):
                    size = (ResizeAndJpegEncode.MAX_RES,
                            ResizeAndJpegEncode.MAX_RES)
                    if "depth" in key:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_depth_image(
                                tf.cast(step["observation"][key], tf.float32), size
                            ),
                            step["observation"][key].dtype,
                        )
                    else:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_image(step["observation"][key], size),
                            tf.uint8,
                        )
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(resize_image_fn)
            return episode

        return ds.map(episode_map_fn)


class FilterSuccess(TfdsModFunction):
    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.filter(lambda e: e["success"])


class FlipImgChannels(TfdsModFunction):
    FLIP_KEYS = ["image"]

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def flip(step):
            for key in cls.FLIP_KEYS:
                if key in step["observation"]:
                    step["observation"][key] = step["observation"][key][..., ::-1]
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(flip)
            return episode

        return ds.map(episode_map_fn)
    

class FlipWristImgChannels(FlipImgChannels):
    FLIP_KEYS = ["wrist_image", "hand_image"]


class AddJointPosition(TfdsModFunction):
    MAX_RES: int = 256

    @classmethod
    def mod_features(
        cls,
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        def add_joint_positions(key, feat):
            """Downsizes image features, encodes as jpeg."""
            if len(feat.shape) >= 2 and feat.shape[0] >= 64 and feat.shape[1] >= 64:  # is image / depth feature
                should_jpeg_encode = (
                    isinstance(feat, tfds.features.Image) and "depth" not in key
                )
                if len(feat.shape) > 2:
                    new_shape = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES, feat.shape[2])
                else:
                    new_shape = (ResizeAndJpegEncode.MAX_RES, ResizeAndJpegEncode.MAX_RES)

                if isinstance(feat, tfds.features.Image):
                    return tfds.features.Image(
                        shape=new_shape,
                        dtype=feat.dtype,
                        encoding_format="jpeg" if should_jpeg_encode else "png",
                        doc=feat.doc,
                    )
                else:
                    return tfds.features.Tensor(
                        shape=new_shape,
                        dtype=feat.dtype,
                        doc=feat.doc,
                    )

            return feat

        return mod_obs_features(features, downsize_and_jpeg)

    @classmethod
    def mod_dataset(cls, ds: tf.data.Dataset) -> tf.data.Dataset:
        def resize_image_fn(step):
            # resize images
            for key in step["observation"]:
                if len(step["observation"][key].shape) >= 2 and (
                    step["observation"][key].shape[0] >= 64
                    or step["observation"][key].shape[1] >= 64
                ):
                    size = (ResizeAndJpegEncode.MAX_RES,
                            ResizeAndJpegEncode.MAX_RES)
                    if "depth" in key:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_depth_image(
                                tf.cast(step["observation"][key], tf.float32), size
                            ),
                            step["observation"][key].dtype,
                        )
                    else:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_image(step["observation"][key], size),
                            tf.uint8,
                        )
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(resize_image_fn)
            return episode

        return ds.map(episode_map_fn)




TFDS_MOD_FUNCTIONS = {
    "resize_and_jpeg_encode": ResizeAndJpegEncode,
    "filter_success": FilterSuccess,
    "flip_image_channels": FlipImgChannels,
    "flip_wrist_image_channels": FlipWristImgChannels,
}


##########
OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_oxe": {
        "transform": bridge_oxe_dataset_transform,
        "ik_params": {
            "source": lambda obs: obs["EEF_state"],
            "urdf": "franka_panda.urdf"
        }
    },
    "bridge_orig": {
        "transform": bridge_orig_dataset_transform,
        "ik_params": {
            "source": lambda obs: obs["EEF_state"],
            "urdf": "franka_panda.urdf"
        }
    },
    "ppgm": {
        "transform": ppgm_dataset_transform,
        "ik_params": {
            "source": lambda obs: obs["cartesian_position"][..., :6],
            "urdf": "kuka_iiwa.urdf"
        }
    },
    "rt1": {
        "transform": rt1_dataset_transform,
        "ik_params": {
            "source": lambda obs: tf.concat([obs["base_pose_tool_reached"], obs["gripper_closed"]], axis=-1),
            "urdf": "universal_robot.urdf"
        }
    },
    "kuka": {
        "transform": kuka_dataset_transform,
        "ik_params": {
            "source": lambda obs: obs["clip_function_input/base_pose_tool_reached"][..., :6],
            "urdf": "kuka_iiwa.urdf"
        }
    },
    "taco_play": {
        "transform": taco_play_dataset_transform,
        "ik_params": {
            "source": lambda obs: obs["state_eef"],
            "urdf": "franka_panda.urdf"
        }
    },
    "jaco_play": {
        "transform": jaco_play_dataset_transform,
        "ik_params": {
            "source": lambda obs: obs["end_effector_cartesian_pos"][..., :6],
            "urdf": "jaco_arm.urdf"
        }
    },
    "berkeley_cable_routing": {
        "transform": berkeley_cable_routing_dataset_transform,
        "ik_params": {
            "source": lambda obs: obs["state"][..., :6],
            "urdf": "ur10.urdf"
        }
    },
    "roboturk": {
        "transform": roboturk_dataset_transform,
        "ik_params": {
            "source": lambda obs: obs["state"][..., :6],
            "urdf": "sawyer.urdf"
        }
    },
    "nyu_door_opening": {
        "transform": nyu_door_opening_dataset_transform,
        "ik_params": {
            "source": lambda obs: obs["state"][..., :6],
            "urdf": "franka_panda.urdf"
        }
    },
    "viola": {
        "ik_params": {
            "source": lambda obs: obs["state"][..., :6],
            "urdf": "ur5.urdf"
        }
    },
    "berkeley_autolab_ur5": {
        "ik_params": {
            "source": lambda obs: obs["robot_state"][..., 6:12],
            "urdf": "ur5.urdf"
        }
    },
    "toto": {
        "ik_params": {
            "source": lambda obs: obs["state"][..., :6],
            "urdf": "xarm.urdf"
        }
    },
    "language_table": {
        "ik_params": {
            "source": lambda obs: tf.zeros((6,)),  # Dummy for non-physical
            "urdf": "dummy.urdf"
        }
    },
    "pusht": {
        "ik_params": {
            "source": lambda obs: obs["state"][..., :6],
            "urdf": "poppy_ergo_jr.urdf"
        }
    },
    "ucsd_pick_place": {
        "ik_params": {
            "source": lambda obs: obs["eef_state"],
            "urdf": "yumi.urdf"
        }
    },
    "robo_net": {
        "ik_params": {
            "source": lambda obs: obs["state"][..., :4],
            "urdf": "stretch.urdf"
        }
    },
    "libero": {
        "ik_params": {
            "source": lambda obs: obs["EEF_state"],
            "urdf": "franka_panda.urdf"
        }
    },
    "berkeley_mvp": {
        "ik_params": {
            "source": lambda obs: obs["state"][..., :6],
            "urdf": "tiago.urdf"
        }
    },
    "cmu_stretch": {
        "ik_params": {
            "source": lambda obs: obs["eef_state"],
            "urdf": "stretch_re1.urdf"
        }
    }
}