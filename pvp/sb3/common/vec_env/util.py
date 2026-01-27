"""
Helpers for dealing with vectorized environments.
"""
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import gym
import numpy as np

# Also import gymnasium for compatibility
try:
    import gymnasium
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False

from pvp.sb3.common.preprocessing import check_for_nested_spaces
from pvp.sb3.common.vec_env.base_vec_env import VecEnvObs


def _is_dict_space(obs_space) -> bool:
    """Check if observation space is a Dict (gym or gymnasium)."""
    if isinstance(obs_space, gym.spaces.Dict):
        return True
    if HAS_GYMNASIUM and isinstance(obs_space, gymnasium.spaces.Dict):
        return True
    return False


def _is_tuple_space(obs_space) -> bool:
    """Check if observation space is a Tuple (gym or gymnasium)."""
    if isinstance(obs_space, gym.spaces.Tuple):
        return True
    if HAS_GYMNASIUM and isinstance(obs_space, gymnasium.spaces.Tuple):
        return True
    return False


def copy_obs_dict(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Deep-copy a dict of numpy arrays.

    :param obs: a dict of numpy arrays.
    :return: a dict of copied numpy arrays.
    """
    assert isinstance(obs, OrderedDict), f"unexpected type for observations '{type(obs)}'"
    return OrderedDict([(k, np.copy(v)) for k, v in obs.items()])


def dict_to_obs(obs_space, obs_dict: Dict[Any, np.ndarray]) -> VecEnvObs:
    """
    Convert an internal representation raw_obs into the appropriate type
    specified by space.

    :param obs_space: an observation space.
    :param obs_dict: a dict of numpy arrays.
    :return: returns an observation of the same type as space.
        If space is Dict, function is identity; if space is Tuple, converts dict to Tuple;
        otherwise, space is unstructured and returns the value raw_obs[None].
    """
    if _is_dict_space(obs_space):
        return obs_dict
    elif _is_tuple_space(obs_space):
        assert len(obs_dict) == len(obs_space.spaces), "size of observation does not match size of observation space"
        return tuple((obs_dict[i] for i in range(len(obs_space.spaces))))
    else:
        assert set(obs_dict.keys()) == {None}, "multiple observation keys for unstructured observation space"
        return obs_dict[None]


def obs_space_info(obs_space) -> Tuple[List[str], Dict[Any, Tuple[int, ...]], Dict[Any, np.dtype]]:
    """
    Get dict-structured information about a gym.Space.

    Dict spaces are represented directly by their dict of subspaces.
    Tuple spaces are converted into a dict with keys indexing into the tuple.
    Unstructured spaces are represented by {None: obs_space}.

    :param obs_space: an observation space
    :return: A tuple (keys, shapes, dtypes):
        keys: a list of dict keys.
        shapes: a dict mapping keys to shapes.
        dtypes: a dict mapping keys to dtypes.
    """
    check_for_nested_spaces(obs_space)
    if _is_dict_space(obs_space):
        # Handle both gym and gymnasium Dict spaces
        if isinstance(obs_space.spaces, OrderedDict):
            subspaces = obs_space.spaces
        else:
            # gymnasium might not use OrderedDict, convert it
            subspaces = OrderedDict(obs_space.spaces)
    elif _is_tuple_space(obs_space):
        subspaces = {i: space for i, space in enumerate(obs_space.spaces)}
    else:
        assert not hasattr(obs_space, "spaces"), f"Unsupported structured space '{type(obs_space)}'"
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes
