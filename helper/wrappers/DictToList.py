import gym
import numpy as np

from .Wrapper import EnvironmentWrapper


class DictToListFull(EnvironmentWrapper):
    def __init__(self, env):
        """
        A wrapper that formats dict-type observation to list-type observation.
        Appends all meaningful unique numbers in the dict-type observation to a
        list. The resulting list has length 347.
        """
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low=-float('Inf'),
                                                high=float('Inf'),
                                                shape=(350, ),
                                                dtype=np.float32)

    def reset(self):
        state_desc = self.env.reset()
        return self._dict_to_list(state_desc)

    def step(self, action):
        state_desc, reward, done, info = self.env.step(action)
        return [self._dict_to_list(state_desc), reward, done, info]

    def _dict_to_list(self, state_desc):
        """
        Return observation list of length 347 given a dict-type observation.

        For more details about the observation, visit this page:
        http://osim-rl.stanford.edu/docs/nips2018/observation/
        """
        res = []

        # Body Observations
        for info_type in ['body_pos', 'body_pos_rot',
                          'body_vel', 'body_vel_rot',
                          'body_acc', 'body_acc_rot']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head', 'pelvis',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += state_desc[info_type][body_part]
        
        # Joint Observations
        # Neglecting `back_0`, `mtp_l`, `subtalar_l` since they do not move
        for info_type in ['joint_pos', 'joint_vel', 'joint_acc']:
            for joint in ['ankle_l', 'ankle_r', 'back', 'ground_pelvis',
                          'hip_l', 'hip_r', 'knee_l', 'knee_r']:
                res += state_desc[info_type][joint]

        # Muscle Observations
        for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r', 
                       'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                       'glut_max_l', 'glut_max_r', 
                       'hamstrings_l', 'hamstrings_r',
                       'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                       'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
            res.append(state_desc['muscles'][muscle]['activation'])
            res.append(state_desc['muscles'][muscle]['fiber_force'])
            res.append(state_desc['muscles'][muscle]['fiber_length'])
            res.append(state_desc['muscles'][muscle]['fiber_velocity'])

        # Force Observations
        # Neglecting forces corresponding to muscles as they are redundant with
        # `fiber_forces` in muscles dictionaries
        for force in ['AnkleLimit_l', 'AnkleLimit_r',
                      'HipAddLimit_l', 'HipAddLimit_r',
                      'HipLimit_l', 'HipLimit_r', 'KneeLimit_l', 'KneeLimit_r']:
            res += state_desc['forces'][force]

        # Center of Mass Observations
        res += state_desc['misc']['mass_center_pos']
        res += state_desc['misc']['mass_center_vel']
        res += state_desc['misc']['mass_center_acc']

        return res


class DictToListLegacy(EnvironmentWrapper):
    def __init__(self, env):
        """
        DEPRECATED
        A legacy wrapper that formats dict-type observation to list-type
        observation. Has the same effect as setting project=False in env.reset()
        or env.step(). The resulting list has length 158. This wrapper contains
        bugs and exists only for legacy purposes.
        """
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low=-float('Inf'),
                                                high=float('Inf'),
                                                shape=(158, ),
                                                dtype=np.float32)

    def reset(self):
        state_desc = self.env.reset()
        return self._get_observation(state_desc)

    def step(self, action):
        state_desc, reward, done, info = self.env.step(action)
        return [self._get_observation(state_desc), reward, done, info]

    def _get_observation(self, state_desc):
        """
        Code from ProstheticsEnv.get_observation(). Contains bug described in
        issue #129 in stanfordnmbl/osim-rl GitHub repository.

        https://github.com/stanfordnmbl/osim-rl/blob/master/osim/env/osim.py
        https://github.com/stanfordnmbl/osim-rl/issues/129
        """
        # Augmented environment from the L2R challenge
        res = []
        pelvis = None

        for body_part in ["pelvis", "head","torso","toes_l","toes_r","talus_l","talus_r"]:
            if body_part in ["toes_r","talus_r"]:
                res += [0] * 9
                continue
            cur = []
            cur += state_desc["body_pos"][body_part][0:2]
            cur += state_desc["body_vel"][body_part][0:2]
            cur += state_desc["body_acc"][body_part][0:2]
            cur += state_desc["body_pos_rot"][body_part][2:]
            cur += state_desc["body_vel_rot"][body_part][2:]
            cur += state_desc["body_acc_rot"][body_part][2:]
            if body_part == "pelvis":
                pelvis = cur
                res += cur[1:]
            else:
                cur_upd = cur
                cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
                cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6,7)]
                res += cur

        for joint in ["ankle_l","ankle_r","back","hip_l","hip_r","knee_l","knee_r"]:
            res += state_desc["joint_pos"][joint]
            res += state_desc["joint_vel"][joint]
            res += state_desc["joint_acc"][joint]

        for muscle in sorted(state_desc["muscles"].keys()):
            res += [state_desc["muscles"][muscle]["activation"]]
            res += [state_desc["muscles"][muscle]["fiber_length"]]
            res += [state_desc["muscles"][muscle]["fiber_velocity"]]

        cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
        res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

        return res
