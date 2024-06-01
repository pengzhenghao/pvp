import garage
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.algos import PPO
from garage.sampler import LocalSampler
from garage.torch.optimizers.optimizer_wrapper import OptimizerWrapper
from garage.envs import GymEnv, normalize

import gym
import numpy as np
from garage.experiment.deterministic import set_seed
from pvp.experiments.metaworld.metaworld_env import HumanInTheLoopEnv, MetaWorldSawyerEnv
import torch
import metaworld

@garage.wrap_experiment
def train_ppo_metaworld(ctxt=None, seed=1):
    # can set seed if you want
    set_seed(seed)

    trainer = garage.Trainer(ctxt)
    # env = HumanInTheLoopEnv(env_name='button-press-v2')
    # env_spec = garage.EnvSpec(observation_space=env.observation_space, action_space=env.action_space,
    #                           max_episode_length=500)

    ml1 = metaworld.ML1('button-press-v2')  # Replace with desired task
    env = ml1.train_classes['button-press-v2']()
    env = GymEnv(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[128, 128],
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
        min_std=0.5,
        max_std=1.5,
    )

    value_function = GaussianMLPValueFunction(
        env_spec=env.spec,
        hidden_sizes=[128, 128],
        hidden_nonlinearity=torch.tanh,
        init_std=1,
    )

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           is_tf_worker=True)

    algo = PPO(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        sampler=sampler,
        discount=0.99,
        center_adv=True,
        gae_lambda=0.95,
        lr_clip_range=0.2,
        policy_optimizer=OptimizerWrapper((torch.optim.Adam, {'lr' : 5e-4}), policy),
        vf_optimizer=OptimizerWrapper((torch.optim.Adam, {'lr' : 5e-4}), policy),
        entropy_method='max',
    )

    trainer.setup(algo, env)

    trainer.train(n_epochs=4000, batch_size=5000)


train_ppo_metaworld()

