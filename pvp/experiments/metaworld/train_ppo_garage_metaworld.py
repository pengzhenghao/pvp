import garage
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.algos import PPO
from garage.sampler import LocalSampler
from garage.torch.optimizers.optimizer_wrapper import OptimizerWrapper
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.experiment.task_sampler import SetTaskSampler

import gym
import numpy as np
from garage.experiment.deterministic import set_seed
from pvp.experiments.metaworld.metaworld_env import HumanInTheLoopEnv, MetaWorldSawyerEnv
import torch
import metaworld

@garage.wrap_experiment
def train_ppo_metaworld(ctxt=None, seed=1, num_train_tasks=10):
    # can set seed if you want
    set_seed(seed)

    ml1 = metaworld.ML1('button-press-v2')
    train_env = MetaWorldSetTaskEnv(ml1, 'train')
    env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                 env=train_env,
                                 wrapper=lambda env, _: normalize(env))
    env = env_sampler.sample(num_train_tasks)
    test_env = MetaWorldSetTaskEnv(ml1, 'test')
    test_env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                      env=test_env,
                                      wrapper=lambda env, _: normalize(env))

    trainer = garage.Trainer(ctxt)
    # env = HumanInTheLoopEnv(env_name='button-press-v2')
    # env_spec = garage.EnvSpec(observation_space=env.observation_space, action_space=env.action_space,
    #                           max_episode_length=500)

    policy = GaussianMLPPolicy(
        env_spec=env[0]().spec,
        hidden_sizes=[128, 128],
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
        min_std=0.5,
        max_std=1.5,
    )

    value_function = GaussianMLPValueFunction(
        env_spec=env[0]().spec,
        hidden_sizes=[128, 128],
        hidden_nonlinearity=torch.tanh,
        init_std=1,
    )

    sampler = LocalSampler(agents=policy,
                           envs=env[0](),
                           max_episode_length=500,
                           is_tf_worker=True)

    algo = PPO(
        env_spec=env[0]().spec,
        policy=policy,
        value_function=value_function,
        sampler=sampler,
        discount=0.99,
        center_adv=False,
        stop_entropy_gradient=True,
        gae_lambda=0.95,
        lr_clip_range=0.2,
        policy_optimizer=OptimizerWrapper((torch.optim.Adam, {'lr' : 5e-4}), policy),
        vf_optimizer=OptimizerWrapper((torch.optim.Adam, {'lr' : 5e-4}), policy),
        entropy_method='max',
        test_env_sampler=test_env_sampler,
    )

    trainer.setup(algo, env)

    trainer.train(n_epochs=4000, batch_size=5000)


train_ppo_metaworld()

