import pybulletgym  # To register tasks in PyBulletGym
import pybullet_envs  # To register tasks in PyBullet
import gym

import statistics
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from Agent import ActorCriticAgent as ACAgent
from environment.env_wrapper import POMDPWrapper
from Memory import ReplayBuffer
import tensorflow as tf
import numpy as np
import tqdm
import Utility
import argparse


def update_critic_parameters(data, optimizer, discount_factor, main_agent, target_agent):
    # Input at time t
    inputs_obs = {"memory": {"obs": data['hist_obs'],
                             "act": data['hist_act']},
                  "features": {"obs": data['obs'],
                               "act": data['act']}}

    # Input at time t+1
    inputs_next_obs = {"memory": {"obs": data['hist_next_obs'],
                                  "act": data['hist_next_act']},
                       "features": {"obs": data['next_obs'],
                                    "act": None}}
    rewards_batch = data["rew"]
    done_batch = data["done"]

    _, next_act = target_agent.get_action(inputs_next_obs)

    inputs_next_obs["features"]["act"] = next_act
    # Target Q-values
    q1_pi_targ = target_agent.critic_1(inputs_next_obs)
    q2_pi_targ = target_agent.critic_2(inputs_next_obs)
    q_pi_targ = tf.math.minimum(q1_pi_targ, q2_pi_targ)

    backup = rewards_batch + discount_factor * (1 - done_batch) * q_pi_targ

    mse = tf.keras.losses.MeanSquaredError()

    # Compute critic loss for both critic network for main agent
    with tf.GradientTape() as tape:
        q1 = main_agent.critic_1(inputs_obs)
        q2 = main_agent.critic_2(inputs_obs)

        # MSE loss against Bellman backup
        loss_q1 = mse(backup, q1)
        loss_q2 = mse(backup, q2)
        loss_q = loss_q1 + loss_q2

    # Update trainable parameters
    q1_q2_parameters = main_agent.critic_1.trainable_variables + main_agent.critic_2.trainable_variables
    grads = tape.gradient(loss_q, q1_q2_parameters)
    optimizer.apply_gradients(zip(grads, q1_q2_parameters))

    return loss_q


def update_actor_parameters(data, optimizer, main_agent):
    inputs_obs = {"memory": {"obs": data['hist_obs'],
                             "act": data['hist_act']},
                  "features": {"obs": data['obs'],
                               "act": None}}

    # Compute actor loss for actor network. Here only parameters of actor net are watched by tape since critic
    # parameters are not required to be updated, but only used for compute the actor loss.
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(main_agent.actor.trainable_variables)
        act_logits = main_agent.actor(inputs_obs)
        inputs_obs["features"]["act"] = act_logits
        q1_pi = main_agent.critic_1(inputs_obs)
        loss_act = tf.reduce_mean(-q1_pi)

    grads = tape.gradient(loss_act, main_agent.actor.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_agent.actor.trainable_variables))

    return loss_act


@tf.function
def update(main_agent, target_agent, upd_counter, batch_sizes, max_history_length,
           freq_policy_update, tau_target_update, disc_factor, critic_optimizer, actor_optimizer):
    loss_q, loss_a = tf.constant(0, dtype=tf.float32), tf.constant(0, dtype=tf.float32)
    sampled_batch = replay_buffer.sample_batch_with_history(batch_size=batch_sizes,
                                                            max_hist_len=max_history_length)

    loss_q = update_critic_parameters(sampled_batch, critic_optimizer, disc_factor, main_agent, target_agent)

    # Delayed policy updates
    if upd_counter % freq_policy_update == 0:
        loss_a = update_actor_parameters(sampled_batch, actor_optimizer, main_agent)

        # Soft update of target parameters according to small tau value
        for par_main, par_targ in zip(main_agent.trainable_variables,
                                      target_agent.trainable_variables):
            par_targ.assign(tau_target_update * par_main + (1 - tau_target_update) * par_targ)

    return loss_q, loss_a


@tf.function
def env_interaction(input_act, main_agent, buffer_l):
    """
    Interaction with environment has been performed through an input dictionary that contains tensors for memory
    and actor network. This method after the interaction returns reward step, done signal and next input
    dictionary, in which  last buffer_l observations and actions are stored, along with observation for the next step.
    :param input_act: input dictionary with observation at time t and previous observations and actions
    :param main_agent: Agent that perform action
    :param buffer_l: number of previous elements to take in consideration
    :return: next_input dictionary for next step
    """
    observation = input_act["features"]["obs"]
    next_input_act = {"memory": {"obs": None,
                                 "act": None},
                      "features": {"obs": None,
                                   "act": None}}

    action, raw_action = main_agent.get_action(input_act)

    next_observation, reward, done = env.tf_step(action)

    next_input_act["features"]["obs"] = next_observation
    last_hist_obs, last_hist_act = input_act["memory"]["obs"], input_act["memory"]["act"]

    observation = tf.expand_dims(observation, axis=0)
    raw_action = tf.expand_dims(raw_action, axis=0)
    # For each iteration stack the last action and observation, keeping only the last buffer_l elements
    next_input_act["memory"]["obs"] = tf.concat([last_hist_obs, observation], axis=1)[:, -buffer_l:, :]
    next_input_act["memory"]["act"] = tf.concat([last_hist_act, raw_action], axis=1)[:, -buffer_l:, :]

    return next_input_act, reward, done


def run_warmup_episodes(num_episodes_warmup):
    """
    Warmup episodes are executed in order to feed Replay buffer memory.
    Randomly selected actions used for interact with environment.
    :param num_episodes_warmup:
    """
    with tqdm.trange(num_episodes_warmup) as warm_episodes:
        for warm_ep in warm_episodes:
            obs_warm = tf.constant(env.reset(), shape=(1, obs_dim), dtype=tf.float32)
            for warm_step in range(episode_steps):
                if continuous:
                    action = tf.convert_to_tensor(env.action_space.sample())
                    raw_action = tf.expand_dims(action, axis=0)
                else:
                    proto_action = tf.random.uniform(minval=lower_act_limit, maxval=upper_act_limit,
                                                     shape=(1, act_dim), dtype=tf.float64)
                    knn = tf.constant(1, dtype=tf.int32)
                    raw_action, action = actor_critic_main.action_space.tf_search_point(proto_action, knn)
                    raw_action = tf.cast(raw_action, dtype=tf.float32)
                    action = tf.squeeze(action)

                next_obs_warm, reward_warm, done_warm = env.tf_step(action)
                done_warm = tf.ones_like(done_warm) if warm_step == episode_steps - 1 else done_warm
                replay_buffer.put(obs_warm, raw_action, reward_warm, next_obs_warm, done_warm)
                obs_warm = next_obs_warm

                if tf.cast(done_warm, dtype=tf.bool):
                    break

            warm_episodes.set_description(f"Warmup episode:[{warm_ep}]")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help="Seed for experiment reproducibility")
    parser.add_argument('--env_name', type=str, default='LunarLander-v2', help="Environment name")
    parser.add_argument('--pomdp_type', type=str, default='remove_velocity', help="Type of pomdp observation")
    parser.add_argument('--config_filename', type=str, default='config_net.toml', help="Name configuration file")
    parser.add_argument('--num_episodes', type=int, default=3000, help="Number of episodes")
    parser.add_argument('--num_warmup_episodes', type=int, default=10, help="Number of warmup episodes")
    args = parser.parse_args()

    # Avoid warnings to be displayed on console
    tf.get_logger().setLevel('ERROR')
    tf.config.run_functions_eagerly(False)

    # Create the environment
    env = POMDPWrapper(env_name=args.env_name, pomdp_type=args.pomdp_type)

    # Set seed
    env.seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    device_name = tf.test.gpu_device_name()

    continuous = None
    try:  # TD3 continuous action space - normal way
        obs_dim = env.observation_space.shape[0] if len(env.observation_space.shape) != 0 else 1
        act_dim = env.action_space.shape[0]
        num_actions = np.inf  # Not used np.inf is only to indicate that are infinite number of actions
        upper_act_limit = env.action_space.high
        lower_act_limit = env.action_space.low
        continuous = True
    except IndexError:  # TD3 discrete action space using Wolpertinger agent
        obs_dim = env.observation_space.shape[0] if len(env.observation_space.shape) != 0 else 1
        act_dim = env.action_space.shape[0] if len(env.action_space.shape) != 0 else 1
        num_actions = env.action_space.n
        lower_act_limit = -1.0
        upper_act_limit = 1.0
        continuous = False

    config_net, dir_checkpoints = Utility.get_configuration(args.env_name, args.pomdp_type, args.config_filename)
    logger = tf.summary.create_file_writer(logdir=dir_checkpoints + "log_dir/", experimental_trackable=True,
                                           max_queue=10)

    # Hyper parameters from arg parsing

    num_episodes = args.num_episodes
    warmup_episodes = args.num_warmup_episodes

    # Hyper parameters from config file

    hyper_parameters = config_net["hyper_parameters"]
    knn_ratio = hyper_parameters["knn_ratio"]
    buffer_capacity = hyper_parameters["replay_buffer_capacity"]
    episode_steps = hyper_parameters["steps_per_episodes"]
    buffer_length = hyper_parameters["buffer_length"]
    gamma = hyper_parameters["discount_factor"]
    tau = hyper_parameters["target_update_rate"]
    lr_critic_parameters = hyper_parameters["learning_rate_critic"]
    lr_actor_parameters = hyper_parameters["learning_rate_actor"]
    batch_size = hyper_parameters["batch_sizes"]
    max_hist_length = hyper_parameters["history_length"]
    std_dev_act_inf = hyper_parameters["std_dev_actor"]
    std_dev_act_update = hyper_parameters["std_dev_actor_target"]
    clip_noise = hyper_parameters["clip_noise"]
    policy_delay = hyper_parameters["policy_delay"]

    # Agent initialization
    actor_critic_main = ACAgent(obs_dim, config_net, act_dim, continuous, upper_act_limit, lower_act_limit, num_actions,
                                knn_ratio, std_dev_act_inf, clip_noise=None, hist_len=None,
                                agent_type="main")
    actor_critic_target = ACAgent(obs_dim, config_net, act_dim, continuous, upper_act_limit, lower_act_limit,
                                  num_actions,
                                  knn_ratio, std_dev_act_update, clip_noise=clip_noise, hist_len=max_hist_length,
                                  agent_type="target")
    actor_critic_target.set_weights(actor_critic_main.get_weights())

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=buffer_capacity, device=device_name)

    # Parameters optimizers
    critic_opt = tf.keras.optimizers.Adam(learning_rate=lr_critic_parameters)
    actor_opt = tf.keras.optimizers.Adam(learning_rate=lr_actor_parameters)

    start_episode = tf.Variable(0)
    # Count each time an update occurs, and save this variable in checkpoint in order to correctly restart updating.
    update_counter = tf.Variable(0)

    checkpoint = tf.train.Checkpoint(main_model=actor_critic_main, target_model=actor_critic_target,
                                     rep_buffer=replay_buffer.buffer, critic_optimizer=critic_opt,
                                     actor_optimizer=actor_opt, ep_number=start_episode, update_counter=update_counter,
                                     logger=logger)

    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=dir_checkpoints, max_to_keep=1)

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
    else:
        # If no previous experience, then execute warmup episodes in order to store data in Replay buffer
        run_warmup_episodes(warmup_episodes)
        start_episode.assign(warmup_episodes - 1)

    start_episode.assign_add(1)

    with tqdm.trange(start_episode.numpy(), num_episodes) as episodes:
        for episode in episodes:
            obs = tf.expand_dims(tf.convert_to_tensor(env.reset(), dtype=tf.float32), axis=0)
            ep_reward = tf.Variable(initial_value=(0,), dtype=tf.float32, shape=(1,))
            ep_length = 0
            loss_critic_values, loss_actor_values = [], []

            act_buffer = tf.Variable(1)
            # Set initial input to actor model where previous experience has been set to zero
            input_act_selection = {"memory": {"obs": tf.zeros([1, 1, obs_dim], dtype=tf.float32),
                                              "act": tf.zeros([1, 1, act_dim], dtype=tf.float32)},
                                   "features": {"obs": obs,
                                                "act": None}}

            with logger.as_default(step=episode):

                for step in tf.range(episode_steps):

                    next_input_act_selection, step_reward, done_step = env_interaction(input_act_selection,
                                                                                       actor_critic_main,
                                                                                       act_buffer)
                    ep_reward.assign_add(step_reward)
                    ep_length += 1

                    obs = input_act_selection["features"]["obs"]
                    next_obs = next_input_act_selection["features"]["obs"]
                    # Get last action, correspondent to the action performed at current step
                    act = next_input_act_selection["memory"]["act"][0][-1:, :]

                    # Force done to one when time horizon has been reached
                    done_step = tf.ones_like(done_step) if step == episode_steps - 1 else done_step
                    replay_buffer.put(obs, act, step_reward, next_obs, done_step)

                    input_act_selection = next_input_act_selection
                    act_buffer.assign(buffer_length)

                    # Update parameters each  step
                    update_counter.assign_add(1)
                    loss_critic, loss_actor = update(actor_critic_main, actor_critic_target, update_counter,
                                                     batch_size, max_hist_length, policy_delay, tau, gamma,
                                                     critic_opt, actor_opt)

                    loss_critic_values.append(loss_critic.numpy())
                    if update_counter.numpy() % policy_delay == 0:
                        loss_actor_values.append(loss_actor.numpy())

                    if tf.cast(done_step, dtype=tf.bool):
                        break

                checkpoint_manager.save()
                checkpoint.ep_number.assign_add(1)
                episodes.set_description(f"Episode:[{episode}]")
                episodes.set_postfix(episode_reward=ep_reward.numpy()[0], episode_length=ep_length)
                tf.summary.scalar(name="Ep_reward", data=ep_reward.numpy()[0], step=episode)
                tf.summary.scalar(name="Ep_length", data=ep_length, step=episode)
                tf.summary.scalar(name="Loss_critic_mean", data=statistics.mean(loss_critic_values), step=episode)
                tf.summary.scalar(name="Loss_actor_mean", data=statistics.mean(loss_actor_values), step=episode)
