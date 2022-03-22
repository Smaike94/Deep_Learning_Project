import tensorflow as tf
from Model import ActorCriticModel
from environment.action_space_TD3 import Discrete_space


class ActorCriticAgent(tf.keras.Model):
    """
    Combined actor-critic network.
    According to TD3 algorithm one actor and two critic network are needed.
    Two type of agent are possible: main and target. They differ for how an action has been selected.
    For main agent the output of actor network has been added to an exploration noise with defined standard deviation.
    Instead, for target type the noise, with its defined standard deviation, has to be clipped before to be added
    to actor output network, according to target policy smoothing described in TD3 algorithm.
    Furthermore, TD3 relies on DDPG that works only for continuous action space, so in order to deal also with discrete
    action space, this implementation of TD3 has been extended including also Wolpertinger policy.
    This variation, after the applications of noise to actor output(the so-called proto-action), performs a further step
    embedding the proto-action into a k-nearest-neighbor mapping, that reduce the continuous proto-action into a
    discrete set.
    In both continuous and discrete action space this action selection has been performed by main and target agent,
    respectively during environment interaction and during the update of critic parameters. Instead, in both cases
    the parameters of actor network are updated taking into account the output of actor network without adding any kind
    of noise.
    """

    def __init__(self, obs_dim, config_net, act_dim, act_space_cont, act_up_limit, act_low_limit,
                 actions, knn_ratio, st_dev_noise, clip_noise, hist_len, agent_type):
        """Initialize."""
        super().__init__()

        self.agent_type = agent_type
        self.actor = ActorCriticModel(obs_dim, config_net["actor_net"], act_dim, act_space_cont, model_type="actor")
        self.critic_1 = ActorCriticModel(obs_dim, config_net["critic_net"], act_dim, act_space_cont,
                                         model_type="critic")
        self.critic_2 = ActorCriticModel(obs_dim, config_net["critic_net"], act_dim, act_space_cont,
                                         model_type="critic")

        self.upper_act_limit = act_up_limit
        self.lower_act_limit = act_low_limit
        self.actions_space_continuous = act_space_cont
        self.std_deviation_act_noise = st_dev_noise
        self.clip_noise = clip_noise
        self.hist_length = hist_len

        if not self.actions_space_continuous:
            num_actions = actions
            self.action_space = Discrete_space(num_actions)
            self.knn = max(1, int(num_actions * knn_ratio))
            self.knn_tensor = tf.constant(self.knn, dtype=tf.int32)

    @tf.function
    def get_action(self, input_act_net):
        if self.actions_space_continuous:
            act = self.continuous_act(input_act_net)
            raw_act = act
            act = tf.squeeze(act)
        else:
            act, raw_act = self.wolp_act(input_act_net)
            act = tf.squeeze(act)
        return act, raw_act

    def continuous_act(self, inputs_act_net):
        action = self.actor(inputs_act_net)

        if self.agent_type == "main":
            action = tf.math.add(action, tf.random.normal(shape=action.shape, stddev=self.std_deviation_act_noise))
            action = tf.clip_by_value(action, clip_value_min=self.lower_act_limit, clip_value_max=self.upper_act_limit)

        elif self.agent_type == "target":
            target_policy_smooth = tf.random.normal(shape=action.shape, stddev=self.std_deviation_act_noise)
            target_policy_smooth = tf.clip_by_value(target_policy_smooth, clip_value_min=-self.clip_noise,
                                                    clip_value_max=self.clip_noise)
            action = tf.math.add(action, target_policy_smooth)
            action = tf.clip_by_value(action, clip_value_min=self.lower_act_limit, clip_value_max=self.upper_act_limit)

        return action

    def wolp_act(self, inputs_act_net):

        proto_action = self.actor(inputs_act_net)

        if self.agent_type == "main":
            proto_action = tf.math.add(proto_action, tf.random.normal(shape=proto_action.shape,
                                                                      stddev=self.std_deviation_act_noise))
            proto_action = tf.clip_by_value(proto_action, clip_value_min=self.lower_act_limit,
                                            clip_value_max=self.upper_act_limit)
        elif self.agent_type == "target":
            target_policy_smooth = tf.random.normal(shape=proto_action.shape, stddev=self.std_deviation_act_noise)
            target_policy_smooth = tf.clip_by_value(target_policy_smooth, clip_value_min=-self.clip_noise,
                                                    clip_value_max=self.clip_noise)
            proto_action = tf.math.add(proto_action, target_policy_smooth)
            proto_action = tf.clip_by_value(proto_action, clip_value_min=self.lower_act_limit,
                                            clip_value_max=self.upper_act_limit)

        proto_action_input = tf.cast(proto_action, dtype=tf.float64)
        # Raw actions are real values that represent one of the possible discrete set of actions.
        # Actions are the integer values that represent one of the possible actions.
        # Raw actions are used to be stored in replay buffer and used to feed networks, instead actions are used
        #  to interact with environment.
        raw_actions, actions = self.action_space.tf_search_point(proto_action_input, self.knn_tensor)

        obs = inputs_act_net["features"]["obs"]
        last_obs = inputs_act_net["memory"]["obs"]
        last_act = inputs_act_net["memory"]["act"]
        num_raw_actions = tf.cast(obs.get_shape()[0], dtype=tf.int64)

        raw_actions = tf.cast(raw_actions, dtype=tf.float32)

        if self.knn > 1:
            # Since if knn is greater than one for each observation, along with its history sequence, there will be a
            # raw actions equal to knn. So, in order to evaluate which raw action is the best, i.e. gave highest
            # Q value, it's necessary to tile, according to the value of knn, each observation and its history sequence,
            # in order to correctly form pairs of observation and raw action.
            # The principle of tile process is the same both in inference that update case, but this last one
            # is more complicated due to ragged tensor, so further operations have to be done.

            obs_dim, act_dim = last_obs.shape[2], last_act.shape[2]
            s_t = tf.tile(obs, tf.constant(value=[1, self.knn]))
            s_t = tf.reshape(s_t, shape=[num_raw_actions, self.knn, obs_dim])

            if isinstance(last_obs, tf.RaggedTensor) and isinstance(last_act, tf.RaggedTensor):  # Update case
                last_obs_not_ragged, last_act_not_ragged = last_obs.to_tensor(), last_act.to_tensor()
                max_hist_len = tf.cast(self.hist_length, dtype=tf.int64)

                # Create boolean mask
                # Making ones
                total_ones = tf.reduce_sum(last_obs.row_lengths())
                ragged_ones = tf.RaggedTensor.from_row_lengths(tf.ones([total_ones]),
                                                               row_lengths=last_obs.row_lengths())
                # Making zeros
                total_zeros = (max_hist_len * num_raw_actions) - total_ones
                zeros_row_lengths = max_hist_len - last_obs.row_lengths()
                ragged_zeros = tf.RaggedTensor.from_row_lengths(tf.zeros([total_zeros]), row_lengths=zeros_row_lengths)

                bool_mask = tf.cast(tf.concat([ragged_ones, ragged_zeros], axis=1), dtype=tf.bool)
                bool_mask = bool_mask.to_tensor()
                bool_mask = tf.tile(bool_mask, tf.constant(value=[1, self.knn]))
                bool_mask = tf.reshape(bool_mask, shape=[num_raw_actions, self.knn, max_hist_len])

                # Apply boolean mask to last_obs  and last_act tiled and reshaped
                last_obs_not_ragged_tile = tf.tile(last_obs_not_ragged, tf.constant(value=[1, self.knn, 1]))
                last_obs_not_ragged_reshaped = tf.reshape(last_obs_not_ragged_tile, shape=[num_raw_actions, self.knn,
                                                                                           max_hist_len,
                                                                                           obs_dim])

                last_act_not_ragged_tile = tf.tile(last_act_not_ragged, tf.constant(value=[1, self.knn, 1]))
                last_act_not_ragged_reshaped = tf.reshape(last_act_not_ragged_tile, shape=[num_raw_actions, self.knn,
                                                                                           max_hist_len,
                                                                                           act_dim])
                # Obtain reshaped and tiled last_obs and last_act
                last_obs = tf.ragged.boolean_mask(last_obs_not_ragged_reshaped, bool_mask)
                last_act = tf.ragged.boolean_mask(last_act_not_ragged_reshaped, bool_mask)

            elif isinstance(last_obs, tf.Tensor) and isinstance(last_act, tf.Tensor):  # Inference case
                last_obs_tiled = tf.tile(last_obs, tf.constant(value=[1, self.knn, 1]))
                last_act_tiled = tf.tile(last_act, tf.constant(value=[1, self.knn, 1]))
                last_obs = tf.reshape(last_obs_tiled, shape=[num_raw_actions, self.knn,
                                                             last_obs.shape[1], obs_dim])
                last_act = tf.reshape(last_act_tiled, shape=[num_raw_actions, self.knn,
                                                             last_act.shape[1], act_dim])

            fn_to_map = lambda i: self.critic_1(dict(memory={"obs": last_obs[i],
                                                             "act": last_act[i]}, features={"obs": s_t[i],
                                                                                            "act": raw_actions[i]}))
            actions_evaluation = tf.map_fn(fn=fn_to_map, elems=tf.range(num_raw_actions), dtype=tf.float32)

            # Return the best action, i.e., wolpertinger action from the full wolpertinger policy
            max_index = tf.math.argmax(actions_evaluation, axis=1)
            raw_actions_max = tf.squeeze(tf.gather(raw_actions, indices=max_index, batch_dims=1), axis=1)
            actions_max = tf.gather(actions, indices=max_index, batch_dims=1)

        else:
            raw_actions_max = raw_actions
            actions_max = actions

        return actions_max, raw_actions_max

