import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity, device):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch_size = 1  # Because one element is added for each episode step
        self.batch_length = capacity
        data_spec = (
            tf.TensorSpec([obs_dim], dtype=tf.float32, name='observation'),
            tf.TensorSpec([act_dim], dtype=tf.float32, name='action'),
            tf.TensorSpec([], dtype=tf.float32, name='reward'),
            tf.TensorSpec([obs_dim], dtype=tf.float32, name='next_observation'),
            tf.TensorSpec([], dtype=tf.float32, name='done'),
        )
        self.buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec, batch_size=self.batch_size, max_length=self.batch_length, device=device)

    def put(self, observation, action, reward, next_observation, done):
        self.buffer.add_batch((observation, action, reward, next_observation, done))

    @staticmethod
    def get_boolean_mask(done_sequence, batch_size, max_hist_len):
        """
        This method create a ragged tensor of indices starting from the done sequence preceding elements at
        given time t. This serves to correctly associate, for each element, its experience, in order to
        not include also elements belonging to other episodes.
        For example if done batch sequence is [[0, 1, 0, 0, 0], [0, 0, 0, 1, 0]] the returned ragged tensor should be
        [[2, 3, 4], [4]], since elements where done is 1 are not considered.
        The special case in which one sequence could end with 1, means that element at time t represent the first one
        in the sampled episode, so no prior experience could be retrieved. In this case also the indices at which this
        happens will be returned in order to add a zero sequence of length one in the final batched sample.

        :param done_sequence: tensor of dimension [batch_size, max_hist_len]
        :param batch_size: number of rows for done batch
        :param max_hist_len: number of columns for done batch
        :return: ragged tensor of indices of dimension [batch_size, None], rows indices where no prior experience occurs
        """
        coordinates_true_done = tf.where(done_sequence == 1)
        unique_obj = tf.unique_with_counts(coordinates_true_done[:, 0])
        rows_with_ones = unique_obj.y

        lengths_rows_with_ones = unique_obj.count
        row_lengths = tf.zeros(shape=[batch_size], dtype=tf.int32)
        row_lengths = tf.tensor_scatter_nd_update(row_lengths, tf.expand_dims(rows_with_ones, axis=-1),
                                                  lengths_rows_with_ones)

        tmp = tf.RaggedTensor.from_row_lengths(coordinates_true_done, row_lengths)
        max_index_for_row = tf.reduce_max(tmp, axis=1)[:, 1:]
        max_index_for_row = tf.where(max_index_for_row >= 0, tf.math.add(max_index_for_row, 1), 0)

        row_indexes_no_mem = tf.where(max_index_for_row[:, 0] == max_hist_len)
        tmp_mod = tf.where(max_index_for_row == max_hist_len, tf.math.add(max_index_for_row, -1), max_index_for_row)
        ragged_indices_to_preserve = tf.ragged.range(starts=tmp_mod[:, 0], limits=max_hist_len)

        return ragged_indices_to_preserve, row_indexes_no_mem

    @tf.function
    def sample_batch_with_history(self, batch_size, max_hist_len):
        """
        Get a random batch sample. The total sequence length sampled is  max_hist_len + 2 , this because
        max_hist_len elements represent the history of observation at time t, present at max_hist_len + 1.
        The last max_hist_len + 2 index represent elements at time t+1, used for fetch history of action at time t+1.
        Gave as result a dictionary including ragged batched tensor for each element stored in replay buffer.

        :param batch_size: number of element in batch
        :param max_hist_len: sequence length
        :return: dictionary including the sampled quantities
        """

        sampled_batch_with_history = self.buffer.get_next(sample_batch_size=batch_size, num_steps=max_hist_len + 2)

        # According to data spec of replay buffer in constructor
        # 0 - Observation at time t
        # 1 - Action at time t
        # 2 - Reward at time t
        # 3 - Observation at time t+1
        # 4 - Done signal at time t

        obs_batch = sampled_batch_with_history[0][0][:, -2, :]
        act_batch = sampled_batch_with_history[0][1][:, -2, :]
        rew_batch = tf.expand_dims(sampled_batch_with_history[0][2][:, -2], axis=-1)
        next_obs_batch = sampled_batch_with_history[0][3][:, -2, :]
        done_batch = tf.expand_dims(sampled_batch_with_history[0][4][:, -2], axis=-1)

        # Two boolean mask are needed, one for time t and one for time t+1
        done_history_seq_t = sampled_batch_with_history[0][4][:, :max_hist_len]
        done_history_seq_next_t = sampled_batch_with_history[0][4][:, 1:max_hist_len + 1]

        range_indices_t, indexes_no_mem_t = self.get_boolean_mask(done_history_seq_t, batch_size, max_hist_len)
        range_indices_next_t, indexes_no_mem_next_t = self.get_boolean_mask(done_history_seq_next_t, batch_size,
                                                                            max_hist_len)

        hist_obs_batch = sampled_batch_with_history[0][0][:, :max_hist_len, :]
        hist_act_batch = sampled_batch_with_history[0][1][:, :max_hist_len, :]
        hist_next_obs_batch = sampled_batch_with_history[0][3][:, :max_hist_len, :]
        hist_next_act_batch = sampled_batch_with_history[0][1][:, 1:max_hist_len + 1, :]

        if len(indexes_no_mem_t) != 0:
            num_seq_to_update = len(indexes_no_mem_t)
            hist_obs_batch = tf.tensor_scatter_nd_update(hist_obs_batch,
                                                         indexes_no_mem_t,
                                                         tf.zeros([num_seq_to_update, max_hist_len, self.obs_dim]))
            hist_act_batch = tf.tensor_scatter_nd_update(hist_act_batch,
                                                         indexes_no_mem_t,
                                                         tf.zeros([num_seq_to_update, max_hist_len, self.act_dim]))

        if len(indexes_no_mem_next_t) != 0:
            num_seq_to_update = len(indexes_no_mem_next_t)
            hist_next_obs_batch = tf.tensor_scatter_nd_update(hist_next_obs_batch,
                                                              indexes_no_mem_next_t,
                                                              tf.zeros([num_seq_to_update, max_hist_len, self.obs_dim]))

            hist_next_act_batch = tf.tensor_scatter_nd_update(hist_next_act_batch,
                                                              indexes_no_mem_next_t,
                                                              tf.zeros([num_seq_to_update, max_hist_len, self.act_dim]))

        hist_obs_batch = tf.gather(hist_obs_batch, range_indices_t, batch_dims=1)
        hist_act_batch = tf.gather(hist_act_batch, range_indices_t, batch_dims=1)
        hist_next_obs_batch = tf.gather(hist_next_obs_batch, range_indices_next_t, batch_dims=1)
        hist_next_act_batch = tf.gather(hist_next_act_batch, range_indices_next_t, batch_dims=1)

        batch_sampled = {"obs": obs_batch,
                         "act": act_batch,
                         "rew": rew_batch,
                         "next_obs": next_obs_batch,
                         "done": done_batch,
                         "hist_obs": hist_obs_batch,
                         "hist_act": hist_act_batch,
                         "hist_next_obs": hist_next_obs_batch,
                         "hist_next_act": hist_next_act_batch}

        return batch_sampled
