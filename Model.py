import tensorflow as tf
from tensorflow.keras import layers


def get_input_emb_layers(act_dim, obs_dim, kernel_init, input_emb_config, past_act=None):
    output_dim_act_emb, output_dim_obs_emb = 0, 0
    if past_act is not None:
        # Input Embedding for memory network
        ragged_input = True
        input_act_shape, input_obs_shape = (None, act_dim), (None, obs_dim)
    else:
        # Input Embedding for features network
        ragged_input = None
        input_act_shape, input_obs_shape = act_dim, obs_dim

    input_actor_embedding_net = tf.keras.Sequential()
    input_actor_embedding_net.add(layers.Input(shape=input_act_shape, ragged=ragged_input))
    for hid_layer_size in input_emb_config["input_emb_act"]:
        input_actor_embedding_net.add(layers.Dense(hid_layer_size, activation="relu",
                                                   kernel_initializer=kernel_init))
    if input_emb_config["input_emb_act"]:
        output_dim_act_emb += input_emb_config["input_emb_act"][-1]

    input_observation_embedding_net = tf.keras.Sequential()
    input_observation_embedding_net.add(layers.Input(shape=input_obs_shape, ragged=ragged_input))
    for hid_layer_size in input_emb_config["input_emb_obs"]:
        input_observation_embedding_net.add(layers.Dense(hid_layer_size, activation="relu",
                                                         kernel_initializer=kernel_init))
    if input_emb_config["input_emb_obs"]:
        output_dim_obs_emb += input_emb_config["input_emb_obs"][-1]

    if past_act is not None:
        if past_act:
            # Return both embedding net
            dim_output = output_dim_act_emb + output_dim_obs_emb
            return input_actor_embedding_net, input_observation_embedding_net, dim_output
        else:
            # Return only observation net
            dim_output = output_dim_obs_emb
            return None, input_observation_embedding_net, dim_output
    else:
        # Return both embedding net for features case
        dim_output = output_dim_act_emb + output_dim_obs_emb
        return input_actor_embedding_net, input_observation_embedding_net, dim_output


class ActorCriticModel(tf.keras.Model):
    """
    Combined actor-critic Model.
    The following implementation follow the structure described by this paper https://arxiv.org/pdf/2102.12344.pdf,
    with the addition of input embedding layers for Wolpertinger architecture in case for discrete action space, applied
    both at input of memory and features network. For Wolpertinger implementation ,
    https://intellabs.github.io/coach/components/agents/policy_optimization/wolpertinger.html?highlight=wolpertinger.
    """

    def __init__(self, obs_dim, config_net: dict, act_dim, actions_space_continuous, model_type):
        """Initialize."""
        super().__init__()
        self.model_type = model_type
        self.action_space_continuous = actions_space_continuous
        self.mem_net_config = config_net["memory_net"]
        self.fet_net_config = config_net["features_net"]
        self.comb_net_config = config_net["combination_net"]
        self.hist_past_act = config_net["hist_past_act"]

        if self.hist_past_act:
            input_dim_mem_net = obs_dim + act_dim
        else:
            input_dim_mem_net = obs_dim
        if self.model_type == "critic":
            self.output_dim = 1
            input_dim_fet_net = obs_dim + act_dim
            # I.e. use linear activation function according to documentation
            activation_output = None
        elif self.model_type == "actor":
            self.output_dim = act_dim
            input_dim_fet_net = obs_dim
            activation_output = "tanh"

        kernel_init = "glorot_uniform"

        # Memory
        # In case of discrete action space both batched observation and action in input of memory network have to
        # embedded separately, according to Wolpertinger network architecture.
        if not self.action_space_continuous:
            input_emb_act_net, \
            input_emb_obs_net, \
            dim_output_emb = get_input_emb_layers(act_dim, obs_dim,
                                                  kernel_init,
                                                  input_emb_config=self.mem_net_config,
                                                  past_act=self.hist_past_act)

            self.input_actor_memory_embedding = input_emb_act_net
            self.input_observation_memory_embedding = input_emb_obs_net
            input_dim_mem_net = dim_output_emb if dim_output_emb != 0 else input_dim_mem_net

        # pre-RNN
        self.memory_network = tf.keras.Sequential()
        self.memory_network.add(layers.Input(shape=(None, input_dim_mem_net), ragged=True))
        for hid_layer_size in self.mem_net_config["pre_rnn_hid_sizes"]:
            self.memory_network.add(layers.Dense(hid_layer_size, activation="relu",
                                                 kernel_initializer=kernel_init))

        # RNN
        if len(self.mem_net_config["rnn_hid_sizes"]) >= 1:
            rnn_cells = []
            for hid_layer_size in self.mem_net_config["rnn_hid_sizes"]:
                rnn_cells.append(layers.LSTMCell(units=hid_layer_size, kernel_initializer=kernel_init))
            self.memory_network.add(layers.RNN(cell=rnn_cells))

        # post-RNN
        for hid_layer_size in self.mem_net_config["post_rnn_hid_sizes"]:
            self.memory_network.add(layers.Dense(hid_layer_size, activation="relu",
                                                 kernel_initializer=kernel_init))

        # Feature extraction
        # According to Wolpertinger architecture observations and actions have to embedded separately.
        if not self.action_space_continuous and self.model_type == "critic":
            input_emb_act_net, \
            input_emb_obs_net, \
            dim_output_emb = get_input_emb_layers(act_dim, obs_dim,
                                                  kernel_init,
                                                  input_emb_config=self.fet_net_config,
                                                  past_act=None)
            self.input_actor_features_embedding = input_emb_act_net
            self.input_observation_features_embedding = input_emb_obs_net
            input_dim_fet_net = dim_output_emb if dim_output_emb != 0 else input_dim_fet_net

        self.features_network = tf.keras.Sequential()
        self.features_network.add(layers.Input(shape=input_dim_fet_net))
        for hid_layer_size in self.fet_net_config["fet_ext_hid_sizes"]:
            self.features_network.add(layers.Dense(hid_layer_size, activation="relu",
                                                   kernel_initializer=kernel_init))

        # Combination of memory and feature extraction

        self.combination_network = tf.keras.Sequential()
        # Retrieve the name of the most recent block of memory network that has a non-empty value for output units
        # and accordingly get value.
        layer_name = [layers_name for layers_name, layers_struct in self.mem_net_config.items() if layers_struct]
        output_size_fet_net = self.fet_net_config["fet_ext_hid_sizes"][-1] if self.fet_net_config[
            "fet_ext_hid_sizes"] else input_dim_fet_net
        input_dim_comb_net = self.mem_net_config[layer_name[-1]][-1] + output_size_fet_net
        self.combination_network.add(layers.Input(shape=input_dim_comb_net))

        for hid_layer_size in self.comb_net_config["comb_net_hid_sizes"]:
            self.combination_network.add(layers.Dense(hid_layer_size, activation="relu",
                                                      kernel_initializer=kernel_init))

        self.combination_network.add(layers.Dense(self.output_dim, activation=activation_output,
                                                  kernel_initializer=kernel_init))

    def call(self, inputs: dict) -> tf.Tensor:
        memory = inputs["memory"]
        features = inputs["features"]
        hist_obs = memory["obs"]
        hist_act = memory["act"]

        obs = features["obs"]
        act = features["act"]

        if self.action_space_continuous:
            if self.hist_past_act:
                memory_inputs = tf.concat([hist_obs, hist_act], -1)
            else:
                memory_inputs = hist_obs
        else:
            if self.hist_past_act:
                out_emb_act_mem = self.input_actor_memory_embedding(hist_act)
                out_emb_obs_mem = self.input_observation_memory_embedding(hist_obs)
                memory_inputs = tf.concat([out_emb_obs_mem, out_emb_act_mem], -1)
            else:
                memory_inputs = self.input_observation_memory_embedding(hist_obs)

        # Feed memory net
        memory_outputs = self.memory_network(memory_inputs)

        # Feed extraction net

        if self.model_type == "critic":
            if self.action_space_continuous:
                features_inputs = tf.concat([obs, act], -1)
            else:
                out_emb_act_fet = self.input_actor_features_embedding(act)
                out_emb_obs_fet = self.input_observation_features_embedding(obs)
                features_inputs = tf.concat([out_emb_obs_fet, out_emb_act_fet], -1)
        elif self.model_type == "actor":
            features_inputs = obs

        features_outputs = self.features_network(features_inputs)

        # Post-combination
        comb_inputs = tf.concat([memory_outputs, features_outputs], -1)
        model_output = self.combination_network(comb_inputs)

        return model_output
