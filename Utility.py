import os
import toml


def get_configuration(environment_name, pomdp_type, config_filename):

    with open(config_filename, 'r') as f:
        actual_config = toml.load(f)

    dir_checkpoints = "ckpts/" + environment_name + "/" + pomdp_type.lower() + "/"

    # Check for already used configurations in order to continue saved runs

    used_configurations = None
    try:
        with open(dir_checkpoints + 'used_configurations.toml', 'r') as f:
            used_configurations = toml.load(f)
    except FileNotFoundError:
        os.makedirs(dir_checkpoints, exist_ok=True)
        used_configurations = {}

    new_configuration = False
    if not used_configurations:
        num_config = 0
        used_configurations[str(num_config)] = actual_config
        dir_checkpoints += "run_config_0/"
        new_configuration = True
    else:
        if actual_config in used_configurations.values():
            num_config = list(used_configurations.values()).index(actual_config)
        else:
            num_config = len(used_configurations.keys())
            used_configurations[str(num_config)] = actual_config
            new_configuration = True
        dir_checkpoints += "run_config_" + str(num_config) + "/"

    if new_configuration:
        os.makedirs(dir_checkpoints, exist_ok=True)
        with open(dir_checkpoints + '../used_configurations.toml', 'w') as f:
            toml.dump(used_configurations, f)

        

#         with open(dir_checkpoints + 'config_' + str(num_config) + ".toml", 'w') as f:
#             toml.dump(actual_config, f)

    return actual_config, dir_checkpoints
