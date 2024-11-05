from copy import deepcopy

def average_state_dicts(state_dicts):
    avg_state_dict = {}
    for key in state_dicts[0]:
        avg_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)
    return avg_state_dict

def interpolate_state_dicts(old, new, epsilon):
    result_state_dict = {}
    for key in old:
        result_state_dict[key] = old[key] + (new[key] - old[key]) * epsilon
    return result_state_dict

    # add_state_dicts(old, scale_state_dicts(subtract_state_dicts(new, old), epsilon))

# def add_state_dicts(*state_dicts):
#     result_state_dict = deepcopy(state_dicts[0])
#     for key in result_state_dict:
#         result_state_dict[key] = sum(d[key] for d in state_dicts)
#     return result_state_dict

# def subtract_state_dicts(state_dict1, state_dict2):
#     result_state_dict = deepcopy(state_dict1)
#     for key in result_state_dict:
#         result_state_dict[key] -= state_dict2[key]
#     return result_state_dict

# def scale_state_dicts(state_dict, scale):
#     result_state_dict = deepcopy(state_dict)
#     for key in result_state_dict:
#         result_state_dict[key] *= scale
#     return result_state_dict
