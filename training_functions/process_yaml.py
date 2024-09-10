import sys
sys.path.append('..')

import yaml

def process_yaml_for_training(yaml_path):

    def merge_dicts(list_of_dicts):
        merged = {}
        for item in list_of_dicts:
            for key, value in item.items():
                if isinstance(value, list) and isinstance(value[0], dict):
                    # If the value is a list of dictionaries, merge them recursively
                    merged[key] = merge_dicts(value)
                else:
                    # Otherwise, just add the key-value pair
                    merged[key] = value
        return merged

    data = None
    with open(yaml_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
    
    hierarchical_dict = {k: merge_dicts(v) for k, v in data.items()}

    return hierarchical_dict

    


