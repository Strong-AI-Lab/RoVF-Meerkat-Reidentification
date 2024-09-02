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
    
    # handle none string values.    
    #if isinstance(data["training_details"]["clip_value"], str):
    #    assert data["training_details"]["clip_value"].lower() == "none", "clip_value must be a float|int or 'none'."
    #    data["training_details"]["clip_value"] = None
    

    #if isinstance(data["dataloader_details"]["override_value"], str):
    #    assert data["dataloader_details"]["override_value"].lower() == "none", "override_value must be a float|int or 'none'."
    #    data["dataloader_details"]["override_value"] = None

    return hierarchical_dict

    


