import yaml

def get_yaml_data(filepath):
    with open(filepath, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data
