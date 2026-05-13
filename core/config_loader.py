import yaml
import os

def load_project_config(project_name):
    path = os.path.join("projects", project_name, "config.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found at {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    if config is None:
        config = {}
    
    config['project_name'] = project_name
    return config