from json import load
from matplotlib.pyplot import get
from learning.utils import load_jsonL_file, save_json_file
from option import get_args
from baseline import main as baseline_entry

def get_exp_config(experiment_config_path):
    experiment_config = load_jsonL_file(json_file_path=experiment_config_path)
    general_config = experiment_config['general']
    args = get_args()
    for parameter in general_config:
        args.__dict__[str(parameter)] = general_config[parameter]
    
    return args




if __name__=="__main__":
    experiment_config = load_jsonL_file(json_file_path='./experiments.json')
    general_config = experiment_config['general']
    experiments_config_list = experiment_config['eperiments']
    args = get_args()

    for experiment_config in experiments_config_list:
        # reset dafult params
        for parameter in general_config:
            args.__dict__[str(parameter)] = general_config[parameter]


        # overwrite default with each experiment specifics
        for parameter in experiment_config:
            args.__dict__[str(parameter)] = experiment_config[parameter]
        
        baseline_entry(args)
        
        
