import math
import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

hyperparameter_defaults  = {
        'epochs': 2,
        'batch_size': 2,
        'weight_decay': 0.0005,
        'optimizer': 'AdamW',
        'learning_rate': 1e-3,
        'seed': 24,
        'k_fold' : 0
}

sweep_config = {
    'name' : 'convnext_test',
    'method': 'bayes',
    'metric' : {
        'name': 'loss',
        'goal': 'minimize'   
    },
    'parameters' : {
        'epochs': {
            'values': [5, 6]
        },
        'batch_size': {
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(1),
            'max': math.log(4),
        },
        'optimizer': {
            'values': ['AdamW', 'SGD']
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        'k_fold':{
            'values': [0] #[0,1,2,3,4]
        }
    },
    'early_terminate':{
        'type': 'hyperband',
        's': 2,
        'eta': 3,
        'max_iter': 27,
    },
}

train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])