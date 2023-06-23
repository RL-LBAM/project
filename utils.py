import torch
import torch.nn as nn
import numpy as np
import random
import sys
import logging
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.serialization import default_restore_location
from collections import Counter

def setup_seed(seed):
	# Set random seeds for different packages to ensure reproducibility
	 torch.manual_seed(seed)
	 random.seed(seed)
	 np.random.seed(seed)
	 torch.cuda.manual_seed(seed)
	 torch.cuda.manual_seed_all(seed)


def load_and_split(args):
	with open('training.csv') as f:
			ncols = len(f.readline().split(','))

	train = torch.from_numpy(np.loadtxt('training.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))
	valid = torch.from_numpy(np.loadtxt('valid.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))
	test = torch.from_numpy(np.loadtxt('test.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))

	maxcounts = torch.cat((train.max(0).values.unsqueeze(0), valid.max(0).values.unsqueeze(0), 
		test.max(0).values.unsqueeze(0)),0).max(0).values
	
	train_loader = DataLoader(train, batch_size = args.batch_size, shuffle = True)
	val_loader = DataLoader(valid, batch_size = args.batch_size, shuffle = True)
	test_loader = DataLoader(test, batch_size = args.batch_size, shuffle = True)

	return train_loader, val_loader, test_loader, maxcounts

def save_checkpoint(save_dir, save_name, state_dict):
	os.makedirs(save_dir, exist_ok=True)
	last_val = state_dict['val_hist'][-1]
	torch.save(state_dict, os.path.join(save_dir, save_name))

	if state_dict['best_val'] == last_val:
		torch.save(state_dict, os.path.join(save_dir, save_name.replace('last','best')))

def load_checkpoint(checkpoint_path, model, optimizer):
	state_dict = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
	model.load_state_dict(state_dict['model'])
	optimizer.load_state_dict(state_dict['optimizer'])
	logging.info('Loaded checkpoint {}'.format(checkpoint_path))
	return state_dict

def init_logging(args):
	handlers = [logging.StreamHandler()]

	if hasattr(args, 'log_file') and args.log_file is not None:
		handlers.append(logging.FileHandler(args.log_file, mode='w'))

	logging.basicConfig(handlers=handlers, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
						level=logging.INFO)
	logging.info('COMMAND: %s' % ' '.join(sys.argv))
	logging.info('Arguments: {}'.format(vars(args)))


def distribution_extractor(data, maxcount, column = True):
    
    normalize = data.shape[1]
    if column == True:
        data = torch.t(data)
        normalize = data.shape[0]
        
    possible_values = range(maxcount+1)
    frequency = []

    
    for row in data:
        row_frequency = Counter(row.tolist())
        row_tensor = [row_frequency.get(value, 0) for value in possible_values]
        row_tensor[-1]+=sum(torch.tensor(list(row_frequency.keys()),dtype = int) > maxcount)
        frequency.append(row_tensor)
    
    return torch.tensor(frequency)/normalize

def inti_para(m):
 	for layer in m.modules():
 		if isinstance(layer, nn.Linear):
 			nn.init.xavier_uniform_(layer.weight)
 			layer.bias.data.fill_(0)




