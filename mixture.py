import torch
import torch.nn as nn
import os
import logging
import argparse
import utils
import numpy as np
from tqdm import tqdm
from model import mixture_model, cate_mixture_model
from collections import OrderedDict

def get_args():
	""" hyper-parameters. """
	parser = argparse.ArgumentParser('Baseline for modelling object co-occurence')

	# Add data arguments
	parser.add_argument("--seed", type=int, default=42, help="random seed")
	parser.add_argument("--experiment_num", type=int, default=5, help="number of experiments to conduct")

	# Add model arguments
	parser.add_argument('--visible_dimension', default=80, type=int, help='dimension of the visible states')	
	parser.add_argument('--loss_function', default='ZIP', type=str, help='likelihood function to calculate loss')
	parser.add_argument('--mixture_number',default=1, type=int, help='number of mixture components')

	# Add training arguments
	parser.add_argument('--batch_size', default=100, type=int, help='batch size for training')
	parser.add_argument('--max_epoch', default=50, type=int, help='force stop training at specified epoch')
	parser.add_argument('--clip_norm', default=4, type=float, help='clip threshold of gradients')
	parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
	parser.add_argument('--patience', default=10, type=int,
						help='number of epochs without improvement on validation set before early stopping')

	# Add checkpoint arguments
	parser.add_argument('--log_file', default='logging.log', help='path to save logs')

	
	
	args = parser.parse_args()
	return args

def main(args):
	utils.init_logging(args)
	savedir = 'Mixture'+str(args.mixture_number)+str(args.loss_function)
	os.makedirs(savedir, exist_ok=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logging.info('using '+str(device))
	with open('training.csv') as f:
		ncols = len(f.readline().split(','))
	train_data = torch.from_numpy(np.loadtxt('training.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))
	test_data = torch.from_numpy(np.loadtxt('test.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))
	val_data = torch.from_numpy(np.loadtxt('valid.csv',delimiter=',',dtype=int, skiprows=1, usecols=range(2, ncols)))

	utils.setup_seed(args.seed)
	seeds = np.random.choice(range(10000),size=args.experiment_num,replace=False)
	best_model = float('inf')
	performance=[]

	for i in range(args.experiment_num):
		logging.info('This is experiment {:01d}'.format(i+1))
		utils.setup_seed(seeds[i])
		train_loader, val_loader, test_loader, maxcounts = utils.load_and_split(args)

		if args.loss_function == 'C':
			model=cate_mixture_model(args,maxcounts)
		else:
			model = mixture_model(args)

		model.to(device)
		optimizer = torch.optim.Adam(model.parameters(), args.lr)
		bad_epochs = 0
		best_val = float('inf')
		

		for epoch in range(args.max_epoch):
			stats=OrderedDict()
			stats['loss'] = 0
			stats['grad_norm'] = 0
			stats['clip'] = 0
			stats['lr'] = 0
			stats['loss']=0


			if bad_epochs == 5:
				optimizer.param_groups[0]['lr']/=2

			progress_bar = tqdm(train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=False)

			# Iterate over the training set
			for i, sample in enumerate(progress_bar):
				model.train()
				optimizer.zero_grad()
				likelihood = model(sample.to(device))
				loss=-likelihood.mean(dim = 0)
				loss.backward()
				for para in model.parameters():
					if para.grad is not None and torch.isnan(para.grad).any():
						noise = torch.randn_like(para.grad)
						para.grad[torch.isnan(para.grad)] = noise[torch.isnan(para.grad)]
				grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
				optimizer.step()
				stats['loss'] += loss.detach().item() 
				stats['grad_norm'] += grad_norm
				stats['lr'] += optimizer.param_groups[0]['lr']
				stats['clip'] += 1 if grad_norm > args.clip_norm else 0
				progress_bar.set_postfix({key: '{:.4g}'.format(value / (i + 1)) for key, value in stats.items()},
										 refresh=True)

			logging.info('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.4g}'.format(
				value / len(progress_bar)) for key, value in stats.items())))

			model.eval()
			with torch.no_grad():
				val_loss = -model(val_data.to(device)).mean(dim=0)

			logging.info('current_val:{:.4g}, previous_best_val:{:.4g}'.format(val_loss, best_val))

			if val_loss < best_val-0.01:
				best_val = val_loss
				bad_epochs = 0
				torch.save(model.state_dict(),os.path.join(savedir,'best.pt'))
			else:
				bad_epochs += 1

			if bad_epochs >= args.patience:
				logging.info('No validation loss improvements observed for {:d} epochs. Early stop!'.format(args.patience))
				break

		performance.append(best_val)
		if best_val < best_model:
			best_model=best_val
			with torch.no_grad():
				model.load_state_dict(torch.load(os.path.join(savedir,'best.pt')))
				loss_test = -model(test_data.to(device))

	loss_test=loss_test.cpu().detach().numpy()
	np.savetxt(os.path.join(savedir,'result.csv'),loss_test,delimiter=',')
	np.savetxt(os.path.join(savedir,'performance.csv'),performance,delimiter=',')


if __name__ == '__main__':
	args = get_args()
	logging.basicConfig(filename=args.log_file, filemode='a', level=logging.INFO,
						format='%(levelname)s: %(message)s')
	if args.log_file is not None:
		
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		logging.getLogger('').addHandler(console)

	main(args)