import torch
import torch.nn as nn
import pyro.distributions as dis

class vae(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.encoder=encoder(args)
		self.decoder=decoder(args)
		self.args = args
		if args.prior == 'vamp':
			self.mixture_components = nn.Parameter(torch.randn(args.prior_hyper, args.visible_dimension))

	def forward(self, x):
		y=x.to(torch.float)
		mean, logvar = self.encoder(y)
		samples = reparameterization(mean, logvar, self.args.sample_size)
		pzero, strength, ratio = self.decoder(samples)
		likelihood = evaluate(x, pzero, strength, ratio, self.args.loss_function).mean(dim = 0)
		if self.args.prior == 'vamp':
			prior_means, prior_logvars = self.encoder(self.mixture_components)
			klloss = kld(mean, logvar, self.args, samples, prior_means, prior_logvars)
		else:
			klloss = kld(mean, logvar, self.args)

		return likelihood, klloss


class encoder(nn.Module):
	def __init__(self, args):
		super().__init__()

		visible = args.visible_dimension
		hidden = args.hidden_dimension
		inter = args.inter_layer
		final = [visible] + inter + [hidden]

		layers = []
		for i in range(len(final) - 2):
			layers.append(nn.Linear(final[i], final[i + 1]))
			layers.append(nn.ReLU())
		self.model = nn.Sequential(*layers)

		self.mean = nn.Linear(final[-2], final[-1])
		self.logvar = nn.Linear(final[-2], final[-1])

	def forward(self, x):
		features = self.model(x)
		mean = self.mean(features)
		logvar = self.logvar(features)

		return mean, logvar

class decoder(nn.Module):
	def __init__(self, args):
		super().__init__()
		
		visible = args.visible_dimension
		hidden = args.hidden_dimension
		inter = args.inter_layer

		final = ([visible] + inter + [hidden])
		final.reverse()

		layers = []
		for i in range(len(final) - 2):
			layers.append(nn.Linear(final[i], final[i + 1]))
			layers.append(nn.ReLU())
		self.model = nn.Sequential(*layers)
		self.pzero = nn.Linear(final[-2], final[-1])
		self.strength = nn.Linear(final[-2], final[-1])   
		self.ratio = nn.Linear(final[-2], final[-1])
	def forward(self,x):
		features = self.model(x)
		pzero = torch.sigmoid(self.pzero(features))
		strength = torch.exp(self.strength(features))+1e-10
		ratio = torch.sigmoid(self.ratio(features))

		strength = torch.where(strength > 50, 50, strength)
		pzero = torch.where(pzero > 0.99, 0.99, pzero)
		ratio = torch.where(ratio > 0.99, 0.99, ratio)
		pzero = torch.where(pzero < 0.01, 0.01, pzero)
		ratio = torch.where(ratio < 0.01, 0.01, ratio)
		
		return pzero, strength, ratio


def reparameterization(mean, logvar, sample_size = 1):
	std = torch.exp(0.5*logvar)
	eps=torch.randn(torch.Size([sample_size]) + std.shape, dtype=std.dtype, device=std.device)
	z = mean + eps*std
	return z.to(device = std.device)

def kld(mean, logvar, args, *others):
	if args.prior == 'gaussian':
		var = torch.exp(logvar)
		kld=-0.5*(1 + logvar - mean**2 - var).sum(dim = -1)
	if args.prior == 'vamp':
		samples = others[0].unsqueeze(2)
		prior_means = others[1]
		prior_logvars = others[2]
		selfentropy = 0.5*(logvar+1).sum(dim=-1)
		single= 0.5*(+prior_logvars+(samples-prior_means)**2/torch.exp(prior_logvars)
			).sum(dim=-1)-torch.log(torch.tensor(args.prior_hyper))
		crossentropy = torch.logsumexp(single, dim=-1).mean(dim=0)
		kld = crossentropy - selfentropy


	return kld

def evaluate(x, pzero, strength, ratio, loss):

	if loss == 'ZIP':
		prob = dis.ZeroInflatedPoisson(strength, gate = pzero)
		
	elif loss == 'ZINB':
		prob = dis.ZeroInflatedNegativeBinomial(strength, probs=ratio, gate = pzero)

	elif loss == 'P':
		prob = torch.distributions.Poisson(strength)

	elif loss == 'NB':
		prob = torch.distributions.NegativeBinomial(strength, probs=ratio)

	return prob.log_prob(x).sum(dim=-1)


		

class ae(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.encoder=encoder(args)
		self.decoder=decoder(args)
		self.args = args

	def forward(self, x):
		y=x.to(torch.float)
		mean, logvar = self.encoder(y)
		
		pzero, strength, ratio = self.decoder(mean)
		likelihood = evaluate(x, pzero, strength, ratio, self.args.loss_function)

		return likelihood

class cate_vae(nn.Module):
	def __init__(self, args, maxcounts):
		super().__init__()
		self.encoder=encoder(args)
		self.decoder=cate_decoder(args, maxcounts)
		self.args = args
		self.loss = nn.CrossEntropyLoss(reduction='none')

	def forward(self,x):
		y=x.to(torch.float)
		mean, logvar = self.encoder(y)
		samples = reparameterization(mean, logvar, self.args.sample_size)
		cates = self.decoder(samples)
		
		loss_all = []
		for j in range(self.args.sample_size):
			self.loss.label_smoothing = 1e-5 if self.training else 0
			losses = [self.loss(cates[i][j], x.long()[:,i]).unsqueeze(1) for i in range(self.args.visible_dimension)]
			loss_all.append(torch.cat(losses, dim=1).sum(dim=-1).unsqueeze(0))

		likelihood = -torch.cat(loss_all,dim=0).mean(dim=0)
		klloss = kld(mean, logvar)

		return likelihood, klloss


class cate_decoder(nn.Module):
	def __init__(self, args, maxcounts):
		super().__init__()
		
		hidden = args.hidden_dimension
		inter = args.inter_layer

		final = (inter + [hidden])
		final.reverse()

		layers = []
		for i in range(len(final) - 1):
			layers.append(nn.Linear(final[i], final[i + 1]))
			layers.append(nn.ReLU())
		self.model = nn.Sequential(*layers)
		self.cate = nn.ModuleList([nn.Linear(final[-1], num_cate + 1) for num_cate in maxcounts])

		
	def forward(self,x):
		features = self.model(x)
		cates = [cate_layer(features) for cate_layer in self.cate]

		return cates

class base_NB(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.pzero = nn.Parameter(torch.randn(args.visible_dimension))
		self.strength = nn.Parameter(torch.randn(args.visible_dimension))
		self.ratio = nn.Parameter(torch.randn(args.visible_dimension))
		self.args = args

	def forward(self, x):
		pzero = torch.sigmoid(self.pzero)
		strength = torch.exp(self.strength)+1e-10
		ratio = torch.sigmoid(self.ratio)
		likelihood = evaluate(x, pzero, strength,ratio,self.args.loss_function)

		return likelihood

class mixture_model(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.mixture_weight = nn.Parameter(torch.randn(args.mixture_number))
		self.pzero = nn.ParameterList([nn.Parameter(torch.randn(args.visible_dimension)) for i in range(args.mixture_number)])
		self.strength = nn.ParameterList([nn.Parameter(torch.randn(args.visible_dimension)) for i in range(args.mixture_number)])
		self.ratio = nn.ParameterList([nn.Parameter(torch.randn(args.visible_dimension)) for i in range(args.mixture_number)])
		self.args = args

	def forward(self, x):
		mixture_weight = torch.softmax(self.mixture_weight,dim=0)
		likelihood = torch.randn(len(mixture_weight),x.shape[0])

		for i in range(len(mixture_weight)):
			pzero = torch.sigmoid(self.pzero[i])
			strength = torch.exp(self.pzero[i])+1e-10
			ratio = torch.sigmoid(self.ratio[i])
			likelihood[i,:] = evaluate(x, pzero, strength,ratio,self.args.loss_function) + torch.log(mixture_weight[i])

		likelihood = torch.logsumexp(likelihood,dim=0)

		return likelihood

class cate_mixture_model(nn.Module):
	def __init__(self, args,maxcounts):
		super().__init__()
		self.mixture_weight=nn.Parameter(torch.randn(args.mixture_number))
		self.loss = nn.CrossEntropyLoss(reduction='none')
		self.cate = nn.ParameterList()
		for i in range(args.mixture_number):
			self.cate.append(nn.ParameterList([nn.Parameter(torch.randn(num+1)) for num in maxcounts]))
		self.args = args

	def forward(self, x):
		mixture_weight = torch.softmax(self.mixture_weight,dim=0)
		likelihood = torch.randn(len(mixture_weight),x.shape[0])
		self.loss.label_smoothing = 1e-5 if self.training else 0

		for i in range(len(mixture_weight)):
			single = self.cate[i]
			losses=[self.loss(single[j].repeat(x.shape[0],1), x[:,j].long()).unsqueeze(1) for j in range(self.args.visible_dimension)]
			loss_all = -torch.cat(losses,dim=1).sum(-1)
			likelihood[i:] = loss_all+torch.log(mixture_weight[i])

		likelihood=torch.logsumexp(likelihood,dim=0)

		return likelihood








