import torch
from torch.nn.functional import normalize as norm

class HDFF_VSA():
	def __init__(self):
		pass

	def _dim_check(self, x, y):
		x = x.unsqueeze(0) if len(x.size()) < 2 else x
		y = y.unsqueeze(0) if len(y.size()) < 2 else y
		return x, y

	def bundle(self, x, y) -> torch.FloatTensor:
		x, y = self._dim_check(x, y)
		return x + y
	
	def bulk_bundle(self, x) -> torch.FloatTensor:
		return torch.sum(x, dim=0)

	def bind(self, x, y) -> torch.FloatTensor:
		x, y = self._dim_check(x, y)
		return x * y

	def similarity(self, x, y) -> torch.FloatTensor:
		# both should be (n_samples, hyper_dim)
		x, y = self._dim_check(x, y) 
		x, y = norm(x), norm(y)
		return torch.mm(x, y.T)

