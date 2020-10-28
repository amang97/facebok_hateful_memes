'''
	* Maan Qraitem 
	* Attention
'''

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class Attention(nn.Module):
	def __init__(self, v_feat_dim, t_feat_dim, num_hidden):
		super(Attention, self).__init__()	
		self.v_feat_dim = v_feat_dim 
		self.t_feat_dim = t_feat_dim 
		self.num_hidden = num_hidden 

		self.linear1 = torch.nn.Parameter(
									data=torch.Tensor(v_feat_dim + t_feat_dim, num_hidden), 
									requires_grad=True)
	

		self.linear2 = torch.nn.Parameter(
									data=torch.Tensor(num_hidden, 1), 
									requires_grad=True)

		self.relu = torch.nn.ReLU() 
		self.softmax = torch.nn.Softmax(dim=-1) 

	def forward(self, v_feat, t_feat): 
		'''		
			v_feat shape = [batch_size, K, v_feat_dim] 
			t_feat shape = [batch_size, t_feat_dim] 
		'''
		K = v_feat.size(1)
		t_feat = t_feat.unsqueeze(1).repeat(1, K, 1) 
		vt_feat = torch.cat((v_feat, t_feat), 2)
		vt_feat = torch.matmul(vt_feat, self.linear1) 
		vt_feat = self.relu(vt_feat) 
		weights = torch.matmul(vt_feat, self.linear2) 
		logits  = self.softmax(weights) 		

		return logits 
		

def main(): 
	attention = Attention(200, 300, 150) 
	v_feat = torch.randn(10, 4, 200) 
	t_feat = torch.randn(10, 300) 
	
	attention(v_feat, t_feat)

if __name__ == "__main__":
	main() 
		
		
