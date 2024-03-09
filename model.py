import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NuclearNormBall
#from geotorch import low_rank



class MatrixCompletion(nn.Module):
    def __init__(self, num_users, num_items, device):
        super().__init__()
        #self.constraint = constraint
        
        init_random = torch.empty(num_users, num_items)
        nn.init.xavier_normal_(init_random)
        self.model = nn.Parameter(init_random, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_users, num_items), requires_grad=True)
        
        # self.user_embeddings = nn.Embedding(num_users, 100)
        # self.item_embeddings = nn.Embedding(num_items, 100)
        
    def forward(self, x):
        user_idx = x[:,0]
        item_idx = x[:,1]
        return self.model[user_idx, item_idx] + self.bias[user_idx, item_idx]
        # user = x[:,0]
        # item = x[:,1]
        # user_embedding = self.user_embeddings(user)
        # item_embedding = self.item_embeddings(item)
        # return torch.sum(user_embedding * item_embedding, dim=1)

    
def nuclear_norm(matrix):
    norm = torch.norm(matrix, p='nuc')
    return norm
    
    
    

# nb_users = 1000
# nb_items = 1700

# mc2 = MatrixCompletion(nb_users, nb_items,100, device)

# ncnorm = NuclearNormBall(nb_users, nb_items, 100)

# x = torch.tensor([[552,630],[4,5],[6,7]])
# user = x[:,0]
# item = x[:,1]
# y = torch.tensor([4.,5.,6.]).to(device)

# pred = mc2(user, item)
# loss = F.mse_loss(pred, y)

# grad = torch.autograd.grad(loss, mc2.model)
# ok = grad[0]

# v = ncnorm.lmo(ok)
# v_proj = ncnorm.euclidean_project(ok)
# U = mc.user_embeddings.weight
# V = mc.item_embeddings.weight
# matrix = torch.matmul(U, V.T)
# nuclear_norm(matrix)

# def nuclear_proj(matrix, k):
#     U,sigma, V= torch.linalg.svd(matrix, full_matrices=False)
#     print(U.shape, sigma.shape, V.shape)
#     sigma = torch.clamp(sigma-k, min=0)
#     sigma = torch.diag(sigma)
#     projected = torch.matmul(torch.matmul(U,sigma), V.T)
#     norm = torch.linalg.norm(projected, ord='nuc')
#     assert norm <= k, "Projection failed, norm {}".format(norm)
#     print("Norm value : {}".format(norm))
    
# nuclear_proj(matrix, 80)