import torch
import torch.nn as nn
from geotorch import low_rank

class MatrixCompletion(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, latent_dim)
        self.item_embeddings = nn.Embedding(num_items, latent_dim)
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_items, 1)
        
    def forward(self, user, item):
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)
        
        rating = (user_emb * item_emb).sum(dim=1) + self.user_biases(user).squeeze() + self.item_biases(item).squeeze()
        reconstruc = torch.matmul(user_emb, item_emb.T)
        return rating
    
def nuclear_norm(matrix):
    norm = torch.norm(matrix, p='nuc')
    return norm
    
mc = MatrixCompletion(num_users= 100, num_items=90, latent_dim=5)
U = mc.user_embeddings.weight
V = mc.item_embeddings.weight
matrix = torch.matmul(U, V.T)
nuclear_norm(matrix)

def nuclear_proj(matrix, k):
    U,sigma, V= torch.linalg.svd(matrix, full_matrices=False)
    print(U.shape, sigma.shape, V.shape)
    sigma = torch.clamp(sigma-k, min=0)
    sigma = torch.diag(sigma)
    projected = torch.matmul(torch.matmul(U,sigma), V.T)
    norm = torch.linalg.norm(projected, ord='nuc')
    assert norm <= k, "Projection failed, norm {}".format(norm)
    print("Norm value : {}".format(norm))
    
nuclear_proj(matrix, 80)