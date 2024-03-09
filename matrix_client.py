from model import MatrixCompletion
from utils import NuclearNormBall, VanillaFW, nuclear_norm, make_feasible, SFW, VanillaFW
from geotorch import low_rank


data_dir = "/Volumes/T7 Touch/TheseProject/FLDATA/MovieLens"
#data_dir = "/srv/storage/energyfl@storage1.toulouse.grid5000.fr/FLDATA/movielens"
ratings_df, movies_df = load_movielens_100k_data(data_dir)
nb_users, nb_movies = len(ratings_df.UserID.unique()), len(movies_df.MovieID.unique())
trainloaders,valloaders, testloader = create_host_dataloaders(ratings_df, 
                                                              movies_df, 
                                                              nb_hosts=10,
                                                              batch_size=64,
                                                              distributed=False)

#x, y = next(iter(trainloaders))
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
matrix_model = MatrixCompletion(nb_users, nb_movies, device)

constraints = []
for name, param in matrix_model.named_parameters():
    constraint = NuclearNormBall(nb_users, nb_movies, 100)
    constraints.append(constraint)

lss = []
#criterion = nn.MSELoss()
optimizer = VanillaFW(matrix_model.parameters(), learning_rate=0.01)
print(optimizer.param_groups)
ell = []
for epoch in range(10):
    epoch_loss = 0.0
    for x,y in trainloaders:
        #print(len(x))
        #x, y = x.to(device), y.to(device)
        pred = matrix_model(x)
        #print(pred, y)
        loss = torch.nn.functional.mse_loss(pred, y) 
        #print("Loss value: {}".format(loss.item()))
        lss.append(loss.item())
        #grads = torch.autograd.grad(loss, matrix_model.parameters())
        #print(grads)
        epoch_loss += loss.item()
        matrix_model.zero_grad()
        loss.backward()
        optimizer.step(constraints=constraints)
        #optimizer.step()
        # with torch.no_grad():
        #     for idx, (name, param) in enumerate(matrix_model.named_parameters()):
        #         #print(param[0], param.grad.nonzero())
        #         #print(name, param.grad.nonzero().shape)
        #         if param.requires_grad:
        #             #print("Grad Available")
        #             v_ = constraints[idx].lmo(param.grad)
        #             d_t = v_ - param
        #             param.mul(1-0.01)
        #             param.add_(d_t, alpha=0.01)
        #             #update_step = param - 0.01*param.grad 
        #             #hehe = constraint.euclidean_project(update_step)
        #             #hehe = param + 0.01*(v_ - param)
        #             #param.copy_(update_step)
        #             #low_rank(param, 'model', 100)
        #             #print(nuclear_norm(param))
        #         #param = param + 0.01*(v_ - param)
        #         #param.data.add_(-0.1*param.grad)
    epoch_loss = epoch_loss/len(trainloaders)
    print(epoch_loss)
    ell.append(epoch_loss)
plt.figure(figsize=(7, 3))
plt.plot(ell)
plt.show()
