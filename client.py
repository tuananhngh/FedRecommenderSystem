# Standard library imports
from collections import OrderedDict
from logging import DEBUG, INFO
import os

# Third party imports
import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    NDArrays,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
import hydra
from omegaconf import DictConfig
from sympy import per
import torch
from typing import Dict, List, Tuple

# Local application imports
from load_movielens import (
    create_user_dataloader,
    create_user_datasets,
    load_movielens_data,
    split_dataset,
)
from matrix_factorization import MatrixFactorizationModel, build_reconstruction_model


def get_parameters(model)->Tuple[List[NDArrays], List[NDArrays]]:
    global_params = []
    local_params = []
    #log(INFO, "FUNCTION GET_PARAMETERS CALLED")
    for name,val in model.state_dict().items():
        if "global_layers" in name:
            global_params.append(val.cpu().numpy())
        elif "local_layers" in name:
            local_params.append(val.cpu().numpy())
    return global_params, local_params

def set_parameters(model, global_params:List[NDArrays])->None:
    #log(INFO, "FUNCTION SET_PARAMETERS CALLED")
    keys = [k for k in model.state_dict().keys() if 'global_layers' in k]
    global_state_dict = OrderedDict({
            k:torch.tensor(v) for k,v in zip(keys, global_params)
        })
    model.load_state_dict(global_state_dict, strict=False)
    
def set_local_parameters(model, local_params:List[NDArrays])->None:
    keys = [k for k in model.state_dict().keys() if 'local_layers' in k]
    local_state_dict = OrderedDict({
            k:torch.tensor(v) for k,v in zip(keys, local_params)
        })
    model.load_state_dict(local_state_dict, strict=False)
    
class Client_ML(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        #self.client_cid = client_cid
        self.trainloader = trainloader
        self.valloader = valloader
        
    def get_parameters(self) -> List[NDArrays]:
        """
        Get the current model's global parameters and send back to the server
        """
        #log(INFO, "Get Parameters called in as CLIENT METHOD")
        ndarray_global, _ = get_parameters(self.model)
        return ndarray_global

        
    def set_parameters(self, parameters:List[NDArrays])->None:
        """
        Receive global parameters from server

        Args:
            ins (FitsIns): _description_
        """
        #params_server = ins.parameters
        #ndarray_server = parameters_to_ndarrays(params_server)
        set_parameters(self.model, parameters)
        
    def fit(self, parameters, config):
        #log(INFO, f'Client {self.client_cid} Fit, config: {ins.config}')
        recon_epochs = config['recon_epochs']
        pers_epochs = config['pers_epochs']
        recon_lr, pers_lr = config['recon_lr'], config['pers_lr']
        #log(INFO,f"{ins.parameters}")
        #received_params = [x.cpu().numpy() for x in parameters] #Original global parameters, convert to ndarrays
        set_parameters(self.model, parameters) #Set client global layers
        recon_loss = recon_train(self.model, recon_epochs, self.trainloader, recon_lr) #Train local
        pers_loss = pers_train(self.model, pers_epochs, self.trainloader, pers_lr) #Train global
        
        #_ , local_layers = get_parameters(recon_model)
        #global_layers, _ = get_parameters(pers_model)
        
        #set_parameters(self.model, global_layers) #Set global layers
        #set_local_parameters(self.model, local_layers) #Set local layers
        
        #Make difference between trained global and global original
        #log(INFO, f"Client {self.client_cid} Fit, recon_loss: {recon_loss}, pers_loss: {pers_loss}")
        updated_global_params, _  = get_parameters(self.model)
        delta = [x - y for x, y in zip(updated_global_params, parameters)]
        return delta, len(self.trainloader.dataset), {"recon_loss":recon_loss, "pers_loss":pers_loss}

        
    def evaluate(self, parameters:List[NDArrays], config)->EvaluateRes:
        self.set_parameters(parameters)
        loss = test(self.model, self.valloader)
        return loss, len(self.valloader.dataset), {"loss":loss}


def recon_train(model, recon_epochs, support_dataloader, lr):
    criterion = torch.nn.MSELoss()
    optimizer_recon = torch.optim.SGD(model.local_layers.parameters(), lr=lr,
                                      weight_decay=0.)
    user = torch.tensor([0], dtype=torch.long).to(model.device) #for compatible with model
    for epoch in range(recon_epochs):
        running_loss = 0.0
        for inputs, targets in support_dataloader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            item = inputs
            #print("Item : {}".format(item.shape))
            outputs = model(user, item)
            #log(INFO, f"Outputs : {outputs}")
            loss = criterion(outputs, targets)
            #log(INFO, f"Loss : {loss}"
            loss.backward()
            optimizer_recon.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(support_dataloader)
        #log(INFO, f"Epoch {epoch+1} Recon loss: {epoch_loss}")
    return epoch_loss

def pers_train(model, pers_epochs, query_dataloader, lr):
    criterion = torch.nn.MSELoss()
    optimizer_pers = torch.optim.SGD(model.global_layers.parameters(), lr=lr,
                                     weight_decay=0.)
    user_input = torch.tensor([0], dtype=torch.long).to(model.device) #for compatible with model
    for epoch in range(pers_epochs):
        running_loss = 0.
        for inputs, targets in query_dataloader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            item_input = inputs
            outputs = model(user_input, item_input)
            
            loss = criterion(outputs, targets)
            optimizer_pers.zero_grad()
            loss.backward()
            optimizer_pers.step()
            
            running_loss += loss.item()
        epoch_loss = running_loss/len(query_dataloader)
        #log(INFO, f"Epoch {epoch+1} Pers loss: {epoch_loss}")
    return epoch_loss


def test(model, testloader):
    model.eval()
    with torch.no_grad():
        criterion = torch.nn.MSELoss()
        running_loss = 0.
        user = torch.tensor([0], dtype=torch.long).to(model.device)
        for inputs, targets in testloader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            item = inputs
            outputs = model(user, item)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
        #log(INFO, f"Val loss: {running_loss/len(testloader)}")
    return running_loss/len(testloader)

def generate_client(model, trainloader, valloader, device):
    def client_fn(client_id)->Client_ML:
        return Client_ML(model, trainloader[int(client_id)], valloader[int(client_id)])
    return client_fn


# #@hydra.main(config_path="conf", config_name="config_file")
# def get_client(cid:str)->Client_ML:
#     hydra.initialize_config_dir(config_dir=os.path.abspath("conf"))
#     cfg = hydra.compose(config_name="config_file")
#     client_cid = int(cid)
#     ratings_df, movies_df = load_movielens_data(cfg.data.data_dir)
#     num_users, num_items = len(ratings_df.UserID.unique()), len(ratings_df.MovieID.unique())
#     user_datasets = create_user_datasets(ratings_df, min_examples_per_user=50, max_clients=4000)
#     #train_users,val_users, test_users = split_dataset(user_datasets, 0.8, 0.1)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
#     trainloader, valloader, testloader = create_user_dataloader(user_datasets[client_cid], 0.8, 0.1, 5)
#     #log(INFO,len(trainloader))
#     model, global_params, local_params = build_reconstruction_model(num_users=1, num_items=num_items, num_latent_factors=50, personal_model=True, add_biases=False, l2_regularizer=0.0, spreadout_lambda=0.0)
#     #client = Client_ML(model, trainloader, valloader, device, client_cid).to_client()
#     #Callback Client
#     def client_fn(cid:str):
#         return Client_ML(model, trainloader, valloader, device, client_cid)
#     return client_fn
