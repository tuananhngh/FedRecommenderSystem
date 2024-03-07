# Standard library imports
import os
import pickle as pkl
from numpy import ndarray

# Third party imports
from omegaconf import DictConfig
import hydra
import flwr as fl
import torch
from flwr.common import (
    ndarrays_to_parameters
)

# Local application imports
from matrix_factorization import MatrixFactorizationModel, build_reconstruction_model
from load_movielens import create_user_datasets, load_movielens_data, get_user_dataloaders
from fedrecon import FedRecon
from server import weighted_average, fit_config_fn, get_on_fit_config, get_parameters, CustomServer
from client import generate_client


@hydra.main(config_path="conf", config_name="config_file",version_base=None)
def main(cfg:DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ratings_df, movies_df = load_movielens_data(cfg.data.data_dir)
    num_users, num_items = len(ratings_df.UserID.unique()), len(ratings_df.MovieID.unique())
    model = MatrixFactorizationModel(num_users, num_items, 50, personal_model=True, add_biases=False, l2_regularizer=0.0, spreadout_lambda=0.0)
    model.to(device)
    global_params, local_params = get_parameters(model)
    initial_params = ndarrays_to_parameters(global_params)
    
    # hist = fl.server.start_server(server_address="[::]:8080",
    #                         config=fl.server.ServerConfig(num_rounds=5), 
    #                         strategy=fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,
    #                                         on_fit_config_fn=get_on_fit_config,
    #                                         initial_parameters=global_params)) 
                            
    user_datasets = create_user_datasets(ratings_df, min_examples_per_user=50, max_clients=4000)
    
    trainloaders, valloaders, testloaders = get_user_dataloaders(user_datasets, 0.8, 0.1, 5)
    client_fn = generate_client(model, trainloaders, valloaders, device)
    nb_clients = len(trainloaders)
    strategy = FedRecon(evaluate_metrics_aggregation_fn=weighted_average,
                        on_fit_config_fn=fit_config_fn,
                        initial_parameters=ndarrays_to_parameters(global_params),
                        fraction_fit=0.1,
                        fraction_evaluate=0.2,
                        )
    
    custom_server = CustomServer(wait_round=50, 
                                 client_manager=fl.server.SimpleClientManager(), 
                                 strategy=strategy)
    
    hist = fl.simulation.start_simulation(config=fl.server.ServerConfig(num_rounds=1000),
                                          client_fn = client_fn,
                                          num_clients=nb_clients,
                                          server = custom_server,
                                          client_resources={"num_cpus": 1, "num_gpus":0.1},
                                          )
    result_path = os.getcwd() + "/result.pkl"
    with open(result_path, 'wb') as f:
        pkl.dump(hist, f)
        
if __name__ == "__main__":  
    main()