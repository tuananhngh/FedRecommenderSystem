from http import client
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import start_server
from typing import List, Tuple, Callable, Dict
from fedrecon import FedRecon
from matrix_factorization import build_reconstruction_model, MatrixFactorizationModel
from load_movielens import load_movielens_data,path_to_1m
from client_ml import main as client_fn
from client_ml import get_parameters, set_parameters
from flwr.common import (
    Metrics,
    FitIns,
    FitRes,
    Scalar,
    parameter
)
import os
import pickle as pkl

def weighted_average(metrics:List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"loss": sum(accuracies) / sum(examples)}

def get_on_fit_config(config: Dict[str, Scalar])->Callable:
    def fit_config_fn(server_round:int)->FitIns:
        return {'recon_epochs': 3, 'recon_lr':0.01, 'pers_epochs': 3, 'pers_lr':0.01 }
    return fit_config_fn

def fit_config_fn(server_round:int)->FitIns:
    return {'recon_epochs': 3, 'recon_lr':0.01, 'pers_epochs': 3, 'pers_lr':0.01 }

def main():
    ratings_df, movies_df = load_movielens_data(path_to_1m)
    num_users, num_items = len(ratings_df.UserID.unique()), len(ratings_df.MovieID.unique())
    model = MatrixFactorizationModel(num_users, num_items, 50, personal_model=True, add_biases=False, l2_regularizer=0.0, spreadout_lambda=0.0)
    global_params, local_params = get_parameters(model)
    # hist = fl.server.start_server(server_address="[::]:8080",
    #                         config=fl.server.ServerConfig(num_rounds=5), 
    #                         strategy=FedAvg(evaluate_metrics_aggregation_fn=weighted_average,
    #                                         on_fit_config_fn=get_on_fit_config,
    #                                         initial_parameters=global_params) 
    
    strategy = FedRecon(evaluate_metrics_aggregation_fn=weighted_average,
                        on_fit_config_fn=fit_config_fn,
                        initial_parameters=global_params,
                        fraction_fit=0.3,
                        fraction_evaluate=0.5)
    
    hist = fl.simulation.start_simulation(config=fl.server.ServerConfig(num_rounds=20),
                                          strategy = strategy,
                                          client_fn = client_fn,
                                          num_clients=300,
                                          )
    result_path = os.getcwd() + "/result.pkl"
    with open(result_path, 'wb') as f:
        pkl.dump(hist, f)
        
if __name__ == "__main__":  
    main()

