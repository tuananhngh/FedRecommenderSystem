import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import start_server
from typing import List, Tuple, Callable, Dict
import torch
from collections import OrderedDict

import hydra
from omegaconf import DictConfig
from fedrecon import FedRecon
from matrix_factorization import build_reconstruction_model, MatrixFactorizationModel
from flwr.common import (
    Metrics,
    FitIns,
    FitRes,
    Scalar,
    parameter,
    NDArrays,
    GetParametersRes,
)
import os
from flwr.common.logger import log
from logging import DEBUG, INFO
import pickle as pkl
from typing import List, Tuple, Optional
import numpy as np
import timeit
from flwr.server import Server, History

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

def weighted_average(metrics:List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"loss": sum(accuracies) / sum(examples)}

def get_on_fit_config(config: DictConfig)->Callable:
    def fit_config_fn(server_round:int):
        return {'recon_epochs': 3, 'recon_lr':0.01, 'pers_epochs': 3, 'pers_lr':0.01 }
    return fit_config_fn

def fit_config_fn(server_round:int)->FitIns:
    return {'recon_epochs': 3, 'recon_lr':0.01, 'pers_epochs': 3, 'pers_lr':0.01 }


class CustomServer(fl.server.Server):
    def __init__(self,wait_round:int,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wait_round = wait_round
        
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()
        
        # Early Stopping
        min_val_loss = float("inf")
        round_no_improve = 0
        
        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )


            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    log(INFO, "Fit progress: (%s, %s)", current_round, loss_fed)
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
                    # Early Stopping
                    if current_round > 100:
                        if loss_fed < min_val_loss:
                            round_no_improve = 0
                            min_val_loss = loss_fed
                        else:
                            round_no_improve += 1
                            if round_no_improve == self.wait_round:
                                log(INFO, "EARLY STOPPING")
                                break
                    
        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history        




