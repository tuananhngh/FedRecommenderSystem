from functools import reduce
import flwr as fl
import numpy as np
from typing import List, Tuple, Optional, Callable, Union, Dict
from flwr.server.strategy import FedAvg, FedAdam
from flwr.server.client_manager import ClientManager, ClientProxy
from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
)

def aggregate(results : List[Tuple[NDArrays, int]]) -> NDArrays:
    total_examples = sum([num_examples for _, num_examples in results])
    weighted_weights = [
        [layer * num_examples for layer in weights]
        for weights, num_examples in results
    ]
    weights_prime = [
        reduce(np.add, layer_update)/total_examples for layer_update in zip(*weighted_weights)
    ]
    return weights_prime
    

class FedRecon(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-1,
    ) -> None:
        super().__init__(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        accept_failures=accept_failures,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
        self.current_weight = parameters_to_ndarrays(initial_parameters)
        self.eta = eta
    
    
    def __repr__(self) -> str:
        return "FedRecon"
    
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""                
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}
        
        fedavg_weight_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)
        
        # delta_t: NDArrays = [x-y for x,y in zip(fedavg_weight_aggregate, self.current_weight)]
        new_weights = [
         x + self.eta*y for x,y in zip(self.current_weight, fedavg_weight_aggregate)
        ]
        self.current_weight = new_weights
                
        return ndarrays_to_parameters(self.current_weight), metrics_aggregated
