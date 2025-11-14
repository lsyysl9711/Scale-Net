# envs/hierarchical_cvrp_env.py
# CORRECTED to handle dataclass objects property.

import torch
import dataclasses  # <<< KEY CHANGE 1: Import the dataclasses module
from .CVRPEnv import Step_State

class HierarchicalCVRPEnv:
    """
    A stateless wrapper that presents a view of a subgraph from a base CVRPEnv.
    It translates actions on the subgraph to actions on the base environment
    and filters the base environment's state for the subgraph.
    """
    def __init__(self, base_env, node_indices):
        self.base_env = base_env
        self.device = self.base_env.device
        
        self.node_indices = node_indices.to(self.device)
        self.batch_size, self.num_nodes_sub = self.node_indices.size()
        
        self.base_to_sub_map = torch.full((self.batch_size, self.base_env.problem_size + 1), -1, dtype=torch.long, device=self.device)
        sub_indices_range = torch.arange(self.num_nodes_sub, device=self.device).expand(self.batch_size, -1)
        self.base_to_sub_map.scatter_(1, self.node_indices, sub_indices_range)

    def get_state(self) -> Step_State:
        """
        Returns the current state of the environment, but filtered for the current subgraph.
        """
        base_step_state = self.base_env.step_state
        
        sub_ninf_mask = torch.gather(
            base_step_state.ninf_mask, 2, 
            self.node_indices.unsqueeze(1).expand(-1, self.base_env.pomo_size, -1)
        )

        current_node_base = base_step_state.current_node
        current_node_sub = torch.gather(
            self.base_to_sub_map.unsqueeze(1), 2, 
            current_node_base.unsqueeze(2)
        ).squeeze(2)

        # --- KEY CHANGE 2: Replace the faulty ._replace() method ---
        # Convert the base state dataclass to a dictionary
        state_dict = dataclasses.asdict(base_step_state)
        
        # Update the dictionary with the new, filtered values
        state_dict['ninf_mask'] = sub_ninf_mask
        state_dict['current_node'] = current_node_sub
        
        # Create a new Step_State object by unpacking the updated dictionary
        return Step_State(**state_dict)

    def step(self, sub_action):
        """
        Takes a batched action in the subgraph, maps it to the global action,
        and steps the base environment.
        """
        expanded_indices = self.node_indices.unsqueeze(1).expand(-1, self.base_env.pomo_size, -1)
        base_action = torch.gather(expanded_indices, 2, sub_action.unsqueeze(2)).squeeze(2)
        
        _, reward, done = self.base_env.step(base_action)
        
        return self.get_state(), reward, done