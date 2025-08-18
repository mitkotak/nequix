from pathlib import Path

import equinox as eqx
import jraph
import numpy as np
import urllib.request
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from nequix.data import (
    atomic_numbers_to_indices,
    dict_to_graphstuple,
    preprocess_graph,
)
from nequix.model import load_model


# TODO: a better heuristic for padding will be much faster
# from https://github.com/google-deepmind/jraph/blob/51f5990/jraph/ogb_examples/train.py#L117
# NB: using numpy instead of jax.numpy can be orders of magnitude faster for some reason
def pad_graph_to_nearest_power_of_two(graphs_tuple: jraph.GraphsTuple, _np=np) -> jraph.GraphsTuple:
    """Pads a batched `GraphsTuple` to the nearest power of two.

    For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
    would pad the `GraphsTuple` nodes and edges:
      7 nodes --> 8 nodes (2^3)
      5 edges --> 8 edges (2^3)

    And since padding is accomplished using `jraph.pad_with_graphs`, an extra
    graph and node is added:
      8 nodes --> 9 nodes
      3 graphs --> 4 graphs

    Args:
      graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

    Returns:
      A graphs_tuple batched to the nearest power of two.
    """

    def _nearest_bigger_power_of_two(x: int) -> int:
        y = 2
        while y < x:
            y *= 2
        return y

    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(_np.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(_np.sum(graphs_tuple.n_edge))
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to)


class NequixCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    URLS = {"nequix-mp-1": "https://figshare.com/files/57245573"}

    def __init__(self, model_name: str = "nequix-mp-1", model_path: str = None, **kwargs):
        super().__init__(**kwargs)
        if model_path is None:
            model_path = Path("~/.cache/nequix/models/").expanduser() / f"{model_name}.nqx"
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(self.URLS[model_name], model_path)

        self.model, self.config = load_model(model_path)
        self.atom_indices = atomic_numbers_to_indices(self.config["atomic_numbers"])

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        graph = dict_to_graphstuple(
            preprocess_graph(atoms, self.atom_indices, self.model.cutoff, False)
        )
        padded_graph = pad_graph_to_nearest_power_of_two(graph)
        energy, forces, stress = eqx.filter_jit(self.model)(padded_graph)
        # take energy and forces without padding
        energy = np.array(energy[0])
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = np.array(forces[: len(atoms)])
        self.results["stress"] = full_3x3_to_voigt_6_stress(np.array(stress[0]))
