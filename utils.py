from typing import Optional, Tuple, NamedTuple, Mapping
import shutil
import os
import sys
import warnings

import mlflow
import mlflow.pytorch
from tueplots import bundles, figsizes
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from loguru import logger

@rank_zero_only
def rank_zero_print(*args, **kwargs):
    """
    Print a message only on the rank 0 process.
    """
    print(*args, **kwargs)

from datastore.base import BaseDatastore
from datastore.mdp import MDPDatastore

DATASTORE_CLASSES = [
    MDPDatastore
]

DATASTORES = {
    datastore.SHORT_NAME: datastore for datastore in DATASTORE_CLASSES
}

def load_datastore(
    datastore_kind: str,
    config_path: str,
) -> MDPDatastore :
    """
    Load  the datastore 
    Parameters
    ----------
    datastore_kind : str
        Kind of datastore to load. Currently only "mdp" is supported.

    config_path : str
        Path to the mllam-data-prep configuration file.

    Returns
    -------
    MDPDatastore
        The loaded datastore.
    """
    
    DatastoreClass = DATASTORES.get(datastore_kind)

    if DatastoreClass is None:
        raise NotImplementedError(
            f"Datastore kind {datastore_kind} is not implemented"
        )

    datastore = DatastoreClass(config_path=config_path)

    return datastore

class BufferList(nn.Module):
    """
    A list of torch buffer tensors that sit together as a Module with no
    parameters and only buffers.

    This should be replaced by a native torch BufferList once implemented.
    See: https://github.com/pytorch/pytorch/issues/37386
    """

    def __init__(self, buffer_tensors, persistent=True):
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    def __getitem__(self, key):
        return getattr(self, f"b{key}")

    def __len__(self):
        return self.n_buffers

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def load_graph(graph_dir_path, device="cpu"):
    """Load all tensors representing the graph from `graph_dir_path`.

    Needs the following files for all graphs:
    - m2m_edge_index.pt
    - g2m_edge_index.pt
    - m2g_edge_index.pt
    - m2m_features.pt
    - g2m_features.pt
    - m2g_features.pt
    - mesh_features.pt

    And in addition for hierarchical graphs:
    - mesh_up_edge_index.pt
    - mesh_down_edge_index.pt
    - mesh_up_features.pt
    - mesh_down_features.pt

    Parameters
    ----------
    graph_dir_path : str
        Path to directory containing the graph files.
    device : str
        Device to load tensors to.

    Returns
    -------
    hierarchical : bool
        Whether the graph is hierarchical.
    graph : dict
        Dictionary containing the graph tensors, with keys as follows:
        - g2m_edge_index
        - m2g_edge_index
        - m2m_edge_index
        - mesh_up_edge_index
        - mesh_down_edge_index
        - g2m_features
        - m2g_features
        - m2m_features
        - mesh_up_features
        - mesh_down_features
        - mesh_static_features

    """

    def loads_file(fn):
        return torch.load(
            os.path.join(graph_dir_path, fn),
            map_location=device,
            weights_only=True,
        )

    # Load edges (edge_index)
    m2m_edge_index = BufferList(
        loads_file("m2m_edge_index.pt"), persistent=False
    )  # List of (2, M_m2m[l])
    g2m_edge_index = loads_file("g2m_edge_index.pt")  # (2, M_g2m)
    m2g_edge_index = loads_file("m2g_edge_index.pt")  # (2, M_m2g)

    n_levels = len(m2m_edge_index)
    hierarchical = n_levels > 1  # Nor just single level mesh graph

    # Load static edge features
    # List of (M_m2m[l], d_edge_f)
    m2m_features = loads_file("m2m_features.pt")
    g2m_features = loads_file("g2m_features.pt")  # (M_g2m, d_edge_f)
    m2g_features = loads_file("m2g_features.pt")  # (M_m2g, d_edge_f)

    # Normalize by dividing with longest edge (found in m2m)
    longest_edge = max(
        torch.max(level_features[:, 0]) for level_features in m2m_features
    )  # Col. 0 is length
    m2m_features = BufferList(
        [level_features / longest_edge for level_features in m2m_features],
        persistent=False,
    )
    g2m_features = g2m_features / longest_edge
    m2g_features = m2g_features / longest_edge

    # Load static node features
    mesh_static_features = loads_file(
        "mesh_features.pt"
    )  # List of (N_mesh[l], d_mesh_static)

    # Some checks for consistency
    assert (
        len(m2m_features) == n_levels
    ), "Inconsistent number of levels in mesh"
    assert (
        len(mesh_static_features) == n_levels
    ), "Inconsistent number of levels in mesh"

    if hierarchical:
        # Load up and down edges and features
        mesh_up_edge_index = BufferList(
            loads_file("mesh_up_edge_index.pt"), persistent=False
        )  # List of (2, M_up[l])
        mesh_down_edge_index = BufferList(
            loads_file("mesh_down_edge_index.pt"), persistent=False
        )  # List of (2, M_down[l])

        mesh_up_features = loads_file(
            "mesh_up_features.pt"
        )  # List of (M_up[l], d_edge_f)
        mesh_down_features = loads_file(
            "mesh_down_features.pt"
        )  # List of (M_down[l], d_edge_f)

        # Rescale
        mesh_up_features = BufferList(
            [
                edge_features / longest_edge
                for edge_features in mesh_up_features
            ],
            persistent=False,
        )
        mesh_down_features = BufferList(
            [
                edge_features / longest_edge
                for edge_features in mesh_down_features
            ],
            persistent=False,
        )

        mesh_static_features = BufferList(
            mesh_static_features, persistent=False
        )
    else:
        # Extract single mesh level
        m2m_edge_index = m2m_edge_index[0]
        m2m_features = m2m_features[0]
        mesh_static_features = mesh_static_features[0]

        (
            mesh_up_edge_index,
            mesh_down_edge_index,
            mesh_up_features,
            mesh_down_features,
        ) = ([], [], [], [])

    return hierarchical, {
        "g2m_edge_index": g2m_edge_index,
        "m2g_edge_index": m2g_edge_index,
        "m2m_edge_index": m2m_edge_index,
        "mesh_up_edge_index": mesh_up_edge_index,
        "mesh_down_edge_index": mesh_down_edge_index,
        "g2m_features": g2m_features,
        "m2g_features": m2g_features,
        "m2m_features": m2m_features,
        "mesh_up_features": mesh_up_features,
        "mesh_down_features": mesh_down_features,
        "mesh_static_features": mesh_static_features,
    }

RADIUS_EARTH = 6371.0  # km
lon_center = 100.0

def make_mlp(dim_list, layer_norm=True):
    hidden_layers = len(dim_list) - 2
    assert hidden_layers >=0, "Need hidden_layer of MLP"
    layers = []
    for layer_i, (dim1,dim2) in enumerate(zip(dim_list[:-1], dim_list[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.SiLU)

    # like Graphcast,LN on output
    if layer_norm:
        layers.append(nn.LayerNorm(dim_list[-1]))

    return nn.Sequential(*layers)

def lat_lon_deg_to_cartesian(node_lat: np.ndarray,
                             node_lon: np.ndarray,
                            ) -> Tuple[np.ndarray, np.ndarray]:
  lon_rad = np.deg2rad(node_lon)
  lat_rad = np.deg2rad(node_lat)

  x = RADIUS_EARTH * lat_rad
  y = RADIUS_EARTH * (lon_rad - np.deg2rad(lon_center))

  xy = np.stack((x, y), axis=-1)

  return xy

def lat_lon_deg_to_spherical(node_lat: np.ndarray,
                             node_lon: np.ndarray,
                            ) -> Tuple[np.ndarray, np.ndarray]:
  phi = np.deg2rad(node_lon)
  theta = np.deg2rad(90 - node_lat)
  return phi, theta


def spherical_to_lat_lon(phi: np.ndarray,
                         theta: np.ndarray,
                        ) -> Tuple[np.ndarray, np.ndarray]:
  lon = np.mod(np.rad2deg(phi), 360)
  lat = 90 - np.rad2deg(theta)
  return lat, lon

def spherical_to_cartesian(
    phi: np.ndarray, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  # Assuming unit radius.
  return (np.cos(phi)*np.sin(theta),
          np.sin(phi)*np.sin(theta),
          np.cos(theta))


per_variable_weights = {
    "uo1.5413750410079956m": 0.1,
    "uo3.8194949626922607m": 0.1,
    "uo6.440614223480225m" : 0.1,
    "uo9.572997093200684m" : 0.1,
    "uo13.467140197753906m": 0.1,
    "uo18.495559692382812m": 0.1,
    "uo25.211410522460938m": 0.1,
    "uo34.43415069580078m" : 0.1,
    "uo47.37369155883789m" : 0.1,
    "vo1.5413750410079956m": 0.1,
    "vo3.8194949626922607m": 0.1,
    "vo6.440614223480225m" : 0.1,
    "vo9.572997093200684m" : 0.1,
    "vo13.467140197753906m": 0.1,
    "vo18.495559692382812m": 0.1,
    "vo25.211410522460938m": 0.1,
    "vo34.43415069580078m" : 0.1,
    "vo47.37369155883789m" : 0.1,
    "so1.5413750410079956m": 0.1,
    "so3.8194949626922607m": 0.1,
    "so6.440614223480225m" : 0.1,
    "so9.572997093200684m" : 0.1,
    "so13.467140197753906m": 0.1,
    "so18.495559692382812m": 0.1,
    "so25.211410522460938m": 0.1,
    "so34.43415069580078m" : 0.1,
    "so47.37369155883789m" : 0.1,
    "thetao1.5413750410079956m": 0.1,
    "thetao3.8194949626922607m": 0.1,
    "thetao6.440614223480225m" : 0.1,
    "thetao9.572997093200684m" : 0.1,
    "thetao13.467140197753906m": 0.1,
    "thetao18.495559692382812m": 0.1,
    "thetao25.211410522460938m": 0.1,
    "thetao34.43415069580078m" : 0.1,
    "thetao47.37369155883789m" : 0.1,
}

def get_manual_state_feature_weights(
    weights: Mapping[str, float], datastore: BaseDatastore
) -> list[float]:
    """
    Return the state feature weights as a list of floats in the order of the
    state features in the datastore.

    Parameters
    ----------
    weights : Manual State Feature Weighting
        Configuration object containing the manual state feature weights.
    datastore : BaseDatastore
        Datastore object containing the state features.

    Returns
    -------
    list[float]
        List of floats containing the state feature weights.
    """
    state_feature_names = datastore.get_vars_names(category="state")
    feature_weight_names = weights.keys()

    # Check that the state_feature_weights dictionary has a weight for each
    # state feature in the datastore.
    if set(feature_weight_names) != set(state_feature_names):
        additional_features = set(feature_weight_names) - set(
            state_feature_names
        )
        missing_features = set(state_feature_names) - set(feature_weight_names)
        raise ValueError(
            f"State feature weights must be provided for each state feature"
            f"in the datastore ({state_feature_names}). {missing_features}"
            " are missing and weights are defined for the features "
            f"{additional_features} which are not in the datastore."
        )

    state_feature_weights = [
        weights.get(feature, 1) for feature in state_feature_names
    ]
    return state_feature_weights


def get_uniform_state_feature_weights(datastore: BaseDatastore) -> list[float]:
    """
    Return the state feature weights as a list of floats in the order of the
    state features in the datastore.

    The weights are uniform, i.e. 1.0/n_features for each feature.

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore object containing the state features.

    Returns
    -------
    list[float]
        List of floats containing the state feature weights.
    """
    state_feature_names = datastore.get_vars_names(category="state")
    n_features = len(state_feature_names)
    return [1.0 / n_features] * n_features


def get_state_feature_weighting(
    weight_config:str, datastore: BaseDatastore
) -> list[float]:
    """
    Return the state feature weights as a list of floats in the order of the
    state features in the datastore. The weights are determined based on the
    configuration in the NeuralLAMConfig object.

    Parameters
    ----------
    weight_config : loss weights config
    datastore : BaseDatastore
        Datastore object containing the state features.

    Returns
    -------
    list[float]
        List of floats containing the state feature weights.
    """
    if weight_config == "manual":
        weights = get_manual_state_feature_weights(per_variable_weights, datastore)
    elif weight_config == "uniform":
        weights = get_uniform_state_feature_weights(datastore)
    else:
        raise NotImplementedError(
            "Unsupported state feature weighting configuration: "
            f"{weight_config}"
        )

    return weights

def inverse_softplus(x, beta=1, threshold=20):
    """
    Inverse of torch.nn.functional.softplus

    Input is clamped to approximately positive values of x, and the function is
    linear for inputs above x*beta for numerical stability.

    Note that this torch.clamp will make gradients 0, but this is not a
    problem as values of x that are this close to 0 have gradients of 0 anyhow.
    """
    x_clamped = torch.clamp(
        x, min=torch.log(torch.tensor(1e-6 + 1)) / beta, max=threshold / beta
    )

    non_linear_part = torch.log(torch.expm1(x_clamped * beta)) / beta

    below_threshold = x * beta <= threshold

    x = torch.where(condition=below_threshold, input=non_linear_part, other=x)

    return x


def inverse_sigmoid(x):
    """
    Inverse of torch.sigmoid

    Sigmoid output takes values in [0,1], this makes sure input is just within
    this interval.
    Note that this torch.clamp will make gradients 0, but this is not a problem
    as values of x that are this close to 0 or 1 have gradients of 0 anyhow.
    """
    x_clamped = torch.clamp(x, min=1e-6, max=1 - 1e-6)
    return torch.log(x_clamped / (1 - x_clamped))


class CustomMLFlowLogger(pl.loggers.MLFlowLogger):
    """
    Custom MLFlow logger that adds the `log_image()` functionality not
    present in the default implementation from pytorch-lightning as
    of version `2.0.3` at least.
    """

    def __init__(self, experiment_name, tracking_uri, run_name):
        super().__init__(
            experiment_name=experiment_name, tracking_uri=tracking_uri
        )

        mlflow.start_run(run_id=self.run_id, log_system_metrics=True)
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.log_param("run_id", self.run_id)

    @property
    def save_dir(self):
        """
        Returns the directory where the MLFlow artifacts are saved.
        Used to define the path to save output when using the logger.

        Returns
        -------
        str
            Path to the directory where the artifacts are saved.
        """
        return "mlruns"

    def log_image(self, key, images, step=None):
        """
        Log a matplotlib figure as an image to MLFlow

        key: str
            Key to log the image under
        images: list
            List of matplotlib figures to log
        step: Union[int, None]
            Step to log the image under. If None, logs under the key directly
        """
        # Third-party
        from botocore.exceptions import NoCredentialsError
        from PIL import Image

        if step is not None:
            key = f"{key}_{step}"

        # Need to save the image to a temporary file, then log that file
        # mlflow.log_image, should do this automatically, but is buggy
        temporary_image = f"{key}.png"
        images[0].savefig(temporary_image)

        img = Image.open(temporary_image)
        try:
            mlflow.log_image(img, f"{key}.png")
        except NoCredentialsError:
            logger.error("Error logging image\nSet AWS credentials")
            sys.exit(1)

@rank_zero_only
def setup_training_logger(datastore, args, run_name):
    """

    Parameters
    ----------
    datastore : Datastore
        Datastore object.

    args : argparse.Namespace
        Arguments from command line.

    run_name : str
        Name of the run.

    Returns
    -------
    logger : pytorch_lightning.loggers.base
        Logger object.
    """

    if args.logger == "wandb":
        logger = pl.loggers.WandbLogger(
            project=args.logger_project,
            name=run_name,
            config=dict(training=vars(args), datastore=datastore._config),
        )
    elif args.logger == "mlflow":
        url = os.getenv("MLFLOW_TRACKING_URI")
        if url is None:
            raise ValueError(
                "MLFlow logger requires setting MLFLOW_TRACKING_URI in env."
            )
        logger = CustomMLFlowLogger(
            experiment_name=args.logger_project,
            tracking_uri=url,
            run_name=run_name,
        )
        logger.log_hyperparams(
            dict(training=vars(args), datastore=datastore._config)
        )

    return logger

def init_training_logger_metrics(training_logger, val_steps):
    """
    Set up logger metrics to track
    """
    experiment = training_logger.experiment
    if isinstance(training_logger, WandbLogger):
        experiment.define_metric("val_mean_loss", summary="min")
        for step in val_steps:
            experiment.define_metric(f"val_loss_unroll{step}", summary="min")
    elif isinstance(training_logger, MLFlowLogger):
        pass
    else:
        warnings.warn(
            "Only WandbLogger & MLFlowLogger is supported for tracking metrics.\
             Experiment results will only go to stdout."
        )

def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of
    the page width.
    """
    # If latex is not available, some visualizations might not render
    # correctly, but will at least not raise an error. Alternatively, use
    # unicode raised numbers.
    usetex = True if shutil.which("latex") else False
    bundle = bundles.neurips2023(usetex=usetex, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle