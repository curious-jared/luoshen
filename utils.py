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
    "zos": 1.0,

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