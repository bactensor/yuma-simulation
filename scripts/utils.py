import os
import torch
import bittensor as bt
import json
import logging
from yuma_simulation._internal.experiment_setup import ExperimentSetup
from yuma_simulation._internal.metagraph_utils import DownloadMetagraph
from yuma_simulation._internal.yumas import YumaParams, YumaSimulationNames, SimulationHyperparameters
from yuma_simulation._internal.cases import MetagraphCase
from argparse import Namespace
from typing import Optional
from common_cli import _create_common_parser

logger = logging.getLogger("main_logger")

def setup_and_download_the_metagraphs(args: Namespace) -> None:
    requested_time_window = args.tempo * args.epochs
    current_block = bt.subtensor().get_current_block()
    start_block = current_block - requested_time_window

    setup = ExperimentSetup(
        netuids=[args.subnet_id],
        start_block=start_block,
        tempo=args.tempo,
        data_points=args.epochs,
        metagraph_storage_path=f"./{args.metagraphs_dir}/subnet_{args.subnet_id}",
        result_path="./results",
        liquid_alpha=False,
    )

    downloader = DownloadMetagraph(setup)
    downloader.run()

def create_output_dir(output_dir: str, subnet_id: int) -> None:
    """
    Creates the output directory if it does not exist.
    """
    if not os.path.exists(f"./{output_dir}/subnet_{subnet_id}"):
        os.makedirs(f"./{output_dir}/subnet_{subnet_id}")
        logger.info(f"Created output directory: {output_dir}")
    else:
        logger.debug(f"Output directory already exists: {output_dir}")


def create_metagraph_case(args: Namespace, metas: list[torch.Tensor], introduce_shift: bool) -> Optional[MetagraphCase]:
    case_type = "shifted" if introduce_shift else "normal"
    try:
        logger.info(f"Creating {case_type} MetagraphCase.")
        case = MetagraphCase(
            shift_validator_id=args.shift_validator_id,
            name="Metagraph Based Dividends",
            metas=metas,
            num_epochs=len(metas),
            introduce_shift=introduce_shift,
            top_validators_ids=args.top_validators
        )
        logger.debug(f"{case_type.capitalize()} MetagraphCase created successfully: {case.name}")
        return case
    except Exception as e:
        logger.error(f"Error while creating {case_type} MetagraphCase.", exc_info=True)
        raise RuntimeError(f"Failed to create {case_type} MetagraphCase") from e

def load_simulation_config():
    """
    Parses command-line arguments and loads the JSON configuration.
    """
    parser = _create_common_parser()
    args = parser.parse_args()

    with open(args.config_file, "r", encoding="utf-8") as f:
        simulation_config = json.load(f)

    global_config = simulation_config.get("global", {})
    for key, value in global_config.items():
        setattr(args, key, value)

    args.yuma_config = {
        "simulation_hyperparameters": simulation_config.get("simulation_hyperparameters", {}),
        "yuma_versions": simulation_config.get("yuma_versions", [])
    }

    return args, simulation_config


def create_simulation_objects(config: dict) -> tuple[SimulationHyperparameters, list[tuple[str, YumaParams]]]:
    simulation_hyperparameters = SimulationHyperparameters(**config.get("simulation_hyperparameters", {}))
    
    yumas = YumaSimulationNames()
    yuma_versions = []
    for version_cfg in config.get("yuma_versions", []):
        version_name = version_cfg["name"]
        params = version_cfg["params"]
        yuma_params = YumaParams(**params)
        version_value = getattr(yumas, version_name)
        yuma_versions.append((version_value, yuma_params))
    
    return simulation_hyperparameters, yuma_versions

