import os
import bittensor as bt
import json
import argparse

from dataclasses import replace
from yuma_simulation._internal.logger_setup import main_logger as logger
from yuma_simulation._internal.simulation_utils import generate_metagraph_case_shifted_validator_comparison_table
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    YumaParams,
    YumaSimulationNames,
)
from yuma_simulation._internal.metagraph_utils import load_metas_from_directory
from yuma_simulation._internal.cases import MetagraphCase

from common_cli import _create_common_parser
from yuma_simulation._internal.experiment_setup import ExperimentSetup
from yuma_simulation._internal.metagraph_utils import (
    DownloadMetagraph,
)

def run_single_scenario(args):
    create_output_dir(args.output_dir, args.subnet_id)

    two_days_blocks = 14400
    current_block = bt.subtensor().get_current_block()
    start_block = current_block - two_days_blocks

    if args.download_new_metagraph:
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

    try:
        logger.info("Loading metagraphs.")
        metas = load_metas_from_directory(
            f"./{args.metagraphs_dir}/subnet_{args.subnet_id}"
        )
    except Exception:
        logger.error("Error while loading metagraphs", exc_info=True)
        return

    if not metas:
        logger.error("No metagraphs loaded. Nothing to be generated.")
        return
    logger.debug(f"Loaded {len(metas)} metagraphs from {args.metagraphs_dir}.")

    logger.info(f"Calculating total dividends")

    simulation_hyperparameters = SimulationHyperparameters()

    file_name = f"./{args.output_dir}/subnet_{args.subnet_id}/metagraph_shifted_validator_comparison_relative_results.csv"
    logger.debug(f"Output file: {file_name}")

    base_yuma_params = YumaParams()
    liquid_alpha_on_yuma_params = replace(base_yuma_params, liquid_alpha=True)  # noqa: F841

    yuma_params = YumaParams(
        bond_alpha=0.025,
        alpha_high=0.9,
        alpha_low=0.7,
    )
    yuma4_liquid_params = replace(yuma_params, liquid_alpha=True)

    yumas = YumaSimulationNames()
    yuma_versions = [
        (yumas.YUMA_RUST, yuma_params),
        (yumas.YUMA4_LIQUID, yuma4_liquid_params)
    ]

    try:
        logger.info("Creating MetagraphCase.")
        case_normal = MetagraphCase(
            shift_validator_id=args.shift_validator_id,
            name="Metagraph simulation",
            metas=metas,
            num_epochs=len(metas),
            introduce_shift=False,
        )
        logger.debug(f"MetagraphCase created successfully: {case_normal.name}")
    except Exception:
        logger.error("Error while creating MetagraphCase.", exc_info=True)
        return
    
    try:
        logger.info("Creating MetagraphCase.")
        case_shifted = MetagraphCase(
            shift_validator_id=args.shift_validator_id,
            name="Metagraph simulation shifted",
            metas=metas,
            num_epochs=len(metas),
            introduce_shift=True,
        )
        logger.debug(f"MetagraphCase created successfully: {case_shifted.name}")
    except Exception:
        logger.error("Error while creating MetagraphCase.", exc_info=True)
        return

    logger.info(
        f"Starting generation of total dividends table."
    )
    dividends_df = generate_metagraph_case_shifted_validator_comparison_table(
        case_normal=case_normal,
        case_shifted=case_shifted,
        yuma_versions=yuma_versions,
        simulation_hyperparameters=simulation_hyperparameters,
        epochs_window=20
    )

    dividends_df = dividends_df.applymap(
        lambda x: f"{x:.3e}" if isinstance(x, (float, int)) and abs(x) < 1e-6 else x
    )

    # Save the dataframe to CSV
    dividends_df.to_csv(file_name, index=False)
    logger.info(f"CSV file {file_name} has been created successfully.")



def create_output_dir(output_dir, subnet_id):
    """
    Creates the output directory if it does not exist.
    """
    if not os.path.exists(f"./{output_dir}/subnet_{subnet_id}"):
        os.makedirs(f"./{output_dir}/subnet_{subnet_id}")
        logger.info(f"Created output directory: {output_dir}")
    else:
        logger.debug(f"Output directory already exists: {output_dir}")


def main():
    parser = _create_common_parser()
    args = parser.parse_args()

    if args.use_json_config:
        # MULTI-RUN MODE:
        with open(args.config_file, "r") as f:
            config_data = json.load(f)

        scenarios = config_data.get("scenarios", [])
        if not scenarios:
            logger.error("No scenarios found in the JSON file.")
            return

        for index, scenario_dict in enumerate(scenarios, start=1):
            logger.info(f"\n===== Running scenario {index} =====")

            scenario_args = argparse.Namespace(**vars(args))
            for key, value in scenario_dict.items():
                setattr(scenario_args, key, value)

            run_single_scenario(scenario_args)

    else:
        # SINGLE-RUN MODE (the old way)
        run_single_scenario(args)

        

if __name__ == "__main__":
    main()
