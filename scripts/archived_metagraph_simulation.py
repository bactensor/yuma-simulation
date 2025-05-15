import argparse
import logging
from argparse import Namespace

from yuma_simulation.v1.api import generate_metagraph_based_chart_table
from yuma_simulation._internal.simulation_utils import _generate_relative_dividends_comparisson_table
from yuma_simulation._internal.metagraph_utils import (
    load_metas_from_directory,
)
from yuma_simulation._internal.logger_setup import setup_logger
from utils import (
    setup_and_download_the_metagraphs,
    create_output_dir,
    create_metagraph_case,
    load_simulation_config,
    create_simulation_objects,
)

def run_single_scenario(args: Namespace) -> None:
    """
    Runs a single subnet scenario using the merged configuration.
    """
    create_output_dir(args.output_dir, args.subnet_id)

    if args.download_new_metagraph:
        logger.info("Downloading metagraphs...")
        try:
            setup_and_download_the_metagraphs(args)
        except Exception as e:
            logger.error(f"Download metagraph failed. Aborting further processing. {e}")
            return

    try:
        logger.info("Loading metagraphs...")
        metas = load_metas_from_directory(f"./{args.metagraphs_dir}/subnet_{args.subnet_id}", args.epochs)
    except Exception as e:
        logger.error(f"Error while loading metagraphs: {e}")
        return

    if not metas:
        logger.error("No metagraphs loaded. Nothing to be generated.")
        return
    logger.debug(f"Loaded {len(metas)} metagraphs from {args.metagraphs_dir}.")

    logger.info("Creating Metagraph cases...")
    try:
        normal_case = create_metagraph_case(args, metas, introduce_shift=False)
    except RuntimeError:
        logger.error("Aborting processing due to failure creating the normal MetagraphCase.")
        raise
    logger.debug(f"Created MetagraphCase: {normal_case.name}")

    try:
        shifted_case = create_metagraph_case(args, metas, introduce_shift=True)
    except RuntimeError:
        logger.error("Aborting processing due to failure creating the shifted MetagraphCase.")
        raise
    logger.debug(f"Created MetagraphCase: {shifted_case.name}")

    simulation_hyperparameters, yuma_versions = create_simulation_objects(args.yuma_config)

    if args.generate_chart_table:
        file_name = f"./{args.output_dir}/subnet_{args.subnet_id}/metagraph_simulation_results_gpu_shift_comparison{args.shift_validator_id}.html"
        logger.info("Creating the charts table...")
        try:
            chart_table = generate_metagraph_based_chart_table_shifted_comparisson(
                yuma_versions=yuma_versions,
                normal_case=normal_case,
                shifted_case=shifted_case,
                yuma_hyperparameters=simulation_hyperparameters,
                epochs_padding=args.epochs_padding,
                draggable_table=True,
            )

            with open(file_name, "w", encoding="utf-8") as f:
                f.write(chart_table.data)
                logger.info(f"Simulation results saved to {file_name}")

        except Exception as e:
            logger.error(f"Error generating the chart table: {e}", exc_info=True)
        
    if args.generate_dividends_table:
        file_name_csv = f"./{args.output_dir}/subnet_{args.subnet_id}/metagraph_dividends_framed_table.csv"
        logger.info("Starting generation of total dividends table.")
        try:
            dividends_df = _generate_relative_dividends_comparisson_table(
                case_normal=normal_case,
                case_shifted=shifted_case,
                yuma_versions=yuma_versions,
                simulation_hyperparameters=simulation_hyperparameters,
                epochs_padding=args.epochs_padding,
                epochs_window=args.epochs_window,
            )
            dividends_df = dividends_df.applymap(
                lambda x: f"{x:.3e}" if isinstance(x, (float, int)) and abs(x) < 1e-6 else x
            )
            dividends_df.to_csv(file_name_csv, index=False)
            logger.info(f"CSV file {file_name_csv} has been created successfully.")
        except Exception as e:
            logger.error(f"Error generating the dividends table: {e}")

def main():
    global logger
    logger = setup_logger("main_logger", "application.log", logging.INFO)

    args, simulation_config = load_simulation_config()
    logger.info("runinng the simulation")
    if args.run_multiple_scenarios:
        # MULTI-RUN MODE: use scenario-specific configurations from the unified config.
        scenarios = simulation_config.get("scenarios", [])
        if not scenarios:
            logger.error("No scenarios found in the unified configuration file.")
            return

        for index, scenario_dict in enumerate(scenarios, start=1):
            logger.info(f"\n===== Running scenario {index} =====")
            # Create a new Namespace that merges CLI/global args with the scenario-specific settings.
            scenario_args = argparse.Namespace(**vars(args))
            for key, value in scenario_dict.items():
                setattr(scenario_args, key, value)
            run_single_scenario(scenario_args)
    else:
        # SINGLE-RUN MODE: simply run with args from CLI merged with global config.
        run_single_scenario(args)

if __name__ == "__main__":
    main()
