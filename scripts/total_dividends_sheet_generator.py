import logging
from dataclasses import replace

from yuma_simulation._internal.cases import get_synthetic_cases
from yuma_simulation._internal.simulation_utils import _generate_total_dividends_table
from yuma_simulation._internal.logger_setup import setup_logger
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    YumaParams,
    YumaSimulationNames,
)

def main():
    global logger
    logger = setup_logger("main_logger", "application.log", logging.INFO)

    # List of bond_penalty values and corresponding file names
    bond_penalty_values = [0, 0.5, 0.99, 1.0]

    for bond_penalty in bond_penalty_values:
        # Define simulation hyperparameters
        simulation_hyperparameters = SimulationHyperparameters(
            bond_penalty=bond_penalty,
        )
        # Make sure the output file name matches the bond_penalty parameter
        file_name = f"simulation_total_dividends_b{bond_penalty}.csv"

        # Define Yuma parameter variations
        base_yuma_params = YumaParams()
        liquid_alpha_on_yuma_params = YumaParams(  # noqa: F841
            liquid_alpha=True,
        )

        yuma4_params = YumaParams(
            bond_alpha=0.025,
            alpha_high=0.9,
            alpha_low=0.7,
        )
        yuma4_liquid_params = replace(yuma4_params, liquid_alpha=True)  # noqa: F841

        yumas = YumaSimulationNames()
        yuma_versions = [
            (yumas.YUMA_RUST, base_yuma_params),
            # (yumas.YUMA, base_yuma_params),
            # (yumas.YUMA_LIQUID, liquid_alpha_on_yuma_params),
            # (yumas.YUMA2, base_yuma_params),
            # (yumas.YUMA3, base_yuma_params),
            # (yumas.YUMA31, base_yuma_params),
            # (yumas.YUMA32, base_yuma_params),
            # (yumas.YUMA4, base_yuma_params),
            (yumas.YUMA4_LIQUID, yuma4_liquid_params),
        ]

        cases = get_synthetic_cases(use_full_matrices=True)

        logger.info(
            f"Starting generation of total dividends table for bond_penalty={bond_penalty}."
        )
        try:
            dividends_df = _generate_total_dividends_table(
                cases=cases,
                yuma_versions=yuma_versions,
                simulation_hyperparameters=simulation_hyperparameters,
            )
        except Exception as e:
            logger.error(f"Error generating the dividends table: {e}")


        dividends_df.to_csv(file_name, index=False, float_format="%.6f")
        logger.info(f"CSV file {file_name} has been created successfully.")


if __name__ == "__main__":
    main()
