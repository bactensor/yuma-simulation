from dataclasses import replace
import logging

from yuma_simulation._internal.cases import get_synthetic_cases
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    YumaParams,
    YumaSimulationNames,
)
from yuma_simulation.v1.api import generate_chart_table
from yuma_simulation._internal.logger_setup import setup_logger


def main():
    global logger
    logger = setup_logger("main_logger", "application.log", logging.INFO)

    # List of bond_penalty values and corresponding file names
    bond_penalty_values = [0, 0.5, 0.99, 1]

    for bond_penalty in bond_penalty_values:
        # Setting global simulation parameters
        simulation_hyperparameters = SimulationHyperparameters(
            bond_penalty=bond_penalty,
        )

        file_name = f"simulation_charts_b{bond_penalty}.html"

        # Setting individual yuma simulations parameters
        base_yuma_params = YumaParams()
        rust_yuma_params = YumaParams(
            bond_moving_avg=0.975,
        )
        liquid_alpha_on_yuma_params = YumaParams(
            liquid_alpha=True,
            alpha_sigmoid_steepness=10.0,
        )
        yuma4_params = YumaParams(
            bond_moving_avg=0.975,
            alpha_low=0.1,
            alpha_high=0.3,
        )
        yuma4_liquid_params = replace(yuma4_params, liquid_alpha=True)

        yumas = YumaSimulationNames()
        yuma_versions = [
            # (yumas.YUMA_RUST, base_yuma_params),
            # (yumas.YUMA, base_yuma_params),
            # (yumas.YUMA_LIQUID, liquid_alpha_on_yuma_params),
            # (yumas.YUMA2, base_yuma_params),
            # (yumas.YUMA3, base_yuma_params),
            # (yumas.YUMA31, base_yuma_params),
            # (yumas.YUMA32, base_yuma_params),
            # (yumas.YUMA4, yuma4_params),
            (yumas.YUMA4_LIQUID, yuma4_liquid_params),
        ]

        cases = get_synthetic_cases(use_full_matrices=True, reset_bonds=True)

        logger.info("Generating chart table...")
        try:
            chart_table = generate_chart_table(
                cases, yuma_versions, simulation_hyperparameters, draggable_table=True
            )
        except Exception as e:
            logger.error(f"Error generating the chart table: {e}", exc_info=True)

        # Save the HTML file
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(chart_table.data)

        logger.info(f"HTML saved to {file_name}")


if __name__ == "__main__":
    main()
