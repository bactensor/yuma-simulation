import pandas as pd
import torch
from IPython.display import HTML
from yuma_simulation._internal.logger_setup import main_logger as logger

from yuma_simulation._internal.cases import BaseCase
from yuma_simulation._internal.charts_utils import (
    _plot_bonds,
    _plot_dividends,
    _plot_relative_dividends,
    _plot_incentives,
    _plot_validator_server_weights,
    _plot_relative_dividends_comparisson,
)
from yuma_simulation._internal.simulation_utils import (
    _generate_draggable_html_table,
    _generate_ipynb_table,
    run_simulation,
    run_dynamic_simulation,
)
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    YumaConfig,
    YumaParams,
    YumaSimulationNames,
)


def generate_chart_table(
    cases: list[BaseCase],
    yuma_versions: list[tuple[str, YumaParams]],
    yuma_hyperparameters: SimulationHyperparameters,
    draggable_table: bool = False,
    chart_types: list[str] = None,
) -> HTML:
    table_data: dict[str, list[str]] = {
        yuma_version: [] for yuma_version, _ in yuma_versions
    }

    def process_chart(
        table_data: dict[str, list[str]], chart_base64_dict: dict[str, str]
    ) -> None:
        for yuma_version, chart_base64 in chart_base64_dict.items():
            table_data[yuma_version].append(chart_base64)

    case_row_ranges = []
    current_row_count = 0

    simulation_names = YumaSimulationNames()
    for idx, case in enumerate(cases):
        current_chart_types = chart_types or (
            ["weights", "dividends", "bonds", "normalized_bonds", "incentives"]
            if idx in [9, 10]
            else ["weights", "dividends", "bonds", "normalized_bonds"]
        )

        case_start = current_row_count
        for chart_type in current_chart_types:
            chart_base64_dict: dict[str, str] = {}
            #TODO(konrad) check if optimization with pushing the yuma versions to outer context is possible
            for yuma_version, yuma_params in yuma_versions:
                if yuma_version == simulation_names.YUMA4_LIQUID_FIXED:
                    case.disable_matrix_fix = False
                else:
                    case.disable_matrix_fix = True
                yuma_config = YumaConfig(
                    simulation=yuma_hyperparameters, yuma_params=yuma_params
                )
                yuma_names = YumaSimulationNames()
                final_case_name = f"{case.name} - {yuma_version}"
                if yuma_version in [
                    yuma_names.YUMA,
                    yuma_names.YUMA_LIQUID,
                    yuma_names.YUMA2,
                ]:
                    final_case_name = (
                        f"{case.name} - beta={yuma_config.bond_penalty}"
                    )
                elif yuma_version == yuma_names.YUMA4_LIQUID:
                    final_case_name = f"{case.name} - {yuma_version} - [{yuma_config.alpha_low}, {yuma_config.alpha_high}]"

                (
                    dividends_per_validator,
                    validators_relative_dividends,
                    bonds_per_epoch,
                    server_incentives_per_epoch,
                ) = run_simulation(
                    case=case,
                    yuma_version=yuma_version,
                    yuma_config=yuma_config,
                )

                if chart_type == "weights":
                    chart_base64 = _plot_validator_server_weights(
                        validators=case.validators,
                        weights_epochs=case.weights_epochs_guard,
                        servers=case.servers,
                        num_epochs=case.num_epochs,
                        case_name=final_case_name,
                        to_base64=True,
                    )
                elif chart_type == "dividends":
                    chart_base64 = _plot_dividends(
                        num_epochs=case.num_epochs,
                        validators=case.validators,
                        dividends_per_validator=dividends_per_validator,
                        case_name=final_case_name,
                        case=case,
                        to_base64=True,
                    )
                elif chart_type == "relative_dividends":
                    chart_base64 = _plot_relative_dividends(
                        validators_relative_dividends=validators_relative_dividends,
                        case_name=final_case_name,
                        case=case,
                        num_epochs=case.num_epochs,
                        to_base64=True,
                    )
                elif chart_type == "bonds":
                    chart_base64 = _plot_bonds(
                        num_epochs=case.num_epochs,
                        validators=case.validators,
                        servers=case.servers,
                        bonds_per_epoch=bonds_per_epoch,
                        case_name=final_case_name,
                        to_base64=True,
                    )
                elif chart_type == "normalized_bonds":
                    chart_base64 = _plot_bonds(
                        num_epochs=case.num_epochs,
                        validators=case.validators,
                        servers=case.servers,
                        bonds_per_epoch=bonds_per_epoch,
                        case_name=final_case_name,
                        to_base64=True,
                        normalize=True,
                    )
                elif chart_type == "incentives":
                    chart_base64 = _plot_incentives(
                        servers=case.servers,
                        server_incentives_per_epoch=server_incentives_per_epoch,
                        num_epochs=case.num_epochs,
                        case_name=final_case_name,
                        to_base64=True,
                    )
                else:
                    raise ValueError("Invalid chart type.")

                chart_base64_dict[yuma_version] = chart_base64

            process_chart(table_data, chart_base64_dict)
            current_row_count += 1

        case_end = current_row_count - 1
        case_row_ranges.append((case_start, case_end, idx))

    summary_table = pd.DataFrame(table_data)

    if draggable_table:
        full_html = _generate_draggable_html_table(
            table_data, summary_table, case_row_ranges
        )
    else:
        full_html = _generate_ipynb_table(table_data, summary_table, case_row_ranges)

    return HTML(full_html)


def generate_metagraph_based_relative_dividends_comparisson_table(
    yuma_versions: list[tuple[str, YumaParams]],
    normal_case: BaseCase,
    shifted_case: BaseCase,
    yuma_hyperparameters: SimulationHyperparameters,
    epochs_padding: int,
    draggable_table: bool = False,
) -> HTML:
    """
    Generate an HTML table with one column per yuma_version.
    
    Each column will have three rows:
      - Row 0: The relative dividends chart for the normal case.
      - Row 1: The relative dividends chart for the shifted case.
      - Row 2: A comparisson chart that uses the validators_relative_dividends data
               from both cases.
    
    It is assumed that a plotting function _plot_relative_dividends_comparisson exists.
    """
    # Create a dictionary with one key per yuma version.
    table_data: dict[str, list[str]] = {yuma_version: [] for yuma_version, _ in yuma_versions}
    
    # For generating proper case names, we mimic generate_chart_table logic.
    yuma_names = YumaSimulationNames()
    
    # Iterate over each yuma_version.
    for yuma_version, yuma_params in yuma_versions:
        # Create the configuration for this simulation.
        yuma_config = YumaConfig(simulation=yuma_hyperparameters, yuma_params=yuma_params)
        
        # Prepare a descriptive case name similar to generate_chart_table.
        if yuma_version in [yuma_names.YUMA, yuma_names.YUMA_LIQUID, yuma_names.YUMA2]:
            final_case_name_normal = f"{normal_case.name} - beta={yuma_config.bond_penalty}"
            final_case_name_shifted = f"{shifted_case.name} - beta={yuma_config.bond_penalty}"
        elif yuma_version == yuma_names.YUMA4_LIQUID:
            final_case_name_normal = f"{normal_case.name} - {yuma_version} - [{yuma_config.alpha_low}, {yuma_config.alpha_high}]"
            final_case_name_shifted = f"{shifted_case.name} - {yuma_version} - [{yuma_config.alpha_low}, {yuma_config.alpha_high}]"
        else:
            final_case_name_normal = f"{normal_case.name} - {yuma_version}"
            final_case_name_shifted = f"{shifted_case.name} - {yuma_version}"
        
        # Run simulation for the normal case.
        _, validators_relative_dividends_normal, _, _ = run_dynamic_simulation(
            case=normal_case,
            yuma_version=yuma_version,
            yuma_config=yuma_config,
        )
        
        # Run simulation for the shifted case.
        _, validators_relative_dividends_shifted, _, _ = run_dynamic_simulation(
            case=shifted_case,
            yuma_version=yuma_version,
            yuma_config=yuma_config,
        )
        
        # Generate chart for the normal case.
        chart_normal = _plot_relative_dividends(
            validators_relative_dividends=validators_relative_dividends_normal,
            case_name=final_case_name_normal,
            case=normal_case,
            num_epochs=normal_case.num_epochs,
            epochs_padding=epochs_padding,
            to_base64=True,
        )
        
        # Generate chart for the shifted case.
        chart_shifted = _plot_relative_dividends(
            validators_relative_dividends=validators_relative_dividends_shifted,
            case_name=final_case_name_shifted,
            case=shifted_case,
            num_epochs=shifted_case.num_epochs,
            epochs_padding=epochs_padding,
            to_base64=True,
        )
        
        # Generate the comparisson chart using both simulation outputs.
        chart_comparisson = _plot_relative_dividends_comparisson(
            validators_relative_dividends_normal=validators_relative_dividends_normal,
            validators_relative_dividends_shifted=validators_relative_dividends_shifted,
            num_epochs=normal_case.num_epochs,  # Assuming both cases use the same number of epochs.
            epochs_padding=epochs_padding,
            case=normal_case,
            to_base64=True,
        )

        chart_comparisson_stake_scaled = _plot_relative_dividends_comparisson(
            validators_relative_dividends_normal=validators_relative_dividends_normal,
            validators_relative_dividends_shifted=validators_relative_dividends_shifted,
            num_epochs=normal_case.num_epochs,  # Assuming both cases use the same number of epochs.
            epochs_padding=epochs_padding,
            case=normal_case,
            to_base64=True,
            use_stakes=True,
        )
        
        # Append the three rows to this column.
        table_data[yuma_version].append(chart_normal)
        table_data[yuma_version].append(chart_shifted)
        table_data[yuma_version].append(chart_comparisson)
        table_data[yuma_version].append(chart_comparisson_stake_scaled)
    
    # Define row ranges for the table (here each row represents a particular chart type).
    # In the generated table, row 0 = normal_case chart, row 1 = shifted_case chart,
    # and row 2 = comparisson chart.
    case_row_ranges = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
    ]
    
    summary_table = pd.DataFrame(table_data)
    
    if draggable_table:
        full_html = _generate_draggable_html_table(table_data, summary_table, case_row_ranges)
    else:
        full_html = _generate_ipynb_table(table_data, summary_table, case_row_ranges)
    
    return HTML(full_html)