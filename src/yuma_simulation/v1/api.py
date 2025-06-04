import pandas as pd
import logging
from IPython.display import HTML

from yuma_simulation._internal.cases import BaseCase
from yuma_simulation._internal.charts_utils import (
    _plot_relative_dividends,
    _plot_relative_dividends_comparisson,
    _generate_chart_for_type,
    _plot_bonds_metagraph_dynamic,
    _plot_validator_server_weights_subplots_dynamic,
    _generate_relative_dividends_summary_html
)
from yuma_simulation._internal.simulation_utils import (
    _generate_draggable_html_table,
    _generate_ipynb_table,
    _run_simulation,
    _run_dynamic_simulation,
    _get_final_case_name,
    _get_final_case_names_dynamic,
)
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    YumaConfig,
    YumaParams,
    YumaSimulationNames,
)

logger = logging.getLogger("main_logger")


def generate_chart_table(
    cases: list[BaseCase],
    yuma_versions: list[tuple[str, YumaParams]],
    yuma_hyperparameters: SimulationHyperparameters,
    draggable_table: bool = False,
    chart_types: list[str] | None = None,
) -> HTML:
    """
    Generates an HTML table of charts for a list of cases across different yuma versions.
    """
    table_data: dict[str, list[str]] = {yuma_version: [] for yuma_version, _ in yuma_versions}
    case_row_ranges = []
    current_row_count = 0

    for idx, case in enumerate(cases):
        current_chart_types = chart_types or case.chart_types
        case_start = current_row_count

        simulation_cache: dict[str, tuple] = {}

        for chart_type in current_chart_types:
            chart_row: dict[str, str] = {}
            for yuma_version, yuma_params in yuma_versions:
                yuma_config = YumaConfig(simulation=yuma_hyperparameters, yuma_params=yuma_params)
                final_case_name = _get_final_case_name(case, yuma_version, yuma_config)

                if chart_type != "weights":
                    if yuma_version not in simulation_cache:
                        simulation_cache[yuma_version] = _run_simulation(
                            case=case,
                            yuma_version=yuma_version,
                            yuma_config=yuma_config,
                        )
                    simulation_results = simulation_cache[yuma_version]
                else:
                    simulation_results = None

                chart_row[yuma_version] = _generate_chart_for_type(
                    chart_type=chart_type,
                    case=case,
                    final_case_name=final_case_name,
                    simulation_results=simulation_results,
                    to_base64=True,
                )

            for yuma_version, chart_base64 in chart_row.items():
                table_data[yuma_version].append(chart_base64)

            current_row_count += 1

        case_row_ranges.append((case_start, current_row_count - 1, idx))

    summary_table = pd.DataFrame(table_data)
    if draggable_table:
        full_html = _generate_draggable_html_table(table_data, summary_table, case_row_ranges)
    else:
        full_html = _generate_ipynb_table(table_data, summary_table, case_row_ranges)

    return HTML(full_html)

def generate_metagraph_based_chart_table(
    chart_versions: tuple[str, YumaParams] | list[tuple[str, YumaParams]],
    summary_versions: tuple[str, YumaParams] | list[tuple[str, YumaParams]],
    normal_case: BaseCase,
    yuma_hyperparameters: SimulationHyperparameters,
    epochs_padding: int,
    diff_versions: tuple[str, str] | None = None,
    draggable_table: bool = False,
) -> HTML:
    """
    Generate charts for `chart_versions` and a summary table across `summary_versions`.

    - `chart_versions`: one or more (version, params) tuples for which to render charts.
    - `summary_versions`: one or more (version, params) tuples to include in the summary table.
    """
    # Normalize inputs to lists
    if isinstance(chart_versions, tuple) and not isinstance(chart_versions[0], (list, tuple)):
        chart_versions = [chart_versions]
    if isinstance(summary_versions, tuple) and not isinstance(summary_versions[0], (list, tuple)):
        summary_versions = [summary_versions]

    # Cache simulation outputs for all summary_versions
    rel_divs_by_version: dict[str, dict[str, list[float]]] = {}
    bonds_by_version: dict[str, list[torch.Tensor]] = {}
    for version, params in summary_versions:
        config = YumaConfig(simulation=yuma_hyperparameters, yuma_params=params)
        _, rel_divs, bonds, _ = _run_dynamic_simulation(
            case=normal_case,
            yuma_version=version,
            yuma_config=config,
        )
        rel_divs_by_version[version] = rel_divs
        bonds_by_version[version] = bonds

    table_data: dict[str, list[str]] = {v: [] for v, _ in chart_versions}
    top_vals = getattr(normal_case, "top_validators_hotkeys", None) or list(normal_case.validators)

    for version, params in chart_versions:
        config = YumaConfig(simulation=yuma_hyperparameters, yuma_params=params)
        final_name = _get_final_case_name(normal_case, version, config)

        
        if version in rel_divs_by_version:
            rel_divs = rel_divs_by_version[version]
            bonds    = bonds_by_version[version]
        else:
            _, rel_divs, bonds, _ = _run_dynamic_simulation(
                case         = normal_case,
                yuma_version = version,
                yuma_config  = config,
            )

        chart_rel = _plot_relative_dividends(
            validators_relative_dividends=rel_divs,
            case_name=final_name,
            case=normal_case,
            num_epochs=normal_case.num_epochs,
            epochs_padding=epochs_padding,
            to_base64=True,
        )
        chart_weights = _plot_validator_server_weights_subplots_dynamic(
            case=normal_case,
            case_name=final_name,
            to_base64=True,
            epochs_padding=epochs_padding,
        )
        chart_bonds = _plot_bonds_metagraph_dynamic(
            case=normal_case,
            bonds_per_epoch=bonds,
            case_name=final_name,
            to_base64=True,
            epochs_padding=epochs_padding,
        )
        chart_bonds_norm = _plot_bonds_metagraph_dynamic(
            case=normal_case,
            bonds_per_epoch=bonds,
            case_name=final_name,
            to_base64=True,
            normalize=True,
            epochs_padding=epochs_padding,
        )
        table_data[version].extend([chart_rel, chart_weights, chart_bonds, chart_bonds_norm])

    summary_html = _generate_relative_dividends_summary_html(
        relative_dividends_by_version={v: rel_divs_by_version[v] for v, _ in summary_versions},
        top_validators=top_vals,
        diff_versions=diff_versions,
        epochs_padding=epochs_padding,
        num_epochs=normal_case.num_epochs,
        alpha_tao_ratio=yuma_hyperparameters.alpha_tao_ratio,
        label_map=normal_case.hotkey_label_map,
    )

    for version, _ in chart_versions:
        table_data[version].append(summary_html)

    total_rows = len(next(iter(table_data.values())))
    case_row_ranges = [(i, i, i) for i in range(total_rows)]
    df_full = pd.DataFrame(table_data)
    full_html = _generate_draggable_html_table(table_data, df_full, case_row_ranges)
    return HTML(full_html)


def generate_metagraph_based_chart_table_shifted_comparisson(
    yuma_versions: list[tuple[str, YumaParams]],
    normal_case: BaseCase,
    shifted_case: BaseCase,
    yuma_hyperparameters: SimulationHyperparameters,
    epochs_padding: int,
    draggable_table: bool = False,
) -> HTML:
    """
    Generates an HTML table with one column per yuma_version.
    
    For each yuma_version the table includes four rows:
      - Row 0: The relative dividends chart for the normal case.
      - Row 1: The relative dividends chart for the shifted case.
      - Row 2: A comparisson chart that uses both cases relative dividends.
      - Row 3: A comparisson chart with stake scaling.
    """
    table_data: dict[str, list[str]] = {yuma_version: [] for yuma_version, _ in yuma_versions}
    yuma_names = YumaSimulationNames()

    for yuma_version, yuma_params in yuma_versions:
        yuma_config = YumaConfig(simulation=yuma_hyperparameters, yuma_params=yuma_params)

        final_case_name_normal, final_case_name_shifted = _get_final_case_names_dynamic(
            normal_case, shifted_case, yuma_version, yuma_config
        )

        _, validators_relative_dividends_normal, bonds_per_epoch, _ = _run_dynamic_simulation(
            case=normal_case,
            yuma_version=yuma_version,
            yuma_config=yuma_config,
        )
        _, validators_relative_dividends_shifted, _, _ = _run_dynamic_simulation(
            case=shifted_case,
            yuma_version=yuma_version,
            yuma_config=yuma_config,
        )

        chart_normal = _plot_relative_dividends(
            validators_relative_dividends=validators_relative_dividends_normal,
            case_name=final_case_name_normal,
            case=normal_case,
            num_epochs=normal_case.num_epochs,
            epochs_padding=epochs_padding,
            to_base64=True,
        )
        chart_shifted = _plot_relative_dividends(
            validators_relative_dividends=validators_relative_dividends_shifted,
            case_name=final_case_name_shifted,
            case=shifted_case,
            num_epochs=shifted_case.num_epochs,
            epochs_padding=epochs_padding,
            to_base64=True,
        )
        chart_comparisson = _plot_relative_dividends_comparisson(
            validators_relative_dividends_normal=validators_relative_dividends_normal,
            validators_relative_dividends_shifted=validators_relative_dividends_shifted,
            num_epochs=normal_case.num_epochs,  # Assuming same epochs.
            epochs_padding=epochs_padding,
            case=normal_case,
            to_base64=True,
        )
        chart_comparisson_stake_scaled = _plot_relative_dividends_comparisson(
            validators_relative_dividends_normal=validators_relative_dividends_normal,
            validators_relative_dividends_shifted=validators_relative_dividends_shifted,
            num_epochs=normal_case.num_epochs,
            epochs_padding=epochs_padding,
            case=normal_case,
            to_base64=True,
            use_stakes=True,
        )
        
        table_data[yuma_version].extend([
            chart_normal,
            chart_shifted,
            chart_comparisson,
            chart_comparisson_stake_scaled,
        ])

    case_row_ranges = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    summary_table = pd.DataFrame(table_data)
    if draggable_table:
        full_html = _generate_draggable_html_table(table_data, summary_table, case_row_ranges)
    else:
        full_html = _generate_ipynb_table(table_data, summary_table, case_row_ranges)

    return HTML(full_html)