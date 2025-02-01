"""
This module provides functionalities to run Yuma simulations, generate charts, and produce tables of results.
It integrates various Yuma versions, handles different chart types, and organizes the outputs into HTML tables.
"""

import pandas as pd
import torch

from yuma_simulation._internal.cases import BaseCase
from yuma_simulation._internal.charts_utils import (
    _calculate_total_dividends,
    _calculate_total_dividends_with_frames,
)
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    Yuma,
    Yuma2,
    Yuma3,
    Yuma4,
    YumaConfig,
    YumaParams,
    YumaRust,
    YumaSimulationNames,
)


def run_simulation(
    case: BaseCase,
    yuma_version: str,
    yuma_config: YumaConfig,
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[torch.Tensor], list[torch.Tensor]]:
    """Runs the Yuma simulation for a given case and Yuma version, returning dividends, bonds and incentive data."""

    # TODO(Konrad): make sure that the validators in the metagraph cases are used per epoch (because they can change)
    dividends_per_validator: dict[str, list[float]] = {
        validator: [] for validator in case.validators
    }
    bonds_per_epoch: list[torch.Tensor] = []
    server_incentives_per_epoch: list[torch.Tensor] = []
    relative_dividends_per_validator: dict[str, list[float]] = {
        validator: [] for validator in case.validators
    }
    B_state: torch.Tensor | None = None
    W_prev: torch.Tensor | None = None
    server_consensus_weight: torch.Tensor | None = None

    simulation_names = YumaSimulationNames()

    for epoch in range(case.num_epochs):
        W: torch.Tensor = case.weights_epochs[epoch]
        S: torch.Tensor = case.stakes_epochs[epoch]

        stakes_tao: torch.Tensor = S * yuma_config.total_subnet_stake
        stakes_units: torch.Tensor = stakes_tao / 1000.0

        # Call the appropriate Yuma function
        if yuma_version in [simulation_names.YUMA, simulation_names.YUMA_LIQUID]:
            result = Yuma(W=W, S=S, B_old=B_state, config=yuma_config)
            B_state = result["validator_ema_bond"]
        elif yuma_version == simulation_names.YUMA2:
            result = Yuma2(W=W, W_prev=W_prev, S=S, B_old=B_state, config=yuma_config)
            B_state = result["validator_ema_bond"]
            W_prev = result["weight"]
        elif yuma_version == simulation_names.YUMA3:
            result = Yuma3(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_bonds"]
        elif yuma_version == simulation_names.YUMA31:
            if B_state is not None and epoch == case.reset_bonds_epoch:
                B_state[:, case.reset_bonds_index] = 0.0
            result = Yuma3(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_bonds"]
        elif yuma_version == simulation_names.YUMA32:
            if (
                B_state is not None
                and epoch == case.reset_bonds_epoch
                and server_consensus_weight is not None
                and server_consensus_weight[case.reset_bonds_index] == 0.0
            ):
                B_state[:, case.reset_bonds_index] = 0.0
            result = Yuma3(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_bonds"]
            server_consensus_weight = result["server_consensus_weight"]
        elif yuma_version in [simulation_names.YUMA4, simulation_names.YUMA4_LIQUID]:
            if (
                B_state is not None
                and epoch == case.reset_bonds_epoch
                and server_consensus_weight is not None
                and server_consensus_weight[case.reset_bonds_index] == 0.0
            ):
                B_state[:, case.reset_bonds_index] = 0.0
            result = Yuma4(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_bonds"]
            server_consensus_weight = result["server_consensus_weight"]
        elif yuma_version == "Yuma 0 (subtensor)":
            result = YumaRust(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_ema_bond"]
        else:
            raise ValueError("Invalid Yuma function.")

        D_normalized: torch.Tensor = result["validator_reward_normalized"]

        E_i: torch.Tensor = yuma_config.validator_emission_ratio * D_normalized
        validator_emission: torch.Tensor = E_i * yuma_config.total_epoch_emission

        for i, validator in enumerate(case.validators):
            stake_unit = float(stakes_units[i].item())
            validator_emission_i = float(validator_emission[i].item())
            if stake_unit > 1e-6:
                dividend_per_1000_tao = validator_emission_i / stake_unit
            else:
                dividend_per_1000_tao = 0.0
            dividends_per_validator[validator].append(dividend_per_1000_tao)

        bonds_per_epoch.append(B_state.clone())
        server_incentives_per_epoch.append(result["server_incentive"])

        S = S / S.sum()

        for i, validator in enumerate(case.validators):
            relative_dividends_per_validator[validator].append(D_normalized[i].item() - S[i].item())

    return dividends_per_validator, relative_dividends_per_validator, bonds_per_epoch, server_incentives_per_epoch


def _generate_draggable_html_table(
    table_data: dict[str, list[str]],
    summary_table: pd.DataFrame,
    case_row_ranges: list[tuple[int, int, int]],
) -> str:
    custom_css_js = """
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        
        .scrollable-table-container {
            background-color: #FFFFFF; 
            width: 100%; 
            height: 100vh;
            overflow: auto;
            border: 1px solid #ccc;
            position: relative; 
            user-select: none;
            scrollbar-width: auto;
            -ms-overflow-style: auto;
            cursor: grab;
        }

        .scrollable-table-container:active {
            cursor: grabbing;
        }

        /* Remove any nth-child rules entirely */

        /* Classes for alternating case groups */
        .case-group-even td {
            background-color: #FFFFFF !important; 
        }
        .case-group-odd td {
            background-color: #F0F0F0 !important;
        }

        .scrollable-table-container img {
            user-select: none;
            -webkit-user-drag: none;
            pointer-events: none;
        }

        .scrollable-table-container::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }

        table {
            border-collapse: collapse;
            margin: 0;
            width: auto;
        }

        td, th {
            padding: 10px;
            vertical-align: top;
            text-align: center;
        }

    </style>

    <script>
        document.addEventListener('DOMContentLoaded', function () {
            const container = document.querySelector('.scrollable-table-container');
            let isDown = false;
            let startX, startY, scrollLeft, scrollTop;

            container.addEventListener('dragstart', function(e) {
                e.preventDefault();
            });

            container.addEventListener('mousedown', (e) => {
                e.preventDefault();
                isDown = true;
                startX = e.clientX;
                startY = e.clientY;
                scrollLeft = container.scrollLeft;
                scrollTop = container.scrollTop;
            });

            document.addEventListener('mouseup', () => {
                isDown = false;
            });

            document.addEventListener('mousemove', (e) => {
                if(!isDown) return;
                e.preventDefault();
                const x = e.clientX;
                const y = e.clientY;
                const walkX = x - startX;
                const walkY = y - startY;
                container.scrollLeft = scrollLeft - walkX;
                container.scrollTop = scrollTop - walkY;
            });
        });
    </script>
    """

    def get_case_index_for_row(row_idx):
        for start, end, c_idx in case_row_ranges:
            if start <= row_idx <= end:
                return c_idx
        return 0

    html_rows = []
    total_rows = len(next(iter(table_data.values())))
    for i in range(total_rows):
        case_idx = get_case_index_for_row(i)
        row_class = "case-group-even" if (case_idx % 2 == 0) else "case-group-odd"
        row_html = f"<tr class='{row_class}'>"
        for yuma_version in summary_table.columns:
            cell_content = summary_table[yuma_version][i]
            row_html += f"<td>{cell_content}</td>"
        row_html += "</tr>"
        html_rows.append(row_html)

    html_table = f"""
    <div class="scrollable-table-container">
        <table>
            <thead>
                <tr>{"".join(f"<th>{col}</th>" for col in summary_table.columns)}</tr>
            </thead>
            <tbody>
                {"".join(html_rows)}
            </tbody>
        </table>
    </div>
    """

    return custom_css_js + html_table


def _generate_ipynb_table(
    table_data: dict[str, list[str]],
    summary_table: pd.DataFrame,
    case_row_ranges: list[tuple[int, int, int]],
) -> str:
    custom_css = """
    <style>
        .scrollable-table-container {
            background-color: #FFFFFF;
            width: 100%; 
            overflow-x: auto;
            overflow-y: hidden;
            white-space: nowrap;
            border: 1px solid #ccc;
        }
        table {
            border-collapse: collapse;
            table-layout: auto;
            width: auto;
        }
        td, th {
            padding: 10px;
            vertical-align: top;
            text-align: center;
        }

        /* Classes for alternating case groups */
        .case-group-even td {
            background-color: #FFFFFF !important;
        }
        .case-group-odd td {
            background-color: #F8F8F8 !important;
        }
    </style>
    """

    def get_case_index_for_row(row_idx):
        for start, end, c_idx in case_row_ranges:
            if start <= row_idx <= end:
                return c_idx
        return 0

    html_rows = []
    num_rows = len(next(iter(table_data.values())))
    for i in range(num_rows):
        case_idx = get_case_index_for_row(i)
        row_class = "case-group-even" if (case_idx % 2 == 0) else "case-group-odd"
        row_html = f"<tr class='{row_class}'>"
        for yuma_version in summary_table.columns:
            cell_content = summary_table[yuma_version][i]
            row_html += f"<td>{cell_content}</td>"
        row_html += "</tr>"
        html_rows.append(row_html)

    html_table = f"""
    <div class="scrollable-table-container">
        <table>
            <thead>
                <tr>{"".join(f"<th>{col}</th>" for col in summary_table.columns)}</tr>
            </thead>
            <tbody>
                {"".join(html_rows)}
            </tbody>
        </table>
    </div>
    """
    return custom_css + html_table


def generate_total_dividends_table(
    cases: list[BaseCase],
    yuma_versions: list[tuple[str, YumaParams]],
    simulation_hyperparameters: SimulationHyperparameters,
    is_metagraph: bool = False,
) -> pd.DataFrame:
    """
    Generates a DataFrame of total dividends for validator names
    across multiple Yuma versions.

    If is_metagraph=True, it will not apply standardization (i.e.,
    use the names in case.validators directly).
    """

    all_column_names = set()
    rows = []

    for case in cases:
        # Decide how to name the validators
        if is_metagraph:
            # Use the validators as-is (e.g., hotkeys from MetagraphCase)
            final_validator_names = case.validators
        else:
            # Apply your original "Validator A/B/C" scheme
            final_validator_names = [
                f"Validator {chr(ord('A') + i)}" for i in range(len(case.validators))
            ]

        # Create a mapping from original name -> final display name
        validator_mapping = dict(zip(case.validators, final_validator_names))

        # Build the row for this case
        row = {"Case": case.name}

        # Run each Yuma version simulation
        for yuma_version, yuma_params in yuma_versions:
            yuma_config = YumaConfig(
                simulation=simulation_hyperparameters,
                yuma_params=yuma_params,
            )

            # Run the simulation
            dividends_per_validator, _, _, _ = run_simulation(
                case=case,
                yuma_version=yuma_version,
                yuma_config=yuma_config,
            )

            total_dividends, _ = _calculate_total_dividends(
                validators=case.validators,
                dividends_per_validator=dividends_per_validator,
                base_validator=case.base_validator,
                num_epochs=case.num_epochs,
            )

            # Map original validator names to our final (display) names
            final_dividends = {
                validator_mapping[original_val]: total_dividends.get(original_val, 0.0)
                for original_val in case.validators
            }

            # Store results in the row
            for val_name in final_validator_names:
                column_name = f"{val_name} - {yuma_version}"
                row[column_name] = final_dividends.get(val_name, 0.0)
                all_column_names.add(column_name)

        rows.append(row)

    # Build the DataFrame
    df = pd.DataFrame(rows)

    columns = ["Case"]
    for yuma_version, _ in yuma_versions:
        version_columns = sorted(
            col for col in all_column_names if col.endswith(f"- {yuma_version}")
        )
        columns.extend(version_columns)

    # Reindex DataFrame so all columns appear. Fill missing with 0.0
    df = df.reindex(columns=columns, fill_value=0.0)

    return df


def generate_metagraph_case_shifted_validator_comparison_table(
    case_normal: BaseCase,
    case_shifted: BaseCase,
    yuma_versions: list[tuple[str, YumaParams]],
    simulation_hyperparameters: SimulationHyperparameters,
    epochs_window: int,
) -> pd.DataFrame:
    """
    Compares the *relative dividends* of a single validator (often the base_validator)
    across two different MetagraphCase objects: normal vs. shifted,
    for multiple Yuma versions. Each Yuma version produces two columns:
      - Normal_<version>
      - Shifted_<version>

    Example final columns for 2 Yuma versions:
      Window | Normal_v1 | Shifted_v1 | Normal_v2 | Shifted_v2
      --------------------------------------------------------
      1-20   | +10.00%   | +20.00%    | -5.00%    | +3.00%
      21-40  | +15.00%   | +25.00%    | -2.00%    | +1.00%
      Total  | +12.50%   | +22.50%    | -3.50%    | +2.00%
    """

    rows = []

    version_frames = {}
    version_collector = {}
    validator_normal = case_normal.base_validator
    validator_shifted = case_shifted.base_validator

    for (yuma_version_name, yuma_params) in yuma_versions:
        single_config = YumaConfig(
            simulation=simulation_hyperparameters,
            yuma_params=yuma_params,
        )

        _, relative_dividends_normal, _, _ = run_simulation(
            case=case_normal,
            yuma_version=yuma_version_name,
            yuma_config=single_config,
        )
        _, relative_dividends_shifted, _, _ = run_simulation(
            case=case_shifted,
            yuma_version=yuma_version_name,
            yuma_config=single_config,
        )

        divs_normal = relative_dividends_normal.get(validator_normal, [])
        divs_shifted = relative_dividends_shifted.get(validator_shifted, [])

        normal_frames, _ = _calculate_total_dividends_with_frames(
            validator_dividends=divs_normal,
            num_epochs=case_normal.num_epochs,
            epochs_window=epochs_window,
            use_relative=True
        )
        shifted_frames, _ = _calculate_total_dividends_with_frames(
            validator_dividends=divs_shifted,
            num_epochs=case_shifted.num_epochs,
            epochs_window=epochs_window,
            use_relative=True
        )

        num_frames = min(len(normal_frames), len(shifted_frames))
        version_frames[yuma_version_name] = {
            "normal_frames": normal_frames,
            "shifted_frames": shifted_frames,
            "num_frames": num_frames
        }

        version_collector[yuma_version_name] = {
            "normal_values": [],
            "shifted_values": []
        }

    max_frames = max(vdata["num_frames"] for vdata in version_frames.values()) if version_frames else 0

    for i in range(max_frames):
        start_epoch = i * epochs_window + 1
        end_epoch = (i + 1) * epochs_window
        end_epoch = min(end_epoch, case_normal.num_epochs, case_shifted.num_epochs)

        row_data = {}
        row_data["Window"] = f"{start_epoch}-{end_epoch}"

        for (yuma_version_name, _) in yuma_versions:
            vdata = version_frames[yuma_version_name]

            if i < vdata["num_frames"]:
                avg_normal = vdata["normal_frames"][i]
                avg_shifted = vdata["shifted_frames"][i]
            else:
                avg_normal = 0.0
                avg_shifted = 0.0

            normal_str = f"{avg_normal * 100:+.2f}%"
            shifted_str = f"{avg_shifted * 100:+.2f}%"

            version_collector[yuma_version_name]["normal_values"].append(avg_normal)
            version_collector[yuma_version_name]["shifted_values"].append(avg_shifted)

            row_data[f"Normal_{yuma_version_name}"] = normal_str
            row_data[f"Shifted_{yuma_version_name}"] = shifted_str

        rows.append(row_data)

    total_row = {"Window": "Total"}
    for (yuma_version_name, _) in yuma_versions:
        normal_list = version_collector[yuma_version_name]["normal_values"]
        shifted_list = version_collector[yuma_version_name]["shifted_values"]

        if normal_list:
            avg_normal_across_frames = sum(normal_list) / len(normal_list)
        else:
            avg_normal_across_frames = 0.0

        if shifted_list:
            avg_shifted_across_frames = sum(shifted_list) / len(shifted_list)
        else:
            avg_shifted_across_frames = 0.0

        total_normal_str = f"{avg_normal_across_frames * 100:+.2f}%"
        total_shifted_str = f"{avg_shifted_across_frames * 100:+.2f}%"

        total_row[f"Normal_{yuma_version_name}"] = total_normal_str
        total_row[f"Shifted_{yuma_version_name}"] = total_shifted_str

    rows.append(total_row)

    df = pd.DataFrame(rows)

    column_order = ["Window"]
    for (yuma_version_name, _) in yuma_versions:
        column_order.append(f"Normal_{yuma_version_name}")
        column_order.append(f"Shifted_{yuma_version_name}")

    column_order = [c for c in column_order if c in df.columns]
    df = df[column_order]

    return df