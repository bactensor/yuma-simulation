"""
This module provides functionalities to run Yuma simulations, generate charts, and produce tables of results.
It integrates various Yuma versions, handles different chart types, and organizes the outputs into HTML tables.
"""

import pandas as pd
import torch
import logging

from yuma_simulation._internal.cases import BaseCase
from yuma_simulation._internal.charts_utils import (
    _calculate_total_dividends,
    _calculate_total_dividends_with_frames,
)
from yuma_simulation._internal.yumas import (
    SimulationHyperparameters,
    Yuma2b,
    Yuma2c,
    Yuma3,
    YumaConfig,
    YumaParams,
    YumaRust,
    YumaSimulationNames,
)

logger = logging.getLogger("main_logger")


def _run_simulation(
    case: BaseCase,
    yuma_version: str,
    yuma_config: YumaConfig,
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[torch.Tensor], list[torch.Tensor]]:
    """Runs the Yuma simulation for a given case and Yuma version, returning dividends, bonds and incentive data."""

    dividends_per_validator: dict[str, list[float]] = {
        validator: [] for validator in case.validators
    }
    bonds_per_epoch: list[torch.Tensor] = []
    server_incentives_per_epoch: list[torch.Tensor] = []
    relative_dividends_per_validator: dict[str, list[float]] = {
        validator: [] for validator in case.validators
    }

    # These states are passed between epochs.
    B_state: torch.Tensor | None = None
    C_state: torch.Tensor | None = None
    W_prev: torch.Tensor | None = None
    server_consensus_weight: torch.Tensor | None = None


    for epoch in range(case.num_epochs):
        W: torch.Tensor = case.weights_epochs[epoch]
        S: torch.Tensor = case.stakes_epochs[epoch]

        simulation_results, B_state, C_state, W_prev, server_consensus_weight = _call_yuma(
            epoch=epoch,
            yuma_version=yuma_version,
            W=W,
            S=S,
            B_state=B_state,
            C_state=C_state,
            W_prev=W_prev,
            server_consensus_weight=server_consensus_weight,
            case=case,
            yuma_config=yuma_config,
        )

        D_normalized: torch.Tensor = simulation_results["validator_reward_normalized"]

        _update_validators_dividends(
            D_normalized=D_normalized,
            S=S,
            yuma_config=yuma_config,
            validators_list=case.validators,
            dividends_per_validator=dividends_per_validator,
        )

        b = B_state.clone()
        i = simulation_results["server_incentive"].clone()

        if case.use_full_matrices:
            b, i = _slice_tensors(
                b,
                i,
                num_validators=len(case.validators),
                num_servers=len(case.servers)
            )

        bonds_per_epoch.append(b)
        server_incentives_per_epoch.append(i)

        _update_validators_relative_dividends(
            D_normalized=D_normalized,
            S=S,
            validators_list=case.validators,
            relative_dividends_per_validator=relative_dividends_per_validator,
        )

    return dividends_per_validator, relative_dividends_per_validator, bonds_per_epoch, server_incentives_per_epoch


def _run_dynamic_simulation(
    case: BaseCase,
    yuma_version: str,
    yuma_config: YumaConfig,
) -> tuple[
    dict[str, list[float]],
    dict[str, list[float]],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    """
    Runs the Yuma simulation for a case whose validators change each epoch.
    Instead of accumulating dividends keyed by a single static list of validators,
    this version merges the per-epoch dividend data into dictionaries mapping each
    validator to its time-series.
    Its used specifically for real metagraphs archived data.
    """
    dividends_per_epoch: list[dict[str, float]] = []
    relative_dividends_per_epoch: list[dict[str, float]] = []
    bonds_per_epoch: list[torch.Tensor] = []
    server_incentives_per_epoch: list[torch.Tensor] = []

    # These states are passed between epochs.
    B_state: torch.Tensor | None = None
    C_state: torch.Tensor | None = None
    W_prev: torch.Tensor | None = None
    server_consensus_weight: torch.Tensor | None = None

    for epoch in range(case.num_epochs):
        W: torch.Tensor = case.weights_epochs[epoch]
        S: torch.Tensor = case.stakes_epochs[epoch]
        current_validators: list[str] = case.validators_epochs[epoch]
        current_miner_indices: list[int] = case.miner_indices_epochs[epoch]

        current_validator_count = len(current_validators)
        current_miner_count = len(current_miner_indices)

        should_align_bond_state = (
            B_state is not None
            and (B_state.shape[0] != current_validator_count or B_state.shape[1] != current_miner_count)
        )
        if should_align_bond_state:
            if epoch > 0:
                old_validators: list[str] = case.validators_epochs[epoch - 1]
                old_miner_indices: list[int] = case.miner_indices_epochs[epoch - 1]
            else:
                old_validators, old_miner_indices = [], []
            B_state = _align_bond_state(
                B_state=B_state,
                current_validators=current_validators,
                current_miner_indices=current_miner_indices,
                old_validators=old_validators,
                old_miner_indices=old_miner_indices,
            )
        
        should_align_consensus_state = (
            C_state is not None 
            and (C_state.shape[0] != current_miner_count)
        )

        if should_align_consensus_state:
            if epoch > 0:
                old_miner_indices = case.miner_indices_epochs[epoch - 1]
            else:
                old_miner_indices = []
            C_state = _align_weights_consensus_state(
                C_state=C_state,
                current_miner_indices=current_miner_indices,
                old_miner_indices=old_miner_indices,
            )

        if W_prev is not None and (
            W_prev.shape[0] != current_validator_count
            or W_prev.shape[1] != current_miner_count
        ):
            old_vals = case.validators_epochs[epoch-1]    if epoch > 0 else []
            old_mins = case.miner_indices_epochs[epoch-1] if epoch > 0 else []
            cur_vals = current_validators
            cur_mins = current_miner_indices

            W_prev = _align_matrix(
                W_prev,
                old_rows=old_vals,
                new_rows=cur_vals,
                old_cols=old_mins,
                new_cols=cur_mins,
            )

        simulation_results, B_state, C_state, W_prev, server_consensus_weight = _call_yuma(
            epoch=epoch,
            yuma_version=yuma_version,
            W=W,
            S=S,
            B_state=B_state,
            C_state=C_state,
            W_prev=W_prev,
            server_consensus_weight=server_consensus_weight,
            case=case,
            yuma_config=yuma_config
        )

        D_normalized: torch.Tensor = simulation_results["validator_reward_normalized"]

        b = B_state.clone()
        i = simulation_results["server_incentive"].clone()

        if case.use_full_matrices:
            b, i = _slice_tensors(
                b,
                i,
                num_validators=len(current_validators),
                num_servers=len(case.servers[epoch])
            )

        bonds_per_epoch.append(b)
        server_incentives_per_epoch.append(i)

        dividends_this_epoch = _compute_dividends_for_epoch(
            D_normalized=D_normalized,
            S=S,
            yuma_config=yuma_config,
            validators_list=current_validators,
        )

        S_norm = S / S.sum()
        relative_dividends_this_epoch: dict[str, float] = {}
        for i, validator in enumerate(current_validators):
            relative_dividends_this_epoch[validator] = D_normalized[i].item() - S_norm[i].item()

        dividends_per_epoch.append(dividends_this_epoch)
        relative_dividends_per_epoch.append(relative_dividends_this_epoch)

    # Merge the per-epoch dictionaries
    merged_dividends = pd.DataFrame(dividends_per_epoch).to_dict(orient="list")
    merged_relative_dividends = pd.DataFrame(relative_dividends_per_epoch).to_dict(orient="list")

    return (merged_dividends, merged_relative_dividends, bonds_per_epoch, server_incentives_per_epoch)

def _call_yuma(
    epoch: int,
    yuma_version: str,
    W: torch.Tensor,
    S: torch.Tensor,
    B_state: torch.Tensor | None,
    C_state: torch.Tensor | None,
    W_prev: torch.Tensor | None,
    server_consensus_weight: torch.Tensor | None,
    case: BaseCase,
    yuma_config: YumaConfig
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Calls the correct Yuma function based on `yuma_version`.
    Handles any bond resets if needed.
    """
    simulation_names = YumaSimulationNames()

    should_reset_bonds = (
        case.reset_bonds and (
        (yuma_version in [
            simulation_names.YUMA2C,
            simulation_names.YUMA3,
            simulation_names.YUMA3_LIQUID,
        ]
        and B_state is not None 
        and epoch == case.reset_bonds_epoch
        ))
    )

    if should_reset_bonds and case.use_full_matrices:
        idx = len(case.validators) + case.reset_bonds_index
        B_state[:, idx] = 0.0
    elif should_reset_bonds:
        B_state[:, case.reset_bonds_index] = 0.0

    if yuma_version == simulation_names.YUMA2B:
        result = Yuma2b(
            W=W,
            W_prev=W_prev,
            S=S,
            B_old=B_state,
            C_old=C_state,
            config=yuma_config,
            num_servers=len(case.servers),
            num_validators=len(case.validators),
            use_full_matrices=case.use_full_matrices
        )
        B_state = result["validator_ema_bond"]
        C_state = result["server_consensus_weight"]
        W_prev = result["weight"]

    elif yuma_version == simulation_names.YUMA2C:
        result = Yuma2c(
            W,
            S,
            B_old=B_state,
            config=yuma_config,
        )
        B_state = result["validator_bonds"]
        C_state = result["server_consensus_weight"]

    elif yuma_version in [simulation_names.YUMA3, simulation_names.YUMA3_LIQUID]:
        result = Yuma3(
            W,
            S,
            B_old=B_state,
            C_old=C_state,
            config=yuma_config,
            num_servers=len(case.servers),
            num_validators=len(case.validators),
            use_full_matrices=case.use_full_matrices
        )
        B_state = result["validator_bonds"]
        C_state = result["server_consensus_weight"]

    elif yuma_version in [simulation_names.YUMA1, simulation_names.YUMA2]:
        result = YumaRust(
            W,
            S,
            B_old=B_state,
            C_old=C_state,
            config=yuma_config,
            num_servers=len(case.servers),
            num_validators=len(case.validators),
            use_full_matrices=case.use_full_matrices
        )
        B_state = result["validator_ema_bond"]
        C_state = result["server_consensus_weight"]

    else:
        raise ValueError(f"Invalid Yuma function: {yuma_version}")

    return result, B_state, C_state, W_prev, server_consensus_weight


def _update_validators_relative_dividends(
    D_normalized: torch.Tensor,
    S: torch.Tensor,
    validators_list: list[str],
    relative_dividends_per_validator: dict[str, list[float]],
) -> dict[str, float]:
    """
    Mutates `relative_dividends_per_validator` by appending new relative dividends
    values for each validator.
    """
    S_norm = S / S.sum()
    for i, validator in enumerate(validators_list):
        relative_dividends_per_validator[validator].append(D_normalized[i].item() - S_norm[i].item())

def _compute_dividend_for_validator(
    i: int, 
    S: torch.Tensor, 
    D_normalized: torch.Tensor, 
    yuma_config: YumaConfig
) -> float:
    """
    Computes the dividend per 1000 tao for the validator at index `i`.
    """
    # Calculate the stakes in tao and convert to stake units (per 1000 tao)
    stakes_tao = S * yuma_config.total_subnet_stake
    stakes_units = stakes_tao / 1000.0

    # Compute the emission for each validator
    emission_ratio = yuma_config.validator_emission_ratio
    E_i = emission_ratio * D_normalized
    validator_emission = E_i * yuma_config.total_epoch_emission

    # Retrieve values as Python floats
    stake_unit = float(stakes_units[i].item())
    emission_val = float(validator_emission[i].item())

    return emission_val / stake_unit if stake_unit > 1e-6 else 0.0

def _update_validators_dividends(
    D_normalized: torch.Tensor,
    S: torch.Tensor,
    yuma_config: YumaConfig,
    validators_list: list[str],
    dividends_per_validator: dict[str, list[float]],
) -> None:
    """
    Updates a dividends dictionary (mapping validator names to a list of dividend values)
    by appending the computed dividend for each validator.
    """
    for i, validator in enumerate(validators_list):
        dividend = _compute_dividend_for_validator(i, S, D_normalized, yuma_config)
        dividends_per_validator[validator].append(dividend)

def _compute_dividends_for_epoch(
    D_normalized: torch.Tensor,
    S: torch.Tensor,
    yuma_config: YumaConfig,
    validators_list: list[str],
) -> dict[str, float]:
    """
    Computes a dictionary mapping each validator (by name) to its dividend for the current epoch.
    """
    dividends_this_epoch = {}
    for i, validator in enumerate(validators_list):
        dividend = _compute_dividend_for_validator(i, S, D_normalized, yuma_config)
        dividends_this_epoch[validator] = dividend
    return dividends_this_epoch


def _align_bond_state(
    B_state: torch.Tensor,
    current_validators: list[str],
    current_miner_indices: list[int],
    old_validators: list[str],
    old_miner_indices: list[int],
) -> torch.Tensor:
    """
    Aligns the previous bond state (B_state) with the current epoch's validators
    and miner indices. Returns a new bond state tensor with shape
      (len(current_validators), len(current_miner_indices)),
    copying over any overlapping entries from the old bond state.
    """

    new_B_state = torch.zeros(
        len(current_validators),
        len(current_miner_indices),
        dtype=B_state.dtype,
        device=B_state.device,
    )
    for i, cur_validator in enumerate(current_validators):
        for j, cur_miner in enumerate(current_miner_indices):
            if cur_validator in old_validators and cur_miner in old_miner_indices:
                old_i = old_validators.index(cur_validator)
                old_j = old_miner_indices.index(cur_miner)
                new_B_state[i, j] = B_state[old_i, old_j]
            else:
                new_B_state[i, j] = 0.0
    return new_B_state

def _align_matrix(
    mat: torch.Tensor,
    old_rows: list[int],
    new_rows: list[int],
    old_cols: list[int],
    new_cols: list[int],
) -> torch.Tensor:
    """
    Reindex `mat` from shape [len(old_rows)×len(old_cols)]
    → [len(new_rows)×len(new_cols)] by:
      • dropping any (row,col) not in the old lists,
      • zero-padding any new row or col.
    """
    out = mat.new_zeros((len(new_rows), len(new_cols)))
    row_map = {uid: i for i, uid in enumerate(old_rows)}
    col_map = {uid: j for j, uid in enumerate(old_cols)}

    for i, r in enumerate(new_rows):
        for j, c in enumerate(new_cols):
            if r in row_map and c in col_map:
                out[i, j] = mat[row_map[r], col_map[c]]
    return out

def _align_weights_consensus_state(
    C_state: torch.Tensor,
    current_miner_indices: list[int],
    old_miner_indices: list[int],
) -> torch.Tensor:
    """
    Align the previous consensus state (C_state) with the current epoch's
    miner indices. Returns a new consensus state tensor with shape
      (len(current_miner_indices),),
    copying over any overlapping entries from the old consensus state.
    """

    new_C_state = torch.zeros(
        len(current_miner_indices),
        dtype=C_state.dtype,
        device=C_state.device,
    )
    for j, cur_miner in enumerate(current_miner_indices):
        if cur_miner in old_miner_indices:
            old_j = old_miner_indices.index(cur_miner)
            new_C_state[j] = C_state[old_j]
        else:
            new_C_state[j] = 0.0

    return new_C_state


def _generate_draggable_html_table(
    table_data: dict[str, list[str]],
    summary_table: pd.DataFrame | None,
    case_row_ranges: list[tuple[int, int, int]],
) -> str:
    """
    Generates a draggable HTML table with custom CSS/JS.
    
    If a summary_table is not provided, it is constructed from table_data.
    """
    if summary_table is None:
        summary_table = pd.DataFrame(table_data)

    custom_css_js = """
    <style>
        body { margin: 0; padding: 0; overflow: hidden; }
        .scrollable-table-container {
            background-color: #FFFFFF; width: 100%; height: 100vh;
            overflow: auto; border: 1px solid #ccc; position: relative;
            user-select: none; scrollbar-width: auto; -ms-overflow-style: auto;
            cursor: grab;
        }
        .scrollable-table-container:active { cursor: grabbing; }
        .case-group-even td { background-color: #FFFFFF !important; }
        .case-group-odd td { background-color: #F0F0F0 !important; }
        .scrollable-table-container img {
            user-select: none; -webkit-user-drag: none; pointer-events: none;
        }
        .scrollable-table-container::-webkit-scrollbar { width: 10px; height: 10px; }
        table { border-collapse: collapse; margin: 0; width: auto; }
        td, th { padding: 10px; vertical-align: top; text-align: center; }
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function () {
            const container = document.querySelector('.scrollable-table-container');
            let isDown = false, startX, startY, scrollLeft, scrollTop;
            container.addEventListener('dragstart', function(e) { e.preventDefault(); });
            container.addEventListener('mousedown', (e) => {
                e.preventDefault();
                isDown = true;
                startX = e.clientX; startY = e.clientY;
                scrollLeft = container.scrollLeft; scrollTop = container.scrollTop;
            });
            document.addEventListener('mouseup', () => { isDown = false; });
            document.addEventListener('mousemove', (e) => {
                if(!isDown) return;
                e.preventDefault();
                const walkX = e.clientX - startX, walkY = e.clientY - startY;
                container.scrollLeft = scrollLeft - walkX;
                container.scrollTop = scrollTop - walkY;
            });
        });
    </script>
    """

    table_html = _build_html_table(summary_table, case_row_ranges)
    container_html = f'<div class="scrollable-table-container">{table_html}</div>'
    return custom_css_js + container_html

def _generate_ipynb_table(
    table_data: dict[str, list[str]],
    summary_table: pd.DataFrame | None,
    case_row_ranges: list[tuple[int, int, int]],
) -> str:
    """
    Generates an HTML table for Jupyter notebooks with custom CSS.
    
    If a summary_table is not provided, it is built from table_data.
    """
    if summary_table is None:
        summary_table = pd.DataFrame(table_data)
    
    custom_css = """
    <style>
        .scrollable-table-container {
            background-color: #FFFFFF; width: 100%;
            overflow-x: auto; overflow-y: hidden; white-space: nowrap;
            border: 1px solid #ccc;
        }
        table { border-collapse: collapse; table-layout: auto; width: auto; }
        td, th { padding: 10px; vertical-align: top; text-align: center; }
        .case-group-even td { background-color: #FFFFFF !important; }
        .case-group-odd td { background-color: #F8F8F8 !important; }
    </style>
    """
    table_html = _build_html_table(summary_table, case_row_ranges)
    container_html = f'<div class="scrollable-table-container">{table_html}</div>'
    return custom_css + container_html

def _generate_total_dividends_table(
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
        if is_metagraph:
            final_validator_names = case.validators
        else:
            final_validator_names = [
                f"Validator {chr(ord('A') + i)}" for i in range(len(case.validators))
            ]

        validator_mapping = dict(zip(case.validators, final_validator_names))
        row = {"Case": case.name}

        for yuma_version, yuma_params in yuma_versions:
            yuma_config = YumaConfig(
                simulation=simulation_hyperparameters,
                yuma_params=yuma_params,
            )

            dividends_per_validator, _, _, _ = _run_simulation(
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

            final_dividends = {
                validator_mapping[original_val]: total_dividends.get(original_val, 0.0)
                for original_val in case.validators
            }

            for val_name in final_validator_names:
                column_name = f"{val_name} - {yuma_version}"
                row[column_name] = final_dividends.get(val_name, 0.0)
                all_column_names.add(column_name)

        rows.append(row)

    df = pd.DataFrame(rows)

    columns = ["Case"]
    for yuma_version, _ in yuma_versions:
        version_columns = sorted(
            col for col in all_column_names if col.endswith(f"- {yuma_version}")
        )
        columns.extend(version_columns)

    df = df.reindex(columns=columns, fill_value=0.0)

    return df

def _generate_relative_dividends_comparisson_table(
    case_normal: BaseCase,
    case_shifted: BaseCase,
    yuma_versions: list[tuple[str, YumaParams]],
    simulation_hyperparameters: SimulationHyperparameters,
    epochs_window: int,
    epochs_padding: int,
) -> pd.DataFrame:
    """
    Compares the *relative dividends* of a single validator (typically the base_validator)
    across two different MetagraphCase objects: normal vs. shifted,
    for multiple Yuma versions. For each version, the table includes three columns:
      - Normal_<version>
      - Shifted_<version>
      - Comparison_<version>
    
    The "Comparison" column is computed, for each epoch, as the difference between the
    shifted and normal dividends divided by the normalized stake (from case_normal).
    The per-window value is the average of these per-epoch comparisons.
    """
    version_frames = {}

    for (yuma_version_name, yuma_params) in yuma_versions:
        frames = _compute_version_frames(
            yuma_version_name=yuma_version_name,
            yuma_params=yuma_params,
            case_normal=case_normal,
            case_shifted=case_shifted,
            simulation_hyperparameters=simulation_hyperparameters,
            epochs_window=epochs_window,
            epochs_padding=epochs_padding,
        )
        version_frames[yuma_version_name] = frames

    num_epochs = case_normal.num_epochs - epochs_padding
    rows = _build_comparison_rows(version_frames, epochs_window, num_epochs, yuma_versions)

    df = pd.DataFrame(rows)
    column_order = ["Window"]
    for (yuma_version_name, _) in yuma_versions:
        column_order.extend([
            f"Normal_{yuma_version_name}",
            f"Shifted_{yuma_version_name}",
            f"Comparison_{yuma_version_name}"
        ])
    df = df[[c for c in column_order if c in df.columns]]
    return df

def _compute_version_frames(
    yuma_version_name: str,
    yuma_params: YumaParams,
    case_normal: BaseCase,
    case_shifted: BaseCase,
    simulation_hyperparameters: SimulationHyperparameters,
    epochs_window: int,
    epochs_padding: int,
) -> dict:
    """
    For a given Yuma version and its parameters, run the dynamic simulations for both
    the normal and shifted cases; compute the relative dividend series for the base
    validator; apply epoch padding; then calculate per-window (frame) averages and totals.
    Returns a dictionary containing:
      - "normal_frames", "shifted_frames", "comparison_frames" (lists of per-window averages)
      - "total_normal", "total_shifted", "total_comparison" (overall totals)
      - "num_frames": the number of computed frames.
    """
    single_config = YumaConfig(
        simulation=simulation_hyperparameters,
        yuma_params=yuma_params,
    )

    _, relative_dividends_normal, _, _ = _run_dynamic_simulation(
        case=case_normal,
        yuma_version=yuma_version_name,
        yuma_config=single_config,
    )
    _, relative_dividends_shifted, _, _ = _run_dynamic_simulation(
        case=case_shifted,
        yuma_version=yuma_version_name,
        yuma_config=single_config,
    )

    validator_normal = case_normal.base_validator
    validator_shifted = case_shifted.base_validator
    divs_normal = relative_dividends_normal.get(validator_normal, [])
    divs_shifted = relative_dividends_shifted.get(validator_shifted, [])

    stakes_series = case_normal.stakes_dataframe[validator_normal].to_list()

    comparison_series = []
    for i in range(len(stakes_series)):
        stake_val = stakes_series[i]
        if stake_val is None or stake_val == 0:
            comp = 0.0
        else:
            comp = (divs_shifted[i] - divs_normal[i]) / stake_val
        comparison_series.append(comp)

    divs_normal = divs_normal[epochs_padding:]
    divs_shifted = divs_shifted[epochs_padding:]
    comparison_series = comparison_series[epochs_padding:]
    num_epochs = case_normal.num_epochs - epochs_padding

    if epochs_window <= 0:
        raise ValueError(f"epochs_window must be > 0. Got {epochs_window}.")
    if num_epochs < epochs_window:
        logger.warning(
            f"Warning: The total number of epochs ({num_epochs}) is smaller than "
            f"the requested epochs_window ({epochs_window}). You will get only one partial window."
        )

    normal_frames, total_normal_divs = _calculate_total_dividends_with_frames(
        validator_dividends=divs_normal,
        num_epochs=num_epochs,
        epochs_window=epochs_window,
        use_relative=True
    )
    shifted_frames, total_shifted_divs = _calculate_total_dividends_with_frames(
        validator_dividends=divs_shifted,
        num_epochs=num_epochs,
        epochs_window=epochs_window,
        use_relative=True
    )
    comparison_frames, total_comparison_divs = _calculate_total_dividends_with_frames(
        validator_dividends=comparison_series,
        num_epochs=num_epochs,
        epochs_window=epochs_window,
        use_relative=True
    )

    return {
        "normal_frames": normal_frames,
        "shifted_frames": shifted_frames,
        "comparison_frames": comparison_frames,
        "num_frames": len(comparison_frames),
        "total_normal": total_normal_divs,
        "total_shifted": total_shifted_divs,
        "total_comparison": total_comparison_divs,
    }

def _build_comparison_rows(
    version_frames: dict[str, dict],
    epochs_window: int,
    num_epochs: int,
    yuma_versions: list[tuple[str, YumaParams]],
) -> list[dict]:
    """
    Builds a list of row dictionaries (one per window) from the per-version frame data.
    Each row has columns:
      - "Window": a string like "1-20"
      - "Normal_<version>", "Shifted_<version>", "Comparison_<version>" for each yuma_version.
    A final "Total" row is also appended.
    """
    rows = []
    max_frames = max(vdata["num_frames"] for vdata in version_frames.values()) if version_frames else 0

    for i in range(max_frames):
        start_epoch = i * epochs_window + 1
        end_epoch = min((i + 1) * epochs_window, num_epochs)
        row_data = {"Window": f"{start_epoch}-{end_epoch}"}

        for (yuma_version_name, _) in yuma_versions:
            vdata = version_frames[yuma_version_name]
            if i < vdata["num_frames"]:
                avg_normal = vdata["normal_frames"][i]
                avg_shifted = vdata["shifted_frames"][i]
                avg_comparison = vdata["comparison_frames"][i]
            else:
                avg_normal = avg_shifted = avg_comparison = 0.0

            row_data[f"Normal_{yuma_version_name}"] = f"{avg_normal * 100:+.2f}%"
            row_data[f"Shifted_{yuma_version_name}"] = f"{avg_shifted * 100:+.2f}%"
            row_data[f"Comparison_{yuma_version_name}"] = f"{avg_comparison * 100:+.2f}%"

        rows.append(row_data)

    total_row = {"Window": "Total"}
    for (yuma_version_name, _) in yuma_versions:
        vdata = version_frames[yuma_version_name]
        total_row[f"Normal_{yuma_version_name}"] = f"{vdata['total_normal'] * 100:+.2f}%"
        total_row[f"Shifted_{yuma_version_name}"] = f"{vdata['total_shifted'] * 100:+.2f}%"
        total_row[f"Comparison_{yuma_version_name}"] = f"{vdata['total_comparison'] * 100:+.2f}%"
    rows.append(total_row)

    return rows

def _get_case_index_for_row(case_row_ranges: list[tuple[int, int, int]], row_idx: int) -> int:
    """
    Given a list of case row ranges (each a tuple (start, end, case_index))
    and a row index, return the corresponding case index.
    """
    for start, end, c_idx in case_row_ranges:
        if start <= row_idx <= end:
            return c_idx
    return 0


def _build_table_rows(summary_table: pd.DataFrame, case_row_ranges: list[tuple[int, int, int]]) -> str:
    """
    Build the HTML rows for the table by iterating over each row of summary_table.
    Uses _get_case_index_for_row() to set alternating classes.
    """
    html_rows = []
    num_rows = len(summary_table.index)
    for i in range(num_rows):
        case_idx = _get_case_index_for_row(case_row_ranges, i)
        row_class = "case-group-even" if (case_idx % 2 == 0) else "case-group-odd"
        row_html = f"<tr class='{row_class}'>"
        for col in summary_table.columns:
            cell_content = summary_table.at[i, col]
            row_html += f"<td>{cell_content}</td>"
        row_html += "</tr>"
        html_rows.append(row_html)
    return "".join(html_rows)


def _build_html_table(summary_table: pd.DataFrame, case_row_ranges: list[tuple[int, int, int]]) -> str:
    """
    Build the complete HTML table (header and body) given a summary_table and
    case row ranges.
    """
    header_html = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in summary_table.columns) + "</tr></thead>"
    body_html = f"<tbody>{_build_table_rows(summary_table, case_row_ranges)}</tbody>"
    table_html = f"<table>{header_html}{body_html}</table>"
    return table_html

def _map_validator_names(case: "BaseCase", is_metagraph: bool) -> dict[str, str]:
    """
    Returns a mapping from the internal validator names to the display names.
    If is_metagraph is True, the internal names are used as-is; otherwise,
    a scheme such as 'Validator A/B/C' is applied.
    """
    if is_metagraph:
        final_names = case.validators
    else:
        final_names = [f"Validator {chr(ord('A') + i)}" for i in range(len(case.validators))]
    return dict(zip(case.validators, final_names))
    


def _get_final_case_name(case: BaseCase, yuma_version: str, yuma_config: YumaConfig) -> str:
    """
    Returns a formatted case name based on the yuma version and configuration.
    """
    yuma_names = YumaSimulationNames()
    final_yuma_name = ""
    if yuma_version == yuma_names.YUMA2B:
        final_yuma_name = f"{case.name} - {yuma_version} - beta={yuma_config.bond_penalty}"
    elif yuma_version == yuma_names.YUMA3_LIQUID:
        final_yuma_name = f"{case.name} - {yuma_version} - [{yuma_config.alpha_low}, {yuma_config.alpha_high}]"
    else:
        final_yuma_name = f"{case.name} - {yuma_version}"
    
    if case.reset_bonds:
        return final_yuma_name + " + bonds reset"
    return final_yuma_name


def _get_final_case_names_dynamic(
    normal_case: BaseCase, shifted_case: BaseCase, yuma_version: str, yuma_config: YumaConfig
) -> tuple[str, str]:
    """
    Returns a tuple (final_case_name_normal, final_case_name_shifted) for the dynamic case,
    based on the yuma version and configuration.
    """
    yuma_names = YumaSimulationNames()
    if yuma_version == yuma_names.YUMA2B:
        final_case_name_normal = f"{normal_case.name} - beta={yuma_config.bond_penalty}"
        final_case_name_shifted = f"{shifted_case.name} - beta={yuma_config.bond_penalty}"
    elif yuma_version == yuma_names.YUMA3_LIQUID:
        final_case_name_normal = f"{normal_case.name} - {yuma_version} - [{yuma_config.alpha_low}, {yuma_config.alpha_high}]"
        final_case_name_shifted = f"{shifted_case.name} - {yuma_version} - [{yuma_config.alpha_low}, {yuma_config.alpha_high}]"
    else:
        final_case_name_normal = f"{normal_case.name} - {yuma_version}"
        final_case_name_shifted = f"{shifted_case.name} - {yuma_version}"
    return final_case_name_normal, final_case_name_shifted

def _slice_tensors(
    *tensors: torch.Tensor,
    num_validators: int,
    num_servers: int,
) -> tuple[torch.Tensor]:
    """
    Applies a uniform slicing rule to each provided tensor:
    """
    sliced_tensors = []
    for tensor in tensors:
        if tensor.dim() == 1:
            sliced_tensors.append(tensor[-num_servers:])
        elif tensor.dim() == 2:
            sliced_tensors.append(tensor[:num_validators, -num_servers:])
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}. Only 1D or 2D allowed.")
    return tuple(sliced_tensors)

def full_matrices(func):
    def wrapper(
        W: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        alpha_sigmoid_steepness: float,
        alpha_low: float,
        alpha_high: float,
        num_validators: int,
        num_servers: int,
        use_full_matrices: bool,
        ):
        if use_full_matrices:
            W_slice, B_slice, C_slice = _slice_tensors(W, B, C,
                                                       num_validators=num_validators,
                                                       num_servers=num_servers)
        else:
            W_slice, B_slice, C_slice = W, B, C

        alpha_slice = func(W_slice, B_slice, C_slice, alpha_sigmoid_steepness, alpha_low, alpha_high)

        if use_full_matrices:
            alpha_full = torch.full_like(W, fill_value=0.0)
            alpha_full[:num_validators, -num_servers:] = alpha_slice
            return alpha_full
        return alpha_slice
    return wrapper

@full_matrices
def _compute_liquid_alpha(
    W: torch.tensor,
    B: torch.tensor,
    C: torch.tensor,
    alpha_sigmoid_steepness: float,
    alpha_low: float,
    alpha_high: float,
    ):
    """
    Liquid alpha is computed using a combination of previous epoch consensus weights, previous epoch bonds, and current epoch weights.

    Buying Bonds:
    When the current epoch weights exceed the previous epoch bonds, it indicates that the validator intends to purchase bonds.
    The greater the discrepancy between the current weights and the previous epoch consensus weights, the more Liquid Alpha 2.0 will shift toward the alpha low value, facilitating faster bond acquisition.

    Selling Bonds:
    When the current epoch weights are lower than the previous epoch bonds, it signals that the validator aims to sell bonds.
    The larger the difference between the current epoch weights and the previous epoch bonds, the more Liquid Alpha 2.0 will adjust toward the alpha low value, enabling faster bond liquidation.
    """
    buy_mask = (W >= B)
    sell_mask = (W < B)
    
    diff_buy = (W - C).clamp(min=0.0, max=1.0)
    diff_sell = (B - W).clamp(min=0.0, max=1.0)
    
    combined_diff = torch.where(buy_mask, diff_buy, diff_sell)
    
    combined_diff = 1.0 / (1.0 + torch.exp(-alpha_sigmoid_steepness * (combined_diff - 0.5)))
    
    alpha_slice = alpha_low + combined_diff * (alpha_high - alpha_low)
    return alpha_slice.clamp(alpha_low, alpha_high)