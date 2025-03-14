"""
This module provides utilities for calculating and visualizing simulation results.
It includes functions for generating plots, calculating dividends, and preparing data for bonds and incentives.
"""

import base64
import io
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from yuma_simulation._internal.cases import BaseCase

logger = logging.getLogger("main_logger")

def _calculate_total_dividends(
    validators: list[str],
    dividends_per_validator: dict[str, list[float]],
    base_validator: str,
    num_epochs: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Calculates total dividends and percentage differences from a base validator."""

    total_dividends: dict[str, float] = {}
    for validator in validators:
        divs: list[float] = dividends_per_validator.get(validator, [])
        total_dividend = sum(divs[:num_epochs])
        total_dividends[validator] = total_dividend

    base_dividend = total_dividends.get(base_validator, None)
    if base_dividend is None or base_dividend == 0.0:
        logger.warning(
            f"Warning: Base validator '{base_validator}' has zero or missing total dividends."
        )
        base_dividend = 1e-6

    percentage_diff_vs_base: dict[str, float] = {}
    for validator, total_dividend in total_dividends.items():
        if validator == base_validator:
            percentage_diff_vs_base[validator] = 0.0
        else:
            percentage_diff = ((total_dividend - base_dividend) / base_dividend) * 100.0
            percentage_diff_vs_base[validator] = percentage_diff

    return total_dividends, percentage_diff_vs_base


def _calculate_total_dividends_with_frames(
    validator_dividends: list[float],
    num_epochs: int,
    epochs_window: int,
    use_relative: bool = False
) -> tuple[list[float], float]:
    """
    Returns a tuple of:
      1) A list of "frame" values over consecutive windows of length `epochs_window`.
      2) The overall "total" (sum for absolute, or average for relative).

    If `use_relative=False` (default), each frame is summed:
      e.g. [sum of epochs 0..9, sum of epochs 10..19, ...]

    If `use_relative=True`, each frame is an average:
      e.g. [avg of epochs 0..9, avg of epochs 10..19, ...]
      And the overall total is the avg of all truncated_divs.

    For example:
      If num_epochs=40, epochs_window=10 => 4 frames (each covering 10 epochs).
      With `use_relative=False`, we sum each window.
      With `use_relative=True`, we average each window.
    """

    # Truncate any extra dividends if validator_dividends is longer than num_epochs
    truncated_divs = validator_dividends[:num_epochs]

    frames_values = []
    for start_idx in range(0, num_epochs, epochs_window):
        end_idx = min(start_idx + epochs_window, num_epochs)
        chunk = truncated_divs[start_idx:end_idx]

        if use_relative:
            val = sum(chunk) / len(chunk)
        else:
            val = sum(chunk)

        frames_values.append(val)

    if use_relative:
        total_value = sum(truncated_divs) / len(truncated_divs)
    else:
        total_value = sum(truncated_divs)

    return frames_values, total_value


def _plot_dividends(
    num_epochs: int,
    validators: list[str],
    dividends_per_validator: dict[str, list[float]],
    case_name: str,
    case: BaseCase,
    to_base64: bool = False,
) -> str | None:
    """
    Generates a plot of dividends over epochs for a set of validators.
    """

    plt.close("all")
    _, ax_main = plt.subplots(figsize=(14, 6))

    top_vals = getattr(case, "top_validators_hotkeys", [])
    if top_vals:
        plot_validator_names = top_vals.copy()
    else:
        plot_validator_names = validators.copy()

    if case.base_validator not in plot_validator_names:
        plot_validator_names.append(case.base_validator)

    validator_styles = _get_validator_styles(validators)

    total_dividends, percentage_diff_vs_base = _calculate_total_dividends(
        validators,
        dividends_per_validator,
        case.base_validator,
        num_epochs,
    )

    num_epochs_calculated: int | None = None
    x: np.ndarray = np.array([])

    for idx, validator in enumerate(plot_validator_names):
        if validator not in dividends_per_validator:
            continue

        dividends = dividends_per_validator[validator]
        dividends_array = np.array(dividends, dtype=float)

        if num_epochs_calculated is None:
            num_epochs_calculated = len(dividends_array)
            x = np.arange(num_epochs_calculated)

        delta = 0.05
        x_shifted = x + idx * delta

        linestyle, marker, markersize, markeredgewidth = validator_styles.get(
            validator, ("-", "o", 5, 1)
        )

        total_dividend = total_dividends.get(validator, 0.0)
        percentage_diff = percentage_diff_vs_base.get(validator, 0.0)

        if abs(total_dividend) < 1e-6:
            total_dividend_str = f"{total_dividend:.3e}"  # Scientific notation
        else:
            total_dividend_str = f"{total_dividend:.6f}"  # Standard float

        if abs(percentage_diff) < 1e-12:
            percentage_str = "(Base)"
        elif percentage_diff > 0:
            percentage_str = f"(+{percentage_diff:.1f}%)"
        else:
            percentage_str = f"({percentage_diff:.1f}%)"

        label = f"{validator}: Total={total_dividend_str} {percentage_str}"

        ax_main.plot(
            x_shifted,
            dividends_array[:num_epochs],  # restrict to requested epochs
            marker=marker,
            markeredgewidth=markeredgewidth,
            markersize=markersize,
            label=label,
            alpha=0.7,
            linestyle=linestyle,
        )

    if num_epochs_calculated is not None:
        _set_default_xticks(ax_main, num_epochs_calculated)

    ax_main.set_xlabel("Time (Epochs)")
    ax_main.set_ylim(bottom=0)
    ax_main.set_ylabel("Dividend per 1,000 Tao per Epoch")
    ax_main.set_title(case_name)
    ax_main.grid(True)
    ax_main.legend()

    from matplotlib.ticker import ScalarFormatter
    ax_main.get_yaxis().set_major_formatter(ScalarFormatter(useMathText=True))
    ax_main.ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    if case_name.startswith("Case 4"):
        ax_main.set_ylim(0, 0.042)

    plt.subplots_adjust(hspace=0.3)

    if to_base64:
        return _plot_to_base64()
    else:
        plt.show()
        return None


def _plot_relative_dividends(
    validators_relative_dividends: dict[str, list[float]],
    case_name: str,
    case: BaseCase,
    num_epochs: int,
    epochs_padding: int = 0,
    to_base64: bool = False
) -> str | None:
    """
    Plots relative dividend values (which may be > 0 or < 0) for a set of validators over epochs,
    using percentages on the Y-axis. A horizontal line at y=0% indicates the 'neutral' line.
    
    If a validator is missing (e.g. deregistered) in a particular epoch, its value for that epoch 
    is set to NaN so that the plotted line is broken. When the validator is present again, the line resumes.
    
    The first `epochs_padding` records are omitted from the plot.
    """
    plt.close("all")
    # Adjust the number of epochs to be plotted.
    plot_epochs = num_epochs - epochs_padding
    if plot_epochs <= 0:
        logger.warning("Epochs padding is too large relative to num_epochs. Nothing to plot.")
        return None

    _, ax = plt.subplots(figsize=(14 * 2, 6 * 2))

    if not validators_relative_dividends:
        logger.warning("No validator data to plot.")
        return None

    all_validators = list(validators_relative_dividends.keys())

    # Use top validators (and base validator) if available.
    top_vals = getattr(case, "top_validators_hotkeys", [])
    if top_vals:
        plot_validator_names = top_vals.copy()
    else:
        plot_validator_names = all_validators.copy()
        
    if case.base_validator not in plot_validator_names:
        plot_validator_names.append(case.base_validator)

    if not plot_validator_names:
        logger.warning("No matching validators to plot.")
        return None

    # Use the adjusted number of epochs.
    x = np.arange(plot_epochs)
    validator_styles = _get_validator_styles(all_validators)

    for idx, validator in enumerate(plot_validator_names):
        # Retrieve dividend series for this validator.
        dividends = validators_relative_dividends.get(validator, [])
        # Ensure there are enough data points to slice.
        if len(dividends) <= epochs_padding:
            continue

        # Slice off the first epochs_padding records.
        dividends = dividends[epochs_padding:]

        # Replace missing values (None) with np.nan.
        dividends = np.array(
            [d if d is not None else np.nan for d in dividends],
            dtype=float,
        )

        delta = 0.05
        x_shifted = x + idx * delta

        total_mean = _compute_mean(dividends)
        sign_str = f"{total_mean * 100:+.5f}%"
        label = f"{validator}: total = {sign_str}"

        linestyle, marker, markersize, markeredgewidth = validator_styles.get(
            validator, ("-", "o", 5, 1)
        )

        ax.plot(
            x_shifted,
            dividends,
            label=label,
            alpha=0.7,
            marker=marker,
            markeredgewidth=markeredgewidth,
            markersize=markersize,
            linestyle=linestyle,
        )

    ax.axhline(y=0, color="black", linewidth=1, linestyle="--", alpha=0.7)

    _set_default_xticks(ax, plot_epochs)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative Dividend (%)")
    ax.set_title(case_name)
    ax.grid(True)

    legend = ax.legend()
    for text in legend.get_texts():
        if text.get_text().startswith(case.shift_validator_hotkey):
            text.set_fontweight('bold')

    def to_percent(y, _):
        return f"{y * 100:.1f}%"
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

    plt.subplots_adjust(hspace=0.3)

    if to_base64:
        return _plot_to_base64()
    else: 
        plt.show()
        return None


def _plot_relative_dividends_comparisson(
    validators_relative_dividends_normal: dict[str, list[float]],
    validators_relative_dividends_shifted: dict[str, list[float]],
    case: BaseCase,
    num_epochs: int,
    epochs_padding: int = 0,
    to_base64: bool = False,
    use_stakes: bool = False
) -> str | None:
    """
    Plots a comparison of dividends for each validator.
    The element-wise differences are plotted (shifted - normal), and the legend shows
    the difference in the means (displayed as a percentage).

    If 'use_stakes' is True and the case provides a stakes_dataframe property,
    then each difference is divided by the normalized stake for that validator at that epoch.
    The mean is recomputed from the newly calculated differences.

    The first `epochs_padding` records are omitted from the plot.
    """
    plt.close("all")
    # Adjust the number of epochs to be plotted.
    plot_epochs = num_epochs - epochs_padding
    if plot_epochs <= 0:
        logger.warning("Epochs padding is too large relative to number of total epochs. Nothing to plot.")
        return None

    _, ax = plt.subplots(figsize=(14 * 2, 6 * 2))

    if not validators_relative_dividends_normal or not validators_relative_dividends_shifted:
        logger.warning("No validator data to plot.")
        return None

    all_validators = list(validators_relative_dividends_normal.keys())

    # Use the case's top validators if available; otherwise, plot all.
    top_vals = getattr(case, "top_validators_hotkeys", [])
    if top_vals:
        plot_validator_names = top_vals.copy()
    else:
        plot_validator_names = all_validators.copy()

    # Ensure that the base (shifted) validator is included.
    base_validator = getattr(case, "base_validator", None)
    if base_validator and base_validator not in plot_validator_names:
        plot_validator_names.append(base_validator)

    x = np.arange(plot_epochs)
    validator_styles = _get_validator_styles(all_validators)

    # Retrieve stakes DataFrame if stakes normalization is requested.
    if use_stakes and hasattr(case, "stakes_dataframe"):
        df_stakes = case.stakes_dataframe
    else:
        df_stakes = None

    for idx, validator in enumerate(plot_validator_names):
        # Retrieve dividend series from both dictionaries.
        normal_dividends = validators_relative_dividends_normal.get(validator, [])
        shifted_dividends = validators_relative_dividends_shifted.get(validator, [])

        # Skip plotting if one of the series is missing or not long enough.
        if not normal_dividends or not shifted_dividends:
            continue
        if len(normal_dividends) <= epochs_padding or len(shifted_dividends) <= epochs_padding:
            continue

        # Slice off the first epochs_padding records.
        normal_dividends = normal_dividends[epochs_padding:]
        shifted_dividends = shifted_dividends[epochs_padding:]

        # Replace missing values (None) with np.nan.
        normal_dividends = np.array(
            [d if d is not None else np.nan for d in normal_dividends],
            dtype=float,
        )
        shifted_dividends = np.array(
            [d if d is not None else np.nan for d in shifted_dividends],
            dtype=float,
        )

        relative_diff = shifted_dividends - normal_dividends

        if df_stakes is not None and validator in df_stakes.columns:
            stakes_series = df_stakes[validator].to_numpy()
            # Ensure stakes series is sliced to match the dividends.
            if len(stakes_series) > epochs_padding:
                stakes_series = stakes_series[epochs_padding:]
            else:
                stakes_series = np.full_like(relative_diff, np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                relative_diff = np.where(stakes_series != 0, relative_diff / stakes_series, np.nan)
            mean_difference = _compute_mean(relative_diff) * 100
        else:
            normal_mean = _compute_mean(normal_dividends)
            shifted_mean = _compute_mean(shifted_dividends)
            mean_difference = (shifted_mean - normal_mean) * 100

        delta = 0.05
        x_shifted = x + idx * delta

        sign_str = f"{mean_difference:+.5f}%"
        label = f"{validator}: mean difference/stake = {sign_str}" if use_stakes else f"{validator}: mean difference = {sign_str}"

        linestyle, marker, markersize, markeredgewidth = validator_styles.get(
            validator, ("-", "o", 5, 1)
        )

        ax.plot(
            x_shifted,
            relative_diff,
            label=label,
            alpha=0.7,
            marker=marker,
            markeredgewidth=markeredgewidth,
            markersize=markersize,
            linestyle=linestyle,
        )

    ax.axhline(y=0, color="black", linewidth=1, linestyle="--", alpha=0.7)

    _set_default_xticks(ax, plot_epochs)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Absolute Difference")
    ax.set_title("Comparison (shifted - normal) scaled by stake" if use_stakes else "Comparison (shifted - normal)")
    ax.grid(True)

    legend = ax.legend()
    for text in legend.get_texts():
        if text.get_text().startswith(case.shift_validator_hotkey):
            text.set_fontweight('bold')

    def to_percent(y, _):
        return f"{y * 100:.1f}%"
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

    plt.subplots_adjust(hspace=0.3)

    if to_base64:
        return _plot_to_base64()
    else:
        plt.show()
        return None


def _plot_bonds(
    num_epochs: int,
    validators: list[str],
    servers: list[str],
    bonds_per_epoch: list[torch.Tensor],
    case_name: str,
    to_base64: bool = False,
    normalize: bool = False,
) -> str | None:
    """Generates a plot of bonds per server for each validator."""

    x = list(range(num_epochs))

    fig, axes = plt.subplots(1, len(servers), figsize=(14, 5), sharex=True, sharey=True)
    if len(servers) == 1:
        axes = [axes]  # type: ignore

    bonds_data = _prepare_bond_data(
        bonds_per_epoch, validators, servers, normalize=normalize
    )
    validator_styles = _get_validator_styles(validators)

    handles: list[plt.Artist] = []
    labels: list[str] = []
    for idx_s, server in enumerate(servers):
        ax = axes[idx_s]
        for idx_v, validator in enumerate(validators):
            bonds = bonds_data[idx_s][idx_v]
            linestyle, marker, markersize, markeredgewidth = validator_styles[validator]

            (line,) = ax.plot(
                x,
                bonds,
                alpha=0.7,
                marker=marker,
                markersize=markersize,
                markeredgewidth=markeredgewidth,
                linestyle=linestyle,
                linewidth=2,
            )
            if idx_s == 0:
                handles.append(line)
                labels.append(validator)

        _set_default_xticks(ax, num_epochs)

        ylabel = "Bond Ratio" if normalize else "Bond Value"
        ax.set_xlabel("Epoch")
        if idx_s == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(server)
        ax.grid(True)

        if normalize:
            ax.set_ylim(0, 1.05)

    fig.suptitle(
        f"Validators bonds per Server{' normalized' if normalize else ''}\n{case_name}",
        fontsize=14,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(validators),
        bbox_to_anchor=(0.5, 0.02),
    )
    plt.tight_layout(rect=(0, 0.05, 0.98, 0.95))

    if to_base64:
        return _plot_to_base64()

    plt.show()
    return None


def _plot_validator_server_weights(
    validators: list[str],
    weights_epochs: list[torch.Tensor],
    servers: list[str],
    num_epochs: int,
    case_name: str,
    to_base64: bool = False,
) -> str | None:
    """Plots validator weights across servers over epochs."""
    from .simulation_utils import _slice_tensors
    weights_epochs = _slice_tensors(*weights_epochs, num_validators=len(validators), num_servers=len(servers))

    validator_styles = _get_validator_styles(validators)

    y_values_all: list[float] = [
        float(weights_epochs[epoch][idx_v][1].item())
        for idx_v in range(len(validators))
        for epoch in range(num_epochs)
    ]
    unique_y_values = sorted(set(y_values_all))
    min_label_distance = 0.05
    close_to_server_threshold = 0.02

    def is_round_number(y: float) -> bool:
        return abs((y * 20) - round(y * 20)) < 1e-6

    y_tick_positions: list[float] = [0.0, 1.0]
    y_tick_labels: list[str] = [servers[0], servers[1]]

    for y in unique_y_values:
        if y in [0.0, 1.0]:
            continue
        if (
            abs(y - 0.0) < close_to_server_threshold
            or abs(y - 1.0) < close_to_server_threshold
        ):
            continue
        if is_round_number(y):
            if all(
                abs(y - existing_y) >= min_label_distance
                for existing_y in y_tick_positions
            ):
                y_tick_positions.append(y)
                y_percentage = y * 100
                label = (
                    f"{y_percentage:.0f}%"
                    if float(y_percentage).is_integer()
                    else f"{y_percentage:.1f}%"
                )
                y_tick_labels.append(label)
        else:
            if all(
                abs(y - existing_y) >= min_label_distance
                for existing_y in y_tick_positions
            ):
                y_tick_positions.append(y)
                y_percentage = y * 100
                label = (
                    f"{y_percentage:.0f}%"
                    if float(y_percentage).is_integer()
                    else f"{y_percentage:.1f}%"
                )
                y_tick_labels.append(label)

    ticks = list(zip(y_tick_positions, y_tick_labels))
    ticks.sort(key=lambda x: x[0])
    y_tick_positions, y_tick_labels = map(list, zip(*ticks))

    fig_height = 1 if len(y_tick_positions) <= 2 else 3
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.set_ylim(-0.05, 1.05)

    for idx_v, validator in enumerate(validators):
        y_values = [
            float(weights_epochs[epoch][idx_v][1].item()) for epoch in range(num_epochs)
        ]
        linestyle, marker, markersize, markeredgewidth = validator_styles[validator]

        ax.plot(
            range(num_epochs),
            y_values,
            label=validator,
            marker=marker,
            linestyle=linestyle,
            markersize=markersize,
            markeredgewidth=markeredgewidth,
            linewidth=2,
        )

    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)

    _set_default_xticks(ax, num_epochs)

    ax.set_xlabel("Epoch")
    ax.set_title(f"Validators Weights to Servers \n{case_name}")
    ax.legend()
    ax.grid(True)

    if to_base64:
        return _plot_to_base64()
    plt.show()
    return None


def _plot_validator_server_weights_subplots(
    validators: list[str],
    weights_epochs: list[torch.Tensor],
    servers: list[str],
    num_epochs: int,
    case_name: str,
    to_base64: bool = False,
) -> str | None:
    """
    Plots validator weights in subplots (one subplot per server) over epochs.
    Each subplot shows lines for all validators, representing how much weight
    they allocate to that server from epoch 0..num_epochs-1.
    """
    from .simulation_utils import _slice_tensors
    weights_epochs = _slice_tensors(
        *weights_epochs, 
        num_validators=len(validators), 
        num_servers=len(servers)
    )

    x = list(range(num_epochs))

    fig, axes = plt.subplots(
        1, 
        len(servers), 
        figsize=(14, 5), 
        sharex=True, 
        sharey=True
    )

    if len(servers) == 1:
        axes = [axes]

    validator_styles = _get_validator_styles(validators)

    handles: list[plt.Artist] = []
    labels: list[str] = []

    for idx_s, server_name in enumerate(servers):
        ax = axes[idx_s]
        for idx_v, validator in enumerate(validators):
            y_values = [
                float(weights_epochs[epoch][idx_v][idx_s].item())
                for epoch in range(num_epochs)
            ]
            linestyle, marker, markersize, markeredgewidth = validator_styles[validator]

            (line,) = ax.plot(
                x,
                y_values,
                alpha=0.7,
                marker=marker,
                markersize=markersize,
                markeredgewidth=markeredgewidth,
                linestyle=linestyle,
                linewidth=2,
                label=validator,
            )

            if idx_s == 0:
                handles.append(line)
                labels.append(validator)

        _set_default_xticks(ax, num_epochs)

        ax.set_xlabel("Epoch")
        if idx_s == 0:
            ax.set_ylabel("Validator Weight")
        ax.set_title(server_name)
        ax.set_ylim(0, 1.05)
        ax.grid(True)

    fig.suptitle(f"Validators' Weights per Server\n{case_name}", fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(validators),
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout(rect=(0, 0.07, 1, 0.95))

    if to_base64:
        return _plot_to_base64()

    plt.show()
    return None


def _plot_incentives(
    servers: list[str],
    server_incentives_per_epoch: list[torch.Tensor],
    num_epochs: int,
    case_name: str,
    to_base64: bool = False,
) -> str | None:
    """Generates a plot of server incentives over epochs."""

    x = np.arange(num_epochs)
    _, ax = plt.subplots(figsize=(14, 3))

    for idx_s, server in enumerate(servers):
        incentives: list[float] = [
            float(server_incentive[idx_s].item())
            for server_incentive in server_incentives_per_epoch
        ]
        ax.plot(x, incentives, label=server)

    _set_default_xticks(ax, num_epochs)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Server Incentive")
    ax.set_title(f"Server Incentives\n{case_name}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True)

    if to_base64:
        return _plot_to_base64()
    plt.show()
    return None


def _plot_to_base64() -> str:
    """Converts a Matplotlib plot to a Base64-encoded string."""

    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    plt.close()
    return f'<img src="data:image/png;base64,{encoded_image}" style="max-width:1200px; height:auto;" draggable="false">'


def _set_default_xticks(ax: Axes, num_epochs: int) -> None:
    tick_locs = [0, 1, 2] + list(range(5, num_epochs, 5))
    tick_labels = [str(i) for i in tick_locs]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels, fontsize=8)


def _prepare_bond_data(
    bonds_per_epoch: list[torch.Tensor],
    validators: list[str],
    servers: list[str],
    normalize: bool,
) -> list[list[list[float]]]:
    """Prepares bond data for plotting, normalizing if specified."""

    num_epochs = len(bonds_per_epoch)
    bonds_data: list[list[list[float]]] = []
    for idx_s, _ in enumerate(servers):
        server_bonds: list[list[float]] = []
        for idx_v, _ in enumerate(validators):
            validator_bonds = [
                float(bonds_per_epoch[epoch][idx_v, idx_s].item())
                for epoch in range(num_epochs)
            ]
            server_bonds.append(validator_bonds)
        bonds_data.append(server_bonds)

    if normalize:
        for idx_s in range(len(servers)):
            for epoch in range(num_epochs):
                epoch_bonds = bonds_data[idx_s]
                values = [epoch_bonds[idx_v][epoch] for idx_v in range(len(validators))]
                total = sum(values)
                if total > 1e-12:
                    for idx_v in range(len(validators)):
                        epoch_bonds[idx_v][epoch] /= total

    return bonds_data


def _get_validator_styles(
    validators: list[str],
) -> dict[str, tuple[str, str, int, int]]:
    combined_styles = [("-", "+", 12, 2), ("--", "x", 12, 1), (":", "o", 4, 1)]
    return {
        validator: combined_styles[idx % len(combined_styles)]
        for idx, validator in enumerate(validators)
    }


def _compute_mean(dividends: np.ndarray) -> float:
    """Computes the mean over valid epochs where the validator is present."""
    if np.all(np.isnan(dividends)):
        return 0.0
    return np.nanmean(dividends)



def _generate_chart_for_type(
    chart_type: str,
    case: BaseCase,
    final_case_name: str,
    simulation_results: tuple | None = None,
    to_base64: bool = True,
    epochs_padding: int = 0
) -> str:
    """
    Dispatches to the correct plotting function based on the chart type.
    For types that need simulation results, the tuple is unpacked as needed.
    """
    if chart_type == "weights":
        return _plot_validator_server_weights(
            validators=case.validators,
            weights_epochs=case.weights_epochs,
            servers=case.servers,
            num_epochs=case.num_epochs,
            case_name=final_case_name,
            to_base64=to_base64,
        )
    elif chart_type == "weights_subplots":
        return  _plot_validator_server_weights_subplots(
            validators=case.validators,
            weights_epochs=case.weights_epochs,
            servers=case.servers,
            num_epochs=case.num_epochs,
            case_name=final_case_name,
            to_base64=to_base64,
        )
    elif chart_type == "dividends":
        dividends_per_validator, *_ = simulation_results
        return _plot_dividends(
            num_epochs=case.num_epochs,
            validators=case.validators,
            dividends_per_validator=dividends_per_validator,
            case_name=final_case_name,
            case=case,
            to_base64=to_base64,
        )
    elif chart_type == "relative_dividends":
        _, validators_relative_dividends, *_ = simulation_results
        return _plot_relative_dividends(
            validators_relative_dividends=validators_relative_dividends,
            case_name=final_case_name,
            case=case,
            num_epochs=case.num_epochs,
            epochs_padding=epochs_padding,
            to_base64=to_base64,
        )
    elif chart_type == "bonds":
        _, _, bonds_per_epoch, *_ = simulation_results
        return _plot_bonds(
            num_epochs=case.num_epochs,
            validators=case.validators,
            servers=case.servers,
            bonds_per_epoch=bonds_per_epoch,
            case_name=final_case_name,
            to_base64=to_base64,
        )
    elif chart_type == "normalized_bonds":
        _, _, bonds_per_epoch, *_ = simulation_results
        return _plot_bonds(
            num_epochs=case.num_epochs,
            validators=case.validators,
            servers=case.servers,
            bonds_per_epoch=bonds_per_epoch,
            case_name=final_case_name,
            to_base64=to_base64,
            normalize=True,
        )
    elif chart_type == "incentives":
        *_, server_incentives_per_epoch = simulation_results
        return _plot_incentives(
            servers=case.servers,
            server_incentives_per_epoch=server_incentives_per_epoch,
            num_epochs=case.num_epochs,
            case_name=final_case_name,
            to_base64=to_base64,
        )
    else:
        raise ValueError("Invalid chart type.")
