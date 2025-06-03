"""
This module provides utilities for calculating and visualizing simulation results.
It includes functions for generating plots, calculating dividends, and preparing data for bonds and incentives.
"""

import base64
import io
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import textwrap
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from yuma_simulation._internal.cases import BaseCase, MetagraphCase
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter

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
    fig, ax_main = plt.subplots(figsize=(14, 6))

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
        return _plot_to_base64(fig)
    else:
        plt.show()
        plt.close(fig)
        return None


def _plot_relative_dividends(
    validators_relative_dividends: dict[str, list[float]],
    case_name: str,
    case: BaseCase,
    num_epochs: int,
    epochs_padding: int = 0,
    to_base64: bool = False
) -> str | None:
    plt.close("all")
    plot_epochs = num_epochs - epochs_padding
    if plot_epochs <= 0 or not validators_relative_dividends:
        logger.warning("Nothing to plot (padding/empty data).")
        return None

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[1, 4], figure=fig)

    ax_text = fig.add_subplot(gs[0])
    ax_text.axis("off")

    para = (
        "“Validator Relative Dividends” is a performance metric that measures the deviation "
        "between a validator’s actual dividends and the hypothetical zero-sum dividends they "
        "would have earned purely in proportion to their stake. This difference highlights "
        "whether a validator has over- or under-performed relative to the stake-weighted "
        "zero-sum baseline across the entire network."
    )
    wrapped_para = textwrap.fill(para, width=160)

    eq = (
        r"$\dfrac{\text{Validator’s Dividends}}{\sum_{\text{all}}\text{Dividends}}"
        r" \;-\; "
        r"\dfrac{\text{Validator’s Stake}}{\sum_{\text{all}}\text{Stake}}$"
    )

    full_text = wrapped_para + "\n\n" + r"$\text{Relative Dividend} =$ " + eq

    ax_text.text(
        0.5, 0.5, full_text,
        ha="center", va="center",
        fontsize=12,
        wrap=False
    )

    ax = fig.add_subplot(gs[1])
    all_validators = list(validators_relative_dividends.keys())
    top_vals = getattr(case, "top_validators_hotkeys", []) or all_validators.copy()
    if case.base_validator not in top_vals:
        top_vals.append(case.base_validator)
    x = np.arange(plot_epochs)
    validator_styles = _get_validator_styles(all_validators)

    for idx, validator in enumerate(top_vals):
        series = validators_relative_dividends.get(validator, [])
        if len(series) <= epochs_padding:
            continue
        arr = np.array([d if d is not None else np.nan for d in series[epochs_padding:]], dtype=float)
        x_shifted = x + idx * 0.05
        mean_pct = _compute_mean(arr) * 100
        label = f"{case.hotkey_label_map.get(validator, validator)}: total = {mean_pct:+.5f}%"
        ls, mk, ms, mew = validator_styles.get(validator, ("-", "o", 5, 1))
        ax.plot(x_shifted, arr, label=label, alpha=0.7,
                linestyle=ls, marker=mk, markersize=ms, markeredgewidth=mew)

    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.7)
    _set_default_xticks(ax, plot_epochs)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Relative Dividend (%)", fontsize=12)
    legend = ax.legend()
    for text in legend.get_texts():
        if text.get_text().startswith(case.shift_validator_hotkey):
            text.set_fontweight("bold")
    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.1f}%"))
    fig.suptitle(f"Validators Relative Dividends\n{case_name}", fontsize=16)

    if to_base64:
        return _plot_to_base64(fig)
    else:
        plt.show()
        plt.close(fig)
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

    fig, ax = plt.subplots(figsize=(14 * 2, 6 * 2))

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
        return _plot_to_base64(fig)
    else:
        plt.show()
        plt.close(fig)
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
        return _plot_to_base64(fig)

    plt.show()
    plt.close(fig)
    return None

def _plot_bonds_metagraph_dynamic(
    case: MetagraphCase,
    bonds_per_epoch:    list[torch.Tensor],
    case_name:          str,
    to_base64:          bool = False,
    normalize:          bool = False,
    legend_validators:  list[str] | None = None,
    epochs_padding:     int = 0,
) -> str | None:

    num_epochs  = case.num_epochs
    plot_epochs = num_epochs - epochs_padding
    if plot_epochs <= 0:
        logger.warning("Nothing to plot (padding >= total_epochs).")
        return None

    validators_epochs = case.validators_epochs
    miners_epochs     = case.servers
    selected_validators = case.top_validators_hotkeys

    bonds_data = _prepare_bond_data_dynamic(
        bonds_per_epoch, validators_epochs, miners_epochs,
        normalize=normalize,
    )

    subset_v = selected_validators or validators_epochs[0]

    subset_m = (case.selected_servers or miners_epochs[0])[:10]

    miner_keys      = miners_epochs[0]
    validator_keys: list[str] = []
    for epoch in validators_epochs:
        for v in epoch:
            if v not in validator_keys:
                validator_keys.append(v)

    m_idx = [miner_keys.index(m)     for m in subset_m]
    v_idx = [validator_keys.index(v) for v in subset_v]

    plot_data: list[list[list[float]]] = []
    for mi in m_idx:
        per_val = []
        for vi in v_idx:
            series = []
            for e in range(epochs_padding, num_epochs):
                series.append(bonds_data[mi][vi][e])
            per_val.append(series)
        plot_data.append(per_val)

    CHART_WIDTH   = 7.0  
    CHART_HEIGHT  = 5.0 
    TEXT_BLOCK_H  = 2.0  
    COLS          = 2    

    num_charts = len(subset_m)
    rows       = math.ceil(num_charts / COLS)  

    fig_w = CHART_WIDTH * COLS          
    fig_h = TEXT_BLOCK_H + (CHART_HEIGHT * rows)

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    chart_block_h = CHART_HEIGHT * rows

    outer_gs = GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[TEXT_BLOCK_H, chart_block_h],
        hspace=0.0,
        figure=fig
    )

    top_gs = GridSpecFromSubplotSpec(
        nrows=3,
        ncols=1,
        subplot_spec=outer_gs[0],
        height_ratios=[0.5, 1.0, 0.5],
        hspace=0.0
    )

    ax_title = fig.add_subplot(top_gs[0])
    ax_title.axis("off")
    title_norm = " normalized" if normalize else ""
    title_str  = f"Validators bonds per Miner{title_norm}\n{case_name}"
    ax_title.text(
        0.5, 0.5,
        title_str,
        ha="center", va="center",
        fontsize=14
    )

    ax_para = fig.add_subplot(top_gs[1])
    ax_para.axis("off")

    if normalize:
        para = (
            "This plot shows each miner’s *normalized* bond ratio from each validator over time.  "
            "At every epoch, each miner’s incoming bonds have been scaled so that their total across "
            "all validators equals 1.\n\n"
        )
        ylabel = "Bond Ratio"
    else:
        para = (
            "This plot shows each validator’s *absolute* bond value to each miner over time.  "
            "At every epoch, the raw bond tensor is used, in the native units of a given simulation version.\n\n"
        )
        ylabel = "Bond Value"

    wrapped = textwrap.fill(para, width=140)
    ax_para.text(
        0.5, 0.5,
        wrapped,
        ha="center", va="center",
        fontsize=11, wrap=False
    )

    ax_legend = fig.add_subplot(top_gs[2])
    ax_legend.axis("off")

    inner_gs = GridSpecFromSubplotSpec(
        nrows=rows,
        ncols=COLS,
        subplot_spec=outer_gs[1],
        wspace=0.3,
        hspace=0.4
    )

    x = list(range(plot_epochs))
    styles = _get_validator_styles(validator_keys)
    handles = []
    labels  = []

    ticks = list(range(0, plot_epochs, 5))
    if (plot_epochs - 1) not in ticks:
        ticks.append(plot_epochs - 1)
    tick_labels = [str(t) for t in ticks]

    for i_miner, miner in enumerate(subset_m):
        r, c = divmod(i_miner, COLS)
        ax = fig.add_subplot(inner_gs[r, c])

        for j, val in enumerate(subset_v):
            ls, mk, ms, mew = styles[val]
            line, = ax.plot(
                x, plot_data[i_miner][j],
                linestyle=ls,
                marker=mk,
                markersize=ms,
                markeredgewidth=mew,
                linewidth=2,
                alpha=0.7
            )
            if i_miner == 0 and (legend_validators or subset_v) and (val in (legend_validators or subset_v)):
                handles.append(line)
                labels.append(case.hotkey_label_map.get(val, val))

        ax.set_title(miner, fontsize=10)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Epoch")
        if r == 0 and c == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True)
        if normalize:
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)

    total_slots = rows * COLS
    for idx in range(num_charts, total_slots):
        r, c = divmod(idx, COLS)
        ax_empty = fig.add_subplot(inner_gs[r, c])
        ax_empty.set_visible(False)

    ncol = min(len(labels), 4)
    ax_legend.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize="small",
        handletextpad=0.3,
        columnspacing=0.5
    )

    fig.tight_layout(w_pad=0.3, h_pad=0.4)

    if to_base64:
        return _plot_to_base64(fig)

    plt.show()
    plt.close(fig)
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
        return _plot_to_base64(fig)
    plt.show()
    plt.close(fig)
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

    fig.suptitle(f"Validators Weights per Server\n{case_name}", fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(validators),
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout(rect=(0, 0.07, 1, 0.95))

    if to_base64:
        return _plot_to_base64(fig)

    plt.show()
    plt.close(fig)
    return None



def _plot_validator_server_weights_subplots_dynamic(
    case: MetagraphCase,
    case_name: str,
    epochs_padding: int = 0,
    to_base64: bool = False,
) -> str | None:
    """
    Dynamic version for metagraph-based weights, skipping the first `epochs_padding` epochs
"""
    total_epochs = case.num_epochs
    plot_epochs  = total_epochs - epochs_padding
    if plot_epochs <= 0:
        print("Nothing to plot (padding >= total_epochs).")
        return None

    subset_vals = case.top_validators_hotkeys or case.validators_epochs[0]
    subset_srvs = case.selected_servers   or case.servers[0][:10]
    hotkey_map  = case.hotkey_label_map

    data_cube: list[list[list[float]]] = []
    for srv in subset_srvs:
        per_val = []
        for val in subset_vals:
            series = []
            for e in range(epochs_padding, total_epochs):
                ve, se, W = case.validators_epochs[e], case.servers[e], case.weights_epochs[e]
                if (val in ve) and (srv in se):
                    r, c = ve.index(val), se.index(srv)
                    series.append(float(W[r, c].item()))
                else:
                    series.append(np.nan)
            per_val.append(series)
        data_cube.append(per_val)

    CHART_WIDTH   = 7  
    CHART_HEIGHT  = 5   
    TEXT_BLOCK_H  = 2    
    COLS          = 2   

    num_charts = len(subset_srvs)
    rows       = math.ceil(num_charts / COLS)

    fig_w = CHART_WIDTH * COLS
    fig_h = TEXT_BLOCK_H + (CHART_HEIGHT * rows)

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)

    chart_block_h = CHART_HEIGHT * rows

    outer_gs = GridSpec(
        nrows=2,
        ncols=1,
        height_ratios=[TEXT_BLOCK_H, chart_block_h],
        hspace=0.0,
        figure=fig
    )

    top_gs = GridSpecFromSubplotSpec(
        nrows=3,
        ncols=1,
        subplot_spec=outer_gs[0],
        height_ratios=[0.5, 1.0, 0.5],
        hspace=0.0
    )

    ax_title = fig.add_subplot(top_gs[0])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.5,
        "Validators Weights per Miner",
        ha="center", va="center",
        fontsize=14
    )

    ax_para = fig.add_subplot(top_gs[1])
    ax_para.axis("off")
    paragraph = (
        "“Validators Weights per Miner” is a visualization that shows how "
        "validators allocate their stake to different miners over time. "
        "Each line represents a validator’s weight on a specific miner."
    )
    ax_para.text(
        0.5, 0.5,
        textwrap.fill(paragraph, width=140),
        ha="center", va="center",
        fontsize=11, wrap=False
    )

    ax_legend = fig.add_subplot(top_gs[2])
    ax_legend.axis("off")

    inner_gs = GridSpecFromSubplotSpec(
        nrows=rows,
        ncols=COLS,
        subplot_spec=outer_gs[1],
        wspace=0.3,
        hspace=0.4
    )

    x       = list(range(plot_epochs))
    styles  = _get_validator_styles(subset_vals)
    handles = []
    labels  = []

    for i_srv, srv in enumerate(subset_srvs):
        r, c = divmod(i_srv, COLS)
        ax = fig.add_subplot(inner_gs[r, c])

        for i_val, val in enumerate(subset_vals):
            series = data_cube[i_srv][i_val]
            ls, mk, ms, mew = styles[val]
            line, = ax.plot(
                x, series,
                linestyle=ls,
                marker=mk,
                markersize=ms,
                markeredgewidth=mew,
                linewidth=2,
                alpha=0.7
            )
            if i_srv == 0:
                handles.append(line)
                labels.append(hotkey_map.get(val, val))

        ax.set_title(srv, fontsize=10)
        ax.grid(True)
        ax.set_ylim(0, 1.05)
        _set_default_xticks(ax, plot_epochs)
        ax.set_xlabel("Epoch")
        if c == 0:
            ax.set_ylabel("Validator Weight")

    total_slots = rows * COLS
    for idx in range(num_charts, total_slots):
        r, c = divmod(idx, COLS)
        ax_empty = fig.add_subplot(inner_gs[r, c])
        ax_empty.set_visible(False)

    ncol = min(len(labels), 4)
    ax_legend.legend(
        handles, labels,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize="small",
        handletextpad=0.3,
        columnspacing=0.5
    )

    fig.tight_layout(w_pad=0.3, h_pad=0.4)

    if to_base64:
        return _plot_to_base64(fig)

    plt.show()
    plt.close(fig)
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
    fig, ax = plt.subplots(figsize=(14, 3))

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
        return _plot_to_base64(fig)
    plt.show()
    plt.close(fig)
    return None

def _plot_to_base64(fig: plt.Figure) -> str:
    """Converts a Matplotlib figure to a Base64-encoded string."""

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    plt.close(fig)
    return f'<img src="data:image/png;base64,{encoded_image}" height:auto;" draggable="false">'


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

def _prepare_bond_data_dynamic(
    bonds_per_epoch:    list[torch.Tensor],
    validators_epochs:  list[list[str]],
    miners_epochs:      list[list[str]],
    normalize:          bool = False,
) -> list[list[list[float]]]:
    """Turn each epoch’s W-matrix into a [miner_slot][validator][epoch] float list,
       padding zeros for missing entries so that normalization and indexing always work."""

    num_epochs = len(bonds_per_epoch)

    validator_keys: list[str] = []
    for vlist in validators_epochs:
        for v in vlist:
            if v not in validator_keys:
                validator_keys.append(v)

    max_miners = max(len(m) for m in miners_epochs)

    data_by_miner: list[dict[str, list[float]]] = [
        {v: [0.0] * num_epochs for v in validator_keys}
        for _ in range(max_miners)
    ]

    for e in range(num_epochs):
        W     = bonds_per_epoch[e]
        vkeys = validators_epochs[e]
        mkeys = miners_epochs[e]
        vmap  = {v: i for i, v in enumerate(vkeys)}
        mmap  = {m: i for i, m in enumerate(mkeys)}

        for mi, miner in enumerate(mkeys):
            for validator in vkeys:
                val = float(W[vmap[validator], mmap[miner]].item())
                data_by_miner[mi][validator][e] = val

    result: list[list[list[float]]] = []
    for miner_dict in data_by_miner:
        rows = [miner_dict[v] for v in validator_keys]

        if normalize:
            for t in range(num_epochs):
                epoch_vals = [row[t] for row in rows]
                total      = sum(epoch_vals)
                if total > 1e-12:
                    for row in rows:
                        row[t] /= total

        result.append(rows)

    return result


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


def _construct_relative_dividends_table(
    relative_dividends_by_version: dict[str, dict[str, list[float]]],
    validators: list[str],
    diff_versions: tuple[str, str] | None = None,
    epochs_padding: int = 0,
    num_epochs: int = 0,
    alpha_tao_ratio: float = 1.0,
) -> pd.DataFrame:
    """
        Constructs a DataFrame comparing scaled relative dividends across Yuma versions:
      - '<version>_mean': mean of (padded) series multiplied by 361 * 0.41 * num_epochs * alpha_tao_ratio
      - if diff_versions is provided, 'diff_<vA>_<vB>': scaled_mean_vA - scaled_mean_vB

    """
    effective_epochs = num_epochs - epochs_padding
    if effective_epochs < 0:
        effective_epochs = 0

    factor = 361 * 0.41 * effective_epochs * alpha_tao_ratio

    rows: list[dict[str, str]] = []
    for v in validators:
        row: dict[str, str] = {"validator": v}
        scaled_means: dict[str, float] = {}
        raw_means: dict[str, float]    = {}

        for version, divs in relative_dividends_by_version.items():
            series = divs.get(v, [])
            trimmed = series[epochs_padding:] if len(series) > epochs_padding else []

            arr = np.array([x if (x is not None) else np.nan for x in trimmed], dtype=float)

            if arr.size > 0:
                base_mean = float(np.nanmean(arr))
            else:
                base_mean = 0.0

            raw_means[version] = base_mean

            scaled = base_mean * factor
            scaled_means[version] = scaled

            scaled_str  = f"{scaled:+.2f} τ"
            raw_pct_str = f"{(base_mean * 100):+.2f}%"
            cell_text   = f"{scaled_str} ({raw_pct_str})"
            row[version] = cell_text

        if diff_versions is not None:
            vA, vB = diff_versions
            diff_col = f"diff_{vA}_{vB}"

            raw_diff    = raw_means.get(vA, 0.0) - raw_means.get(vB, 0.0)
            scaled_diff = scaled_means.get(vA, 0.0) - scaled_means.get(vB, 0.0)

            scaled_str_diff = f"{scaled_diff:+.2f} τ"
            raw_pct_diff    = f"{(raw_diff * 100):+.2f}%"
            cell_diff_text  = f"{scaled_str_diff} ({raw_pct_diff})"
            row[diff_col]   = cell_diff_text

        rows.append(row)

    df = pd.DataFrame(rows).set_index("validator")
    return df

def _generate_relative_dividends_summary_html(
    relative_dividends_by_version: dict[str, dict[str, list[float]]],
    top_validators: list[str],
    diff_versions: tuple[str, str] | None = None,
    epochs_padding: int = 0,
    num_epochs: int = 0,
    alpha_tao_ratio: float = 1.0,
    label_map: dict[str, str] | None = None,
) -> str:
    """
    Build a Bootstrap‐styled HTML table for the scaled relative dividends
    of `top_validators` across Yuma versions, with optional display name mapping.

    Scaled means by 361 * 0.41 * (num_epochs - epochs_padding) * alpha_tao_ratio.
    If `diff_versions` is provided, includes a diff column 'diff_vA_vB'.
    If `label_map` is given, uses that to replace validator IDs in the index.
    """

    df = _construct_relative_dividends_table(
        relative_dividends_by_version,
        top_validators,
        diff_versions=diff_versions,
        epochs_padding=epochs_padding,
        num_epochs=num_epochs,
        alpha_tao_ratio=alpha_tao_ratio,
    )

    if label_map is not None:
        df.index = [label_map.get(v, v) for v in df.index]

    tao_icon = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 467.715 408.195" '
        'preserveAspectRatio="xMaxYMax slice" '
        'fill="black" '
        'width="1.2em" height="1.2em" '
        'style="display:inline-block; '
            'vertical-align:middle; '
            'transform:translateY(-0.1em); '
            'margin-left:0.05em;">'
        '<path d="M271.215,286.17c-11.76,7.89-23.85,8.31-36.075,2.865c-11.43-5.1-16.695-14.64-16.725-26.955c-0.09-35.49-0.03-70.98-0.03-106.485'
        'c0-2.16,0-4.305,0-7.08c-22.05,0-43.815,0-65.52,0c-1.38-13.335,9.93-27.885,22.83-29.73c5.4-0.765,10.89-1.185,16.35-1.2'
        'c38.85-0.105,77.685-0.06,116.535-0.06c2.13,0,4.275,0,6.42,0c-0.18,16.5-11.715,30.15-30.33,30.63c-18.915,0.495-37.845,0.105-56.985,0.435'
        'c9.9,4.125,17.7,10.455,21.255,20.7c1.5,4.335,2.4,9.105,2.445,13.68c0.225,26.415-0.15,52.845,0.195,79.26C251.805,279.99,258.36,282.9,271.215,286.17z"/>'
        '</svg>'
    )

    for col in df.columns:
        df[col] = df[col].map(lambda x: (
            '<span style="white-space:nowrap; line-height:1em;">'
            + (x if isinstance(x, str) else f"{float(x):+.2f} τ")
            + tao_icon
            + '</span>'
        ))

    table_html = df.to_html(
        classes="table table-striped table-bordered",
        border=0,
        index=True,
        escape=False,
    )

    title_html = '<h4 class="mt-4">Relative Dividends Summary</h4>'
    desc_html = (
        f'<p class="mb-3">'
        f"This table shows, for each of your top validators, the mean “relative dividend” "
        f"across different Yuma versions, after scaling by "
        f"<code>361 × 0.41 × (epochs-number) × alpha-tao-ratio</code>. "
        f"(Higher numbers → better performance.)"
        f'</p>'
    )

    return f"{title_html}{desc_html}{table_html}"
