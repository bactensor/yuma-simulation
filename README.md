# Yuma Consensus Simulation Package
&nbsp;[![Continuous Integration](https://github.com/DarnoX-reef/yuma-simulation/workflows/Continuous%20Integration/badge.svg)](https://github.com/DarnoX-reef/yuma-simulation/actions?query=workflow%3A%22Continuous+Integration%22)&nbsp;[![License](https://img.shields.io/pypi/l/yuma_simulation.svg?label=License)](https://pypi.python.org/pypi/yuma_simulation)&nbsp;[![python versions](https://img.shields.io/pypi/pyversions/yuma_simulation.svg?label=python%20versions)](https://pypi.python.org/pypi/yuma_simulation)&nbsp;[![PyPI version](https://img.shields.io/pypi/v/yuma_simulation.svg?label=PyPI%20version)](https://pypi.python.org/pypi/yuma_simulation)

## Overview
This package provides a suite of tools and simulation frameworks for exploring Bittensor Yuma Consensus mechanisms. Developed in Python, it supports multiple versions of Yuma simulations and includes features for both synthetic and real-world scenarios.

Simulations can be executed using predefined synthetic cases, designed to analyze and address challenges in various Yuma iterations. Alternatively, simulations can leverage archived real metagraph data from selected subnets, enabling realistic case studies. The output data can be visualized through interactive .html charts or exported as raw .csv files for further analysis.

In addition to core simulations, the package includes specialized scripts to generate data for both synthetic and real-world cases. A notable feature of the real-case simulations is the "shifted validator" mode, which introduces delayed weight commits by shifting the weight state of a specified validator to the subsequent epoch.

## Usage

### Using the archived metagraph based simulation script
There is a json configuration 'simulation_config.json' that enables you to tailor the simulation runs, outputs, and data management to your specific requirements. Adjust the values as needed to experiment with different subnets, hyperparameters, and simulation versions.

Output Options:
Enable or disable chart and dividend table generation using generate_chart_table and generate_dividends_table. At least one must be set to true to produce outputs.

Directories:
Set output_dir and metagraphs_dir to point to directories where input data and outputs will be saved.

Data Initialization:
On the first run, set download_new_metagraph to true so that the necessary metagraph data is downloaded.

Epoch Parameters:
Adjust epochs_padding to ignore an appropriate number of initial epochs (typically 20–40) to allow the bonds and weights to stabilize, and set epochs_window to define how many epochs are averaged together in the output tables.

Scenarios:
Each scenario under the scenarios array specifies a distinct simulation run for a particular subnet. Configure the subnet ID, number of epochs, tempo, the validator to shift (simulate higher GPU load), and which top validators to include in the output charts and tables.

Hyperparameters & Versions:
The simulation_hyperparameters and yuma_versions sections allow you to fine-tune simulation behavior. For each Yuma version listed, the script will run the simulations using the provided parameters.
To see available Yuma version refer to YumaSimulationNames dataclass in the yumas.py - the versions should be provided as fields from this structue. ("YUMA1", "YUMA2" etc.)

Example usage to run the script using multiple scenarios from the json configuration:

```
python ./scripts/archived_metagraph_simulation.py --run-multiple-scenarios
```

- **HTML charts**  
  For each Yuma version, the table shows four rows per subnet run:  
  1. Normal case dividends  
  2. Shifted case dividends  
  3. Combined comparison dividends  
  4. Stake‐scaled comparison  

- **Dividends CSV**  
  Columns for each version:  
  ```text
  Normal_<VERSION>
  Shifted_<VERSION>
  Comparison_<VERSION>
  ```

### Using the synthetic cases chart generation script

The chart_table_generation script runs a suite of “synthetic” metagraph simulations across a sweep of bond-penalty values and Yuma versions, then outputs an HTML chart table for each penalty. It requires only the default configuration in code—no external JSON config is provided.

Example usage to run the script:

```
python ./scripts/charts_table_generator.py
```

## Versioning

This package uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
TL;DR you are safe to use [compatible release version specifier](https://packaging.python.org/en/latest/specifications/version-specifiers/#compatible-release) `~=MAJOR.MINOR` in your `pyproject.toml` or `requirements.txt`.

Additionally, this package uses [ApiVer](https://www.youtube.com/watch?v=FgcoAKchPjk) to further reduce the risk of breaking changes.
This means, the public API of this package is explicitly versioned, e.g. `yuma_simulation.v1`, and will not change in a backwards-incompatible way even when `yuma_simulation.v2` is released.

Internal packages, i.e. prefixed by `yuma_simulation._` do not share these guarantees and may change in a backwards-incompatible way at any time even in patch releases.


## Development


Pre-requisites:
- [pdm](https://pdm.fming.dev/)
- [nox](https://nox.thea.codes/en/stable/)
- [docker](https://www.docker.com/) and [docker compose plugin](https://docs.docker.com/compose/)


Ideally, you should run `nox -t format lint` before every commit to ensure that the code is properly formatted and linted.
Before submitting a PR, make sure that tests pass as well, you can do so using:
```
nox -t check # equivalent to `nox -t format lint test`
```

If you wish to install dependencies into `.venv` so your IDE can pick them up, you can do so using:
```
pdm install --dev
```

### Modifying or Adding Yuma Versions

All available Yuma versions are defined in `yumas.py` alongside their parameter dataclasses:

- **Global simulation parameters**  
  The `SimulationHyperparameters` class controls settings applied across every Yuma run (e.g. `bond_penalty`, `liquid_alpha_consensus_mode`).

- **Per-version parameters**  
  The `YumaParams` class encapsulates version-specific settings (e.g. `bond_moving_avg`, `alpha_low`, `alpha_high`, `liquid_alpha`, `alpha_sigmoid_steepness`).

To introduce a new Yuma version:

1. **Update `YumaSimulationNames`**  
   Add a new field (e.g. `YUMA4B`) in `YumaSimulationNames`.

2. **Define its `YumaParams`**  
   Pass your custom values when you build the `(version, params)` tuple in your script.

3. **Extend consensus logic if needed**  
   Any special “reset bonds” or output-calculation rules live in `simulation_utils.py` under `_call_yuma()`. Hook or override logic there if your new version requires non-standard behavior.

---

### Adding or Extending Synthetic Cases

Synthetic test cases live in `cases.py` as dataclasses that inherit from `BaseCase`. Each case includes:

- `name`  
- `base_validator` (the reference validator for relative‐dividends plots)  
- `validators` (list of validator names to plot)  
- `servers` (list of server/miner names to plot)  
- `num_epochs`  
- **Bond-reset configuration** (optional):  
  - `reset_bonds` (bool)  
  - `reset_bonds_index` (which validator is reset)  
  - `reset_bonds_epoch` (when the reset happens)  
- **Matrix flags**:  
  - `use_full_matrices` (toggle full vs. per-case matrices)  
  - `_get_base_weights_epochs` & `_get_base_stakes_epochs` (per-case matrices for non-full mode)  

Most built-in cases only override:
- `name`
- `validators`
- `base_validator`
- `_get_base_weights_epochs`
- `_get_base_stakes_epochs`

To add your own case:

1. **Create a new dataclass** inheriting from `BaseCase`.  
2. **Override** the `BaseCase` attributes.
4. **Choose chart types** via the `chart_types` set:  
   - Use `weights_subplots` instead of `weights` when you have more than two servers.  

For examples, see cases 15–18 in `cases.py`, which demonstrate multi-server configurations and custom chart settings.  

### Release process

Run `nox -s make_release -- X.Y.Z` where `X.Y.Z` is the version you're releasing and follow the printed instructions.