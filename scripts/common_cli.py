"""
This module defines a common CLI parser for metagraph-based data simulation scripts.
It provides various arguments to configure the simulation, including parameters
for subnet id, bond penalties, epochs, tempo, and output settings.
"""

import argparse

def _create_common_parser():
    parser = argparse.ArgumentParser(
        description="Common CLI for Yuma simulation scripts."
    )

    # Group 1: Scenario Specific Arguments (these may be overridden by the JSON config)
    scenarios_specific_args = parser.add_argument_group("Scenario Specific Arguments")
    scenarios_specific_args.add_argument(
        "--subnet-id",
        type=int,
        default=0,
        help="The subnet id to run the diagnostics on (overridden by JSON config).",
    )
    scenarios_specific_args.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of epochs to simulate (overridden by JSON config).",
    )
    scenarios_specific_args.add_argument(
        "--tempo",
        type=int,
        default=360,
        help="Tempo value for the simulation (overridden by JSON config).",
    )
    scenarios_specific_args.add_argument(
        "--shift-validator-id",
        type=int,
        default=0,
        help="The uid of the validator with the weights shifted by 1 epoch back (overridden by JSON config).",
    )
    scenarios_specific_args.add_argument(
        "--top-validators",
        type=str,
        nargs="+",
        help="List of top validator indices (overridden by JSON config).",
    )

    # Group 2: Global Simulation Arguments
    global_simulation_args = parser.add_argument_group("Global Simulation Arguments")
    global_simulation_args.add_argument(
        "--config-file",
        type=str,
        default="./simulation_config.json",
        help="Path to JSON config with scenarios and YUMA parameters.",
    )
    global_simulation_args.add_argument(
        "--run-multiple-scenarios",
        action="store_true",
        help="Use a simulation config file to run multiple scenarios.",
    )
    return parser
