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

    # Group 1: Overwritten by JSON config
    scenarios_specific_args = parser.add_argument_group("Scenario Specific Arguments")
    scenarios_specific_args.add_argument(
        "--subnet-id",
        type=int,
        default=0,
        help="The subnet id to run the diagnostics on (overwritten by JSON config).",
    )
    scenarios_specific_args.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of epochs to simulate (overwritten by JSON config).",
    )
    scenarios_specific_args.add_argument(
        "--tempo",
        type=int,
        default=360,
        help="Tempo value for the simulation (overwritten by JSON config).",
    )
    scenarios_specific_args.add_argument(
        "--shift-validator-id",
        type=int,
        default=0,
        help="The uid of the validator with the weights shifted by 1 epoch back (overwritten by JSON config).",
    )
    scenarios_specific_args.add_argument(
        "--top-validators",
        type=str,
        nargs="+",
        help="List of top validator indices (space-separated hotkeys) that will be plotted in the charts.",
    )


    # Group 2: Global simulation arguments (not overwritten by JSON config)
    global_simulation_args = parser.add_argument_group("Global Simulation Arguments")
    global_simulation_args.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="The dir to store simulation results.",
    )
    global_simulation_args.add_argument(
        "--metagraphs-dir",
        type=str,
        default="metagraphs",
        help="Directory which stores the downloaded metagraphs.",
    )
    global_simulation_args.add_argument(
        "--output-prefix",
        type=str,
        default="metagraph_simulation_results",
        help="Prefix for output file names.",
    )
    global_simulation_args.add_argument(
        "--draggable-table",
        action="store_true",
        help="Make the table draggable.",
    )
    global_simulation_args.add_argument(
        "--download-new-metagraph",
        action="store_true",
        help="Download a new metagraph.",
    )
    global_simulation_args.add_argument(
        "--introduce-shift",
        action="store_true",
        help="Introduce shift in the simulation.",
    )
    global_simulation_args.add_argument(
        "--use-json-config",
        action="store_true",
        help="Use a JSON config file to run multiple scenarios.",
    )
    global_simulation_args.add_argument(
        "--config-file",
        type=str,
        default="metagraph_subnets_config.json",
        help="Path to JSON file with multiple scenario definitions.",
    )
    global_simulation_args.add_argument(
        "--epochs-padding",
        type=int,
        default=0,
        help="The first x epochs that are not plotted in the charts."
    )
    return parser
