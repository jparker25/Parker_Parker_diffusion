"""
run_mixed_society.py

Script to run mixed society via command line.

Authors: John E. Parker (2026)
"""

# python libraries
import argparse

# user modules
import simulate_society as ss
from helpers import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "-rc", "--contract", nargs=1, default=1, type=int, help="Choice of social contract."
)
parser.add_argument(
    "-f", "--frac_alts", nargs=1, default=0.5, type=float, help="Fraction of altruists."
)
parser.add_argument(
    "-N", nargs=1, default=4, type=int, help="Number of agents in society."
)
parser.add_argument(
    "-t",
    "--trials",
    nargs=1,
    default=1000,
    type=int,
    help="Number of societies to run with implementation.",
)
parser.add_argument(
    "-T", "--time", nargs=1, default=1000, type=float, help="Max simulation time."
)
parser.add_argument(
    "-dt", nargs=1, default=0.01, type=float, help="Time step for simulation."
)
parser.add_argument(
    "-sd",
    "--save_dir",
    nargs=1,
    default="../data",
    type=str,
    help="Path to directory to save data.",
)
parser.add_argument(
    "-nf",
    "--dont_save_files",
    action="store_true",
    help="If given, don't save meta_data outputs.",
)
parser.add_argument(
    "-sa",
    "--save_agents",
    action="store_true",
    help="If given, store agent wealth trajectories.",
)

parser.add_argument(
    "-p",
    "--print_sim_info",
    action="store_true",
    help="If given, print simulation info.",
)

args = parser.parse_args()

social_contract_dict = {
    1: "Alts share with everyone on redistribution.",
    2: "Alts share only with everyone on redistribution.",
    3: "Alts share with altruists only with altruists trigger redistribution.",
}

if args.print_sim_info:
    print(
        f"Running society of {args.N} agents, with {args.frac_alts} fraction as altruists. \
        \n\n  Social contract is {args.contract}:\n\t {social_contract_dict[args.contract]}\
        \n\n  Simulation:\n\t Trials -> {args.trials} \n\t Max Time -> {args.time} \n\t Time step -> {args.dt} \
        \n\n  Save information:\n\t Save directory -> {args.save_dir} \n\t Save outputs -> {not args.dont_save_files} \n\t Save agents -> {args.save_agents}\
        "
    )

ss.run_mixed_society_trials(
    N=args.N,
    redistribution_contract=args.contract,
    fraction_alts=args.frac_alts,
    trials=args.trials,
    T=args.time,
    dt=args.dt,
    save_dir=args.save_dir,
    save_files=not args.dont_save_files,
    save_agents=args.save_agents,
)
