"""
run_multiple_society.py

Script to run multiple societies via command line.

Authors: John E. Parker (2026)
"""

# python libraries
import argparse
from tqdm import tqdm
from itertools import product

# user modules
import simulate_society as ss
from helpers import *

parser = argparse.ArgumentParser()

parser.add_argument(
    "-rf",
    "--range_frac_alts",
    nargs=3,
    default=[0, 1, 5],
    type=float,
    help="Min, Max (inclusive), and N for range of frac altruists.",
)
parser.add_argument(
    "-N",
    nargs=3,
    default=[3, 7, 1],
    type=int,
    help="Min, Max (inclusive), and step for range of number of agents.",
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

frac_alt_values = np.linspace(
    args.range_frac_alts[0], args.range_frac_alts[1], args.range_frac_alts[2]
)

num_agent_values = np.arange(args.N[0], args.N[1] + args.N[2], args.N[2])

social_contract_dict = {
    1: "Alts share with everyone on redistribution.",
    2: "Alts share only with everyone on redistribution.",
    3: "Alts share with altruists only with altruists trigger redistribution.",
}

if args.print_sim_info:
    print(
        f"Running {frac_alt_values.shape[0] * num_agent_values.shape[0]} societies of {num_agent_values[0]} to {num_agent_values[-1]} agents, with {frac_alt_values[0]} to {frac_alt_values[-1]} fraction as altruists. \
        \n\n  All social contracts simulated, giving {frac_alt_values.shape[0] * num_agent_values.shape[0]} x 3 = {3*frac_alt_values.shape[0] * num_agent_values.shape[0]} societies.\
        \n\n  Simulation:\n\t Trials -> {args.trials} \n\t Max Time -> {args.time} \n\t Time step -> {args.dt} \
        \n\n  Save information:\n\t Save directory -> {args.save_dir} \n\t Save outputs -> {not args.dont_save_files} \n\t Save agents -> {args.save_agents}\
        "
    )

societies = list(product(num_agent_values, frac_alt_values, [1, 2, 3]))

for n, frac, contract in tqdm(societies, desc="Running societies..."):
    ss.run_mixed_society_trials(
        N=int(n),
        redistribution_contract=int(contract),
        fraction_alts=frac,
        trials=args.trials,
        T=args.time,
        dt=args.dt,
        save_dir=f"{args.save_dir}/n_{n}_frac_alt_{frac}_contract_{contract}",
        save_files=not args.dont_save_files,
        save_agents=args.save_agents,
    )
