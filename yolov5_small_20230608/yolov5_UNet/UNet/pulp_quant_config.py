# The codes are modify from https://coin-or.github.io/pulp/CaseStudies/a_blending_problem.html

from pulp import *
import argparse


def IntegerLinearPrograming(layers=None, bits=None, sensitivity=None, BOPS=None, size=None, latency=None,
                            target_BOPS_rate=1.0, target_size_rate=1.0, target_latency_rate=1.0):
    layer_num = len(layers)

    # calculate target BOPS, params, and latency
    if target_BOPS_rate < 1:
        BOPS_all_biggest_bit = sum([BOPS[l][0] for l in range(layer_num)])
        target_BOPS = BOPS_all_biggest_bit * target_BOPS_rate
        print("target_BOPS_rate:{} \t target_size_rate:{} \t target_latency_rate:{}".format(target_BOPS_rate,
                                                                                            target_size_rate,
                                                                                            target_latency_rate))
    if target_size_rate < 1:
        size_all_biggest_bit = sum([size[l][0] for l in range(layer_num)])
        target_size = size_all_biggest_bit * target_size_rate
        print("BOPS_all_biggest_bit:{} \t size_all_biggest_bit:{} \t latency_all_biggest_bit:{}".format(
            BOPS_all_biggest_bit, size_all_biggest_bit, latency_all_biggest_bit))
    if target_latency_rate < 1:
        latency_all_biggest_bit = sum([latency[l][0] for l in range(layer_num)])
        target_latency = latency_all_biggest_bit * target_latency_rate
        print("target_BOPS:{} \t target_size:{} \t target_latency:{}".format(target_BOPS, target_size, target_latency))

    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("The_Quant_Config_Problem", LpMinimize)

    # Creates the decision variables
    config = LpVariable.dicts("Config", (layers, bits), cat="Binary")

    # Creates a list of tuples containing all the possible (layer, bit) for quant config
    Config_list = [(l, b) for l in layers for b in bits]

    # Convert data into a dictionary
    sensitivity = makeDict([layers, bits], sensitivity, 0)
    if target_BOPS_rate < 1:
        BOPS = makeDict([layers, bits], BOPS, 0)
    if target_size_rate < 1:
        size = makeDict([layers, bits], size, 0)
    if target_latency_rate < 1:
        latency = makeDict([layers, bits], latency, 0)

    # The objective function is added to 'prob' first
    prob += (
        lpSum([sensitivity[l][b] * config[l][b] for (l, b) in Config_list]),
        "Total_Sensitivity_of_Quant_Config",
    )

    # The constraints are added to 'prob'
    for l in layers:
        prob += lpSum([config[l][b] for b in bits]) == 1

    if target_BOPS_rate < 1:
        prob += (
            lpSum([BOPS[l][b] * config[l][b] for (l, b) in Config_list]) <= target_BOPS,
            "FlopsRequirement",
        )

    if target_size_rate < 1:
        prob += (
            lpSum([size[l][b] * config[l][b] for (l, b) in Config_list]) <= target_size,
            "SizeRequirement",
        )

    if target_latency_rate < 1:
        prob += (
            lpSum([latency[l][b] * config[l][b] for (l, b) in Config_list]) <= target_latency,
            "LatencyRequirement",
        )

    # The problem data is written to an .lp file
    prob.writeLP("QuantConfigModel.lp")

    # The problem is solved using PuLP's choice of Solver
    prob.solve()

    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])

    # Each of the variables is printed with it's resolved optimum value
    for v in prob.variables():
        print(v.name, "=", v.varValue)

    # The optimised objective function value is printed to the screen
    print("Total Sensitity of Quant Config = ", value(prob.objective))

    # Flops constraints
    if target_BOPS_rate < 1:
        total_BOPS = sum([BOPS[l][b] * value(config[l][b]) for (l, b) in Config_list])
        print("Total Flops of Quant Config = {}".format(total_BOPS))

    # Size constraints
    if target_size_rate < 1:
        total_size = sum([size[l][b] * value(config[l][b]) for (l, b) in Config_list])
        print("Total Size of Quant Config = {}".format(total_size))

    # Latency constraints
    if target_latency_rate < 1:
        total_latency = sum([latency[l][b] * value(config[l][b]) for (l, b) in Config_list])
        print("Total Latency of Quant Config = {}".format(total_latency))

    name_bit_dict = {}
    # Each of the variables is printed with it's resolved optimum value
    for v in prob.variables():
        if v.varValue == 1.0:
            name_bit_dict['_'.join(v.name.split("_")[1:-1])] = v.name.split("_")[-1]

    return name_bit_dict


if __name__ == '__main__':
    # example bash
    # python pulp_quant_config.py -tfr 0.8 -tsr 0.8 -tlr 0.9
    parser = argparse.ArgumentParser(description='Pulp for Integer Linear Programing')
    parser.add_argument('-tfr', '--target_BOPS_rate', default=1.0, type=float)
    parser.add_argument('-tsr', '--target_size_rate', default=1.0, type=float)
    parser.add_argument('-tlr', '--target_latency_rate', default=1.0, type=float)

    args = parser.parse_args()

    ############################################# human design test data #############################################
    # Creates a list of the Layers
    layers = ["layer1", "layer2", "layer3", "layer4", "layer5"]

    # Creates a list of the bits
    bits = ["16bit", "8bit"]

    # A dictionary of the sensitivity of each config is created
    sensitivity = [[50, 120],
                   [45, 100],
                   [40, 94],
                   [42, 105],
                   [55, 140]]

    # A dictionary of the BOPS of each config is created
    BOPS = [[250, 18],
            [245, 200],
            [240, 194],
            [242, 175],
            [255, 140]]

    # A dictionary of the size of each config is created
    size = [[250, 120],
            [245, 100],
            [240, 94],
            [242, 105],
            [255, 140]]

    # A dictionary of the latency of each config is created
    latency = [[250, 120],
               [245, 100],
               [240, 94],
               [242, 105],
               [255, 140]]

    ############################################# human design test data #############################################

    IntegerLinearPrograming(layers=layers, bits=bits, sensitivity=sensitivity,
                            BOPS=BOPS, size=size, latency=latency,
                            target_BOPS_rate=args.target_BOPS_rate,
                            target_size_rate=args.target_size_rate,
                            target_latency_rate=args.target_latency_rate)