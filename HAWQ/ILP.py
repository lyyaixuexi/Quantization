import numpy as np
from pulp import *
import pulp

class args():
    def __init__(self):
        pass
args = args
args.model_size_limit = 0.5 # the model size limit, 0.25 means, 4bit_model_size * 0.75 + 8bit_model_size * 0.25
args.bops_limit = 0.5 # same definition as above
args.latency_limit = 0.5 # same definition as above

# those are different matrics

# Hutchinson_trace means the trace of Hessian for each weight matrix.
# Particular, it has 19 elements since ResNet18 has 18 layers, removing the first and last layer
# we still have 16 layers (the other three come from residual connection layer)
# These values are already normlized, i.e., Trace / # arameters
Hutchinson_trace = np.array([0.06857826, 0.03162379, 0.03298575, 0.01205663, 0.02222431, 0.00596336, 0.06931772, 0.00807129, 0.00372905, 0.00530698, 0.00209011, 0.00737569, 0.00210454, 0.00151197, 0.00158041,0.00078146, 0.00451841, 0.00098745, 0.00072944])
# Delta Weight 8 bit Square means \| W_fp32 - W_int8  \|_2^2
delta_weights_8bit_square = np.array([0.0235, 0.0125, 0.0102, 0.0082, 0.0145, 0.0344, 0.0023, 0.0287, 0.0148, 0.0333, 0.0682, 0.0027, 0.0448, 0.0336, 0.0576, 0.1130, 0.0102, 0.0947, 0.0532]) #  = (w_fp32 - w_int8)^2
# Delta Weight 4 bit Square means \| W_fp32 - W_int4  \|_2^2
delta_weights_4bit_square = np.array([6.7430, 3.9691, 3.3281, 2.6796, 4.7277, 10.5966, 0.6827, 9.0942, 4.8857, 10.7599, 21.7546, 0.8603, 14.5324, 10.9651, 18.7706, 36.4044, 3.1572, 29.6994, 17.4016]) #  = (w_fp32 - w_int4)^2
# number of paramers of each layer
parameters = np.array([ 36864, 36864, 36864, 36864, 73728, 147456, 8192, 147456, 147456, 294912, 589824, 32768, 589824, 589824, 1179648, 2359296, 131072, 2359296, 2359296]) / 1024 / 1024 # make it millition based (1024 is due to Byte to MB see next cell for model size computation)
# Bit Operators of each layer
bops = np.array([115605504, 115605504, 115605504, 115605504, 57802752, 115605504, 6422528, 115605504, 115605504, 57802752, 115605504, 6422528, 115605504, 115605504, 57802752, 115605504, 6422528, 115605504, 115605504]) / 1000000 # num of bops for each layer/block
# latency for INT4 and INT8, measured on T4 GPU
latency_int4 = np.array([0.21094404, 0.21092674, 0.21104113, 0.21086851, 0.13642465, 0.19167506, 0.02532183, 0.19148203, 0.19142914, 0.11395316, 0.20556011, 0.01917474, 0.20566918, 0.20566509, 0.13185102, 0.22287786, 0.01790088, 0.22304611, 0.22286099])
latency_int8 = np.array([0.36189111, 0.36211718, 0.31141909, 0.30454471, 0.19184896, 0.38948934, 0.0334169, 0.38904905, 0.3892859, 0.19134735, 0.34307431, 0.02802354, 0.34313329, 0.34310756, 0.21117103, 0.37376585, 0.02896843, 0.37398187, 0.37405185])

# do some calculatations first
# model size
model_size_32bit = np.sum(parameters) * 4. # MB
model_size_8bit = model_size_32bit / 4. # 8bit model is 1/4 of 32bit model
model_size_4bit = model_size_32bit / 8. # 4bit model is 1/8 of 32bit model
# as mentioned previous, that's how we set the model size limit
model_size_limit = model_size_4bit + (model_size_8bit - model_size_4bit) * args.model_size_limit

# bops
bops_8bit = bops / 4. / 4. # For Wx, we have two matrices, so that we need the (bops / 4 / 4)
bops_4bit = bops / 8. / 8. # Similar to above line
bops_limit = np.sum(bops_4bit) + (np.sum(bops_8bit) - np.sum(bops_4bit)) * args.bops_limit # similar to model size

# latency
latency_limit = np.sum(latency_int4) + (np.sum(latency_int8) - np.sum(latency_int4)) * args.latency_limit # similar to model size

# Let's construct the problem
num_variable = Hutchinson_trace.shape[0]

# first get the variables, here 1 means 4 bit and 2 means 8 bit
variable = {}
for i in range(num_variable):
    variable[f"x{i}"] = LpVariable(f"x{i}", 1, 2, cat=LpInteger)

prob = LpProblem("Model_Size", LpMinimize)
prob += sum([0.5 * variable[f"x{i}"] * parameters[i] for i in range(num_variable) ]) <= model_size_limit # 1 million 8 bit numbers means 1 Mb, here 0.5 * 2 = 1, similar for 4 bit

# add downsampling layer constraint, here we make the residual connetction
# layer have the same bit as the main stream
prob += variable[f"x4"] ==variable[f"x6"]
prob += variable[f"x9"] ==variable[f"x11"]
prob += variable[f"x14"] ==variable[f"x16"]

sensitivity_difference_between_4_8 = Hutchinson_trace * ( delta_weights_8bit_square - delta_weights_4bit_square ) # here is the sensitivity different between 4 and 8

# for fixed bops, we want large models, as well as smaller sensitivy
# here sensitivity_difference_between is negative numbers, so if x = 1, means using 4 bit, gives us 0, and if x = 2, means using 8 bits, gives us negavie numers. It will prefer 8 bit
# negative model size is negaive number. It will prefer 8 bit.
# both prefer 8 bit but the bops is constrained, so we get a tradeoff.
prob += sum( [ (variable[f"x{i}"] - 1) * sensitivity_difference_between_4_8[i] for i in range(num_variable) ] )

# solve the problem
status = prob.solve(GLPK_CMD(msg=1, options=["--tmlim", "10000","--simplex"]))
# status = prob.solve(COIN_CMD(msg=1, options=['dualSimplex']))
# status = prob.solve(GLPK(msg=1, options=["--tmlim", "10","--dualSimplex"]))

# get the result
LpStatus[status]

result = []
for i in range(num_variable):
    result.append(value(variable[f"x{i}"]))
result = np.array(result)

print(result)
# Note that this model size does not count the first and last layer of the network
# In the paper, we manually added this two layers in the final result
print('Model Size:', np.sum(result * parameters * 4 * 4 / 32))
result_4 = (result == 1)
result_8 = (result == 2)
print("Bops: ", np.sum(bops_4bit[result_4]) + np.sum(bops_8bit[result_8]))
print("Latency: ", np.sum(latency_int4[result_4]) + np.sum(latency_int8[result_8]))