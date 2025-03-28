import kinetic_analysis
import numpy as np
import matplotlib.pyplot as plt
import math
from kinetic_analysis import get_parser, Import_Kinetic_Data, Kinetic_Solver, get_inputs, graph_kinetic_data

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

parser = get_parser()
args = parser.parse_args()

inputs = get_inputs()
substrate = inputs.gen_substrate(args.substrate)

data = Import_Kinetic_Data(args.file_name, substrate)
df = data.import_data(args.wells) #(4-13) for Wells A, (16-25) for Wells B, (28-37) for Wells C

vvalues_all = data.gen_vvalues(df, time_min=5, time_max=15, steps=10)

sum_value_guess = []
sum_value_min = []
kinetic_parameters_all = []

for i in range(len(vvalues_all)):
    vvalues = vvalues_all[i]
    vm = (vvalues[0] + vvalues[1] + vvalues[2])/3
    hv = vm/2
    hv = int(hv)
    vkm = find_nearest(vvalues, hv)
    ind = np.where(vvalues == vkm)
    ind = ind[0]
    ind = ind.astype(int)
    if ind.size == 0:
        ind = 0
        print("One or More V Values are NaN, Move on to Next V Value")
    else:
        val = Kinetic_Solver(2, vm, substrate[ind[0]+1])
        s = val.sums(2, vm, substrate[ind[0]+1], vvalues, substrate)
        sum_value_guess.append(s)
        eq_to_min = val.full_equation(substrate, vvalues)
        df_dvmax, df_dh, df_dkm = val.partial_diff(eq_to_min)
        sol = val.minimize(df_dvmax, df_dh, df_dkm)
        val_min = Kinetic_Solver(sol[0], sol[1], sol[2])
        s_min = val_min.sums(sol[0], sol[1], sol[2], vvalues, substrate)
        sum_value_min.append(s_min)
        kinetic_parameters_all.append(sol)
        print(f"Done Calculating Kinetic Parameters at V Value {i}")

best_v = np.min(sum_value_min)
sum_value_min = np.array(sum_value_min)
ind_min = np.where(sum_value_min == best_v)
ind_min = ind_min[0].astype(int)
kinetic_parameters = kinetic_parameters_all[ind_min[0]]
vvalues = vvalues_all[ind_min[0]]

with open(f"{args.output}.txt", "w") as file:
    file.write(f"The velocity values matching the best-fit data are {vvalues}\n")
    file.write(f"The sum of squares is {sum_value_min[ind_min[0]]}\n")
    file.write(f"The Hill Coefficient is {kinetic_parameters[0]}\n")
    file.write(f"the Vmax Value is {kinetic_parameters[1]}\n")
    file.write(f"the Km Value is {kinetic_parameters[2]}\n")
print(f"The velocity values matching the best-fit data are {vvalues}")
print(f"The sum of squares is {sum_value_min[ind_min[0]]}")
print(f"The Hill Coefficient is {kinetic_parameters[0]}, the Vmax Value is {kinetic_parameters[1]}, the Km Value is {kinetic_parameters[2]}")

