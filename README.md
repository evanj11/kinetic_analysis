# kinetic_analysis
Read kinetic fluorescence data from csv and output either MM or Hill graph (get_kinetic_parameters.py only outputs kinetic parameters (Hill coefficient, Vmax value, Km value) to text file)

Requires sympy, numpy, matplotlib, and pandas

python gen_graphs.py -f "/path/to/kinetic/data" -w/--wells {first well to read data} {last well to read data} -s/--substrate {number of substrate concentrations} {dilution factor} {first substrate concentration} --output {file name for output graphs}

Requires access to kinetic_analysis.py script in the same folder as the gen_graphs.py script

Can read data fitting both Hill and Michelis-Menten kinetics
  MM = Hill coefficient ~1 

EXAMPLE:
python gen_graphs.py -f "/path/to/kinetic/data" -w 3 23 -s 20 1.5 80 -o "hillplot"

get_kinetic_parameters.py works the same way
