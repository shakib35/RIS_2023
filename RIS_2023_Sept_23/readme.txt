This program is based on the "RIS-enhanced CoMP-NOMA with MIMO Aspects Efficiency" paper. This code was converted from MATLAB to python code and further modified to perform the desired task.

Main Program:
RIS_Full_Problem.py - Handles main operation of program.


Support Programs:
RIS_Channel_Generation.py - Creates a new random RIS system channel configuration based on specified parameters.
Optimize_RIS_Elements.py - Optimizes the RIS elements by converting the subsection of the problem into a modified knapsack problem.
Subproblem_1.py - Contains all functions related to Subproblem 1.
Subproblem_2.py - Contains all functions related to Subproblem 2.


Algorithm pseudo code:
test_parameters = ["potential lists of params: ris_elements, signal_power, iot_devices, etc."]
results = []
for p test_parameters:
    new_ris_config = RIS_Channel_Generation(set_param=p)    //generate new RIS Config
    tau0, tau1 = Optimize_RIS_Elements(new_ris_config)      //optimize RIS elements for harvesting and reflection
    new_ris_config.tau0, new_ris_config.tau1 = tau0, tau1   //update RIS system values

    sub_2_solve = new_ris_config                            //quick hack to set up optimization loop
    opt_loop_results = []
    for i in range(0,10):                                   //loop Sub_1 -> Sub_2 -> Sub_1 -> Sub_2 -> ...
        sub_1_solve = Subproblem_1(sub_2_solve)             //solve subproblem 1
        sub_2_solve = Subproblem_2(sub_1_solve)             //solve subproblem 2
        opt_output = sub_2_solve.value
        opt_loop_results.append(opt_output)
        if last 3 values of results
            decreases monotonically:                        //optimization stopping criteria
            break                                           //kill sub-loop
    results.append(opt_loop_results[-1])                    //capture final optimized output values









Directions to Run Program:
1) Run RIS_Full_Problem.py to produce data points for each noise power scenario. Specify the number of data points required for each sub-scenario. The program will output data to the appropriate subdirectory for the noise power scenario. Data is already caculated for the -30dBm and the -100dBm noise scenarios.
2) Run RIS_Plotting.py to plot data points and create data graphics. Graphics will be stored within the appropriate noise power scenario. Graphs are already generated for the -30dBm and the -100dBm noise scenarios.