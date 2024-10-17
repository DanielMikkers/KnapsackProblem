from Classes import GenData, KnapsackSolver, NeuralNetworkDP
import numpy as np
import matplotlib.pyplot as plt
import time

np.set_printoptions(edgeitems=10,linewidth=180)

#creating lists to append all values to
acc_ratio_n_bin = []
acc_ratio_n_dp = []
acc_ratio_n_gh = []

acc_ratio_W_bin = []
acc_ratio_W_dp = []
acc_ratio_W_gh = []

time_n_bin = []
time_n_dp = []
time_n_gh = []

time_W_bin = []
time_W_dp = []
time_W_gh = []

#creating relevant arrays and values for plotting relevant measures
n_start = 1
n_end = 3
n_space = np.round(np.logspace(n_start,n_end)).astype(int)
W_fix = 500

W_start = 2
W_end = 4
W_space = np.round(np.logspace(W_start,W_end)).astype(int)
n_fix = 50

#obtaining the optimal values and obtaining the difference between
#BP and DP/GH as a function of n
for i in n_space:
    data = GenData(i,W_fix)
    w,v = data.generate()

    opt_bin = KnapsackSolver(solver='Bin').knapsackSolver(i, W_fix, v, w)
    opt_dyn = KnapsackSolver(solver='DP').knapsackSolver(i, W_fix, v, w)
    opt_grh = KnapsackSolver(solver='GH').knapsackSolver(i, W_fix, v, w)

    opt_val_bin = np.sum(v[opt_bin])
    opt_val_dp = np.sum(v[opt_dyn])
    opt_val_gh = np.sum(v[opt_grh])

    acc_ratio_n_bin.append(opt_val_bin-opt_val_bin)
    acc_ratio_n_dp.append(opt_val_bin-opt_val_dp)
    acc_ratio_n_gh.append(opt_val_dp-opt_val_gh)

acc_ratio_n_bin = np.array(acc_ratio_n_bin)
acc_ratio_n_dp = np.array(acc_ratio_n_dp)
acc_ratio_n_gh = np.array(acc_ratio_n_gh)

#plotting the found differences
plt.figure(dpi=300)
plt.plot(n_space,acc_ratio_n_dp, '-ro', label='DP')
plt.plot(n_space,acc_ratio_n_gh, '-gs', label='GH')
plt.xscale('log')
plt.yscale('symlog')
plt.ylim(-1,300)
plt.grid(True, which="both", ls="-", color='0.65')
plt.xlabel(r'$n$ [number of items]')
plt.ylabel(r'$OPT_{BP}-OPT_{DP,GH}$')
plt.legend()
#plt.savefig('acc_plot_n_W_500.png')
plt.show()

#obtaining the optimal values and obtaining the difference between
#BP and DP/GH as a function of W
for i in W_space:
    data = GenData(n_fix,i)
    w,v = data.generate()

    opt_bin = KnapsackSolver(solver='Bin').knapsackSolver(n_fix, i, v, w)
    opt_dyn = KnapsackSolver(solver='DP').knapsackSolver(n_fix, i, v, w)
    opt_grh = KnapsackSolver(solver='GH').knapsackSolver(n_fix, i, v, w)

    opt_val_bin = np.sum(v[opt_bin])
    opt_val_dp = np.sum(v[opt_dyn])
    opt_val_gh = np.sum(v[opt_grh])

    acc_ratio_W_bin.append(opt_val_bin-opt_val_bin)
    acc_ratio_W_dp.append(opt_val_bin-opt_val_dp)
    acc_ratio_W_gh.append(opt_val_bin-opt_val_gh)

acc_ratio_W_bin = np.array(acc_ratio_W_bin)
acc_ratio_W_dp = np.array(acc_ratio_W_dp)
acc_ratio_W_gh = np.array(acc_ratio_W_gh)

#plotting the differences
plt.figure(dpi=300)
plt.plot(W_space,acc_ratio_W_dp, '-ro', label='DP')
plt.plot(W_space,acc_ratio_W_gh, '-gs', label='GH')
plt.xscale('log')
plt.yscale('symlog')
plt.ylim(-1,2000)
plt.grid(True, which="both", ls="-", color='0.65')
plt.xlabel(r'$W$ [capacity of knapsack]')
plt.ylabel(r'$OPT_{BP}-OPT_{DP,GH}$')
plt.legend()
#plt.savefig('acc_plot_W_n_50.png')
plt.show()

#obtaining the execution time of the three methods as a function of n
for i in n_space:
    data = GenData(i,W_fix)
    w,v = data.generate()

    start_bin = time.time()
    opt_bin = KnapsackSolver(solver='Bin').knapsackSolver(i, W_fix, v, w)
    end_bin = time.time()
    execution_time_bin = end_bin - start_bin
    time_n_bin.append(execution_time_bin)
    
    start_dyn = time.time()
    opt_dyn = KnapsackSolver(solver='DP').knapsackSolver(i, W_fix, v, w)
    end_dyn = time.time()
    execution_time_dyn = end_dyn - start_dyn
    time_n_dp.append(execution_time_dyn)
    
    start_grh = time.time()
    opt_grh = KnapsackSolver(solver='Bin').knapsackSolver(i, W_fix, v, w)
    end_grh = time.time()
    execution_grh = end_grh - start_grh
    time_n_gh.append(execution_grh)

time_n_bin = np.array(time_n_bin)
time_n_dp = np.array(time_n_dp)
time_n_gh = np.array(time_n_gh)

#plotting execution time as a function of n
plt.figure(dpi=300)
plt.plot(n_space,time_n_bin, '-b^', label='BP')
plt.plot(n_space,time_n_dp, '-ro', label='DP')
plt.plot(n_space,time_n_gh, '-gs', label='GH')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="-", color='0.65')
plt.xlabel(r'$n$ [number of items]')
plt.ylabel(r'Execution time [s]')
plt.legend()
#plt.savefig('ex_time_plot_n_W_500.png')
plt.show()

#obtianing the execution time as a function of W
for i in W_space:
    data = GenData(n_fix,i)
    w,v = data.generate()

    start_bin = time.time()
    opt_bin = KnapsackSolver(solver='Bin').knapsackSolver(n_fix, i, v, w)
    end_bin = time.time()
    execution_time_bin = end_bin - start_bin
    time_W_bin.append(execution_time_bin)
    
    start_dyn = time.time()
    opt_dyn = KnapsackSolver(solver='DP').knapsackSolver(n_fix, i, v, w)
    end_dyn = time.time()
    execution_time_dyn = end_dyn - start_dyn
    time_W_dp.append(execution_time_dyn)
    
    start_grh = time.time()
    opt_grh = KnapsackSolver(solver='Bin').knapsackSolver(n_fix, i, v, w)
    end_grh = time.time()
    execution_grh = end_grh - start_grh
    time_W_gh.append(execution_grh)

time_W_bin = np.array(time_W_bin)
time_W_dp = np.array(time_W_dp)
time_W_gh = np.array(time_W_gh)

#plotting the execution time as a function of W
plt.figure(dpi=300)
plt.plot(W_space,time_W_bin, '-b^', label='BP')
plt.plot(W_space,time_W_dp, '-ro', label='DP')
plt.plot(W_space,time_W_gh, '-gs', label='GH')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="-", color='0.65')
plt.xlabel(r'$W$ [capacity of knapsack]')
plt.ylabel(r'Execution time [s]')
plt.legend()
#plt.savefig('ex_time_plot_W_n_50.png')
plt.show()

