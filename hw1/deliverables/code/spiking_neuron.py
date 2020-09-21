###############################################################################
# Imports
###############################################################################
import math
import numpy as np
import pathlib
from matplotlib import pyplot as plt


class SpikingNeuron:
    def __init__(
            self,
            a = 0.02,
            b = 0.25,
            c = -65.0,
            d = 6,
            v_mem_init = -64,
            spike_thresh = 30.0,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v_mem = v_mem_init
        self.u_mem = self.b * self.v_mem
        self.spike_thresh = spike_thresh

    # Function to run 1 step of the simulation and return v_mem and spike value
    def step(self, I, TAU = 0.25):
        ret_val = dict()
        self.v_mem += TAU * (0.04 * (self.v_mem**2) + 5 * self.v_mem + 140
                - self.u_mem + I)
        self.u_mem += TAU * self.a * (self.b * self.v_mem - self.u_mem)
        if (self.v_mem > self.spike_thresh):
            self.reset_neuron()
            return (self.spike_thresh, 1)
        return (self.v_mem, 0)

    def reset_neuron(self):
        self.v_mem = self.c
        self.u_mem += self.d


if __name__ == '__main__':
    ###########################################################################
    # Common values between problems
    ###########################################################################
    # File locations
    CODE_DIR = pathlib.Path(__file__).parent.absolute() # Dir of code files
    ROOT_DIR = CODE_DIR.parent # Root project dir
    IMG_DIR = ROOT_DIR.joinpath('images') # Dir to save images to
    IMG_DIR.mkdir(mode=0o775, exist_ok=True) # Create image dir if needed
    # Define constants
    SHOW_PLOTS = False # Set to true to display plots while code is running
    SIM_TIME = 1000 # Sim time in milliseconds
    TAU = 0.25 # Sim time step in milliseconds
    SIM_STEPS = int(SIM_TIME / TAU) + 1 # total number of sim steps
    # Function to draw subplots for v_mem vs time
    def draw_subplot(t_span,v_mem_log, idx, num_subplots, title):
        subplot_id = 100*num_subplots + 10 + idx
        plt.subplot(subplot_id)
        plt.title(title)
        plt.xlim(0, SIM_TIME)
        plt.xticks(ticks=[0, t_span.max()], labels=[0, SIM_TIME])
        plt.ylim(-90, 40)
        plt.ylabel('$V_m$')
        plt.plot(t_span, v_mem_log)

    ###########################################################################
    # Problem 1
    ###########################################################################
    input_signals = [] # Keep track of input values for plotting
    v_mem_logs = dict() # Store membrane potential plots
    t_spans = dict()
    spike_trains = dict()
    input_signal = 0.0 # Strength of the input signal
    max_input_signal = 20.0 # Last input signal
    print('Simulating Problem 1')
    while (input_signal <= 20.0): # Run sims for I = 0 to 20
        neur = SpikingNeuron() # Neuron to be simulated
        t_span = np.zeros(SIM_STEPS) # Track time values
        v_mem_log = np.zeros(SIM_STEPS) # Track v_mem values
        spike_train = np.zeros(SIM_STEPS) # Track spike train
        t = 0
        for i in range(SIM_STEPS): # Run through simulation steps
            t_span[i] = t
            v_mem, spike = neur.step(input_signal, TAU) # Update neuron
            v_mem_log[i] = v_mem
            spike_train[i] = spike
            t += TAU
        # Log results and increment input_signal
        input_signals.append(input_signal)
        v_mem_logs[input_signal] = v_mem_log
        t_spans[input_signal] = t_span
        spike_trains[input_signal] = spike_train
        input_signal += 0.5
    # Draw desired membrane potential graphs
    input_vals = [1.0, 5.0, 10.0, 15.0, 20.0]
    plt.figure(figsize=(8,10))
    plt.suptitle('Regular Spiking Membrane Potential vs Time Step')
    for i, input_val in enumerate(input_vals):
        t_span = t_spans[input_val]
        v_mem_log = v_mem_logs[input_val]
        draw_subplot(t_span, v_mem_log, i+1, len(input_vals),
                f'I = {input_val}')
    plt.xlabel('Time Step (ms)')
    if (SHOW_PLOTS):
        plt.show()
    plt_name = str(IMG_DIR.joinpath(f'v_mem_plot_single.png'))
    plt.savefig(f'{plt_name}')
    plt.close()
    # Calculate spike rate values
    spike_rates_single = []
    for input_signal in input_signals:
        spike_train = spike_trains[input_signal]
        spike_train = spike_train[int(200/TAU):] # Discard first 200 ms
        spike_rates_single.append(spike_train.sum() / 800.0) # Calculate avg
    # Plot mean spike rates
    plt.figure()
    plt.title('Mean Spiking Rate vs Synaptic Current')
    plt.xlabel('Synaptic Current, I')
    plt.ylabel('Spiking Rate, R')
    plt.plot(input_signals, spike_rates_single)
    if (SHOW_PLOTS):
        plt.show()
    plt_name = str(IMG_DIR.joinpath(f'spike_rate_single.png'))
    plt.savefig(f'{plt_name}')
    plt.close()

    ###########################################################################
    # Problem 2
    ###########################################################################
    INPUT_A = 5.0
    WEIGHT_BA = 10.0
    input_signals = [] # Keep track of input values for plotting
    v_mem_logs = dict() # Store membrane potential plots
    t_spans = dict()
    spike_trains = dict()
    input_signal = 0.0 # Strength of the input signal
    max_input_signal = 20.0 # Last input signal
    print('Simulating Problem 2')
    while (input_signal <= 20.0): # Run sims for I = 0 to 20
        neur_a = SpikingNeuron()
        neur_b = SpikingNeuron()
        t_span = np.zeros(SIM_STEPS)
        v_mem_log = np.zeros(SIM_STEPS)
        spike_train = np.zeros(SIM_STEPS)
        t = 0
        for i in range(SIM_STEPS):
            t_span[i] = t
            _, spike_a = neur_a.step(INPUT_A) # Update neuron A
            # Update neuron B, combining input_signal with spike train from A
            v_mem_b,spike_b = neur_b.step(input_signal + (spike_a * WEIGHT_BA))
            v_mem_log[i] = v_mem_b
            spike_train[i] = spike_b
            t += TAU
        # Log results and increment input_signal
        input_signals.append(input_signal)
        v_mem_logs[input_signal] = v_mem_log
        t_spans[input_signal] = t_span
        spike_trains[input_signal] = spike_train
        input_signal += 0.5
    # Draw desired membrane potential graphs
    input_vals = [1.0, 5.0, 10.0, 15.0, 20.0]
    plt.figure(figsize=(8,10))
    plt.suptitle('Regular Spiking Membrane Potential vs Time Step')
    for i, input_val in enumerate(input_vals):
        t_span = t_spans[input_val]
        v_mem_log = v_mem_logs[input_val]
        draw_subplot(t_span, v_mem_log, i+1, len(input_vals),
                f'$I_B$ = {input_val}')
    plt.xlabel('Time Step (ms)')
    if (SHOW_PLOTS):
        plt.show()
    plt_name = str(IMG_DIR.joinpath(f'v_mem_plot_network.png'))
    plt.savefig(f'{plt_name}')
    plt.close()
    # Calculate spike rate values
    spike_rates_network = []
    for input_signal in input_signals:
        spike_train = spike_trains[input_signal]
        spike_train = spike_train[int(200/TAU):] # Discard first 200 ms
        spike_rates_network.append(spike_train.sum() / 800.0) # Calculate avg
    plt.figure()
    plt.title('Mean Spiking Rate vs Synaptic Current')
    plt.xlabel('Synaptic Current, I')
    plt.ylabel('Spiking Rate, R')
    plt.plot(input_signals, spike_rates_single, label='Single Neuron')
    plt.plot(input_signals, spike_rates_network, label='Simple Network')
    plt.legend(loc='upper left')
    if (SHOW_PLOTS):
        plt.show()
    plt_name = str(IMG_DIR.joinpath(f'spike_rate_network.png'))
    plt.savefig(f'{plt_name}')
    plt.close()

