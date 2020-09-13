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
            c = -65,
            d = 6,
            v_mem_init = -64,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v_mem = v_mem_init
        self.u_mem = self.b * self.v_mem

    def step(self, I, TAU = 0.25):
        self.v_mem += TAU * (0.04 * (self.v_mem**2) + 5 * self.v_mem + 140 - self.u_mem + I)
        self.u_mem += TAU * self.a * (self.b * self.v_mem - self.u_mem)

    def reset_spike(self):
        self.v_mem = self.c
        self.u_mem += self.d

    def get_v_mem(self):
        return self.v_mem

    def get_u_mem(self):
        return self.u_mem


if __name__ == '__main__':
    # File locations
    CODE_DIR = pathlib.Path(__file__).parent.absolute()
    print(CODE_DIR)
    ROOT_DIR = CODE_DIR.parent
    IMG_DIR = ROOT_DIR.joinpath('images')
    IMG_DIR.mkdir(mode=0o775, exist_ok=True)
    # Define constants
    SHOW_PLOTS = False
    SIM_TIME = 1000 # Simulation time in milliseconds
    TAU = 0.25 # Simulation time step in milliseconds
    SIM_STEPS = int(SIM_TIME / TAU) + 1 # simulation steps
    INPUT_START_TIME = 0 # Time to start input signal
    SPIKE_THRESH = 30.0 # Threshold to determine spikes

    # Simulation
    input_signals = []
    v_mem_logs = dict()
    u_mem_logs = dict()
    t_spans = dict()
    spike_trains = dict()
    input_signal = 0.0 # Strength of the input signal
    max_input_signal = 20.0 # Last input signal
    while (input_signal <= 20.0):
        print(f'Simulating input_signal = {input_signal}')
        neur = SpikingNeuron() # Neuron to be simulated
        t_span = np.zeros(SIM_STEPS)
        v_mem_log = np.zeros(SIM_STEPS)
        u_mem_log = np.zeros(SIM_STEPS)
        spike_train = np.zeros(SIM_STEPS)
        t = 0
        I = 0
        for i in range(SIM_STEPS):
            t_span[i] = t
            if (t > INPUT_START_TIME):
                I = input_signal
            neur.step(I, TAU)
            t += TAU
            v_mem_log[i] = neur.get_v_mem()
            if (v_mem_log[i] > SPIKE_THRESH):
                v_mem_log[i] = SPIKE_THRESH
                neur.reset_spike()
                spike_train[i] = 1
            u_mem_log[i] = neur.get_u_mem()
        # Log results and increment input_signal
        input_signals.append(input_signal)
        v_mem_logs[input_signal] = v_mem_log
        u_mem_logs[input_signal] = u_mem_log
        t_spans[input_signal] = t_span
        spike_trains[input_signal] = spike_train
        input_signal += 0.5

    # Function to draw subplots
    def draw_subplot(t_span,v_mem_log, idx, num_subplots, title):
        subplot_id = 100*num_subplots + 10 + idx
        plt.subplot(subplot_id)
        plt.title(title)
        plt.xlim(0, SIM_TIME)
        plt.xticks(ticks=[0, t_span.max()], labels=[0, SIM_TIME])
        plt.ylim(-90, 40)
        plt.ylabel('$V_m$')
        plt.plot(t_span, v_mem_log)
    # Draw desired membrane potential graphs
    input_vals = [1.0, 5.0, 10.0, 15.0, 20.0]
    plt.figure(figsize=(8,10))
    plt.suptitle('Regular Spiking Membrane Potential vs Time Step')
    for i, input_val in enumerate(input_vals):
        t_span = t_spans[input_val]
        v_mem_log = v_mem_logs[input_val]
        draw_subplot(t_span, v_mem_log, i+1, len(input_vals), f'I = {input_val}')
    plt.xlabel('Time Step (ms)')
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if (SHOW_PLOTS):
        plt.show()
    plt_name = str(IMG_DIR.joinpath(f'spiking_plot.png'))
    plt.savefig(f'{plt_name}')
    plt.close()

    # Calculate spike rate values
    spike_rates = []
    for input_signal in input_signals:
        spike_train = spike_trains[input_signal]
        spike_train = spike_train[int(200/TAU):]
        spike_rates.append(spike_train.sum() / 800.0)
    plt.figure()
    plt.title('Mean Spiking Rate vs Synaptic Current')
    plt.xlabel('Synaptic Current, I')
    plt.ylabel('Spiking Rate, R')
    plt.plot(input_signals, spike_rates)
    if (SHOW_PLOTS):
        plt.show()
    plt_name = str(IMG_DIR.joinpath(f'spike_rate.png'))
    plt.savefig(f'{plt_name}')
    plt.close()

