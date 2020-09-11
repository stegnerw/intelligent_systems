###############################################################################
# Imports
###############################################################################
import math
import numpy as np
from matplotlib import pyplot as plt


class SpikingNeuron:
    def __init__(
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
        self.u_mem = self.b * v_mem

    def step(self, I, tau = 0.25):
        v_mem = v_mem + tau * (0.04 * v_mem**2 + 5 * v_mem + 140 - u_mem + I)
        u_mem = u_mem + tau * a * (b * v_mem - u_mem)

    def get_v_mem(self):
        return self.v_mem

    def get_u_mem(self):
        return self.u_mem


if __name__ == '__main__':
    pass
###############################################################################
# Plotting
###############################################################################
# subplot(2,1,1)
# plot(t_span,v_mem_arr)
# axis([0 max(t_span) -90 40])
# xlabel('time step')
# ylabel('v_mem_m')
# xticks([0 max(t_span)])
# xticklabels([0 time_steps])
# title('Regular Spiking')

# subplot(2,1,2)
# plot(t_span,spike_ts,'r')
# axis([0 max(t_span) 0 1.5])
# xlabel('time step')
# xticks([0 max(t_span)])
# xticklabels([0 time_steps])
# yticks([0 1])
