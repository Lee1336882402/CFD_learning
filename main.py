import numpy as np
from source import lwe_FDM
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

def init_fuc(zero_array):
    M_x = zero_array.shape[0]
    result = np.zeros_like(zero_array)
    for i in range(M_x):
        if i/(M_x-1)<0.25:
            result[i]=0
        elif i/(M_x-1)<0.75:
            result[i]=1
        else:
            result[i]=0
    return result

if __name__ == '__main__':
    solver = lwe_FDM(init_fuc,method='Lax_Wendroff')
    #solver.forward(100)
    #step_interval = int(10/solver.delta_t/40)
    #solver.gif_create(step_interval)
    plot_time = [0.1,1,10]
    solver.picture(plot_time)