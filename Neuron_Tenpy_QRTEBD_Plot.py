import os
import numpy as np
import matplotlib.pyplot as plt

def Plot_Boson_Ocupation_And_Spin_Magnetisaion(times, occupation, magnetization, Name):
    fig, axs = plt.subplots(2, 1, figsize=(10, 7)) # axs[0] = boson occupation, axs[1] = spin magnetisation
    for i, n in enumerate(occupation.T): # .T gives one row per site
        axs[0].plot(times, n, label=f"Mode {i}")
    for i, sz in enumerate(magnetization.T):
        axs[1].plot(times, sz, label=f"Spin {i}")
    axs[0].set_ylabel(r"$\langle n \rangle$")
    axs[1].set_ylabel(r"$\langle Z \rangle$")
    axs[0].set_title(Name)
    for ax in axs: # Shared formatting for both panels
        ax.set_xlabel("Time")
        ax.legend()
    fig.tight_layout()
    return fig

Name = 'QRTEBD_S00000_N8_J0.1_T500_Bond32_Step0.1_cbe0.1'
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output', f'{Name}.npz')
with np.load(path) as data:
    times = data['times']
    occupation = data['occupation']
    magnetization = data['magnetization']
fig = Plot_Boson_Ocupation_And_Spin_Magnetisaion(times, occupation, magnetization, Name)
plt.show()
