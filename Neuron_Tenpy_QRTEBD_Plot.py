import os
import numpy as np
import matplotlib.pyplot as plt

def Load_Run(Name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output', f'{Name}.npz')
    with np.load(path) as data:
        return data['times'], data['occupation'], data['magnetization']
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
def Plot_Boson_Ocupation_And_Spin_Magnetisaion_Convergence(runs, Names):
    N_sites = max(occ.shape[1] for _, occ, _ in runs)
    fig, axs = plt.subplots(N_sites, 2, figsize=(12, 2.2 * N_sites), sharex=True, squeeze=False)
    # axs[i, 0] = boson occupation of mode i, axs[i, 1] = spin magnetisation of spin i
    colors = plt.cm.tab10.colors
    for k, ((times, occupation, magnetization), Name) in enumerate(zip(runs, Names)):
        c = colors[k % len(colors)] # Same colour for a given file on every panel
        for i in range(occupation.shape[1]):
            axs[i, 0].plot(times, occupation[:, i], color=c, label=Name)
        for i in range(magnetization.shape[1]):
            axs[i, 1].plot(times, magnetization[:, i], color=c, label=Name)
    for i in range(N_sites):
        axs[i, 0].set_ylabel(rf"Mode {i}  $\langle n \rangle$")
        axs[i, 1].set_ylabel(rf"Spin {i}  $\langle Z \rangle$")
    axs[0, 0].set_title("Boson occupation")
    axs[0, 1].set_title("Spin magnetisation")
    for ax in axs[-1]:
        ax.set_xlabel("Time")
    handles, labels = axs[0, 0].get_legend_handles_labels() # One shared legend for the figure
    fig.legend(handles, labels, loc='upper center', ncol=min(len(Names), 2), fontsize='small')
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.03 * ((len(Names) + 1) // 2) - 0.01])
    return fig

Names = ['QRTEBD_S00000_N8_J0.1_T500_Bond32_Step0.1_cbe0.1']
if isinstance(Names, str): # Allow a single name given as a bare string
    Names = [Names]

if len(Names) == 1:
    Name = Names[0]
    times, occupation, magnetization = Load_Run(Name)
    fig = Plot_Boson_Ocupation_And_Spin_Magnetisaion(times, occupation, magnetization, Name)
else:
    runs = [Load_Run(n) for n in Names]
    fig = Plot_Boson_Ocupation_And_Spin_Magnetisaion_Convergence(runs, Names)
plt.show()
