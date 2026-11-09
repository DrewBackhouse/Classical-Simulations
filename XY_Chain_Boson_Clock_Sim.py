# Classical-clock version of the Neuron model (Neuron_Tenpy_QRTEBD.py).
# The spin-boson term  lambda * sx (x) |N-1><N-1|  is replaced by  lambda * P_top(t) * sx,
# where P_top(t) is the top Fock state population of a free boson evolving under g(a + a^dag).
# No spin-boson entanglement and no back-action, the boson just acts as a classical clock.
import os
from qutip import tensor, qeye, sigmax, sigmay, sigmaz, basis, sesolve, expect, destroy, fock, fock_dm, coefficient
import numpy as np
import matplotlib.pyplot as plt

def Boson_Clock(Number_Of_Fock_States, Displacement_Coefficent, Time_List):
    a = destroy(Number_Of_Fock_States)
    Result = sesolve(Displacement_Coefficent * (a + a.dag()), fock(Number_Of_Fock_States, 0), Time_List)
    States = Result.states
    Top_Fock_State_Desnity_Matrix = fock(Number_Of_Fock_States, Number_Of_Fock_States-1) * fock(Number_Of_Fock_States, Number_Of_Fock_States-1).dag()
    Number_Operator = destroy(Number_Of_Fock_States).dag() * destroy(Number_Of_Fock_States)
    P_top = np.real(expect(Top_Fock_State_Desnity_Matrix, States))
    exp_n = expect(Number_Operator, States)
    return P_top, exp_n
def Initial_State_To_Operator(Initial_State):
    return tensor([basis(2, i) for i in Initial_State])
def GHZ_State_To_Operator(Number_Of_Spins):
    All_Up = tensor([basis(2, 0) for _ in range(Number_Of_Spins)])
    All_Down = tensor([basis(2, 1) for _ in range(Number_Of_Spins)])
    return (All_Up + All_Down) / np.sqrt(2)
def Hamiltonian(Initial_State, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent, P_top, Time_List):
    sx_list, sy_list, sz_list = [], [], []
    for i in range(len(Initial_State)):
        op_list = [qeye(2) for _ in range(len(Initial_State))]
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list))
        op_list[i] = sigmay()
        sy_list.append(tensor(op_list))
        op_list[i] = sigmaz()
        sz_list.append(tensor(op_list))

    H_XY = sum(Spin_Interaction_Coefficent * (sx_list[i] * sx_list[i + 1] + sy_list[i] * sy_list[i + 1]) / 2 for i in range(len(Initial_State) - 1))
    H_Drive = Spin_Boson_Interaction_Coefficent * sum(sx_list)
    H = [H_XY, [H_Drive, coefficient(P_top, tlist=Time_List)]] # H(t) = H_{XY} + P_{top}(t)H_{Drive}
    return H, sz_list
def Qutip_Sim(H, Initial_State_Operator, Time_List, sz_list):
    Result = sesolve(H, Initial_State_Operator, Time_List, options={"max_step": 0.05}) # Resolve the narrow P_top peaks
    exp_sz = np.array([expect(op, Result.states) for op in sz_list])
    return exp_sz
def Plot_Comparison(Time_List, exp_n, exp_sz, Full_Data, Name):
    fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axs[0].plot(Time_List, exp_n)
    axs[0].set_ylabel(r"$\langle n \rangle(t)$ Isolated Boson")
    axs[0].set_title(Name)
    for i, sz in enumerate(exp_sz):
        axs[1].plot(Time_List, sz, label=f"Spin {i}")
    axs[1].set_ylabel(r"$\langle Z \rangle$ classical clock")
    if Full_Data is not None:
        for i, n in enumerate(Full_Data["occupation"].T):
            axs[2].plot(Full_Data["times"], n, label=f"Boson {i}")
    else:
        axs[2].text(0.5, 0.5, "No matching TeNPy output found", ha="center", transform=axs[2].transAxes)
    axs[2].set_ylabel(r"$\langle n \rangle$ full quantum")
    axs[2].set_xlabel("Time")
    if Full_Data is not None:
        for i, sz in enumerate(Full_Data["magnetization"].T):
            axs[3].plot(Full_Data["times"], sz, label=f"Spin {i}")
    else:
        axs[3].text(0.5, 0.5, "No matching TeNPy output found", ha="center", transform=axs[3].transAxes)
    axs[3].set_ylabel(r"$\langle Z \rangle$ full quantum")
    axs[3].set_xlabel("Time")
    for ax in axs[1:]:
        ax.legend()
    fig.tight_layout()
    return fig


Initial_State = [0, 0, 1, 0, 0]
Number_Of_Spins = 5
Number_Of_Fock_States = 8
Displacement_Coefficent = 1
Spin_Interaction_Coefficent = 0.01
Spin_Boson_Interaction_Coefficent = (np.pi/4) * Displacement_Coefficent * (Number_Of_Fock_States - 0.5)**0.5
Simulation_Time = 500
Time_List = np.linspace(0, Simulation_Time, int(20 * Simulation_Time) + 1)

Max_Bond_Dimension = 64
cbe_expand = 0.1
Timestep = 0.1

P_top, exp_n = Boson_Clock(Number_Of_Fock_States, Displacement_Coefficent, Time_List)
H, sz_list = Hamiltonian(Initial_State, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent, P_top, Time_List)
Initial_State_Vector = Initial_State_To_Operator(Initial_State)
# Initial_State_Vector = GHZ_State_To_Operator(Number_Of_Spins)
exp_sz = Qutip_Sim(H, Initial_State_Vector, Time_List, sz_list)

Full_Name = f'QRTEBD_S{"".join(map(str, Initial_State))}_N{Number_Of_Fock_States}_J{Spin_Interaction_Coefficent}_T{Simulation_Time}_Bond{Max_Bond_Dimension}_Step{Timestep}_cbe{cbe_expand}'
Full_Path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output', f'{Full_Name}.npz')
Full_Data = np.load(Full_Path) if os.path.exists(Full_Path) else None
fig = Plot_Comparison(Time_List, exp_n, exp_sz, Full_Data, f'Classical clock vs {Full_Name}')
plt.show()
