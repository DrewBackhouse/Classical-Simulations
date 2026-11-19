import os
from scipy.special import comb
from qutip import tensor, qeye, sigmax, sigmay, sigmaz, basis, sesolve, expect, destroy, fock, fock_dm, coefficient
import numpy as np
import matplotlib.pyplot as plt

def Boson_Clock(Number_Of_Fock_States, Time_List):
    Bosonic_Hop = sum(np.sqrt((n+1) * (2 * Number_Of_Fock_States - n - 1)) * basis(Number_Of_Fock_States, n+1) * basis(Number_Of_Fock_States, n).dag() for n in range(Number_Of_Fock_States-1))
    Hamiltonian = Bosonic_Hop + Bosonic_Hop.dag() + Number_Of_Fock_States * fock(Number_Of_Fock_States, Number_Of_Fock_States-1)* fock(Number_Of_Fock_States, Number_Of_Fock_States-1).dag()
    Result = sesolve(Hamiltonian, fock(Number_Of_Fock_States, 0), Time_List)
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
def Plot(Time_List, exp_n, exp_sz):
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axs[0].plot(Time_List, exp_n)
    axs[0].set_ylabel(r"$\langle n \rangle(t)$ Isolated Boson")
    axs[0].set_title(f"L{Number_Of_Spins}_N{Number_Of_Fock_States}_J{Spin_Interaction_Coefficent}")
    for i, sz in enumerate(exp_sz):
        axs[1].plot(Time_List, sz, label=f"Spin {i}")
    axs[1].set_ylabel(r"$\langle Z \rangle$ classical clock")
    axs[1].legend()
    
    fig.tight_layout()
    return fig


Initial_State = [0, 0, 1, 0,0]
Number_Of_Spins = 5
Number_Of_Fock_States = 5
Spin_Interaction_Coefficent = 0.1
A = np.pi/2 * comb(2*Number_Of_Fock_States-1, Number_Of_Fock_States) * comb(2*Number_Of_Fock_States-2, Number_Of_Fock_States-1) / 4.0**(2*Number_Of_Fock_States-2)
Spin_Boson_Interaction_Coefficent = np.pi / (2*A)
Simulation_Time = 50
Time_List = np.linspace(0, Simulation_Time, int(20 * Simulation_Time) + 1)

P_top, exp_n = Boson_Clock(Number_Of_Fock_States, Time_List)
H, sz_list = Hamiltonian(Initial_State, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent, P_top, Time_List)
Initial_State_Vector = Initial_State_To_Operator(Initial_State)
# Initial_State_Vector = GHZ_State_To_Operator(Number_Of_Spins)
exp_sz = Qutip_Sim(H, Initial_State_Vector, Time_List, sz_list)
fig = Plot(Time_List, exp_n, exp_sz)
plt.show()
