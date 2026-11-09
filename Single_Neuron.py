from qutip import sesolve, expect, destroy, fock, tensor, basis, qeye, sigmax, sigmaz
import numpy as np
import matplotlib.pyplot as plt

def States_And_Operators(Number_Of_Fock_States, Displacement_Coeficent, Spin_Boson_Interaction_Coefficent):
    Initial_State = tensor(basis(2,0), fock(Number_Of_Fock_States, 0))
    Initial_State_GHZ = tensor((basis(2,0)+basis(2,1))/np.sqrt(2), fock(Number_Of_Fock_States, 0))
    Displacement_Operator = tensor(qeye(2),destroy(Number_Of_Fock_States) + destroy(Number_Of_Fock_States).dag())
    Interaction_Operator = tensor(sigmax(), fock(Number_Of_Fock_States,Number_Of_Fock_States-1) * fock(Number_Of_Fock_States,Number_Of_Fock_States-1).dag())
    Hamiltonian = Displacement_Coeficent * Displacement_Operator + Spin_Boson_Interaction_Coefficent * Interaction_Operator
    n_op = tensor(qeye(2),destroy(Number_Of_Fock_States).dag() * destroy(Number_Of_Fock_States))
    sz_op = tensor(sigmaz(), qeye(Number_Of_Fock_States))
    return Hamiltonian, Initial_State, n_op, sz_op, Initial_State_GHZ
def Qutip_Sim(Hamiltonian, Initial_State, n_op, sz_op, Time):
    Time_List = np.linspace(0, Time, int(20*Time)+1)
    Result = sesolve(Hamiltonian, Initial_State, Time_List)
    States = Result.states
    exp_n = expect(n_op, States)
    exp_sz = expect(sz_op, States)
    return exp_n, exp_sz, Time_List
def Plot_Exp_n_and_sz(exp_n, exp_sz, Time_List):
    fig, axs = plt.subplots(2,1)
    axs[0].plot(Time_List, exp_n)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel(r"$\langle n \rangle$")
    axs[1].plot(Time_List, exp_sz)
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel(r"$\langle Z \rangle$")
    fig.tight_layout()
    return fig

Number_Of_Fock_States = 8
Displacement_Coeficent = 1
Spin_Boson_Interaction_Coefficent = Displacement_Coeficent * (Number_Of_Fock_States -0.5)**0.5
Time = 500

Hamiltonian, Initial_State, n_op, sz_op, Initial_State_GHZ = States_And_Operators(Number_Of_Fock_States, Displacement_Coeficent, Spin_Boson_Interaction_Coefficent)
exp_n, exp_sz, Time_List = Qutip_Sim(Hamiltonian, Initial_State_GHZ, n_op, sz_op, Time)
fig = Plot_Exp_n_and_sz(exp_n, exp_sz, Time_List)
plt.show()
