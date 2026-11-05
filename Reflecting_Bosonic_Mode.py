from qutip import sesolve, expect, destroy, fock
import numpy as np
import matplotlib.pyplot as plt

def States_And_Operators(Number_Of_Fock_States, Displacement_Coeficent):
    Initial_State = fock(Number_Of_Fock_States, 0)
    Hamiltonian = Displacement_Coeficent * (destroy(Number_Of_Fock_States) + destroy(Number_Of_Fock_States).dag())
    n_op = destroy(Number_Of_Fock_States).dag() * destroy(Number_Of_Fock_States)
    return Hamiltonian, Initial_State, n_op
def Qutip_Sim(Hamiltonian, Initial_State, n_op, Time):
    Time_List = np.linspace(0, Time, int(20*Time)+1)
    Result = sesolve(Hamiltonian, Initial_State, Time_List)
    States = Result.states
    exp_n = expect(n_op, States)
    return exp_n, Time_List
def Plot_Exp_n(exp_n, Time_List):
    fig, ax = plt.subplots()
    ax.plot(Time_List, exp_n)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\langle n \rangle$")
    return fig

Number_Of_Fock_States = 20
Displacement_Coeficent = 1
Time = 500

Hamiltonian, Initial_State, n_op = States_And_Operators(Number_Of_Fock_States, Displacement_Coeficent)
exp_n, Time_List = Qutip_Sim(Hamiltonian, Initial_State, n_op, Time)
fig = Plot_Exp_n(exp_n, Time_List)
plt.show()
