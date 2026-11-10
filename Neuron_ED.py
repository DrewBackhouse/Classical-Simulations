from qutip import sesolve, destroy, fock, tensor, basis, qeye, sigmax, sigmay, sigmaz, coherent
import numpy as np
import matplotlib.pyplot as plt

def GHZ_Spin_State(Number_Of_Bosonic_Modes):
    up = tensor([basis(2,0) for _ in range(Number_Of_Bosonic_Modes)])
    down = tensor([basis(2,1) for _ in range(Number_Of_Bosonic_Modes)])
    State = (up + down)/np.sqrt(2)
    return State
def Coherent_Bosonic_State(Number_Of_Fock_States, Initial_Coherent_Bosonic_State):
    State = tensor([coherent(Number_Of_Fock_States, alpha) for alpha in Initial_Coherent_Bosonic_State])
    return State
def States_And_Operators(Initial_Spin_State, Initial_Bosonic_State, Number_Of_Bosonic_Modes, Number_Of_Fock_States, Displacement_Coeficent, Spin_Interaction_Coefficient, Spin_Boson_Interaction_Coefficent):
    Initial_State = tensor(Initial_Spin_State, Initial_Bosonic_State)

    Bosonic_Identity_List = [qeye(Number_Of_Fock_States) for _ in range(Number_Of_Bosonic_Modes)]
    Spin_Identity_List = [qeye(2) for _ in range(Number_Of_Bosonic_Modes)]
    
    Bosonic_Displacement_Operator = destroy(Number_Of_Fock_States) + destroy(Number_Of_Fock_States).dag()
    Bosonic_Displacement_Operator_List = [] 
    for i in range(Number_Of_Bosonic_Modes):
        Operator_List = Bosonic_Identity_List.copy()
        Operator_List[i] = Bosonic_Displacement_Operator
        Bosonic_Displacement_Operator_List.append(tensor(Spin_Identity_List + Operator_List))
    Bosonic_Displacement_Hamiltonian = Displacement_Coeficent * sum(Bosonic_Displacement_Operator_List)
    
    sx_list, sy_list, sz_list = [], [], []
    for i in range(Number_Of_Bosonic_Modes):
        op_list = Spin_Identity_List.copy()
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list + Bosonic_Identity_List))
        op_list[i] = sigmay()
        sy_list.append(tensor(op_list + Bosonic_Identity_List))
        op_list[i] = sigmaz()
        sz_list.append(tensor(op_list + Bosonic_Identity_List))
    Spin_Interaction_Hamiltonian = Spin_Interaction_Coefficient * sum(sx_list[i] * sx_list[i + 1] + sy_list[i] * sy_list[i + 1] for i in range(Number_Of_Bosonic_Modes - 1))/2
    
    Bosonic_Projection_Operator = fock(Number_Of_Fock_States, Number_Of_Fock_States-1) * fock(Number_Of_Fock_States, Number_Of_Fock_States-1).dag()
    Interaction_Operator_List = []
    for i in range(Number_Of_Bosonic_Modes):
        Spin_Operator_List = Spin_Identity_List.copy()
        Bosonic_Operator_List = Bosonic_Identity_List.copy()
        Spin_Operator_List[i] = sigmax()
        Bosonic_Operator_List[i] = Bosonic_Projection_Operator
        Interaction_Operator_List.append(tensor(Spin_Operator_List + Bosonic_Operator_List))
    Interaction_Hamiltonian = Spin_Boson_Interaction_Coefficent * sum(Interaction_Operator_List)

    Hamiltonian = Bosonic_Displacement_Hamiltonian + Spin_Interaction_Hamiltonian + Interaction_Hamiltonian

    Bosonic_Number_Operator = destroy(Number_Of_Fock_States).dag() * destroy(Number_Of_Fock_States)
    n_list = []
    for i in range(Number_Of_Bosonic_Modes):
        Operator_List = Bosonic_Identity_List.copy()
        Operator_List[i] = Bosonic_Number_Operator
        n_list.append(tensor(Spin_Identity_List + Operator_List))

    return Initial_State, Hamiltonian, sz_list, n_list
def Qutip_Sim(Hamiltonian, Initial_State, n_list, sz_list, Time):
    Time_List = np.linspace(0, Time, int(20*Time)+1)
    Result = sesolve(Hamiltonian, Initial_State, Time_List, e_ops = sz_list + n_list)
    N= len(sz_list)
    exp_sz = np.array(Result.expect[:N])
    exp_n = np.array(Result.expect[N:])
    return exp_n, exp_sz, Time_List
def Plot_Exp_n_and_sz(exp_n, exp_sz, Time_List):
    fig, axs = plt.subplots(2,1)
    for i, n in enumerate(exp_n):
        axs[0].plot(Time_List, n, label=f"Boson {i}")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel(r"$\langle n \rangle$")
    axs[0].legend()
    for i, sz in enumerate(exp_sz):
        axs[1].plot(Time_List, sz, label=f"Spin {i}")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel(r"$\langle Z \rangle$")
    axs[1].legend()
    fig.tight_layout()
    return fig


Number_Of_Fock_States = 8
Number_Of_Bosonic_Modes = 4
Initial_Coherent_Bosonic_State = [0, 0, 0, 0]
Displacement_Coeficent = 1
Spin_Interaction_Coefficient = 0.1
Spin_Boson_Interaction_Coefficent = Displacement_Coeficent * (Number_Of_Fock_States -0.5)**0.5
Time = 300

Initial_Spin_State = GHZ_Spin_State(Number_Of_Bosonic_Modes)
Initial_Bosonic_State = Coherent_Bosonic_State(Number_Of_Fock_States, Initial_Coherent_Bosonic_State)
Initial_State, Hamiltonian, sz_list, n_list = States_And_Operators(Initial_Spin_State, Initial_Bosonic_State, Number_Of_Bosonic_Modes, Number_Of_Fock_States, Displacement_Coeficent, Spin_Interaction_Coefficient, Spin_Boson_Interaction_Coefficent)
exp_n, exp_sz, Time_List = Qutip_Sim(Hamiltonian, Initial_State, n_list, sz_list, Time)
fig = Plot_Exp_n_and_sz(exp_n, exp_sz, Time_List)
plt.show()
