from qutip import tensor, qeye, sigmax, sigmay, sigmaz, basis, sesolve, expect
import numpy as np
import matplotlib.pyplot as plt

def Initial_State_To_Operator(Initial_State):
    Initial_State_Operator = tensor([basis(2, i) for i in Initial_State])
    return Initial_State_Operator

def Hamiltonian(Initial_State, Spin_Interaction_Coefficient):
    sx_list, sy_list, sz_list = [], [], []
    for i in range(len(Initial_State)):
        op_list = [qeye(2) for _ in range(len(Initial_State))]
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list))
        op_list[i] = sigmay()
        sy_list.append(tensor(op_list))
        op_list[i] = sigmaz()
        sz_list.append(tensor(op_list))

    H = sum(Spin_Interaction_Coefficient * (sx_list[i] * sx_list[i + 1] + sy_list[i] * sy_list[i + 1]) / 2 for i in range(len(Initial_State) - 1))

    return H, sz_list

def Qutip_Sim(Hamiltonian, Initial_State_Operator, Time, sz_list):
    Time_List = np.linspace(0, Time, 200)
    Result = sesolve(Hamiltonian, Initial_State_Operator, Time_List)
    States = Result.states
    exp_sz = np.array([expect(op, States) for op in sz_list])
    return exp_sz, Time_List

def Plot_Exp_sz(exp_sz, Time_List):
    fig, ax = plt.subplots()
    for i, sz in enumerate(exp_sz):
        ax.plot(Time_List, sz, label=f"Spin {i}")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\langle Z \rangle$")
    return fig


Initial_State = [1,0,0]
Spin_Interaction_Coefficient = 0.01
Time = 500

Initial_State_Operator = Initial_State_To_Operator(Initial_State)
H, sz_list = Hamiltonian(Initial_State, Spin_Interaction_Coefficient)
exp_sz, Time_List = Qutip_Sim(H, Initial_State_Operator, Time, sz_list)
fig = Plot_Exp_sz(exp_sz, Time_List)
plt.show()
