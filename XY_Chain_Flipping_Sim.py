from qutip import tensor, qeye, sigmax, sigmay, sigmaz, basis, sesolve, expect
import numpy as np
import matplotlib.pyplot as plt

def Initial_State_To_Operator(Initial_State):
    Initial_State_Operator = tensor([basis(2, i) for i in Initial_State])
    return Initial_State_Operator

def Pulse_Envelope(t, args):
    if t >= args["Delta_t"] and (t % args["Delta_t"]) < args["Pulse_Width"]:
        return 1.0
    return 0.0

def Hamiltonian(Initial_State, Spin_Interaction_Coefficient, Pulse_Width):
    sx_list, sy_list, sz_list = [], [], []
    for i in range(len(Initial_State)):
        op_list = [qeye(2) for _ in range(len(Initial_State))]
        op_list[i] = sigmax()
        sx_list.append(tensor(op_list))
        op_list[i] = sigmay()
        sy_list.append(tensor(op_list))
        op_list[i] = sigmaz()
        sz_list.append(tensor(op_list))

    H_XY = sum(Spin_Interaction_Coefficient * (sx_list[i] * sx_list[i + 1] + sy_list[i] * sy_list[i + 1]) / 2 for i in range(len(Initial_State) - 1))
    H_Drive = (np.pi / Pulse_Width) / 2 * sum(sx_list)
    H = [H_XY, [H_Drive, Pulse_Envelope]]

    return H, sz_list

def Qutip_Sim(Hamiltonian, Initial_State_Operator, Time, sz_list, Delta_t, Pulse_Width):
    Time_List = np.linspace(0, Time, 5000)
    # max_step stops the adaptive solver stepping over the narrow pulses
    Result = sesolve(Hamiltonian, Initial_State_Operator, Time_List,
                     args={"Delta_t": Delta_t, "Pulse_Width": Pulse_Width},
                     options={"max_step": Pulse_Width / 5})
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


Initial_State = [0,0,0,1,0,0,0]
Spin_Interaction_Coefficient = 0.005
Time = 500
Delta_t = 500
Pulse_Width = 5

Initial_State_Operator = Initial_State_To_Operator(Initial_State)
H, sz_list = Hamiltonian(Initial_State, Spin_Interaction_Coefficient, Pulse_Width)
exp_sz, Time_List = Qutip_Sim(H, Initial_State_Operator, Time, sz_list, Delta_t, Pulse_Width)
fig = Plot_Exp_sz(exp_sz, Time_List)
for t_pulse in np.arange(Delta_t, Time, Delta_t):
    fig.axes[0].axvspan(t_pulse, t_pulse + Pulse_Width, color="grey", alpha=0.3)
plt.show()
