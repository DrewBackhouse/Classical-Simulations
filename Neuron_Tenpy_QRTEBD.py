# ssh k2148668@hpc.create.kcl.ac.uk
# rsync -av --exclude '__pycache__' --exclude '.DS_Store' --exclude 'Output' --exclude 'TeNPyvenv' --exclude 'logs' Tensor_Networks_TeNPy_CREATE/ k2148668@hpc.create.kcl.ac.uk:~/Tensor_Networks_TeNPy_CREATE/
# ./submit.sh --ls "4" --bonds "32 64" --steps "0.1 0.05" --mem 1G --time 6:00:00   (see submit.sh, the first run also builds TeNPyvenv with setup_venv.sh)
# Check all output logs with: grep -h "done in" logs/*.out
# rsync -av k2148668@hpc.create.kcl.ac.uk:~/Tensor_Networks_TeNPy_CREATE/Output/ Tensor_Networks_TeNPy_CREATE/Output/

import os
import argparse
import numpy as np
import tenpy.linalg.np_conserved as npc
import matplotlib.pyplot as plt
from tenpy.algorithms.tebd import QRBasedTEBDEngine
from tenpy.networks.site import Site
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Chain
from tenpy.models.model import NearestNeighborModel

def Command_Line_Parameters(Initial_State, Number_Of_Fock_States, Displacement_Coefficent, Spin_Interaction_Coefficent, Simulation_Time, Timestep, Steps_Per_Sample, Max_Bond_Dimension, cbe_expand, Tebd_Order):
    parser = argparse.ArgumentParser() # Change default values from the command line
    parser.add_argument('--init', type=str, default=''.join(map(str, Initial_State)), help='Initial qubits, 0 = up, 1 = down, one per site (e.g. 010), its length sets L')
    parser.add_argument('--N', type=int, default=Number_Of_Fock_States, help='Number of Fock states per bosonic mode')
    parser.add_argument('--J', type=float, default=Spin_Interaction_Coefficent, help='Spin-spin interaction coefficient')
    parser.add_argument('--T', type=float, default=Simulation_Time, help='Total evolution time')
    parser.add_argument('--timestep', type=float, default=Timestep)
    parser.add_argument('--steps_per_sample', type=int, default=Steps_Per_Sample, help='Trotter steps between measurements')
    parser.add_argument('--max_bond', type=int, default=Max_Bond_Dimension)
    parser.add_argument('--cbe_expand', type=float, default=cbe_expand, help='QR-TEBD bond expansion rate per step')
    parser.add_argument('--tebd_order', type=int, default=Tebd_Order, choices=[2, 4])
    args = parser.parse_args()
    if set(args.init) - {'0', '1'} or len(args.init) < 2:
        parser.error(f"--init must be a string of at least two 0s and 1s (TEBD needs a bond), got '{args.init}'")

    Initial_State = [int(s) for s in args.init]
    Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (args.N - 0.5)**0.5 # Recomputed as it depends on N
    return Initial_State, args.N, args.J, Spin_Boson_Interaction_Coefficent, args.T, args.timestep, args.steps_per_sample, args.max_bond, args.cbe_expand, args.tebd_order
def Initial_State_To_MPS(Initial_State, Number_Of_Fock_States):
    site = Site(npc.LegCharge.from_trivial(2 * Number_Of_Fock_States))
    states = [np.kron(np.eye(2)[s], np.eye(Number_Of_Fock_States)[0]) for s in Initial_State]
    Initial_MPS = MPS.from_product_state([site] * len(Initial_State), states, bc="finite")
    return Initial_MPS
def QR_Based_TEBD_Hamiltonian(Initial_State, Number_Of_Fock_States, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent):

    a = np.diag(np.sqrt(np.arange(1, Number_Of_Fock_States)), 1)
    Displacement_operator = np.kron(np.eye(2), a + a.T)
    Pauli_X = np.array([[0, 1], [1, 0]])
    Pauli_Y = np.array([[0, -1j], [1j, 0]])
    Pauli_X_Operator = np.kron(Pauli_X, np.eye(Number_Of_Fock_States))
    Pauli_Y_Operator = np.kron(Pauli_Y, np.eye(Number_Of_Fock_States))
    Projection_Operator = np.kron(Pauli_X, np.diag(np.eye(Number_Of_Fock_States)[Number_Of_Fock_States - 1]))

    H1 = Displacement_Coefficent * Displacement_operator + Spin_Boson_Interaction_Coefficent * Projection_Operator

    site = Site(npc.LegCharge.from_trivial(2 * Number_Of_Fock_States))
    lattice = Chain(len(Initial_State), site, bc="open", bc_MPS="finite")

    H_bond = [None] # H_bond[i] acts on sites (i-1, i), none for i = 0 on a finite chain
    for i in range(1, len(Initial_State)):
        H = Spin_Interaction_Coefficent / 2 * (np.kron(Pauli_X_Operator, Pauli_X_Operator) + np.kron(Pauli_Y_Operator, Pauli_Y_Operator)) + np.kron(np.eye(2 * Number_Of_Fock_States), H1) # Site i's H1 goes on the bond to its left
        if i == 1:
            H = H + np.kron(H1, np.eye(2 * Number_Of_Fock_States)) # Site 0 has no bond to its left
        H = H.reshape(2 * Number_Of_Fock_States, 2 * Number_Of_Fock_States, 2 * Number_Of_Fock_States, 2 * Number_Of_Fock_States).transpose(0, 2, 1, 3) # (p0, p1, p0*, p1*) -> (p0, p0*, p1, p1*)
        H_bond.append(npc.Array.from_ndarray(H, [site.leg, site.leg.conj(), site.leg, site.leg.conj()], labels=['p0', 'p0*', 'p1', 'p1*']))

    Hamiltonian = NearestNeighborModel(lattice, H_bond)
    return Hamiltonian
def QR_Based_TEBD_Sim(Initial_MPS, Hamiltonian, Number_Of_Fock_States, Simulation_Time, Timestep, Steps_Per_Sample, Max_Bond_Dimension, cbe_expand, Tebd_Order):
    psi = Initial_MPS.copy() # evolve works in place, keep the input state untouched
    leg = psi.sites[0].leg
    Number_Operator = npc.Array.from_ndarray(np.kron(np.eye(2), np.diag(np.arange(Number_Of_Fock_States))).astype(complex), [leg, leg.conj()], labels=['p', 'p*'])
    Pauli_Z_Operator = npc.Array.from_ndarray(np.kron(np.diag([1, -1]), np.eye(Number_Of_Fock_States)).astype(complex), [leg, leg.conj()], labels=['p', 'p*'])

    engine = QRBasedTEBDEngine(psi, Hamiltonian, dict(trunc_params=dict(chi_max=Max_Bond_Dimension, svd_min=1e-10), cbe_expand=cbe_expand))
    engine.calc_U(Tebd_Order, Timestep) # Builds the Trotter gates exp(-i H_bond dt), must come before evolve

    times = np.arange(round(Simulation_Time / (Steps_Per_Sample * Timestep)) + 1) * Steps_Per_Sample * Timestep
    occupation = np.zeros((len(times), psi.L))
    magnetization = np.zeros((len(times), psi.L))
    for it in range(len(times)):
        if it > 0:
            engine.evolve(Steps_Per_Sample, Timestep) # One Trotter step of Timestep
        occupation[it] = np.real(psi.expectation_value(Number_Operator))
        magnetization[it] = np.real(psi.expectation_value(Pauli_Z_Operator))

    return times, occupation, magnetization
def Plot_Boson_Ocupation_And_Spin_Magnetisaion(times, occupation, magnetization):
    fig, axs = plt.subplots(2, 1, figsize=(10, 7)) # axs[0] = boson occupation, axs[1] = spin magnetisation
    for i, n in enumerate(occupation.T): # .T gives one row per site
        axs[0].plot(times, n, label=f"Mode {i}")
    for i, sz in enumerate(magnetization.T):
        axs[1].plot(times, sz, label=f"Spin {i}")
    axs[0].set_ylabel(r"$\langle n \rangle$")
    axs[1].set_ylabel(r"$\langle Z \rangle$")
    for ax in axs: # Shared formatting for both panels
        ax.set_xlabel("Time")
        ax.legend()
    fig.tight_layout()
    return fig


Initial_State = [0, 0, 0, 0, 0]
Number_Of_Fock_States = 4
Displacement_Coefficent = 1
Spin_Interaction_Coefficent = 0.1
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_Of_Fock_States -0.5)**0.5

Simulation_Time = 100
Timestep = 0.1
Steps_Per_Sample = 1
Max_Bond_Dimension = 32
cbe_expand = 0.1
Tebd_Order = 2

Initial_State, Number_Of_Fock_States, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent, Simulation_Time, Timestep, Steps_Per_Sample, Max_Bond_Dimension, cbe_expand, Tebd_Order = Command_Line_Parameters(Initial_State, Number_Of_Fock_States, Displacement_Coefficent, Spin_Interaction_Coefficent, Simulation_Time, Timestep, Steps_Per_Sample, Max_Bond_Dimension, cbe_expand, Tebd_Order)

Initial_MPS = Initial_State_To_MPS(Initial_State, Number_Of_Fock_States)
Hamiltonian = QR_Based_TEBD_Hamiltonian(Initial_State, Number_Of_Fock_States, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
times, occupation, magnetization = QR_Based_TEBD_Sim(Initial_MPS, Hamiltonian, Number_Of_Fock_States, Simulation_Time, Timestep, Steps_Per_Sample, Max_Bond_Dimension, cbe_expand, Tebd_Order)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output', f'QRTEBD_S{"".join(map(str, Initial_State))}_N{Number_Of_Fock_States}_J{Spin_Interaction_Coefficent}_T{Simulation_Time}_Bond{Max_Bond_Dimension}_Step{Timestep}_cbe{cbe_expand}.npz'), times=times, occupation=occupation, magnetization=magnetization) # Next to this script, not the working directory
fig = Plot_Boson_Ocupation_And_Spin_Magnetisaion(times, occupation, magnetization)
plt.show()
