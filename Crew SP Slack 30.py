import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import pennylane as qml

QT_10 = np.array([
[-78,  40,  30,  50,  20,   0,   0, -20, -40,  20,  40],
[  0, -76,  30,  40,  30,   0,   0, -20, -40,  20,  40],
[  0,   0, -76,  40,  40, -20, -40,   0,   0,  20,  40],
[  0,   0,   0, -77,  40, -20, -40, -20, -40,  20,  40],
[  0,   0,   0,   0, -68, -20, -40,   0,   0,  20,  40],
[  0,   0,   0,   0,   0,  30,  40,   0,   0,   0,   0],
[  0,   0,   0,   0,   0,   0,  80,   0,   0,   0,   0],
[  0,   0,   0,   0,   0,   0,   0,  30,  40,   0,   0],
[  0,   0,   0,   0,   0,   0,   0,   0,  80,   0,   0],
[  0,   0,   0,   0,   0,   0,   0,   0,   0, -50,  40],
[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, -80]])
offset_10 = 150

QT_5 = np.array([
[-38,  20,  15,  25,  10,   0,   0, -10, -20,  10,  20],
[  0, -36,  15,  20,  15,   0,   0, -10, -20,  10,  20],
[  0,   0, -36,  20,  20, -10, -20,   0,   0,  10,  20],
[  0,   0,   0, -37,  20, -10, -20, -10, -20,  10,  20],
[  0,   0,   0,   0, -33, -10, -20,   0,   0,  10,  20],
[  0,   0,   0,   0,   0,  15,  20,   0,   0,   0,   0],
[  0,   0,   0,   0,   0,   0,  40,   0,   0,   0,   0],
[  0,   0,   0,   0,   0,   0,   0,  15,  20,   0,   0],
[  0,   0,   0,   0,   0,   0,   0,   0,  40,   0,   0],
[  0,   0,   0,   0,   0,   0,   0,   0,   0, -25,  20],
[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, -40]])
offset_5 = 75

QT_20 = np.array([
[-158,   80,   60,  100,   40,   0,   0, -40, -80,   40,   80],
[   0, -156,   60,   80,   60,   0,   0, -40, -80,   40,   80],
[   0,    0, -156,   80,   80, -40, -80,   0,   0,   40,   80],
[   0,    0,    0, -157,   80, -40, -80, -40, -80,   40,   80],
[   0,    0,    0,    0, -138, -40, -80,   0,   0,   40,   80],
[   0,    0,    0,    0,    0,  60,  80,   0,   0,    0,    0],
[   0,    0,    0,    0,    0,   0, 160,   0,   0,    0,    0],
[   0,    0,    0,    0,    0,   0,   0,  60,  80,    0,    0],
[   0,    0,    0,    0,    0,   0,   0,   0, 160,    0,    0],
[   0,    0,    0,    0,    0,   0,   0,   0,   0, -100,   80],
[   0,    0,    0,    0,    0,   0,   0,   0,   0,    0, -160]])
offset_20 = 300

offset = offset_20
QT = QT_20

n_qubits = len(QT)

# -----------------------------   QAOA circuit ------------------------------------
shots = 5000  # Number of samples used
dev = qml.device("default.qubit", shots=shots)


@qml.qnode(dev)
def qaoa_circuit(gammas, betas, h, J, num_qubits):
    wmax = max(
        np.max(np.abs(list(h.values()))), np.max(np.abs(list(h.values())))
    )  # Normalizing the Hamiltonian is a good idea
    p = len(gammas)
    # Apply the initial layer of Hadamard gates to all qubits
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    # repeat p layers the circuit shown in Fig. 1
    for layer in range(p):
        # ---------- COST HAMILTONIAN ----------
        for ki, v in h.items():  # single-qubit terms
            qml.RZ(2 * gammas[layer] * v / wmax, wires=ki[0])
        for kij, vij in J.items():  # two-qubit terms
            qml.CNOT(wires=[kij[0], kij[1]])
            qml.RZ(2 * gammas[layer] * vij / wmax, wires=kij[1])
            qml.CNOT(wires=[kij[0], kij[1]])
        # ---------- MIXER HAMILTONIAN ----------
        for i in range(num_qubits):
            qml.RX(-2 * betas[layer], wires=i)
    return qml.sample()


def samples_dict(samples, n_items):
    """Just sorting the outputs in a dictionary"""
    results = defaultdict(int)
    for sample in samples:
        print(f"Inside samples_dict {sample}")
        results["".join(str(i) for i in sample)[:n_items]] += 1
    return results

betas = np.linspace(0, 1, 15)[::-1]  # Parameters for the mixer Hamiltonian
gammas = np.linspace(0, 1, 15)  # Parameters for the cost Hamiltonian (Our Knapsack problem)

def from_Q_to_Ising(Q, offset):
    """Convert the matrix Q of Eq.3 into Eq.13 elements J and h"""
    n_qubits = len(Q)  # Get the number of qubits (variables) in the QUBO matrix
    # Create default dictionaries to store h and pairwise interactions J
    h = defaultdict(int)
    J = defaultdict(int)

    # Loop over each qubit (variable) in the QUBO matrix
    for i in range(n_qubits):
        # Update the magnetic field for qubit i based on its diagonal element in Q
        h[(i,)] -= Q[i, i] / 2
        # Update the offset based on the diagonal element in Q
        offset += Q[i, i] / 2
        # Loop over other qubits (variables) to calculate pairwise interactions
        for j in range(i + 1, n_qubits):
            # Update the pairwise interaction strength (J) between qubits i and j
            J[(i, j)] += Q[i, j] / 4
            # Update the magnetic fields for qubits i and j based on their interactions in Q
            h[(i,)] -= Q[i, j] / 4
            h[(j,)] -= Q[i, j] / 4
            # Update the offset based on the interaction strength between qubits i and j
            offset += Q[i, j] / 4
    # Return the magnetic fields, pairwise interactions, and the updated offset
    return h, J, offset


def energy_Ising(z, h, J, offset):
    """
    Calculate the energy of an Ising model given spin configurations.

    Parameters:
    - z: A dictionary representing the spin configurations for each qubit.
    - h: A dictionary representing the magnetic fields for each qubit.
    - J: A dictionary representing the pairwise interactions between qubits.
    - offset: An offset value.

    Returns:
    - energy: The total energy of the Ising model.
    """
    if isinstance(z, str):
        z = [(1 if int(i) == 0 else -1) for i in z]
    # Initialize the energy with the offset term
    energy = offset
    # Loop over the magnetic fields (h) for each qubit and update the energy
    for k, v in h.items():
        energy += v * z[k[0]]
    # Loop over the pairwise interactions (J) between qubits and update the energy
    for k, v in J.items():
        energy += v * z[k[0]] * z[k[1]]
    # Return the total energy of the Ising model
    return energy

h, J, zoffset = from_Q_to_Ising(QT, offset)  # Eq.13 for our problem

def run_qaoa():
    samples_slack = samples_dict(
        qaoa_circuit(gammas, betas, h, J, num_qubits=len(QT)), n_qubits
    )
    return samples_slack

# Guardar tiempos de ejecución
execution_times = []
num_factibles = []
num_opt = []

for i in range(30):
    start_time = time.time()
    print(f"\n--- Corrida {i + 1} ---")
    samples = run_qaoa()
    end_time = time.time()

    elapsed_time = end_time - start_time
    execution_times.append(elapsed_time)

    # Checar si son factibles

    mat = np.array([
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1],
    [1, 1, 0, 1, 0],
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 0],
    [-1, -1, -1, -1, -1]])

    b = np.array([1, 1, 1, 1, 1, 1, -3])

    # Contar soluciones factibles
    conteo = 0
    conteo_opt = 0

    # Calcular los resultados como los primeros 5 valores de la solución multiplicados por la matriz para el escalar
    for clave, veces in samples.items():
        # Extraer las soluciones como vector
        vector = [int(bit) for bit in clave[:5]]
        vector_completo = [int(bit) for bit in clave]

        # Convertir en un vector numpy
        vector= np.array(vector)
        vector_completo = np.array(vector_completo)

        # Verificar factibilidad con la matriz de restricciones
        if np.all(np.dot(mat, vector) >= b):
            conteo += veces

            if (offset + vector_completo @ QT @ vector_completo) == 8:
              conteo_opt += veces

    num_factibles.append(conteo)
    num_opt.append(conteo_opt)

plt.plot(execution_times, marker='o')
plt.xlabel('Corrida')
plt.ylabel('Tiempo (s)')
plt.title('Tiempos de ejecución de QAOA')
plt.grid(True)
#plt.show()

plt.plot(num_factibles, marker='o')
plt.xlabel('Corrida')
plt.ylabel('Soluciones factibles')
plt.title('Número de soluciones factibles por corrida')
plt.grid(True)
#plt.show()

plt.plot(num_opt, marker='o')
plt.xlabel('Corrida')
plt.ylabel('Soluciones óptimas')
plt.title('Número de soluciones óptimas por corrida')
plt.grid(True)
#plt.show()

print(execution_times)
print(num_factibles)
print(num_opt)