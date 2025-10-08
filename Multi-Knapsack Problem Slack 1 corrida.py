import numpy as np
from collections import defaultdict
import pennylane as qml
import matplotlib.pyplot as plt

QT_10 = np.array([
[-2553,  1200,  1020,  1200,   960,   60,  120,  240,  120,  240,   480,   960],
[    0, -3204,  1360,  1600,  1280,   80,  160,  320,  160,  320,   640,  1280],
[    0,     0, -2825,  1380,  1080,   60,  120,  240,  140,  280,   560,  1120],
[    0,     0,     0, -3154,  1240,   40,   80,  160,  180,  360,   720,  1440],
[    0,     0,     0,     0, -2683,   80,  160,  320,  120,  240,   480,   960],
[    0,     0,     0,     0,     0, -190,   40,   80,    0,    0,     0,     0],
[    0,     0,     0,     0,     0,    0, -360,  160,    0,    0,     0,     0],
[    0,     0,     0,     0,     0,    0,    0, -640,    0,    0,     0,     0],
[    0,     0,     0,     0,     0,    0,    0,    0, -390,   40,    80,   160],
[    0,     0,     0,     0,     0,    0,    0,    0,    0, -760,   160,   320],
[    0,     0,     0,     0,     0,    0,    0,    0,    0,    0, -1440,   640],
[    0,     0,     0,     0,     0,    0,    0,    0,    0,    0,     0, -2560]])
offset_10 = 5000

QT_5 = np.array([
 [-1278,   600,   510,   600,   480,  30,   60,  120,   60,  120,  240,   480],
 [    0, -1604,   680,   800,   640,  40,   80,  160,   80,  160,  320,   640],
 [    0,     0, -1415,   690,   540,  30,   60,  120,   70,  140,  280,   560],
 [    0,     0,     0, -1579,   620,  20,   40,   80,   90,  180,  360,   720],
 [    0,     0,     0,     0, -1343,  40,   80,  160,   60,  120,  240,   480],
 [    0,     0,     0,     0,     0, -95,   20,   40,    0,    0,    0,     0],
 [    0,     0,     0,     0,     0,   0, -180,   80,    0,    0,    0,     0],
 [    0,     0,     0,     0,     0,   0,    0, -320,    0,    0,    0,     0],
 [    0,     0,     0,     0,     0,   0,    0,    0, -195,   20,   40,    80],
 [    0,     0,     0,     0,     0,   0,    0,    0,    0, -380,   80,   160],
 [    0,     0,     0,     0,     0,   0,    0,    0,    0,    0, -720,   320],
 [    0,     0,     0,     0,     0,   0,    0,    0,    0,    0,    0, -1280]])
offset_5 = 2500

QT_20 = np.array([
[-5103,  2400,  2040,  2400,  1920,  120,  240,   480,  240,   480,   960,  1920],
[    0, -6404,  2720,  3200,  2560,  160,  320,   640,  320,   640,  1280,  2560],
[    0,     0, -5645,  2760,  2160,  120,  240,   480,  280,   560,  1120,  2240],
[    0,     0,     0, -6304,  2480,   80,  160,   320,  360,   720,  1440,  2880],
[    0,     0,     0,     0, -5363,  160,  320,   640,  240,   480,   960,  1920],
[    0,     0,     0,     0,     0, -380,   80,   160,    0,     0,     0,     0],
[    0,     0,     0,     0,     0,    0, -720,   320,    0,     0,     0,     0],
[    0,     0,     0,     0,     0,    0,    0, -1280,    0,     0,     0,     0],
[    0,     0,     0,     0,     0,    0,    0,     0, -780,    80,   160,   320],
[    0,     0,     0,     0,     0,    0,    0,     0,    0, -1520,   320,   640],
[    0,     0,     0,     0,     0,    0,    0,     0,    0,     0, -2880,  1280],
[    0,     0,     0,     0,     0,    0,    0,     0,    0,     0,     0, -5120]])
offset_20 = 10000

QT_15 = np.array([
 [-3828,  1800,  1530,  1800,  1440,   90,  180,  360,  180,   360,   720,  1440],
 [    0, -4804,  2040,  2400,  1920,  120,  240,  480,  240,   480,   960,  1920],
 [    0,     0, -4235,  2070,  1620,   90,  180,  360,  210,   420,   840,  1680],
 [    0,     0,     0, -4729,  1860,   60,  120,  240,  270,   540,  1080,  2160],
 [    0,     0,     0,     0, -4023,  120,  240,  480,  180,   360,   720,  1440],
 [    0,     0,     0,     0,     0, -285,   60,  120,    0,     0,     0,     0],
 [    0,     0,     0,     0,     0,    0, -540,  240,    0,     0,     0,     0],
 [    0,     0,     0,     0,     0,    0,    0, -960,    0,     0,     0,     0],
 [    0,     0,     0,     0,     0,    0,    0,    0, -585,    60,   120,   240],
 [    0,     0,     0,     0,     0,    0,    0,    0,    0, -1140,   240,   480],
 [    0,     0,     0,     0,     0,    0,    0,    0,    0,     0, -2160,   960],
 [    0,     0,     0,     0,     0,    0,    0,    0,    0,     0,     0, -3840]])
offset_15 = 7500

QT = QT_20
offset = offset_20

n_qubits = len(QT)

# -----------------------------   QAOA circuit ------------------------------------

shots = 10000  # Number of samples used
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

betas = np.linspace(0, 1, 20)[::-1]  # Parameters for the mixer Hamiltonian
gammas = np.linspace(0, 1, 20)  # Parameters for the cost Hamiltonian (Our Knapsack problem)

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

samples_slack = samples_dict(qaoa_circuit(gammas, betas, h, J, num_qubits=len(QT)), n_qubits) # The solutions are generated here

keys = list(samples_slack.keys())

print(samples_slack)
#for key in keys:
 #   solution = list(map(int,key))
  #  print(solution)

# Checar si son factibles

mat = np.array([
[3, 4, 3, 2, 4],
[6, 8, 7, 9, 6]])

b = np.array([10, 20])

# Listas para almacenar resultados y repeticiones
resultados = []
repeticiones = []
factibles = []

# Calcular los resultados como los primeros 5 valores de la solución multiplicados por la matriz para el escalar
for clave, veces in samples_slack.items():
    # Extraer los primeros 5 bits de la clave (vector corto)
    vector_corto = [int(bit) for bit in clave[:5]]  # Usamos solo los primeros 5 bits

    # Convertir en un vector numpy
    vector_corto = np.array(vector_corto)

    # Verificar factibilidad con la matriz de restricciones
    factible = np.all(np.dot(mat, vector_corto) <= b)  # Comparación con b_restricciones

    # Extraer el vector completo de la solución (toda la cadena binaria)
    vector_completo = [int(bit) for bit in clave]  # Usamos todo el vector

    # Convertir en un vector numpy
    vector_completo = np.array(vector_completo)

    # Multiplicar por la matriz para el cálculo del escalar
    resultado = vector_completo @ QT @ vector_completo

    # Guardar el resultado escalar, las repeticiones y si es factible o no
    resultados.append(resultado)
    repeticiones.append(veces)
    factibles.append(factible)

# Recorrer las listas de atrás hacia adelante y eliminar los elementos con repetición 1
for i in range(len(repeticiones) - 1, -1, -1):
    if repeticiones[i] <= 3:
        del resultados[i]
        del repeticiones[i]
        del factibles[i]

# Diccionarios para agrupar repeticiones separadas por factibles e infactibles
factibles_agrupados = defaultdict(int)
infactibles_agrupados = defaultdict(int)

# Agrupar las repeticiones sumando factibles e infactibles por cada resultado
for resultado, rep, factible in zip(resultados, repeticiones, factibles):
    if factible:
        factibles_agrupados[resultado] += rep  # Sumar solo factibles
    else:
        infactibles_agrupados[resultado] += rep  # Sumar solo infactibles

# Obtener los valores únicos de resultados ordenados
resultados_unicos = sorted(set(resultados))  # Ordenamos para mantener el eje X limpio

# Obtener las repeticiones de cada tipo (si no existe el valor, poner 0)
repeticiones_factibles = [factibles_agrupados[r] for r in resultados_unicos]
repeticiones_infactibles = [infactibles_agrupados[r] for r in resultados_unicos]

# Graficar el histograma apilado
plt.figure(figsize=(30, 6))

plt.bar(
    [str(r) for r in resultados_unicos], repeticiones_factibles,
    color='green', label="Factibles"
)

plt.bar(
    [str(r) for r in resultados_unicos], repeticiones_infactibles,
    color='red', bottom=repeticiones_factibles, label="Infactibles"
)

# Etiquetas y formato
plt.xlabel('Resultado (Escalar)')
plt.ylabel('Repeticiones')
plt.title('Histograma de Resultados Factibles vs Infactibles')
plt.xticks(rotation=90)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar gráfico
plt.show()

# Generar el box plot
plt.figure(figsize=(10, 5))
plt.boxplot(resultados, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))

# Etiquetas
plt.xlabel("Valores de Resultado")
plt.title("Box Plot de Resultados Escalares")
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Mostrar gráfico
plt.show() # No toma en cuenta las repeticiones
