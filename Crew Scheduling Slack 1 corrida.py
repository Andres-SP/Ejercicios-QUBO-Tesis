import numpy as np
from collections import defaultdict
import pennylane as qml 
import matplotlib.pyplot as plt

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

samples_slack = samples_dict(qaoa_circuit(gammas, betas, h, J, num_qubits=len(QT)), n_qubits) # The solutions are generated here

keys = list(samples_slack.keys())

print(samples_slack)
#for key in keys:
 #   solution = list(map(int,key))
  #  print(solution)

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
    factible = np.all(np.dot(mat, vector_corto) >= b)  # Comparación con b_restricciones

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
    if repeticiones[i] <= 2:
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
plt.figure(figsize=(12, 6))

plt.bar(
    [str(r) for r in resultados_unicos], repeticiones_factibles,
    color='green', label="Factibles"
)

plt.bar(
    [str(r) for r in resultados_unicos], repeticiones_infactibles,
    color='red', bottom=repeticiones_factibles, label="Infactibles"
)

# Etiquetas y formato
plt.xlabel('Energía')
plt.ylabel('Repeticiones')
plt.title('Histograma de soluciones factibles e infactibles')
plt.xticks(rotation=90)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar gráfico
plt.show()

# Generar el box plot
plt.figure(figsize=(8, 5))
plt.boxplot(resultados, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))

# Etiquetas
plt.xlabel("Valores de Energía")
plt.title("Box Plot de Soluciones")
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Mostrar gráfico
plt.show() # No toma en cuenta las repeticiones
