import numpy as np
from collections import defaultdict
import pennylane as qml
from docplex.mp.model import Model
from openqaoa.problems import FromDocplex2IsingModel
import matplotlib.pyplot as plt
import time

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

# 3. Crew Scheduling Problem -----------------------------------------------------------------------------------
def crew_scheduling_problem(obj_func, set_constraints):
    nu_teams = set_constraints.shape[1]
    nu_constr = set_constraints.shape[0]

    mdl = Model(name="Crew Scheduling Problem")
    x = mdl.binary_var_list(nu_teams, name="x")

    objective = mdl.sum(x[i] * obj_func[i] for i in range(nu_teams))
    mdl.minimize(objective)

    for j in range(nu_constr - 1):
        mdl.add_constraint(mdl.sum(x[k] * set_constraints[j, k] for k in range(nu_teams)) >= 1)

    mdl.add_constraint(mdl.sum(x[k] * set_constraints[6, k] for k in range(nu_teams)) <= 3)

    num_vars = mdl.number_of_variables
    print(mdl.export_as_lp_string())
    return mdl

obj_function = [2, 4, 4, 3, 2]
set_constraints = np.array([
                            [1, 0, 0, 1, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1],
                            [1, 1, 0, 1, 0],
                            [0, 1, 0, 0, 1],
                            [1, 0, 1, 0, 0],
                            [1, 1, 1, 1, 1]
                           ])

mdl = crew_scheduling_problem(obj_function,set_constraints)

lambda_1, lambda_2 = (
    20,
    20,
)  # Parameters of the unbalanced penalization function (They are in the main paper)
ising_hamiltonian = FromDocplex2IsingModel(
    mdl,
    unbalanced_const=True,
    strength_ineq=[lambda_1, lambda_2],  # https://arxiv.org/abs/2211.13914
).ising_model

h_new = {
    tuple(i): w for i, w in zip(ising_hamiltonian.terms, ising_hamiltonian.weights) if len(i) == 1
}
J_new = {
    tuple(i): w for i, w in zip(ising_hamiltonian.terms, ising_hamiltonian.weights) if len(i) == 2
}

def run_qaoa():
    samples_unbalanced = samples_dict(
        qaoa_circuit(gammas, betas, h_new, J_new, num_qubits=5), 5
    )
    return samples_unbalanced

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

    c = np.array([2, 4, 4, 3, 2])

    # Contar soluciones factibles
    conteo = 0
    conteo_opt = 0

    # Calcular los resultados como los primeros 5 valores de la solución multiplicados por la matriz para el escalar
    for clave, veces in samples.items():
        # Extraer las soluciones como vector
        vector = [int(bit) for bit in clave]

        # Convertir en un vector numpy
        vector= np.array(vector)

        # Verificar factibilidad con la matriz de restricciones
        if np.all(np.dot(mat, vector) >= b):
            conteo += veces

            if np.dot(vector, c) == 8:
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
