# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:34:37 2025

@author: andre
"""

import sympy as sp

Par = 15

x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12')
fo = -3*x1 - 4*x2 - 5*x3 - 4*x4 - 3*x5
restricciones = [3*x1 + 4*x2 + 3*x3 + 2*x4 + 4*x5 + x6 + 2*x7 + 4*x8 - 10,
                 6*x1 + 8*x2 + 7*x3 + 9*x4 + 6*x5 + x9 + 2*x10 + 4*x11
                 + 8*x12 - 20]
variables = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12]

n = len(variables)
matriz = sp.zeros(n, n)

indices = []
constante = 0

for r in restricciones:
    r2 = sp.expand(Par*(r**2))
    p = sp.poly(r2, *variables)
    coef = p.coeffs()
    mon = p.monoms()

    for coef, mono in zip(coef, mon):
        # Crear una lista de índices para las variables presentes en el monomio
        indices = [i for i, exp in enumerate(mono) if exp > 0]
        
        if len(indices) == 0:
            # Término constante, no se agrega a la matriz
            constante += coef
            continue
        elif len(indices) == 1:  # Término de una variable
            i = indices[0]
            matriz[i, i] += coef
        elif len(indices) == 2:  # Término de dos variables
            i, j = indices
            matriz[i, j] += coef

p_fo = sp.poly(fo, *variables)
coef_fo = p_fo.coeffs()
mon_fo = p_fo.monoms()

for coef_fo, mono_fo in zip(coef_fo, mon_fo):
    # Crear una lista de índices para las variables presentes en el monomio
    indices_fo = [i for i, exp in enumerate(mono_fo) if exp > 0]
    
    if len(indices_fo) == 0:
        # Término constante, no se agrega a la matriz
        constante += coef_fo
        continue
    elif len(indices_fo) == 1:  # Término de una variable
        i = indices_fo[0]
        matriz[i, i] += coef_fo
    elif len(indices_fo) == 2:  # Término de dos variables
        i, j = indices_fo
        matriz[i, j] += coef_fo
