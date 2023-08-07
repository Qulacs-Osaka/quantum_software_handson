from qulacs import QuantumCircuit
from Adder import add_sum_gate, add_carry_gate, add_inv_carry_gate

def add_subtracter_gate(circ, subtrahendIds, minuendIds, carryIds):
    
    digit = len(subtrahendIds)
    
    if digit + 1 != len(minuendIds):
        raise ValueError("The minuend register's qubit num must be equal to the subtrahend register's qubit num + 1")
        
    if digit != len(carryIds):
        raise ValueError("The subtrahend register's qubit num must be equal to the carry register's qubit num")
        
    # flip each qubit in the subtrahend register to get 2's complement of it - 1
    for i in range(digit):
        circ.add_X_gate(minuendIds[i])
        
    add_carry_gate(circ, subtrahendIds[0], minuendIds[0], carryIds[0], carryIds[1])
    for i in range(1, digit - 1):
        add_carry_gate(circ, subtrahendIds[i], minuendIds[i], carryIds[i], carryIds[i + 1])
        
    circ.add_CNOT_gate(subtrahendIds[digit - 1], minuendIds[digit - 1])
    add_sum_gate(circ, carryIds[digit - 1], subtrahendIds[digit - 1], minuendIds[digit - 1])
    
    for i in range(digit - 2, -1, -1):
        add_inv_carry_gate(circ, subtrahendIds[i], minuendIds[i], carryIds[i], carryIds[i + 1])
        add_sum_gate(circ, carryIds[i], subtrahendIds[i], minuendIds[i])
    
    # flip each qubit in the minuend register
    for i in range(digit):
        circ.add_X_gate(minuendIds[i])
        
    