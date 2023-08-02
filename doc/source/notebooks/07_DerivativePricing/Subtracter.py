from qulacs.gate import TOFFOLI, CNOT, merge, Identity, X
from Adder import sum_gate, carry_gate, inv_carry_gate

def subtracter_gate(subtrahendIds, minuendIds, carryIds):
    
    digit = len(subtrahendIds)
    
    if digit + 1 != len(minuendIds):
        raise ValueError("The minuend register's qubit num must be equal to the subtrahend register's qubit num + 1")
        
    if digit != len(carryIds):
        raise ValueError("The subtrahend register's qubit num must be equal to the carry register's qubit num")
        
    # flip each qubit in the subtrahend register to get 2's complement of it - 1
    ret = Identity(0)
    for i in range(digit):
        ret = merge(ret, X(minuendIds[i]))    
        
    ret = merge(ret, carry_gate(subtrahendIds[0], minuendIds[0], carryIds[0], carryIds[1]))
    for i in range(1, digit - 1):
        ret = merge(ret, carry_gate(subtrahendIds[i], minuendIds[i], carryIds[i], carryIds[i + 1]))
        
    ret = merge(ret, CNOT(subtrahendIds[digit - 1], minuendIds[digit - 1]))
    ret = merge(ret, sum_gate(carryIds[digit - 1], subtrahendIds[digit - 1], minuendIds[digit - 1]))
    
    for i in range(digit - 2, -1, -1):
        ret = merge(ret, inv_carry_gate(subtrahendIds[i], minuendIds[i], carryIds[i], carryIds[i + 1]))
        ret = merge(ret, sum_gate(carryIds[i], subtrahendIds[i], minuendIds[i]))
    
    # flip each qubit in the minuend register
    for i in range(digit):
        ret = merge(ret, X(minuendIds[i]))      
    
    return ret
        
    