from qulacs import QuantumCircuit
from qulacs.gate import TOFFOLI

def add_sum_gate(circ, ia, ib, iout):
    circ.add_CNOT_gate(ib, iout)
    circ.add_CNOT_gate(ia, iout)
    
def add_carry_gate(circ, ia, ib, icarry, iout):
    circ.add_gate(TOFFOLI(ia, ib, iout))
    circ.add_CNOT_gate(ia, ib)
    circ.add_gate(TOFFOLI(ib, icarry, iout))
    
def add_inv_carry_gate(circ, ia, ib, icarry, iout):
    circ.add_gate(TOFFOLI(ib, icarry, iout))
    circ.add_CNOT_gate(ia, ib)
    circ.add_gate(TOFFOLI(ia, ib, iout))

def add_adder_gate(circ, addendIds, augendIds, carryIds):
    
    digit = len(addendIds)
    
    if digit + 1 != len(augendIds):
        raise ValueError("The augend register's qubit num must be equal to the addend register's qubit num + 1")
        
    if digit != len(carryIds):
        raise ValueError("The addend register's qubit num must be equal to the carry register's qubit num")
        
    add_carry_gate(circ, addendIds[0], augendIds[0], carryIds[0], carryIds[1])
    for i in range(1, digit - 1):
        add_carry_gate(circ, addendIds[i], augendIds[i], carryIds[i], carryIds[i + 1])
       
    add_carry_gate(circ, addendIds[digit - 1], augendIds[digit - 1], carryIds[digit - 1], augendIds[digit])
    circ.add_CNOT_gate(addendIds[digit - 1], augendIds[digit - 1])
    add_sum_gate(circ, carryIds[digit - 1], addendIds[digit - 1], augendIds[digit - 1])
    
    for i in range(digit - 2, -1, -1):
        add_inv_carry_gate(circ, addendIds[i], augendIds[i], carryIds[i], carryIds[i + 1])
        add_sum_gate(circ, carryIds[i], addendIds[i], augendIds[i])
    