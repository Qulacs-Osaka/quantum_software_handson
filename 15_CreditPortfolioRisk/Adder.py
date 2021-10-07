from qulacs.gate import TOFFOLI, CNOT, merge

def sum_gate(ia, ib, iout):
    return merge(CNOT(ib, iout), CNOT(ia, iout))
    
def carry_gate(ia, ib, icarry, iout):
    return merge([TOFFOLI(ia, ib, iout), CNOT(ia, ib), TOFFOLI(ib, icarry, iout)])
    
def inv_carry_gate(ia, ib, icarry, iout):
    return merge([TOFFOLI(ib, icarry, iout), CNOT(ia, ib), TOFFOLI(ia, ib, iout)])

def adder_gate(addendIds, augendIds, carryIds):
    
    digit = len(addendIds)
    
    if digit + 1 != len(augendIds):
        raise ValueError("The augend register's qubit num must be equal to the addend register's qubit num + 1")
        
    if digit != len(carryIds):
        raise ValueError("The addend register's qubit num must be equal to the carry register's qubit num")
        
    ret = carry_gate(addendIds[0], augendIds[0], carryIds[0], carryIds[1])
    for i in range(1, digit - 1):
        ret = merge(ret, carry_gate(addendIds[i], augendIds[i], carryIds[i], carryIds[i + 1]))
        
    ret = merge(ret, carry_gate(addendIds[digit - 1], augendIds[digit - 1], carryIds[digit - 1], augendIds[digit]))
    ret = merge(ret, CNOT(addendIds[digit - 1], augendIds[digit - 1]))
    ret = merge(ret, sum_gate(carryIds[digit - 1], addendIds[digit - 1], augendIds[digit - 1]))
    
    for i in range(digit - 2, -1, -1):
        ret = merge(ret, inv_carry_gate(addendIds[i], augendIds[i], carryIds[i], carryIds[i + 1]))
        ret = merge(ret, sum_gate(carryIds[i], addendIds[i], augendIds[i]))
    
    return ret
        
    