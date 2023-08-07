from qulacs import QuantumCircuit
from qulacs.gate import X, CNOT, merge

def add_const_setter_gate(circ, num, numIds):
    
    gateList = []
    
    binNum = list(reversed(list('{:b}'.format(num))))
    for i in range(len(binNum)):
        if binNum[i] == '1':
            gateList += [X(numIds[i])]
    
    circ.add_gate(merge(gateList))

def add_ctrl_const_setter_gate(circ, num, ctrlId, numIds, ctrlIs1):
    
    gateList = []
    
    if not ctrlIs1:
        gateList += [X(ctrlId)]
    
    binNum = list(reversed(list('{:b}'.format(num))))
    for i in range(len(binNum)):
        if binNum[i] == '1':
            gateList += [CNOT(ctrlId, numIds[i])]
    
    if not ctrlIs1:
        gateList += [X(ctrlId)]
    
    circ.add_gate(merge(gateList))
    
    
        
    