from qulacs.gate import X, CNOT, merge

def const_setter_gate(num, numIds):
    
    gateList = []
    
    binNum = list(reversed(list('{:b}'.format(num))))
    for i in range(len(binNum)):
        if binNum[i] == '1':
            gateList += [X(numIds[i])]
    
    return merge(gateList)

def ctrl_const_setter_gate(num, ctrlId, numIds, ctrlIs1):
    
    gateList = []
    
    if not ctrlIs1:
        gateList += [X(ctrlId)]
    
    binNum = list(reversed(list('{:b}'.format(num))))
    for i in range(len(binNum)):
        if binNum[i] == '1':
            gateList += [CNOT(ctrlId, numIds[i])]
    
    if not ctrlIs1:
        gateList += [X(ctrlId)]
    
    return merge(gateList)
    
    
        
    