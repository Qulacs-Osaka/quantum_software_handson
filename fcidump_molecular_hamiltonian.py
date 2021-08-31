import re
import numpy as np
from pyscf import __config__
from openfermion.ops import InteractionOperator

MOLPRO_ORBSYM = getattr(__config__, 'fcidump_molpro_orbsym', False)

# Mapping Pyscf symmetry numbering to Molpro symmetry numbering for each irrep.
# See also pyscf.symm.param.IRREP_ID_TABLE
# https://www.molpro.net/info/current/doc/manual/node36.html
ORBSYM_MAP = {
    'D2h': (1,         # Ag
            4,         # B1g
            6,         # B2g
            7,         # B3g
            8,         # Au
            5,         # B1u
            3,         # B2u
            2),        # B3u
    'C2v': (1,         # A1
            4,         # A2
            2,         # B1
            3),        # B2
    'C2h': (1,         # Ag
            4,         # Bg
            2,         # Au
            3),        # Bu
    'D2' : (1,         # A
            4,         # B1
            3,         # B2
            2),        # B3
    'Cs' : (1,         # A'
            2),        # A"
    'C2' : (1,         # A
            2),        # B
    'Ci' : (1,         # Ag
            2),        # Au
    'C1' : (1,)
}

# Taken from PySCF
def read(filename, molpro_orbsym=MOLPRO_ORBSYM):
    '''Parse FCIDUMP.  Return a dictionary to hold the integrals and
    parameters with keys:  H1, H2, ECORE, NORB, NELEC, MS, ORBSYM, ISYM
    Kwargs:
        molpro_orbsym (bool): Whether the orbsym in the FCIDUMP file is in
            Molpro orbsym convention as documented in
            https://www.molpro.net/info/current/doc/manual/node36.html
            In return, orbsym is converted to pyscf symmetry convention
    '''
    print('Parsing %s' % filename)
    finp = open(filename, 'r')

    data = []
    for i in range(10):
        line = finp.readline().upper()
        data.append(line)
        if '&END' in line:
            break
    else:
        raise RuntimeError('Problematic FCIDUMP header')

    result = {}
    tokens = ','.join(data).replace('&FCI', '').replace('&END', '')
    tokens = tokens.replace(' ', '').replace('\n', '').replace(',,', ',')
    for token in re.split(',(?=[a-zA-Z])', tokens):
        key, val = token.split('=')
        if key in ('NORB', 'NELEC', 'MS2', 'ISYM'):
            result[key] = int(val.replace(',', ''))
        elif key in ('ORBSYM',):
            result[key] = [int(x) for x in val.replace(',', ' ').split()]
        else:
            result[key] = val

    # Convert to molpr orbsym convert_orbsym
    if 'ORBSYM' in result:
        if molpro_orbsym:
            # Guess which point group the orbsym belongs to. FCIDUMP does not
            # save the point group information, the guess might be wrong if
            # the high symmetry numbering of orbitals are not presented.
            orbsym = result['ORBSYM']
            if max(orbsym) > 4:
                result['ORBSYM'] = [ORBSYM_MAP['D2h'].index(i) for i in orbsym]
            elif max(orbsym) > 2:
                # Fortunately, without molecular orientation, B2 and B3 in D2
                # are not distinguishable
                result['ORBSYM'] = [ORBSYM_MAP['C2v'].index(i) for i in orbsym]
            elif max(orbsym) == 2:
                result['ORBSYM'] = [i-1 for i in orbsym]
            elif max(orbsym) == 1:
                result['ORBSYM'] = [0] * len(orbsym)
            else:
                raise RuntimeError('Unknown orbsym')
        elif max(result['ORBSYM']) >= 8:
            raise RuntimeError('Unknown orbsym convention')

    norb = result['NORB']
    norb_pair = norb * (norb+1) // 2
    h1e = np.zeros((norb,norb))
    h2e = np.zeros((norb,norb,norb,norb))
    dat = finp.readline().split()
    while dat:
        i, j, k, l = [int(x) for x in dat[1:5]]
        hijkl = float(dat[0])
        if k != 0:
            h2e[i-1,j-1,k-1,l-1] = hijkl # (ij|kl)
            h2e[i-1,j-1,l-1,k-1] = hijkl # (ij|lk)
            h2e[j-1,i-1,k-1,l-1] = hijkl # (ji|kl)
            h2e[j-1,i-1,l-1,k-1] = hijkl # (ji|lk)
            h2e[l-1,k-1,j-1,i-1] = hijkl # (lk|ji)
            h2e[k-1,l-1,i-1,j-1] = hijkl # (kl|ij)
            h2e[k-1,l-1,j-1,i-1] = hijkl # (kl|ji)
            h2e[l-1,k-1,i-1,j-1] = hijkl # (lk|ij)
        elif k == 0:
            if j != 0:
                h1e[i-1,j-1] = float(dat[0])
            else:
                result['ECORE'] = float(dat[0])
        dat = finp.readline().split()

    idx, idy = np.tril_indices(norb, -1)
    if np.linalg.norm(h1e[idy,idx]) == 0:
        h1e[idy,idx] = h1e[idx,idy]
    elif np.linalg.norm(h1e[idx,idy]) == 0:
        h1e[idx,idy] = h1e[idy,idx]
    result['H1'] = h1e
    result['H2'] = h2e
    finp.close()
    return result

def activespace_indices(mf, nelec, norb):
    nelectron = mf.mol.nelectron
    if (nelectron % 2) :
        nelectron = nelectron+1
    if (nelec % 2) :
        nelec = nelec + 1
    ndocc = (nelectron) // 2
    nlower = nelec // 2
    nclosed = ndocc - nlower
    nupper = norb - nlower
    max_active = nupper + ndocc
    closed = [i for i in range(nclosed*2)]
    active = [i for i in range(nclosed*2,max_active*2)]
    return closed, active

def init_H_coeffs(n_qubits):
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros((n_qubits, n_qubits,
                                      n_qubits, n_qubits))
    return one_body_coefficients, two_body_coefficients

def get_molecular_hamiltonian_from_fcidump(fname):

    #read FCIDUMP
    myfcidump = read(fname)

    nelec = myfcidump['NELEC']
    norb = myfcidump['NORB']


    one_body_integrals = myfcidump['H1']
    eri_mo = myfcidump['H2']
    two_body_integrals = np.asarray(
        eri_mo.transpose(0, 2, 3, 1), order='C')

    constant = myfcidump['ECORE']

    n_qubits = 2 * one_body_integrals.shape[0]

    one_body_coefficients, two_body_coefficients = init_H_coeffs(n_qubits)

    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):
            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[
                p, q]
            one_body_coefficients[2 * p + 1, 2 *
                                  q + 1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):
                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1,
                                          2 * r + 1, 2 * s] = (
                        two_body_integrals[p, q, r, s] / 2.)
                    two_body_coefficients[2 * p + 1, 2 * q,
                                          2 * r, 2 * s + 1] = (
                        two_body_integrals[p, q, r, s] / 2.)

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q,
                                          2 * r, 2 * s] = (
                        two_body_integrals[p, q, r, s] / 2.)
                    two_body_coefficients[2 * p + 1, 2 * q + 1,
                                          2 * r + 1, 2 * s + 1] = (
                        two_body_integrals[p, q, r, s] / 2.)

    # Cast to InteractionOperator class and return.
    molecular_hamiltonian = InteractionOperator(
        constant, one_body_coefficients, two_body_coefficients)
    return molecular_hamiltonian


if __name__=='__main__':
    molecular_hamiltonian = get_molecular_hamiltonian_from_fcidump('FCIDUMP.h2o.full')

    from openfermion.transforms import get_fermion_operator, jordan_wigner
    from openfermion.transforms import get_sparse_operator
    jw_hamiltonian = jordan_wigner(get_fermion_operator
                                   (molecular_hamiltonian))
    jw_matrix = get_sparse_operator(jw_hamiltonian)
    from openfermion.utils import get_ground_state
    fci_energy, hogehoge = get_ground_state(jw_matrix)
    print ("FCI ENERGY", fci_energy)
    from scipy.sparse.linalg import eigs
    from scipy.linalg import eig
    eigenenergies, eigenvecs = eigs(jw_matrix)
    print("DIAG jw_matrix", eigenenergies[0])
    print("DIAG jw_matrix", eigenenergies)

    n_qubit = 14
    from qulacs import QuantumState
    from qulacsplus.utils.parsers.openfermion_parsers.operator_parser import parse_of_operators
    q_h = parse_of_operators(n_qubit,jw_hamiltonian)
    hf = QuantumState(n_qubit)
    n_electron = 10
    hf.set_computational_basis(
            int('0b'+'0'*(n_qubit - n_electron)+'1'*(n_electron),2))
    E_HF_active = q_h.get_expectation_value(hf)
    print ("E-HF active", E_HF_active)
