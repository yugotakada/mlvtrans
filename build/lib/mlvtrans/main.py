#!/usr/bin/env python3
import numpy as np
import stim

def load_example(example_name):
    if example_name == "31_11_5":
        # --- [[31,11,5]] quantum BCH code  ---
        # coset representatives are calculated from stim.
        H_matrix = np.array([
            [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
        ], dtype=int)
        A_matrix=coset_rep_calcu(H_matrix)
        desired_logical_S = (-1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1)

        return H_matrix, A_matrix, desired_logical_S

    if example_name == "16_4_3":
        # --- [[16,4,3]] color code  ---
        # coset representatives are calculated from stim.
        H_matrix = np.array([
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        ], dtype=int)
        A_matrix=coset_rep_calcu(H_matrix)
        desired_logical_S = (1, 1, 1, 1)

        return H_matrix, A_matrix, desired_logical_S
    
    if example_name == "6_2_2":
        # --- [[6,2,2]] code  ---
        H_matrix = np.array([
            [1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ], dtype=int)
        A_matrix = np.array([
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ], dtype=int)
        desired_logical_S = (1, 1)

        return H_matrix, A_matrix, desired_logical_S
    
    if example_name == "4_2_2":
        # --- [[4,2,2]] code  ---
        H_matrix = np.array([
            [1, 1, 1, 1],
        ], dtype=int)
        A_matrix = np.array([
            [0, 0, 1, 1],
            [1, 0, 1, 0],
        ], dtype=int)
        desired_logical_S = (-1, 1)

        return H_matrix, A_matrix, desired_logical_S
    
    if example_name == "15_7_3":
        # --- [[15,7,3]] quantum Hamming code  ---
        H_matrix = np.array([
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=int)
        A_matrix = np.array([
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=int)
        desired_logical_S = (1, 1, 1, 1, 1, 1, 1)

        return H_matrix, A_matrix, desired_logical_S

    elif example_name == "19_1_5":
        # --- distance 5 hexagonal color code ---
        num_qubit = 19
        num_stabilizer = 9
        num_logical_qubits = 1
        h_indices = {
            0: [0,1,2,3],
            1: [1,3,5,6],
            2: [2,3,4,5,7,8],
            3: [4,7,10,11],
            4: [5,6,8,9,12,13],
            5: [10,11,14,15],
            6: [7,8,11,12,15,16],
            7: [12,13,16,17],
            8: [9,13,17,18],
        }
        a_indices = {
            0: [14,15,16,17,18],
        }
        H_matrix = np.zeros((num_stabilizer, num_qubit), dtype=int)
        for row, cols in h_indices.items():
            H_matrix[row, cols] = 1
        A_matrix = np.zeros((num_logical_qubits, num_qubit), dtype=int)
        for row, cols in a_indices.items():
            A_matrix[row, cols] = 1
        logical_s_dict = {
            0: -1,
        }
        desired_logical_S = tuple(logical_s_dict.get(i, 1) for i in range(num_logical_qubits))

        return H_matrix, A_matrix, desired_logical_S

    else:
        raise Exception(f"Unknown example name: {example_name}")
    
def get_logical_pairs(
    stabilizers: list[stim.PauliString],
) -> list[tuple[stim.PauliString, stim.PauliString]]:
    completed_tableau = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_redundant=True,
        allow_underconstrained=True,
    )

    logicals = []
    for k in range(len(completed_tableau))[::-1]:
        z = completed_tableau.z_output(k)
        if z in stabilizers:
            break
        x = completed_tableau.x_output(k)
        logicals.append((x, z))

    return logicals

def pauli_string_from_binary_row(binary_row, pauli_char):
    s = ''.join(pauli_char if bit == 1 else '_' for bit in binary_row)
    return stim.PauliString("+" + s)

def coset_rep_calcu(H_matrix):
    stabilizers = []
    for row in H_matrix:
        stabilizers.append(pauli_string_from_binary_row(row, "X"))
    for row in H_matrix:
        stabilizers.append(pauli_string_from_binary_row(row, "Z"))
    logical_pairs = get_logical_pairs(stabilizers)

    x_logicals = [x for x, _ in logical_pairs]

    coset_reps = []
    for pauli in x_logicals:
        pauli_str = str(pauli)
        if pauli_str[0] in ['+', '-']:
            pauli_str = pauli_str[1:]
        row = [1 if char == 'X' else 0 for char in pauli_str]
        coset_reps.append(row)

    coset_rep_matrix = np.array(coset_reps, dtype=int)
    return coset_rep_matrix

def has_odd_vector(A):
    for vec in A:
        if np.sum(vec) % 2 == 1:
            return True
    return False

def support(vec):
    return tuple(np.nonzero(vec)[0])

def is_compatible(Lx, Lz):
    for x, z in zip(Lx, Lz):
        if support(x) != support(z):
            return False
        if np.sum(x) % 2 == 0:
            return False
    return True

def algorithm1(A):
    w = [vec.copy() for vec in A]
    s = len(w)
    Lx = []
    Lz = []

    while s > 0:
        # Look for the smallest index p such that w[p]·w[p] = 1 (mod 2)
        p_found = None
        for p in range(s):
            if np.dot(w[p], w[p]) % 2 == 1:
                p_found = p
                break
        # Step 1
        if p_found is not None:
            Lx.append(w[p_found].copy())
            Lz.append(w[p_found].copy())
            new_w = []
            for q in range(s):
                if q == p_found:
                    continue
                dot_val = np.dot(w[q], w[p_found]) % 2
                if dot_val == 1:
                    new_vec = np.bitwise_xor(w[q], w[p_found])
                else:
                    new_vec = w[q].copy()
                new_w.append(new_vec)
            w = new_w
            s = len(w)
        else:
            # Step 2
            p_found = None
            for p in range(1, s):
                if np.dot(w[0], w[p]) % 2 == 1:
                    p_found = p
                    break
            if p_found is None:
                raise Exception("Algorithm1: No valid pair found; algorithm fails.")
            Lx.append(w[0].copy())
            Lz.append(w[p_found].copy())
            Lx.append(w[p_found].copy())
            Lz.append(w[0].copy())
            new_w = []
            for q in range(1, p_found):
                dot_val = np.dot(w[q], w[p_found]) % 2
                if dot_val == 1:
                    new_vec = np.bitwise_xor(w[q], w[0])
                else:
                    new_vec = w[q].copy()
                new_w.append(new_vec)
            for q in range(p_found+1, s):
                dot_val1 = np.dot(w[q], w[0]) % 2
                dot_val2 = np.dot(w[q], w[p_found]) % 2
                new_vec = w[q].copy()
                if dot_val1 == 1:
                    new_vec = np.bitwise_xor(new_vec, w[p_found])
                if dot_val2 == 1:
                    new_vec = np.bitwise_xor(new_vec, w[0])
                new_w.append(new_vec)
            w = new_w
            s = len(w)
    return Lx, Lz

def find_swapped_pair(Lx, Lz):
    n = len(Lx)
    for j in range(n):
        if support(Lx[j]) == support(Lz[j]):
            continue
        for k in range(j+1, n):
            if support(Lx[k]) == support(Lz[k]):
                continue
            if support(Lx[j]) == support(Lz[k]) and support(Lx[k]) == support(Lz[j]):
                return j, k
    return None

def find_good_index(Lx, Lz):
    for i in range(len(Lx)):
        if support(Lx[i]) == support(Lz[i]):
            return i
    return None

def apply_basis_change(Lx, Lz, i, j, k):
    new_i_x = np.bitwise_xor(np.bitwise_xor(Lx[i], Lx[j]), Lx[k])
    new_i_z = np.bitwise_xor(np.bitwise_xor(Lz[i], Lz[j]), Lz[k])
    new_j_x = np.bitwise_xor(Lx[i], Lx[j])
    new_j_z = np.bitwise_xor(Lz[i], Lz[k])
    new_k_x = np.bitwise_xor(Lx[i], Lx[k])
    new_k_z = np.bitwise_xor(Lz[i], Lz[j])
    Lx[i] = new_i_x
    Lz[i] = new_i_z
    Lx[j] = new_j_x
    Lz[j] = new_j_z
    Lx[k] = new_k_x
    Lz[k] = new_k_z
    return Lx, Lz

def make_basis_compatible(Lx, Lz):
    while not is_compatible(Lx, Lz):
        swapped = find_swapped_pair(Lx, Lz)
        if swapped is None:
            break
        j, k = swapped
        i = find_good_index(Lx, Lz)
        if i is None:
            break
        Lx, Lz = apply_basis_change(Lx, Lz, i, j, k)
    return Lx, Lz

def solve_GF2(M, delta):
    M = M.copy().astype(np.int64)
    delta = delta.copy().astype(np.int64)
    k, n = M.shape
    A_aug = np.concatenate((M, delta.reshape(-1, 1)), axis=1)
    rank = 0
    pivot_cols = []
    for col in range(n):
        pivot_row = None
        for row in range(rank, k):
            if A_aug[row, col] % 2 == 1:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        if pivot_row != rank:
            A_aug[[rank, pivot_row], :] = A_aug[[pivot_row, rank], :]
        pivot_cols.append(col)
        for row in range(rank+1, k):
            if A_aug[row, col] % 2 == 1:
                A_aug[row, :] = (A_aug[row, :] + A_aug[rank, :]) % 2
        rank += 1
        if rank == k:
            break
    for row in range(rank, k):
        if A_aug[row, -1] % 2 == 1:
            raise Exception("No solution exists for the GF(2) system.")
    x = np.zeros(n, dtype=np.int64)
    for i in range(rank-1, -1, -1):
        col = pivot_cols[i]
        s = A_aug[i, -1]
        for j in range(col+1, n):
            s = (s - A_aug[i, j]*x[j]) % 2
        x[col] = s % 2
    return x

def compute_v_from_H(H):
    d_list = []
    for row in H:
        wt = int(np.sum(row))
        if wt % 4 == 0:
            d_list.append(0)
        elif wt % 4 == 2:
            d_list.append(1)
        else:
            raise Exception("Stabilizer weight is not even; invalid self-dual CSS code.")
    d = np.array(d_list, dtype=int)
    x = solve_GF2(H, d)
    return x

def compute_physical_S_from_logical_S(H_matrix, desired_logical_S, compatible_basis):
    # Compute v from H.
    v = compute_v_from_H(H_matrix)
    # Default physical S
    physical_S = np.array([1 if vi == 0 else -1 for vi in v], dtype=int)
    # Compute the logical actions by the default physical S.
    k = len(compatible_basis)
    q_tilde = []
    for j in range(k):
        l_j = compatible_basis[j]
        wt = int(np.sum(l_j))
        s_j = np.dot(v, l_j)
        q_j = wt - 2 * s_j
        mod_q = q_j % 4
        if mod_q == 1:
            q_tilde.append(1)
        elif mod_q == 3:
            q_tilde.append(-1)
        else:
            raise Exception(f"Unexpected q_j = {q_j} for logical qubit {j}")
        
    # Adjust the physical S configuration by applying logical Zs.
    for j in range(k):
        if desired_logical_S[j] != q_tilde[j]:
            for i in support(compatible_basis[j]):
                physical_S[i] *= -1
    return physical_S

def verify_physical_S(physical_S, H, compatible_basis, desired_logical_S):
    v = np.where(physical_S == 1, 0, 1)

    # --- Check: Stabilizer group preservation ---
    for idx, row in enumerate(H):
        row = row.astype(int)
        wt = int(np.sum(row))
        b_i = np.dot(v, row)
        p_i = wt - 2 * b_i

        if p_i % 4 != 0:
            print(f"Stabilizer {idx} is not preserved.")
            return False

    # --- Check: Logical actions ---
    k = len(compatible_basis)
    if k != len(desired_logical_S):
        print("Error: The number of logical qubits does not match the number of desired logical phase-type operators.")
        return False
    q_tilde = []
    for j in range(k):
        l_j = compatible_basis[j]
        wt = int(np.sum(l_j))
        s_j = np.dot(v, l_j)
        q_j = wt - 2 * s_j
        mod_q = q_j % 4
        if mod_q == 1:
            q_tilde.append(1)
        elif mod_q == 3:
            q_tilde.append(-1)
        else:
            print(f"Unexpected q_j = {q_j} for logical qubit {j}.")
            return False
    for j in range(k):
        if desired_logical_S[j] != q_tilde[j]:
            print(f"Logical qubit {j}: q_tilde is {q_tilde[j]}, but desired is {desired_logical_S[j]}")
            return False

    return True

def format_S_configuration_with_indices(phy_log_S, phy_log):
    width = 3
    index_line = ""
    symbol_line = ""
    for i, val in enumerate(phy_log_S):
        index_line += f"{i}".center(width)
        if phy_log == 0:
            base = "S"
            dagger = "\u2020" if val == -1 else ""
        else:
            base = "S\u0305"
            dagger = "\u2020" if val == -1 else ""
        cell = " " + base + (dagger if dagger else " ")
        symbol_line += cell

    s_dagger_line = " ".join(str(i) for i, val in enumerate(phy_log_S) if val == -1)
    
    return index_line, symbol_line, s_dagger_line

def is_valid_compatible(Lx, Lz, H_matrix):
    n = len(Lx)
    for i in range(n):
        if support(Lx[i]) != support(Lz[i]):
            print(f"Error: support(Lx[{i}]) != support(Lz[{i}]).")
            return False
        if np.sum(Lx[i]) % 2 == 0:
            print(f"Error: Lx[{i}] has even weight ({np.sum(Lx[i])}).")
            return False
        for j, h in enumerate(H_matrix):
            if np.dot(Lx[i], h) % 2 != 0:
                print(f"Error: Lx[{i}] and H[{j}] anticommute.")
                return False
    for i in range(n):
        for j in range(n):
            overlap = np.dot(Lx[i], Lz[j]) % 2
            if i == j:
                if overlap != 1:
                    print(f"Error: Lx[{i}] and Lz[{i}] commute.")
                    return False
            else:
                if overlap != 0:
                    print(f"Error: Lx[{i}] and Lz[{j}] anticommute.")
                    return False
    return True


#------ use ------
def print_desired_logical_S(*, desired_logical_S):
    index_line, symbol_line, s_dagger_line = format_S_configuration_with_indices(desired_logical_S, 1)
    print("\nDesired logical phase-type gates:")
    print(index_line)
    print(symbol_line)
    print("Indices with S\u0305\u2020:", s_dagger_line)
    return

def alg1(*, A_matrix):
    A = [A_matrix[i, :] for i in range(A_matrix.shape[0])]
    Lx, Lz = algorithm1(A)
    print("\nInitial symplectic basis from Algorithm 1:")
    for i in range(len(Lx)):
        print(f"Pair {i+1}:")
        print(f"  Lx = {Lx[i]}   support = {support(Lx[i])}   weight = {np.sum(Lx[i])}")
        print(f"  Lz = {Lz[i]}   support = {support(Lz[i])}   weight = {np.sum(Lz[i])}")
    return Lx, Lz

def comp_basis(*, A_matrix):
    A = [A_matrix[i, :] for i in range(A_matrix.shape[0])]
    if not has_odd_vector(A):
        print("\nCompatible symplectic basis does not exist: no vector in A has odd weight.")
        return
    Lx, Lz = algorithm1(A)
    Lx, Lz = make_basis_compatible(Lx, Lz)
    print("\nCompatible symplectic basis:")
    for i in range(len(Lx)):
        print(f"Pair {i+1}:")
        print(f"  Lx = {Lx[i]}   support = {support(Lx[i])}   weight = {np.sum(Lx[i])}")
        print(f"  Lz = {Lz[i]}   support = {support(Lz[i])}   weight = {np.sum(Lz[i])}")
    return Lx, Lz

def lemma2(*, Lx, Lz):
    if not has_odd_vector(Lx):
        print("\nCompatible symplectic basis does not exist: no vector in A has odd weight.")
        return
    Lx, Lz = make_basis_compatible(Lx, Lz)
    print("\nCompatible symplectic basis:")
    for i in range(len(Lx)):
        print(f"Pair {i+1}:")
        print(f"  Lx = {Lx[i]}   support = {support(Lx[i])}   weight = {np.sum(Lx[i])}")
        print(f"  Lz = {Lz[i]}   support = {support(Lz[i])}   weight = {np.sum(Lz[i])}")
    return Lx, Lz

def phys_S_for_logi_S(*, H_matrix, desired_logical_S, Lx):
    # Compute the physical S configuration.
    physical_S = compute_physical_S_from_logical_S(H_matrix, desired_logical_S, Lx)
    index_line, symbol_line, s_dagger_line = format_S_configuration_with_indices(physical_S, 0)
    print("\nPhysical phase-type gates to achieve the desired logical phase-type gates:")
    print(index_line)
    print(symbol_line)
    print("Indices with S†:", s_dagger_line)
    return physical_S

def check_compatible(*, H_matrix, Lx, Lz):
    if is_valid_compatible(Lx, Lz, H_matrix):
        print("\nIs the symplectic basis compatible?: YES")
    else:
        print("\nIs the symplectic basis compatible?: NO")
    return

def check_S(*, H_matrix, desired_logical_S, Lx, physical_S):
    if verify_physical_S(physical_S, H_matrix, Lx, desired_logical_S):
        print("\nDo the physical phase-type gates give the desired logical phase-type gates?: YES")
    else:
        print("\nDo the physical phase-type gates give the desired logical phase-type gates?: NO")
    return

def run_matrix(*, H_matrix, A_matrix, desired_logical_S):
    print("Parity-check matrix H:")
    print(H_matrix)
    print("\nCoset representatives matrix A:")
    print(A_matrix)
    index_line, symbol_line, s_dagger_line = format_S_configuration_with_indices(desired_logical_S, 1)
    print("\nDesired logical phase-type gates:")
    print(index_line)
    print(symbol_line)
    print("Indices with S\u0305\u2020:", s_dagger_line)
    
    A = [A_matrix[i, :] for i in range(A_matrix.shape[0])]
    
    # Check that at least one vector in A has odd weight.
    if not has_odd_vector(A):
        print("\nCompatible symplectic basis does not exist: no vector in A has odd weight.")
        return
    
    # Apply Algorithm 1 to produce the initial symplectic basis.
    Lx, Lz = algorithm1(A)
    print("\nInitial symplectic basis from Algorithm 1:")
    for i in range(len(Lx)):
        print(f"Pair {i+1}:")
        print(f"  Lx = {Lx[i]}   support = {support(Lx[i])}   weight = {np.sum(Lx[i])}")
        print(f"  Lz = {Lz[i]}   support = {support(Lz[i])}   weight = {np.sum(Lz[i])}")
    
    # Apply lemma 2 repeatedly until the basis is compatible.
    Lx, Lz = make_basis_compatible(Lx, Lz)
    print("\nCompatible symplectic basis:")
    for i in range(len(Lx)):
        print(f"Pair {i+1}:")
        print(f"  Lx = {Lx[i]}   support = {support(Lx[i])}   weight = {np.sum(Lx[i])}")
        print(f"  Lz = {Lz[i]}   support = {support(Lz[i])}   weight = {np.sum(Lz[i])}")

    # Compute the physical S configuration.
    physical_S = compute_physical_S_from_logical_S(H_matrix, desired_logical_S, Lx)
    index_line, symbol_line, s_dagger_line = format_S_configuration_with_indices(physical_S, 0)
    print("\nPhysical phase-type gates to achieve the desired logical phase-type gates:")
    print(index_line)
    print(symbol_line)
    print("Indices with S†:", s_dagger_line)

    # ---------- FINAL CHECK ----------
    if is_valid_compatible(Lx, Lz, H_matrix):
        print("\nIs the symplectic basis compatible?: YES")
    else:
        print("\nIs the symplectic basis compatible?: NO")
    
    if verify_physical_S(physical_S, H_matrix, Lx, desired_logical_S):
        print("\nDo the physical phase-type gates give the desired logical phase-type gates?: YES")
    else:
        print("\nDo the physical phase-type gates give the desired logical phase-type gates?: NO")
    # ---------- FINAL CHECK ----------
        
def run_index(*, num_qubit, num_stabilizer, num_logical_qubits, h_indices, a_indices, logical_s_dict):
    H_matrix = np.zeros((num_stabilizer, num_qubit), dtype=int)
    for row, cols in h_indices.items():
        H_matrix[row, cols] = 1
    A_matrix = np.zeros((num_logical_qubits, num_qubit), dtype=int)
    for row, cols in a_indices.items():
        A_matrix[row, cols] = 1
    desired_logical_S = tuple(logical_s_dict.get(i, 1) for i in range(num_logical_qubits))
    run_matrix(H_matrix=H_matrix, A_matrix=A_matrix, desired_logical_S=desired_logical_S)

def run_matrix_auto_coset(*, H_matrix, desired_logical_S):
    A_matrix=coset_rep_calcu(H_matrix)
    run_matrix(H_matrix=H_matrix, A_matrix=A_matrix, desired_logical_S=desired_logical_S)

def run_index_auto_coset(*, num_qubit, num_stabilizer, num_logical_qubits, h_indices, logical_s_dict):
    H_matrix = np.zeros((num_stabilizer, num_qubit), dtype=int)
    for row, cols in h_indices.items():
        H_matrix[row, cols] = 1
    desired_logical_S = tuple(logical_s_dict.get(i, 1) for i in range(num_logical_qubits))
    A_matrix=coset_rep_calcu(H_matrix)
    run_matrix(H_matrix=H_matrix, A_matrix=A_matrix, desired_logical_S=desired_logical_S)

def examples(*, example_choice):
    example = example_choice
    H_matrix, A_matrix, desired_logical_S = load_example(example)
    run_matrix(H_matrix=H_matrix, A_matrix=A_matrix, desired_logical_S=desired_logical_S)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='mlvtrans: construct a compatible symplectic basis and transversal phase-type gates for self-dual CSS codes')
    parser.add_argument(
        '--example',
        type=str,
        choices=["31_11_5", "16_4_3", "6_2_2", "4_2_2", "15_7_3", "19_1_5"],
        default="31_11_5",
        help='example name'
    )
    args = parser.parse_args()
    examples(example_choice=args.example)

if __name__ == "__main__":
    main()
