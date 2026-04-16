
"""

    Matrix module
    =============

    Defines useful functions to deal with symbolic matrices.

"""

from sympy import MutableSparseMatrix

def block_matrix_inverse(P, blocks_extent):
    """Function to invert a symbolic matrix :math:`P` devided by blocks.

    Parameters
    ----------
    P: ~sympy.matrices.immutable.ImmutableSparseMatrix or ~sympy.matrices.mutable.MutableSparseMatrix
        The block matrix to invert.
    blocks_extent: list(tuple)
        The extent of each block, as a list of 2-tuple.

    Warnings
    --------
    To be fast, this function doesn't check if the matrix :math:`P` is invertible !

    """
    be = blocks_extent.copy()
    PP = P.copy()
    B_list = list()
    C_list = list()
    Am1_list = list()
    while len(be) >= 2:
        rest_block_extent = (be[1][0], be[-1][1])
        Ainv, B, C, D, PPP = _block_matrix_inverse_2x2(PP, (be[0], rest_block_extent))

        B_list.append(B)
        C_list.append(C)
        Am1_list.append(Ainv)

        PP = PPP
        ret = be[0][1]
        be = list(map(lambda p: (p[0]-ret, p[1]-ret), be[1:]))

    PP = PP.adjugate() / PP.det()
    for Am1, B, C in zip(Am1_list[::-1], B_list[::-1], C_list[::-1]):
        Ashape = Am1.shape[0]
        Pshape = PP.shape[0]
        mat_shape = Ashape + Pshape
        Pp1 = MutableSparseMatrix(mat_shape, mat_shape, {})
        Pp1[slice(0, Ashape), slice(0, Ashape)] = Am1 + Am1 @ B @ PP @ C @ Am1
        Pp1[slice(0, Ashape), slice(Ashape, mat_shape)] = - Am1 @ B @ PP
        Pp1[slice(Ashape, mat_shape), slice(0, Ashape)] = - PP @ C @ Am1
        Pp1[slice(Ashape, mat_shape), slice(Ashape, mat_shape)] = PP
        PP = Pp1

    return PP

def _block_matrix_inverse_2x2(P, blocks_extent):
    be = blocks_extent
    A = P[slice(*be[0]), slice(*be[0])]
    B = P[slice(*be[0]), slice(*be[1])]
    C = P[slice(*be[1]), slice(*be[0])]
    D = P[slice(*be[1]), slice(*be[1])]
    Ainv = A.adjugate()/A.det()

    return Ainv, B, C, D, D - C @ Ainv @ B


if __name__ == '__main__':
    A = MutableSparseMatrix(3, 3, {})
    A[0,0] = 1
    A[1,1] = 3
    A[2,2] = 2
    C = A /2
    D = 3 * A
    G = MutableSparseMatrix(9, 9, {})
    G[:3, :3] = A
    G[:3, 3:6] = -C
    G[:3, 6:] = D
    G[3:6, :3] = C
    G[3:6, 3:6] = C
    G[3:6, 6:] = - D
    G[6:, :3] = - C
    G[6:, 3:6] = A
    G[6:, 6:] = D
    bl = [(0, 3), (3, 6), (6, 9)]
    Ginv = block_matrix_inverse(G, bl)

