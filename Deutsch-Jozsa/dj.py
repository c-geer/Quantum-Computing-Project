from sparsematrix import SparseMatrix

A = [[0, 1],[2, 3]]
B = SparseMatrix.from_dense_matrix(A)

if isinstance(B, SparseMatrix):
    print(B[0,1])