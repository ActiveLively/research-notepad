import numpy as np
from einsum import einsum   # pip install einsum

# Set a seed so results are reproducible
np.random.seed(0)

# Example arrays
x = np.random.randn(5)          # vector length 5
y = np.random.randn(5)          # vector length 5
A = np.random.randn(5, 5)       # 5x5 matrix
B = np.random.randn(5, 5)       # 5x5 matrix
M = np.random.randn(3, 5, 5)    # batch of 3 matrices
X = np.random.randn(10, 4)      # dataset: 10 samples, 4 features
Q = np.random.randn(2, 3, 4, 6) # queries: batch=2, heads=3, seq=4, dim=6
K = np.random.randn(2, 3, 5, 6) # keys:    batch=2, heads=3, seq=5, dim=6

# --- Practice Problems ---

# 1. Vector inner product
dot = einsum("i,i->", x, y)
print("Dot product:", dot)

# 2. Matrix-vector product
matvec = einsum("ij,j->i", A, x)
print("Matrix-vector:", matvec)

# 3. Matrix transpose
transpose = einsum("ij->ji", A)
print("Transpose shape:", transpose.shape)

# 4. Outer product
outer = einsum("i,j->ij", x, y)
print("Outer product shape:", outer.shape)

# 5. Matrix-matrix multiplication
matmul = einsum("ik,kj->ij", A, B)
print("Matrix-matrix shape:", matmul.shape)

# 6. Batch matrix-vector multiplication
batch_matvec = einsum("bij,bj->bi", M, x[:5])
print("Batch matvec shape:", batch_matvec.shape)

# 7. Trace of a matrix
trace = einsum("ii->", A)
print("Trace:", trace)

# 8. Bilinear form xᵀAy
bilinear = einsum("i,ij,j->", x, A, y)
print("Bilinear form:", bilinear)

# 9. Frobenius inner product of two matrices
frobenius = einsum("ij,ij->", A, B)
print("Frobenius inner product:", frobenius)

# 10. Attention scores (QKᵀ)
scores = einsum("bhid,bhjd->bhij", Q, K)
print("Attention scores shape:", scores.shape)

# 11. Covariance matrix XᵀX
cov = einsum("nd,ne->de", X, X)
print("Covariance shape:", cov.shape)