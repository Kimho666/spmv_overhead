import torch
import torch.nn as nn

import mxq_inference_engine

#torch.backends.cuda.matmul.allow_tf32 = False
#torch.backends.cudnn.allow_tf32 = False


DEV = torch.device('cuda')

# M = 12288 * 4
# N = 12288
M = 1
N = 8192
K = 28672


# DTYPE = torch.half
# B = torch.randn((K, N), device=DEV, dtype=DTYPE)
# A = torch.randn((M, K), device=DEV, dtype=DTYPE)
# C = torch.zeros((M, N), device=DEV, dtype=DTYPE)

# COUNT = 100
# import time
# torch.matmul(A, B, out=C) 
# torch.cuda.synchronize()
# tick = time.time()
# for _ in range(COUNT):
#     torch.matmul(A, B, out=C) 
#     torch.cuda.synchronize()
# print('FP16:', (time.time() - tick) / COUNT *1000,'ms')
# t_fp16 = (time.time() - tick) / COUNT

# AA = torch.randn((4096, 1), device=DEV, dtype=DTYPE)

DTYPE = torch.half
group_size = 128
pack_num = 8 # int32 = 4b * 8
B = torch.randn((N, int(K/pack_num)), device=DEV, dtype=DTYPE)
A = torch.randn((1, M, K), device=DEV, dtype=DTYPE)
C = torch.zeros((1, M, N), device=DEV, dtype=DTYPE)
DTYPE = torch.half
B = B.to(torch.int)
A = A.to(DTYPE)
C = C.to(DTYPE)


B = torch.randint(-1000000000, 1000000000, (N, int(K/pack_num)), device=DEV, dtype=torch.int)
scales = torch.randn((N, int(K/group_size)), device=DEV, dtype=DTYPE)
# zeros = torch.randn(N, device=DEV, dtype=torch.int)
zeros = torch.ones((N, int(K/group_size/pack_num)), device=DEV, dtype=torch.int)



W = torch.rand((N,N), device=DEV)
X = torch.rand((1,N), device=DEV).half()
Y = torch.zeros((1,N), device=DEV).half()

# ol_ratio = [0.0003, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.01, 0.02, 0.05, 0.1]
ol_ratio = [0.002]
for ol in ol_ratio:
    print(f'============= {ol*100}% =============')
    idx = int(N * (1-ol))-1
    thres = W.sort(dim=1).values[:,idx].reshape(N,1)

    # idx = int(N * N * (1-ol))
    # thres = W.flatten().sort().values[idx]
    # print(thres)
    mask = W > thres
    # print(mask)
    W_coo = W * mask.half()
    W_csr = W_coo.to_sparse_csr()

    rows = W_csr.crow_indices().int()
    cols = W_csr.col_indices().int()
    spmat = W_csr.values().half()
    num_rows = N
    # print(rows)
    # print(rows.size())
    # print(cols)
    # print(cols.size())
    # print(spmat)
    # print(spmat.size())
    # print(num_rows)
    # print(X)

    # Y = mxq_inference_engine.spmv_forward_cuda(X, rows, cols, spmat, num_rows, 1)
    # Y_check = torch.sparse.mm(W_csr.cpu(),X.float().cpu())

    # print(Y)
    # print(Y_check)

    for _ in range(1000):
        Y = mxq_inference_engine.gemv_spmv_forward_cuda(A, B, scales, zeros, group_size, X, rows, cols, spmat, num_rows, 1)
        torch.cuda.synchronize()

    COUNT = 100000
    import time
    tick = time.time()

    for _ in range(COUNT):
        Y = mxq_inference_engine.gemv_spmv_forward_cuda(A, B, scales, zeros, group_size, X, rows, cols, spmat, num_rows, 1)
        torch.cuda.synchronize()

    t1 = (time.time() - tick) / COUNT *1000000
    print(f'[Test] awq+sparse_4bit: {t1:.4f}us')


    tick = time.time()

    for _ in range(COUNT):
        Y = mxq_inference_engine.gemv_forward_cuda(A, B, scales, zeros, group_size)
        torch.cuda.synchronize()

    t2 = (time.time() - tick) / COUNT *1000000
    print(f'[Test] awq_4bit: {t2:.4f}us')

    d_ratio = (t1-t2) / t2 *100

    print(f'[Test] speed slow: {d_ratio:.4f}%')
