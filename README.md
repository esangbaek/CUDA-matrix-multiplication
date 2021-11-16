# Locality and Tiled Matrix Multiplication

## CUDA-matrix-multiplication
Matrix multiplication using cuda shared memory

&nbsp;

To implement a tiled dense matrix multiplication using shared memory

To learn to assess the benefit of tiling

&nbsp;

### How to make executable file and run

```bash
$ ls
Makefile global_matmul.cu shared_matmul.cu test.sh

$ make all

non-tiled ( Accesses to cuda global memory )
$ ./non_tiled_matmul <M row size> <M col size> <N row size> <N col size>

[example]
$ ./non_tiled_matmul 1024 512 512 1024

tiled ( Accesses to cuda shared memory )
$ ./tiled_matmul <M row size> <M col size> <N row size> <N col size>

[example]
$ ./tiled_matmul 4096 2048 2048 4096
```

&nbsp;

### Quick test

```bash
$ bash test.sh

Automatically run non_tiled_matmul and tiled_matmul using argument 
32 32 32 32
256 256 256 256
4096 2048 2048 4096
```



### About code

- 계산 후 검증의 편의를 위해 행렬의 모든 성분을 1로 만들어서 계산하였음

  성공적인 계산 : 곱셈할 첫번째 행렬의 column개수 = 결과 행렬 각 성분의 값
