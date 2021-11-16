CC := nvcc

all: tiled_matmul non_tiled_matmul

tiled_matmul : shared_matmul.cu
	$(CC) -o $@ $<

non_tiled_matmul : global_matmul.cu
	$(CC) -o $@ $<

clean:
	rm -f tiled_matmul
	rm -f non_tiled_matmul
