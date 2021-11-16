#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TILE_WIDTH 32

__global__ void MatrixMulKernelGlobal(float* d_M, float* d_N, float* d_P, int Md_row, int Md_col, int Nd_col)
{
	//using global memory
	int Row = blockIdx.x * blockDim.x + threadIdx.x;
	int Col = blockIdx.y * blockDim.y + threadIdx.y;
	
	if((Row < Md_row) && (Col < Nd_col))
	{
		float Pvalue = 0;
		for(int k = 0; k < Md_col; k++)
		{
			Pvalue += d_M[Row*Md_col+k] * d_N[k*Nd_col+Col];
		}
		d_P[Row * Nd_col+Col] = Pvalue;
	}
}

int main(int argc, const char **argv)
{
	int Md_row, Md_col, Nd_row, Nd_col;
	if(argc == 5)
	{
		Md_row = atoi(argv[1]);
		Md_col = atoi(argv[2]);
		Nd_row = atoi(argv[3]);
		Nd_col = atoi(argv[4]);
		if(Md_col != Nd_row)
		{
			printf("Invalid matrix format!\n");
			exit(0);
		}
	}
	else{
		printf("usage: %s <M row size> <M col size> <N row size> <N col size>\n", argv[0]);
		exit(0);
	}
	//Host
	float *Md, *Nd, *Pd;
	//Device
	float *d_M, *d_N, *d_P;

	cudaEvent_t start, end;

	float time_ms=0;

	cudaEventCreate(&start);
	cudaEventCreate(&end);


	Md = (float*)malloc(Md_row * Md_col * sizeof(float));
	Nd = (float*)malloc(Nd_row * Nd_col * sizeof(float));
	Pd = (float*)malloc(Md_row * Nd_col * sizeof(float));
	for(int i=0;i<Md_row * Md_col;i++)
	{
		Md[i] = 1.0;
	}
	for(int j=0;j<Nd_row * Nd_col;j++)
	{
		Nd[j] = 1.0;
	}

	cudaMalloc((void **)&d_M, Md_row * Md_col * sizeof(float));
	cudaMalloc((void **)&d_N, Nd_row * Nd_col * sizeof(float));
	cudaMalloc((void **)&d_P, Md_row * Nd_col * sizeof(float));
	cudaMemset(d_M, 0, Md_row * Md_col * sizeof(float));
	cudaMemset(d_N, 0, Nd_row * Nd_col * sizeof(float));
	cudaMemset(d_P, 0, Md_row * Nd_col * sizeof(float));
	cudaMemcpy(d_M, Md, Md_row * Md_col * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, Nd, Nd_row * Nd_col * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);

	//Kernel code
	dim3 dimGrid(ceil(Md_row/(TILE_WIDTH*1.0)), ceil(Nd_col/(TILE_WIDTH*1.0)),1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	MatrixMulKernelGlobal<<< dimGrid, dimBlock >>>(d_M, d_N, d_P, Md_row, Md_col, Nd_col);

	cudaEventRecord(end, 0);

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time_ms, start, end);

	cudaDeviceSynchronize();

	cudaMemcpy(Pd, d_P, Md_row * Nd_col * sizeof(float), cudaMemcpyDeviceToHost);
		
	printf("Accesses to Global Memory\n");
	printf("TILE WIDTH : %d\n", TILE_WIDTH);
	printf("(%d x %d),(%d x %d) Matrix\n\n",Md_row, Md_col, Nd_row, Nd_col);
	printf("Execution time for kernel : %f ms\n", time_ms);
	
	float result = (float)Md_col;
	
	for(int i=0;i<Md_row;i++)
	{
		for(int j=0;j<Nd_col;j++)
		{
			if(result!=Pd[i*Nd_col+j])
			{
				printf("Wrong answer\n");
				goto quit;
			}
		}
	}
quit:
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
	free(Md);
	free(Nd);
	free(Pd);

	return 0;
}
