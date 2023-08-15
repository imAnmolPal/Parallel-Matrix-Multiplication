#include<iostream>
// #include<sys/time.h>
#include<cuda.h>
using namespace std;

#define block_Dim 16

__global__ void _matrix_mult_1(int *a, int *b, int *c, int p, int q, int r){
	extern __shared__ int sharedMem[];
	
	/* In row1 we will store entire row of left side matrix */
	int *row1 = sharedMem;
	/* In row2 we will store entire row of right side matrix */
	int *row2 = sharedMem + q;
	/*In result[] we will store the intermediate A[i*k] * B[k*j] */
	int *result = sharedMem + 2*q;

	// Find cordinates of elements of resultant matrix 
	int x_i = blockIdx.x;
	int y_i = blockIdx.y;

	// Left side matrix has order p x q and right side matrix has order r x q
	// In matrix mulitplication AxB, A is left side matrix and B is right side matrix

	int leftSideMatrixElement = a[x_i*q+threadIdx.x];
	int rightSideMatrixELement = b[y_i*q+threadIdx.x];

	row1[threadIdx.x]=leftSideMatrixElement;
	row2[threadIdx.x]=rightSideMatrixELement;

	result[threadIdx.x]=row1[threadIdx.x]*row2[threadIdx.x];
	__syncthreads();
	if(threadIdx.x==0){
		int sum = 0;
		for(int i=0;i<q;i++){
			sum+=result[i];
		}
		c[x_i*r+y_i] = sum;
	}
}
__global__ void _matrix_mult_(int *a, int *b, int *c, int p, int q, int r){
	int row = blockIdx.x;
	int col = threadIdx.x;
	// c[row * r + col] = row * r + col;
	int sum = 0;
	for(int i = 0; i < q; i++) 
    {
        sum += a[row * q + i] * b[i * r + col];
    }
    c[row * r + col] = sum;
}

__global__ void _matrix_transpose(int* d, int* d_trans, int rows, int cols) 
{
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) 
    {
        int pos = row * cols + col;
        int trans_pos = col * rows + row;
        d_trans[trans_pos] = d[pos];
    }
}

__global__ void _matrix_transpose1(int* d, int* d_trans, int rows, int cols) 
{
	extern __shared__ int sharedMem[];
    int x_i = blockIdx.x;
    int y_i = threadIdx.x;

    sharedMem[y_i]= d[blockIdx.x * cols + threadIdx.x];
    __syncthreads();
    d_trans[y_i*rows+x_i]=sharedMem[y_i];
}

__global__ void _add_matrix_(int *f, int *g, int *e, int rows, int cols){
	int row = blockIdx.x;
	int col = threadIdx.x;

	e[row*cols+col]=f[row*cols+col]+g[row*cols+col];
	// e[row*cols+col]=row*cols+col;
}

__global__ void optimisedTranspose(int *d, int* d_trans, int rows, int cols){
	__shared__ int sharedMem[block_Dim*(block_Dim+1)]; // block_dim = block ki dimension

	int row = blockIdx.x * block_Dim + threadIdx.x;
	int col = blockIdx.y * block_Dim + threadIdx.y;

	if((row<rows) && (col<cols)){
		// int trans_Index = col * rows + row;
		// printf("threadx.x=%d, threadIdx.y=%d->[%d][%d]\n",threadIdx.x, threadIdx.y,row, col);
		sharedMem[threadIdx.y*(block_Dim+1)+threadIdx.x] = d[row * cols + col];
		printf("threadx.x=%d, threadIdx.y=%d->%d=[%d,%d]\n",threadIdx.x, threadIdx.y,row * cols + col,row ,col);
	}

	//now we will use synchronise to ensure all all threads have written to shared memory 
	__syncthreads();
	// printf("synced");
	int rowT = blockIdx.y * block_Dim + threadIdx.x;
	int colT = blockIdx.x * block_Dim + threadIdx.y;

	if((rowT < cols) && (colT < rows)){
		int trans_Index = rowT * rows + colT;
		d_trans[trans_Index] = sharedMem[threadIdx.x*(block_Dim+1)+threadIdx.y];
		// printf("threadx.x=%d, threadIdx.y=%d->[%d]\n",threadIdx.x, threadIdx.y,colT * cols + rowT);
		printf("threadx.x=%d, threadIdx.y=%d->%d = [%d][%d]\n",threadIdx.x, threadIdx.y,colT * cols + rowT,rowT, colT);
	}

}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */

	


	/* ****************************************************************** */
	// Let's calculate A * B first and store its result in matrix F.
	int *h_matrixF; // host matrix
	h_matrixF = (int*) malloc(p * r * sizeof(int)); // allocate memory to host matrix

	int *d_matrixF; // device matrix
	cudaMalloc(&d_matrixF, p * r * sizeof(int)); // allocate memory to device matrix 	
	cudaMemcpy(d_matrixF, h_matrixF, p * r * sizeof(int), cudaMemcpyHostToDevice);

	/*We will take transpose of matrix B, in order to exploit coallecsing during multiplication.*/
	int *h_matrixBt;
	h_matrixBt = (int*) malloc(r * q * sizeof(int));

	int *d_matrixBt;
	cudaMalloc(&d_matrixBt, r * q * sizeof(int));
	cudaMemcpy(d_matrixBt, h_matrixBt, q * r * sizeof(int), cudaMemcpyHostToDevice);

	/*Kernel to take transpose where I have exploited shared memory.*/
	_matrix_transpose1<<<q,r, r*sizeof(int) >>>(d_matrixB, d_matrixBt, q, r);	
	cudaMemcpy(h_matrixBt, d_matrixBt, q * r * sizeof(int), cudaMemcpyDeviceToHost);
	
	/*----------------Transpose code I did with the help of
	 internet but didn't make use of it in assignment-----------*/

	// int block_Dim = 16;
	//int x = ceil(float(r)/block_Dim);
	//int y = ceil(float(q)/block_Dim);
	//dim3 grid(x, y, 1);
    //dim3 threads(block_Dim, block_Dim, 1);

	//optimisedTranspose<<<grid, threads>>>(d_matrixD, d_matrixDt, r, q);


	/*Kernel to multiply matrix A and matrix B where I have exploited shared memory
	and concept of coalescing.*/
	
	/* launch the kernel to compute A*B and save its result in matrix F // p=m   q=n   r=k*/

	dim3 numBlock(p,r);
	int numThread = q;
	_matrix_mult_1<<<numBlock,numThread, 3*q*sizeof(int)>>>(d_matrixA, d_matrixBt, d_matrixF, p, q, r);

	// _matrix_mult_<<<p,r>>>(d_matrixA, d_matrixB, d_matrixF, p, q, r);

	/* write code to bring F matrix back to host*/
	cudaMemcpy(h_matrixF, d_matrixF, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// // int block_Dim = 16;
	// int x = ceil(float(r)/block_Dim);
	// int y = ceil(float(q)/block_Dim);
	// dim3 grid(x, y, 1);
    // dim3 threads(block_Dim, block_Dim, 1);

	// optimisedTranspose<<<grid, threads>>>(d_matrixD, d_matrixDt, r, q);	
	

	
	/* launch the kernel to compute A*B and save its result in matrix G*/
	int *h_matrixG; // host matrix
	h_matrixG = (int*) malloc(p * r * sizeof(int)); // allocate memory to host matrix

	int *d_matrixG; // device matrix
	cudaMalloc(&d_matrixG, p * r * sizeof(int)); // allocate memory to device matrix
	cudaMemcpy(d_matrixG, h_matrixG, p * r * sizeof(int), cudaMemcpyHostToDevice);
	
	// _matrix_mult_<<<p,r>>>(d_matrixC, d_matrixDt, d_matrixG, p, q, r);
	_matrix_mult_1<<<numBlock,numThread, 3*q*sizeof(int)>>>(d_matrixC, d_matrixD, d_matrixG, p, q, r);
	

	/* write code to bring G matrix back to host*/
	cudaMemcpy(h_matrixG, d_matrixG, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	/* Now we will launch the kernel to add AB + CD and store it in matrix E. */
	
	_add_matrix_<<<p,r>>>(d_matrixF, d_matrixG,d_matrixE, p, r);
	
	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);
	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	// struct timeval t1, t2;
	// double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	// gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	// cudaDeviceSynchronize();
	// gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	// seconds = t2.tv_sec - t1.tv_sec;
	// microSeconds = t2.tv_usec - t1.tv_usec;
	// printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
