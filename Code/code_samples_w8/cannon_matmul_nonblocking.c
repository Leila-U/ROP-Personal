// ------------
// This code is provided solely for the personal and private use of
// students taking the CSC367H5 course at the University of Toronto.
// Copying for purposes other than this use is expressly prohibited.
// All forms of distribution of this code, whether as given or with
// any changes, are expressly prohibited.
//
// Authors: Bogdan Simion
//
// All of the files in this directory and all subdirectories are:
// Copyright (c) 2019 Bogdan Simion
// -------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/**
 * Multiply matrices A and B and generate the result in C.
 * Each process will generate its own tile of A and tile of B,
 * and calculate the corresponding tile of C by the end of the 
 * algorithm. 
 * The tiles of A, B, and C get printed to files by each of the
 * processes. Once we get to collective operations you'll learn
 * how to collect data from each process (e.g., to reconstruct 
 * the matrix C at the master node (or any other node), or how 
 * to distribute chunks of data to workers.
 *
 * Example sequential run:
 * $ ./cannon_matmul 4 seq
 * OR, for the sake of overkill:
 * $ mpirun -np 1 ./cannon_matmul 4 seq
 * 
 * Example parallel run:
 * $ mpirun -np 4 ./cannon_matmul 4 par
 * => Each process gets a 2x2 tile
 * OR try a larger matrix (each process gets a 4x4 tile):
 * $ mpirun -np 4 ./cannon_matmul 16 par 
 * 
 * Note: make sure you set the hostfile if you want to run it in
 * truly distributed fashion, otherwise this defaults to localhost
 * with the number of cores indicated (simulated via shared memory).
 */

/* Print matrix to a file */
void print_mat(int *m, int size, FILE *f) {
	int i, j;
	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++) {
			fprintf(f, "%d ", m[i*size+j]);
		}
		fprintf(f, "\n");
	}
}

/* Sequential matrix multiplication algorithm for matrices a and b */
void matmul_seq(int *a, int *b, int *c, int size) {
	int i, j, k;

	for(i = 0; i < size; i++) {
		for(j = 0; j < size; j++) {
			for(k = 0; k < size; k++) {
				c[i*size+j] += a[i*size+k] * b[k*size+j];
			}
		}
	}
}

/* Parallel matrix multiplication algorithm using MPI */
void matmul_mpi(int *a, int *b, int *c, int size, MPI_Comm comm) {
	int i, j, comm_size;
	int dims[2], periods[2];
	int myrank, mycartrank, mycartcoords[2];
	int uprank, downrank, leftrank, rightrank, coords[2];
	int shiftsrc, shiftdst;
	MPI_Status status;
	MPI_Comm cart_comm;
	MPI_Request reqs[4];
	int *a_buf[2], *b_buf[2];

	/* Get the communicator related information */
	MPI_Comm_size(comm, &comm_size);
	MPI_Comm_rank(comm, &myrank);

	/* Dimensions for Cartesian topology: sqrt(N) x sqrt(N), where N is the 
	   communicator size (the number of processes in the communicator) 
		This does assume here that comm_size is a square number (e.g., 4), 
		otherwise some small adjustments must be made. 
	*/
	dims[0] = dims[1] = sqrt(comm_size);

	/* The 2D Grid is periodic (wraps around) in both directions */
	periods[0] = periods[1] = 1;

	/* Create Cartesian communicator in cart_comm, allow MPI to reorder ranks (reorder=1) */
	MPI_Cart_create(comm, 2, dims, periods, 1, &cart_comm);

	/* Retrieve my rank and coordinates in the Cartesian topology */
	MPI_Comm_rank(cart_comm, &mycartrank);
	MPI_Cart_coords(cart_comm, mycartrank, 2, mycartcoords);
	
	/* Retrieve ranks of the up/down and left/right shifts */
	MPI_Cart_shift(cart_comm, 1, 1, &leftrank, &rightrank); 
	MPI_Cart_shift(cart_comm, 0, 1, &uprank, &downrank); 

	/* Setup the buffers for async communication */
	a_buf[0] = a;
	a_buf[1] = (int*) malloc(size*size*sizeof(int));
	b_buf[0] = b;
	b_buf[1] = (int*) malloc(size*size*sizeof(int));

	/* Get to initial configuration, by shifting A and B as discussed in class. */
	MPI_Cart_shift(cart_comm, 1, -mycartcoords[0], &shiftsrc, &shiftdst);
	MPI_Sendrecv_replace(a_buf[0], size*size, MPI_INT, 
	                     shiftdst, 1, shiftsrc, 1, cart_comm, &status);

	MPI_Cart_shift(cart_comm, 0, -mycartcoords[1], &shiftsrc, &shiftdst);
	MPI_Sendrecv_replace(b_buf[0], size*size, MPI_INT,
	                     shiftdst, 1, shiftsrc, 1, cart_comm, &status);

	/* Go through sqrt(N) steps, multiplying a pair of chunks at each step, then shifting */
	for(i = 0; i < dims[0]; i++) {

		/* Send and Recv a tile asynchronously from one a_buf to another (modulo the iteration)
		to the left, and from one b_buf to another upwards. 
		This creates 4 asynchronous requests, which we wait upon later to ensure completion,
		before moving onto the next iteration. */
		MPI_Isend(a_buf[i%2], size*size, MPI_INT,
		          leftrank, 1, cart_comm, &reqs[0]);
		MPI_Isend(b_buf[i%2], size*size, MPI_INT,
		          uprank, 1, cart_comm, &reqs[1]);
		MPI_Irecv(a_buf[(i+1)%2], size*size, MPI_INT,
		          rightrank, 1, cart_comm, &reqs[2]);
		MPI_Irecv(b_buf[(i+1)%2], size*size, MPI_INT,
		          downrank, 1, cart_comm, &reqs[3]);

		/* Multiply a and b chunks sequentially into c chunks */
		matmul_seq(a_buf[i%2], b_buf[i%2], c, size);

		for(j = 0; j < 4; j++) {
			MPI_Wait(&reqs[j], &status);
		}
	}

	/* Restore the initial layout of a and b */
	MPI_Cart_shift(cart_comm, 1, mycartcoords[0], &shiftsrc, &shiftdst);
	MPI_Sendrecv_replace(a_buf[i%2], size*size, MPI_INT,
	                     shiftdst, 1, shiftsrc, 1, cart_comm, &status);

	MPI_Cart_shift(cart_comm, 0, mycartcoords[1], &shiftsrc, &shiftdst);
	MPI_Sendrecv_replace(b_buf[i%2], size*size, MPI_INT, 
	                     shiftdst, 1, shiftsrc, 1, cart_comm, &status);

	free(a_buf[1]);
	free(b_buf[1]);	
	MPI_Comm_free(&cart_comm);
}


int main(int argc, char *argv[]) {

	int *a, *b, *c;
	int mat_size;
	int i, j, par;
	double start, stop;
	int myrank, comm_size, chunksz, dim;
	char stra[100], strb[100], strc[100];
	FILE *fa, *fb, *fc;

	MPI_Init(&argc, &argv);

	if(argc != 3) {
		printf("Usage: %s <matrix_size> <seq|par>\n", argv[0]);
		MPI_Finalize();
		exit(1);
	}

	if(!strcmp(argv[2], "seq")) {
		par = 0;
	}
	else if(!strcmp(argv[2], "par")){
		par = 1;
	}
	else {
		printf("Undefined argument: %s\n", argv[2]);
		MPI_Finalize();
		exit(1);
	}

	mat_size = atoi(argv[1]);

	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	/* Make each process write its chunk to a file, for fun... */
	sprintf(stra, "Input_A_%d.txt", myrank);
	sprintf(strb, "Input_B_%d.txt", myrank);
	sprintf(strc, "Output_C_%d.txt", myrank);
	fa = fopen(stra, "w");
	fb = fopen(strb, "w");
	fc = fopen(strc, "w");
	if(fa == NULL || fb == NULL || fc == NULL) {
		printf("Rank %d cannot open file\n", myrank);
		MPI_Finalize();
		exit(1);
	}

	/* Warning: we assume the comm_size is a square, for the purpose of
	this example. Feel free to rewrite this in a more generalizable way. */
	dim = (int)sqrt(comm_size);
	chunksz = mat_size / dim;
	a = (int*)malloc(chunksz*chunksz*sizeof(int*));
	b = (int*)malloc(chunksz*chunksz*sizeof(int*));
	c = (int*)malloc(chunksz*chunksz*sizeof(int*));

	/* Warning: This is just an example of how to generate a and b chunks 
	Depending on the number of processes, you have to really be careful 
	to see what the tiles look like, to determine if the final result 
	(the matrix C tiles) is correct! */
	for(i = 0; i < chunksz; i++) {
		for(j = 0; j < chunksz; j++) {
			int x = myrank / dim;
			int y = myrank % dim;
			/* Feel free to change this to other matrices */
			a[i*chunksz+j] = i + j + (x+y)*chunksz; 
			b[i*chunksz+j] = 1; 
			c[i*chunksz+j] = 0;
		}
	}	

	/* Each process writes its chunks of a and b for posterity to see and wonder */
	print_mat(a, chunksz, fa);
	print_mat(b, chunksz, fb);

	/* Wait for everyone before starting the timing */
	MPI_Barrier(MPI_COMM_WORLD);

	/* Time the execution only in the master rank (and yes, another rank may technically 
	be way ahead when the master starts the timer, but in practice that's not a big deal) */
	if (myrank == 0) {
		start = MPI_Wtime();
	}

	if (par) {
		matmul_mpi(a, b, c, chunksz, MPI_COMM_WORLD);
	}
	else {
		matmul_seq(a, b, c, mat_size);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (myrank == 0) {	
		stop = MPI_Wtime();
		printf("Elapsed time: %lf\n", stop-start);
	}
	
	/* Each process writes its amazing achievement - a tile of the C result */
	print_mat(c, chunksz, fc);

	free(a);
	free(b);
	free(c);

	MPI_Finalize();
	return 0;
}
