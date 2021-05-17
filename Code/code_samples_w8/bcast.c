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
#include <assert.h>
#include <mpi.h>

void custom_bcast(void* data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator) {

	int rank, size;
	MPI_Comm_rank(communicator, &rank);
	MPI_Comm_size(communicator, &size);

	if(rank == root) {
		/* The root process sends the data to everyone */
		int i;
		for (i = 0; i < size; i++) {
			if (i != rank) {
				MPI_Send(data, count, datatype, i, 0, communicator);
			}
		}
	} 
	else {
		/* A regular receiver process will receive data from the root */
		MPI_Recv(data, count, datatype, root, 0, communicator, MPI_STATUS_IGNORE);
	}
}

int main(int argc, char** argv) {

	int *data;
	int num_items, num_runs, my_rank, i;
	double custom_bcast_time = 0, mpi_bcast_time = 0;

	MPI_Init(NULL, NULL);

	if (argc != 3) {
		printf("Usage: %s <number_of_items> <number_of_runs>\n", argv[0]);
		exit(1);
	}

	num_items = atoi(argv[1]);
	num_runs = atoi(argv[2]);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	data = (int*)malloc(num_items * sizeof(int));
	if(data == NULL) {
		printf("Uh-oh, out of memory, get me outta here...\n");
		exit(1);	
	}
	
	for(i = 0; i < num_items; i++) {
		data[i] = i;
	}

	for(i = 0; i < num_runs; i++) {
		/* Must ensure all processes get to this, before starting timer */
		MPI_Barrier(MPI_COMM_WORLD);
		custom_bcast_time -= MPI_Wtime();
		custom_bcast(data, num_items, MPI_INT, 0, MPI_COMM_WORLD);
		// Synchronize again before obtaining final time
		MPI_Barrier(MPI_COMM_WORLD);
		custom_bcast_time += MPI_Wtime();

		MPI_Barrier(MPI_COMM_WORLD);
		mpi_bcast_time -= MPI_Wtime();
		MPI_Bcast(data, num_items, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		mpi_bcast_time += MPI_Wtime();
	}

	/* Master prints the timings */
	if (my_rank == 0) {
		printf("Average custom_bcast time = %lf\n", custom_bcast_time / num_runs);
		printf("Average MPI_Bcast time    = %lf\n", mpi_bcast_time / num_runs);
	}

	free(data);
	MPI_Finalize();
	return 0;
}

