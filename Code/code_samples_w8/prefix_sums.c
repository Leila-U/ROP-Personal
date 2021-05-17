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
#include <mpi.h>
#include <assert.h>
#include <time.h>

/* Generate an array of float numbers, where each item has a value
   between rank*num_items and (rank+1)*num_items */
float *create_local_array(int num_items, int rank) {
	int i;
	float *items = (float *)malloc(sizeof(float) * num_items);
	for(i = 0; i < num_items; i++) {
		items[i] = rank * num_items + i + 1;
	}
	return items;
}

int main(int argc, char** argv) {

	float *array, global_sum, local_sum;
	int num_items_per_proc, my_rank, comm_size, i;
 	MPI_Init(&argc, &argv);

	if (argc != 2) {
		printf("Usage: %s <num_items_per_process>\n", argv[0]);
		exit(1);
	}
	num_items_per_proc = atoi(argv[1]);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	array = create_local_array(num_items_per_proc, my_rank);

	local_sum = 0;
	for (i = 0; i < num_items_per_proc; i++) {
		local_sum += array[i];
	}

	/* Calculate prefix sums */
	MPI_Scan(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

	printf("\tSum[P%d] = %f\n", my_rank, global_sum);

	free(array);

	MPI_Finalize();
	return 0;
}
