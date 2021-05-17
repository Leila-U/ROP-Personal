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

/* Compute the average of an array of numbers */
float compute_average(float *array, int num_items) {
	int i; float arr_sum = 0;
	for (i = 0; i < num_items; i++) {
		arr_sum += array[i];
	}
	return arr_sum / num_items;
}

int main(int argc, char** argv) {

	float *array=NULL, *array_chunk, *all_chunk_averages=NULL;
	float global_avg, local_avg_chunk;
	int num_items_per_proc, my_rank, comm_size;
 	MPI_Init(&argc, &argv);

	if (argc != 2) {
		printf("Usage: %s <num_items_per_process>\n", argv[0]);
		exit(1);
	}
	num_items_per_proc = atoi(argv[1]);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	if(my_rank == 0) {
		array = create_local_array(num_items_per_proc*comm_size, my_rank);
	}
	/* Array to hold the scattered items */
	array_chunk = malloc(num_items_per_proc * sizeof(float));
  
	/* Scatter master (rank 0) array into the chunk arrays of all procs */
	MPI_Scatter(array, num_items_per_proc, MPI_FLOAT, 
	            array_chunk, num_items_per_proc, MPI_FLOAT, 
	            0, MPI_COMM_WORLD);

	/* Calculate average per chunk */
	local_avg_chunk = compute_average(array_chunk, num_items_per_proc);

	if(my_rank == 0) {
		all_chunk_averages = malloc(comm_size * sizeof(float));	
	}

	/* Gather into rank 0 all the partial averages */
	MPI_Gather(&local_avg_chunk, 1, MPI_FLOAT, 
	           all_chunk_averages, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if(my_rank == 0) {
		/* Calculate the total average from all gathered partial averages. 
		   This only works because each process computed an average on an equal 
	  	 number of elements, otherwise doing this wouldn't be correct. */
		global_avg = compute_average(all_chunk_averages, comm_size);
		printf("Average[P%d] = %f\n", my_rank, global_avg);
	}

	if (my_rank == 0) {
		free(array);
		free(all_chunk_averages);
	}
	free(array_chunk);
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
