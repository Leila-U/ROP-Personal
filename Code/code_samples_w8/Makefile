MPICC?=mpicc

EXECS=hello cannon_matmul cannon_matmul_nonblocking bcast reduce_avg reduce_avg_allreduce prefix_sums scatter_gather_avg scatter_allgather_avg

all: ${EXECS}

hello: hello.c
	${MPICC} -o $@ $^

cannon_matmul: cannon_matmul.c
	${MPICC} -o $@ $^ -lm

cannon_matmul_nonblocking: cannon_matmul_nonblocking.c
	${MPICC} -o $@ $^ -lm

bcast: bcast.c
	${MPICC} -o $@ $^

reduce_avg: reduce_avg.c
	${MPICC} -o $@ $^

reduce_avg_allreduce: reduce_avg_allreduce.c
	${MPICC} -o $@ $^

prefix_sums: prefix_sums.c
	${MPICC} -o $@ $^

scatter_gather_avg: scatter_gather_avg.c
	${MPICC} -o $@ $^

scatter_allgather_avg: scatter_allgather_avg.c
	${MPICC} -o $@ $^

clean:
	rm -f ${EXECS}
