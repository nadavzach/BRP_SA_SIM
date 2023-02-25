CC = g++
CFLAGS = -g -Wall -O3

all: smt_sa_tb

smt_sa_tb: smt_sa_tb.o smt_sa_os.o grid_os.o node_mem.o node_pu_os.o fifo.o
	$(CC) $(CFLAGS) -o smt_sa_tb smt_sa_tb.o smt_sa_os.o grid_os.o node_mem.o node_pu_os.o fifo.o

smt_sa_tb.o: smt_sa_tb.cpp smt_sa_os.o
	$(CC) $(CFLAGS) -c smt_sa_tb.cpp

smt_sa_os.o: smt_sa_os.cpp grid_os.o node_mem.o node_pu_os.o fifo.o
	$(CC) $(CFLAGS) -c smt_sa_os.cpp

fifo.o: fifo.cpp
	$(CC) $(CFLAGS) -c fifo.cpp

node_pu_os.o: node_pu_os.cpp fifo.o
	$(CC) $(CFLAGS) -c node_pu_os.cpp

node_mem.o: node_mem.cpp fifo.o node.o
	$(CC) $(CFLAGS) -c node_mem.cpp

grid_os.o: grid_os.cpp node_pu_os.o node_mem.o
	$(CC) $(CFLAGS) -c grid_os.cpp

node.o: node.cpp fifo.o
	$(CC) $(CFLAGS) -c node.cpp

clean:
	$(RM) count *.o *~

