CC = mpicc
CFLAGS = -O2 -Wall -Wextra
H5_DIR = /path/to/hdf5/install

INCLUDES = -I. -I$(H5_DIR)/include
LDFLAGS = -L$(H5_DIR)/lib
LIBS = -lhdf5 -lm

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<

all: hdf5_noncontig

hdf5_noncontig: hdf5_noncontig.o random.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	rm -f *.o hdf5_noncontig
