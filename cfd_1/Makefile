BUILDDIR := build
SRCDIR := src
EXECT := $(BUILDDIR)/bin

CC=mpicc
# CFLAGS=-O3 -Wall -g -pg 
# CFLAGS=-O3 -Wall -g -fopenmp
CFLAGS=-O3 -Wall -g

$(shell mkdir -p $(EXECT))


all: bin2ppm diffbin pingpong colcopy karman

clean:
	rm -rf $(BUILDDIR)

karman: $(BUILDDIR)/alloc.o $(BUILDDIR)/boundary.o $(BUILDDIR)/init.o $(BUILDDIR)/karman.o $(BUILDDIR)/simulation.o
	$(CC) $(CFLAGS) -o $(EXECT)/$@ $^ -lm

karman-par: $(BUILDDIR)/alloc.o $(BUILDDIR)/boundary.o $(BUILDDIR)/init.o $(BUILDDIR)/karman-par.o $(BUILDDIR)/simulation.o
	$(CC) $(CFLAGS) -o $(EXECT)/$@ $^ -lm

bin2ppm: $(BUILDDIR)/bin2ppm.o $(BUILDDIR)/alloc.o
	$(CC) $(CFLAGS) -o $(EXECT)/$@ $^ -lm

diffbin: $(SRCDIR)/diffbin.c
	$(CC) $(CFLAGS) -o $(EXECT)/$@ $^ -lm

pingpong: $(BUILDDIR)/pingpong.o
	$(CC) $(CFLAGS) -o $(EXECT)/$@ $^

colcopy: $(BUILDDIR)/colcopy.o $(BUILDDIR)/alloc.o
	$(CC) $(CFLAGS) -o $(EXECT)/$@ $^


$(BUILDDIR)/%.o : $(SRCDIR)/%.c
	@echo compiling $<
	$(maketargetdir)
	$(CC) $(CFLAGS) -o $@ -c $<

define maketargetdir
	-@mkdir -p $(dir $@) > /dev/null 2>&1
endef

