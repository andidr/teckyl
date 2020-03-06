TFLAGS=-O2
CFLAGS=$(TFLAGS)

BUILDDIR ?= .

all: $(BUILDDIR)/mm-halo-linalg $(BUILDDIR)/mm-halo-stdloops

$(BUILDDIR)/mm-halo-%: main.c $(BUILDDIR)/mm-halo-%.o
	$(CC) -std=c99 -o $@ $^ $(CFLAGS)

$(BUILDDIR)/mm-halo-linalg.o: mm-halo.tc
	../../../teckyl-genobject -o $@ $^ $(TFLAGS)

$(BUILDDIR)/mm-halo-stdloops.o: mm-halo.tc
	../../../teckyl-genobject -o $@ $^ $(TFLAGS) --force-std-loops

clean:
	rm -f $(BUILDDIR)/*.o $(BUILDDIR)/mm-halo-linalg $(BUILDDIR)/mm-halo-stdloops

run:
	$(BUILDDIR)/mm-halo-linalg
	$(BUILDDIR)/mm-halo-stdloops
