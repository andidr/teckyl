TFLAGS=-O2
CFLAGS=$(TFLAGS)

BUILDDIR ?= .

all: $(BUILDDIR)/mm-linalg $(BUILDDIR)/mm-stdloops

$(BUILDDIR)/mm-%: main.c $(BUILDDIR)/mm-%.o
	$(CC) -std=c99 -o $@ $^ $(CFLAGS)

$(BUILDDIR)/mm-linalg.o: mm.tc
	../../../teckyl-genobject -o $@ $^ $(TFLAGS)

$(BUILDDIR)/mm-stdloops.o: mm.tc
	../../../teckyl-genobject -o $@ $^ $(TFLAGS) --force-std-loops

clean:
	rm -f $(BUILDDIR)/*.o $(BUILDDIR)/mm-linalg $(BUILDDIR)/mm-stdloops

run:
	$(BUILDDIR)/mm-linalg
	$(BUILDDIR)/mm-stdloops
