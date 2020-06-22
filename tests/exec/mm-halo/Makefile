TFLAGS=-O2
CFLAGS=$(TFLAGS)

BUILDDIR ?= .

VERSIONS=$(BUILDDIR)/mm-halo-linalg.generic $(BUILDDIR)/mm-halo-scf.for

all: $(VERSIONS)

$(BUILDDIR)/mm-halo-%: main.c $(BUILDDIR)/mm-halo-%.o
	$(CC) -std=c99 -o $@ $^ $(CFLAGS)

$(BUILDDIR)/mm-halo-%.o: mm-halo.tc
	../../../teckyl-genobject -o $@ $^ $(TFLAGS) --body-op=$*

clean:
	rm -f $(BUILDDIR)/*.o $(VERSIONS)

run:
	for VERSION in $(VERSIONS) ; \
	do \
		$$VERSION || exit 1 ; \
	done
