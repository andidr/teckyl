TFLAGS=-O2
CFLAGS=$(TFLAGS)

BUILDDIR ?= .

VERSIONS=$(BUILDDIR)/mm-linalg.generic $(BUILDDIR)/mm-scf.for

all: $(VERSIONS)

$(BUILDDIR)/mm-%: main.c $(BUILDDIR)/mm-%.o
	$(CC) -std=c99 -o $@ $^ $(CFLAGS)

$(BUILDDIR)/mm-%.o: mm.tc
	../../../teckyl-genobject -o $@ $^ $(TFLAGS) --body-op=$*

clean:
	rm -f $(BUILDDIR)/*.o $(VERSIONS)

run:
	for VERSION in $(VERSIONS) ; \
	do \
		$$VERSION || exit 1 ; \
	done
