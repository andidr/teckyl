TFLAGS=-O2
CFLAGS=$(TFLAGS)

BUILDDIR ?= .

VERSIONS=$(BUILDDIR)/mm-static-linalg.generic $(BUILDDIR)/mm-static-scf.for

all: $(VERSIONS)

$(BUILDDIR)/mm-static-%: main.c $(BUILDDIR)/mm-static-%.o
	$(CC) -std=c99 -o $@ $^ $(CFLAGS)

$(BUILDDIR)/mm-static-%.o: mm-static.tc
	../../../teckyl-genobject -o $@ $^ $(TFLAGS) --body-op=$*

clean:
	rm -f $(BUILDDIR)/*.o $(VERSIONS)

run:
	for VERSION in $(VERSIONS) ; \
	do \
		$$VERSION || exit 1 ; \
	done
