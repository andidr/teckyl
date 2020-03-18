#include <stdio.h>
#include <string.h>
#include "../lib/memref.h"

#define DECL_MM_TEST(SUFFIX, TYPE, VECTYPE)				\
	/* Generated matrix multiplication function under test */	\
	extern void mm_##SUFFIX(					\
		const TYPE* a_allocatedptr, const TYPE* a_alignedptr,	\
		int64_t a_offset, int64_t a_sizes0, int64_t a_sizes1,	\
		int64_t a_strides0, int64_t a_strides1,			\
									\
		const TYPE* b_allocatedptr, const TYPE* b_alignedptr,	\
		int64_t b_offset, int64_t b_sizes0, int64_t b_sizes1,	\
		int64_t b_strides0, int64_t b_strides1,			\
									\
		TYPE* o_allocatedptr, TYPE* o_alignedptr,		\
		int64_t o_offset, int64_t o_sizes0, int64_t o_sizes1,	\
		int64_t o_strides0, int64_t o_strides1);		\
									\
	/* Reference implementation of a matrix multiplication */	\
	void mm_refimpl_##SUFFIX(const struct VECTYPE* a,		\
				 const struct VECTYPE* b,		\
				 struct VECTYPE* o)			\
	{								\
		TYPE accu;						\
									\
		for(int64_t y = 0; y < o->sizes[0]; y++) {		\
			for(int64_t x = 0; x < o->sizes[1]; x++) {	\
				accu = 0;				\
									\
				for(int64_t k = 0; k < a->sizes[1]; k++){\
					accu += VECTYPE##_get(a, k, y) *\
						VECTYPE##_get(b, x, k); \
				}					\
									\
				VECTYPE##_set(o, x, y, accu);		\
			}						\
		}							\
	}								\
									\
	/* Initialize matrix with value x+y at position (x, y) */	\
	void init_matrix_##SUFFIX(struct VECTYPE* m)			\
	{								\
		for(int64_t y = 0; y < m->sizes[0]; y++)		\
			for(int64_t x = 0; x < m->sizes[1]; x++)	\
				VECTYPE##_set(m, x, y, x+y);		\
	}								\
									\
	/* Executes the implementation under test and compares the */	\
	/* result with the reference implementation. If the results */	\
	/* differ, an error message is displayed o,n stderr and the */	\
	/* process is killed with a nonzero exit code. */		\
	int test_##SUFFIX(int verbose)					\
	{								\
		struct VECTYPE a, b, o, o_ref;				\
		int n = 6;						\
		int k = 9;						\
		int m = 12;						\
									\
		if(VECTYPE##_alloc(&a, n, k) ||				\
		   VECTYPE##_alloc(&b, k, m) ||				\
		   VECTYPE##_alloc(&o, n, m) ||				\
		   VECTYPE##_alloc(&o_ref, n, m))			\
		{							\
		        fputs("Allocation failed [" #SUFFIX "]",	\
			      stderr);					\
		        exit(1);					\
		}							\
									\
		init_matrix_##SUFFIX(&a);				\
		init_matrix_##SUFFIX(&b);				\
									\
		if(verbose) {						\
			puts("A [" #SUFFIX "]:");			\
			VECTYPE##_dump(&a);				\
			puts("");					\
									\
			puts("B [" #SUFFIX "]:");			\
			VECTYPE##_dump(&b);				\
			puts("");					\
									\
			puts("O [" #SUFFIX "]:");			\
			VECTYPE##_dump(&o);				\
			puts("");					\
		}							\
									\
		mm_##SUFFIX(VEC2D_ARGS(&a),				\
			    VEC2D_ARGS(&b),				\
			    VEC2D_ARGS(&o));				\
		mm_refimpl_##SUFFIX(&a, &b, &o_ref);			\
									\
		if(verbose) {						\
			puts("Result O [" #SUFFIX "]:");		\
			VECTYPE##_dump(&o);				\
			puts("");					\
									\
			puts("Reference O [" #SUFFIX "]:");		\
			VECTYPE##_dump(&o_ref);				\
			puts("");					\
		}							\
									\
		if(!VECTYPE##_compare(&o, &o_ref)) {			\
			fputs("Result differs from reference result "	\
			      "[" #SUFFIX "]\n", stderr);		\
			exit(1);					\
		}							\
									\
		VECTYPE##_destroy(&a);					\
		VECTYPE##_destroy(&b);					\
		VECTYPE##_destroy(&o);					\
		VECTYPE##_destroy(&o_ref);				\
									\
		return 0;						\
	}								\

DECL_MM_TEST(u8, uint8_t, vec_u82d)
DECL_MM_TEST(u16, uint16_t, vec_u162d)
DECL_MM_TEST(u32, uint32_t, vec_u322d)
DECL_MM_TEST(u64, uint64_t, vec_u642d)

DECL_MM_TEST(i8, uint8_t, vec_i82d)
DECL_MM_TEST(i16, uint16_t, vec_i162d)
DECL_MM_TEST(i32, uint32_t, vec_i322d)
DECL_MM_TEST(i64, uint64_t, vec_i642d)

DECL_MM_TEST(f32, float, vec_f2d)
DECL_MM_TEST(f64, double, vec_d2d)

void die_usage(const char* program_name)
{
	fprintf(stderr, "Usage: %s [-v]\n", program_name);
	exit(1);
}

int main(int argc, char** argv)
{
	int verbose = 0;

	if(argc > 2)
		die_usage(argv[0]);

	if(argc == 2) {
		if(strcmp(argv[1], "-v") == 0)
			verbose = 1;
		else
			die_usage(argv[0]);
	}

	test_u8(verbose);
	test_u16(verbose);
	test_u32(verbose);
        test_u64(verbose);

	test_i8(verbose);
	test_i16(verbose);
	test_i32(verbose);
        test_i64(verbose);

	return 0;
}
