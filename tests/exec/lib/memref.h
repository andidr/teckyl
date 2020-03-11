#ifndef MEMREF_H
#define MEMREF_H

#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>

#define DECL_VEC2D_TYPE(NAME, TYPE, FORMAT)				\
	/* Data layout information for a 2d TYPE memref */		\
	struct NAME {							\
		TYPE *allocatedPtr;					\
		TYPE *alignedPtr;					\
		int64_t offset;						\
		int64_t sizes[2];					\
		int64_t strides[2];					\
	};								\
									\
	/* Allocates and initializes a 2d TYPE memref. Returns 0 on	\
	 * success, otherwise 1.					\
	 */								\
	static inline int						\
	NAME##_alloc(struct NAME* v, size_t n, size_t m)		\
	{								\
	        TYPE* f;						\
									\
		if(!(f = calloc(n*m, sizeof(TYPE))))			\
			return 1;					\
									\
		v->allocatedPtr = f;					\
		v->alignedPtr = f;					\
		v->offset = 0;						\
		v->sizes[0] = n;					\
		v->sizes[1] = m;					\
		v->strides[0] = m;					\
		v->strides[1] = 1;					\
									\
		return 0;						\
	}								\
									\
	/* Destroys a 2d TYPE memref */					\
	static inline int NAME##_destroy(struct NAME* v)		\
	{								\
		free(v->allocatedPtr);					\
	}								\
									\
	/* Returns the element at position (`x`, `y`) of a 2d TYPE	\
	 * memref `v` */						\
	static inline TYPE						\
	NAME##_get(const struct NAME* v, int64_t x, int64_t y)	\
	{								\
		return *(v->allocatedPtr + y*v->sizes[1] + x);		\
	}								\
									\
	/* Assigns `f` to the element at position (`x`, `y`) of a 2d	\
	 * TYPE memref `v`						\
	 */								\
	static inline void						\
	NAME##_set(struct NAME* v, int64_t x, int64_t y, TYPE f)	\
	{								\
		*(v->allocatedPtr + y*v->sizes[1] + x) = f;		\
	}								\
									\
	/* Compares the values of two 2d TYPE memrefs. Returns 1 if	\
	 * they are equal, otherwise 0.					\
	 */								\
	static inline int						\
	NAME##_compare(const struct NAME* a, const struct NAME* b)\
	{								\
		/* Compare shapes */					\
		if(a->sizes[0] != b->sizes[0] ||			\
		   a->sizes[1] != b->sizes[1])				\
		{							\
			return 0;					\
		}							\
									\
		/* Compare elements */					\
		for(int64_t y = 0; y < a->sizes[0]; y++) {		\
			for(int64_t x = 0; x < a->sizes[1]; x++) {	\
				if(NAME##_get(a, x, y) !=		\
				   NAME##_get(b, x, y))			\
				{					\
					return 0;			\
				}					\
			}						\
		}							\
									\
		return 1;						\
	}								\
									\
	/* Dumps a 2d TYPE memref `v` to stdout. */			\
	static inline void NAME##_dump(const struct NAME* v)		\
	{								\
		for(int64_t y = 0; y < v->sizes[0]; y++) {		\
			for(int64_t x = 0; x < v->sizes[1]; x++) {	\
				printf(FORMAT "%s",			\
				       *(v->allocatedPtr +		\
					 y*v->sizes[1] + x),		\
				       x == v->sizes[1]-1 ? "" : " ");	\
			}						\
									\
			puts("");					\
		}							\
	}

DECL_VEC2D_TYPE(vec_f2d, float, "%f")
DECL_VEC2D_TYPE(vec_u82d, uint8_t, "%" PRIu8)
DECL_VEC2D_TYPE(vec_u162d, uint16_t, "%" PRIu16)
DECL_VEC2D_TYPE(vec_u322d, uint32_t, "%" PRIu32)
DECL_VEC2D_TYPE(vec_u642d, uint64_t, "%" PRIu64)
DECL_VEC2D_TYPE(vec_i82d, int8_t, "%" PRId8)
DECL_VEC2D_TYPE(vec_i162d, int16_t, "%" PRId16)
DECL_VEC2D_TYPE(vec_i322d, int32_t, "%" PRId32)
DECL_VEC2D_TYPE(vec_i642d, int64_t, "%" PRId64)


/* Generates a comma-separated list of arguments from the fields of a
 * 2d memref */
#define VEC2D_ARGS(v)                           \
  (v)->allocatedPtr,                            \
    (v)->alignedPtr,                            \
    (v)->offset,                                \
    (v)->sizes[0],                              \
    (v)->sizes[1],                              \
    (v)->strides[0],                            \
    (v)->strides[1]


#endif
