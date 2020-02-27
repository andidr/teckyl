#ifndef MEMREF_H
#define MEMREF_H

#include <stdint.h>
#include <stdlib.h>

/* Data layout information for a 2d float memref */
struct vec_f2d {
	float *allocatedPtr;
	float *alignedPtr;
	int64_t offset;
	int64_t sizes[2];
	int64_t strides[2];
};

/* Allocates and initializes a 2d float memref. Returns 0 on success,
 * otherwise 1.
 */
static inline int vec_f2d_alloc(struct vec_f2d* v, size_t n, size_t m)
{
	float* f;

	if(!(f = calloc(n*m, sizeof(float))))
		return 1;

	v->allocatedPtr = f;
	v->alignedPtr = f;
	v->offset = 0;
	v->sizes[0] = n;
	v->sizes[1] = m;
	v->strides[0] = m;
	v->strides[1] = 1;

	return 0;
}

/* Destroys a 2d float memref */
static inline int vec_f2d_destroy(struct vec_f2d* v)
{
	free(v->allocatedPtr);
}

/* Returns the element at position (`x`, `y`) of a 2d float memref `v` */
static inline float vec_f2d_get(const struct vec_f2d* v, int64_t x, int64_t y)
{
	return *(v->allocatedPtr + y*v->sizes[1] + x);
}

/* Assigns `f` to the element at position (`x`, `y`) of a 2d float
 * memref `v`
 */
static inline void vec_f2d_set(struct vec_f2d* v, int64_t x, int64_t y, float f)
{
	*(v->allocatedPtr + y*v->sizes[1] + x) = f;
}

/* Compares the values of two 2d float memrefs. Returns 1 if they are
 * equal, otherwise 0.
 */
static inline int vec_f2d_compare(const struct vec_f2d* a, const struct vec_f2d* b)
{
	/* Compare shapes */
	if(a->sizes[0] != b->sizes[0] ||
	   a->sizes[1] != b->sizes[1])
	{
		return 0;
	}

	/* Compare elements */
	for(int64_t y = 0; y < a->sizes[0]; y++)
		for(int64_t x = 0; x < a->sizes[1]; x++)
			if(vec_f2d_get(a, x, y) != vec_f2d_get(b, x, y))
				return 0;

	return 1;
}

/* Dumps a 2d float memref `v` to stdout. */
static inline void vec_f2d_dump(const struct vec_f2d* v)
{
	for(int64_t y = 0; y < v->sizes[0]; y++) {
		for(int64_t x = 0; x < v->sizes[1]; x++) {
			printf("%f%s",
			       *(v->allocatedPtr + y*v->sizes[1] + x),
			       x == v->sizes[1]-1 ? "" : " ");
		}

		puts("");
	}
}

#endif
