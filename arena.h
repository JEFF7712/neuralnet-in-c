#ifndef ARENA_H
#define ARENA_H

#include <stdlib.h>

typedef struct {
  unsigned char* buffer;
  size_t capacity;
  size_t offset;
} Arena;

void arena_init(Arena *a, size_t size);
void* arena_alloc(Arena *a, size_t size, size_t align);
void arena_reset(Arena *a);

#endif