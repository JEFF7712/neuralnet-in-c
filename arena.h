#ifndef ARENA_H
#define ARENA_H

#include <stdlib.h>

typedef struct ArenaBlock {
  unsigned char* buffer;
  size_t capacity;
  size_t offset;
  struct ArenaBlock* next;
} ArenaBlock;

typedef struct Arena {
  ArenaBlock* head;
  ArenaBlock* current;
} Arena;

void arena_init(Arena *a, size_t size);
void* arena_alloc(Arena *a, size_t size, size_t align);
void* arena_calloc(Arena *a, size_t size, size_t align);
void arena_reset(Arena *a);
void arena_destroy(Arena *a);

#endif