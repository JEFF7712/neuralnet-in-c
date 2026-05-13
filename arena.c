#include "arena.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

void arena_init(Arena *a, size_t size) {
  a->buffer = (unsigned char*)malloc(size);
  a->capacity = size;
  a->offset = 0;
}

void* arena_alloc(Arena *a, size_t size, size_t align) {
  uintptr_t curr_ptr = (uintptr_t)a->offset + (uintptr_t)a->buffer;
  uintptr_t aligned_ptr = (curr_ptr + align - 1) & ~(align - 1);
  size_t padding = (size_t)(aligned_ptr - curr_ptr);
  
  if (size + a->offset + padding > a->capacity) {
    return NULL;
  }
  
  size_t aligned_offset = a->offset + padding;
  a->offset = aligned_offset + size;
  
  return &a->buffer[aligned_offset];
}

void arena_reset(Arena *a) {
  a->offset = 0;
}

void arena_destroy(Arena *a) {
  free(a->buffer);
  a->buffer = NULL;
  a->capacity = 0;
  a->offset = 0;
}