#include "arena.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

void arena_init(Arena *a, size_t size) {
  a->head = (ArenaBlock*)malloc(sizeof(ArenaBlock));
  if (a->head == NULL) {
    fprintf(stderr, "Fatal: Failed to allocate ArenaBlock\n");
    exit(1);
  }
  
  a->head->buffer = (unsigned char*)malloc(size);
  if (a->head->buffer == NULL) {
    fprintf(stderr, "Fatal: Failed to allocate Arena buffer\n");
    free(a->head);
    exit(1);
  }
  
  a->head->capacity = size;
  a->head->offset = 0;
  a->head->next = NULL;
  
  a->current = a->head;
}

void* arena_alloc(Arena *a, size_t size, size_t align) {
  assert((align & (align - 1)) == 0);
  
  while (1) {
    uintptr_t curr_ptr = (uintptr_t)a->current->offset + (uintptr_t)a->current->buffer;
    uintptr_t aligned_ptr = (curr_ptr + align - 1) & ~(align - 1);
    size_t padding = (size_t)(aligned_ptr - curr_ptr);
    
    if (size + a->current->offset + padding <= a->current->capacity) {
      size_t aligned_offset = a->current->offset + padding;
      a->current->offset = aligned_offset + size;
      return &a->current->buffer[aligned_offset];
    }
    
    if (a->current->next == NULL) {
      ArenaBlock* new_block = (ArenaBlock*)malloc(sizeof(ArenaBlock));
      if (!new_block) return NULL;
      
      size_t new_cap = (a->head->capacity > size) ? a->head->capacity : size;
      
      new_block->buffer = (unsigned char*)malloc(new_cap);
      if (!new_block->buffer) {
        free(new_block);
        return NULL;
      }
        
        new_block->capacity = new_cap;
        new_block->offset = 0;
        new_block->next = NULL;
        
        a->current->next = new_block;
      }
      
      a->current = a->current->next;
    }
}

void* arena_calloc(Arena *a, size_t size, size_t align) {
  void* ptr = arena_alloc(a, size, align);
  if (ptr) {
    memset(ptr, 0, size);
  }
  return ptr;
}

void arena_reset(Arena *a) {
  ArenaBlock* curr = a->head;
  while (curr != NULL) {
    curr->offset = 0;
    curr = curr->next;
  }
  a->current = a->head;
}

void arena_destroy(Arena *a) {
  ArenaBlock* curr = a->head;
  while (curr != NULL) {
    ArenaBlock* next = curr->next;
    free(curr->buffer);
    free(curr);
    curr = next;
  }
  a->head = NULL;
  a->current = NULL;
}