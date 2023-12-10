#ifndef __THREADPOOL__
#define __THREADPOOL__

#include <pthread.h>
#include <stdlib.h>
#include <stdbool.h>

// Queue params.
#define QUEUE_RESIZE_COEF 2
#define QUEUE_SIZE_INIT 128

typedef void (*Job)(void *);
typedef struct Queue Queue;

typedef struct Task {
    Job func;
    void *arg;
} Task;

typedef struct Queue {
    Task **vec;
    size_t len;
    size_t cap;
} Queue;

typedef struct ThreadPool {
    pthread_t *workers;
    pthread_mutex_t running_lock;
    pthread_mutex_t qlock;
    pthread_cond_t new_task;
    pthread_cond_t finished;
    size_t running;
    Queue *tasks;
    size_t len;
    bool exit;
} ThreadPool;

ThreadPool *thpool_new(size_t nthreads);
int thpool_spawn(ThreadPool *pool, Job job, void *arg);
void thpool_wait(ThreadPool *pool);
size_t thpool_running(ThreadPool *pool);
size_t thpool_len(ThreadPool *pool);
void thpool_del(ThreadPool *pool);

#endif // __THREADPOOL__