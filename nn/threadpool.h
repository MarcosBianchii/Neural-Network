#ifndef __THREADPOOL__
#define __THREADPOOL__

#include <pthread.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_THREADS 128
#define THREADS_IF_ZERO 3

typedef void (*Job)(void *);
typedef struct Queue Queue;

typedef struct Worker {
    pthread_t handle;
} Worker;

typedef struct ThreadPool {
    Worker workers[MAX_THREADS];
    pthread_mutex_t running_lock;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    size_t running;
    Queue *tasks;
    size_t len;
    bool exit;
} ThreadPool;

ThreadPool *threadpool_new(size_t n);
int threadpool_spawn(ThreadPool *pool, Job job, void *arg);
void threadpool_wait(ThreadPool *pool);
size_t threadpool_running(ThreadPool *pool);
size_t threadpool_len(ThreadPool *pool);
void threadpool_del(ThreadPool *pool);

#endif // __THREADPOOL__