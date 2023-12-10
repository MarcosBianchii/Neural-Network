#include "threadpool.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

#define QUEUE_RESIZE_COEF 2
#define QUEUE_SIZE_INIT 128

typedef struct Task {
    Job func;
    void *arg;
} Task;

typedef struct Queue {
    Task **vec;
    size_t len;
    size_t cap;
} Queue;

Task *
task_new(Job func, void *arg)
{
    Task *task = malloc(sizeof(Task));
    if (!task) return NULL;

    task->func = func;
    task->arg = arg;
    return task;

}

void
task_del(Task *task)
{
    free(task);
}

Queue *
queue_new(size_t n)
{
    Queue *queue = malloc(sizeof(Queue));
    if (!queue) return NULL;

    queue->vec = malloc(sizeof(Task) * n);
    if (!queue->vec) {
        free(queue);
        return NULL;
    }

    queue->len = 0;
    queue->cap = n;
    return queue;
}

int
queue_resize(Queue *queue, size_t new_cap)
{
    if (new_cap < queue->len) {
        return -1;
    }

    queue->vec = realloc(queue->vec, sizeof(Task) * new_cap);
    queue->cap = new_cap;
    return 0;
}

int
queue_push(Queue *queue, Task *task)
{
    if (queue->len == queue->cap) {
        if (queue_resize(queue, QUEUE_RESIZE_COEF * queue->cap) == -1) {
            return 1;
        }
    }

    queue->vec[queue->len++] = task;
    return 0;
}

bool
queue_empty(Queue *queue)
{
    return queue->len == 0;
}

Task *
queue_pop(Queue *queue)
{
    if (queue_empty(queue)) {
        return NULL;
    }

    Task *task = queue->vec[0];
    queue->len--;

    for (size_t i = 0; i < queue->len; i++) {
        queue->vec[i] = queue->vec[i + 1];
    }

    return task;
}

void
queue_del(Queue *queue)
{
    free(queue->vec);
    free(queue);
}

void *
__f(void *__send)
{
    ThreadPool *pool = (ThreadPool *) __send;
    while (1) {
        pthread_mutex_lock(&pool->lock);
        while (pool->tasks->len == 0 && !pool->exit) {
            pthread_cond_wait(&pool->cond, &pool->lock);
        }

        if (!queue_empty(pool->tasks)) {
            Task *task = queue_pop(pool->tasks);
            pthread_mutex_unlock(&pool->lock);

            pthread_mutex_lock(&pool->running_lock);
            pool->running++;
            pthread_mutex_unlock(&pool->running_lock);

            task->func(task->arg);
            task_del(task);

            pthread_mutex_lock(&pool->running_lock);
            pool->running--;
            pthread_mutex_unlock(&pool->running_lock);

            if (threadpool_running(pool) == 0 && queue_empty(pool->tasks)) {
                pthread_cond_signal(&pool->cond);
            }

            if (pool->exit) {
                break;
            }
        }

        else if (pool->exit) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }
    }

    return NULL;
}

// Instanciates a new pool with `n` workers.
// If `n` is 0, it will default to `THREADS_IF_ZERO`.
ThreadPool *
threadpool_new(size_t n)
{
    ThreadPool *pool = calloc(1, sizeof(ThreadPool));
    pool->tasks = queue_new(QUEUE_SIZE_INIT);
    if (!pool || !pool->tasks) {
        free(pool);
        return NULL;
    }

    if (n == 0) n = THREADS_IF_ZERO;
    pthread_mutex_init(&pool->running_lock, NULL);
    pthread_mutex_init(&pool->lock, NULL);
    pthread_cond_init(&pool->cond, NULL);

    for (size_t i = 0; i < n; i++) {
        pool->workers[i] = (Worker) { .handle = 0 };
        pthread_create(&pool->workers[i].handle, NULL, __f, pool);
        pool->len++;
    }

    return pool;
}

// Assigns 'job' to the first available worker.
// Returns 0 on success, 1 if the queue is full or
// the running task could not be created, 2 if the
// parameters are invalid.
int
threadpool_spawn(ThreadPool *pool, Job job, void *arg)
{
    if (!pool || !job || pool->exit)
        return 2;

    Task *task = task_new(job, arg);
    if (!task) return 1;

    pthread_mutex_lock(&pool->lock);

    int res = queue_push(pool->tasks, task);
    pthread_mutex_unlock(&pool->lock);
    pthread_cond_signal(&pool->cond);
    return res;
}

// Returns the amount of containing threads.
size_t
threadpool_len(ThreadPool *pool)
{
    return pool ? pool->len : 0;
}

// Returns the amount of running workers in the pool.
size_t
threadpool_running(ThreadPool *pool)
{
    return pool ? pool->running : 0;
}

// Will block the calling thread until every task
// in the task queue is finished.
void
threadpool_wait(ThreadPool *pool)
{
    if (!pool) return;

    pthread_mutex_lock(&pool->lock);

    while (!queue_empty(pool->tasks) || threadpool_running(pool) > 0) {
        pthread_cond_wait(&pool->cond, &pool->lock);
    }

    pthread_mutex_unlock(&pool->lock);
}

// Free's the memory used by 'pool'.
//
// # Considerations
//
// This function will block until all workers have finished their task.
// It will not wait for the task queue to be empty, just for the running
// workers to finish. Consider calling `threadpool_wait` before this.
void
threadpool_del(ThreadPool *pool)
{
    if (!pool || pool->exit) return;

    pthread_mutex_lock(&pool->lock);
    pool->exit = true;
    pthread_mutex_unlock(&pool->lock);
    pthread_cond_broadcast(&pool->cond);

    for (size_t i = 0; i < pool->len; i++) {
        printf("Joining thread %li\n", i);
        pthread_join(pool->workers[i].handle, NULL);
    }

    pthread_mutex_destroy(&pool->running_lock);
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->cond);
    queue_del(pool->tasks);
    free(pool);
}