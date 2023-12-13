#include "threadpool.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static Task *
task_new(Job func, void *arg)
{
    Task *task = malloc(sizeof(Task));
    if (!task) return NULL;

    task->func = func;
    task->arg = arg;
    return task;

}

static void
task_del(Task *task)
{
    free(task);
}

static Queue *
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

static int
queue_resize(Queue *queue, size_t new_cap)
{
    if (new_cap < queue->len) {
        return -1;
    }

    queue->vec = realloc(queue->vec, sizeof(Task) * new_cap);
    queue->cap = new_cap;
    return 0;
}

static int
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

static bool
queue_empty(Queue *queue)
{
    return queue->len == 0;
}

static Task *
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

static void
queue_del(Queue *queue)
{
    for (size_t i = 0; i < queue->len; i++) {
        task_del(queue->vec[i]);
    }

    free(queue->vec);
    free(queue);
}

static void *
__f(void *__send)
{
    ThreadPool *pool = (ThreadPool *) __send;
    while (1) {
        pthread_mutex_lock(&pool->work_lock);
        while (queue_empty(pool->tasks) && !pool->exit) {
            pthread_cond_wait(&pool->new_task, &pool->work_lock);
        }

        if (pool->exit) {
            pthread_mutex_unlock(&pool->work_lock);
            break;
        }

        if (!queue_empty(pool->tasks)) {
            pool->running++;

            // Get task from queue.
            Task *task = queue_pop(pool->tasks);
            pthread_mutex_unlock(&pool->work_lock);

            task->func(task->arg);
            task_del(task);

            // Decrement running tasks count.
            pthread_mutex_lock(&pool->work_lock);
            pool->running--;
        }

        // Signal that this task has finished.
        pthread_cond_signal(&pool->finished);
        pthread_mutex_unlock(&pool->work_lock);
    }

    return NULL;
}

// Instanciates a new pool with `n` workers.
ThreadPool *
thpool_new(size_t nthreads)
{
    if (nthreads == 0) return NULL;

    ThreadPool *pool = calloc(1, sizeof(ThreadPool));
    if (!pool) return NULL;

    // Queue is the same size as the amount of workers.
    Queue *tasks = queue_new(nthreads);
    if (!tasks) {
        free(pool);
        return NULL;
    }

    pthread_t *workers = malloc(sizeof(pthread_t) * nthreads);
    if (!workers) {
        queue_del(tasks);
        free(pool);
        return NULL;
    }

    // Initialize pool.
    *pool = (ThreadPool) {
        .tasks = tasks,
        .workers = workers,
        .len = nthreads,
    };

    pthread_mutex_init(&pool->work_lock, NULL);
    pthread_cond_init(&pool->new_task, NULL);
    pthread_cond_init(&pool->finished, NULL);

    for (size_t i = 0; i < nthreads; i++) {
        pthread_create(&pool->workers[i], NULL, __f, pool);
    }

    return pool;
}

// Assigns 'job' to the first available worker. Returns `0` on success and `1` on failure.
int
thpool_spawn(ThreadPool *pool, Job job, void *arg)
{
    if (!pool || !job || pool->exit)
        return 1;

    Task *task = task_new(job, arg);
    if (!task) return 1;

    pthread_mutex_lock(&pool->work_lock);
    int res = queue_push(pool->tasks, task);
    pthread_cond_signal(&pool->new_task);
    pthread_mutex_unlock(&pool->work_lock);
    return res;
}

// Returns the amount of running workers in the pool.
size_t
thpool_running(ThreadPool *pool)
{
    return pool ? pool->running : 0;
}

// Returns the amount of containing threads.
size_t
thpool_len(ThreadPool *pool)
{
    return pool ? pool->len : 0;
}

// Will block the calling thread until every task
// in the queue is finished.
void
thpool_wait(ThreadPool *pool)
{
    if (!pool) return;

    pthread_mutex_lock(&pool->work_lock);

    while (!queue_empty(pool->tasks) || thpool_running(pool) > 0) {
        pthread_cond_wait(&pool->finished, &pool->work_lock);
    }

    pthread_mutex_unlock(&pool->work_lock);
}

// Prints the state of the pool.
void
thpool_show(ThreadPool *pool)
{
    printf("pool: %p\n", pool);
    if (!pool) return;

    printf("    running: %li\n", thpool_running(pool));
    printf("    len: %li\n", thpool_len(pool));
    printf("    exit: %s\n", pool->exit ? "true" : "false");

    Queue *queue = pool->tasks;
    printf("queue: %p\n", queue);
    if (!queue) return;

    printf("    tasks: %li\n", queue->len);
    printf("    capacity: %li\n", queue->cap);
}

// Free's the memory used by 'pool'.
//
// # Considerations
//
// This function will block until all workers have finished their task.
// It will not wait for the task queue to be empty, just for the running
// workers to finish. Consider calling `threadpool_wait` before this.
void
thpool_del(ThreadPool *pool)
{
    if (!pool || pool->exit) return;

    pthread_mutex_lock(&pool->work_lock);
    pool->exit = true;
    pthread_cond_broadcast(&pool->new_task);
    pthread_mutex_unlock(&pool->work_lock);

    for (size_t i = 0; i < pool->len; i++) {
        pthread_join(pool->workers[i], NULL);
    }

    pthread_mutex_destroy(&pool->work_lock);
    pthread_cond_destroy(&pool->new_task);
    pthread_cond_destroy(&pool->finished);
    queue_del(pool->tasks);
    free(pool);
}
