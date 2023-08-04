#ifndef __SET_H__
#define __SET_H__

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Set {
    double *data, *free_ptr;
    size_t n, m, stride;
} Set;

// Gives an entry point to specific data in the set.
#define SET_AT(set, i, j) (set).data[(i)*(set).stride+(j)]

void set_print_no_nl(Set s, const char *str) {
    printf("\033[0;37m%s: ", str);
    char buff[16];
    for (size_t i = 0; i < s.n; i++) {
        printf("\033[0;30m[\033[0;37m  ");
        for (size_t j = 0; j < s.m; j++) {
            double v = SET_AT(s, i, j);
            snprintf(buff, 6, "%.3lf", v);
            printf(v < 0 ? "%.2lf  " : "%.3lf  ", v);
        }
        printf("\033[0;30m]");
    }
    printf("\033[0;37m");
}

double absf(double v);
void set_print_with_str(Set s, const char *str, size_t u, size_t v) {
    assert(u < v);
    printf("\033[0;37m%s:\n", str);
    char buff[16];
    for (; u < v; u++) {
        printf("\033[0;30m[  ");
        for (size_t j = 0; j < s.m; j++) {
            double v = SET_AT(s, u, j);
            snprintf(buff, 6, "%.3lf", absf(v));
            printf(v < 0 ? "\033[0;31m%s  " : (v == 0 ? "\033[0;30m%s  " : "\033[0;32m%s  "), buff);
        }
        puts("\033[0;30m]");
    }
    puts("\033[0;37m");
}

// Formats and prints s.
#define set_print(s) set_print_with_str(s, #s, 0, (s).n)
// Formats and prints s from rows i to j.
#define set_print_win(s, i, j) set_print_with_str(s, #s, i, j)

// Asserts s is a valid set.
static void set_assert(Set s) {
    assert(s.data != NULL);
}

// Returns an empty set.
static Set set_new(size_t n, size_t m) {
    Set s = {
        .data = calloc(n*m, sizeof(double)),
        .free_ptr = s.data,
        .n = n,
        .m = m,
        .stride = m,
    };

    set_assert(s);
    return s;
}

// Returns a set made from a C style matrix.
Set set_from(size_t n, size_t m, double data[n][m]) {
    Set s = set_new(n, m);
    size_t len = n*m;
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            SET_AT(s, i, j) = data[i][j];
    return s;
}

// Returns a set made from a CSV file.
Set set_from_csv(const char *csv, const char *sep) {
    FILE *f = fopen(csv, "r");
    assert(f != NULL);

    size_t n = 0, m = 0;
    char buffer[1024];
    while (fscanf(f, "%s", buffer) == 1) {
        char *token = strtok(buffer, sep);
        while (token != NULL) {
            if (n == 0) m++;
            token = strtok(NULL, sep);
        }

        n++;
    }

    Set s = set_new(n, m);
    rewind(f);
    for (size_t i = 0; i < n; i++) {
        fscanf(f, "%s", buffer);
        char *token = strtok(buffer, sep);
        for (size_t j = 0; j < m; j++) {
            SET_AT(s, i, j) = atof(token);
            token = strtok(NULL, sep);
        }
    }

    fclose(f);
    return s;
}

// Returns a sub-set of the i'th row of s.
// The returned Set doesn't need to be free'd using set_del().
Set set_row(Set s, size_t i) {
    return (Set) {
        .data = &s.data[i*s.stride],
        .free_ptr = NULL,
        .n = 1,
        .m = s.m,
        .stride = s.stride,
    };
}

// Returns a sub-set of the j'th col of s.
// The returned Set doesn't need to be free'd using set_del().
Set set_col(Set s, size_t j) {
    return (Set) {
        .data = &s.data[j],
        .free_ptr = NULL,
        .n = s.n,
        .m = 1,
        .stride = s.m,
    };
}

// Returns a sub-set of cols of s from [0,i].
// The returned Set doesn't need to be free'd using set_del().
Set set_get_x(Set s, size_t i) {
    return (Set) {
        .data = s.data,
        .free_ptr = NULL,
        .n = s.n,
        .m = i,
        .stride = s.m,
    };
}

// Returns a sub-set of cols of s from [i,m].
// The returned Set doesn't need to be free'd using set_del().
Set set_get_y(Set s, size_t i) {
    return (Set) {
        .data = &s.data[i],
        .free_ptr = NULL,
        .n = s.n,
        .m = s.m - i,
        .stride = s.m,
    };
}

// Frees s.
void set_del(Set s) {
    free(s.free_ptr);
}

#endif // __SET_H__