#include "set.h"
#include "colors.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double static absd(double x) {
    return x < 0 ? -x : x;
}

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

// Returns a set made from a CSV file.
Set set_from_csv(const char *csv, const char *sep) {
    FILE *f = fopen(csv, "r");
    assert(f != NULL);

    size_t n = 0, m = 0;
    char buffer[4096];
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
        if (fscanf(f, "%s", buffer) != 1) {
            perror("Error reading CSV file\n");
            set_del(s);
            exit(1);
        }

        char *token = strtok(buffer, sep);
        for (size_t j = 0; j < m; j++) {
            SET_AT(s, i, j) = atof(token);
            token = strtok(NULL, sep);
        }
    }

    fclose(f);
    return s;
}

// Returns a set made from a C style matrix.
Set set_from(size_t n, size_t m, double data[n][m]) {
    Set s = set_new(n, m);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++)
            SET_AT(s, i, j) = data[i][j];
    return s;
}

// Returns a sub-set of the i'th row of s.
// The returned Set doesn't need to be free'd using set_del().
Set set_row(Set s, size_t i) {
    return (Set) {
        .data = &SET_AT(s, i, 0),
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
        .data = &SET_AT(s, 0, j),
        .free_ptr = NULL,
        .n = s.n,
        .m = 1,
        .stride = s.m,
    };
}

// Returns a sub-set of cols of s in the interval [0,i).
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

// Returns a sub-set of cols of s in the interval [i,m).
// The returned Set doesn't need to be free'd using set_del().
Set set_get_y(Set s, size_t i) {
    return (Set) {
        .data = &SET_AT(s, 0, i),
        .free_ptr = NULL,
        .n = s.n,
        .m = s.m - i,
        .stride = s.m,
    };
}

// Returns a sub-set of rows of s in the interval [from,to).
// The returned Set doesn't need to be free'd using set_del().
Set set_batch(Set s, size_t from, size_t to) {
    assert(from <= to);
    to = to < s.n ? to : s.n;
    return (Set) {
        .data = &SET_AT(s, from, 0),
        .free_ptr = NULL,
        .n = to - from,
        .m = s.m,
        .stride = s.stride,
    };
}

// Shuffles the given set and returns it.
Set set_shuffle(Set s) {
    for (size_t i = 0; i < s.n; i++) {
        size_t j = (rand() % (s.n - i)) + i;
        for (size_t k = 0; k < s.m; k++) {
            double tmp = SET_AT(s, i, k);
            SET_AT(s, i, k) = SET_AT(s, j, k);
            SET_AT(s, j, k) = tmp;
        }
    }

    return s;
}

// Copies src into dst and returns it.
Set set_copy(Set dst, Set src) {
    for (size_t i = 0; i < src.n; i++) {
        for (size_t j = 0; j < src.m; j++) {
            SET_AT(dst, i, j) = SET_AT(src, i, j);
        }
    }

    return dst;
}

void set_print_with_str(Set s, const char *str, size_t from, size_t to) {
    assert(from < to);
    printf(WHITE"%s:\n", str);
    char buff[16];
    for (size_t i = from; i < to; i++) {
        printf(BLACK"[  ");
        for (size_t j = 0; j < s.m; j++) {
            double v = SET_AT(s, i, j);
            snprintf(buff, 6, "%.3lf", absd(v));
            printf(v < 0 ? RED"%s  " : (v == 0 ? WHITE"%s  " : GREEN"%s  "), buff);
        }
        puts(BLACK"]");
    }
    puts(WHITE);
}

// Frees s.
void set_del(Set s) {
    free(s.free_ptr);
}
