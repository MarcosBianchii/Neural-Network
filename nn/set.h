#ifndef __SET_H__
#define __SET_H__

#include <stdlib.h>

typedef struct Set {
    double *data, *free_ptr;
    size_t n, m, stride;
} Set;

// Gives an entry point to specific data in the set.
#define SET_AT(set, i, j) (set).data[(i)*(set).stride+(j)]

// Formats and prints s.
#define set_print(s) set_print_with_str(s, #s, 0, (s).n)

Set set_from(size_t n, size_t m, double data[n][m]);
Set set_from_csv(const char *csv, const char *sep);
Set set_row(Set s, size_t i);
Set set_col(Set s, size_t j);
Set set_get_x(Set s, size_t i);
Set set_get_y(Set s, size_t i);
Set set_batch(Set s, size_t from, size_t to);
Set set_shuffle(Set s);
Set set_copy(Set s);
void set_print_with_str(Set s, const char *str, size_t u, size_t v);
void set_del(Set s);

#endif // __SET_H__