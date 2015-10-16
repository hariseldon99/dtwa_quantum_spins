#include <stdio.h>

#include "lorenzo_bbgky.h"

int main (void)
{
    char *data, *data2;
    int len, len2;

    data = test_get_data_nulls(&len);
    test_data_print(data, len);

    test_get_data_nulls_out(&data2, &len2);
    test_data_print(data2, len2);
}
