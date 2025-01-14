#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>

// Function declaration
int func1(_Bool x1, _Bool x2, _Bool x3, _Bool x4, _Bool x5, _Bool x6, _Bool x7, _Bool x8, _Bool x9, _Bool x10,
          _Bool r1, _Bool r2, _Bool r3, _Bool r4, _Bool r5) {
    // Boolean logic for the `x` variables using `r` variables
    if (r1 && !r2) {
        x1 = x1 || x3 || !x5;
    } else {
        x2 = !x2 && x1 && x4;
    }

    if (r2 || (r3 && !r4)) {
        x3 = x2 && !x4;
    } else {
        x4 = x3 || (!x2 && x5);
    }

    x5 = (x1 && x3) || (!x4);
    x6 = !x1 || (x5 && x3);
    x7 = x6 && (!x5 || x4);

    if (r4 || !r5) {
        x8 = !x7 || x2;
    } else {
        x9 = x8 && x3 && !x6;
    }

    x10 = x9 || (!x8 && x7 && x2);

    // Return an integer representation of the Boolean variables
    return (512*(int)x1 + 256*(int)x2 + 128*(int)x3 + 64*(int)x4 + 32*(int)x5 +
            16*(int)x6 + 8*(int)x7 + 4*(int)x8 + 2*(int)x9 + (int)x10);
}

int main(int argc, char *argv[]) {
    if (argc != 16) { // Check if exactly 15 arguments are provided
        printf("Usage: %s <x1> <x2> <x3> <x4> <x5> <x6> <x7> <x8> <x9> <x10> <r1> <r2> <r3> <r4> <r5>\n", argv[0]);
        return 1;
    }

    // Convert command-line arguments to _Bool
    _Bool x1 = atoi(argv[1]);
    _Bool x2 = atoi(argv[2]);
    _Bool x3 = atoi(argv[3]);
    _Bool x4 = atoi(argv[4]);
    _Bool x5 = atoi(argv[5]);
    _Bool x6 = atoi(argv[6]);
    _Bool x7 = atoi(argv[7]);
    _Bool x8 = atoi(argv[8]);
    _Bool x9 = atoi(argv[9]);
    _Bool x10 = atoi(argv[10]);
    _Bool r1 = atoi(argv[11]);
    _Bool r2 = atoi(argv[12]);
    _Bool r3 = atoi(argv[13]);
    _Bool r4 = atoi(argv[14]);
    _Bool r5 = atoi(argv[15]);

    // Call func1 with the arguments
    int result = func1(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, r1, r2, r3, r4, r5);

    // Print the result
    printf("%d\n", result);

    return 0;
}
