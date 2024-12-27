#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10


#include <stdio.h>
#include <stdlib.h>

// Function declaration
int func1(_Bool x1, _Bool x2, _Bool x3, _Bool x4, _Bool x5, _Bool x6, _Bool x7, _Bool x8, _Bool x9, _Bool x10, _Bool x11, _Bool x12, _Bool x13, _Bool x14, _Bool x15, _Bool x16, _Bool x17,
          _Bool r1, _Bool r2, _Bool r3, _Bool r4, _Bool r5, _Bool r6, _Bool r7) {
    x1 = !x1 || x4;
    if (r1 && !r4)
    {
        x2 = x2 && x1 && !x5;
    }
    else 
    {
        x2 = (!x2 || x1) && !x3;
    }

    if (r2 || (r3 && !r4))
    {
        x3 = !x3 && x5;
    }
    else
    {
        x4 = x3 && !x2 && x5;
    }
    
    x1 = x2 || (!x3) || x5;
    x2 = x3 || (x2 && !x1) || !x5;
    x5 = x4 && !x1;
    
    // Extended logic for the remaining type 1 variables
    if (r5 || r6)
    {
        x6 = x6 || x5;
    }
    else 
    {
        x7 = x7 && !x6 && x4;
    }
    
    if (r7 && !r6)
    {
        x9 = !x6 && x7;
    }
    else
    {
        x10 = x5 && x11 && !x2;
    }
    
    x12 = x14 || (x11 && !x4);
    x13 = x12 && (!x7 || x11);
    x14 = x13 || (x14 && x6) || !x7;
    x15 = !x10 || (x15 && !x11);
    x16 = x15 || (x12 && !x14);
    x17 = x13 && (!x16 || x5);
    
    return (65536*(int)x1 + 32768*(int)x2 + 16384*(int)x3 + 8192*(int)x4 + 4096*(int)x5 +
            2048*(int)x6 + 1024*(int)x7 + 512*(int)x8 + 256*(int)x9 + 128*(int)x10 +
            64*(int)x11 + 32*(int)x12 + 16*(int)x13 + 8*(int)x14 + 4*(int)x15 +
            2*(int)x16 + (int)x17);
}

int main(int argc, char *argv[]) {
    if (argc != 25) { // Check if exactly 24 arguments are provided
        printf("Usage: %s <x1> <x2> <x3> <x4> <x5> <x6> <x7> <x8> <x9> <x10> <x11> <x12> <x13> <x14> <x15> <x16> <x17> <r1> <r2> <r3> <r4> <r5> <r6> <r7>\n", argv[0]);
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
    _Bool x11 = atoi(argv[11]);
    _Bool x12 = atoi(argv[12]);
    _Bool x13 = atoi(argv[13]);
    _Bool x14 = atoi(argv[14]);
    _Bool x15 = atoi(argv[15]);
    _Bool x16 = atoi(argv[16]);
    _Bool x17 = atoi(argv[17]);
    _Bool r1 = atoi(argv[18]);
    _Bool r2 = atoi(argv[19]);
    _Bool r3 = atoi(argv[20]);
    _Bool r4 = atoi(argv[21]);
    _Bool r5 = atoi(argv[22]);
    _Bool r6 = atoi(argv[23]);
    _Bool r7 = atoi(argv[24]);

    // Call func1 with the arguments
    int result = func1(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17, r1,r2,r3,r4,r5,r6,r7);

    // Print the result
    printf("%d\n", result);

    return 0;
}
