#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
#include <stdio.h>
#include <stdlib.h>

// Function declaration
int func1(_Bool x1, _Bool x2, _Bool x3, _Bool x4, _Bool x5, _Bool x6, _Bool x7, _Bool x8, _Bool x9, _Bool x10, _Bool x11, _Bool x12, _Bool r1, _Bool r2, _Bool r3, _Bool r4, _Bool r5, _Bool r6) {



    x1 = !x1 || x4;
    if (r1 && !r4)
    {
        x2 = x2 && x1 && !x5;
    }
    else 
    {
        x4 = (!x2 || x1) && !x3;
    }

    if (r2 || (r3 && !r4))
    {
        x3 = !x3 && x5;
    }
    else
    {
        x6 = x3 && !x2 && x5;
    }
    
    x7 = x2 || (!x3) || x5;
    x8 = x3 || (x2 && !x1) || !x5;
    x9 = x4 && !x1;
    
    // Extended logic for the remaining type 1 variables
    if (r5 || r6)
    {
        x11 = x6 || x5;
    }
    else 
    {
        x12 = x7 && !x6 && x4;
    }
    
    if (r4 && !r6)
    {
        x10 = !x6 && x7;
    }
    else
    {
        x5 = x5 && x11 && !x2;
    }
    
    x11 = x4 || (x9 && !x4);
    x12 = x12 && (!x7 || x11);
    x4 = x3 || (x4 && x6) || !x7;
    x6 = !x10 || (x5 && !x11);
    x8 = x6 || (x2 && !x4);
    x11 = x3 && (!x7 || x5);

       
    
    return (2048*(int)x1 + 1024*(int)x2 + 512*(int)x3 + 256*(int)x4 + 128*(int)x5 + 64*(int)x6 + 32*(int)x7 + 16*(int)x8 + 8*(int)x9 + 4*(int)x10 + 2*(int)x11 + (int)x12);
}



int main(int argc, char *argv[]) {
    if (argc != 19) { // Check if exactly 9 arguments are provided
        printf("Usage: %s <x1> <x2> <x3> <x4> <x5> <x6> <x7> <x8> <x9> <x10> <x11> <x12> <r1> <r2> <r3> <r4> <r5> <r6>\n", argv[0]);
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
    _Bool r1 = atoi(argv[13]);
    _Bool r2 = atoi(argv[14]);
    _Bool r3 = atoi(argv[15]);
    _Bool r4 = atoi(argv[16]);
    _Bool r5 = atoi(argv[17]);
    _Bool r6 = atoi(argv[18]);


    // Call func1 with the arguments
    int result = func1(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,r1,r2,r3,r4,r5,r6);

    // Print the result
    printf("%d\n", result);


    return 0;
}
