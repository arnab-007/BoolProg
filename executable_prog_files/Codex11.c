#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
#include <stdio.h>
#include <stdlib.h>

// Function declaration
int func1(_Bool x1, _Bool x2, _Bool x3, _Bool x4, _Bool x5, _Bool x6, _Bool x7, _Bool x8, _Bool r1, _Bool r2, _Bool r3, _Bool r4) {



    x1 = (!x1 || x4) && x7;
    if (r1 && !r4) 
    {
        x2 = x2 && x1 && !x5 && x8;
    } 
    else 
    {
        x2 = (!x2 || x1) && !x3 && x6;
    }

    if (r2 || (r3 && !r4)) 
    {
        x3 = (!x3 && x5) || x6;
    } 
    else 
    {
        x4 = (x3 && !x2 && x5) || !x7;
    }

    x1 = (x2 || !x3 || x5) && x8;
    x2 = (x3 || (x2 && !x1) || !x5) && !x7;
    x5 = x4 && !x1 && x6;

    if (r2 && r4) 
    {
        x6 = (x1 || x3) && (!x2 || x8);
    } 
    else 
    {
        x7 = (!x6 || x4) && x8;
    }

    x8 = x7 || (x3 && !x5) || x6;

       
    
    return (128*(int)x1 + 64*(int)x2 + 32*(int)x3 + 16*(int)x4 + 8*(int)x5 + 4*(int)x6 + 2*(int)x7 + (int)x8);
}



int main(int argc, char *argv[]) {
    if (argc != 13) { // Check if exactly 9 arguments are provided
        printf("Usage: %s <x1> <x2> <x3> <x4> <x5> <x6> <x7> <x8> <r1> <r2> <r3> <r4>\n", argv[0]);
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
    _Bool r1 = atoi(argv[9]);
    _Bool r2 = atoi(argv[10]);
    _Bool r3 = atoi(argv[11]);
    _Bool r4 = atoi(argv[12]);

    // Call func1 with the arguments
    int result = func1(x1,x2,x3,x4,x5,x6,x7,x8,r1,r2,r3,r4);

    // Print the result
    printf("%d\n", result);


    return 0;
}
