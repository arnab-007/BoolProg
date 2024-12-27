#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
#include <stdio.h>
#include <stdlib.h>

// Function declaration
int func1(_Bool a, _Bool b, _Bool c, _Bool d, _Bool e, _Bool r1, _Bool r2, _Bool r3, _Bool r4) {
    a = !a || d;
    if (r1 && !r4)
    {
        b = b && a && !e;
    }
    else 
    {
        b = (!b || a) && !c;
    }
    if (r2 || (r3 && !r4))
    {
        c = !c && e;
    }
    else
    {
        d = c && !b && e;
    }
    
    a = b || (!c) || e;
    b = c || (b && !a) || !e;
    e = d && !a;
       
    
    return (16*(int)a + 8*(int)b + 4*(int)c + 2*(int)d + (int)e);
}

int main(int argc, char *argv[]) {
    if (argc != 10) { // Check if exactly 9 arguments are provided
        printf("Usage: %s <x1> <x2> <x3> <x4> <x5> <r1> <r2> <r3> <r4>\n", argv[0]);
        return 1;
    }

    // Convert command-line arguments to _Bool
    _Bool x1 = atoi(argv[1]);
    _Bool x2 = atoi(argv[2]);
    _Bool x3 = atoi(argv[3]);
    _Bool x4 = atoi(argv[4]);
    _Bool x5 = atoi(argv[5]);
    _Bool r1 = atoi(argv[6]);
    _Bool r2 = atoi(argv[7]);
    _Bool r3 = atoi(argv[8]);
    _Bool r4 = atoi(argv[9]);

    // Call func1 with the arguments
    int result = func1(x1,x2,x3,x4,x5,r1,r2,r3,r4);

    // Print the result
    printf("%d\n", result);


    return 0;
}