#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
#include <stdio.h>
#include <stdlib.h>

// Function declaration
int func1(_Bool x1, _Bool x2, _Bool r1, _Bool r2) {
    if (r1 || r2)
    {
        x1 = x1 || x2;
        x2 = x1 && x2;
    }  
    else
    {
        x1 = x1 && x2;
        x2 = x1 || x2;
    }
    
    return (2*(int)x1 + (int)x2);
}

int main(int argc, char *argv[]) {
    if (argc != 5) { // Check if exactly 4 arguments are provided
        printf("Usage: %s <x1> <x2> <r1> <r2>\n", argv[0]);
        return 1;
    }

    // Convert command-line arguments to _Bool
    _Bool x1 = atoi(argv[1]);
    _Bool x2 = atoi(argv[2]);
    _Bool r1 = atoi(argv[3]);
    _Bool r2 = atoi(argv[4]);

    // Call func1 with the arguments
    int result = func1(x1,x2,r1,r2);

    // Print the result
    printf("%d\n", result);


    return 0;
}


