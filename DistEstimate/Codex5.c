#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
#include <stdio.h>
#include <stdlib.h>

// Function declaration
int func1(_Bool x1, _Bool x2, _Bool x3) {
    x1 = !x1;
    x2 = x2 && x1;
    x2 = x2 || x1;
    x3 = x3 && x2;
    x1 = x2 || (!x3);
    x2 = x3 || (x2 && !x1);
    
    return (4*(int)x1 + 2*(int)x2 + (int)x3);
}

int main(int argc, char *argv[]) {
    if (argc != 4) { // Check if exactly 3 arguments are provided
        printf("Usage: %s <x1> <x2> <x3>\n", argv[0]);
        return 1;
    }

    // Convert command-line arguments to _Bool
    _Bool x1 = atoi(argv[1]);
    _Bool x2 = atoi(argv[2]);
    _Bool x3 = atoi(argv[3]);

    // Call func1 with the arguments
    int result = func1(x1,x2,x3);

    // Print the result
    printf("%d\n", result);


    return 0;
}


