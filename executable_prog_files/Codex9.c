#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
#include <stdio.h>
#include <stdlib.h>

// Function declaration
int func1(_Bool x1, _Bool x2, _Bool x3, _Bool x4, _Bool x5, _Bool r1, _Bool r2, _Bool r3, _Bool r4) {
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
       
    
    return (16*(int)x1 + 8*(int)x2 + 4*(int)x3 + 2*(int)x4 + (int)x5);
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



/*
int main()
{
    _Bool x1,x2,x3,x4,x5,r1,r2,r3,r4;
    x1 = 0;
    x2 = 0;
    x3 = 0;
    x4 = 0;
    x5 = 0;
    r1 = 0;
    r2 = 0;
    r3 = 0;
    r4 = 0;


    for (x1 =0;x1<=1;x1++){
        for (x2 =0;x2<=1;x1++){
            for (x3 =0;x3<=1;x1++)
            {
                for (x4 =0;x4<=1;x1++){
                    for (x5 =0;x5<=1;x1++){
                        for (r1 =0;r1<=1;x1++){
                            for (r2 =0;r2<=1;x1++){
                                for (r3 =0;r3<=1;x1++){
                                    for (r4 =0;r4<=1;x1++){
                                        printf("%d\n", func1(x1,x2,x3,x4,x5,r1,r2,r3,r4));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


}

*/