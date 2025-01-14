#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
__CPROVER_bool main() {
    
    __CPROVER_bool x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,r1,r2,r3,r4,r5;



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


    assert(!x1 || !x2 || !x3 || !x4 || !x5 || !x6 || !x7 || !x8 || !x9 || !x10);

    return true;
}