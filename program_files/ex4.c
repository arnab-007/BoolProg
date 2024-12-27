#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
_Bool main() {
    
    _Bool x1,x2,x3,x4,x5,r1,r2,r3,r4;
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
       
    assert(!x1 || !x2 || !x3 || !x4 || !x5);
    
}
