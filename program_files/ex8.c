#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
__CPROVER_bool main() {
    
    __CPROVER_bool a,b,r1,r2;



    if (r1 || r2)
    {
        a = a || b;
        b = a && b;
    }  
    else
    {
        a = a && b;
        b = a || b;
    }

        
    
    
    assert(!a || !b);
    return true;
}