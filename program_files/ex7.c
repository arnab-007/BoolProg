#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
__CPROVER_bool main() {
    
    __CPROVER_bool a,b;



       
    a = a || b;
    b = a && b;
        
    
    
    assert(!a || !b);
    return true;
}