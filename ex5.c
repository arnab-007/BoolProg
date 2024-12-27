#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
__CPROVER_bool main() {
    
    __CPROVER_bool a,b,c;


    a = !a;
    b = b && a;
    b = b || a;
    c = c && b;
    
    
    a = b || (!c);
    b = c || (b && !a);
    
    assert(!a || !b || !c);
    return true;
}