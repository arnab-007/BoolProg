#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
__CPROVER_bool main() {
    
    __CPROVER_bool a,b,c,r1,r2,r3;
    a = !a;
    if (r1)
    {
    	b = b && a;
    }
    else 
    {
    	b = b || a;
    }
    if (r2 || r3)
    {
    	c = !c;
    }
    else
    {
    	c = c && b;
    }
    
    a = b || (!c);
    b = c || (b && !a);
    
    assert(!c);
    return true;
}
