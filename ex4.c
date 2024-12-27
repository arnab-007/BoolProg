#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
bool main() {
    
    bool a,b,c,d,e,r1,r2,r3,r4;
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
       
    assert(!a || !b || !c || !d || !e);
    return true;
}
