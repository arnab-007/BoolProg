#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
__CPROVER_bool main() {
    
    __CPROVER_bool a,b,c,d,e;
    a = !a || d;
    b = b && a && !e;
    b = (!b || a) && !c;
    c = !c && e;
    d = c && !b && e;
    a = b || (!c) || e;
    b = c || (b && !a) || !e;
    e = d && !a;
       
    assert(!a || !b || !c || !d || !e);
    return true;
}
