#include <stdlib.h> 
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <stdio.h> 

#define size 10

 
_Bool func1(_Bool a,_Bool b,_Bool c,_Bool d,_Bool e) {
    
    
    a = !a || d;
    b = b && a && !e;
    b = (!b || a) && !c;
    c = !c && e;
    d = c && !b && e;
    a = b || (!c) || e;
    b = c || (b && !a) || !e;
    e = d && !a;
       
    return 0;
}

int main(int argc, char *argv[])

{

    _Bool a,b,c,d,e;
    a = (_Bool)argv[1];
    b = (_Bool)argv[2];
    c = (_Bool)argv[3];
    d = (_Bool)argv[4];
    e = (_Bool)argv[5];
    printf("%d\n", func1(a,b,c,d,e));
}