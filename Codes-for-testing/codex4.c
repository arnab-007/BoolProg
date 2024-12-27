#include <stdio.h>
#include <stdbool.h>

int prog(int tempA,int tempB,int tempC,int tempD,int tempE,int tempR1,int tempR2,int tempR3,int tempR4) {


    bool a = false, b = false, c = false, d = false, e = false;
    bool r1 = false, r2 = false, r3 = false, r4 = false;


       // Convert input values to bool
    a = tempA ? true : false;
    b = tempB ? true : false;
    c = tempC ? true : false;
    d = tempD ? true : false;
    e = tempE ? true : false;
    r1 = tempR1 ? true : false;
    r2 = tempR2 ? true : false;
    r3 = tempR3 ? true : false;
    r4 = tempR4 ? true : false;
    
    
    // Apply the logic
    a = !a || d;
    if (r1 && !r4) {
        b = b && a && !e;
    } else {
        b = (!b || a) && !c;
    }
    if (r2 || (r3 && !r4)) {
        c = !c && e;
    } else {
        d = c && !b && e;
    }

    a = b || (!c) || e;
    b = c || (b && !a) || !e;
    e = d && !a;

    

    return (16*a+8*b+4*c+2*d+e);
}

int main()

{
    int tempA, tempB, tempC, tempD, tempE, tempR1, tempR2, tempR3, tempR4,result;
    
    for (tempA = 0;tempA<2;tempA++)
    {
        for (tempB = 0;tempB<2;tempB++){
            for (tempC = 0;tempC<2;tempC++){
                for (tempD = 0;tempD<2;tempD++)
                {
                    for (tempE = 0;tempE<2;tempE++){
                        for (tempR1 = 0;tempR1<2;tempR1++){
                            for (tempR2 = 0;tempR2<2;tempR2++)
                            {
                                for (tempR3 = 0;tempR3<2;tempR3++){
                                    {
                                        for (tempR4 = 0;tempR4<2;tempR4++){
                                            result = prog(tempA, tempB, tempC, tempD, tempE, tempR1, tempR2, tempR3, tempR4);
                                            printf("%d\n",result);
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

   
}
