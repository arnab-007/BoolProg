#include <stdbool.h>
int geo0(float p1) {
    int z = 0;
    bool flip = false;
    while (flip == false) {
        bool d = p1 > 0.5;  // Simplify for SATABS
        if (d) {
            flip = true;
        } else {
            z = z + 1;
        }
    }
    return z;
}

int main() {
    int z = geo0(0.5);
    return 0;
}






/*
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>



int geo0(float p1) {
    int z = 0;
    bool flip = false;
    srand(time(NULL));
    while (flip == false) {
        bool d = (float)rand() / RAND_MAX < p1;
        if (d) {
            flip = true;
        } else {
            z = z + 1;
        }
    }
    return z;
}

int main() {
    printf("z = %d\n", geo0(0.5));
    return 0;
}
*/