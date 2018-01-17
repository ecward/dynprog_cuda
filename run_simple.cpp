#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>



//
// declare external routine
//
extern
int simple(/*int nblocks, int nthreads*/);

//
// main code
//
int main(int argc, char **argv)
{
    simple();
    return 0;
}
