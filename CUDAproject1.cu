/*
 * Joseph Salazar
 * salazjos@oregonstate.edu
 * CUDAproject1.cu - Monte Carlo Simulation - Use CUDA to calculate
 * the volume between two surfaces using titles that are on the top
 * and bottom surface. Identical to project2.cpp except uses CUDA
 * instead of OpenMP.
 *
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#ifndef BLOCKSIZE
#define BLOCKSIZE		32		// number of threads per block
#endif

#ifndef SIZE
#define SIZE			1*1024*1024	// array size
#endif

#ifndef NUMTRIALS
#define NUMTRIALS		100		// to make the timing more accurate
#endif

#ifndef TOLERANCE
#define TOLERANCE		0.00001f	// tolerance to relative error
#endif

// ranges for the random numbers:
const float XCMIN =     0.0;
const float XCMAX =     2.0;
const float YCMIN =     0.0;
const float YCMAX =     2.0;
const float RMIN  =     0.5;
const float RMAX  =     2.0;

// function prototypes:
float   Ranf(float, float);
int     Ranf(int, int);
void    TimeOfDaySeed();
int     sumHits(int *, int);
__global__ void MonteCarlo( float *, float *, float *, int *);

int main(int argc, char* argv[ ]){

    int dev = findCudaDevice(argc, (const char **)argv);

    // seed the random number generator
    TimeOfDaySeed();

    //delcare the arrays
    // allocate host memory:
    float *h_xcs = new float[NUMTRIALS];
    float *h_ycs = new float[NUMTRIALS];
    float *h_rs  = new float[NUMTRIALS];
    int   *h_numhits = new int[NUMTRIALS] {0};

    // fill the random-value arrays:
    for(int n = 0; n < NUMTRIALS; n++)
    {
        h_xcs[n] = Ranf(XCMIN, XCMAX);
        h_ycs[n] = Ranf(YCMIN, YCMAX);
        h_rs[n]  = Ranf(RMIN, RMAX);
    }

    // declare device arrays:
    float *d_xcs;
    float *d_ycs;
    float *d_rs;
    int   *d_numhits;

    //TODO: is this correct?
    dim3 dimsXCS(NUMTRIALS, 1, 1);
    dim3 dimsYCS(NUMTRIALS, 1, 1);
    dim3 dimsRS (NUMTRIALS, 1, 1);
    dim3 dimsHITS(NUMTRIALS, 1, 1);

    //allocate device memory for each array.
    cudaError_t status;
    status = cudaMalloc(reinterpret_cast<void **>(&d_xcs), NUMTRIALS*sizeof(float));
    checkCudaErrors(status);
    status = cudaMalloc(reinterpret_cast<void **>(&d_ycs), NUMTRIALS*sizeof(float));
    checkCudaErrors(status);
    status = cudaMalloc(reinterpret_cast<void **>(&d_rs),  NUMTRIALS*sizeof(float));
    checkCudaErrors(status);
    status = cudaMalloc(reinterpret_cast<void **>(&d_numhits),  NUMTRIALS*sizeof(int));
    checkCudaErrors(status);

    // copy host memory to the device:
    status = cudaMemcpy(d_xcs, h_xcs, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors( status );
    status = cudaMemcpy(d_ycs, h_ycs, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors( status );
    status = cudaMemcpy(d_rs, h_rs, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors( status );
    status = cudaMemcpy(d_numhits, h_numhits,NUMTRIALS*sizeof(int),cudaMemcpyHostToDevice);
    checkCudaErrors( status );

    // setup the execution parameters:
    dim3 threads(BLOCKSIZE, 1, 1);
    dim3 grid( NUMTRIALS / threads.x, 1, 1);

    // Create and start timer
    cudaDeviceSynchronize();

    // allocate CUDA events for timing:
    cudaEvent_t start, stop;
    status = cudaEventCreate(&start);
    checkCudaErrors(status);
    status = cudaEventCreate(&stop);
    checkCudaErrors(status);

    // record the start event:
    status = cudaEventRecord(start, NULL);
    checkCudaErrors(status);

    // execute the kernel:
    //run monte carlo on gpu device
    MonteCarlo<<<grid,threads>>>(d_xcs, d_ycs, d_rs, d_numhits);

    // record the stop event:
    status = cudaEventRecord(stop, NULL);
    checkCudaErrors(status);

    // wait for the stop event to complete:
    status = cudaEventSynchronize(stop);
    checkCudaErrors(status);

    //calculate elapsed time
    float msecTotal = 0.0f;
    status = cudaEventElapsedTime(&msecTotal, start, stop);
    checkCudaErrors(status);

    //compute performance
    double secondsTotal = 0.001 * (double)msecTotal;
    double multsPerSecond =  (float)NUMTRIALS / secondsTotal;
    double megaMultsPerSecond = multsPerSecond / 1000000.;
    printf("BLOCKSIZE: %d\n", BLOCKSIZE);
    printf("NUMTRIALS: %d\n", NUMTRIALS);
    fprintf( stderr, "Array Size = %10d, MegaMultReductions/Second = %10.2lf\n", NUMTRIALS, megaMultsPerSecond );

    // copy result from the device to the host:
    status = cudaMemcpy( h_numhits, d_numhits,(NUMTRIALS)*sizeof(float),cudaMemcpyDeviceToHost);
    checkCudaErrors( status );

    //calculate and print probability
    int sum = sumHits(h_numhits, NUMTRIALS);
    float currentProb = (float)sum/(float)NUMTRIALS;
    printf("Probability = %2.4f\n\n", currentProb);

    // clean up host memory:
    delete [] h_xcs;
    delete [] h_ycs;
    delete [] h_rs;
    delete [] h_numhits;

    //clean up device memory
    status = cudaFree(d_xcs);
    checkCudaErrors(status);
    status = cudaFree(d_ycs);
    checkCudaErrors(status);
    status = cudaFree(d_rs);
    checkCudaErrors(status);
    status = cudaFree(d_numhits);
    checkCudaErrors(status);

    return 0;
}

int sumHits(int *hitsArray; int array_size){
    int sum = 0;
    for(int i = 0; i < array_size; i++){
        sum += hitsArray[i];
    }
    return sum;
}

//Instructor provided
float Ranf( float low, float high ){
    float r = (float) rand();               // 0 - RAND_MAX
    float t = r  /  (float) RAND_MAX;       // 0. - 1.
    return   low  +  t * ( high - low );
}

//Instructor provided
int Ranf( int ilow, int ihigh ){
    float low = (float)ilow;
    float high = ceil( (float)ihigh );
    return (int) Ranf(low,high);
}

//Instructor provided
void TimeOfDaySeed( ){
    struct tm y2k = { 0 };
    y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
    y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

    time_t  timer;
    time( &timer );
    double seconds = difftime( timer, mktime(&y2k) );
    unsigned int seed = (unsigned int)( 1000.*seconds );    // milliseconds
    srand( seed );
}

//Instructor provided
__global__ void MonteCarlo( float *xcs, float *ycs, float *rs, int *numHits){

    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

    // randomize the location and radius of the circle:
    float xc = xcs[gid];
    float yc = ycs[gid];
    float  r =  rs[gid];

    // solve for the intersection using the quadratic formula:
    float a = 2.;
    float b = -2.*( xc + yc );
    float c = xc*xc + yc*yc - r*r;
    float d = b*b - 4.*a*c;

    if(d > 0){
        // hits the circle:
        // get the first intersection:
        d = sqrt( d );
        float t1 = (-b + d ) / ( 2.*a );    // time to intersect the circle
        float t2 = (-b - d ) / ( 2.*a );    // time to intersect the circle
        float tmin = t1 < t2 ? t1 : t2;        // only care about the first intersection

        if(tmin > 0){
            // where does it intersect the circle?
            float xcir = tmin;
            float ycir = tmin;

            // get the unitized normal vector at the point of intersection:
            float nx = xcir - xc;
            float ny = ycir - yc;
            float _n = sqrt( nx*nx + ny*ny );
            nx /= _n;    // unit vector
            ny /= _n;    // unit vector

            // get the unitized incoming vector:
            float inx = xcir - 0.;
            float iny = ycir - 0.;
            float in  = sqrt( inx*inx + iny*iny );
            inx /= in;    // unit vector
            iny /= in;    // unit vector

            // get the outgoing (bounced) vector:
            float dot  = inx*nx + iny*ny;
            float outx = inx - 2.*nx*dot;    // angle of reflection = angle of incidence`
            float outy = iny - 2.*ny*dot;    // angle of reflection = angle of incidence`

            // find out if it hits the infinite plate:
            float t = ( 0. - ycir ) / outy;

            if(t > 0){ //beam hits the plate
                numHits[gid]++;;
            }
        }
    }
}