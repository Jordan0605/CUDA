/**********************************************************************
 * DESCRIPTION:
 *   Serial Concurrent Wave Equation - C Version
 *   This program implements the concurrent wave equation
 *********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265

int nsteps, tpoints;

/**********************************************************************
 *	Checks input values from parameters
 *********************************************************************/
void check_param(void)
{
   char tchar[20];

   /* check number of points, number of iterations */
   while ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS)) {
      printf("Enter number of points along vibrating string [%d-%d]: "
           ,MINPOINTS, MAXPOINTS);
      scanf("%s", tchar);
      tpoints = atoi(tchar);
      if ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS))
         printf("Invalid. Please enter value between %d and %d\n",
                 MINPOINTS, MAXPOINTS);
   }
   while ((nsteps < 1) || (nsteps > MAXSTEPS)) {
      printf("Enter number of time steps [1-%d]: ", MAXSTEPS);
      scanf("%s", tchar);
      nsteps = atoi(tchar);
      if ((nsteps < 1) || (nsteps > MAXSTEPS))
         printf("Invalid. Please enter value between 1 and %d\n", MAXSTEPS);
   }

   printf("Using points = %d, steps = %d\n", tpoints, nsteps);

}

__global__ void exe(float* finalval, int tpoints, int nsteps){
  float values, newval, oldval;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  //init
  float x = (float)(i - 1) / (tpoints - 1);
  values = sin(2.0 * PI * x);
  oldval = values;
  //update
  if(i == 0 || i == tpoints) values = 0;
  else{
    for(int i=1 ; i<=nsteps ; i++){
      //tau = 0.3 , sqtau = 0.09 , new = 2 * values - oldval + 0.09 * (-2) * values
      //new = 1.82 * values - oldval
      newval = 1.82 * values - oldval;
      oldval = values;
      values = newval;
    }
  }
  finalval[i] = values;
}

void printfinal(float* values)
{
   int i;

   for (i = 1; i <= tpoints; i++) {
      printf("%6.4f ", values[i]);
      if (i%10 == 0)
         printf("\n");
   }
}

int main(int argc, char *argv[])
{
  float finalval[MAXPOINTS + 2];
  float* final_D;
  int size = (MAXPOINTS + 2) * sizeof(float);
  cudaMalloc((void**)&final_D, size);

	sscanf(argv[1],"%d",&tpoints);
	sscanf(argv[2],"%d",&nsteps);
	check_param();

	printf("Initializing points on the line...\n");
	//init_line();
	printf("Updating all points for all time steps...\n");
  int threadPerBlock = 1024;
  int numBlock = tpoints / threadPerBlock + 1;
	//update();
  exe<<<numBlock,threadPerBlock>>>(final_D, tpoints, nsteps);

  cudaMemcpy(finalval, final_D, size, cudaMemcpyDeviceToHost);
  cudaFree(final_D);

	printf("Printing final results...\n");
	printfinal(finalval);
	printf("\nDone.\n\n");

	return 0;
}
