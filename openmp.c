#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define ARRAYSIZE(x)  (sizeof(x)/sizeof(*(x)))
void main(){

	printf("Preparing Data For Training\n");

const char filename[] = "cancer_attributes.csv";
int attributes[683][9];
   FILE *file = fopen(filename, "r");
   if ( file )
   { size_t i, j, k;char buffer[BUFSIZ], *ptr;
      for ( i = 0; fgets(buffer, sizeof buffer, file); ++i )
      {for ( j = 0, ptr = buffer; j < ARRAYSIZE(*attributes); ++j, ++ptr )
         {attributes[i][j] = (int)strtol(ptr, &ptr, 10);}
      }
      fclose(file);
  }

/*
  for (int i =0;i<683;i++){
  	for(int j=0;j<9;j++)
  		{printf("%d ",attributes[i][j]);}
  	printf("\n");
  }

*/


const char cancer_file_name[] = "cancer_classes.csv";
int classes[683][2];
   FILE *cancer_file = fopen(cancer_file_name, "r");
   if ( cancer_file )
   { 
      size_t i1, j1, k1;char buffer1[BUFSIZ], *ptr1;
      for ( i1 = 0; fgets(buffer1, sizeof buffer1, cancer_file); ++i1 )
      {for ( j1 = 0, ptr1 = buffer1; j1 < ARRAYSIZE(*classes); ++j1, ++ptr1 )
         {classes[i1][j1] = (int)strtol(ptr1, &ptr1, 10);}
      }
      fclose(cancer_file);}


/*
for (int i =0;i<683;i++){
  	for(int j=0;j<2;j++)
  		{printf("%d ",classes[i][j]);}
  	printf("\n");
  }

*/
printf("Training Begins\n");

double node1weights[500][9],node2weights[500][500],node3weights[500][500],outputlayer[2][500];
printf("Initializing Weights of layer 1\n");
for(int i=0;i<500;i++)for(int j=0;j<9;j++)node1weights[i][j]=0.25;
printf("Initializing Weights of layer 2\n");
for(int i=0;i<500;i++)for(int j=0;j<500;j++)node2weights[i][j]=0.25;
printf("Initializing Weights of layer 3\n");
for(int i=0;i<500;i++)for(int j=0;j<500;j++)node3weights[i][j]=0.25;
printf("initializing Weights for outer layer\n");
for(int i=0;i<2;i++)for(int j=0;j<500;j++)outputlayer[i][j]=0.25;

double forpassl1[500],forpassl2[500],forpassl3[500],forpassout[2];
double errl1[500],errl2[500],errl3[500],errout[2];
double errRate = 0.01;

printf("Starting Training\n");
for(int training_sample = 0;training_sample<683;training_sample++)
{

#pragma omp parallel shared(node1weights,node2weights,node3weights,outputlayer,forpassl1,forpassl2,forpassl3,forpassout) num_threads(8)
	{
		//hidden layer 1 forward pass 1
		#pragma omp for schedule(dynamic,5)
		for(int i=0;i<500;i++){
			forpassl1[i]=0;
			int multifir = 0;
			for(int j=0;j<9;j++){
				multifir=multifir+node1weights[i][j]*attributes[training_sample][j];
			}
			forpassl1[i]=1/(1+exp(-multifir));
		}
		

		#pragma omp barrier

		//hidden layer 2 forward pass 2

		#pragma omp for schedule(dynamic,5)
		for(int i=0;i<500;i++){
			forpassl2[i]=0;
			int multifir = 0;
			for(int j=0;j<500;j++){
				multifir=multifir+node2weights[i][j]*forpassl1[j];
			}
			forpassl2[i]=1/(1+exp(-multifir));
		}
		#pragma omp barrier

		//hidden layer 3 forward pass 3

		#pragma omp for schedule(dynamic,5)
		for(int i=0;i<500;i++){
			forpassl3[i]=0;
			int multifir = 0;
			for(int j=0;j<500;j++){
				multifir=multifir+node3weights[i][j]*forpassl2[j];
			}
			forpassl3[i]=1/(1+exp(-multifir));
		}
		#pragma omp barrier
		//output layer forward pass 4

		#pragma omp for schedule(dynamic,5)
		for(int i=0;i<2;i++){
			forpassout[i]=0;
			int multifir = 0;
			for(int j=0;j<500;j++){
				multifir=multifir+outputlayer[i][j]*forpassl3[j];
			}
			forpassout[i]=1/(1+exp(-multifir));
		}
		#pragma omp barrier

		//error at output layer 5

		#pragma omp for schedule(dynamic,5)
		for(int i=0;i<2;i++){
			errout[i]=forpassout[i]*(1-forpassout[i])*(classes[training_sample][i]-forpassout[i]);
		}

		#pragma omp barrier

		//error at hidden layer 3 6

		#pragma omp for schedule(dynamic,5)
		for(int i=0;i<500;i++)
		{	int sum=0;
			for(int k=0;k<2;k++)
				{sum=sum+errout[k]*outputlayer[k][i];}

			errl3[i]=forpassl3[i]*(1-forpassl3[i])*sum;
		}
		#pragma omp barrier

		//error at hidden layer 2 7

		#pragma omp for schedule(dynamic,5)
		for(int i=0;i<500;i++)
		{	int sum=0;
			for(int k=0;k<500;k++)
				{sum=sum+errl3[k]*node3weights[k][i];}

			errl2[i]=forpassl2[i]*(1-forpassl2[i])*sum;
		}
		#pragma omp barrier

		//error at hidden layer 1 8

		#pragma omp for schedule(dynamic,5)
		for(int i=0;i<500;i++)
		{	int sum=0;
			for(int k=0;k<500;k++)
				{sum=sum+errl2[k]*node2weights[k][i];}

			errl1[i]=forpassl1[i]*(1-forpassl1[i])*sum;
		}
		#pragma omp barrier

		//changing weights in output layer 9
		#pragma omp for schedule(dynamic,5)
		for(int k=0;k<2;k++)
			for(int j=0;j<500;j++)
			{
				outputlayer[k][j] = outputlayer[k][j] + errRate*(errout[k]*forpassl3[j]);
			}

		#pragma omp barrier

		// changing weights in hidden layer 3 10
		#pragma omp for schedule(dynamic,5)
		for(int k=0;k<500;k++)
			for(int j=0;j<500;j++)
			{
				node3weights[k][j] = node3weights[k][j] + errRate*(errl3[k]*forpassl2[j]);
			}

		#pragma omp barrier

		//changing weights in hidden layer 2 11

		#pragma omp for schedule(dynamic,5)
		for(int k=0;k<500;k++)
			for(int j=0;j<500;j++)
			{
				node2weights[k][j] = node2weights[k][j] + errRate*(errl2[k]*forpassl1[j]);
			}

		#pragma omp barrier

		//changing weights in hidden layer 1 12

		#pragma omp for schedule(dynamic,5)
		for(int k=0;k<500;k++)
			for(int j=0;j<9;j++)
			{
				node1weights[k][j] = node1weights[k][j] + errRate*(errl1[k]*attributes[training_sample][j]);
			}
	}
}
printf("Training Complete\n");
}
