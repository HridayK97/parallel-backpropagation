#include <stdio.h>

#define ARRAYSIZE(x)  (sizeof(x)/sizeof(*(x)))



int attributes[683][9];
int classes[683][2];
double node1weights[500][9],node2weights[500][500],node3weights[500][500],outputlayer[2][500];



double errRate = 0.01;


__global__ void feed_forward_loop1(int training_sample,double *forpassl1,double node1weights[500][9],int attributes[683][9])
{

for(int i=0;i<500;i++){
			forpassl1[i]=0;
			double multifir = 0;
			for(int j=0;j<9;j++){
				multifir=(multifir)+node1weights[i][j]*(attributes[training_sample][j]);
			}
			int e=-4;int ee=e;
			double T = 1;if(e<0){ee=-e;}
			for(int k=1;k<=ee;k++)T = T*multifir;
			T = (e<0) ? 1/T : T;
			forpassl1[i]=1/(1+T);
		}

}
__global__ void feed_forward_loop2(int training_sample,double *forpassl2,double node2weights[500][500],double *forpassl1)
{
for(int i=0;i<500;i++){
			forpassl2[i]=0;
			int multifir = 0;
			for(int j=0;j<500;j++){
				multifir=multifir+node2weights[i][j]*forpassl1[j];
			}
			int e=-4;int ee=e;
                        double T = 1;if(e<0){ee=-e;}
                        for(int k=1;k<=ee;k++)T = T*multifir;
                        T = (e<0) ? 1/T : T;
                        forpassl1[i]=1/(1+T);}
}

__global__ void feed_forward_loop3(int training_sample,double *forpassl3,double node3weights[500][500],double *forpassl2)
{
for(int i=0;i<500;i++){
                        forpassl3[i]=0;
                        int multifir = 0;
                        for(int j=0;j<500;j++){
                                multifir=multifir+node3weights[i][j]*forpassl2[j];
                        }
                        int e=-4;int ee=e;
                        double T = 1;if(e<0){ee=-e;}
                        for(int k=1;k<=ee;k++)T = T*multifir;
                        T = (e<0) ? 1/T : T;
                        forpassl3[i]=1/(1+T);}
}

__global__ void feed_forward_loop4(double *forpassout,double outputlayer[2][500],double *forpassl3)
{

for(int i=0;i<2;i++){
			forpassout[i]=0;
			int multifir = 0;
			for(int j=0;j<500;j++){
				multifir=multifir+outputlayer[i][j]*forpassl3[j];
			}
			
int e=-4;int ee=e;
                        double T = 1;if(e<0){ee=-e;}
                        for(int k=1;k<=ee;k++)T = T*multifir;
                        T = (e<0) ? 1/T : T;
                        forpassout[i]=1/(1+T);
		}
}

__global__ void feed_forward_loop5(int training_sample,double *errout,double *forpassout,int classes[683][2])
{

for(int i=0;i<2;i++){
			errout[i]=forpassout[i]*(1-forpassout[i])*(classes[training_sample][i]-forpassout[i]);
		}
}
__global__ void feed_forward_loop6(double *errout,double outputlayer[2][500],double *errl3,double *forpassl3)
{

for(int i=0;i<500;i++)
		{	int sum=0;
			for(int k=0;k<2;k++)
				{sum=sum+errout[k]*outputlayer[k][i];}

			errl3[i]=forpassl3[i]*(1-forpassl3[i])*sum;
		}
}
__global__ void feed_forward_loop7(double *errl3,double node3weights[500][500],double *forpassl2,double *errl2)
{

for(int i=0;i<500;i++)
		{	int sum=0;
			for(int k=0;k<500;k++)
				{sum=sum+errl3[k]*node3weights[k][i];}

			errl2[i]=forpassl2[i]*(1-forpassl2[i])*sum;
		}

}

__global__ void feed_forward_loop8(double *errl2,double node2weights[500][500],double *forpassl1,double *errl1)
{


for(int i=0;i<500;i++)
		{	int sum=0;
			for(int k=0;k<500;k++)
				{sum=sum+errl2[k]*node2weights[k][i];}

			errl1[i]=forpassl1[i]*(1-forpassl1[i])*sum;
		}

}

__global__ void feed_forward_loop9(double outputlayer[2][500],double errRate,double *errout,double *forpassl3)
{

for(int k=0;k<2;k++)
			for(int j=0;j<500;j++)
			{
				outputlayer[k][j] = outputlayer[k][j] + errRate*(errout[k]*forpassl3[j]);
			}



}

__global__ void feed_forward_loop10(double node3weights[500][500],double *forpassl2,double errRate,double *errl3)
{

for(int k=0;k<500;k++)
			for(int j=0;j<500;j++)
			{
				node3weights[k][j] = node3weights[k][j] + errRate*(errl3[k]*forpassl2[j]);
			}


}

__global__ void feed_forward_loop11(double node2weights[500][500],double *forpassl1,double errRate,double *errl2)
{

for(int k=0;k<500;k++)
			for(int j=0;j<500;j++)
			{
				node2weights[k][j] = node2weights[k][j] + errRate*(errl2[k]*forpassl1[j]);
			}

}
__global__ void feed_forward_loop12(int training_sample,double node1weights[500][9],double errRate,double *errl1,int attributes[683][9])
{

for(int k=0;k<500;k++)
			for(int j=0;j<9;j++)
			{
				node1weights[k][j] = node1weights[k][j] + errRate*(errl1[k]*attributes[training_sample][j]);
			}



}


int main()
{
printf("Preparing Data For Training\n");
const char filename[] = "cancer_attributes.csv";

int **attributes =(int **)cudaMallocManaged()


double *forpassl1,*forpassl2,*forpassl3,*forpassout;
cudaMallocManaged(&forpassl1,500*sizeof(int));
cudaMallocManaged(&forpassl2,500*sizeof(int));
cudaMallocManaged(&forpassl3,500*sizeof(int));
cudaMallocManaged(&forpassout,2*sizeof(int));

double *errl1,*errl2,*errl3,*errout;
cudaMallocManaged(&errl1,500*sizeof(double));
cudaMallocManaged(&errl2,500*sizeof(double));
cudaMallocManaged(&errl3,500*sizeof(double));
cudaMallocManaged(&errout,2*sizeof(double));

FILE *file = fopen(filename, "r");
   if ( file )
   { size_t i, j;char buffer[BUFSIZ], *ptr;
      for ( i = 0; fgets(buffer, sizeof buffer, file); ++i )
      {for ( j = 0, ptr = buffer; j < ARRAYSIZE(*attributes); ++j, ++ptr )
         {attributes[i][j] = (int)strtol(ptr, &ptr, 10);}
      }
      fclose(file);
  }


for (int i =0;i<683;i++){
  for(int j=0;j<9;j++)
  {printf("%d ",attributes[i][j]);}
  printf("\n");
  }


const char cancer_file_name[] = "cancer_classes.csv";


   FILE *cancer_file = fopen(cancer_file_name, "r");
   if ( cancer_file )
   { 
      size_t i1, j1;char buffer1[BUFSIZ], *ptr1;
      for ( i1 = 0; fgets(buffer1, sizeof buffer1, cancer_file); ++i1 )
      {for ( j1 = 0, ptr1 = buffer1; j1 < ARRAYSIZE(*classes); ++j1, ++ptr1 )
         {classes[i1][j1] = (int)strtol(ptr1, &ptr1, 10);}
      }
      fclose(cancer_file);}

for (int i =0;i<683;i++){
  for(int j=0;j<2;j++)
  {printf("%d ",classes[i][j]);}
  printf("\n");
 }


printf("Training Begins\n");


printf("Initializing Weights of layer 1\n");
for(int i=0;i<500;i++)for(int j=0;j<9;j++)node1weights[i][j]=0.25;
printf("Initializing Weights of layer 2\n");
for(int i=0;i<500;i++)for(int j=0;j<500;j++)node2weights[i][j]=0.25;
printf("Initializing Weights of layer 3\n");
for(int i=0;i<500;i++)for(int j=0;j<500;j++)node3weights[i][j]=0.25;
printf("initializing Weights for outer layer\n");
for(int i=0;i<2;i++)for(int j=0;j<500;j++)outputlayer[i][j]=0.25;




printf("Starting Training\n");



for(int training_sample = 0;training_sample<683;training_sample++)
{



feed_forward_loop1<<<1000,256>>>(training_sample,forpassl1,node1weights,attributes);

cudaDeviceSynchronize();

feed_forward_loop2<<<1000,256>>>(training_sample,forpassl2,node2weights,forpassl1);

cudaDeviceSynchronize();

feed_forward_loop3<<<1000,256>>>(training_sample,forpassl3,node3weights,forpassl2);

cudaDeviceSynchronize();

feed_forward_loop4<<<1000,256>>>(forpassout,outputlayer,forpassl3);
cudaDeviceSynchronize();

feed_forward_loop5<<<1000,256>>>(training_sample,errout,forpassout,classes);
cudaDeviceSynchronize();

feed_forward_loop6<<<1000,256>>>(errout,outputlayer,errl3,forpassl3);
cudaDeviceSynchronize();

feed_forward_loop7<<<1000,256>>>(errl3,node3weights,forpassl2,errl2);
cudaDeviceSynchronize();

feed_forward_loop8<<<1000,256>>>(errl2,node2weights,forpassl1,errl1);
cudaDeviceSynchronize();


feed_forward_loop9<<<1000,256>>>(outputlayer,errRate,errout,forpassl3);
cudaDeviceSynchronize();

feed_forward_loop10<<<1000,256>>>(node3weights,forpassl2,errRate,errl3);
cudaDeviceSynchronize();

feed_forward_loop11<<<1000,256>>>(node2weights,forpassl1,errRate,errl2);
cudaDeviceSynchronize();

feed_forward_loop12<<<1000,256>>>(training_sample,node1weights,errRate,errl1,attributes);
cudaDeviceSynchronize();
}
printf("Training Complete\n");


}
