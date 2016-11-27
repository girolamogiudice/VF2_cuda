#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <dirent.h>
#include <cstdlib>
#include <string.h>
#include <cutil.h>
#include <shrUtils.h>
#include <cstdio>
#include <windows.h>
#ifdef _WIN32
#include <windows.h>
#elif MACOS
#include <sys/param.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif



char graph_label[256];
char graph_label_query[256];
int no_of_nodes=0,no_of_nodes_query=0;
int no_of_edge=0,no_of_edge_query=0;FILE *fp;
FILE *fp1;
//Structure to hold a node information

struct edge
	{
		int edge_start;
		int edge_end;
	};
struct __align__(8)  node 
	{
		unsigned long int node_start;
		unsigned long int node_end;
	};
double tot=0;

int *is_already_present;
int *h_result;
int *h_node_label;
int *h_node_label_query;
edge *h_edge;
edge *h_edge_query;
node *h_node;
node *h_node_query;
int maxThreadsPerBlock;
int numcore;
void find(char *path_query,const char *path,char *result_dir);
void read_and_construct_graph(int argc, char** argv);
void read_and_construct_query(int argc, char** argv,const char *path,char *result_dir,float tot);

//void graph_find();
int verify();
int getNumCores();
void create_for_vento(int h_increment,char *path_query,const char *path,char *result_dir);
	
	// d_node,h_node_query[i].node_start,h_node_query[i].node_end,increment,d_result,no_of_edge,d_over);
__global__ void Kernel(node *d_node,int node_start,int node_end,int *increment,int *d_result,int no_of_edge,int *d_present,int maxThreadsPerBlock) 
{
	
	int tid = blockIdx.x*maxThreadsPerBlock + threadIdx.x;
	if(tid<no_of_edge) 
		{
	
			if ( d_node[tid].node_start==node_start && d_node[tid].node_end==node_end || d_node[tid].node_start==node_end && d_node[tid].node_end==node_start) 
				{
					*d_present=1;
					d_result[atomicAdd(increment,1)]=tid;
				}
		}	
}
	

	
	
int main( int argc, char** argv) 
{
	FILE *fpres = fopen("result.txt","w");

	//FILE *fpres = fopen("result.txt","a");
	time_t begin,end;
	char directory[512];
	char result_dir[512];
	begin=clock();
	verify();
	end = clock();
	tot =(double)(end-begin)/CLOCKS_PER_SEC;
	printf("Tempo verifica %.20lf\n",tot);
	int num_of_core=getNumCores();
	int core_to_use;
	printf("\n");
	printf("CPU Analysis");
	printf("\n  Number Of Core:%d\n",num_of_core);
	printf("how many core(s) do you want to use?\n");
	scanf("%d",&core_to_use);
	if(core_to_use>num_of_core) 
		{
			printf("to many cory please insert a number between 1 and %d\n",num_of_core);
			scanf("%d\n",&core_to_use);
		}
	printf("please insert query directory\n");
	scanf("%s",&directory);
	printf("please insert result directory\n");
	scanf("%s",&result_dir);
	read_and_construct_graph( argc, argv);
 	printf("\n----------------------------------------------------------------------------------------------------------------------\n");
	read_and_construct_query(argc,argv,directory,result_dir,tot);
}

void Usage(int argc, char**argv) 
{
	fprintf(stderr,"Usage: %s <graph> \n", argv[0]);
}

int verify()
{
	CUdevice dev;
	int major = 0, minor = 0;
	int deviceCount = 0;
	char deviceName[256];

	// note your project will need to link with cuda.lib files on windows
	printf("CUDA Device Query (Driver API) statically linked version \n");
		CUresult err = cuInit(0);

    CU_SAFE_CALL_NO_SYNC(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0) 
	{
		printf("There is no device supporting CUDA\n");		
	}
    for (dev = 0; dev < deviceCount; ++dev) 
		{
			CU_SAFE_CALL_NO_SYNC( cuDeviceComputeCapability(&major, &minor, dev) );

				if (dev == 0) 
					{
					// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
						if (major == 9999 && minor == 9999)
							printf("There is no device supporting CUDA.\n");
						else if (deviceCount == 1)
							printf("There is 1 device supporting CUDA\n");
						else
							printf("There are %d devices supporting CUDA\n", deviceCount);
					}
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetName(deviceName, 256, dev) );
			printf("\nDevice %d: \"%s\"\n", dev, deviceName);
			printf("  CUDA Capability Major/Minor version number:    %d.%d\n", major, minor);
			if(major==1 && minor==0)
				printf("sorry but graph match requires Cuda Capabilities >=1.1\n");
			int multiProcessorCount;
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev ) );
			printf("  Multiprocessors x Cores/MP = Cores:            %d (MP) x %d (Cores/MP) = %d (Cores)\n", 
					 multiProcessorCount, ConvertSMVer2Cores(major, minor), 
					 ConvertSMVer2Cores(major, minor) * multiProcessorCount);
			int totalConstantMemory;
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &totalConstantMemory, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, dev ) );
			printf("  Total amount of constant memory:               %u bytes\n", totalConstantMemory);
			int sharedMemPerBlock;
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &sharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, dev ) );
			printf("  Total amount of shared memory per block:       %u bytes\n", sharedMemPerBlock);
			int regsPerBlock;
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, dev ) );
			printf("  Total number of registers available per block: %d\n", regsPerBlock);
			int warpSize;
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev ) );
			printf("  Warp size:                                     %d\n",	warpSize);
			
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev ) );
			printf("  Maximum number of threads per block:           %d\n",	maxThreadsPerBlock);
			int blockDim[3];
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &blockDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, dev ) );
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &blockDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, dev ) );
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &blockDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, dev ) );
			printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n", blockDim[0], blockDim[1], blockDim[2]);
			int gridDim[3];
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &gridDim[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev ) );
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &gridDim[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, dev ) );
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &gridDim[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, dev ) );
			printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n", gridDim[0], gridDim[1], gridDim[2]);
			int clockRate;
			CU_SAFE_CALL_NO_SYNC( cuDeviceGetAttribute( &clockRate, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev ) );
			printf("  Clock rate:                                    %.2f GHz\n", clockRate * 1e-6f);
}
return(0);
}

void read_and_construct_graph( int argc, char** argv) 
{
    char *input_f;
	if(argc!=2) 
	{
		Usage(argc, argv);
		exit(0);
	}
	input_f = argv[1];
	printf("\n");
	printf("**********Reading Graph File**********\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
		{
			printf("Error Reading graph file\n");
			return;
		}	
	fscanf(fp,"%s",graph_label);
	fscanf(fp,"%d",&no_of_nodes);
	h_node_label = (int*) malloc(sizeof(int)*no_of_nodes);
	int label=0;
	
	//reading and construct graph
	for( int i = 0; i < no_of_nodes; i++) 
		{	
			fscanf(fp,"%d",&label);
			h_node_label[i] = label;
		}
	fscanf(fp,"%d",&no_of_edge);
	h_edge = (edge*) malloc(sizeof(edge)*no_of_edge);
	int start;
	int end;
	//printf("no of edge %d\n",no_of_edge);
	for(int i=0;i<no_of_edge;i++) 
	{
		fscanf(fp,"%d %d",&start,&end);
		h_edge[i].edge_start = start ;
		h_edge[i].edge_end = end ;
	//printf("result %d	%d	\n",h_edge[i].edge_start,h_edge[i].edge_end);

	}
	fclose(fp);
	printf("graph name: %s \nno. of node: %d\nno. of edge: %d\n",graph_label,no_of_nodes,no_of_edge); 

}
void read_and_construct_query(int argc, char** argv,const char *path,char *result_dir,float tot)
{	
	
	char *input_f1;
	DIR *pdir = NULL; // remember, it's good practice to initialise a pointer to NULL!
    pdir = opendir (path); // "." will refer to the current directory
    struct dirent *pent = NULL;
    if (pdir == NULL) // if pdir wasn't initialised correctly
		{ // print an error message and exit the program
			printf ("\nERROR! pdir could not be initialised correctly");
			return; // exit the function
		} // end if
 
    while (pent = readdir (pdir)) // while there is still something in the directory to list
		{	
			int start_query=0;
			int end_query=0;
			int label_query=0;
			char str[128];
			strcpy (str,path);
				if (pent == NULL) // if pent has not been initialised correctly
					{ // print an error message, and exit the program
						printf ("\nERROR! pent could not be initialised correctly");
						return; // exit the function
					}
					printf("query %s\n",pent->d_name);
				if (strstr(pent->d_name,"txt")) 
					{

						strcat (str,pent->d_name);
						input_f1 = str;
						printf("**********Reading Query File**********\n");
						//Read in Graph from a file
						fp1 = fopen(input_f1,"r");
						if (!fp1)
						{
							printf("Error Reading graph file\n");
							return;
						} 
		

			//reading and construct graph query
			fscanf(fp1,"%s",graph_label_query);
			fscanf(fp1,"%d",&no_of_nodes_query);
			h_node_label_query = (int*) malloc(sizeof(int)*no_of_nodes_query);
			for ( int i = 0; i < no_of_nodes_query; i++) 
				{
					fscanf(fp1,"%d",&label_query);
					h_node_label_query[i] = label_query;
					//printf("h_node_query %d\n",h_node_label_query[i]);
				}

			fscanf(fp1,"%d",&no_of_edge_query);
			h_edge_query = (edge*) malloc(sizeof(edge)*no_of_edge_query);

			//printf("no of edge %d\n",no_of_edge_query);
			for	(int i=0;i<no_of_edge_query;i++) 
				{
					fscanf(fp1,"%d %d",&start_query,&end_query);
					h_edge_query[i].edge_start= start_query ;
					h_edge_query[i].edge_end = end_query ;	
					//printf("h_edge start %d , h_edge end %d \n",h_edge_query[i].edge_start,h_edge_query[i].edge_end);
				}

			fclose(fp1);
			printf("**********graph & query struct constructed**********\n");

			is_already_present=(int*) malloc(sizeof(int)*no_of_edge_query);
			node query;
			for(int i=0;i<no_of_edge_query;i++)
				is_already_present[i]=0;

			h_node = (node*) malloc(sizeof(node)*no_of_edge);
			h_node_query = (node*) malloc(sizeof(node)*no_of_edge_query);

			for (int i=0;i<no_of_edge;i++) 
				{
					h_node[i].node_start=h_node_label[h_edge[i].edge_start];
					h_node[i].node_end=h_node_label[h_edge[i].edge_end];
					//printf("h_node start %d ,h_node end %d\n",h_node[i].node_start,h_node[i].node_end);
				}

			for (int i=0;i<no_of_edge_query;i++) 
				{
					h_node_query[i].node_start=h_node_label_query[h_edge_query[i].edge_start];
					h_node_query[i].node_end=h_node_label_query[h_edge_query[i].edge_end];
					//printf("h_node start %d ,h_node end %d\n",h_node_query[i].node_start,h_node_query[i].node_end);
				}				

			for ( int i=0;i<no_of_edge_query;i++) 
				{
					query.node_start=h_node_query[i].node_start;
					query.node_end=h_node_query[i].node_end;
					//printf("node start %d node end %d\n",query.node_start,query.node_end);
				for (int j=0;j<no_of_edge_query;j++) 
					{
						if (j==i) 
							continue ;
						if ( query.node_start==h_node_query[j].node_start && query.node_end==h_node_query[j].node_end || query.node_start==h_node_query[j].node_end && query.node_end==h_node_query[j].node_start )
							is_already_present[j]=1;
					}				
					is_already_present[i]=0;
				}
			find(str,path,result_dir);
					}
		}
  closedir (pdir);
 
}

void find(char *path_query,const char *path,char *result_dir) 
{	
	
	
	int is_not_present=0;
	int h_increment=0;
	int num_of_blocks = 1;
	int num_of_threads_per_block = maxThreadsPerBlock;
	int *d_result;
	int h_present;
	int *d_present;
	int *d_increment;
	node *d_node;
	node *d_node_query;
	time_t begin,end;
	h_result = (int*) malloc(sizeof(int)*5000); //max number of occurrency in a graph
				
		if(no_of_edge>maxThreadsPerBlock)
			{
				num_of_blocks = (int)ceil(no_of_edge/(double)maxThreadsPerBlock); 
				num_of_threads_per_block = maxThreadsPerBlock; 
			}
	//Copy the Node list to device memory

		
	//print information about graph target and query
	printf("\nquery name: %s\nno. of node query: %d\nno. of edge query %d\n",graph_label_query,no_of_nodes_query,no_of_edge_query); 


	begin = clock();	
	
	cudaMalloc( (void**) &d_increment, sizeof(int));
	cudaMalloc( (void**) &d_present, sizeof(int));
	cudaMalloc( (void**) &d_node, sizeof(node)*no_of_edge) ;
	cudaMemcpy( d_node, h_node, sizeof(node)*no_of_edge, cudaMemcpyHostToDevice) ;
	cudaMalloc( (void**) &d_node_query, sizeof(node)*no_of_edge_query) ;
	cudaMemcpy( d_node_query, h_node_query, sizeof(node)*no_of_edge_query, cudaMemcpyHostToDevice) ;
	cudaMalloc( (void**) &d_result, sizeof(int)*5000) ; //max number of occurrency in a graph (see line 288)
	cudaMemcpy( d_result, h_result, sizeof(int)*5000, cudaMemcpyHostToDevice) ; //max number of occurrency in a graph(see line 288)
	cudaMemcpy( d_increment, &h_increment, sizeof(int), cudaMemcpyHostToDevice) ;



	for(int i=0;i<no_of_edge_query;i++) 
		{
			if (is_already_present[i]==1) 
				continue;
			h_present=0 ;
			//printf("node start %d node end %d\n",h_node_query[i].node_start,h_node_query[i].node_end);
			//printf("i %d ,h_present %d,node start %d node end %d\n",i,h_present,h_node_query[i].node_start,h_node_query[i].node_end,increment);
			cudaMemcpy( d_present, &h_present, sizeof(int), cudaMemcpyHostToDevice) ;
			Kernel<<<num_of_blocks,num_of_threads_per_block >>>( d_node,h_node_query[i].node_start,h_node_query[i].node_end,d_increment,d_result,no_of_edge,d_present,maxThreadsPerBlock);
			cudaThreadSynchronize();
			cudaMemcpy( &h_present, d_present, sizeof(int), cudaMemcpyDeviceToHost) ;
		

			if (h_present== 0 ) 
				{
					printf("query non presente\n");
					printf("----------------------------------------------------------------------------------------------------------------------\n");
					
					is_not_present=1;
					break ;
					//exit(0);

				}
		}	
	
	
	cudaMemcpy( h_result, d_result, sizeof(int)*5000, cudaMemcpyDeviceToHost) ;
	
	
	end = clock();
	cudaMemcpy( &h_increment, d_increment, sizeof(int), cudaMemcpyDeviceToHost) ;
	
	cudaFree(d_node);
	cudaFree(d_increment);
	cudaFree(d_result);
	cudaFree(d_present);
	cudaFree(d_node_query);
	free(h_node);
	
		//printf("increment dopo %d\n",h_increment);
	float time_subgraph = (double)(end-begin)/CLOCKS_PER_SEC;
	printf("subgraph created in: %.20lf\n",time_subgraph);

	if (is_not_present!=1) 
		{	
			create_for_vento( h_increment,path_query,path,result_dir);
		}
}


void create_for_vento(int h_increment,char *path_query,const char *path,char *result_dir) 
{
	edge *h_vento_edge = (edge*) malloc(sizeof(edge)*h_increment);
	int *nodi=(int *)malloc(sizeof(int)*10000);
	time_t begin,end;
	int edge;
	int *result=(int *)malloc(sizeof(int)*2*h_increment);
	for (int i=0;i<h_increment;i++) 
		result[i]=h_edge[h_result[i]].edge_start;
	for (int i=h_increment;i<h_increment*2;i++) 
		result[i]=h_edge[h_result[i-h_increment]].edge_end;
	//ho unito i 2 indici dimensione 2*h_increment

	int *change=(int *)malloc(2*h_increment*sizeof(int));
	for (int i=0;i<2*h_increment;i++) 
		{
			change[i]=0;
		}

	int data=0;
	int index=0;
	for(int i=0;i<2*h_increment;i++) 
		{
			if (change[i]==1)  
				continue ;

			edge=result[i];
			nodi[index]=edge;
			index++;


			for (int j=0;j<2*h_increment;j++) 
				{
			
					if( edge == result[j] && change[j]==0) 
						{
							result[j]=data;
							change[j]=1;
						}

				}
			data++;

		}
		//ho creato result[i] che è un vettore che si deve adattare a sing che diviso per 2 mi darà arco iniziale e finale.


	for (int j=0;j<h_increment;j++) 
		h_vento_edge[j].edge_start=result[j];
	for (int i=0;i<h_increment;i++)
		h_vento_edge[i].edge_end=result[i+h_increment];


	char query[]="#";
	char name_file[512];
	strcpy (name_file,path);

			


	FILE *fpo = fopen("graph.txt","w");
	fprintf(fpo,"%s \n",query);
	fprintf(fpo,"%d \n",data);
	for(int j=0;j<data;j++)
		fprintf(fpo,"%d\n",h_node_label[nodi[j]]);
	fprintf(fpo,"%d \n",h_increment);
	for(int i=0;i<h_increment;i++)
		fprintf(fpo,"%d %d\n",h_vento_edge[i].edge_start,h_vento_edge[i].edge_end);	
	fclose(fpo);
	printf("subgraph composed of %d node and %d edge\n",data,h_increment);

	char strvento[80];

	strcpy (strvento,"Vento.exe");
	strcat (strvento," graph.txt ");
 
	strcat (strvento," ");
	strcat (strvento,path_query);
	begin=clock();
	###########here call vento. It works only for windows##########
	system(strvento);
	end=clock();
	float time_vento = (double)(end-begin)/CLOCKS_PER_SEC;
	printf("vento match in: %.20lf\n",time_vento);
	
 	printf("----------------------------------------------------------------------------------------------------------------------\n");
	
	

}

int getNumCores() 
{
	#ifdef WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
	#elif MACOS
		int nm[2];
		size_t len = 4;
		uint32_t count;

		nm[0] = CTL_HW; nm[1] = HW_AVAILCPU;
		sysctl(nm, 2, &count, &len, NULL, 0);

    if(count < 1) 	
		{
			nm[1] = HW_NCPU;
			sysctl(nm, 2, &count, &len, NULL, 0);
			if(count < 1) { count = 1; }
		}
    return count;
	#else
		return sysconf(_SC_NPROCESSORS_ONLN);
	#endif
}
