/*	

Copyright 2012 The Trustees of Indiana University.  All rights reserved.
CGL MapReduce Framework on GPUs and CPUs
Code Name: Panda 0.2
File: main.cu 
Time: 2012-07-01 
Developer: Hui Li (lihui@indiana.edu)

This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

*/

#include "Panda.h"
#include "UserAPI.h"
#include <ctype.h>


//-----------------------------------------------------------------------
//usage: C-means datafile
//param: datafile 
//-----------------------------------------------------------------------


static float *GenPointsFloat(int numPt, int dim)
{
	float *matrix = (float*)malloc(sizeof(float)*numPt*dim);
	srand(time(0));
	for (int i = 0; i < numPt; i++)
		for (int j = 0; j < dim; j++)
			matrix[i*dim+j] = (float)((rand() % 100)/73.0);
	return matrix;
}//static float 

static float *GenInitCentersFloat(float* points, int numPt, int dim, int K)
{
	float* centers = (float*)malloc(sizeof(float)*K*dim);

	for (int i = 0; i < K; i++)
		for (int j = 0; j < dim; j++)
			centers[i*dim+j] = points[i*dim + j];
	return centers;
}//

int main(int argc, char** argv) 
{		
	if (argc != 7)
	{
		printf("Panda C-means\n");
		printf("usage: %s [numPt] [Dimensions] [numClusters] [numMapperPerGPU] [maxIter] [numGpus]\n", argv[0]);
		exit(-1);
	}//if

	//printf("start %s  %s  %s\n",argv[0],argv[1],argv[2]);
	int numPt = atoi(argv[1]);
	int dim = atoi(argv[2]);
	int K = atoi(argv[3]);
	int numMapper = atoi(argv[4]);
	int maxIter = atoi(argv[5]);
	int num_gpus = atoi(argv[6]);
	double ratio = 0;
	
	panda_context *panda = CreatePandaContext();
	panda->num_gpus = num_gpus;
	panda->num_cpus_groups = 0;//num_cpus_groups;
	panda->ratio = 0.0;

	ShowLog("numPt:%d	dim:%d	K:%d	numMapper:%d	maxIter:%d",numPt,dim,K,numMapper,maxIter);
	float* h_points = GenPointsFloat(numPt, dim);
	float* h_cluster = GenInitCentersFloat(h_points, numPt, dim, K);
	
	int numgpus = 0;
	cudaGetDeviceCount(&numgpus);
	if (num_gpus >= numgpus)		num_gpus = numgpus;
			
	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*num_gpus);
	thread_info_t *thread_info = (thread_info_t*)malloc(sizeof(thread_info_t)*num_gpus);
		
	for (int i=0; i<num_gpus; i++){

		job_configuration *gpu_job_conf = CreateJobConf();
		gpu_job_conf->num_gpus = num_gpus;
		gpu_job_conf->num_mappers = numMapper;
		gpu_job_conf->auto_tuning = false;
		gpu_job_conf->ratio = (double)ratio;
		gpu_job_conf->auto_tuning_sample_rate = -1;//sample_rate;
		gpu_job_conf->iterative_support = false;
		
		int tid = i;		
		float* d_points	=	NULL;
		float* d_cluster =	NULL;
		int* d_change	=	NULL;
		int* d_clusterId =	NULL;
		
		float* d_tempClusters = NULL;
		float* d_tempDenominators = NULL;
		
		checkCudaErrors(cudaSetDevice(tid));
				
		checkCudaErrors(cudaMalloc((void**)&d_points, numPt*dim*sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_points, h_points, numPt*dim*sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&d_clusterId, numPt*sizeof(int)));
		checkCudaErrors(cudaMemset(d_clusterId, 0, numPt*sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_cluster, K*dim*sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_cluster, h_cluster, K*dim*sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&d_change, sizeof(int)));
		checkCudaErrors(cudaMemset(d_change, 0, sizeof(int)));
		
		checkCudaErrors(cudaMalloc((void**)&d_tempClusters,K*dim*numMapper*sizeof(float)));
		checkCudaErrors(cudaMemset(d_tempClusters, 0, sizeof(float)*K*dim*numMapper));
		checkCudaErrors(cudaMalloc((void**)&d_tempDenominators,numMapper * K * sizeof(float)));
		
		checkCudaErrors(cudaMemset(d_tempDenominators, 0, sizeof(float)*K*numMapper));
		
		thread_info[i].tid = i;
		//thread_info[i].num_gpus = num_gpus;
		thread_info[i].device_type = GPU_ACC;
		
		cudaDeviceProp gpu_dev;
		cudaGetDeviceProperties(&gpu_dev, i);

		ShowLog("Configure Device ID:%d: Device Name:%s", i, gpu_dev.name);
		thread_info[i].device_name = gpu_dev.name;
				
		KM_VAL_T val;
		val.ptrPoints = (int *)d_points;
		val.ptrClusters = (int *)d_cluster;
		val.d_Points = d_points;
		val.d_Clusters = d_cluster;
		val.ptrChange = d_change;
		
		KM_KEY_T key;
		key.dim = dim;
		key.K = K;
		key.ptrClusterId = d_clusterId;
		
		int numPtPerGPU = numPt/num_gpus;
		int start = i*numPtPerGPU;
		int end = start+numPtPerGPU;
		if (i==num_gpus-1)
			end = numPt;
		
		int numPtPerMap = (end-start)/numMapper;
		ShowLog("numPtPerMap:%d startPt:%d  endPt:%d numPt:%d",numPtPerMap,start,end,numPt);

		int start_i,end_i;
		start_i = start;
		for (int j = 0; j < numMapper; j++)
		{	
			end_i = start_i + numPtPerMap;
			if (i<(end-start)%numMapper)
				end_i++;
			
			//ShowLog("start_i:%d, start_j:%d",start_i,end_i);
			//key.point_id = start_i;
			key.start = start_i;
			key.end = end_i;
			key.global_map_id = i*numMapper+j;
			key.local_map_id = j;

			val.d_Points = d_points;
			val.d_tempDenominators = d_tempDenominators;
			val.d_tempClusters = d_tempClusters;

			AddPandaTask(gpu_job_conf, &key, &val, sizeof(KM_KEY_T), sizeof(KM_VAL_T));
			start_i = end_i;
		}//for

		thread_info[i].job_conf = gpu_job_conf;
		thread_info[i].device_type = GPU_ACC;
	}//for
	double t1 = PandaTimer();
		
	int iter = 0;
	while (iter<maxIter)
	{
		PandaMetaScheduler(thread_info, panda);
		/*for (int i=0; i<num_gpus; i++){
			if (pthread_create(&(no_threads[i]),NULL,Panda_Map,(char *)&(thread_info[i]))!=0) 
				perror("Thread creation failed!\n");
		}//for num_gpus
		for (int i=0; i<num_gpus; i++){
			void *exitstat;
			if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
		}//for
		int gpu_id;
		cudaGetDevice(&gpu_id);
		ShowLog("current gpu_id:%d",gpu_id);
		if(gpu_id !=(num_gpus-1)){
			checkCudaErrors(cudaSetDevice(num_gpus-1));
			ShowLog("changing GPU context to device:%d",num_gpus-1);
		}//if*/
		
		/*	for (int i=1; i<num_gpus; i++){
			Panda_Shuffle_Merge_GPU(thread_info[i-1].d_g_state, thread_info[i].d_g_state);
		}//for
		//cudaThreadSynchronize();
		Panda_Reduce(&thread_info[num_gpus-1]);	*/
		iter++;
		cudaThreadSynchronize();
	}//while iterations

	double t2 = PandaTimer();
	ShowLog("Panda C-means take %f sec", t2-t1);
	DoLog2Disk("== Panda C-means numPt:%d	dim:%d	K:%d	numMapper:%d	maxIter:%d take %f sec",numPt,dim,K,numMapper,maxIter, t2-t1);

	return 0;
}//		
