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
		printf("usage: %s [numPt]  [gpu/cpu ratio] [numMapperPerCPU][numMapperPerGPU] [maxIter] [numGpus]\n", argv[0]);
		exit(-1);//[Dimensions] [numClusters]
	}//if
	
	//printf("start %s  %s  %s\n",argv[0],argv[1],argv[2]);
	int numPt = atoi(argv[1]);
	int dim = 100;//atoi(argv[2]);
	int K = 10;//atoi(argv[3]);
	float ratio = atof(argv[2]);
	int numMapperCPU = atoi(argv[3]);
	int numMapperGPU = atoi(argv[4]);
	int maxIter = atoi(argv[5]);
	int num_gpus = atoi(argv[6]);
	int num_cpus_groups = 1;
	
	panda_context *panda = CreatePandaContext();
	panda->num_gpus = num_gpus;
	panda->num_cpus_groups = num_cpus_groups;
	panda->ratio = 0.0;

	ShowLog("numPt:%d	dim:%d	K:%d	numMapperGPU:%d numMapperCPU:%d	maxIter:%d",numPt,dim,K,numMapperGPU,numMapperCPU,maxIter);
	float* h_points = GenPointsFloat(numPt, dim);
	float* h_cluster = GenInitCentersFloat(h_points, numPt, dim, K);
	
	int numgpus = 0;
	cudaGetDeviceCount(&numgpus);
	if (num_gpus >= numgpus)		num_gpus = numgpus;
			
	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*(num_gpus+num_cpus_groups));
	thread_info_t *thread_info = (thread_info_t*)malloc(sizeof(thread_info_t)*(num_gpus+num_cpus_groups));
		
	int gpuWorkLoad = int(numPt * ratio);
	int cpuWorkLoad = numPt - gpuWorkLoad;
	if (num_gpus == 0) gpuWorkLoad = 0;
	if (num_cpus_groups == 0) cpuWorkLoad = 0;


	ShowLog("numPt:%d cpu workload:%d gpu workload:%d", numPt, cpuWorkLoad, gpuWorkLoad );

	for (int dev_id=0; dev_id<num_gpus; dev_id++){

		job_configuration *gpu_job_conf = CreateJobConf();
		gpu_job_conf->num_gpus = num_gpus;
		gpu_job_conf->num_mappers = numMapperGPU;
		gpu_job_conf->auto_tuning = false;
		gpu_job_conf->ratio = (double)ratio;
		gpu_job_conf->auto_tuning_sample_rate = -1;//sample_rate;
		gpu_job_conf->iterative_support = false;
		//gpu_job_conf->local_combiner = true;

		int tid = dev_id;		
		float* d_points	=	NULL;
		float* d_cluster =	NULL;
		//int* d_change	=	NULL;
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
		//checkCudaErrors(cudaMalloc((void**)&d_change, sizeof(int)));
		//checkCudaErrors(cudaMemset(d_change, 0, sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_tempClusters,K*dim*numMapperGPU*sizeof(float)));
		checkCudaErrors(cudaMemset(d_tempClusters, 0, sizeof(float)*K*dim*numMapperGPU));
		checkCudaErrors(cudaMalloc((void**)&d_tempDenominators,numMapperGPU * K * sizeof(float)));
		
		checkCudaErrors(cudaMemset(d_tempDenominators, 0, sizeof(float)*K*numMapperGPU));
		
		thread_info[dev_id].tid = dev_id;
		//thread_info[dev_id].num_gpus = num_gpus;
		thread_info[dev_id].device_type = GPU_ACC;
		
		cudaDeviceProp gpu_dev;
		cudaGetDeviceProperties(&gpu_dev, dev_id);

		ShowLog("Configure Device ID:%d: Device Name:%s", dev_id, gpu_dev.name);
		thread_info[dev_id].device_name = gpu_dev.name;
				
		KM_VAL_T val;
		val.ptrPoints = (int *)d_points;
		val.ptrClusters = (int *)d_cluster;
		val.d_Points = d_points;
		val.d_Clusters = d_cluster;
		//val.ptrChange = d_change;
		
		KM_KEY_T key;
		key.dim = dim;
		key.K = K;
		key.ptrClusterId = d_clusterId;
		
		int numPtPerGPU = gpuWorkLoad/num_gpus;
		int start = dev_id*numPtPerGPU;
		int end = start+numPtPerGPU;
		if (dev_id==num_gpus-1)
			end = gpuWorkLoad;
		
		int numPtPerMap = (end-start)/numMapperGPU;
		ShowLog("GPU numPtPerMap:%d startPt:%d  endPt:%d numPt:%d gpuWorkLoad:%d",numPtPerMap,start,end,numPt,gpuWorkLoad);

		int start_i,end_i;
		start_i = start;
		for (int j = 0; j < numMapperGPU; j++)
		{	
			end_i = start_i + numPtPerMap;
			if (dev_id<(end-start)%numMapperGPU)
				end_i++;
			
			//ShowLog("start_i:%d, start_j:%d",start_i,end_i);
			//key.point_id = start_i;
			key.start = start_i;
			key.end = end_i;
			key.global_map_id = dev_id*numMapperGPU+j;
			key.local_map_id = j;

			val.d_Points = d_points;
			val.d_tempDenominators = d_tempDenominators;
			val.d_tempClusters = d_tempClusters;

			AddPandaTask(gpu_job_conf, &key, &val, sizeof(KM_KEY_T), sizeof(KM_VAL_T));
			start_i = end_i;
		}//for
		thread_info[dev_id].job_conf = gpu_job_conf;
		thread_info[dev_id].device_type = GPU_ACC;
	}//for

	for (int dev_id=num_gpus; dev_id < num_gpus+num_cpus_groups; dev_id++){

		job_configuration *cpu_job_conf = CreateJobConf();
		cpu_job_conf->num_cpus_groups = num_cpus_groups;
		cpu_job_conf->num_cpus_cores = getCPUCoresNum();
		//cpu_job_conf->local_combiner = true;
		cpu_job_conf->ratio = (double)ratio;
		cpu_job_conf->auto_tuning_sample_rate = -1;
		cpu_job_conf->iterative_support = false;
		cpu_job_conf->local_combiner = false;

		int tid = dev_id;		
		float* d_points	=	NULL;
		float* d_cluster =	NULL;
		int* d_change	=	NULL;
		int* d_clusterId =	NULL;
		
		float* h_tempClusters = NULL;
		float* h_tempDenominators = NULL;

		h_tempClusters = (float *)malloc(K*dim*numMapperGPU*sizeof(float));
		h_tempDenominators = (float *)malloc(numMapperGPU*K*sizeof(float));

		/*
		checkCudaErrors(cudaSetDevice(tid));
		checkCudaErrors(cudaMalloc((void**)&d_points, numPt*dim*sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_points, h_points, numPt*dim*sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void**)&d_clusterId, numPt*sizeof(int)));
		checkCudaErrors(cudaMemset(d_clusterId, 0, numPt*sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_cluster, K*dim*sizeof(int)));
		checkCudaErrors(cudaMemcpy(d_cluster, h_cluster, K*dim*sizeof(int), cudaMemcpyHostToDevice));
		//checkCudaErrors(cudaMalloc((void**)&d_change, sizeof(int)));
		//checkCudaErrors(cudaMemset(d_change, 0, sizeof(int)));
		checkCudaErrors(cudaMalloc((void**)&d_tempClusters,K*dim*numMapper*sizeof(float)));
		checkCudaErrors(cudaMemset(d_tempClusters, 0, sizeof(float)*K*dim*numMapper));
		checkCudaErrors(cudaMalloc((void**)&d_tempDenominators,numMapper * K * sizeof(float)));
		checkCudaErrors(cudaMemset(d_tempDenominators, 0, sizeof(float)*K*numMapper));
		cudaGetDeviceProperties(&gpu_dev, i);
		ShowLog("Configure Device ID:%d: Device Name:%s", i, gpu_dev.name);
		*/		

		thread_info[dev_id].device_name = "CPU";
				
		KM_VAL_T val;
		val.ptrPoints = (int *)h_points;
		val.ptrClusters = (int *)h_cluster;
		val.d_Points = h_points;
		val.d_Clusters = h_cluster;
		//val.ptrChange = d_change;
		
		KM_KEY_T key;
		key.dim = dim;
		key.K = K;
		int * h_clusterId = (int*)malloc(sizeof(int)*numPt);
		key.ptrClusterId = h_clusterId;
		
		int numPtPerCPU = cpuWorkLoad/num_cpus_groups;
		int start = cpuWorkLoad+ (dev_id-num_gpus)*numPtPerCPU;
		int end = start+numPtPerCPU;

		if (dev_id==num_gpus+num_cpus_groups-1)
			end = numPt;
		
		int numPtPerMap = (end-start)/numMapperCPU;
		ShowLog("CPU numPtPerMap:%d startPt:%d  endPt:%d numPt:%d cpuWorkLoad:%d",numPtPerMap,start,end,numPt,cpuWorkLoad);

		int start_i, end_i;
		start_i = start;
		for (int j = 0; j < numMapperCPU; j++)
		{	
			end_i = start_i + numPtPerMap;
			if (dev_id<(end-start)%numMapperCPU)
				end_i++;
			
			//ShowLog("start_i:%d, start_j:%d",start_i,end_i);
			//key.point_id = start_i;
			key.start = start_i;
			key.end = end_i;
			key.global_map_id = dev_id*numMapperCPU+j;
			key.local_map_id = j;

			//val.d_Points = h_points;
			val.d_tempDenominators = h_tempDenominators;
			val.d_tempClusters = h_tempClusters;

			AddPandaTask(cpu_job_conf, &key, &val, sizeof(KM_KEY_T), sizeof(KM_VAL_T));
			start_i = end_i;
		}//for
		
		thread_info[dev_id].job_conf = cpu_job_conf;
		thread_info[dev_id].device_type = CPU_ACC;
		thread_info[dev_id].tid = dev_id;
		
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
	DoLog2Disk("== Panda C-means numPt:%d	dim:%d	K:%d	numMapperGPU:%d	numMapperCPU:%d maxIter:%d take %f sec",numPt,dim,K,numMapperGPU,numMapperCPU,maxIter, t2-t1);

	return 0;
}//		