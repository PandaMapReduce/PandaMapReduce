/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: Global.h 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#ifndef __USER_H__
#define __USER_H__


#include "Panda.h"

__device__ void gpu_map(void *KEY, void*VAL, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx);

__device__ void gpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context *d_g_state, int map_task_idx);

__device__ void gpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context d_g_state);

__device__ int gpu_compare(const void *d_a, int len_a, const void *d_b, int len_b);


void cpu_map(void *KEY, void*VAL, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx);

void cpu_reduce(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context* d_g_state);

int cpu_compare(const void *d_a, int len_a, const void *d_b, int len_b);

void cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context *d_g_state, int map_task_idx);


typedef struct
{
        //int point_id;
		int local_map_id;
        int dim;
        int K;
        int* ptrClusterId;
        int start;
        int end;
        int global_map_id;

} KM_KEY_T;

typedef struct
{
        int* ptrPoints;
        int* ptrClusters;
        int* ptrChange;

        float *d_tempClusters;
        float *d_tempDenominators;
        float *d_Clusters;
        float *d_Points;

} KM_VAL_T;


#endif
