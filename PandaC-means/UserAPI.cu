/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	Code Name: Panda 0.1
	File: reduce.cu 
	Time: 2012-07-01 
	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.
 
 */

#ifndef __REDUCE_CU__
#define __REDUCE_CU__

#include "Panda.h"
#include "UserAPI.h"

//invoke cmeans_cpu_map_cpp compiled with g++
void cpu_map(void *key, void *val, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx){
	
	cmeans_cpu_map_cpp(key, val, keySize, valSize);
	CPUEmitMapOutput(key, val, sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state, map_task_idx);

}


void cpu_map2(void *key, void *val, int keySize, int valSize, cpu_context *d_g_state, int map_task_idx){

	KM_KEY_T* pKey = (KM_KEY_T*)key;
	KM_VAL_T* pVal = (KM_VAL_T*)val;
	
	int dim = pKey->dim;
	//int dim_4;
	int K = pKey->K;
	int start = pKey->start;
	int end = pKey->end;
	int index = pKey->local_map_id;
	//TODO there could be problem here when running C-means with more than one GPU
	//index = 0;

	float *point	= (float*)(pVal->d_Points);
	float *cluster	= (float*)(pVal->d_Clusters);

	float * tempClusters = pVal->d_tempClusters+index*dim*K;
	float * tempDenominators = pVal->d_tempDenominators+index*K;

	float denominator = 0.0f;
	float membershipValue = 0.0f;

	float *distances = (float *)malloc(sizeof(float)*K);
	float *numerator = (float *)malloc(sizeof(float)*K);
	
	for(int i=0; i<K; i++){
		distances[i]=0.0f;
		numerator[i]=0.0f;
	}//for

	//printf("map_task_id 0:%d thread_id:%d\n",map_task_idx,THREAD_ID);
	for (int i=start; i<end; i++){
		float *curPoint = (float*)(pVal->d_Points + i*dim);
		for (int k = 0; k < K; ++k)
		{
			float* curCluster = (float*)(pVal->d_Clusters + k*dim);
			distances[k] = 0.0;
			//printf("dim:%d\n",dim);
			//dim_4 = dim;
			float delta = 0.0;	
			
			for (int j = 0; j < dim; ++j)
			{
				delta = curPoint[j]-curCluster[j];
				distances[k] += (delta*delta);
			}//for
			
			numerator[k] = powf(distances[k],2.0f/(2.0-1.0))+1e-30;
			denominator  = denominator + 1.0f/(numerator[k]+1e-30);
		}//for

		for (int k = 0; k < K; ++k)
		{
			membershipValue = 1.0f/powf(numerator[k]*denominator,(float)2.0);
			for(int d =0; d<dim; d++){
				//float pt = curePoint[d].x;
				tempClusters[k*dim+d] += (curPoint[d])*membershipValue;
				
			}
			tempDenominators[k]+= membershipValue;
		}//for 
	}//for
	//printf("map_task_id 1:%d\n",map_task_idx);
	
	free(distances);
	free(numerator);
	
	//TODO
	pKey->local_map_id = 0;
	pKey->end = 0;
	pKey->start = 0;
	pKey->global_map_id = 0;
	
	CPUEmitMapOutput(key, val, sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state, map_task_idx);

}//void

__device__ void gpu_map(void *key, void *val, int keySize, int valSize, gpu_context *d_g_state, int map_task_idx){

	KM_KEY_T* pKey = (KM_KEY_T*)key;
	KM_VAL_T* pVal = (KM_VAL_T*)val;
	
	int dim = pKey->dim;
	int dim_4;
	int K = pKey->K;
	int start = pKey->start;
	int end = pKey->end;
	int index = pKey->local_map_id;
	//TODO there could be problem here when running C-means with more than one GPU
	index = 0;

	float4 *point =(float4*)(pVal->d_Points);
	float* cluster = (float*)(pVal->d_Clusters);

	float * tempClusters = pVal->d_tempClusters+index*dim*K;
	float * tempDenominators = pVal->d_tempDenominators+index*K;

	float denominator = 0.0f;
	float membershipValue = 0.0f;

	float *distances = (float *)malloc(sizeof(float)*K);
	float *numerator = (float *)malloc(sizeof(float)*K);
	
	
	for(int i=0; i<K; i++){
		distances[i]=0.0f;
		numerator[i]=0.0f;
	}//for

	//printf("map_task_id 0:%d thread_id:%d\n",map_task_idx,THREAD_ID);
	for (int i=start; i<end; i++){
		float4* curPoint = (float4*)(pVal->d_Points + i*dim);
		for (int k = 0; k < K; ++k)
		{
			float4* curCluster = (float4*)(pVal->d_Clusters + k*dim);
			distances[k] = 0.0;
			//printf("dim:%d\n",dim);
			dim_4 = dim/4;
			float delta = 0.0;	
			
			for (int j = 0; j < dim_4; ++j)
			{
				float4 pt = curPoint[j];
				float4 cl = curCluster[j];

				delta = pt.x-cl.x;
				distances[k] += (delta*delta);
				delta = pt.y-cl.y;
				distances[k] += (delta*delta);
				delta = pt.z-cl.z;
				distances[k] += (delta*delta);
				delta = pt.w-cl.w;
				distances[k] += (delta*delta);

			}//for
				
			int remainder = dim & 0x00000003;
			float* rPoint = (float*)(curPoint+dim_4);
			float* rCluster = (float*)(curCluster+dim_4);
			
			for (int j = 0; j < remainder; j++)
			{
				float pt = rPoint[j];
				float cl = rCluster[j];
				delta = pt - cl;
				distances[k] += (delta*delta);				
			}			
			numerator[k] = powf(distances[k],2.0f/(2.0-1.0))+1e-30;
			denominator  = denominator + 1.0f/(numerator[k]+1e-30);
		}//for

		for (int k = 0; k < K; ++k)
		{
			membershipValue = 1.0f/powf(numerator[k]*denominator,(float)2.0);
			for(int d =0;d<dim_4;d++){
				//float pt = curePoint[d].x;
				tempClusters[k*dim+d] += (curPoint[d].x)*membershipValue;
				tempClusters[k*dim+d] += (curPoint[d].y)*membershipValue;
				tempClusters[k*dim+d] += (curPoint[d].z)*membershipValue;
				tempClusters[k*dim+d] += (curPoint[d].w)*membershipValue;
			}
			tempDenominators[k]+= membershipValue;
		}//for 
	}//for
	//printf("map_task_id 1:%d\n",map_task_idx);
	
	free(distances);
	free(numerator);
	
	//TODO
	pKey->local_map_id = 0;
	pKey->end = 0;
	pKey->start = 0;
	pKey->global_map_id = 0;
	
	GPUEmitMapOutput(key, val, sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state, map_task_idx);
	
}//map2





__device__ int gpu_compare(const void *key_a, int len_a, const void *key_b, int len_b)
{
	//KM_KEY_T *ka = (KM_KEY_T*)key_a;
	//KM_KEY_T *kb = (KM_KEY_T*)key_b;

	return 0;

	/*
	if (ka->i > kb->i)
		return 1;

	if (ka->i > kb->i)
		return -1;

	if (ka->i == kb->i)
		return 0;
	*/
}


int cpu_compare(const void *key_a, int len_a, const void *key_b, int len_b)
{
	//KM_KEY_T *ka = (KM_KEY_T*)key_a;
	//KM_KEY_T *kb = (KM_KEY_T*)key_b;

	return 0;

	/*
	if (ka->i > kb->i)
		return 1;

	if (ka->i > kb->i)
		return -1;

	if (ka->i == kb->i)
		return 0;
		*/

}


void cpu_reduce(void *key, val_t* vals, int keySize, int valCount, cpu_context* d_g_state){
	cmeans_cpu_reduce_cpp(key,  vals, keySize, valCount);
	CPUEmitReduceOutput(key,vals,sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state);
}

//-------------------------------------------------------------------------
//Reduce Function in this application
//-------------------------------------------------------------------------

void cpu_reduce2(void *key, val_t* vals, int keySize, int valCount, cpu_context* d_g_state)
{

		KM_KEY_T* pKey = (KM_KEY_T*)key;
        int dim = pKey->dim;
        int K = pKey->K;



        float* myClusters = (float*) malloc(sizeof(float)*dim*K);
        float* myDenominators = (float*) malloc(sizeof(float)*K);
        memset(myClusters,0,sizeof(float)*dim*K);
        memset(myDenominators,0,sizeof(float)*K);

        float *tempClusters = NULL;
        float *tempDenominators = NULL;
		

        for (int i = 0; i < valCount; i++)
        {
                int index = pKey->local_map_id;


				KM_VAL_T* pVal = (KM_VAL_T*)(vals[i].val);
                tempClusters = pVal->d_tempClusters + index*K*dim;
                tempDenominators = pVal->d_tempDenominators+ index*K;
                for (int k = 0; k< K; k++){
                        for (int j = 0; j< dim; j++)
                                myClusters[k*dim+j] += tempClusters[k*dim+j];
                        myDenominators[k] += tempDenominators[k];
                }//for
        }//end for


        for (int k = 0; k< K; k++){
			for (int i = 0; i < dim; i++){
						//printf("K:%d dim:%d myDenominators[i]:%f",K,dim,myDenominators[i]);
                        myClusters[i] /= ((float)myDenominators[i]+0.0001);
						//printf("%f ",myClusters[i]);
			}//for
			//printf("\n");
        }//for

		
		free(myClusters);
		free(myDenominators);

		CPUEmitReduceOutput(key,vals,sizeof(KM_KEY_T), sizeof(KM_VAL_T), d_g_state);

}

__device__ void gpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, gpu_context *d_g_state, int map_task_idx){
		
		
}//reduce2

void cpu_combiner(void *KEY, val_t* VAL, int keySize, int valCount, cpu_context *d_g_state, int map_task_idx){
		
		
}//reduce2

__device__ void gpu_reduce(void *key, val_t* vals, int keySize, int valCount, gpu_context d_g_state)
{
		//printf("valCount:%d\n",valCount);
		KM_KEY_T* pKey = (KM_KEY_T*)key;
        //KM_VAL_T* pVal = (KM_VAL_T*)vals;
        int dim = pKey->dim;
        int K = pKey->K;
				
        float* myClusters = (float*) malloc(sizeof(float)*dim*K);
        float* myDenominators = (float*) malloc(sizeof(float)*K);
        memset(myClusters,0,sizeof(float)*dim*K);
        memset(myDenominators,0,sizeof(float)*K);

        float *tempClusters = NULL;
        float *tempDenominators = NULL;
        for (int i = 0; i < valCount; i++)
        {
                int index = pKey->local_map_id;
				KM_VAL_T* pVal = (KM_VAL_T*)(vals[i].val);
                tempClusters = pVal->d_tempClusters + index*K*dim;
                tempDenominators = pVal->d_tempDenominators+ index*K;
                for (int k = 0; k< K; k++){
                        for (int j = 0; j< dim; j++)
                                myClusters[k*dim+j] += tempClusters[k*dim+j];
                        myDenominators[k] += tempDenominators[k];
                }//for
        }//end for

        for (int k = 0; k< K; k++){
			for (int i = 0; i < dim; i++){
                        myClusters[i] /= (float)myDenominators[i];
						//printf("%f ",myClusters[i]);
			}//for
			//printf("\n");
        }//for

		//printf("TID reduce2:%d\n",TID);
		GPUEmitReduceOuput(key,vals,sizeof(KM_KEY_T), sizeof(KM_VAL_T), &d_g_state);
		
		free(myClusters);
		free(myDenominators);
				
}//reduce2

#endif //__REDUCE_CU__
