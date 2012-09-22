/*	
	Copyright 2012 The Trustees of Indiana University.  All rights reserved.
	CGL MapReduce Framework on GPUs and CPUs
	
	Code Name: Panda 
	
	File: PandaSched.cu 
	First Version:		2012-07-01 V0.1
	Current Version:	2012-09-01 V0.3	
	Last Updates:		2012-09-02

	Developer: Hui Li (lihui@indiana.edu)

	This is the source code for Panda, a MapReduce runtime on GPUs and CPUs.

 */

#ifndef _PANDASCHED_CU_
#define _PANDASCHED_CU_

// includes, kernels
#include "Panda.h"
#include "UserAPI.h"


//--------------------------------------------------
//  PandaMetaScheduler
//--------------------------------------------------
/*
 * 1) input a set of panda worker (thread)
 * 2) each panda worker consist of one panda job and pand device
 * 3) copy input data from pand job to pand device 
 */



//For version 0.3
void PandaMetaScheduler(thread_info_t *thread_info, panda_context *panda){

	int num_gpus = panda->num_gpus;
	int num_cpus_groups = panda->num_cpus_groups;
	float ratio = panda->ratio;
	
	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*(num_gpus + num_cpus_groups));
	
	int assigned_gpu_id = 0;
	int assigned_cpu_group_id = 0;
	for (int dev_id=0; dev_id<(num_gpus + num_cpus_groups); dev_id++){

		if (thread_info[dev_id].device_type == GPU_ACC){
			
			gpu_context *d_g_state = CreateGPUContext();
			job_configuration *gpu_job_conf = (job_configuration *)(thread_info[dev_id].job_conf);

			d_g_state->num_mappers = gpu_job_conf->num_mappers;
			d_g_state->num_reducers = gpu_job_conf->num_reducers;
			d_g_state->num_gpus = num_gpus;
			d_g_state->gpu_id = assigned_gpu_id;
			d_g_state->local_combiner = gpu_job_conf->local_combiner;
			d_g_state->iterative_support = gpu_job_conf->iterative_support;

			thread_info[dev_id].tid = dev_id;
			thread_info[dev_id].d_g_state = d_g_state;

			ShowLog("Assigned Dev_ID:[%d] GPU_ACC TID:%d",assigned_gpu_id,thread_info[dev_id].tid);
			assigned_gpu_id++;

		}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
			
			cpu_context *d_g_state = CreateCPUContext();
			job_configuration *cpu_job_conf = (job_configuration *)(thread_info[dev_id].job_conf);
			
			d_g_state->cpu_group_id = assigned_cpu_group_id;
			d_g_state->local_combiner = cpu_job_conf->local_combiner;
			d_g_state->iterative_support = cpu_job_conf->iterative_support;

			thread_info[dev_id].tid = dev_id;
			thread_info[dev_id].d_g_state = d_g_state;

			ShowLog("Assigned Dev_ID:[%d] CPU_ACC TID:%d",dev_id,thread_info[dev_id].tid);
			assigned_cpu_group_id++;

		}//if
	}//for
	
	///////////////////////////////////////////////////
		
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){

		if (thread_info[dev_id].device_type == GPU_ACC){

				job_configuration *gpu_job_conf = (job_configuration *)(thread_info[dev_id].job_conf);
				int start_task_id = 0;
				int end_task_id = gpu_job_conf->num_input_record;
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);
				
				AddMapInputRecordGPU(d_g_state,(gpu_job_conf->input_keyval_arr), start_task_id,end_task_id);
				
		}//if
	
		if (thread_info[dev_id].device_type == CPU_ACC){

				job_configuration *cpu_job_conf = (job_configuration *)(thread_info[dev_id].job_conf);
				int start_task_id = 0;
				int end_task_id = cpu_job_conf->num_input_record;
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				
				//ShowWarn("cpu_job_conf->num_input_record:%d ",cpu_job_conf->num_input_record);
				AddMapInputRecordCPU(d_g_state,(cpu_job_conf->input_keyval_arr),start_task_id, end_task_id);
				
		}//if
	}//for

	//1) initial map; 2)run map task; 3) run local combiner
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){
		if (pthread_create(&(no_threads[dev_id]), NULL, Panda_Map, (char *)&(thread_info[dev_id])) != 0) 
			perror("Thread creation failed!\n");
	}//for

	for (int i = 0; i < num_gpus + num_cpus_groups; i++){
		void *exitstat;
		if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
	}//for
	
	
	for (int i = 0; i < num_gpus + num_cpus_groups; i++){

		if (thread_info[i].device_type == CPU_ACC){
			
			PandaShuffleMergeCPU((panda_context*)panda, (cpu_context*)(thread_info[i].d_g_state));
			//ShowLog("==>CPU d_g_state_1->sorted_keyvals_arr_len:%d",panda->sorted_keyvals_arr_len);
		}//if

		if (thread_info[i].device_type == GPU_ACC){
			
			PandaShuffleMergeGPU((panda_context*)panda, (gpu_context*)(thread_info[i].d_g_state));
			//ShowLog("==>GPU d_g_state_1->sorted_keyvals_arr_len:%d",panda->sorted_keyvals_arr_len);
		}//if
			
	}//for
	
	int num_sorted_intermediate_record = panda->sorted_keyvals_arr_len;
	int num_tasks_4_cpu = num_sorted_intermediate_record*(panda->ratio);
	int num_tasks_4_gpu = num_sorted_intermediate_record*(1 - panda->ratio);

	int task_per_cpu = 0;
	if (num_cpus_groups>0) task_per_cpu = num_tasks_4_cpu/(num_cpus_groups);

	int task_per_gpu = 0;
	if (num_gpus>0) task_per_gpu = num_tasks_4_gpu/(num_gpus);
	int *split = (int*)malloc(sizeof(int)*(num_gpus+num_cpus_groups));
	
	for (int i=0; i<num_gpus; i++){
		if (i==0) 	split[0] = task_per_gpu;
		else if (i== num_gpus -1) split[i] = num_tasks_4_gpu;
		else split[i] = split[i-1] + task_per_gpu;
		//ShowLog("i:%d  %d",i, split[i]);
	}//for
	
	for (int i=num_gpus; i<num_gpus+num_cpus_groups; i++){
		if (i==0) 	split[0] = task_per_cpu;
		else if (i == num_gpus + num_cpus_groups - 1) split[i] = num_tasks_4_gpu + num_tasks_4_cpu;
		else split[i] = split[i-1] + task_per_cpu;
		//ShowLog("i:%d  %d",i, split[i]);
	}//for
	split[num_gpus + num_cpus_groups - 1] = num_sorted_intermediate_record;

	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){
	
		int start_row_id = 0;
		if (dev_id>0) start_row_id = split[dev_id-1];
		int end_row_id = split[dev_id];
				
		if (thread_info[dev_id].device_type == GPU_ACC){
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);
				AddReduceInputRecordGPU(d_g_state,(panda->sorted_intermediate_keyvals_arr), start_row_id, end_row_id);
		}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				d_g_state->sorted_keyvals_arr_len = 0;
				AddReduceInputRecordCPU(d_g_state,(panda->sorted_intermediate_keyvals_arr), start_row_id, end_row_id);
		}//if

	}//for

	for (int dev_id = 0; dev_id < (num_gpus+num_cpus_groups); dev_id++){
		if (pthread_create(&(no_threads[dev_id]),NULL,Panda_Reduce,(char *)&(thread_info[dev_id]))!=0) 
			perror("Thread creation failed!\n");
	}//for
		
	for (int i=0; i < num_gpus + num_cpus_groups; i++){
		void *exitstat;
		if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
	}//for

	//TODO Reduce Merge
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	//Panda_Reduce_Merge(&thread_info[num_gpus-1]);															 //
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	
	int total_output_records = 0;
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){
	
		if (thread_info[dev_id].device_type == GPU_ACC){
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);
				total_output_records += d_g_state->d_reduced_keyval_arr_len;
		}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				total_output_records += d_g_state->sorted_keyvals_arr_len;
		}//if
		
	}//for

	ShowLog("number of reduce output:%d\n",total_output_records);
	ShowLog("=====panda mapreduce job finished=====");

}//PandaMetaScheduler


//Scheduler for version 0.2 depressed

//Ratio = Tcpu/Tgpu
//Tcpu = (execution time on CPU cores for sampled tasks)/(#sampled tasks)
//Tgpu = (execution time on 1 GPU for sampled tasks)/(#sampled tasks)
//smart scheduler for auto tuning; measure the performance of sample data  

float AutoTuningScheduler(thread_info_t *thread_info, panda_context *panda){
	
	ShowLog("AutoTuningScheduler");
	int num_gpus = panda->num_gpus;
	int num_cpus_cores = getCPUCoresNum();//job_conf->num_cpus;
	int num_cpus_groups = panda->num_cpus_groups;
	
	num_gpus = 1;
	num_cpus_groups = 1;

	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*(num_gpus + num_cpus_groups));
	
	int cpu_sampled_tasks_num = 0;
	int gpu_sampled_tasks_num = 0;

	int start_row_id = 0;
	int end_row_id = 0;//job_conf->num_cpus_cores*2; //(job_conf->num_input_record/100);

	int cpu_index = -1;
	int gpu_index = -1;

	for (int tid=0; tid<num_gpus+num_cpus_groups; tid++){
	
		if (thread_info[tid].device_type == GPU_ACC){
			if (gpu_index>=0)
				continue;
			gpu_index = tid;

			gpu_context *d_g_state = CreateGPUContext();
			d_g_state->num_gpus = num_gpus;
			thread_info[tid].d_g_state = d_g_state;

			job_configuration *gpu_job_conf = (job_configuration *)(thread_info[tid].job_conf);
			gpu_sampled_tasks_num = gpu_job_conf->num_input_record;
			start_row_id = 0;
			end_row_id = gpu_job_conf->num_input_record;
			AddMapInputRecordGPU(d_g_state,(gpu_job_conf->input_keyval_arr), start_row_id, end_row_id);
			
		}//if
		
		if (thread_info[tid].device_type == CPU_ACC){
			if (cpu_index>=0)
				continue;
			cpu_index = tid;

			cpu_context *d_g_state = CreateCPUContext();
			//d_g_state->num_cpus_groups = num_cpus_groups;
			thread_info[tid].d_g_state = d_g_state;

			job_configuration *cpu_job_conf = (job_configuration *)(thread_info[tid].job_conf);
			cpu_sampled_tasks_num = cpu_job_conf->num_input_record;
			start_row_id = 0;
			end_row_id = cpu_job_conf->num_input_record;
			AddMapInputRecordCPU(d_g_state,(cpu_job_conf->input_keyval_arr), start_row_id, end_row_id);

		}//if
	}//for 
	
	//cpu_sampled_tasks_num = num_cpus_cores*job_conf->auto_tuning_sample_rate;
	//gpu_sampled_tasks_num = getGPUCoresNum()*job_conf->auto_tuning_sample_rate;
	//if (cpu_sampled_tasks_num>job_conf->num_input_record)
	//if (gpu_sampled_tasks_num>job_conf->num_input_record)
		
	double t1 = PandaTimer();
	Panda_Map((void *)&(thread_info[gpu_index]));
	double t2 = PandaTimer();
	//start_row_id cpu 
	Panda_Map((void *)&(thread_info[cpu_index]));
	double t3 = PandaTimer();
	
	double t_cpu = (t3-t2);///cpu_sampled_tasks_num;
	double t_gpu = (t2-t1);///gpu_sampled_tasks_num;

	if (t_gpu<0.0001)
		t_gpu=0.0001;
	
	//double ratio = (t_cpu*gpu_sampled_tasks_num)/(t_gpu*cpu_sampled_tasks_num);
	
	double ratio = (t_cpu)/(t_gpu);
	ShowLog("cpu time:%f gpu time:%f ratio:%f", (t_cpu), (t_gpu), ratio);
	/*
	char log[128];
	sprintf(log,"	cpu_sampled_tasks:%d cpu time:%f cpu time per task:%f", cpu_sampled_tasks_num, t_cpu, t_cpu/(cpu_sampled_tasks_num));
	DoDiskLog(log);
	sprintf(log,"	gpu_sampled_tasks:%d gpu time:%f gpu time per task:%f	ratio:%f", gpu_sampled_tasks_num, t_gpu, t_gpu/(gpu_sampled_tasks_num), ratio);
	DoDiskLog(log);
	*/
	
	return (ratio);
	
}//void

void PandaDynamicMetaScheduler(thread_info_t *thread_info, panda_context *panda){

	int num_gpus = panda->num_gpus;
	int num_cpus_groups = panda->num_cpus_groups;
	float ratio = panda->ratio;
	
	pthread_t *no_threads = (pthread_t*)malloc(sizeof(pthread_t)*(num_gpus + num_cpus_groups));
	
	for (int dev_id=0; dev_id<(num_gpus + num_cpus_groups); dev_id++){

		int assigned_gpu_id = 0;
		if (thread_info[dev_id].device_type == GPU_ACC){
			
			job_configuration* gpu_job_conf = (job_configuration*)(thread_info[dev_id].job_conf);
			gpu_context *d_g_state = CreateGPUContext();
			d_g_state->num_mappers = gpu_job_conf->num_mappers;
			d_g_state->num_reducers = gpu_job_conf->num_reducers;
			d_g_state->num_gpus = num_gpus;
			d_g_state->gpu_id = assigned_gpu_id;

			thread_info[dev_id].tid = dev_id;
			thread_info[dev_id].d_g_state = d_g_state;

			ShowLog("Assigned Dev_ID:[%d] GPU_ACC TID:%d",assigned_gpu_id,thread_info[dev_id].tid);
			assigned_gpu_id++;
		}//if

		int cpu_group_id = 0;
		if (thread_info[dev_id].device_type == CPU_ACC){
			
			cpu_context *d_g_state = CreateCPUContext();
			d_g_state->cpu_group_id = cpu_group_id;
			thread_info[dev_id].tid = dev_id;
			thread_info[dev_id].d_g_state = d_g_state;

			ShowLog("Assigned Dev_ID:[%d] CPU_ACC TID:%d",dev_id,thread_info[dev_id].tid);
			cpu_group_id++;
		}//if
	}//for

	///////////////////////////////////////////////////
	
	
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){

		if (thread_info[dev_id].device_type == GPU_ACC){

				job_configuration *gpu_job_conf = (job_configuration *)(thread_info[dev_id].job_conf);
				int start_row_id = 0;
				int end_id = gpu_job_conf->num_input_record;
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);

				AddMapInputRecordGPU(d_g_state,(gpu_job_conf->input_keyval_arr), start_row_id,end_id);
				
		}//if
	
		if (thread_info[dev_id].device_type == CPU_ACC){

				job_configuration *cpu_job_conf = (job_configuration *)(thread_info[dev_id].job_conf);
				int start_row_id = 0;
				int end_id = cpu_job_conf->num_input_record;
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				
				AddMapInputRecordCPU(d_g_state,(cpu_job_conf->input_keyval_arr),start_row_id, end_id);
				
		}//if
	}//for
	
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){
		if (pthread_create(&(no_threads[dev_id]), NULL, Panda_Map, (char *)&(thread_info[dev_id])) != 0) 
			perror("Thread creation failed!\n");
	}//for

	for (int i = 0; i < num_gpus + num_cpus_groups; i++){
		void *exitstat;
		if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
	}//for

	//ShowLog("start to merge results of GPU's and CPU's device to Panda scheduler");
	for (int i = 0; i < num_gpus+num_cpus_groups; i++){

		if (thread_info[i].device_type == CPU_ACC)
			PandaShuffleMergeCPU((panda_context*)panda, (cpu_context*)(thread_info[i].d_g_state));

		if (thread_info[i].device_type == GPU_ACC)
			
			((panda_context*)panda, (gpu_context*)(thread_info[i].d_g_state));
			
	}//for
	
	//TODO reduce task ratio 
	int num_sorted_intermediate_record = panda->sorted_keyvals_arr_len;
	int records_per_device = num_sorted_intermediate_record/(num_gpus + num_cpus_groups*ratio);
	
	int *split = (int*)malloc(sizeof(int)*(num_gpus+num_cpus_groups));
	
	for (int i=0; i<num_gpus; i++){
	
				if (i==0) 
				split[0] = records_per_device;
				else
				split[i] = split[i-1] + records_per_device;
				
	}//for
	
	for (int i=num_gpus; i<num_gpus+num_cpus_groups; i++){
	
				if (i==0) 
				split[0] = records_per_device*ratio;
				else 
				split[i] = split[i-1] + records_per_device*ratio;
								
	}//for
	split[num_gpus + num_cpus_groups-1] = num_sorted_intermediate_record;

	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){
	
		int start_row_id = 0;
		if (dev_id>0) start_row_id = split[dev_id-1];
		int end_row_id = split[dev_id];
				
		if (thread_info[dev_id].device_type == GPU_ACC){
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);
				ShowLog("start:%d end:%d",start_row_id,end_row_id);
				AddReduceInputRecordGPU(d_g_state,(panda->sorted_intermediate_keyvals_arr), start_row_id, end_row_id);
		}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				ShowLog("start:%d end:%d",start_row_id,end_row_id);
				AddReduceInputRecordCPU(d_g_state,(panda->sorted_intermediate_keyvals_arr), start_row_id, end_row_id);
		}//if
	}//for

	for (int dev_id = 0; dev_id < (num_gpus+num_cpus_groups); dev_id++){
		if (pthread_create(&(no_threads[dev_id]),NULL,Panda_Reduce,(char *)&(thread_info[dev_id]))!=0) 
			perror("Thread creation failed!\n");
	}//for
		
	for (int i=0; i < num_gpus + num_cpus_groups; i++){
		void *exitstat;
		if (pthread_join(no_threads[i],&exitstat)!=0) perror("joining failed");
	}//for

	//TODO Reduce Merge
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	//Panda_Reduce_Merge(&thread_info[num_gpus-1]);															 //
	///////////////////////////////////////////////////////////////////////////////////////////////////////
	
	int total_output_records = 0;
	for (int dev_id = 0; dev_id<(num_gpus+num_cpus_groups); dev_id++){
	
		if (thread_info[dev_id].device_type == GPU_ACC){
				gpu_context* d_g_state = (gpu_context*)(thread_info[dev_id].d_g_state);
				total_output_records += d_g_state->d_reduced_keyval_arr_len;
		}//if

		if (thread_info[dev_id].device_type == CPU_ACC){
				cpu_context* d_g_state = (cpu_context*)(thread_info[dev_id].d_g_state);
				total_output_records += d_g_state->sorted_keyvals_arr_len;
		}//if
		
	}//for
	ShowLog("number of reduce output:%d\n",total_output_records);
	ShowLog("=====panda mapreduce job finished=====");

}//PandaMetaScheduler


#endif // _PRESCHED_CU_
