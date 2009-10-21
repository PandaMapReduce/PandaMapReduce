#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include <cmeans.h>
#include <cmeansMultiGPUcu.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
//#include <cmeans_kernel.cu>
#include "timers.h"


/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else



bool InitCUDA(void)
{
    int count = 0;
    int i = 0;
    int device = -1;
    int num_procs = 0;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }

    printf("There are %d devices.\n",count);
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            printf("Device #%d, Version: %d.%d\n",i,prop.major,prop.minor);
            // Check if CUDA capable device
            if(prop.major >= 1) {
                if(prop.multiProcessorCount > num_procs) {
                    device = i;
                    num_procs = prop.multiProcessorCount;
                }
            }
        }
    }
    if(device == -1) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return false;
    }

    device = 0;
    printf("Using Device %d\n",device);
    CUDA_SAFE_CALL(cudaSetDevice(device));

    printf("CUDA initialized.\n");
    return true;
}

#endif

unsigned int timer_io; // Timer for I/O, such as reading FCS file and outputting result files
unsigned int timer_memcpy; // Timer for GPU <---> CPU memory copying
unsigned int timer_cpu; // Timer for processing on CPU
unsigned int timer_gpu; // Timer for kernels on the GPU
unsigned int timer_total; // Total time


void printCudaError() {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(error));
    }
}

/************************************************************************/
/* C-means Main                                                            */
/************************************************************************/
int main(int argc, char* argv[])
{
   
    CUT_SAFE_CALL(cutCreateTimer(&timer_io));
    CUT_SAFE_CALL(cutCreateTimer(&timer_memcpy));
    CUT_SAFE_CALL(cutCreateTimer(&timer_cpu));
    CUT_SAFE_CALL(cutCreateTimer(&timer_gpu));
    CUT_SAFE_CALL(cutCreateTimer(&timer_total));
    
    CUT_SAFE_CALL(cutStartTimer(timer_total));
    CUT_SAFE_CALL(cutStartTimer(timer_io));
    
    // [program name]  [data file]
    if(argc != 2){
        printf("Usage Error: must supply data file. e.g. programe_name @opt(flags) file.in\n");
        //char tmp45[8];
        //scanf(tmp45, "%s");
        return 1;
    }

    float* myEvents = ParseSampleInput(argv[1]);
#if FAKE
    free(myEvents);
    myEvents = generateEvents();
#endif
    if(myEvents == NULL){
        return 1;
    }
     
    printf("Parsed file\n");
    
    int num_gpus = 0;       // number of CUDA GPUs

    /////////////////////////////////////////////////////////////////
    // determine the number of CUDA capable GPUs
    //
    cudaGetDeviceCount(&num_gpus);
    if(num_gpus < 1)
    {
        printf("no CUDA capable devices were detected\n");
        return 1;
    }

    /////////////////////////////////////////////////////////////////
    // display CPU and GPU configuration
    //
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);
    for(int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }
    printf("---------------------------\n");
    
    //if(!InitCUDA()) {
    //    return 0;
    //}
    //CUT_DEVICE_INIT(argc, argv);
    
    srand((unsigned)(time(0)));
    
    
    
    CUT_SAFE_CALL(cutStopTimer(timer_io));
    CUT_SAFE_CALL(cutStartTimer(timer_cpu));
    
    clock_t total_start;
    total_start = clock();

    // Allocate arrays for the cluster centers
    float* myClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS);
    float* newClusters = (float*)malloc(sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS);

    // Select random cluster centers
    generateInitialClusters(myClusters, myEvents);

    // Create an array of arrays for temporary cluster centers from each GPU
    float** tempClusters = (float**) malloc(sizeof(float*)*num_gpus);
    float** tempDenominators = (float**) malloc(sizeof(float*)*num_gpus);
    for(int i=0; i < num_gpus; i++) {
        tempClusters[i] = (float*) malloc(sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS);
        tempDenominators[i] = (float*) malloc(sizeof(float)*NUM_CLUSTERS);
        memcpy(tempClusters[i],myClusters,sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS);
    }
    
    float diff; // used to track difference in cluster centers between iterations

    // Transpose the events matrix
    /*
    float* temp = (float*)malloc(sizeof(float)*NUM_EVENTS*ALL_DIMENSIONS);
    for(int i=0; i<NUM_EVENTS; i++) {
        for(int j=0; j<ALL_DIMENSIONS; j++) {
            temp[j*NUM_EVENTS+i] = myEvents[i*ALL_DIMENSIONS+j];
        }
    }
    memcpy(myEvents,temp,sizeof(float)*NUM_EVENTS*ALL_DIMENSIONS);
    free(temp);
    */
    
    
    CUT_SAFE_CALL(cutStopTimer(timer_cpu));
    
    ////////////////////////////////////////////////////////////////
    // run as many CPU threads as there are CUDA devices
    //   each CPU thread controls a different device, processing its
    //   portion of the data.  It's possible to use more CPU threads
    //   than there are CUDA devices, in which case several CPU
    //   threads will be allocating resources and launching kernels
    //   on the same device.  For example, try omp_set_num_threads(2*num_gpus);
    //   Recall that all variables declared inside an "omp parallel" scope are
    //   local to each CPU thread
    //
    omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
    //omp_set_num_threads(2*num_gpus);// create twice as many CPU threads as there are CUDA devices
    #pragma omp parallel shared(myClusters,diff,tempClusters,tempDenominators)
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        printf("hello from thread %d of %d\n",cpu_thread_id,num_cpu_threads);

        // set and check the CUDA device for this CPU thread
        int gpu_id = -1;
        cudaSetDevice(cpu_thread_id % num_gpus);        // "% num_gpus" allows more CPU threads than GPU devices
        cudaGetDevice(&gpu_id);

        printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
    

#if !CPU_ONLY    
        CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
        float* d_distanceMatrix;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_distanceMatrix, sizeof(float)*NUM_EVENTS*NUM_CLUSTERS));
        float* d_E;// = AllocateEvents(myEvents);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_E, sizeof(float)*NUM_EVENTS*ALL_DIMENSIONS));
        float* d_C;// = AllocateClusters(myClusters);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_C, sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS));
        float* d_nC;// = AllocateCM(cM);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_nC, sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS));
        float* d_denoms;// = AllocateCM(cM);
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_denoms, sizeof(float)*NUM_CLUSTERS));
        int size = sizeof(float)*ALL_DIMENSIONS*NUM_EVENTS;
        CUDA_SAFE_CALL(cudaMemcpy(d_E, myEvents, size, cudaMemcpyHostToDevice));
        size = sizeof(float)*ALL_DIMENSIONS*NUM_CLUSTERS;
        CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));
        CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
#endif
        clock_t cpu_start, cpu_stop;
        cpu_start = clock();
        printf("Starting C-means\n");
        float averageTime = 0;
        int iterations = 0;
        
        // Compute starting/finishing indexes for the events for each gpu
        int start = cpu_thread_id*NUM_EVENTS/num_gpus;
        int finish = (cpu_thread_id+1)*NUM_EVENTS/num_gpus;
        if(cpu_thread_id == (num_gpus-1)) {
            finish = NUM_EVENTS;
        }
        //start = 0; finish = NUM_EVENTS;
        printf("GPU %d, Starting Event: %d, Ending Event: %d\n",cpu_thread_id,start,finish);

        do{
#if CPU_ONLY
            CUT_SAFE_CALL(cutStartTimer(timer_cpu));
            clock_t cpu_start, cpu_stop;
            cpu_start = clock();

            UpdateClusterCentersCPU(myClusters, myEvents, newClusters);

            cpu_stop = clock();
            printf("Processing time for CPU: %f (ms) \n", (float)(cpu_stop - cpu_start)/(float)(CLOCKS_PER_SEC)*(float)1e3);
            averageTime += (float)(cpu_stop - cpu_start)/(float)(CLOCKS_PER_SEC)*(float)1e3;
            CUT_SAFE_CALL(cutStopTimer(timer_cpu));
#else
            unsigned int timer = 0;
            CUT_SAFE_CALL(cutCreateTimer(&timer));
            CUT_SAFE_CALL(cutStartTimer(timer));

            size = sizeof(float)*ALL_DIMENSIONS*NUM_CLUSTERS;

            CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
            CUDA_SAFE_CALL(cudaMemcpy(d_C, myClusters, size, cudaMemcpyHostToDevice));
            CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
            
            dim3 BLOCK_DIM(1, NUM_THREADS, 1);

            CUT_SAFE_CALL(cutStartTimer(timer_gpu));
            printf("Launching ComputeDistanceMatrix kernel\n");
            ComputeDistanceMatrix<<< NUM_CLUSTERS, 320  >>>(d_C, d_E, d_distanceMatrix, start, finish);

            cudaThreadSynchronize();
            printCudaError();

            printf("Launching UpdateClusterCentersGPU kernel\n");
            UpdateClusterCentersGPU<<< NUM_BLOCKS, BLOCK_DIM >>>(d_C, d_E, d_nC, d_distanceMatrix, d_denoms, start, finish);
            cudaThreadSynchronize();
            printCudaError();
            
            CUT_SAFE_CALL(cutStopTimer(timer_gpu));
            
            // Copy partial centers and denominators to host
            CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
            CUDA_SAFE_CALL(cudaMemcpy(tempClusters[cpu_thread_id], d_nC, sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS, cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(tempDenominators[cpu_thread_id], d_denoms, sizeof(float)*NUM_CLUSTERS, cudaMemcpyDeviceToHost));
            printCudaError();
            CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
            
            CUT_SAFE_CALL(cutStopTimer(timer));
            float thisTime = cutGetTimerValue(timer);
            printf("Processing time for GPU %d: %f (ms) \n", cpu_thread_id, thisTime);
            averageTime += thisTime;
            CUT_SAFE_CALL(cutDeleteTimer(timer));

#endif
            CUT_SAFE_CALL(cutStartTimer(timer_cpu));
        
            #pragma omp barrier
            if(cpu_thread_id == 0) {
                // Sum up the partial cluster centers (numerators)
                for(int i=1; i < num_gpus; i++) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        for(int d=0; d < ALL_DIMENSIONS; d++) {
                            tempClusters[0][c*ALL_DIMENSIONS+d] += tempClusters[i][c*ALL_DIMENSIONS+d];
                        }
                    }
                }

                // Sum up the denominator for each cluster
                for(int i=1; i < num_gpus; i++) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        tempDenominators[0][c] += tempDenominators[i][c];
                    }
                }

                // Divide to get the final clusters
                for(int i=1; i < num_gpus; i++) {
                    for(int c=0; c < NUM_CLUSTERS; c++) {
                        for(int d=0; d < ALL_DIMENSIONS; d++) {
                            tempClusters[0][c*ALL_DIMENSIONS+d] /= tempDenominators[0][c];
                        }
                    }
                }
                diff = 0.0;
                for(int i=0; i < NUM_CLUSTERS; i++){
                    //printf("GPU %d, Cluster %d: ",cpu_thread_id,i);
                    for(int k = 0; k < ALL_DIMENSIONS; k++){
                        //printf("%f ",tempClusters[cpu_thread_id][i*ALL_DIMENSIONS + k]);
                        diff += myClusters[i*ALL_DIMENSIONS + k] - tempClusters[cpu_thread_id][i*ALL_DIMENSIONS + k];
                    }
                    //printf("\n");
                }
                memcpy(myClusters,tempClusters[cpu_thread_id],sizeof(float)*ALL_DIMENSIONS*NUM_CLUSTERS);
                printf("Diff = %f\n", diff);
                printf("Done with iteration #%d\n", iterations);
            }
            
            #pragma omp barrier
            iterations++;
            
            CUT_SAFE_CALL(cutStopTimer(timer_cpu));
            printf("\n");

        } while(abs(diff) > THRESHOLD && iterations < 150); 
        
        if(iterations == 150){
            printf("Warning: c-means did not converge to the %f threshold provided\n", THRESHOLD);
        }
        cpu_stop = clock();
        
        CUT_SAFE_CALL(cutStartTimer(timer_io));
        
        averageTime /= iterations;
        printf("\nTotal Processing time: %f (s) \n", (float)(cpu_stop - cpu_start)/(float)(CLOCKS_PER_SEC));
        printf("C-means complete\n");
        printf("\n");

        for(int i=0; i < NUM_CLUSTERS; i++){
            printf("GPU %d, Center %d: ",cpu_thread_id,i);
            for(int k = 0; k < ALL_DIMENSIONS; k++)
                printf("%f\t", myClusters[i*ALL_DIMENSIONS + k]);
            printf("\n");
        }
        
        exit(0);  // NOTE - Stopping early until we figure out how to make it parallel!!!

        CUT_SAFE_CALL(cutStopTimer(timer_io));
        
        int* finalClusterConfig;
        float mdlTime = 0;
        
#if !MDL_on_GPU
        finalClusterConfig = MDL(myEvents, myClusters, &mdlTime, argv[1]);
#else
        finalClusterConfig = MDLGPU(d_E, d_nC, &mdlTime, argv[1]);
        mdlTime /= 1000.0; // CUDA timer returns time in milliseconds, normalize to seconds
#endif


        CUT_SAFE_CALL(cutStartTimer(timer_io));

        printf("Final Clusters are:\n");
        int newCount = 0;
        for(int i = 0; i < NUM_CLUSTERS; i++){
            if(finalClusterConfig[i]){
                for(int j = 0; j < ALL_DIMENSIONS; j++){
                    newClusters[newCount * ALL_DIMENSIONS + j] = myClusters[i*ALL_DIMENSIONS + j];
                    printf("%f\t", myClusters[i*ALL_DIMENSIONS + j]);
                }
                newCount++;
                printf("\n");
            }
        }
        
        CUT_SAFE_CALL(cutStopTimer(timer_io));

        FindCharacteristics(myEvents, newClusters, newCount, averageTime, mdlTime, iterations, argv[1], total_start);
        
        free(newClusters);
        free(myClusters);
        free(myEvents);
    #if !CPU_ONLY
        CUDA_SAFE_CALL(cudaFree(d_E));
        CUDA_SAFE_CALL(cudaFree(d_C));
        CUDA_SAFE_CALL(cudaFree(d_nC));
    #endif

        CUT_SAFE_CALL(cutStopTimer(timer_total));
        printf("\n\n"); 
        printf("Total Time (ms): %f\n.",cutGetTimerValue(timer_total));
        printf("I/O Time (ms): %f\n.",cutGetTimerValue(timer_io));
        printf("GPU memcpy Time (ms): %f\n.",cutGetTimerValue(timer_memcpy));
        printf("CPU processing Time (ms): %f\n.",cutGetTimerValue(timer_cpu));
        printf("GPU processing Time (ms): %f\n.",cutGetTimerValue(timer_gpu));
        printf("\n\n"); 
        
        //CUT_EXIT(argc, argv);
        printf("\n\n");
    
    } // end of omp_parallel block
    
    return 0;
}

float* generateEvents(){
    float* allEvents = (float*) malloc(NUM_EVENTS*ALL_DIMENSIONS*sizeof(float));
    //generateEvents around (10,10,10), (20, 10, 50), and (50, 50, 0)
    int i, j;
    for(i = 0; i < NUM_EVENTS; i++){
        for(j =0; j < 3; j++){
                
        if(i < NUM_EVENTS/3){
            allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 7;
        }
        else if(i < NUM_EVENTS*2/3){
            switch(j){
                case 0: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 47; break;
                case 1: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 27; break;
                case 2: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 7; break;
                default: printf("error!\n");
            }
        }
        else {
            switch(j){
                case 0: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 47; break;
                case 1: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*6 + 47; break;
                case 2: allEvents[i*3 + j] = rand()/(float(RAND_MAX)+1)*3 ; break;
                default: printf("error!\n");
            }

        }
        }
    }
    return allEvents;
}

void generateInitialClusters(float* clusters, float* events){
    int seed;
    for(int i = 0; i < NUM_CLUSTERS; i++){
        seed = rand() % NUM_EVENTS;
        for(int j = 0; j < ALL_DIMENSIONS; j++){
            clusters[i*ALL_DIMENSIONS + j] = events[seed*ALL_DIMENSIONS + j];
        }
    }
    
}



__host__ float CalculateDistanceCPU(const float* clusters, const float* events, int clusterIndex, int eventIndex){

    float sum = 0;
#if DISTANCE_MEASURE == 0
    for(int i = 0; i < ALL_DIMENSIONS; i++){
        float tmp = events[eventIndex*ALL_DIMENSIONS + i] - clusters[clusterIndex*ALL_DIMENSIONS + i];
        sum += tmp*tmp;
    }
    sum = sqrt(sum);
#endif
#if DISTANCE_MEASURE == 1
    for(int i = 0; i < ALL_DIMENSIONS; i++){
        float tmp = events[eventIndex*ALL_DIMENSIONS + i] - clusters[clusterIndex*ALL_DIMENSIONS + i];
        sum += abs(tmp);
    }
#endif
#if DISTANCE_MEASURE == 2
    for(int i = 0; i < ALL_DIMENSIONS; i++){
        float tmp = abs(events[eventIndex*ALL_DIMENSIONS + i] - clusters[clusterIndex*ALL_DIMENSIONS + i]);
        if(tmp > sum)
            sum = tmp;
    }
#endif
    return sum;
}


__host__ float MembershipValue(const float* clusters, const float* events, int clusterIndex, int eventIndex){
    float myClustDist = CalculateDistanceCPU(clusters, events, clusterIndex, eventIndex);
    float sum =0;
    float otherClustDist;
    for(int j = 0; j< NUM_CLUSTERS; j++){
        otherClustDist = CalculateDistanceCPU(clusters, events, j, eventIndex); 
        if(otherClustDist < .000001)
            return 0.0;
        sum += pow((float)(myClustDist/otherClustDist),float(2/(FUZZINESS-1)));
    }
    return 1/sum;
}



void UpdateClusterCentersCPU(const float* oldClusters, const float* events, float* newClusters){
    
    
    //float membershipValue, sum, denominator;
    float membershipValue, denominator;
    float* numerator = (float*)malloc(sizeof(float)*ALL_DIMENSIONS);
    float* denominators = (float*)malloc(sizeof(float)*NUM_CLUSTERS);
    float* distances = (float*)malloc(sizeof(float)*NUM_CLUSTERS);

    
    for(int i = 0; i < NUM_CLUSTERS; i++){
      denominator = 0.0;
      for(int j = 0; j < ALL_DIMENSIONS; j++)
        numerator[j] = 0;
      for(int j = 0; j < NUM_EVENTS; j++){
        membershipValue = MembershipValue(oldClusters, events, i, j);
        for(int k = 0; k < ALL_DIMENSIONS; k++){
          numerator[k] += events[j*ALL_DIMENSIONS + k]*membershipValue;
        }
        
        denominator += membershipValue;
      }  
      for(int j = 0; j < ALL_DIMENSIONS; j++){
          newClusters[i*ALL_DIMENSIONS + j] = numerator[j]/denominator;
      }  
    }
    

    /*
    memset(newClusters,0.0,sizeof(float)*NUM_CLUSTERS*ALL_DIMENSIONS);    
    memset(denominators,0.0,sizeof(float)*NUM_CLUSTERS);    

    for(int i = 0; i < NUM_EVENTS; i++){
        for(int j = 0; j < ALL_DIMENSIONS; j++)
            numerator[j] = 0;

        // Compute distance from this event to each cluster
        for(int j = 0; j < NUM_CLUSTERS; j++){
            distances[j] = CalculateDistanceCPU(oldClusters,events,j,i);
        }

        // Find sum of all distances
        sum = 0.0;
        for(int j = 0; j < NUM_CLUSTERS; j++) {
            sum += distances[j];
        }

        for(int j = 0; j < NUM_CLUSTERS; j++){
            membershipValue = distances[j] / sum;
            //printf("%f\n",membershipValue);
            if(isnan(membershipValue)) {
                printf("Event #%d: MembershipValue: %f, sum: %f\n",i,membershipValue,sum);
            }

            // Add contribution to the center for each dimension for this cluster
            for(int k = 0; k < ALL_DIMENSIONS; k++){
              newClusters[j*ALL_DIMENSIONS+k] += events[i*ALL_DIMENSIONS + k]*membershipValue;
            }

            denominators[j] += membershipValue;
        }  
    }
    for(int k = 0; k < NUM_CLUSTERS; k++){
        for(int j = 0; j < ALL_DIMENSIONS; j++) {
            newClusters[k*ALL_DIMENSIONS + j] /= denominators[k];
            //printf("%f ",newClusters[k*ALL_DIMENSIONS + j]);
        }
        //printf("\n");
    } 
    //printf("\n"); 
    */
    
    free(numerator);
    free(denominators);
    free(distances);
}




float* ParseSampleInput(const char* filename){
    FILE* myfile = fopen(filename, "r");
    if(myfile == NULL){
        printf("Error: File DNE\n");
        return NULL;
    }
    char myline[1024];
    
    float* retVal = (float*)malloc(sizeof(float)*NUM_EVENTS*ALL_DIMENSIONS);
    myfile = fopen(filename, "r");
#if !LINE_LABELS

    for(int i = 0; i < NUM_EVENTS; i++){
        fgets(myline, 1024, myfile);
        retVal[i*ALL_DIMENSIONS] = (float)atof(strtok(myline, DELIMITER));
        for(int j = 1; j < ALL_DIMENSIONS; j++){
            retVal[i*ALL_DIMENSIONS + j] = (float)atof(strtok(NULL, DELIMITER));
        }
    }
#else
    fgets(myline, 1024, myfile);
    for(int i = 0; i < NUM_EVENTS; i++){
        fgets(myline, 1024, myfile);
        strtok(myline, DELIMITER);
        for(int j = 0; j < ALL_DIMENSIONS; j++){
            retVal[i*ALL_DIMENSIONS + j] = (float)atof(strtok(NULL, DELIMITER));
        }
    }
#endif
    
    fclose(myfile);
    
    
    return retVal;
}

void FreeMatrix(float* d_matrix){
    CUDA_SAFE_CALL(cudaFree(d_matrix));
}

float* BuildQGPU(float* d_events, float* d_clusters, float* mdlTime){
    float* d_matrix;
    int size = sizeof(float) * NUM_CLUSTERS*NUM_CLUSTERS;

    unsigned int timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));
    CUT_SAFE_CALL(cutStartTimer(timer_memcpy));


    cudaMalloc((void**)&d_matrix, size);
    cudaThreadSynchronize();
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");
    //CalculateQMatrixGPU<<<NUM_CLUSTERS,Q_THREADS>>>(d_events, d_clusters, d_matrix);

    CUT_SAFE_CALL(cutStopTimer(timer_memcpy));
    CUT_SAFE_CALL(cutStartTimer(timer_gpu));

    dim3 grid(NUM_CLUSTERS, NUM_CLUSTERS);
    printf("Launching Q Matrix Kernel\n");
    CalculateQMatrixGPUUpgrade<<<grid, Q_THREADS>>>(d_events, d_clusters, d_matrix);
    cudaThreadSynchronize();
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    CUT_SAFE_CALL(cutStopTimer(timer_gpu));
    

    CUT_SAFE_CALL(cutStartTimer(timer_memcpy));
    float* matrix = (float*)malloc(size);
    printf("Copying results to CPU\n");
    cudaError_t error = cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");
    CUT_SAFE_CALL(cutStopTimer(timer_memcpy));

    CUT_SAFE_CALL(cutStopTimer(timer));
    *mdlTime = cutGetTimerValue(timer);
    printf("Processing time for GPU: %f (ms) \n", *mdlTime);
    CUT_SAFE_CALL(cutDeleteTimer(timer));
        
    const char * message = cudaGetErrorString(error);
    printf("Error: %s\n", message);

    
    FreeMatrix(d_matrix);
    return matrix;
}

/*float FindScoreGPU(float* d_matrix, long config){
    float* d_score;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_score, sizeof(float)));
    EvaluateSolutionGPU<<<1, 1>>>(d_matrix, config, d_score);
    float* score = (float*)malloc(sizeof(float));
    CUDA_SAFE_CALL(cudaMemcpy(score, d_score, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_score));
    return *score;
}*/
