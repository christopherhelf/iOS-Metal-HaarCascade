//
//  GroupingTest3.metal
//  iOS-HaarCascade-Improved
//
//  Created by Christopher Helf on 15.12.16.
//  Copyright © 2016 Christopher Helf. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

bool fetchNextPosition(int minNeighbors, device int* neighbors, threadgroup int& counter, int maxCount, threadgroup int& position, threadgroup int* visited);
bool fetchNextPosition(int minNeighbors, device int* neighbors, threadgroup int& counter, int maxCount, threadgroup int& position, threadgroup int* visited) {
    while (counter < maxCount) {
        if (!visited[counter] && neighbors[counter] >= minNeighbors) {
            position = counter;
            counter++;
            return true;
        } else {
            counter++;
        }
    }
    return false;
}

void processWithPosition(
                         uint threadIdx,
                         threadgroup int& position,
                         threadgroup int* visited,
                         int currentCluster,
                         device int* clusters,
                         threadgroup atomic_int& continueToProcess,
                         threadgroup int& shouldContinueShared,
                         int numberOfThreads,
                         threadgroup int* frontier,
                         int numWindows,
                         device int* neighbors,
                         device int* offset,
                         device int* adjacencies);
void processWithPosition(
                         uint threadIdx,
                         threadgroup int& position,
                         threadgroup int* visited,
                         int currentCluster,
                         device int* clusters,
                         threadgroup atomic_int& continueToProcess,
                         threadgroup int& shouldContinueShared,
                         int numberOfThreads,
                         threadgroup int* frontier,
                         int numWindows,
                         device int* neighbors,
                         device int* offset,
                         device int* adjacencies) {
    
    // continue all threads until we have no more nodes to process
    while(shouldContinueShared) {
    
        if (!threadIdx) {
            atomic_store_explicit(&continueToProcess, false, memory_order_relaxed);
        }
        
        threadgroup_barrier( mem_flags::mem_threadgroup );
        
        // iterate through all windows
        for(int i=threadIdx; i<numWindows; i+=numberOfThreads) {
        
            // check whether this was marked
            if ( frontier[i] == position && !visited[i] ) {
            
                // set this node as visited
                visited[i] = 1;
                // set the cluster
                clusters[i] = currentCluster;
            
                // get infos about the neighbors
                int nbCount = neighbors[i];
                int nbOffset = offset[i];
            
                // iterate through all neighbors
                for(int neighborPos=nbOffset; neighborPos<nbOffset+nbCount; neighborPos++) {
                
                    // get the actual id of the neighbor
                    int neighborIdx = adjacencies[neighborPos];
                    
                    if (neighborIdx > 0) {
                        // if the neighbor wasnt visited yet, mark for next processing
                        frontier[neighborIdx] = position;
                        atomic_fetch_or_explicit(&continueToProcess, true, memory_order_relaxed);
                    }
                
                }
            
            }
            
        }
        
        threadgroup_barrier( mem_flags::mem_device );
        
        if (!threadIdx) {
            shouldContinueShared = atomic_load_explicit(&continueToProcess, memory_order_relaxed);
        }
         
        // wait for the flag to be set
        threadgroup_barrier( mem_flags::mem_threadgroup );
        
        if (!shouldContinueShared) {
            break;
        }
        
    }
    
}

kernel void haarCascadeDoGrouping(constant int &numWindows                      [[ buffer(0) ]],
                                  constant int &minNeighbors                    [[ buffer(1) ]],
                                  device int* neighbors                         [[ buffer(2) ]],
                                  device int* offset                            [[ buffer(3) ]],
                                  device int* adjacencies                       [[ buffer(4) ]],
                                  device int& clusterCount                      [[ buffer(5) ]],
                                  device int* clusters                          [[ buffer(6) ]],
                                  
                                  threadgroup int& blockCounter                 [[ threadgroup (0) ]],
                                  threadgroup int& position                     [[ threadgroup (1) ]],
                                  threadgroup int* visited                      [[ threadgroup (2) ]],
                                  threadgroup int& shouldContinueShared         [[ threadgroup (3) ]],
                                  threadgroup int* frontier                     [[ threadgroup (4) ]],
                                  threadgroup atomic_int& continueToProcess     [[ threadgroup (5) ]],
                                  
                                  uint numberOfThreads                          [[threads_per_threadgroup]],
                                  uint threadIdx                                [[thread_index_in_threadgroup]]

) {
    
    // set the initial continue flag
    if (!threadIdx) {
        shouldContinueShared = true;
        clusterCount = 0;
        blockCounter = 0;
    }
    
    // reset the visited flags and the clusters
    for(int i=threadIdx; i<numWindows; i+=numberOfThreads) {
        visited[i] = 0;
        clusters[i] = -1;
        frontier[i] = -1;
    }
    
    threadgroup_barrier( mem_flags::mem_device );
    
    // continue until we have nodes to process
    while (shouldContinueShared) {
        
        // first thread finds new position to process and sets the shared flag
        if (!threadIdx) {
            if (!fetchNextPosition(minNeighbors, neighbors, blockCounter, numWindows, position, visited)) {
                shouldContinueShared = false;
            } else {
                shouldContinueShared = true;
                frontier[position] = position;
                clusterCount++;
            }
        }
        
        // wait for first thread
        threadgroup_barrier( mem_flags::mem_threadgroup );
                
        // if we have blocks to process
        if (!shouldContinueShared) {
            return;
        }
        
        // process
        processWithPosition(threadIdx,position,visited,clusterCount,clusters,continueToProcess,shouldContinueShared,numberOfThreads,frontier,numWindows,neighbors,offset,adjacencies);
        
        // reset the flag
        if (!threadIdx) {
            shouldContinueShared = true;
        }
        
        threadgroup_barrier( mem_flags::mem_threadgroup );
        
    }
    
}
