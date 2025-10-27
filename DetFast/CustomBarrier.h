#pragma once
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

class CustomBarrier {
public:
    CustomBarrier(int nThreads)
        : nThreads(nThreads), Count(0), Phase(0) {
    }

    // Called by each worker after completing its block
    void arrive(int loop) {
        Count.fetch_add(1, std::memory_order_acq_rel);
        while (Phase.load(std::memory_order_acquire) == loop)
            std::this_thread::yield();
    }

    // Called by main thread after all threads reach this phase
    void release() {
        Count.store(0, std::memory_order_release);
        Phase.fetch_add(1, std::memory_order_acq_rel);
    }

    // Wait helpers used by main thread
    void waitAll() const {
        while (Count.load(std::memory_order_acquire) < nThreads)
            std::this_thread::yield();
    }

private:
    const int nThreads;
    std::atomic<int> Count;
    std::atomic<int> Phase;
};

//// ============================================================================
//// Example usage
//// ============================================================================
//
//void Worker(int id, int nLoops, CustomBarrier& barrier) {
//    int loop = 0;
//    while (loop < nLoops) {
//        // ---------- Upper block ----------
//        printf("-%d,%d ", id, loop);
//        barrier.arrive(loop); // wait for all threads to reach barrier
//        //serial block is executed in the main thread before next operation
//
//        // ---------- Lower block ----------
//        printf("*%d,%d ", id, loop);
//        ++loop;
//    }
//}
//
//int main() {
//    clock_t start_application = clock();
//    clock_t current_time = clock();
//
//    const int nThreads = 8;
//    const int nLoops = 12;
//    const int nEpochs = 2;
//
//    for (int epoch = 0; epoch < nEpochs; ++epoch) {
//
//        CustomBarrier barrier(nThreads);
//        std::vector<std::thread> threads;
//        threads.reserve(nThreads);
//        for (int id = 0; id < nThreads; ++id)
//            threads.emplace_back(Worker, id, nLoops, std::ref(barrier));
//
//        for (int loop = 0; loop < nLoops; ++loop) {
//            barrier.waitAll();
//            printf("Serial part, loop %d\n", loop);
//            barrier.release();
//        }
//
//        for (auto& t : threads) t.join();
//        printf("\nDone.\n");
//    }
//
//    current_time = clock();
//    printf("\nElapsed time %2.3f\n", (double)(current_time - start_application) / CLOCKS_PER_SEC);
//}