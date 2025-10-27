//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

// License
// If the end user somehow manages to make billions of US dollars using this code,
// and happens to meet the developer begging for change outside a McDonald's,
// they are under no obligation to buy the developer a sandwich.

// Symmetry Clause
// Likewise, if the developer becomes rich and famous by publishing this code,
// and meets an unfortunate end user who went bankrupt using it,
// the developer is also under no obligation to buy the end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://link.springer.com/article/10.1007/s10994-025-06800-6

//Website:
//http://OpenKAN.org

//This is Parallel Newton-Kaczmarz method for Kolmogorov-Arnold networks. The features are random matrices,
//targets are their determinants. Accuracy metric is Pearson correlation coefficient.
//Typical processing time for 16 cores laptop. 
//1. 4 by 4, 16 features, 100'000 records, termination at Pearson > 0.97, near 1 second.
//2. 5 by 5, 25 features, 10'000'000 records, termination at Pearson > 0.95, from 100 to 250 seconds.

//Makefile
//# Compiler and flags
//CXX = g++
//CXXFLAGS = -O2 -std=c++17 -Wall
//
//# Target name (final executable)
//TARGET = DetFast
//
//# Source files
//SRCS = DetFast.cpp
//
//# Object files
//OBJS = $(SRCS:.cpp=.o)
//
//# Default rule
//$(TARGET): $(OBJS)
//	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)
//
//# Compile .cpp to .o
//%.o: %.cpp
//	$(CXX) $(CXXFLAGS) -c $< -o $@
//
//# Clean rule
//clean:
//	rm -f $(OBJS) $(TARGET)

#include <iostream>
#include <random>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include "KANAddendPL.h"
#include "Helper.h"
#include "CustomBarrier.h"

const int THREADS = 16;  //for fast execution must match the number of cores. 
std::atomic<double> RESIDUAL{ 0 };
std::vector<std::atomic<double>> PARTIAL_MODELS(THREADS);

void Worker(int id, const std::vector<KANAddendPL*>& addends, const std::unique_ptr<std::unique_ptr<double[]>[]>& features,
    CustomBarrier& barrier, int nTrainingRecords, double mu) {
    int local_round = 0;

    while (local_round < nTrainingRecords) {
        //compute individual contributions of all addends
        double model = 0.0;
        for (auto& addend : addends) {
            model += addend->ComputeUsingInput(features[local_round]);
        }
        PARTIAL_MODELS[id].store(model, std::memory_order_release);

        //sync with main and execute serial block
        barrier.arrive(local_round);

        //continue after serial
        double residual = RESIDUAL.load(std::memory_order_acquire);
        residual *= mu;

        //update all addends
        for (auto& addend : addends) {
            addend->UpdateUsingMemory(residual);
        }

        //proceed to next record
        ++local_round;
    }
}

void Training() {

    //This is for 5 by 5 matrices
    //Dataset
    int nTrainingRecords = 10'000'000;
    int nValidationRecords = 2'000'000;
    int nMatrixSize = 5;
    double min = 0.0;
    double max = 10.0;

    //Hyperparameters
    int nAddends = 208;  //must be divisible by number of threads
    double mu = 0.1;
    double termination = 0.91;

    ////This is for 4 by 4 matrices
    ////Dataset
    //int nTrainingRecords = 100'000;
    //int nValidationRecords = 20'000;
    //int nMatrixSize = 4;
    //double min = 0.0;
    //double max = 10.0;

    ////Hyperparameters
    //int nAddends = 64;  //must be divisible by number of threads
    //double mu = 0.2;
    //double termination = 0.97;

    mu /= nAddends;
    int nFeatures = nMatrixSize * nMatrixSize;
    auto inputs_training = Helper::GenerateInput(nTrainingRecords, nFeatures, min, max);
    auto inputs_validation = Helper::GenerateInput(nValidationRecords, nFeatures, min, max);
    auto target_training = Helper::ComputeDeterminantTarget(inputs_training, nMatrixSize, nTrainingRecords);
    auto target_validation = Helper::ComputeDeterminantTarget(inputs_validation, nMatrixSize, nValidationRecords);

    printf("Dataset is generated\n");

    auto tstart = std::chrono::high_resolution_clock::now();

    //find initialization parameters
    std::vector<double> argmin;
    std::vector<double> argmax;
    double targetMin;
    double targetMax;
    Helper::FindMinMax(argmin, argmax, targetMin, targetMax, inputs_training, target_training,
        nTrainingRecords, nFeatures);

    auto xmin = std::make_unique<double[]>(argmin.size());
    auto xmax = std::make_unique<double[]>(argmax.size());
    for (size_t i = 0; i < argmin.size(); ++i) {
        xmin[i] = argmin[i];
        xmax[i] = argmax[i];
    }

    //initialize objects
    std::vector<std::unique_ptr<KANAddendPL>> addends;
    for (int i = 0; i < nAddends; ++i) {
        addends.push_back(std::make_unique<KANAddendPL>(xmin, xmax, targetMin / nAddends, targetMax / nAddends, 5, 22, nFeatures));
    }

    //first concurrent stage is pretraining, random pairs are trained individually, it is only good for very approximate model
    printf("Pretraining started ...\n");
    auto pairs = Helper::MakePairs(nAddends);

    //pairs of addends
    std::vector<std::thread> pre_threads;
    for (auto& p : pairs) {
        int first = p.first;
        int second = p.second;
        pre_threads.emplace_back([first, second, &inputs_training, &target_training, mu, nTrainingRecords, &addends]() {
            for (int epoch = 0; epoch < 2; ++epoch) {
                for (int i = 0; i < nTrainingRecords; ++i) {
                    double model = addends[first]->ComputeUsingInput(inputs_training[i]);
                    model += addends[second]->ComputeUsingInput(inputs_training[i]);
                    double residual = target_training[i] - model;
                    residual *= mu;
                    addends[first]->UpdateUsingMemory(residual);
                    addends[second]->UpdateUsingMemory(residual);
                }
            }
            });
    }

    for (auto& t : pre_threads) {
        t.join();
    }
    auto tend_pre = std::chrono::high_resolution_clock::now();
    auto ms_pre = std::chrono::duration_cast<std::chrono::milliseconds>(tend_pre - tstart);
    printf("Pretraining ended, time %2.1f\n", static_cast<double>(ms_pre.count()));

    //second concurrent stage
    for (int epoch = 0; epoch < 16; ++epoch) {  //it does not need 16 epochs, it terminates earlier
        //start all threads
        CustomBarrier barrier(THREADS);
        std::vector<std::thread> threads;
        threads.reserve(THREADS);
        int block_size = nAddends / THREADS;
        for (int i = 0; i < THREADS; ++i) {
            std::vector<KANAddendPL*> thread_addends;
            for (int j = i * block_size; j < (i + 1) * block_size; ++j) {
                thread_addends.push_back(addends[j].get());
            }
            threads.emplace_back(Worker, i, std::move(thread_addends), std::ref(inputs_training), 
                std::ref(barrier), nTrainingRecords, mu);
        }

        //navigating via dataset with synchronization
        for (int record = 0; record < nTrainingRecords; ++record) {
            barrier.waitAll(); //this waits until all contibutions to model

            //this is serial block
            double model = 0.0;
            for (int t = 0; t < THREADS; ++t)
                model += PARTIAL_MODELS[t].load(std::memory_order_acquire);
            double residual = target_training[record] - model;
            RESIDUAL.store(residual, std::memory_order_release);
            //end of serial block

            //this makes to proceed to the next record
            barrier.release();
        }

        //terminate threads
        for (auto& t : threads) {
            t.join();
        }

        //validation on idependent data, it is sequential, but assumed that validation set is short
        auto model_validation = std::make_unique<double[]>(nValidationRecords);
        double error = 0.0;
        for (int i = 0; i < nValidationRecords; ++i) {
            double vmodel = 0.0;
            for (int j = 0; j < nAddends; ++j) {
                vmodel += addends[j]->ComputeUsingInput(inputs_validation[i], true);
            }
            model_validation[i] = vmodel;
            error += (target_validation[i] - vmodel) * (target_validation[i] - vmodel);
        }
        error /= nValidationRecords;
        error = sqrt(error) / (targetMax - targetMin);
        double validation_pearson = Helper::Pearson(model_validation, target_validation, nValidationRecords);
        auto tend = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);
        printf("Pearson for validation %6.3f, RRMSE %6.3f, time in ms %7.1f\n", validation_pearson, error, static_cast<double>(ms.count()));
        if (validation_pearson > termination) break;
    }
}

int main() {
    printf("Processing started ...\n");
    Training();
}


