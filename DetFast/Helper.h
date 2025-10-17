#pragma once
#include <memory>
#include <vector>
#include <chrono>
#include <random>
#include <cfloat>
#include <algorithm>

class Helper
{
public:
    static double Pearson(const std::unique_ptr<double[]>& x, const std::unique_ptr<double[]>& y, int len) {
        double xy = 0.0;
        double x2 = 0.0;
        double y2 = 0.0;
        for (int i = 0; i < len; ++i)
        {
            xy += x[i] * y[i];
            x2 += x[i] * x[i];
            y2 += y[i] * y[i];
        }
        xy /= (double)(len);
        x2 /= (double)(len);
        y2 /= (double)(len);
        double xav = 0.0;
        for (int i = 0; i < len; ++i)
        {
            xav += x[i];
        }
        xav /= len;
        double yav = 0.0;
        for (int i = 0; i < len; ++i)
        {
            yav += y[i];
        }
        yav /= len;
        double ro = xy - xav * yav;
        ro /= sqrt(x2 - xav * xav);
        ro /= sqrt(y2 - yav * yav);
        return ro;
    }
    //this gives identical result
    static double Pearson2(const std::unique_ptr<double[]>& x, const std::unique_ptr<double[]>& y, int len) {
        double xmean = 0.0;
        double ymean = 0.0;
        for (int i = 0; i < len; ++i) {
            xmean += x[i];
            ymean += y[i];
        }
        xmean /= len;
        ymean /= len;

        double covariance = 0.0;
        for (int i = 0; i < len; ++i) {
            covariance += (x[i] - xmean) * (y[i] - ymean);
        }

        double stdX = 0.0;
        double stdY = 0.0;
        for (int i = 0; i < len; ++i) {
            stdX += (x[i] - xmean) * (x[i] - xmean);
            stdY += (y[i] - ymean) * (y[i] - ymean);
        }
        stdX = sqrt(stdX);
        stdY = sqrt(stdY);
        return covariance / stdX / stdY;
    }
    static void ShowMatrix(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                printf("%5.3f ", matrix[i][j]);
            }
            printf("\n");
        }
    }
    static void ShowVector(std::unique_ptr<double[]>& ptr, int N) {
        int cnt = 0;
        for (int i = 0; i < N; ++i) {
            printf("%5.2f ", ptr[i]);
            if (++cnt >= 10) {
                printf("\n");
                cnt = 0;
            }
        }
    }
    static void SwapRows(std::unique_ptr<double[]>& row1, std::unique_ptr<double[]>& row2, int cols) {
        auto ptr = std::make_unique<double[]>(cols);
        for (int i = 0; i < cols; ++i) {
            ptr[i] = row1[i];
        }
        for (int i = 0; i < cols; ++i) {
            row1[i] = row2[i];
        }
        for (int i = 0; i < cols; ++i) {
            row2[i] = ptr[i];
        }
    }
    static void SwapScalars(double& x1, double& x2) {
        double buff = x1;
        x1 = x2;
        x2 = buff;
    }
    static void Shuffle(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, std::unique_ptr<double[]>& vector, int rows, int cols) {
        for (int i = 0; i < 2 * rows; ++i) {
            int n1 = rand() % rows;
            int n2 = rand() % rows;
            SwapRows(matrix[n1], matrix[n2], cols);
            SwapScalars(vector[n1], vector[n2]);
        }
    }
    static void FindMinMax(std::vector<double>& xmin, std::vector<double>& xmax,
        double& targetMin, double& targetMax,
        std::unique_ptr<std::unique_ptr<double[]>[]>& matrix,
        std::unique_ptr<double[]>& target, int nRows, int nCols) {

        for (int i = 0; i < nCols; ++i) {
            xmin.push_back(DBL_MAX);
            xmax.push_back(-DBL_MAX);
        }
        for (int i = 0; i < nRows; ++i) {
            for (int j = 0; j < nCols; ++j) {
                if (matrix[i][j] < xmin[j]) xmin[j] = static_cast<double>(matrix[i][j]);
                if (matrix[i][j] > xmax[j]) xmax[j] = static_cast<double>(matrix[i][j]);
            }
        }
        targetMin = DBL_MAX;
        targetMax = -DBL_MAX;
        for (int j = 0; j < nRows; ++j) {
            if (target[j] < targetMin) targetMin = target[j];
            if (target[j] > targetMax) targetMax = target[j];
        }
    }
    static void IndividualLimits2Sum(double xmin, double xmax, int N, double& sumMin, double& sumMax) {
        double sumMean = N * (xmin + xmax) / 2.0;
        double sumVariance = N * (xmin - xmax) * (xmin - xmax) / 12.0;
        double STD = sqrt(sumVariance);
        sumMin = sumMean - 1.96 * STD;
        sumMax = sumMean + 1.96 * STD;
    }
    static void Sum2IndividualLimits(double sumMin, double sumMax, int N, double& xmin, double& xmax) {
        double varCoeff = N / 12.0;
        double STDCoeff = sqrt(varCoeff);
        double confCoeff = STDCoeff * 1.96;
        double xmin_plus_xmax = (sumMin + sumMax) / N;
        double xmax_minus_xmin = (sumMax - sumMin) / 2 / confCoeff;
        xmax = (xmin_plus_xmax + xmax_minus_xmin) / 2.0;
        xmin = xmin_plus_xmax - xmax;
    }

    //determinants
    //static std::unique_ptr<std::unique_ptr<double[]>[]> GenerateInput(int nRecords, int nFeatures, double min, double max) {
    //    auto x = std::make_unique<std::unique_ptr<double[]>[]>(nRecords);
    //    for (int i = 0; i < nRecords; ++i) {
    //        x[i] = std::make_unique<double[]>(nFeatures);
    //        for (int j = 0; j < nFeatures; ++j) {
    //            x[i][j] = static_cast<double>((rand() % 10000) / 10000.0);
    //            x[i][j] *= (max - min);
    //            x[i][j] += min;
    //        }
    //    }
    //    return x;
    //}

    static std::unique_ptr<std::unique_ptr<double[]>[]> GenerateInput(int nRecords, int nFeatures, double min, double max) {
        constexpr int nThreads = 8; // fixed inside function
        auto x = std::make_unique<std::unique_ptr<double[]>[]>(nRecords);

        // allocate rows first
        for (int i = 0; i < nRecords; ++i) {
            x[i] = std::make_unique<double[]>(nFeatures);
        }

        // Prepare different seeds (generated on the parent thread)
        std::random_device rd;
        std::vector<std::uint64_t> seeds(nThreads);
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int t = 0; t < nThreads; ++t) {
            // mix rd() with time and thread index to avoid identical seeds
            seeds[t] = (static_cast<std::uint64_t>(rd()) << 32) ^ rd() ^ (now + (std::uint64_t)t * 0x9e3779b97f4a7c15ULL);
        }

        // launch anonymous worker threads (capturing seed by value)
        std::vector<std::thread> threads;
        threads.reserve(nThreads);
        for (int t = 0; t < nThreads; ++t) {
            threads.emplace_back(
                [t, nThreads, nRecords, nFeatures, min, max, &x, seed = seeds[t]]()
                {
                    std::mt19937_64 rng(seed);
                    std::uniform_real_distribution<double> dist(min, max);
                    for (int i = t; i < nRecords; i += nThreads) {
                        double* row = x[i].get();
                        for (int j = 0; j < nFeatures; ++j) {
                            row[j] = dist(rng);
                        }
                    }
                }
            );
        }

        for (auto& th : threads) th.join();
        return x;
    }

    static double determinant(const std::vector<std::vector<double>>& matrix) {
        size_t n = matrix.size();
        if (n == 1) {
            return matrix[0][0];
        }
        if (n == 2) {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }
        double det = 0.0;
        for (size_t col = 0; col < n; ++col) {
            std::vector<std::vector<double>> subMatrix(n - 1, std::vector<double>(n - 1));
            for (size_t i = 1; i < n; ++i) {
                int subCol = 0;
                for (size_t j = 0; j < n; ++j) {
                    if (j == col) continue;
                    subMatrix[i - 1][subCol++] = matrix[i][j];
                }
            }
            det += (col % 2 == 0 ? 1 : -1) * matrix[0][col] * determinant(subMatrix);
        }
        return det;
    }

    static double ComputeDeterminant(std::unique_ptr<double[]>& input, int N) {
        std::vector<std::vector<double>> matrix(N, std::vector<double>(N, 0.0));
        int cnt = 0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                matrix[i][j] = input[cnt++];
            }
        }
        return determinant(matrix);
    }

    //static std::unique_ptr<double[]> ComputeDeterminantTarget(const std::unique_ptr<std::unique_ptr<double[]>[]>& x, int nMatrixSize, int nRecords) {
    //    auto target = std::make_unique<double[]>(nRecords);
    //    int counter = 0;
    //    while (true) {
    //        target[counter] = ComputeDeterminant(x[counter], nMatrixSize);
    //        if (++counter >= nRecords) break;
    //    }
    //    return target;
    //}

    static std::unique_ptr<double[]> ComputeDeterminantTarget(
        const std::unique_ptr<std::unique_ptr<double[]>[]>& x,
        int nMatrixSize,
        int nRecords)
    {
        constexpr int nThreads = 8;  // fixed inside
        auto target = std::make_unique<double[]>(nRecords);

        // Worker: each thread handles every nThreads-th record
        auto worker = [&](int tid) {
            for (int i = tid; i < nRecords; i += nThreads) {
                target[i] = ComputeDeterminant(x[i], nMatrixSize);
            }
            };

        // Launch threads
        std::vector<std::thread> threads;
        threads.reserve(nThreads);
        for (int t = 0; t < nThreads; ++t) {
            threads.emplace_back(worker, t);
        }

        for (auto& th : threads) th.join();

        return target;
    }
    //end determinants

    static std::vector<std::pair<int, int>> MakePairs(int N) {
        if (0 != N % 2) {
            printf("Uneven vector size in MakePairs\n");
            exit(0);
        }
        std::vector<int> data;
        for (int i = 0; i < N; ++i) {
            data.push_back(i);
        }
        unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(data.begin(), data.end(), std::default_random_engine(seed));

        std::vector<std::pair<int, int>> result;
        for (int i = 0; i < N / 2; ++i) {
            std::pair<int, int> p = { data[i * 2], data[i * 2 + 1] };
            result.push_back(p);
        }
        return result;
    }
};

