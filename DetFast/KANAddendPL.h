#pragma once
#include "UrysohnPL.h"
#include "UnivariatePL.h"
#include <memory>

class KANAddendPL
{
public:
    KANAddendPL(std::unique_ptr<double[]>& xmin, std::unique_ptr<double[]>& xmax,
        double targetMin, double targetMax,
        int inner, int outer, int number_of_inputs) {
        _lastInnerValue = 0.0;

        std::unique_ptr<int[]> interior_structure = std::make_unique<int[]>(number_of_inputs);
        for (int i = 0; i < number_of_inputs; i++)
        {
            interior_structure[i] = static_cast<int>(inner);
        }
        _u = std::make_unique<UrysohnPL>(xmin, xmax, targetMin, targetMax, interior_structure, number_of_inputs);
        _univariate = std::make_unique<UnivariatePL>(targetMin, targetMax, targetMin, targetMax, outer);
    }
    KANAddendPL(const KANAddendPL& addend) {
        _lastInnerValue = addend._lastInnerValue;
        _univariate = std::make_unique<UnivariatePL>(*addend._univariate);
        _u = std::make_unique<UrysohnPL>(*addend._u);
    }
    void UpdateUsingMemory(double diff) {
        double derrivative = _univariate->GetDerivative(_lastInnerValue);
        _u->UpdateUsingMemory(diff * derrivative);
        _univariate->UpdateUsingMemory(diff);
    }
    void UpdateUsingInput(std::unique_ptr<double[]>& input, double diff) {
        double value = _u->GetValueUsingInput(input);
        double derrivative = _univariate->GetDerivative(value);
        _u->UpdateUsingInput(diff * derrivative, input);
        _univariate->UpdateUsingInput(value, diff);
    }
    double ComputeUsingInput(std::unique_ptr<double[]>& input, bool noUpdate = false) {
        _lastInnerValue = _u->GetValueUsingInput(input, noUpdate);
        return _univariate->GetFunctionUsingInput(_lastInnerValue, noUpdate);
    }
    void IncrementInner() {
        _u->IncrementInner();
    }
    void IncrementOuter() {
        _univariate->IncrementPoints();
    }
    int HowManyOuter() {
        return _univariate->HowManyPoints();
    }
    int HowManyInner() {
        return _u->_univariateList[0]->HowManyPoints();
    }
    std::unique_ptr<double[]> GetAllOuterPoints() {
        return _univariate->GetAllPoints();
    }
    std::unique_ptr<UrysohnPL> _u;
private:
    double _lastInnerValue;
    std::unique_ptr<UnivariatePL> _univariate;
};

