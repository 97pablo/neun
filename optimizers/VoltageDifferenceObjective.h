
#ifndef VOLTAGE_DIFFERENCE_OBJECTIVE_H_
#define VOLTAGE_DIFFERENCE_OBJECTIVE_H_

#include "DifferentialNeuronWrapper.h"
#include <vector>
#include <iostream>

template <typename TNeuron>
class VoltageDifferenceObjective
{
public:
    typedef TNeuron Neuron;
    enum parameter
    {
        step,
        time,
        n_parameters,
    };

    struct ConstructorArgs
    {
        double params[n_parameters];
    };

    std::vector<double> generateVoltages()
    {
        std::vector<double> out;
        // Struct to initialize neuron model parameters
        typename Neuron::ConstructorArgs args;

        // Set the parameter values
        args.params[Neuron::cm] = 1 * 7.854e-3;
        args.params[Neuron::vna] = 50;
        args.params[Neuron::vk] = -77;
        args.params[Neuron::vl] = -54.387;
        args.params[Neuron::gna] = 120 * 7.854e-3;
        args.params[Neuron::gk] = 36 * 7.854e-3;
        args.params[Neuron::gl] = 0.3 * 7.854e-3;

        // Initialize a new neuron model
        Neuron n(args);

        // You can also initialize the variables of the neuron model to a given value
        n.set(Neuron::v, -80);
        n.set(Neuron::m, 0.1);
        n.set(Neuron::n, 0.7);
        n.set(Neuron::h, 0.01);

        // Set the integration step
        const double step = this->params[parameter::step];
        const double time = this->params[parameter::time];
        const std::size_t nSamples = static_cast<std::size_t>(time / step);

        for (int i = 0; i < nSamples; i++)
        {
            double voltage = n.get(Neuron::v);
            out.push_back(voltage);

            n.step(step);
        }

        return out;
    }

    VoltageDifferenceObjective(ConstructorArgs &args)
    {
        std::copy(args.params, args.params + n_parameters, this->params);

        this->targetVoltages = this->generateVoltages();
    }

    double evaluate(Neuron n)
    {
        const double step = this->params[parameter::step];
        const double time = this->params[parameter::time];
        const std::size_t nSamples = static_cast<std::size_t>(time / step);

        double sqDiff = 0;
        for (int i = 0; i < nSamples; i++)
        {
            // calculate voltage difference for each point and aggregate in difference
            double voltage = n.get(Neuron::v);
            double difference = std::abs(voltage - this->targetVoltages[i]);
            sqDiff += difference * difference;

            n.step(step);
        }

        // this line is needed to minimize difference, instead of maximizing it
        double fitness = 1.0 / (1.0 + sqDiff);
        return std::max(1e-9, fitness);
    }

protected:
    double params[n_parameters];

private:
    std::vector<double> targetVoltages;
};

#endif // VOLTAGE_DIFFERENCE_OBJECTIVE_H_
