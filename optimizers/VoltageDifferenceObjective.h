
#ifndef VOLTAGE_DIFFERENCE_OBJECTIVE_H_
#define VOLTAGE_DIFFERENCE_OBJECTIVE_H_

#include "DifferentialNeuronWrapper.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <limits>
#include <optimizerUtils.h>

#define TOLERANCE 0.3

template <typename TNetwork>
class VoltageDifferenceObjective
{
public:
    typedef TNetwork Network;
    typedef typename Network::Neuron Neuron;
    typedef typename Neuron::variable NeuronVariable;

    enum parameter
    {
        step,
        time,
        tolerance,
        input,
        n_parameters,
    };

    struct ConstructorArgs
    {
        double params[n_parameters];
    };

    std::vector<double> generateVoltages(Neuron &n)
    {
        std::vector<double> out;

        // Set the integration step
        const double step = this->params[parameter::step];
        const double time = this->params[parameter::time];
        const std::size_t nSamples = static_cast<std::size_t>(time / step);

        for (size_t i = 0; i < nSamples; i++)
        {
            double voltage = n.get(graphedVariable);
            out.push_back(voltage);

            n.add_synaptic_input(this->params[input]);

            n.step(step);
        }

        return out;
    }

    VoltageDifferenceObjective(ConstructorArgs &args, Neuron &n, NeuronVariable graphedVariable)
    {
        std::copy(args.params, args.params + n_parameters, this->params);
        this->graphedVariable = graphedVariable;
        this->targetVoltages = this->generateVoltages(n);
    }

    double get_error(std::vector<double> &voltages)
    {
        const double step = this->params[parameter::step];
        const double time = this->params[parameter::time];
        const int nSamples = static_cast<int>(time / step);

        double score = 0;
        for (int i = 0; i < nSamples; i++)
        {
            double diff = targetVoltages[i] - voltages[i];
            score += diff * diff;
        }
        return score / nSamples;
    }
    double evaluate(Network &net)
    {
        const double step = this->params[parameter::step];
        const double time = this->params[parameter::time];
        const int nSamples = static_cast<int>(time / step);

        std::vector<double> voltages;
        for (int i = 0; i < nSamples; i++)
        {
            // std::cout << targetVoltages[i] << " " << voltages[i] << std::endl;
            double voltage = net.get_neuron(0)->get(graphedVariable);
            voltages.push_back(voltage);
            net.step(step);
        }

        return std::max(1e-9, 1.0 / (1.0 + get_error(voltages)));
    }

    void save_voltages(std::ofstream &file)
    {
        for (size_t i = 0; i < targetVoltages.size(); i++)
        {
            file << i * this->params[step] << " " << targetVoltages[i] << std::endl;
        }
    }

protected:
    double params[n_parameters];

private:
    std::vector<double> targetVoltages;

    NeuronVariable graphedVariable;
};

#endif // VOLTAGE_DIFFERENCE_OBJECTIVE_H_
