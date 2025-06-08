#ifndef AMPLITUDE_OBJECTIVE_H_
#define AMPLITUDE_OBJECTIVE_H_

#include "DifferentialNeuronWrapper.h"
#include "optimizerUtils.h"
#include <vector>
#include <iostream>
#include <numeric>

template <typename TNetwork>
class AmplitudeObjective
{
public:
    typedef TNetwork Network;
    typedef typename Network::Neuron Neuron;

    enum parameter
    {
        step,
        time,
        peak_tolerance,
        amplitude,
        amp_tolerance,
        n_peaks,
        n_parameters,
    };

    struct ConstructorArgs
    {
        double params[n_parameters];
    };

    AmplitudeObjective(ConstructorArgs &args)
    {
        std::copy(args.params, args.params + n_parameters, this->params);
    }

    double get_quality_score(std::vector<int> &peaks, std::vector<double> &voltages)
    {
        if (peaks.empty())
        {
            return 0.0;
        }

        // uses gaussian reward to ensure a a smooth score falloff
        const double amp = params[parameter::amplitude];
        const double tol = params[parameter::amp_tolerance];
        const double sigma = amp * tol;
        double sum = 0.0;
        for (int p_idx : peaks)
        {
            double v = voltages[p_idx];
            double z = (v - amp) / sigma;
            sum += std::exp(-0.5 * z * z);
        }

        double qualityScore = sum / peaks.size();

        return qualityScore;
    }

    double get_count_score(std::vector<int> peaks)
    {
        double count_score = static_cast<double>(peaks.size()) /
                             static_cast<double>(params[n_peaks]);
        return std::min(1.0, count_score);
    }

    double evaluate(Network n)
    {
        const double step = params[parameter::step];
        const double time = params[parameter::time];
        const int nSamples = static_cast<int>(time / step);
        const int nNeurons = n.get_neurons().size();

        // simulates neuron
        std::vector<std::vector<double>> voltages(nNeurons, std::vector<double>(nSamples, 0.0));
        for (int i = 0; i < nSamples; i++)
        {
            n.step(step);
            for (int j = 0; j < nNeurons; j++)
            {
                double v = n.get_neuron(j)->get(Neuron::v);
                voltages[j][i] = v;
            }
        }

        // std::cout << "Neurons: " << std::endl;
        //  calculate scores for each neuron
        double fitness = 0;
        for (int i = 0; i < nNeurons; i++)
        {
            std::vector<int> peaks = detect_peaks(voltages[i], params[parameter::peak_tolerance]);
            double quality = get_quality_score(peaks, voltages[i]);
            double count = get_count_score(peaks);

            /*for (auto p : peaks)
            {
                std::cout << p << " ";
            }
            std::cout << std::endl;*/

            // std::cout << "quality: " << quality << " count: " << count << std::endl;
            fitness += quality + count;
        }

        fitness /= nNeurons * 2;
        return fitness;
    }

private:
    double params[parameter::n_parameters];
};

#endif // AMPLITUDE_OBJECTIVE_H_
