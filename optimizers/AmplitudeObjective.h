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
        interval_pre,
        interval_post,
        int_tolerance,
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

    double get_amplitude_score(std::vector<int> &peaks, std::vector<double> &voltages)
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
            sum += gaussian_reward(amp, voltages[p_idx], sigma);
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

    /*
    double get_interval_score(std::vector<int> peaks_n1, std::vector<int> peaks_n2)
    {
        const double interval = params[parameter::interval];
        const double step = params[parameter::step];
        const double tol = params[parameter::int_tolerance];
        const double sigma = interval * tol;

        if (peaks_n1.empty() || peaks_n2.empty())
        {
            return 0.0;
        }

        double sum = 0.0;
        int usable_peaks = 0;
        size_t j;

        // score left peaks of the first neuron
        j = 0;
        for (int peak : peaks_n1)
        {
            // finds the closest peak from the left
            while (j + 1 < peaks_n2.size() && peaks_n2[j + 1] <= peak)
                ++j;
            // if no peak is found to the left, skip this peak
            if (peaks_n2[j] > peak)
                continue;

            usable_peaks++;
            // find the time interval between current and left peak
            double diff = peak * step - peaks_n2[j] * step;
            sum += gaussian_reward(interval, diff, sigma);
        }

        // score left peaks of the second neuron
        j = 0;
        for (int peak : peaks_n2)
        {
            // finds the closest peak from the left
            while (j + 1 < peaks_n1.size() && peaks_n1[j + 1] <= peak)
                ++j;
            // if no peak is found to the left, skip this peak
            if (peaks_n1[j] > peak)
                continue;

            // find the time interval between current and left peak
            usable_peaks++;
            double diff = peak * step - peaks_n1[j] * step;
            sum += gaussian_reward(interval, diff, sigma);
        }

        // return the average reward
        return sum / usable_peaks;
    }
    */

    double get_interval_score(std::vector<int> peaks_n1, std::vector<int> peaks_n2)
    {
        const double iprev = params[parameter::interval_pre];
        const double ipost = params[parameter::interval_post];

        const double step = params[parameter::step];
        const double tol = params[parameter::int_tolerance];

        if (peaks_n1.empty() || peaks_n2.empty())
        {
            return 0.0;
        }

        size_t j = 0;
        double score = 0.0;
        int unused_peaks;

        while (j < peaks_n2.size() && peaks_n2[j] < peaks_n1[0])
            j++;
        unused_peaks = j;

        for (size_t i = 0; i < peaks_n1.size() - 1; i++)
        {
            int left_peak = peaks_n1[i];
            int right_peak = peaks_n1[i + 1];

            while (j < peaks_n2.size() && peaks_n2[j] < right_peak)
            {
                int peak = peaks_n2[j];
                double diff_left = (peak - left_peak) * step;
                double diff_right = (right_peak - peak) * step;

                double score_left = gaussian_reward(iprev, diff_left, iprev * tol);
                double score_right = gaussian_reward(ipost, diff_right, ipost * tol);

                score += (score_left + score_right) / 2;
                j++;
            }
        }

        while (j < peaks_n2.size())
        {
            double diff = (peaks_n2[j] - peaks_n1[peaks_n1.size() - 1]) * step;
            score += gaussian_reward(ipost, diff, ipost * tol);
            j++;
        }

        return score / (peaks_n2.size() - unused_peaks);
    }

    /*
    double get_interval_score(std::vector<int> peaks_n1, std::vector<int> peaks_n2)
    {
        const double interval = params[parameter::interval];
        const double step = params[parameter::step];
        const double tol = params[parameter::int_tolerance];
        const double sigma = interval * tol;

        if (peaks_n1.empty() || peaks_n2.empty())
        {
            return 0.0;
        }

        size_t j = 0;
        double score = 0.0;
        int unused_peaks;

        while (j < peaks_n2.size() && peaks_n2[j] < peaks_n1[0])
            j++;

        unused_peaks = j;

        for (size_t i = 0; i < peaks_n1.size() - 1; i++)
        {
            int peak_n1 = peaks_n1[i];

            while (j < peaks_n2.size() && peaks_n2[j] < peaks_n1[i + 1])
            {
                int peak_n2 = peaks_n2[j];
                double diff = (peak_n2 - peak_n1) * step;

                // std::cout << diff << std::endl;
                score += gaussian_reward(interval, diff, sigma);
                j++;
            }
        }

        int last_peak = peaks_n1[peaks_n1.size() - 1];
        while (j < peaks_n2.size())
        {
            int peak_n2 = peaks_n2[j];
            double diff = (peak_n2 - last_peak) * step;

            score += gaussian_reward(interval, diff, sigma);
            j++;
        }

        return score / (peaks_n2.size() - unused_peaks);
    }
        */

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
        double count_fitness = 0;
        double amplitude_fitness = 0;
        double interval_fitness = 0;

        std::vector<std::vector<int>> peaks(nNeurons);
        for (int i = 0; i < nNeurons; i++)
        {
            peaks[i] = detect_peaks(voltages[i], params[parameter::peak_tolerance]);
            amplitude_fitness += get_amplitude_score(peaks[i], voltages[i]);
            count_fitness += get_count_score(peaks[i]);

            /*
            for (auto p : peaks[i])
            {
                std::cout << p * step << " ";
            }
            std::cout << std::endl;
            */

            // std::cout << "quality: " << quality << " count: " << count << std::endl;
        }
        amplitude_fitness /= nNeurons;
        count_fitness /= nNeurons;
        interval_fitness = get_interval_score(peaks[0], peaks[1]);

        std::cout << "amp " << amplitude_fitness << " count " << count_fitness << " int " << interval_fitness << std::endl;

        return (amplitude_fitness + count_fitness + interval_fitness) / 3;
    }

private:
    double params[parameter::n_parameters];
};

#endif // AMPLITUDE_OBJECTIVE_H_
