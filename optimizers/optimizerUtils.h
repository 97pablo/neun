#ifndef OPTIMIZER_UTLIS_H_
#define OPTIMIZER_UTLIS_H_

#include <vector>
#include <numeric>
#include <iostream>

std::vector<int> detect_peaks(std::vector<double> &voltages, double tolerance)
{

    // variables for finding peak prominence
    double runningMin = voltages[0];
    std::vector<double> leftMins;
    std::vector<double> rightMins;
    double globalMin = voltages[0];
    double globalMax = voltages[0];

    // finds local maximums
    std::vector<int> peaks;
    for (size_t i = 1; i < voltages.size() - 1; i++)
    {
        double prev = voltages[i - 1];
        double curr = voltages[i];
        double next = voltages[i + 1];

        // find global range
        globalMax = std::max(globalMax, curr);
        globalMin = std::min(globalMin, curr);

        runningMin = std::min(runningMin, curr);
        if (curr > prev && curr > next)
        {
            leftMins.push_back(runningMin);
            if (!peaks.empty())
            {
                rightMins.push_back(runningMin);
            }

            peaks.push_back(i);
            runningMin = curr;
        }
    }
    rightMins.push_back(runningMin);

    const double global_range = globalMax - globalMin;
    const double minProminence = tolerance * global_range;

    // finds peaks that have a prominence above the threshold
    std::vector<int> refinedPeaks;
    for (size_t i = 0; i < peaks.size(); i++)
    {
        double curr = voltages[peaks[i]];
        double prominence = curr - std::max(leftMins[i], rightMins[i]);

        if (prominence > minProminence)
        {
            refinedPeaks.push_back(peaks[i]);
        }
    }

    return refinedPeaks;
}

double gaussian_reward(double expected, double obtained, double sigma)
{
    double z = (expected - obtained) / sigma;
    return std::exp(-0.5 * z * z);
}

double smooth_square_error(double expected, double obtained)
{
    double difference = (expected - obtained);
    double error = difference * difference;
    return 1 / (1 + error);
}

#endif // OPTIMIZER_UTLIS_H_
