#ifndef OPTIMIZER_UTLIS_H_
#define OPTIMIZER_UTLIS_H_

#include <vector>
#include <numeric>

std::vector<int> detect_peaks(std::vector<double> &voltages, double tolerance)
{

    // variables for finding peak prominence
    double runningMin = voltages[0];
    std::vector<double> leftMins;
    std::vector<double> rightMins;

    // finds local maximums
    std::vector<int> peaks;
    for (size_t i = 1; i < voltages.size() - 1; i++)
    {
        double prev = voltages[i - 1];
        double curr = voltages[i];
        double next = voltages[i + 1];

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

    // finds peaks that have a prominence above the threshold
    std::vector<int> refinedPeaks;

    for (size_t i = 0; i < peaks.size(); i++)
    {
        double curr = voltages[peaks[i]];
        double prominence = curr - std::max(leftMins[i], rightMins[i]);

        if (prominence > tolerance * std::fabs(curr))
        {
            refinedPeaks.push_back(peaks[i]);
        }
    }

    return refinedPeaks;
}

#endif // OPTIMIZER_UTLIS_H_
