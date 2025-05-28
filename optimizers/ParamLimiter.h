#ifndef PARAM_LIMITER_H_
#define PARAM_LIMITER_H_

template <typename Neuron>
class ParamLimiter
{
public:
    typedef typename Neuron::variable variable;
    typedef typename Neuron::parameter parameter;

    ParamLimiter()
    {
        for (int i = 0; i < parameter::n_parameters; ++i)
        {
            paramMins[i] = -1.0e3f;
            paramMaxs[i] = 1.0e3f;
        }
        for (int i = 0; i < variable::n_variables; ++i)
        {
            varMins[i] = -100.0f;
            varMaxs[i] = 100.0f;
        }
    }

    float getMin(parameter param) const
    {
        int i = static_cast<int>(param);
        return paramMins[i];
    }

    float getMin(variable var) const
    {
        int i = static_cast<int>(var);
        return varMins[i];
    }

    float getMax(parameter param) const
    {
        int i = static_cast<int>(param);
        return paramMaxs[i];
    }

    float getMax(variable var) const
    {
        int i = static_cast<int>(var);
        return varMaxs[i];
    }

    void addLimits(parameter param, float min, float max)
    {
        int index = static_cast<int>(param);
        paramMaxs[index] = max;
        paramMins[index] = min;
    }

    void addLimits(variable var, float min, float max)
    {
        int index = static_cast<int>(var);
        varMaxs[index] = max;
        varMins[index] = min;
    }

private:
    float paramMins[parameter::n_parameters];
    float paramMaxs[parameter::n_parameters];

    float varMins[variable::n_variables];
    float varMaxs[variable::n_variables];
};

#endif // PARAM_LIMITER_H_