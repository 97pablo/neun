/*************************************************************

Copyright (c) 2006, Fernando Herrero Carrón
Copyright (c) 2020, Angel Lareo <angel.lareo@gmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of the author nor the names of his contributors
      may be used to endorse or promote products derived from this
      software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************/

#include <DifferentialNeuronWrapper.h>
#include <HodgkinHuxleyModel.h>
#include <RungeKutta4.h>
#include <NeuronOptimizerWrapper.h>
#include <OptimizationObjectiveWrapper.h>
#include <ParamLimiter.h>

#include <VoltageDifferenceObjective.h>
#include <NeuronGAOptimizer.h>

#include <vector>

typedef RungeKutta4 Integrator;
typedef DifferentialNeuronWrapper<HodgkinHuxleyModel<float>, Integrator>
    Neuron;
typedef ParamLimiter<Neuron> Limiter;
typedef OptimizationObjectiveWrapper<VoltageDifferenceObjective<Neuron>> Objective;
typedef NeuronOptimizerWrapper<NeuronGAOptimizer<Objective>> Optimizer;

int main(int argc, char **argv)
{
  // Initializes the patameters for the optimization objective
  Objective::ConstructorArgs objectiveArgs;
  objectiveArgs.params[Objective::time] = 50;
  objectiveArgs.params[Objective::step] = 0.001;
  Objective objective(objectiveArgs);

  // Establishes bounds for the values of each parameter
  Limiter limiter;
  limiter.addLimits(Neuron::cm, 0.5, 2.0);
  limiter.addLimits(Neuron::vna, 30.0, 70.0);
  limiter.addLimits(Neuron::vk, -100.0, -60.0);
  limiter.addLimits(Neuron::vl, -80.0, -40.0);
  limiter.addLimits(Neuron::gna, 50.0, 300.0);
  limiter.addLimits(Neuron::gk, 10.0, 100.0);
  limiter.addLimits(Neuron::gl, 0.05, 1.0);

  limiter.addLimits(Neuron::v, -100.0, -20.0);
  limiter.addLimits(Neuron::m, 0.0, 0.2);
  limiter.addLimits(Neuron::n, 0.4, 0.8);
  limiter.addLimits(Neuron::h, 0.0, 1.0);

  // Initializes the paramters for the optimizer
  Optimizer::ConstructorArgs optimizerArgs;
  optimizerArgs.params[Optimizer::pConv] = 0.99;
  optimizerArgs.params[Optimizer::pRepl] = 0.25;
  optimizerArgs.params[Optimizer::popSize] = 1000;
  optimizerArgs.params[Optimizer::pCross] = 0.9;
  optimizerArgs.params[Optimizer::pMut] = 0.05;
  optimizerArgs.params[Optimizer::nGens] = 500;
  optimizerArgs.params[Optimizer::nElite] = 1;

  // Creates the optimizer with the setup
  Optimizer optimizer(optimizerArgs, objective, limiter);

  std::cout << "Optimizing neuron:\n";
  Neuron optimizedNeuron = optimizer.generate();

  std::cout << "Optimized Neuron Parameters:\n";
  std::cout << "cm   = " << optimizedNeuron.get(Neuron::cm) << "\n";
  std::cout << "vna  = " << optimizedNeuron.get(Neuron::vna) << "\n";
  std::cout << "vk   = " << optimizedNeuron.get(Neuron::vk) << "\n";
  std::cout << "vl   = " << optimizedNeuron.get(Neuron::vl) << "\n";
  std::cout << "gna  = " << optimizedNeuron.get(Neuron::gna) << "\n";
  std::cout << "gk   = " << optimizedNeuron.get(Neuron::gk) << "\n";
  std::cout << "gl   = " << optimizedNeuron.get(Neuron::gl) << "\n";

  std::cout << "\nNeuron State Variables:\n";
  std::cout << "v = " << optimizedNeuron.get(Neuron::v) << "\n";
  std::cout << "m = " << optimizedNeuron.get(Neuron::m) << "\n";
  std::cout << "n = " << optimizedNeuron.get(Neuron::n) << "\n";
  std::cout << "h = " << optimizedNeuron.get(Neuron::h) << "\n";

  std::ofstream data("example2.txt");

  // print results
  const double step = 0.001;
  double simulation_time = 100;
  for (double time = 0; time < simulation_time; time += step)
  {
    optimizedNeuron.step(step);
    double v = optimizedNeuron.get(Neuron::v);
    data << time << ' ' << v << '\n';
  }

  data.close();

  return 0;
}
