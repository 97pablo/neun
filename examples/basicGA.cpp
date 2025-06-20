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
#include <DynamicalSystemLimiter.h>

#include <VoltageDifferenceObjective.h>
#include <NetworkGAOptimizer.h>

#include <vector>
#include <ElectricalSynapsis.h>
#include <NeuronNetwork.h>

typedef RungeKutta4 Integrator;
typedef HodgkinHuxleyModel<float> HH;
typedef DifferentialNeuronWrapper<HH, Integrator> Neuron;
typedef ElectricalSynapsis<Neuron, Neuron> Synapsis;
typedef NeuronNetwork<Neuron, Synapsis> Network;

typedef VoltageDifferenceObjective<Network> Objective;
typedef NetworkGAOptimizer<Objective> Optimizer;
typedef Optimizer::Blueprint Blueprint;
typedef Blueprint::NeuronLimiter Limiter;

#define STEP 0.001
#define TIME 100
#define INPUT 0.5

double input(double time, const Network &net)
{
  return INPUT;
}

int main(int argc, char **argv)
{
  Neuron::ConstructorArgs neuronArgs;
  // Set the parameter values
  neuronArgs.params[Neuron::cm] = 1 * 7.854e-3;
  neuronArgs.params[Neuron::vna] = 50;
  neuronArgs.params[Neuron::vk] = -77;
  neuronArgs.params[Neuron::vl] = -54.387;
  neuronArgs.params[Neuron::gna] = 120 * 7.854e-3;
  neuronArgs.params[Neuron::gk] = 36 * 7.854e-3;
  neuronArgs.params[Neuron::gl] = 0.3 * 7.854e-3;

  // Initialize a new neuron model
  Neuron n(neuronArgs);

  // You can also initialize the variables of the neuron model to a given value
  n.set(Neuron::v, -80);
  n.set(Neuron::m, 0.1);
  n.set(Neuron::n, 0.7);
  n.set(Neuron::h, 0.01);

  // Initializes the patameters for the optimization objective
  Objective::ConstructorArgs objectiveArgs;
  objectiveArgs.params[Objective::time] = TIME;
  objectiveArgs.params[Objective::step] = STEP;
  objectiveArgs.params[Objective::input] = INPUT;
  Objective objective(objectiveArgs, n, Neuron::v);
  // save the voltages to a file to graph later
  // save the base data to a file
  std::ofstream base_data("basic.txt");
  objective.save_voltages(base_data);
  base_data.close();

  // Establishes bounds for the values of each parameter

  Limiter limiter;
  limiter.addLimits(Neuron::cm, 1e-3, 1.0);
  limiter.addLimits(Neuron::vna, -100.0, 100.0);
  limiter.addLimits(Neuron::vk, -100.0, -100.0);
  limiter.addLimits(Neuron::vl, -100.0, 100.0);
  limiter.addLimits(Neuron::gna, 0, 2.0);
  limiter.addLimits(Neuron::gk, 0, 2.0);
  limiter.addLimits(Neuron::gl, 0, 2.0);

  limiter.addLimits(Neuron::v, -100.0, -60.0);
  limiter.addLimits(Neuron::m, 0.0, 1.0);
  limiter.addLimits(Neuron::n, 0.0, 1.0);
  limiter.addLimits(Neuron::h, 0.0, 1.0);

  // Initializes the paramters for the optimizer
  Optimizer::ConstructorArgs optimizerArgs;
  optimizerArgs.params[Optimizer::pConv] = 0.99;
  optimizerArgs.params[Optimizer::pRepl] = 0.9;
  optimizerArgs.params[Optimizer::popSize] = 1000;
  optimizerArgs.params[Optimizer::pCross] = 0.9;
  optimizerArgs.params[Optimizer::pMut] = 0.15;
  optimizerArgs.params[Optimizer::nGens] = 500;

  Blueprint b;
  b.add_neuron(limiter);
  b.add_synaptic_input(0, input);

  // Creates the optimizer with the setup
  Optimizer optimizer(optimizerArgs, objective, b, 12345);

  std::cout << "Optimizing neuron:\n";
  Network optimizedNetwork = optimizer.generate();

  Neuron *optimizedNeuron = optimizedNetwork.get_neuron(0);

  std::cout << "Optimized Neuron Parameters:\n";
  std::cout << "cm   = " << optimizedNeuron->get(Neuron::cm) << "\n";
  std::cout << "vna  = " << optimizedNeuron->get(Neuron::vna) << "\n";
  std::cout << "vk   = " << optimizedNeuron->get(Neuron::vk) << "\n";
  std::cout << "vl   = " << optimizedNeuron->get(Neuron::vl) << "\n";
  std::cout << "gna  = " << optimizedNeuron->get(Neuron::gna) << "\n";
  std::cout << "gk   = " << optimizedNeuron->get(Neuron::gk) << "\n";
  std::cout << "gl   = " << optimizedNeuron->get(Neuron::gl) << "\n";

  std::cout << "\nNeuron State Variables:\n";
  std::cout << "v = " << optimizedNeuron->get(Neuron::v) << "\n";
  std::cout << "m = " << optimizedNeuron->get(Neuron::m) << "\n";
  std::cout << "n = " << optimizedNeuron->get(Neuron::n) << "\n";
  std::cout << "h = " << optimizedNeuron->get(Neuron::h) << "\n";

  std::ofstream data("recreated.txt");
  optimizedNetwork.simulate(TIME, STEP, data, Neuron::v, Synapsis::i1);
  data.close();

  return 0;
}
