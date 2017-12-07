/*******************************************************************************
 * This file is part of libraries to evaluate performance of Background 
 * Subtraction algorithms.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
#ifndef _DNN_ALGORITHM_H
#define _DNN_ALGORITHM_H

//#include <opencv2/opencv.hpp>
//using namespace cv;

class DeepNeuralNetworkAlgorithmI
{
public:
  DeepNeuralNetworkAlgorithmI() {}
  virtual ~DeepNeuralNetworkAlgorithmI() {}
  virtual void SetAlgorithmParameters()         = 0;
  virtual void LoadConfigParameters(std::string)= 0;
  virtual void SaveConfigParameters()           = 0;
  virtual void Initialization()                 = 0;
  virtual void GetBackground(OutputArray)       = 0;
  virtual void GetForeground(OutputArray)       = 0;
  virtual void Update(InputArray, OutputArray)  = 0;
  virtual void Process(InputArray, OutputArray) = 0;
  virtual void GetClassProb(int*, double*)      = 0;
  virtual void LoadModel(std::string)           = 0;
  virtual void SaveModel()                      = 0;
  virtual std::string PrintParameters()         = 0;
  virtual const std::string Name()              = 0;
  virtual std::string ElapsedTimeAsString()     = 0;
  virtual double ElapsedTime()                  = 0;
  virtual std::string className()               = 0;
  virtual std::string className(int)            = 0;
  virtual int         classId()                 = 0;
  virtual double      classProbability()        = 0;
  virtual double      classProbability(int)     = 0;
};

#endif
