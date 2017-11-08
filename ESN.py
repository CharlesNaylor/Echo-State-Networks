# Copyright 2017 Francesco Mannella (francesco.mannella@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this fileexcept in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module implementing the EchoStateRNN Cell.

This module provides the EchoStateRNN Cell, implementing the leaky ESN as 
described in  http://goo.gl/bqGAJu.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell_impl

# to pass to py_func
def np_eigenvals(x):
    return np.linalg.eigvals(x).astype('complex64')

class EchoStateRNNCell(rnn_cell_impl.RNNCell):
    """Echo-state RNN cell.

    """

    def __init__(self, num_units, num_inputs=1, num_readouts=1, decay=0.1, epsilon=1e-10, alpha=0.4, 
                 sparseness=0.0, rng=None, batch=True, activation=None, saturation=None, 
                 optimize_vars=None, reuse=None):
        """
        Args:
            num_units: int, The number of units in the RNN cell.
            num_inputs: int, The number of input units to the RNN cell.
            num_readouts: int, The number of readout units of the RNN cell.
            decay: float, Decay of the ODE of each unit. Default: 0.1.
            epsilon: float, Discount from spectral radius 1. Default: 1e-10.
            alpha: float [0,1], the proporsion of infinitesimal expansion vs infinitesimal rotation
                of the dynamical system defined by the inner weights
            sparseness: float [0,1], sparseness of the inner weight matrix. Default: 0.
            rng: np.random.RandomState, random number generator. Default: None.
            batch: if the readout activation are computed after the simulation. (Cannot have feedback). Default: True.
            activation: Nonlinearity to use for the output.  Default: `tanh`.
            activation: Nonlinearity to use for the neural step.  Default: `tanh`.
            optimize: Python boolean describing whether to optimize rho and alpha. 
            optimize_vars: list containing variables to be optimized
            reuse: (optional) Python boolean describing whether to reuse 
                variables in an existing scope. If not `True`, and 
                the existing scope already has the given variables, 
                an error is raised.
        """

        # Basic RNNCell initialization
        super(EchoStateRNNCell, self).__init__()
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self.saturation = saturation or math_ops.tanh
        self._reuse = reuse
        self.num_inputs = num_inputs
        self.num_readouts = num_readouts
        self.epsilon = epsilon
        self.sparseness = sparseness
        self.batch = batch
        
                
        # Random number generator initialization
        self.rng = rng
        if rng is None:
            self.rng = np.random.RandomState()
        
        # build initializers for tensorflow variables 
        self.w = self.buildInputWeights()
        self.wout = self.buildOutputWeights()
        self.wfb = self.buildfbWeights()
        self.u = self.buildEchoStateWeights()
        
        self.W = tf.get_variable('W',initializer = self.w, trainable = False)   
        self.U = tf.get_variable('U', initializer = self.u, trainable = False)
        self.Wfb = tf.get_variable('Wfb',initializer = self.w, trainable = False)   
        self.Wout = tf.get_variable('Wout',initializer = self.wout, trainable = False)   
       
        self.state = tf.get_variable('state', initializer = 
                                     tf.zeros([1,self._num_units], dtype=tf.float32 ), trainable = False)  
        self.output = tf.get_variable('output', initializer = 
                                      tf.zeros([1, self._num_units], dtype=tf.float32 ), trainable = False) 
     
        self.readouts = tf.get_variable('readouts',initializer = 
                                        tf.zeros([1, self._num_units], dtype=tf.float32 ), trainable = False)  

        # alpha and rho default as tf non trainables  
        self.optimize_table = {"alpha": False, 
                               "rho": False, 
                               "decay": False,
                               "sw": False}
        
        # Set tf trainables  
        if optimize_vars is not None:
            for var in ["alpha", "rho", "decay", "sw" ]:
                if var in optimize_vars:
                    self.optimize_table[var] = True
                
        self.decay = tf.get_variable('decay', initializer = decay, 
                                     trainable = self.optimize_table["decay"])
        self.alpha = tf.get_variable('alpha', initializer = alpha, 
                                     trainable = self.optimize_table["alpha"])
        self.rho = tf.get_variable('Rho', initializer = 0.5 if self.optimize_table["rho"] 
                                   else 1 - self.epsilon, trainable = self.optimize_table["rho"]) 
        self.sw = tf.get_variable('sw', initializer = 4.0 if self.optimize_table["sw"] 
                                   else 1 - self.epsilon, trainable = self.optimize_table["sw"])   
        
        self.init()
    
    def init(self):
        self.setEchoStateProperty()
 
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state, readouts = None):
        """ Echo-state RNN: """
        
        if  not self.batch and readouts is not None:
           # x = x + h*(f(W*inp + Wfb*rout + U*g(x)) - x). 
           new_state = self.state + self.decay*(
                    self.saturation(
                        tf.matmul(inputs, self.W * self.sw) + 
                        tf.matmul(readouts, self.Wfb) + 
                        tf.matmul(self._activation(state), self.U * self.rho_one * self.rho) 
                    )
                - self.state)
        else:
           # x = x + h*(f(W*inp + U*g(x)) - x). 
           new_state = self.state + self.decay*(
                    self.saturation(
                        tf.matmul(inputs, self.W * self.sw) + 
                        tf.matmul(self._activation(state), self.U * self.rho_one * self.rho) 
                    )
                - self.state)       
        self.output = self._activation(new_state)
        self.readouts = tf.matmul(self.output , self.Wout) 

        return self.output, new_state   
    
    def reset(self):
        self.state = tf.zeros([1, self._num_units], dtype=tf.float32)
        self.output = tf.zeros([1, self._num_units], dtype=tf.float32)
        self.readouts = tf.zeros([1, self.num_readouts], dtype=tf.float32)
        
 
    def setEchoStateProperty(self):
        """ optimize U to obtain alpha-imporooved echo-state property """
        self.U.assign(self.u)
        self.U = self.set_alpha(self.U)
        self.U = self.normalizeEchoStateWeights(self.U)
        self.rho_one = self.buildEchoStateRho(self.U) 
 
    def set_alpha(self, W):
        """ Decompose rotation and translation """

        return 0.5*(self.alpha*(W + tf.transpose(W)) + (1 - self.alpha)*(W - tf.transpose(W)))
        
    def buildInputWeights(self):
        """            
            Returns:
            
            A 1-D tensor representing the 
            input weights to the ESN    
        """  

        # Input weight tensor initializer
        return self.rng.uniform(-1, 1, [self.num_inputs, self._num_units]).astype("float32") 
    
    def buildOutputWeights(self):
        """            
            Returns:
            
            A 1-D tensor representing the 
            output weights from the ESN  to the readouts 
        """  

        # Input weight tensor initializer
        return self.rng.uniform(-1, 1, [self._num_units, self.num_readouts]).astype("float32") 
     
    def buildfbWeights(self):
        """            
            Returns:
            
            A 1-D tensor representing the 
            output weights from the  readouts to the ESN
        """  

        # Input weight tensor initializer
        return self.rng.uniform(-1, 1, [self.num_readouts, self._num_units]).astype("float32")       
    
    def buildEchoStateWeights(self):
        """            
            Returns:
            
            A 1-D tensor representing the 
            inner weights to an ESN (to be optimized)        
        """    

        # Inner weight tensor initializer
        # 1) Build random matrix
        W = self.rng.randn(self._num_units, self._num_units).astype("float32") * \
                (self.rng.rand(self._num_units, self._num_units) < 1 - self.sparseness) 
        return W
    
    def normalizeEchoStateWeights(self, W):
        # 2) Normalize to spectral radius 1

        eigvals = tf.py_func(np_eigenvals, [W], tf.complex64) 
        W /= tf.reduce_max(tf.abs(eigvals)) 

        return W
              
    def buildEchoStateRho(self, W):
        """Build the inner weight matrix initialixer W  so that  
        
            1 - epsilon < rho(W)  < 1,
        
            where 
        
            Wd = decay * W + (1 - decay) * I. 
        
            See Proposition 2 in Jaeger et al. (2007) http://goo.gl/bqGAJu.
            See also https://goo.gl/U6ALDd. 

            Returns:

                A 2-D tensor representing the 
                inner weights of an ESN      
        """
            
        # Correct spectral radius for leaky units. The iteration 
        #    has to reach this value 
        target = 1.0
        # spectral radius and eigenvalues
        eigvals = tf.py_func(np_eigenvals, [W], tf.complex64) 
        x = tf.real(eigvals) 
        y = tf.imag(eigvals)  
        # solve quadratic equations
        a = x**2 * self.decay**2 + y**2 * self.decay**2
        b = 2 * x * self.decay - 2 * x * self.decay**2
        c = 1 + self.decay**2 - 2 * self.decay - target**2
        # just get the positive solutions
        sol = (tf.sqrt(b**2 - 4*a*c) - b)/(2*a)
        # and take the minor amongst them
        effective_rho = tf.reduce_min(sol)
        rho = effective_rho
        # 4 defne the tf variable      

        return rho
