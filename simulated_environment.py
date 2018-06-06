#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
import pickle
import numpy as np


class Environment(object):
  """This class is used for simulating runs and collecting statistics."""

  def __init__(self, results_file, timeout):
    """Prepares an instance that can simulate runs based on a measurements file.

    Args:
      results_file: the location of the pickle dump containing the results
        of the runtime measurements.
      timeout: the timeout used for the runtime measurements.
    """
    self._timeout = timeout

    # Load measurements.
    with open(results_file, 'r') as f:
      results = pickle.load(f)
    self._results = [results[k] for k in sorted(results.keys())]
    self._instance_count = len(self._results[0])
    self.reset()

  def reset(self):
    """Reset the state of the environment."""
    # Total runtime, without resuming, of any configuration ran on any instance.
    # Without resuming means that if the same configuration-instance pair is
    # run, `total_runtime` will track the time taken as if the execution had to
    # be restarted, rather than resumed from when it was last interrupted
    # due to a timeout.
    self._total_runtime = 0
    # Total runtime, with resuming, of any configuration ran on any instance.
    self._total_resumed_runtime = 0
    # Dict mapping a configuration to how long it was run, with resuming, on all
    # instances combined.
    self._runtime_per_config = collections.defaultdict(float)
    # Dict mapping a configuration and an instance to how long it ran so far
    # in total, with resuming. Summing the runtimes for all instances for a
    # configuration will be equal to the relevant value in `runtime_per_config`.
    self._ran_so_far = collections.defaultdict(
        lambda: collections.defaultdict(float))

  def get_num_configs(self):
    return len(self._results)

  def get_num_instances(self):
    return self._instance_count

  def get_total_runtime(self):
    return self._total_runtime

  def get_total_resumed_runtime(self):
    return self._total_resumed_runtime

  def run(self, config_id, timeout, instance_id=None):
    """Simulates a run of a configuration on an instance with a timeout.

    Args:
      config_id: specifies which configuration to run. Integer from 0 to
        get_num_configs() - 1.
      timeout: the timeout to simulate the run with.
      instance_id: the instance to run. If not specified, a random instance
        will be run.

    Raises:
      ValueError: if the supplied timeout is larger than self.timeout, the
        requested simulation cannot be completed and this error will be raised.

    Returns:
      A tuple of whether the simulated run timed out, and how long it ran.
    """
    if timeout > self._timeout:
      raise ValueError('timeout provided is too high to be simulated.')
    if instance_id is None:
      instance_id = np.random.randint(self._instance_count)
    runtime = min(timeout,
                  self._results[config_id][instance_id % self._instance_count])
    self._total_runtime += runtime
    resumed_runtime = runtime - self._ran_so_far[config_id][instance_id]
    self._runtime_per_config[config_id] += resumed_runtime
    self._ran_so_far[config_id][instance_id] = runtime
    self._total_resumed_runtime += resumed_runtime
    return (timeout <=
            self._results[config_id][instance_id % self._instance_count],
            runtime)

  def print_config_stats(self, config_id, tau=None):
    """Prints statistics about a particular configuration."""

    # Compute average runtime capped at TIMEOUT.
    average = np.mean([min(self._timeout, r) for r in self._results[config_id]])
    print('avg runtime capped at the dataset\'s timeout: {}'.format(average))
    timeout_count = 0
    for t in self._results[config_id]:
      if t > self._timeout:
        timeout_count += 1
    print('fraction of instances timing out at the timeout of the dataset: {}'.
          format(float(timeout_count) / len(self._results[config_id])))
    if tau is not None:
      timeout_count = 0
      for t in self._results[config_id]:
        if t > tau:
          timeout_count += 1
      print('fraction of instances timing out at tau: {}'.format(
          float(timeout_count) / len(self._results[config_id])))
    with open('runtime_per_config.dump', 'wb') as outf:
      pickle.dump(self._runtime_per_config, outf)
