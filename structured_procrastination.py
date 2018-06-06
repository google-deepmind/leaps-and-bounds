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

import argparse
import heapq
import numpy as np
import simulated_environment

C = 12.  # constant for l_i


def structured_procrastination(env, n, epsilon, delta, zeta, k0, k_bar,
                               theta_multiplier):
  """Implementation of Structured Procrastination."""
  # The names of the variables used here agree with the pseudocode in the paper,
  # except q is used instead of the paper's upper-case Q, and qq is used instead
  # of the paper's lower-case q. The pseudocode overloads l: here, ll is used to
  # represent the scalar (appears as l in paper), and l is used to represent the
  # array (appears as l_i in the paper). For efficiency, we implement the argmin
  # in line 10 of the paper with a heap.
  k, l, q, qq, r, r_sum, heap = [], [], [], [], [], [], []
  beta = np.log2(k_bar / k0)
  for i in xrange(n):  # Line 2 in paper.
    k.append(0)
    l.append(
        int(np.ceil(C / (epsilon * epsilon) * np.log(3 * beta * n / zeta))))
    q.append([])
    qq.append(0)
    r.append([])
    r_sum.append(0)
    heapq.heappush(heap, (0, i))
    for ll in xrange(l[i]):  # Line 6 in paper.
      r[i].append(0)
      q[i].append((ll, k0))

  # Main loop.
  current_delta = 1
  iter_count = 0
  while current_delta > delta:  # Line 9, but stop when target delta reached.
    iter_count += 1
    _, i = heapq.heappop(heap)
    ll, theta = q[i].pop(0)
    if r[i][ll] == 0:  # Line 12 in paper.
      k[i] += 1
      qq[i] = int(
          np.ceil(
              C /
              (epsilon * epsilon) * np.log(3 * beta * n * k[i] * k[i] / zeta)))

    did_timeout, elapsed = env.run(config_id=i, timeout=theta, instance_id=ll)
    if not did_timeout:  # Line 15 in paper.
      r_sum[i] += elapsed - r[i][ll]
      r[i][ll] = elapsed
    else:
      r_sum[i] += theta - r[i][ll]
      r[i][ll] = theta
      q[i].append((ll, theta_multiplier * theta))
    while len(q[i]) < qq[i]:  # Line 20 in paper.
      l[i] += 1
      r[i].append(0)
      q[i].insert(0, (l[i] - 1, theta))

    heapq.heappush(heap, (r_sum[i] / k[i], i))  # Bookeeping for the heap.

    if iter_count % 10000 == 0:
      # Recalculate delta, as in line 25 of the paper. We recalculate it within
      # this while loop, because instead of an anytime algorithm, we want one
      # that stops when the target delta is reached.
      i_star = np.argmax(r_sum)
      current_delta = np.sqrt(1 + epsilon) * qq[i_star] / k[i_star]
      print('iter_count={}, delta={}, theta={}, total runtime so far={}'.format(
          iter_count, current_delta, theta, env.get_total_runtime()))

  return i_star, current_delta


def main():
  parser = argparse.ArgumentParser(
      description='Executes Structured Procrastination '
      'with a simulated environment.')
  parser.add_argument('--epsilon', help='Epsilon from the paper',
                      type=float, default=0.2)
  parser.add_argument('--delta', help='Delta from the paper',
                      type=float, default=0.2)
  parser.add_argument('--zeta', help='Zeta from the paper',
                      type=float, default=0.1)
  parser.add_argument('--k0', help='Kappa_0 from the paper',
                      type=float, default=1.)
  parser.add_argument('--k-bar', help='bar{Kappa} from the paper',
                      type=float, default=1000000.)
  parser.add_argument('--theta-multiplier',
                      help='Theta multiplier from the paper',
                      type=float, default=2.0)
  parser.add_argument('--measurements-filename',
                      help='Filename to load measurement results from',
                      type=str, default='measurements.dump')
  parser.add_argument('--measurements-timeout',
                      help='Timeout (seconds) used for the measurements',
                      type=float, default=900.)
  args = vars(parser.parse_args())

  epsilon = args['epsilon']
  delta = args['delta']
  zeta = args['zeta']
  k0 = args['k0']
  k_bar = args['k_bar']
  theta_multiplier = args['theta_multiplier']
  results_file = args['measurements_filename']
  timeout = args['measurements_timeout']

  env = simulated_environment.Environment(results_file, timeout)
  num_configs = env.get_num_configs()
  best_config_index, delta = structured_procrastination(
      env, num_configs, epsilon, delta, zeta, k0, k_bar, theta_multiplier)
  print('best_config_index={}, delta={}'.format(best_config_index, delta))
  env.print_config_stats(best_config_index)
  def format_runtime(runtime):
    return '{}s = {}m = {}h = {}d'.format(
        runtime, runtime / 60, runtime / 3600, runtime / (3600 * 24))
  print('total runtime: ' + format_runtime(env.get_total_runtime()))
  print('total resumed runtime: ' +
        format_runtime(env.get_total_resumed_runtime()))


if __name__ == '__main__':
  main()
