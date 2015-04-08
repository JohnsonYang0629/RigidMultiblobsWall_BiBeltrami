''' 
Script to compare free tetrahedron translational MSD to 
that generated by using IBAMR with stiff springs.
'''

import cPickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
sys.path.append('..')

import tetrahedron_free as tf
from translational_diffusion_coefficient import calculate_average_mu_parallel_and_perpendicular
from utils import MSDStatistics
from utils import plot_time_dependent_msd


IBAMR_TIME = np.array([0 , 16 , 32 , 48 , 64 , 80 , 96 , 112, 128, 
                       144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304])

IBAMR_PARALLEL = np.array(
  [0., 0.23030084, 0.46050293, 0.69003878,
   0.91991073, 1.1500172, 1.3800153, 1.6088495, 1.8364486,
   2.0640567, 2.2921163, 2.5206239, 2.7501723, 2.979825,
   3.2094257, 3.4390972, 3.6691305, 3.9003276, 4.1311499,  
   4.3611359])

IBAMR_PARALLEL_STD = np.array(
  [0., 0.001477575, 0.003386016, 0.006319060, 0.010594295, 0.015719995,
   0.022184539, 0.029280662, 0.037885666, 0.045756387, 0.053230181,
   0.060988031, 0.068676815, 0.077319251, 0.086739617, 0.096781223,
   0.10675642, 0.11709404, 0.12807461, 0.1388481])


IBAMR_PERP = np.array(
  [0., 0.15078125, 0.27641909, 0.38550696, 0.481194, 0.56581186, 
   0.64170116, 0.71043655, 0.77321308, 0.82912696, 0.88002487, 
   0.92730235, 0.97134252, 1.0123946, 1.0509769, 1.0867704, 
   1.1207078, 1.152183, 1.1801148, 1.2056205])

IBAMR_PERP_STD = np.array(
  [0., 0.001645053, 0.003407508, 0.005898420, 0.009283213, 0.013286326, 
   0.017028906, 0.020202788, 0.023350492, 0.026647023, 0.030101367, 
   0.033308575, 0.036714922, 0.039723968, 0.043054832, 0.046361506, 
   0.049286502, 0.052712193, 0.055929978, 0.059303183])


if __name__ == '__main__':
  data_name = ('tetrahedron-msd-dt-0.2-N-1000000-end-800.0-scheme-RFD-'
               'runs-4-final-com.pkl')

  data_file = os.path.join('.', 'data', 
                            data_name)
  
  with open(data_file, 'rb') as f:
    msd_statistics = cPickle.load(f)
    msd_statistics.print_params()

  # Combine 0,0 and 1,1 into msd_parallel
  for scheme in msd_statistics.data:
    for dt in msd_statistics.data[scheme]:
      for k in range(len(msd_statistics.data[scheme][dt][1])):
        msd_statistics.data[scheme][dt][1][k][0][0] = (
          msd_statistics.data[scheme][dt][1][k][0][0] +
          msd_statistics.data[scheme][dt][1][k][1][1])
        msd_statistics.data[scheme][dt][2][k][0][0] = np.sqrt(
          msd_statistics.data[scheme][dt][2][k][0][0]**2 +
          msd_statistics.data[scheme][dt][2][k][1][1]**2)


  average_mob_and_friction = calculate_average_mu_parallel_and_perpendicular(20000)
  mu_parallel_com =  average_mob_and_friction[0] # 0.0711/2.
  print "average parallel mob ", mu_parallel_com
  mu_perp_com = 0.0263
  zz_msd_com = 1.633
  rot_msd_com = 0.167
    
  mu_parallel_center = 0.0711/2.
  mu_perp_center = 0.0263
  zz_msd_center = 1.52
  rot_msd_center = 0.169

  mu_parallel_vertex = 0.117/2.
  mu_perp_vertex = 0.0487
  zz_msd_vertex = 2.517
  rot_msd_vertex = 0.167956760304

  figure_numbers = [1, 5, 1, 2, 3, 4]
  labels= [' Parallel MSD', ' YY-MSD', ' Perpendicular MSD', ' Rotational MSD', ' Rotational MSD', ' Rotational MSD']
  styles = ['o', '^', 's', 'o', '.', '.']
  translation_end = 200.0
  for l in range(6):
    ind = [l, l]
    plot_time_dependent_msd(msd_statistics, ind, figure_numbers[l],
                            error_indices=[0, 2, 3], label=labels[l], symbol=styles[l],
                            num_err_bars=40)
    plt.figure(figure_numbers[l])
    if l in [0]:
      plt.plot([0.0, translation_end], 
               [0.0, translation_end*4.*tf.KT*mu_parallel_com], 'k-',
               lw=2, label=r'Parallel Mobility')
      plt.errorbar(IBAMR_TIME, 2.*IBAMR_PARALLEL, yerr = 4.*IBAMR_PARALLEL_STD,
                   c='red', label='IBAMR Parallel')
    elif l == 2:
      plt.plot([0.0, translation_end],
               [zz_msd_com, zz_msd_com], 'k--',
               lw=2, label='Asymptotic Perpendicular MSD')
      plt.errorbar(IBAMR_TIME, IBAMR_PERP, yerr = 2.*IBAMR_PERP_STD,
                   c='red', label='IBAMR Perpendicular')
      plt.xlim([0., translation_end])
      plt.ylim([0., translation_end*4.*tf.KT*mu_parallel_com])

    if l == 3:
      plt.plot([0.0, 350.],
                  [rot_msd_com, rot_msd_com], 'k--', lw=2, 
                  label='Asymptotic Rotational MSD')
      plt.xlim([0., 350.])

    plt.title('MSD(t) for Tetrahedron')
    plt.legend(loc='best', prop={'size': 11})
    plt.savefig('./figures/TimeDependentRotationalMSD-Component-%s-%s.pdf' % 
                   (l, l))
