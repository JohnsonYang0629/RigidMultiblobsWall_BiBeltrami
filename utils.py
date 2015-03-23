'''File with utilities for the scripts and functions in this project.'''
import logging
import matplotlib
matplotlib.use('Agg')
from matplotlib import animation
from matplotlib import pyplot
import numpy as np
import os


DT_STYLES = {}  # Used for plotting different timesteps of MSD.

# Fake log-like class to redirect stdout to log file.
class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''
 
   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())


# Static Variable decorator for calculating acceptance rate.
def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate

class MSDStatistics(object):
  '''
  Class to hold the means and std deviations of the time dependent
  MSD for multiple schemes and timesteps.  data is a dictionary of
  dictionaries, holding runs indexed by scheme and timestep in that 
  order.
  Each run is organized as a list of 3 arrays: [time, mean, std]
  mean and std are matrices (the rotational MSD).
  '''
  def __init__(self, params):
    self.data = {}
    self.params = params


  def add_run(self, scheme_name, dt, run_data):
    ''' 
    Add a run.  Create the entry if need be. 
    run is organized as a list of 3 arrays: [time, mean, std]
    In that order.
    '''
    if scheme_name not in self.data:
      self.data[scheme_name] = dict()
    self.data[scheme_name][dt] = run_data


  def print_params(self):
     print "Parameters are: "
     print self.params
     

def plot_time_dependent_msd(msd_statistics, ind, figure, color=None, symbol=None,
                            label=None, error_indices=[0, 1, 2, 3, 4, 5], data_name=None,
                            num_err_bars=None):
  ''' 
  Plot the <ind> entry of the rotational MSD as 
  a function of time on given figure (integer).  
  This uses the msd_statistics object
  that is saved by the *_rotational_msd.py script.
  ind contains the indices of the entry of the MSD matrix to be plotted.
  ind = [row index, column index].
  '''
  scheme_colors = {'RFD': 'g', 'FIXMAN': 'b', 'EM': 'r'}
  pyplot.figure(figure)
  # Types of lines for different dts.
  write_data = True
  data_write_type = 'w'
  if not data_name:
     data_name = "MSD-component-%s-%s.txt" % (ind[0], ind[1])
  if write_data:
    np.set_printoptions(threshold=np.nan)
    with open(os.path.join('.', 'data', data_name), data_write_type) as f:
      f.write('  \n')
  if not num_err_bars:
     num_err_bars = 40
  linestyles = [':', '--', '-.', '']
  for scheme in msd_statistics.data.keys():
    for dt in msd_statistics.data[scheme].keys():
      if dt in DT_STYLES.keys():
        if not symbol:
           style = ''
           nosymbol_style = DT_STYLES[dt]
        else:
           style = symbol #+ DT_STYLES[dt]
           nosymbol_style = DT_STYLES[dt]
      else:
        if not symbol:
           style = '' #linestyles[len(DT_STYLES)]
           DT_STYLES[dt] = linestyles[len(DT_STYLES)]
           nosymbol_style = DT_STYLES[dt]
        else:
           DT_STYLES[dt] = linestyles[len(DT_STYLES)]
           style = symbol #+ DT_STYLES[dt]
           nosymbol_style = DT_STYLES[dt]
      # Extract the entry specified by ind to plot.
      num_steps = len(msd_statistics.data[scheme][dt][0])
      # Don't put error bars at every point
      err_idx = [int(num_steps*k/num_err_bars) for k in range(num_err_bars)]
      msd_entries = np.array([msd_statistics.data[scheme][dt][1][_][ind[0]][ind[1]]
                              for _ in range(num_steps)])
      msd_entries_std = np.array(
        [msd_statistics.data[scheme][dt][2][_][ind[0]][ind[1]]
         for _ in range(num_steps)])
      # Set label and style.
      if label:
         #HACK, use scheme in Label + given.
         # ('dt = %s ' % dt) + 
         if scheme == 'FIXMAN':
            plot_label = scheme.capitalize() + label
         else:
            plot_label = scheme + label
      else:
         plot_label = '%s, dt=%s' % (scheme, dt)

      if color:
         plot_style = color + style
         nosymbol_plot_style = color + nosymbol_style
         err_bar_color = color
      else:
         plot_style = scheme_colors[scheme] + style
         nosymbol_plot_style = scheme_colors[scheme] + nosymbol_style
         err_bar_color = scheme_colors[scheme]

      pyplot.plot(np.array(msd_statistics.data[scheme][dt][0])[err_idx],
                  msd_entries[err_idx],
                  plot_style,
                  label = plot_label)
      pyplot.plot(msd_statistics.data[scheme][dt][0],
                  msd_entries,
                  nosymbol_plot_style)
      
      if ind[0] == 0 and scheme == 'RFD':
        #HACK line of best fit for Translational MSD.
#        fit_line = np.polyfit(msd_statistics.data[scheme][dt][0], msd_entries, 1)
#        print 'np.polyfit is ', fit_line
#        slope_ratio = fit_line[0]/(4.*0.2*0.06)
#        pyplot.plot(msd_statistics.data[scheme][dt][0], 
#                    fit_line[0]*np.array(msd_statistics.data[scheme][dt][0]),
#                    'k-.',
#                    label='%.2f * Average Mobility' % slope_ratio)
         pass

      if write_data:
        with open(os.path.join('.', 'data', data_name),'a') as f:
          f.write("scheme %s \n" % scheme)
          f.write("dt %s \n" % dt)
          f.write("time: %s \n" % msd_statistics.data[scheme][dt][0])
          f.write("MSD component: %s \n" % msd_entries)
          f.write("Std Dev:  %s \n" % msd_entries_std)

      if ind[0] in error_indices:
         pyplot.errorbar(np.array(msd_statistics.data[scheme][dt][0])[err_idx],
                         msd_entries[err_idx],
                         yerr = 2.*msd_entries_std[err_idx],
                         fmt = err_bar_color + '.')
  pyplot.ylabel('MSD')
  pyplot.xlabel('time')

def log_time_progress(elapsed_time, time_units, total_time_units):
  ''' Write elapsed time and expected duration to progress log.'''
  progress_logger = logging.getLogger('progress_logger')  
  expected_duration = elapsed_time*total_time_units/time_units
  if elapsed_time > 60.0:
    progress_logger.info('Elapsed Time: %.2f Minutes.' % 
                         (float(elapsed_time/60.)))
  else:
    progress_logger.info('Elapsed Time: %.2f Seconds' % float(elapsed_time))
  if expected_duration > 60.0:
    progress_logger.info('Expected Duration: %.2f Minutes.' % 
                         (float(expected_duration/60.)))
  else:
      progress_logger.info('Expected Duration: %.2f Seconds' % 
                           float(expected_duration))


def _calc_total_msd_from_matrix_and_center(original_center, original_rot_matrix, 
                                       final_center, rot_matrix):
  ''' 
  Calculate 6x6 MSD including orientation and location.  This is
  calculated from precomputed center of the tetrahedron and rotation
  matrix data to avoid repeating computation.
  '''
  u_hat = np.zeros(3)
  for i in range(3):
    e = np.zeros(3)
    e[i] = 1.
    u_hat += 0.5*np.cross(np.inner(original_rot_matrix, e),
                          np.inner(rot_matrix, e))
    
  dx = np.array(final_center) - np.array(original_center)
  displacement = np.concatenate([dx, u_hat])
  return np.outer(displacement, displacement)

def calc_msd_data_from_trajectory(trajectory_data, calc_center_function, end):
  ''' Calculate rotational and translational (6x6) MSD matrix given a dictionary of
  trajectory data.  Return a numpy array of 6x6 MSD matrices, one for each time.'''

  burn_in = 0  # We start from an equilibrium sample, so we don't need burn_in.
  orientations = trajectory_data['orientation']
  locations = trajectory_data['location']
  average_rotational_msd = np.array([np.zeros((6, 6)) 
                                     for _ in range(trajectory_length)])
  lagged_rotation_trajectory = []
  lagged_location_trajectory = []
  for k in range(len(locations)):
    if k > burn_in: 
       orientation = Quaternion(orientations[k])
       lagged_rotation_trajectory.append(orientation.rotation_matrix())
       lagged_location_trajectory.append(calc_center_function(locations[k], orientation))
    if len(lagged_location_trajectory) > trajectory_length:
      lagged_location_trajectory = lagged_location_trajectory[1:]
      lagged_rotation_trajectory = lagged_rotation_trajectory[1:]
      for l in range(trajectory_length):
        current_rot_msd = (calc_total_msd_from_matrix_and_center(
           lagged_location_trajectory[0],
           lagged_trajectory[0],
           lagged_location_trajectory[l],
           lagged_trajectory[l]))
        average_rotational_msd[l] += current_rot_msd

  average_rotational_msd = (average_rotational_msd/
                            (n_steps - trajectory_length - burn_in))
  
  return average_rotational_msd
   
   
      
      
   
   

   
