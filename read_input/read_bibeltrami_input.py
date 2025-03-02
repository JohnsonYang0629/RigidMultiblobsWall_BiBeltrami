'''
Simple class to read the input files to run a simulation.
'''

import numpy as np
import ntpath
import sys


class ReadInput(object):
  """
  Simple class to read the input files to generate required input/command files.
  """

  def __init__(self, entries):
    """ Constructor takes the name of the input file """
    self.entries = entries
    self.input_file = entries
    self.options = {}
    number_of_structures = 0

    # Read input file
    comment_symbols = ['#']
    with open(self.input_file, 'r') as f:
      # Loop over lines
      for line in f:
        # Strip comments
        if comment_symbols[0] in line:
          line, comment = line.split(comment_symbols[0], 1)

        # Save options to dictionary, Value may be more than one word
        line = line.strip()
        if line != '':
          option, value = line.split(None, 1)
          if option == 'structure':
            option += str(number_of_structures)
            number_of_structures += 1
          self.options[option] = value

    # Set option to file or default values
    self.pde_solver = str(self.options.get('pde_solver') or 'extrinsic')
    self.k0 = int(self.options.get('k-NN') or 51)
    self.degree = int(self.options.get('degree') or 2)
    self.nu = float(self.options.get('viscosity') or 2.0)
    self.gamma = float(self.options.get('coupling_coefficient_gamma') or 3.0)
    self.slip_save_type = str(self.options.get('slip_save_type') or 'velocity_field')
    self.slip_file_name = str(self.options.get('slip_file_name') or 'sphere_bibeltrami')
    self.save_directory = str(self.options.get('save_directory') or '/slip_files/')
    self.save_potential = str(self.options.get('save_potential') or 'False')
    self.error_display = str(self.options.get('error_display') or 'False')

    # Create list with vertex_file for each structure
    self.num_bodies = number_of_structures
    self.structures = []
    self.structures_ID = []
    for i in range(number_of_structures):
      option = 'structure' + str(i)
      structure_files = str.split(str(self.options.get(option)))
      self.structures.append(structure_files)

    # Create structures ID for each kind
    for struct in self.structures:
      # First, remove directory from structure name
      head, tail = ntpath.split(struct[0])
      # then, remove end (.vertex)
      tail = tail[:-7]
      self.structures_ID.append(tail)

    return
