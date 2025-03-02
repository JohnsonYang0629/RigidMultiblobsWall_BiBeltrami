import sys
import os
import argparse
import distutils
import numpy as np
from rbf_matrix import *

found_functions = False
path_to_append = ''
while found_functions is False:
    try:
        from read_input import read_vertex_file
        from read_input import read_bibeltrami_input
        from body import body
        found_functions = True
    except ImportError as exc:
        sys.stderr.write('Error: failed to import settings module ({})\n'.format(exc))
        path_to_append += '../'
        print('searching functions in path ', path_to_append)
        sys.path.append(path_to_append)
        if len(path_to_append) > 21:
            print('\nProjected functions not found. Edit path in generate_surface_slip.py')
            sys.exit()


def save_ndarray(filename: str, filepath: str, input_value, style_number: int, blob_radius=0):
    os.chdir(filepath)
    length = np.shape(input_value)[0]
    if style_number == 0:
        file = open(filename, mode='w')
        for i in range(0, length):
            str_value = ' '.join(str(j) for j in input_value[i, :])
            line = str_value + '\n'
            file.write(line)
        file.close()
    elif style_number == -1:
        # flush the designated file
        file = open(filename, mode='w')
        file.write(str(''))
        file.close()
    else:
        file = open(filename, mode='w')
        file.write(str(style_number) + '\t' + str(blob_radius) + '\n')
        file.close()
        for i in range(0, length):
            file = open(filename, mode='a')
            str_value = ' '.join(str(j) for j in input_value[i, :])
            line = str_value + '\n'
            file.write(line)
            file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate surface slip files with given force density.')
    parser.add_argument('--input-file', dest='input_file', type=str, default='test_input.txt', help='name of the input file')
    args = parser.parse_args()
    input_file = args.input_file

    read = read_bibeltrami_input.ReadInput(input_file)

    num_bodies = read.num_bodies
    nu = read.nu  # viscosity
    gamma = read.gamma  # coupling coefficient
    k0 = read.k0  # k-NN
    degree = read.degree  # maximum degree
    pde_solver = read.pde_solver
    error_display = read.error_display

    slip_save_type = read.slip_save_type
    save_directory = read.save_directory
    structures = read.structures
    structures_ID = read.structures_ID

    # Create rigid bodies
    bodies = []
    body_types = []

    body_names = []
    blobs_offset = 0
    Laplace_flag = None
    for ID, structure in enumerate(structures):
        print('Creating structures = ', structures_ID[ID])
        # Read vertex and clones files
        struct_ref_config = read_vertex_file.read_vertex_file(structure[0])
        body_names.append(structures_ID[ID])
        # Create each body of type structure
        b = body.Body(np.array([0, 0, 0]), np.array([1, 0, 0, 0]), struct_ref_config, 0)
        b.ID = structures_ID[ID]
        # Append bodies to total bodies list
        bodies.append(b)

    for b in bodies:
        print('Solving ', slip_save_type, ' for structure = ', b.ID)
        body_config = b.reference_configuration
        velocity_field, velocity_field_true, vec_potential_field, vec_potential_field_true = \
            solve_velocity(body_config, nu, gamma, k0, degree, pde_solver)

        # write results into files
        velocity_field_file_name = b.ID + ".slip"
        potential_field_file_name = b.ID + ".potential"

        if slip_save_type.find("velocity") != -1:
            if slip_save_type.find("potential") == -1:
                save_ndarray(velocity_field_file_name, save_directory, velocity_field, style_number=b.Nblobs)
            elif slip_save_type.find("potential") != -1:
                save_ndarray(velocity_field_file_name, save_directory, velocity_field, style_number=b.Nblobs)
                save_ndarray(potential_field_file_name, save_directory, vec_potential_field, style_number=0)

        if error_display == "True":
            error_potential = np.linalg.norm(vec_potential_field - vec_potential_field_true -
                                             np.mean(vec_potential_field) + np.mean(vec_potential_field_true)) \
                              / math.sqrt(vec_potential_field.shape[0])
            error_velocity = np.linalg.norm(velocity_field - velocity_field_true -
                                            np.mean(velocity_field) + np.mean(velocity_field_true)) \
                             / math.sqrt(velocity_field.shape[0])
            print("Error for vector potential field = ", error_potential)
            print("Error for velocity field = ", error_velocity)

