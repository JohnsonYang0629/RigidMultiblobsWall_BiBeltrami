import numpy as np
from mobility import mobility as mob


def gauss_weights(N):
  '''
  Compute Legendre points and weights for Gauss quadrature.
  From Spectral Methods in MATLAB, Trefethen 2000, Program Gauss.m (page 129).
  '''
  s = np.arange(1, N)
  beta = 0.5 / np.sqrt(1.0 - 1.0 / (2 * s)**2)
  T = np.diag(beta, k=1) + np.diag(beta, k=-1)
  eig_values, eig_vectors = np.linalg.eigh(T)
  w = 2 * eig_vectors[0,:]**2
  return eig_values, w


def parametrization(p):
  '''
  Set parametrization, (u,v), with p+1 points along u and (2*p+2) along v.
  In total 2*p**2 points because at the poles we only have one point.

  Return parametrization and weights.
  '''
  # Precomputation
  Nu = p + 1
  Nv = 2 * (p + 1)
  N = Nu * Nv
  t, w_gauss = gauss_weights(Nu)
  u = np.arccos(t)
  v = np.linspace(0, 2*np.pi, Nv, endpoint=False)
  uu, vv = np.meshgrid(u, v, indexing = 'ij')

  # Parametrization
  uv = np.zeros((N,2))
  uv[:,0] = uu.flatten()
  uv[:,1] = vv.flatten()

  # Weights precomputation
  uw = w_gauss / np.sin(u)
  vw = np.ones(v.size) * 2 * np.pi / Nv  
  uuw, vvw = np.meshgrid(uw, vw, indexing = 'ij')
  
  # Weights
  w = uuw.flatten() * vvw.flatten()
  
  return uv, w


def sphere(a, uv):
  '''
  Return the points on a sphere of radius a parametrized by uv.
  '''
  # Generate coordinates
  x = np.zeros((uv.shape[0], 3))
  x[:,0] = a * np.cos(uv[:,1]) * np.sin(uv[:,0])
  x[:,1] = a * np.sin(uv[:,1]) * np.sin(uv[:,0])
  x[:,2] = a * np.cos(uv[:,0]) 
  return x



def plot_velocity_field(bodies, lambda_blobs, eta, sphere_radius, p, output, frame_body=-1, *args, **kwargs):
  '''
  This function plots the velocity field to a Chebyshev-Fourier spherical grid of radius "sphere_radius" centered at (0,0,0).
  If frame_body=index the bodies are translated and rotated so the configuration of body index is x=(0,0,0, 1,0,0,0).
  '''

  # Create sphere
  uv, uv_weights = parametrization(p)
  grid_coor = sphere(sphere_radius, uv)

  # Get r_vectors, rotation to body 0 frame of reference if frame_body=True
  r_vectors = np.empty((lambda_blobs.size // 3, 3))
  offset = 0
  if frame_body >= 0:
    R0 = bodies[frame_body].orientation.rotation_matrix().T
    theta0 = bodies[frame_body].orientation.inverse()
    for b in bodies:
      location = np.dot(R0, (b.location - bodies[frame_body].location))
      orientation = theta0 * b.orientation
      num_blobs = b.Nblobs
      r_vectors[offset:(offset+num_blobs)] = b.get_r_vectors(location=location, orientation=orientation)
      offset += num_blobs
    lambda_blobs = lambda_blobs.reshape((lambda_blobs.size // 3, 3))
    for i in range(lambda_blobs.shape[0]):
      lambda_blobs[i] = np.dot(R0, lambda_blobs[i])
  else:    
    for b in bodies:
      location = b.location
      orientation = b.orientation
      num_blobs = b.Nblobs
      r_vectors[offset:(offset+num_blobs)] = b.get_r_vectors(location=location, orientation=orientation)
      offset += num_blobs

  # Set radius of blobs and grid nodes (= 0)
  radius_source = np.zeros(r_vectors.size // 3) 
  offset = 0
  for b in bodies:
    num_blobs = b.Nblobs
    radius_source[offset:(offset+num_blobs)] = b.blobs_radius
    offset += num_blobs
  radius_target = np.zeros(grid_coor.size // 3) 
  
  # Compute velocity field 
  mobility_vector_prod_implementation = kwargs.get('mobility_vector_prod_implementation')
  if mobility_vector_prod_implementation == 'python':
    grid_velocity = mob.mobility_vector_product_source_target_one_wall(r_vectors, 
                                                                       grid_coor, 
                                                                       lambda_blobs, 
                                                                       radius_source, 
                                                                       radius_target, 
                                                                       eta, 
                                                                       *args, 
                                                                       **kwargs) 
  elif mobility_vector_prod_implementation == 'C++':
    grid_velocity = mob.boosted_mobility_vector_product_source_target(r_vectors, 
                                                                      grid_coor, 
                                                                      lambda_blobs, 
                                                                      radius_source, 
                                                                      radius_target, 
                                                                      eta, 
                                                                      *args, 
                                                                      **kwargs)
  elif mobility_vector_prod_implementation == 'numba_no_wall':
    grid_velocity = mob.no_wall_mobility_trans_times_force_source_target_numba(r_vectors, 
                                                                               grid_coor, 
                                                                               lambda_blobs, 
                                                                               radius_source, 
                                                                               radius_target, 
                                                                               eta, 
                                                                               *args, 
                                                                               **kwargs) 
  else:
    grid_velocity = mob.single_wall_mobility_trans_times_force_source_target_pycuda(r_vectors, 
                                                                                    grid_coor, 
                                                                                    lambda_blobs, 
                                                                                    radius_source, 
                                                                                    radius_target, 
                                                                                    eta, 
                                                                                    *args, 
                                                                                    **kwargs) 
 
  # Write velocity field.
  header = 'R=' + str(sphere_radius) + ', p=' + str(p) + ', N=' + str(uv_weights.size) + ', centered body=' + str(frame_body) + ', 7 Columns: grid point (x,y,z), quadrature weight, velocity (vx,vy,vz)'
  result = np.zeros((grid_coor.shape[0], 7))
  result[:,0:3] = grid_coor
  result[:,3] = uv_weights
  grid_velocity = grid_velocity.reshape((grid_velocity.size // 3, 3)) 
  result[:,4:] = grid_velocity
  np.savetxt(output, result, header=header) 
  return

