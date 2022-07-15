import numpy as np
import mcubes
  
  # Create a data volume (30 x 30 x 30)
X, Y, Z = np.mgrid[:32, :32, :32]
u = (X-15)**2 + (Y-15)**2 + (Z-15)**2 - 8**2
  
  # Extract the 0-isosurface
vertices, triangles = mcubes.marching_cubes(u, 0)
  
  # Export the result to sphere.dae
mcubes.export_mesh(vertices, triangles, "sphere.dae", "MySphere")
