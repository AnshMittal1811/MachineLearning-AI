	
import numpy as np

def export_obj(vertices, triangles, filename):
   
#    Exports a mesh in the (.obj) format.
    
    with open(filename, 'w') as fh:
        
        for v in vertices:
            fh.write("v {} {} {}\n".format(*v))
            
        for f in triangles:
            fh.write("f {} {} {}\n".format(*(f + 1)))
