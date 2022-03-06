import os
import numpy as np

from tqdm.auto import tqdm

cam_K = {
    'fx': 481.2,
    'fy': -480.0,
    'cx': 319.5,
    'cy': 239.5
}

def toImagePlane(depth, x, y):

    # taken from the c++ code implementation at https://www.doc.ic.ac.uk/~ahanda/VaFRIC/codes.html in file VaFRIC.cpp#getEuclidean2PlanarDepth
    x_plane = (x - cam_K['cx']) / cam_K['fx']
    y_plane = (y - cam_K['cy']) / cam_K['fy']
    z = depth / np.sqrt(x_plane ** 2 + y_plane ** 2 + 1)

    return z

def main(input):
    depth_files = sorted([os.path.join(input, f) for f in os.listdir(input) if
                          f.endswith('.depth.npy') and not f.endswith('.gl.depth.npy')])
    gl_depth_files = sorted([os.path.join(input, f) for f in os.listdir(input) if
                          f.endswith('.gl.depth.npy')])

    if len(depth_files) != len(gl_depth_files):
        raise ValueError("Number of .depth.npy files ({}) not equal to number of .gl.depth.npy files ({}).".format(len(depth_files), len(gl_depth_files)))

    print("Comparing {} depth files".format(len(depth_files)))

    for depth_file, gl_depth_file in tqdm(zip(depth_files, gl_depth_files)):

        depth = np.load(os.path.join(input, depth_file))
        # CONVERT ICL DEPTH AS MENTIONED IN THEIR DATASET SPECIFICATION
        depth = np.fromfunction(lambda y, x: toImagePlane(depth, x, y), depth.shape, dtype=depth.dtype)

        gl_depth = np.load(os.path.join(input, gl_depth_file))

        is_close = np.allclose(depth, gl_depth)

        if not is_close:
            min_icl = np.min(depth)
            max_icl = np.max(depth)

            min_gl = np.min(gl_depth)
            max_gl = np.max(gl_depth)

            mask = np.isclose(depth, gl_depth)

            print("ICL [{},{}] vs. GL [{},{}]".format(min_icl, max_icl, min_gl, max_gl))

            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=depth.shape)
            fig.suptitle("THESE ARE NOT EQUAL")
            fig.add_subplot(1, 3, 1)
            plt.title("ICL DEPTH")
            plt.imshow(depth)
            fig.add_subplot(1, 3, 2)
            plt.title("GL DEPTH")
            plt.imshow(gl_depth)
            fig.add_subplot(1, 3, 3)
            plt.title("EQUAL AREAS")
            plt.imshow(mask)
            plt.show()




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compares .depth.npy files with .gl.depth.npy files and outputs if they are equal (allclose)')
    parser.add_argument('--input', metavar='path', required=True,
                        help='path/to/input/directory')

    args = parser.parse_args()
    main(args.input)
