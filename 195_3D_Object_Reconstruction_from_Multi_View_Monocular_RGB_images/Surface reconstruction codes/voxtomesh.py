from marching_cubes import march
from numpy import load
from pyqtgraph.opengl import GLViewWidget, MeshData
from pyqtgraph.opengl.items.GLMeshItem import GLMeshItem

from PyQt4.QtGui import QApplication

volume = load("test/data/input/sample.npy")  # 128x128x128 uint8 volume

# extract the mesh where the values are larger than or equal to 1
# everything else is ignored
vertices, normals, faces = march(volume, 0)  # zero smoothing rounds
smooth_vertices, smooth_normals, faces = march(volume, 4)  # 4 smoothing rounds

app = QApplication([])
view = GLViewWidget()

mesh = MeshData(vertices / 100, faces)  # scale down - because camera is at a fixed position 
# or mesh = MeshData(smooth_vertices / 100, faces)
mesh._vertexNormals = normals
# or mesh._vertexNormals = smooth_normals

item = GLMeshItem(meshdata=mesh, color=[1, 0, 0, 1], shader="normalColor")

view.addItem(item)
view.show()
app.exec_()

