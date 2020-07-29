import pyrender2
import numpy as np
import trimesh
from PIL import Image

# mesh = trimesh.load('0_0_20.ply')
mesh = trimesh.load('./example/fuze.obj')

# print (mesh)

vertices = mesh.vertices
faces = mesh.faces

vertices = np.array(vertices, dtype=np.float).transpose([1, 0])
faces = np.array(faces, dtype=np.float).transpose([1, 0])
# print (vertices)
vertices = vertices.copy(order='C')
faces = faces.copy(order='C')
# print (vertices)

# vertices: a 3xN double numpy array
# faces: a 3xN double array (indices of vertices array)
# cam_intr: (fx, fy, px, py) double vector
# img_size: (width, height) int vector 

# N = 10
# vertices = np.ones((3, 10), dtype=np.float)
# faces = np.ones((3, 10), dtype=np.float)
fx = fy = 50
cam_intr = np.array([fx, fy, 160, 120], dtype=np.float)
img_size = np.array([320, 240], dtype=np.int32)

print (vertices.shape, faces.shape, cam_intr.shape, img_size.shape)

depth, mask, img = pyrender2.render(vertices, faces, cam_intr, img_size)
# print (depth.shape)

print (img.shape)
print (depth)
depth = Image.fromarray(np.uint8(depth*255).clip(0,255), 'L')

# img = Image.fromarray(np.uint8(img*255), 'RGB')

depth.save('./example/depth.png')
# img.save('img.png')


