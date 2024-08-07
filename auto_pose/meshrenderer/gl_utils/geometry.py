# -*- coding: utf-8 -*-
import os
import numpy as np
import hashlib

import pyassimp
import pyassimp.postprocess
import progressbar



def load(filename):
    scene = pyassimp.load(filename, processing=pyassimp.postprocess.aiProcess_GenUVCoords|pyassimp.postprocess.aiProcess_Triangulate )
    mesh = scene.meshes[0]
    return mesh.vertices, mesh.normals, mesh.texturecoords[0,:,:2]

def load_meshes(obj_files, vertex_tmp_store_folder):
    bar = progressbar.ProgressBar()
    meshes = []
    for path in bar(obj_files):
        hashed_file_name = hashlib.md5((''.join(path) + 'load_meshes').encode('utf-8') ).hexdigest() + '.npz'

        out_file = os.path.join(vertex_tmp_store_folder, hashed_file_name)
        if os.path.exists(out_file):
            file = np.load(out_file)
            meshes.append((file['positions'],
                           file['normals'],
                           file['colors'],
                           file['indexes']))
        else:
            meshes.append(load_mesh(path))
            np.savez(out_file, 
                     positions=meshes[-1][0],
                     normals=meshes[-1][1],
                     colors=meshes[-1][2],
                     indexes=meshes[-1][3])
    return meshes

def load_mesh(path):
    if '.obj' in path:
        from . import inout
        mesh = inout.load_obj(path)

    elif '.ply' in path:
        model = inout.load_ply(path)

        positions = np.array(model['pts'] ).astype(np.float32)
        normals = np.array(model['normals']).astype(np.float32)
        faces = np.array(model['faces']).astype(np.uint32).flatten()

        if 'colors' in model:
            colors = np.array(model['colors']).astype(np.uint32)
        else:
            colors = np.empty_like(positions, dtype=np.float32)
            colors[:,:] = 160

        mesh = (positions,normals,colors,faces)

    else:
        raise ValueError('Only obj and ply are supported')
        with pyassimp.load(path, pyassimp.postprocess.aiProcess_Triangulate) as scene:

            positions = []
            normals = []
            faces = []

            mesh = scene.meshes[0]
            for mesh in scene.meshes:
                positions.extend(mesh.vertices)
                normals.extend(mesh.normals)
                faces.extend(mesh.faces)
            return np.array(positions),np.array(normals),np.array(faces)

    return mesh

def calc_normals(vertices):
    normals = np.empty_like(vertices)
    N = vertices.shape[0]
    for i in range(0, N-1, 3):
        v1 = vertices[i]
        v2 = vertices[i+1]
        v3 = vertices[i+2]
        normal = np.cross(v2-v1, v3-v1)
        norm = np.linalg.norm(normal)
        normal = np.zeros(3) if norm == 0 else normal / norm
        normals[i] = normal
        normals[i+1] = normal
        normals[i+2] = normal
    return normals

# def calc_normals_with_faces(vertices, faces):
#     normals = np.empty_like(vertices)
#     N = vertices.shape[0]
#     for face in faces:
#         v1 = vertices[face[0]]
#         v2 = vertices[face[1]]
#         v3 = vertices[face[2]]
#         normal = np.cross(v2-v1, v3-v1)
#         norm = np.linalg.norm(normal)
#         normal = np.zeros(3) if norm == 0 else normal / norm
#         normals[i] = normal
#         normals[i+1] = normal
#         normals[i+2] = normal
#     return normals

# src: https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/6.pbr/2.2.2.ibl_specular_textured/ibl_specular_textured.cpp
def sphere(x_segments, y_segments):

    N = (x_segments+1) * (y_segments+1)
    positions = np.empty((N, 3), dtype=np.float32)
    uv = np.empty((N, 2), dtype=np.float32)
    normals = np.empty((N, 3), dtype=np.float32)

    i = 0
    for y in range(y_segments+1):
        for x in range(x_segments+1):
            xSegment = float(x) / float(x_segments)
            ySegment = float(y) / float(y_segments)
            xPos = np.cos(xSegment * 2.0 * np.pi) * np.sin(ySegment * np.pi)
            yPos = np.cos(ySegment * np.pi)
            zPos = np.sin(xSegment * 2.0 * np.pi) * np.sin(ySegment * np.pi)

            positions[i] = (xPos, yPos, zPos)
            uv[i] = (xSegment, ySegment)
            normals[i] = (xPos, yPos, zPos)
            i += 1

    indices = []
    oddRow = False
    for y in range(y_segments):
        if not oddRow:
            for x in range(x_segments+1):
                indices.append(y     * (x_segments + 1) + x)
                indices.append((y+1) * (x_segments + 1) + x)
        else:
            for x in reversed(range(x_segments+1)):
                indices.append((y+1) * (x_segments + 1) + x)
                indices.append(y     * (x_segments + 1) + x)
        oddRow = not oddRow
    indices = np.array(indices, dtype=np.uint32)

    return positions, uv, normals, indices

def cube():
    positions = np.array([
        [-1.0, -1.0, -1.0],
        [1.0,  1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0,  1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [1.0, -1.0,  1.0],
        [1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0],
        [-1.0, -1.0,  1.0],
        [-1.0,  1.0,  1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [-1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0],
        [1.0, -1.0, -1.0],
        [1.0,  1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0,  1.0,  1.0],
        [1.0, -1.0,  1.0],
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0,  1.0],
        [1.0, -1.0,  1.0],
        [-1.0, -1.0,  1.0],
        [-1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [1.0,  1.0 , 1.0],
        [1.0,  1.0, -1.0],
        [1.0,  1.0,  1.0],
        [-1.0,  1.0, -1.0],
        [-1.0,  1.0,  1.0]], dtype=np.float32)

    normals = np.array([
         [0.0,  0.0, -1.0],
         [0.0,  0.0, -1.0],
         [0.0,  0.0, -1.0],
         [0.0,  0.0, -1.0],
         [0.0,  0.0, -1.0],
         [0.0,  0.0, -1.0],
         [0.0,  0.0,  1.0],
         [0.0,  0.0,  1.0],
         [0.0,  0.0,  1.0],
         [0.0,  0.0,  1.0],
         [0.0,  0.0,  1.0],
         [0.0,  0.0,  1.0],
        [-1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
         [1.0,  0.0,  0.0],
         [1.0,  0.0,  0.0],
         [1.0,  0.0,  0.0],
         [1.0,  0.0,  0.0],
         [1.0,  0.0,  0.0],
         [1.0,  0.0,  0.0],
         [0.0, -1.0,  0.0],
         [0.0, -1.0,  0.0],
         [0.0, -1.0,  0.0],
         [0.0, -1.0,  0.0],
         [0.0, -1.0,  0.0],
         [0.0, -1.0,  0.0],
         [0.0,  1.0,  0.0],
         [0.0,  1.0,  0.0],
         [0.0,  1.0,  0.0],
         [0.0,  1.0,  0.0],
         [0.0,  1.0,  0.0],
         [0.0,  1.0,  0.0]], dtype=np.float32)

    uv = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]], dtype=np.float32)
    return positions, uv, normals

def cube2(min, max):
    positions = np.array([
        [-1.0, -1.0, -1.0],
        [1.0,  1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0,  1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [1.0, -1.0,  1.0],
        [1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0],
        [-1.0, -1.0,  1.0],
        [-1.0,  1.0,  1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [-1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0],
        [1.0, -1.0, -1.0],
        [1.0,  1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0,  1.0,  1.0],
        [1.0, -1.0,  1.0],
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0,  1.0],
        [1.0, -1.0,  1.0],
        [-1.0, -1.0,  1.0],
        [-1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [1.0,  1.0 , 1.0],
        [1.0,  1.0, -1.0],
        [1.0,  1.0,  1.0],
        [-1.0,  1.0, -1.0],
        [-1.0,  1.0,  1.0]], dtype=np.float32)

    normals = np.array([
         [0.0,  0.0, -1.0],
         [0.0,  0.0, -1.0],
         [0.0,  0.0, -1.0],
         [0.0,  0.0, -1.0],
         [0.0,  0.0, -1.0],
         [0.0,  0.0, -1.0],
         [0.0,  0.0,  1.0],
         [0.0,  0.0,  1.0],
         [0.0,  0.0,  1.0],
         [0.0,  0.0,  1.0],
         [0.0,  0.0,  1.0],
         [0.0,  0.0,  1.0],
        [-1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
         [1.0,  0.0,  0.0],
         [1.0,  0.0,  0.0],
         [1.0,  0.0,  0.0],
         [1.0,  0.0,  0.0],
         [1.0,  0.0,  0.0],
         [1.0,  0.0,  0.0],
         [0.0, -1.0,  0.0],
         [0.0, -1.0,  0.0],
         [0.0, -1.0,  0.0],
         [0.0, -1.0,  0.0],
         [0.0, -1.0,  0.0],
         [0.0, -1.0,  0.0],
         [0.0,  1.0,  0.0],
         [0.0,  1.0,  0.0],
         [0.0,  1.0,  0.0],
         [0.0,  1.0,  0.0],
         [0.0,  1.0,  0.0],
         [0.0,  1.0,  0.0]], dtype=np.float32)

    uv = np.array([
            [min, min],
            [max, max],
            [max, min],
            [max, max],
            [min, min],
            [min, max],
            [min, min],
            [max, min],
            [max, max],
            [max, max],
            [min, max],
            [min, min],
            [max, min],
            [max, max],
            [min, max],
            [min, max],
            [min, min],
            [max, min],
            [max, min],
            [min, max],
            [max, max],
            [min, max],
            [max, min],
            [min, min],
            [min, max],
            [max, max],
            [max, min],
            [max, min],
            [min, min],
            [min, max],
            [min, max],
            [max, min],
            [max, max],
            [max, min],
            [min, max],
            [min, min]], dtype=np.float32)
    return positions, uv, normals


def quad(reverse_uv=False):
    positions = np.array([
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0] ], dtype=np.float32)
    if reverse_uv:
        uv = np.array([
         [0.0, 0.0],
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 1.0]], dtype=np.float32)
    else:
        uv = np.array([
         [0.0, 1.0],
         [0.0, 0.0],
         [1.0, 1.0],
         [1.0, 0.0]], dtype=np.float32)
    return positions, uv


cube_vertices_texture = np.array([
    -0.5, -0.5, -0.5,  0.0, 0.0,
     0.5, -0.5, -0.5,  1.0, 0.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
    -0.5,  0.5, -0.5,  0.0, 1.0,
    -0.5, -0.5, -0.5,  0.0, 0.0,

    -0.5, -0.5,  0.5,  0.0, 0.0,
     0.5, -0.5,  0.5,  1.0, 0.0,
     0.5,  0.5,  0.5,  1.0, 1.0,
     0.5,  0.5,  0.5,  1.0, 1.0,
    -0.5,  0.5,  0.5,  0.0, 1.0,
    -0.5, -0.5,  0.5,  0.0, 0.0,

    -0.5,  0.5,  0.5,  1.0, 0.0,
    -0.5,  0.5, -0.5,  1.0, 1.0,
    -0.5, -0.5, -0.5,  0.0, 1.0,
    -0.5, -0.5, -0.5,  0.0, 1.0,
    -0.5, -0.5,  0.5,  0.0, 0.0,
    -0.5,  0.5,  0.5,  1.0, 0.0,

     0.5,  0.5,  0.5,  1.0, 0.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
     0.5, -0.5, -0.5,  0.0, 1.0,
     0.5, -0.5, -0.5,  0.0, 1.0,
     0.5, -0.5,  0.5,  0.0, 0.0,
     0.5,  0.5,  0.5,  1.0, 0.0,

    -0.5, -0.5, -0.5,  0.0, 1.0,
     0.5, -0.5, -0.5,  1.0, 1.0,
     0.5, -0.5,  0.5,  1.0, 0.0,
     0.5, -0.5,  0.5,  1.0, 0.0,
    -0.5, -0.5,  0.5,  0.0, 0.0,
    -0.5, -0.5, -0.5,  0.0, 1.0,

    -0.5,  0.5, -0.5,  0.0, 1.0,
     0.5,  0.5, -0.5,  1.0, 1.0,
     0.5,  0.5,  0.5,  1.0, 0.0,
     0.5,  0.5,  0.5,  1.0, 0.0,
    -0.5,  0.5,  0.5,  0.0, 0.0,
    -0.5,  0.5, -0.5,  0.0, 1.0
], dtype=np.float32)

def quad_bitangent():
    verts = np.array([   [-1.0, 1.0, 0.0],
                            [-1.0, -1.0, 0.0],
                            [1.0, -1.0, 0.0],
                            [1.0, 1.0, 0.0]], dtype=np.float32)
    uv = np.array([
         [0.0, 1.0],
         [0.0, 0.0],
         [1.0, 0.0],
         [1.0, 1.0]], dtype=np.float32)

    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    edge1 = verts[1] - verts[0]
    edge2 = verts[2] - verts[0]
    deltaUV1 = uv[1] - uv[0]
    deltaUV2 = uv[2] - uv[0]

    f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])

    tangent1 = f * np.array([   deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0],
                                deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1],
                                deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2]], dtype=np.float32)
    tangent1 /= np.linalg.norm(tangent1)

    bitangent1 = f * np.array([ -deltaUV2[0] * edge1[0] + deltaUV1[0] * edge2[0],
                                -deltaUV2[0] * edge1[1] + deltaUV1[0] * edge2[1],
                                -deltaUV2[0] * edge1[2] + deltaUV1[0] * edge2[2]], dtype=np.float32)
    bitangent1 /= np.linalg.norm(bitangent1)

    edge1 = verts[2] - verts[0];
    edge2 = verts[3] - verts[0];
    deltaUV1 = uv[2] - uv[0];
    deltaUV2 = uv[3] - uv[0];

    f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])

    tangent2 = f * np.array([   deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0],
                                deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1],
                                deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2]], dtype=np.float32)
    tangent2 /= np.linalg.norm(tangent2)

    bitangent2 = f * np.array([ -deltaUV2[0] * edge1[0] + deltaUV1[0] * edge2[0],
                                -deltaUV2[0] * edge1[1] + deltaUV1[0] * edge2[1],
                                -deltaUV2[0] * edge1[2] + deltaUV1[0] * edge2[2]], dtype=np.float32)
    bitangent2 /= np.linalg.norm(bitangent2)


    return np.array([ verts[0][0], verts[0][1], verts[0][2], uv[0][0], uv[0][1], normal[0], normal[1], normal[2], tangent1[0], tangent1[1], tangent1[2], bitangent1[0], bitangent1[1], bitangent1[2],
                      verts[1][0], verts[1][1], verts[1][2], uv[1][0], uv[1][1], normal[0], normal[1], normal[2], tangent1[0], tangent1[1], tangent1[2], bitangent1[0], bitangent1[1], bitangent1[2],
                      verts[2][0], verts[2][1], verts[2][2], uv[2][0], uv[2][1], normal[0], normal[1], normal[2], tangent1[0], tangent1[1], tangent1[2], bitangent1[0], bitangent1[1], bitangent1[2],
                      verts[0][0], verts[0][1], verts[0][2], uv[0][0], uv[0][1], normal[0], normal[1], normal[2], tangent2[0], tangent2[1], tangent2[2], bitangent2[0], bitangent2[1], bitangent2[2],
                      verts[2][0], verts[2][1], verts[2][2], uv[2][0], uv[2][1], normal[0], normal[1], normal[2], tangent2[0], tangent2[1], tangent2[2], bitangent2[0], bitangent2[1], bitangent2[2],
                      verts[3][0], verts[3][1], verts[3][2], uv[3][0], uv[3][1], normal[0], normal[1], normal[2], tangent2[0], tangent2[1], tangent2[2], bitangent2[0], bitangent2[1], bitangent2[2]], dtype=np.float32)



quad_vert_tex_normal_tangent_bitangent = np.array([     -1, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
                                                         1, -1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
                                                         1,  1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0,
                                                        -1,  1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0], dtype=np.float32)