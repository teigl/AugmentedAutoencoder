# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os
import struct
import numpy as np

def load_mtl(path):
    materials = {}
    with open(path) as file:

        material = None
        for line in file:
            tokens = line.split()
            if not len(tokens):
                continue

            key = tokens[0]
            if key == 'newmtl':
                material = tokens[1]
                materials[material] = {}
            elif not material: 
                continue
            else:
                materials[material][key] = [float(x) for x in tokens[1:]]
    return materials

def load_obj(path):
    print()
    print('Loading obj')

    vs = []
    vts = []
    vns = []

    positions = []
    normals = []
    colors = []
    indexes = []

    vertexes = {}
    next_index = 0
    ambient_colors = {}

    with open(path) as file:

        for line in file:

            tokens = line.split()
            if not len(tokens):
                continue

            key = tokens[0]
            values = tokens[1:]

            if key == 'mtllib':
                cwd = os.getcwd()
                os.chdir(os.path.dirname(path))
                materials = load_mtl(tokens[1])
                os.chdir(cwd)
                for material in materials:
                    if 'Ka' in materials[material]:
                        ambient_colors[material] = \
                            [255*x for x in materials[material]['Ka']]

            elif key == 'v':
                vs.append([float(x) for x in values])
            elif key == 'vt':
                vts.append([float(x) for x in values])
            elif key == 'vn':
                vns.append([float(x) for x in values])
            elif key == 'usemtl':
                material = values[0]
            elif key == 'f':
                
                for value in values:
                    if not value in vertexes:
                        inds = value.split("/")
                        positions.append(vs[int(inds[0]) - 1])
                        normals.append(vns[int(inds[2]) - 1])
                        if material in ambient_colors:
                            colors.append(ambient_colors[material])
                        else:
                            colors.append([160,160,160])

                        vertexes[value] = next_index
                        indexes.append(next_index)
                        next_index += 1
                    else:
                        indexes.append(vertexes[value])
        
    return (np.array(positions, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(colors, dtype=np.float32),
            np.array(indexes, dtype=np.uint32))
            
def load_ply(path):
    """
    Loads a 3D mesh model from a PLY file.

    :param path: Path to a PLY file.
    :return: The loaded model given by a dictionary with items:
    'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
    'faces' (mx3 ndarray) - the latter three are optional.
    """
    f = open(path, 'r')

    n_pts = 0
    n_faces = 0
    face_n_corners = 3 # Only triangular faces are supported
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False

    # Read header
    while True:
        line = f.readline().rstrip('\n').rstrip('\r') # Strip the newline character(s)
        if line.startswith('element vertex'):
            n_pts = int(line.split()[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith('element face'):
            n_faces = int(line.split()[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith('element'): # Some other element
            header_vertex_section = False
            header_face_section = False
        elif line.startswith('property') and header_vertex_section:
            # (name of the property, data type)
            pt_props.append((line.split()[-1], line.split()[-2]))
        elif line.startswith('property list') and header_face_section:
            elems = line.split()
            if elems[-1] == 'vertex_indices':
                # (name of the property, data type)
                face_props.append(('n_corners', elems[2]))
                for i in range(face_n_corners):
                    face_props.append(('ind_' + str(i), elems[3]))
            else:
                print(('Warning: Not supported face property: ' + elems[-1]))
        elif line.startswith('format'):
            if 'binary' in line:
                is_binary = True
        elif line.startswith('end_header'):
            break

    # Prepare data structures
    model = {}
    model['pts'] = np.zeros((n_pts, 3), float)
    if n_faces > 0:
        model['faces'] = np.zeros((n_faces, face_n_corners), float)

    pt_props_names = [p[0] for p in pt_props]
    is_normal = False
    if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
        is_normal = True
        model['normals'] = np.zeros((n_pts, 3), float)

    is_color = False
    if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
        is_color = True
        model['colors'] = np.zeros((n_pts, 3), float)

    is_texture = False
    if {'texture_u', 'texture_v'}.issubset(set(pt_props_names)):
        is_texture = True
        model['texture_uv'] = np.zeros((n_pts, 2), float)

    formats = { # For binary format
        'float': ('f', 4),
        'double': ('d', 8),
        'int': ('i', 4),
        'uchar': ('B', 1)
    }

    # Load vertices
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                      'red', 'green', 'blue', 'texture_u', 'texture_v']
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split()
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model['pts'][pt_id, 0] = float(prop_vals['x'])
        model['pts'][pt_id, 1] = float(prop_vals['y'])
        model['pts'][pt_id, 2] = float(prop_vals['z'])

        if is_normal:
            model['normals'][pt_id, 0] = float(prop_vals['nx'])
            model['normals'][pt_id, 1] = float(prop_vals['ny'])
            model['normals'][pt_id, 2] = float(prop_vals['nz'])

        if is_color:
            model['colors'][pt_id, 0] = float(prop_vals['red'])
            model['colors'][pt_id, 1] = float(prop_vals['green'])
            model['colors'][pt_id, 2] = float(prop_vals['blue'])

        if is_texture:
            model['texture_uv'][pt_id, 0] = float(prop_vals['texture_u'])
            model['texture_uv'][pt_id, 1] = float(prop_vals['texture_v'])

    # Load faces
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == 'n_corners':
                    if val != face_n_corners:
                        print('Error: Only triangular faces are supported.')
                        print(('Number of face corners: ' + str(val)))
                        exit(-1)
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split()
            for prop_id, prop in enumerate(face_props):
                if prop[0] == 'n_corners':
                    if int(elems[prop_id]) != face_n_corners:
                        print('Error: Only triangular faces are supported.')
                        print(('Number of face corners: ' + str(int(elems[prop_id]))))
                        exit(-1)
                else:
                    prop_vals[prop[0]] = elems[prop_id]

        model['faces'][face_id, 0] = int(prop_vals['ind_0'])
        model['faces'][face_id, 1] = int(prop_vals['ind_1'])
        model['faces'][face_id, 2] = int(prop_vals['ind_2'])

    f.close()

    return model