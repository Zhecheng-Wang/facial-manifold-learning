from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
from utils import load_blendshape, SPDataset
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the data
blendshapes = load_blendshape(model="SP")
dataset = SPDataset()
base_vertices = blendshapes.V  # Base vertices (rest shape)
base_faces = blendshapes.F     # Base faces

@app.route('/face-data')
def get_face_data():
    # Debug: Check vertex and face data types and shapes
    print("=== Data Validation ===")
    print("Base vertices:", type(base_vertices), base_vertices.dtype, base_vertices.shape)
    print("Base faces:", type(base_faces), base_faces.dtype, base_faces.shape)
    
    # Ensure vertices are float32 and faces are uint32
    vertices_float32 = base_vertices.astype(np.float32)
    faces_uint32 = base_faces.astype(np.uint32)
    
    # Validate data
    print("\n=== Data Statistics ===")
    print("Vertices - min:", np.min(vertices_float32), "max:", np.max(vertices_float32))
    print("Faces - min:", np.min(faces_uint32), "max:", np.max(faces_uint32))
    print("Has NaN:", np.isnan(vertices_float32).any())
    
    # Check for invalid values
    nan_mask = np.isnan(vertices_float32)
    inf_mask = np.isinf(vertices_float32)
    if nan_mask.any() or inf_mask.any():
        nan_indices = np.where(nan_mask)[0]
        inf_indices = np.where(inf_mask)[0]
        print("\n=== Invalid Values Found ===")
        print(f"NaN values at indices: {nan_indices[:10]} ...")
        print(f"Inf values at indices: {inf_indices[:10]} ...")
        return jsonify({'error': 'Invalid vertex data detected'}), 400
    
    # Convert to lists with explicit type checking
    base_vertices_list = vertices_float32.tolist()
    base_faces_list = faces_uint32.tolist()
    
    # Check face indices are valid
    if np.any(faces_uint32 >= len(vertices_float32)):
        invalid_faces = np.where(faces_uint32 >= len(vertices_float32))[0]
        print("\n=== Invalid Face Indices Found ===")
        print(f"Face indices exceeding vertex count at: {invalid_faces[:10]} ...")
        return jsonify({'error': 'Invalid face indices detected'}), 400
    
    # Prepare response data
    response_data = {
        'baseVertices': base_vertices_list,
        'baseFaces': base_faces_list,
        'numVertices': int(base_vertices.shape[0]),  # Send actual number of vertices
        'numFaces': int(base_faces.shape[0]),        # Send actual number of faces
        'blendshapes': []
    }
    
    # Prepare blendshape data
    for i, name in enumerate(blendshapes.names):
        # Get the blendshape vertices (difference from base)
        delta = blendshapes.delta[i].astype(np.float32)  # Ensure float32
        
        # Get the full blendshape vertices (not just magnitudes)
        full_delta = blendshapes.blendshapes[i].astype(np.float32)  # This should be (5110, 3)
        
        # Flatten the full delta array in row-major order [x0,y0,z0, x1,y1,z1, ...]
        vertices = full_delta.reshape(-1).tolist()  # Now has 15330 values
        
        # Calculate center point (average of non-zero vertices)
        non_zero_mask = np.any(blendshapes.blendshapes[i] != 0, axis=1)
        if np.any(non_zero_mask):
            center = np.mean(base_vertices[non_zero_mask], axis=0).astype(np.float32)
        else:
            center = np.zeros(3, dtype=np.float32)
        
        # Calculate normal
        if np.any(non_zero_mask):
            affected_faces = np.any(np.isin(base_faces, np.where(non_zero_mask)[0]), axis=1)
            if np.any(affected_faces):
                face_vertices = base_vertices[base_faces[affected_faces]]
                v1 = face_vertices[:, 1] - face_vertices[:, 0]
                v2 = face_vertices[:, 2] - face_vertices[:, 0]
                face_normals = np.cross(v1, v2)
                face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)
                normal = np.mean(face_normals, axis=0).astype(np.float32)
            else:
                normal = np.array([0, 1, 0], dtype=np.float32)
        else:
            normal = np.array([0, 1, 0], dtype=np.float32)
        
        # Calculate maximum displacement
        max_displacement = float(np.max(np.abs(delta)))
        
        response_data['blendshapes'].append({
            'name': name,
            'vertices': vertices,  # Now has full xyz deltas
            'center': center.tolist(),
            'normal': normal.tolist(),
            'maxDisplacement': max_displacement
        })
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(port=5001, debug=True) 