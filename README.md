# Facial Manifold Exploration

### Installation

```bash
conda create -n manifold python=3.11
pip install -r requirements.txt
```

### Usage
To run the interactive viewer.
```bash
cd src
python interactive.py
```

### Web UI
To run the web-based interactive viewer, run backend server (ideally in tmux), then run web viewer:
```bash
cd src
tmux new -s backend
python server.py
```
Ctrl-b + d to detach from tmux session.

Then run the web viewer in another terminal:
```bash
cd ui
npm install
npm start
```

The web UI will be available at `http://localhost:3000`. It provides two different ways to control the face:
1. Slider-based controller with grouped blendshapes
2. Interactive 3D sphere controller that allows direct manipulation of blendshapes on the face

### Data Format

The system uses a 3D facial mesh with blendshapes for animation. Here's the data structure:

#### Base Mesh
- **Vertices**: Array of 5110 vertices, each with xyz coordinates
  - Format: `[x0,y0,z0, x1,y1,z1, ...]` (15330 values total)
  - Type: `Float32Array`
- **Faces**: Array of triangle indices
  - Format: `[i0,j0,k0, i1,j1,k1, ...]`
  - Type: `Uint32Array`

#### Blendshapes
Each blendshape contains:
- **Name**: String identifier (e.g., "faceMuscles.frontalis")
- **Vertices**: Delta displacements from base mesh
  - Format: `[dx0,dy0,dz0, dx1,dy1,dz1, ...]` (15330 values total)
  - Type: `Float32Array`
  - Each value represents the xyz displacement for that vertex
- **Center**: Average position of affected vertices
  - Format: `[x, y, z]`
- **Normal**: Average normal vector of affected region
  - Format: `[nx, ny, nz]`
- **MaxDisplacement**: Maximum magnitude of displacement

#### API Response Format
```json
{
  "baseVertices": [x0,y0,z0, x1,y1,z1, ...],
  "baseFaces": [i0,j0,k0, i1,j1,k1, ...],
  "blendshapes": [
    {
      "name": "blendshape_name",
      "vertices": [dx0,dy0,dz0, dx1,dy1,dz1, ...],
      "center": [x, y, z],
      "normal": [nx, ny, nz],
      "maxDisplacement": float
    },
    ...
  ]
}
```