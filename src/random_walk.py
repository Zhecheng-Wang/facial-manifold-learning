import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from blendshapes import *

class ManifoldExplorer:
    def __init__(self, model:CollisionBlendshapeModel, step_size=0.1):
        self.model = model
        self.step_size = step_size
        self.weights = np.zeros(self.model.weights.shape)
        self.history = []

    def valid_weights(self, weights):
        # all weights should be between -1 and 1
        return np.all(weights >= -1) and np.all(weights <= 1)

    def random_step(self):
        direction_index = np.random.randint(0, self.weights.shape[0])
        direction = np.zeros(self.weights.shape)
        direction[direction_index] = 1
        next_weights = self.weights + direction * self.step_size
        # if next weights are invalid or mesh has intersection, try another direction
        is_invalid = not self.valid_weights(next_weights)
        has_intersection = self.model.has_intersections(next_weights)
        # while not self.valid_weights(next_weights) or model.has_intersections(next_weights):
        step = 0
        while is_invalid or has_intersection:
            print(f"{step=}: {is_invalid=}, {has_intersection=}")
            step += 1
            if step >= 100:
                return None
            direction_index = np.random.randint(0, self.weights.shape[0])
            direction = np.zeros(self.weights.shape)
            direction[direction_index] = 1
            next_weights = self.weights + direction * self.step_size
            is_invalid = not self.valid_weights(next_weights)
            has_intersection = self.model.has_intersections(next_weights)
        self.weights = next_weights
        self.history.append(self.weights.copy())
        return self.model.eval(self.weights)

manifold_explorer = None
playing = False
manifold_visual = None
save_path = os.path.join(os.pardir, "output", "history.npy")

def draw():
    global playing

    if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space)):
        playing = not playing

    if playing:
        V = manifold_explorer.random_step()
        if V is None:
            playing = False
            print("Failed to find a valid direction")
        else:
            manifold_visual.update_vertex_positions(V)
        if psim.Button("Stop"):
            playing = False
    elif psim.Button("Play"):
        playing = True
    
    psim.InputText("", save_path)
    psim.SameLine()
    if psim.Button("Save"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(f"{save_path}", np.array(manifold_explorer.history))
        print(f"Saved history to {save_path}")

if __name__ == "__main__":
    # load the blendshape model
    PROJ_PATH = os.path.dirname(os.path.abspath(__file__))
    BLENDSHAPES_PATH = os.path.join(os.pardir, "data", "Apple blendshapes51 OBJs", "OBJs")
    model = load_blendshape_model(BLENDSHAPES_PATH)
    # initialize manifold explorer
    step_size = 0.1
    manifold_explorer = ManifoldExplorer(model, step_size)   
    # initialize polyscope
    ps.set_autocenter_structures(True)
    ps.set_program_name("Manifold Explorer")
    ps.set_verbosity(0)
    ps.set_SSAA_factor(3)
    ps.set_max_fps(60)
    ps.set_ground_plane_mode("none")
    ps.init()
    manifold_visual = ps.register_surface_mesh("manifold", model.eval(), model.F, smooth_shade=True, edge_width=0.5, color=(0.8, 0.8, 0.8))
    ps.set_user_callback(draw)
    ps.look_at(np.array([0, 0, 0.3]), np.mean(model.V, axis=0))
    ps.show()