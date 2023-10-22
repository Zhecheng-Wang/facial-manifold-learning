import pygame
import random
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np


def lookAt(eye, center, up):
    f = (center - eye)
    f /= np.linalg.norm(f)
    right = np.cross(up, f)
    right /= np.linalg.norm(right)
    u = np.cross(f, right)
    u /= np.linalg.norm(u)
    m = np.eye(4)
    m[0, :-1] = right
    m[1, :-1] = u
    m[2, :-1] = -f
    m[:-1, -1] = -eye

    return m.T

class BasicBlendshapeModel:
    def __init__(self):
        self.name_map = {'eyeblink_r': 'eyeblinkright', 'eyelookdown_r': 'eyelookdownright', 'eyelookin_r': 'eyelookinright', 'eyelookout_r': 'eyelookoutright', 'eyelookup_r': 'eyelookupright', 'eyesquint_r': 'eyesquintright', 'eyewide_r': 'eyewideright', 'eyeblink_l': 'eyeblinkleft', 'eyelookdown_l': 'eyelookdownleft', 'eyelookin_l': 'eyelookinleft', 'eyelookout_l': 'eyelookoutleft', 'eyelookup_l': 'eyelookupleft', 'eyesquint_l': 'eyesquintleft', 'eyewide_l': 'eyewideleft', 'jawforward': 'jawforward', 'jawright': 'jawright', 'jawleft': 'jawleft', 'jawopen': 'jawopen', 'mouthclose': 'mouthclose', 'mouthfunnel': 'mouthfunnel', 'mouthpucker': 'mouthpucker', 'mouthright': 'mouthright', 'mouthleft': 'mouthleft', 'mouthsmile_r': 'mouthsmileright', 'mouthsmile_l': 'mouthsmileleft', 'mouthfrown_r': 'mouthfrownright', 'mouthfrown_l': 'mouthfrownleft', 'mouthdimple_r': 'mouthdimpleright', 'mouthdimple_l': 'mouthdimpleleft', 'mouthstretch_r': 'mouthstretchright', 'mouthstretch_l': 'mouthstretchleft', 'mouthrolllower': 'mouthrolllower', 'mouthrollupper': 'mouthrollupper', 'mouthshruglower': 'mouthshruglower', 'mouthshrugupper': 'mouthshrugupper', 'mouthpress_r': 'mouthpressright', 'mouthpress_l': 'mouthpressleft', 'mouthlowerdown_r': 'mouthlowerdownright', 'mouthlowerdown_l': 'mouthlowerdownleft', 'mouthupperup_r': 'mouthupperupright', 'mouthupperup_l': 'mouthupperupleft', 'browdown_r': 'browdownright', 'browdown_l': 'browdownleft', 'browinnerup': 'browinnerup', 'browouterup_r': 'browouterupright', 'browouterup_l': 'browouterupleft', 'cheekpuff': 'cheekpuff', 'cheeksquint_r': 'cheeksquintright', 'cheeksquint_l': 'cheeksquintleft', 'nosesneer_r': 'nosesneerright', 'nosesneer_l': 'nosesneerleft'}
        self.neutral_mesh = None
        self.blendshape_mesh = {}
        self.weight = {}
        self.translation = np.array([0, 0, 0])
        self.scale_factor = 1

    def translate(self, delta_pos):
        self.translation += delta_pos

    def scale(self, scaling=1):
        self.scale_factor = scaling

    def eval(self):
        out_vers = self.neutral_mesh["vertices"].copy()
        
        for i in self.weight:
            shape_i = self.blendshape_mesh[i]["vertices"]
            out_vers += shape_i * self.weight[i]
        out_vers = out_vers * self.scale_factor + np.expand_dims(self.translation, axis=0)
        out_faces = self.neutral_mesh["faces"]
        return {"vertices":out_vers, "faces":out_faces}
    def facing_dire(self):
        facing_dir = np.mean(self.neutral_mesh.face_normals, axis=0)
        facing_dir /= np.linalg.norm(facing_dir)
        return facing_dir
def load_blendshape_model(path, model):
    neutral_path = os.path.join(path, "Neutral.obj")
    blendshape_paths = []
    folder_content = os.listdir(path)
    blendshape_name = []
    for b in folder_content:
        if b[-4:] != "gltf" and b[:7] != "Neutral":
            blendshape_paths.append(os.path.join(path, b))
            blendshape_name.append(b.split(".")[0].lower())
    model.neutral_mesh = load_obj(neutral_path)
    for i in range(0, len(blendshape_paths)):
        model.blendshape_mesh[blendshape_name[i]] = load_obj(blendshape_paths[i])
        model.blendshape_mesh[blendshape_name[i]]["vertices"] -= model.neutral_mesh["vertices"]
    model.weights = {}
    for i in model.blendshape_mesh:
        model.weight[i] = 0
    return model
def load_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('v '):
                vertex = line.split()[1:]
                vertices.append(list(map(float, vertex)))
            elif line.startswith('f '):
                face = line.split()[1:]
                faces.append(list(map(int, face)))
    return {'vertices': np.array(vertices), 'faces': np.array(faces)}
def draw_model(model):
    glBegin(GL_TRIANGLES)
    for face in model['faces']:
        for vertex_index in face:
            glVertex3fv(model['vertices'][vertex_index - 1])
    
    glEnd()
def is_intersecting(current_blend_weights, blendshape):
    # Replace this with your implementation to check if the mesh is self-intersecting
    return False
def draw_mesh_edges(vertices, faces):
    glColor3f(1, 0, 0)  # Set edge color to white
    glBegin(GL_LINES) 
    for face in faces:
        for i in range(len(face)):
            v1 = vertices[face[i]-1]
            v2 = vertices[face[(i + 1) % len(face)]-1]
            glVertex3fv(v1)
            glVertex3fv(v2)
    glEnd()
def generate_direction(blend_weight_keys, current_blend_weights, current_direction, current_speed, boundary_blendweights, blendshape, holding, intersecting):
    if not holding and not intersecting:
        # Replace this with your implementation to generate a new direction
        return current_direction
    else:
        direction_arr = [random.uniform(-1, 1) for _ in blend_weight_keys]
        direction_dict = {}
        for i in range(0, len(blend_weight_keys)):
            direction_dict[blend_weight_keys[i]] = direction_arr[i]
        return direction_dict

def generate_speed(current_blend_weights, current_direction, current_speed, boundary_blendweights, blendshape, holding, intersecting):
    # Replace this with your implementation to generate a new speed
    return 0.01

# Initialize Pygame
pygame.init()
display = (800, 600)
aspect_ratio = float(display[0]) / float(display[1])
display_surf = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
# set up perspective and camera position
glTranslatef(0, 0, -6) # the camera parameters are (left right, up down, forward backward)
# Initialize state variables
boundary_blendweights = []
current_speed = 0.3
# load the blendshape model
model = BasicBlendshapeModel()
model = load_blendshape_model("/Users/evanpan/Documents/GitHub/staggered_face/data/Apple blendshapes51 OBJs/OBJs", model)
current_blend_weights = {}  # Replace with your initial blend weights
blend_weight_keys = list(model.weight.keys()) # to ensure the dict is ordered
for i in blend_weight_keys:
    current_blend_weights[i] = 0
current_direction = {}
for i in blend_weight_keys:
    current_direction[i] = 0
model.scale(5)
model.translate(np.array([0, 0, 0]))
# Main loop for each timestamp
holding = False
intersecting = False


# Modify the view using a viewing matrix
camera_position = np.array([0.0, 0, 0])       # Camera position
center = np.array([0, 0, -2])   # Point to look at
up = np.array([0, 1.0, 0])        # Up direction
# Compute the view matrix
view_matrix = lookAt(camera_position, center, up)
# Apply the view matrix
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()
glLoadMatrixf(view_matrix)
while True:  # Replace num_timestamps with the actual number of timestamps
    holding = False
    intersecting = is_intersecting(current_blend_weights, model)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == KEYDOWN and event.key == K_SPACE:
            holding = True
    new_direction = generate_direction(blend_weight_keys, current_blend_weights, current_direction, current_speed, boundary_blendweights, model, holding, intersecting)
    new_speed = generate_speed(current_blend_weights, current_direction, current_speed, boundary_blendweights, model, holding, intersecting)
    if holding or intersecting:
        boundary_blendweights.append(current_blend_weights)
    current_speed = new_speed
    current_direction = new_direction
    for i in blend_weight_keys:
        current_blend_weights[i] = current_direction[i] * current_speed + current_blend_weights[i]
        if current_blend_weights[i] > 1:
            current_blend_weights[i] = 1
        if current_blend_weights[i] < 0:
            current_blend_weights[i] = 0
    # Display the blendshape with current_blend_weights using Pygame
    model.weight = current_blend_weights
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPushMatrix()
    # glTranslatef(*model.translation)
    # glScalef(model.scale_factor, model.scale_factor, model.scale_factor)
    mesh = model.eval()
    # draw_model(mesh)
    draw_mesh_edges(mesh["vertices"], mesh["faces"])
    glPopMatrix()
    pygame.display.flip()

# Quit Pygame when done
pygame.quit()
