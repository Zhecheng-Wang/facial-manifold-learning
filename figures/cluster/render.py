import os
import sys
import numpy as np
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
src_path = os.path.join(PROJ_ROOT, "src")
sys.path.append(src_path)
FIGURES_FOLDER = os.path.join(PROJ_ROOT, "figures")
FIGURE_NAME = os.path.dirname(os.path.abspath(__file__))
BLENDER_TOOLBOX_PATH = os.path.join(FIGURES_FOLDER, "BlenderToolbox")
sys.path.append(BLENDER_TOOLBOX_PATH)
import BlenderToolBox as bt
import bpy

def render_frame(vertex_scalars, meshPath, outputPath, numSamples):
    ## initialize blender
    imgRes_x = 1920 # recommend > 1080 
    imgRes_y = 1080 # recommend > 1080 
    exposure = 1.5
    use_GPU = True
    bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure, use_GPU)

    ## read mesh
    location = (2.2, 0, 1.6) # (GUI: click mesh > Transform > Location)
    rotation = (60, 0, 90) # (GUI: click mesh > Transform > Rotation)
    scale = (1.5,1.5,1.5) # (GUI: click mesh > Transform > Scale)
    mesh = bt.readMesh(meshPath, location, rotation, scale)

    ## set shading
    bpy.ops.object.shade_smooth()

    ## subdivision
    bt.subdivision(mesh, level = 1)

    ###########################################
    ## Set your material here (see other demo scripts)

    color_type = 'vertex'
    color_map = 'red'
    mesh = bt.setMeshScalars(mesh, vertex_scalars, color_map, color_type)
    meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
    bt.setMat_VColor(mesh, meshVColor)

    ## End material
    ###########################################

    ## set invisible plane (shadow catcher)
    bt.invisibleGround(shadowBrightness=0.9)

    ## set camera 
    camLocation = (3, 0, 2)
    lookAtLocation = (0,0,0.5)
    focalLength = 45 # (UI: click camera > Object Data > Focal Length)
    cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

    ## set light
    lightAngle = (6, -30, -155) 
    strength = 2
    shadowSoftness = 0.3
    sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

    ## set ambient light
    bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

    ## set gray shadow to completely white with a threshold (optional but recommended)
    bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

    # ## save blender file so that you can adjust parameters in the UI
    # bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')

    ## save rendering
    bt.renderImage(outputPath, cam)

if __name__ == "__main__":
    # path to rest pose
    BLENDSHAPES_PATH = os.path.join(PROJ_ROOT, "data", "AppleAR", "OBJs")
    rest_face_path = os.path.join(BLENDSHAPES_PATH, "Neutral.obj")
    # read color values
    path = os.path.join(FIGURES_FOLDER, FIGURE_NAME, "cluster")
    save_path = path + "_render"
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file) 
                render_file_path = os.path.join(save_path, file.replace(".npy", ".png"))
                vertex_scalars = np.load(file_path)
                render_frame(vertex_scalars, rest_face_path, render_file_path,  50)