import numpy as np
from blendshapes import BasicBlendshapes

def compute_ruzicka_similarity(blendshapes:BasicBlendshapes):
    """compute Ruzicka similarity between blendshapes
    Args:
        blendshapes (BasicBlendshapes): blendshape model
    Returns:
        similarity (np.ndarray): similarity matrix
    """
    # number of blendshapes
    n_blendshapes = len(blendshapes)
    # compute displacement magnitude for each blendshape
    blendshapes_norm = np.linalg.norm(blendshapes.blendshapes, axis=2)
    # Ruzicka similarity
    similarity = np.zeros((n_blendshapes, n_blendshapes))
    for i in range(n_blendshapes):
        similarity[i, i] = 1.0
        for j in range(i+1, n_blendshapes):
            similarity[i, j] = np.sum(np.minimum(blendshapes_norm[i], blendshapes_norm[j])) / np.sum(np.maximum(blendshapes_norm[i], blendshapes_norm[j]))
            similarity[j, i] = similarity[i, j]
    return similarity

def compute_maximum_deformation_position_similarity(blendshapes:BasicBlendshapes):
    n_blendshapes = len(blendshapes)
    blend_shape_maximum_deformation_position = np.mean(blendshapes.blendshapes + np.expand_dims(blendshapes.V, axis=0), axis=1)
    # reflected
    reflected_blend_shape_maximum_deformation_position = blend_shape_maximum_deformation_position.copy()
    reflected_blend_shape_maximum_deformation_position[:, 1] = -reflected_blend_shape_maximum_deformation_position[:, 1]
    # compluete similarity matrix
    similarity = np.zeros((n_blendshapes, n_blendshapes))
    for i in range(n_blendshapes):
        similarity[i, i] = 1.0
        for j in range(i+1, n_blendshapes):
            similarity_ij = np.linalg.norm(blend_shape_maximum_deformation_position[i] - blend_shape_maximum_deformation_position[j])
            # similarity_ij_reflected = np.linalg.norm(blend_shape_maximum_deformation_position[i] - reflected_blend_shape_maximum_deformation_position[j])
            # similarity[i, j] = np.max([similarity_ij, similarity_ij_reflected])
            similarity[i, j] = similarity_ij
            similarity[j, i] = similarity[i, j]
    similarity = similarity/np.max(similarity)
    return similarity

def compute_jaccard_similarity(blendshapes:BasicBlendshapes, threshold=0.1, sym=False):
    """compute Jaccard similarity between blendshapes
    Args:
        blendshapes (BasicBlendshapes): blendshape model
        threshold (float, optional): threshold for activating a vertex. Defaults to 0.1.
    Returns:
        similarity (np.ndarray): similarity matrix
    """
    # number of blendshapes
    n_blendshapes = len(blendshapes)
    # compute displacement magnitude for each blendshape
    if sym:
        reflected_bs = blendshapes.copy()
        reflected_bs[:, :, 0] *= -1
        blendshapes_norm = np.linalg.norm((blendshapes.blendshapes + reflected_bs)/2, axis=2)
    else:
        blendshapes_norm = np.linalg.norm(blendshapes.blendshapes, axis=2)
    blendshapes_activation = blendshapes_norm > threshold
    # Jaccard similarity
    similarity = np.zeros((n_blendshapes, n_blendshapes))
    for i in range(n_blendshapes):
        similarity[i, i] = 1.0
        for j in range(i+1, n_blendshapes):
            if not sym:
                similarity[i, j] = np.sum(blendshapes_activation[i] & blendshapes_activation[j]) / np.sum(blendshapes_activation[i] | blendshapes_activation[j])
            else:
                similarity = np.sum(blendshapes_activation[i] & blendshapes_activation[j]) / np.sum(blendshapes_activation[i] | blendshapes_activation[j])
                # sym_similarity = np.sum(blendshapes_activation[i] & blendshapes_activation[j]) / np.sum(blendshapes_activation[i] | blendshapes_activation[j]
            similarity[j, i] = similarity[i, j]
    return similarity
