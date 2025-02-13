import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display

def update_visualization(bone_index, mesh, skinning_weights_np, image_widget, side_view = widgets.fixed(None)):
    """
    Update the mesh vertex colors based on the skinning weights for the given bone index,
    render the scene to a PNG image, and update the Image widget.
    """
    # Extract the skinning weights for the selected bone.
    weights = skinning_weights_np[:, bone_index]  # shape: (1014,)

    # Normalize weights to [0, 1]
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    # Map normalized weights to colors using the colormap.
    cmap = plt.get_cmap('seismic')
    vertex_colors = cmap(weights_norm)[:, :3]  # Take only RGB

    # Convert to 8-bit integers.
    vertex_colors = (vertex_colors * 255).astype(np.uint8)

    # Update the mesh vertex colors.
    mesh.visual.vertex_colors = vertex_colors

    # Get the scene
    scene = mesh.scene()

    if side_view:
        scene.set_camera(angles=(0, np.pi / 2, 0), distance=2.0)

    # Render the scene to a PNG image.
    png = scene.save_image(resolution=(600, 600))

    # Update the widget
    image_widget.value = png

    return image_widget

def display_view(mesh, skinning_weights_np, predicted_skinning_weights_np, dropdown, side_view = widgets.fixed(None)):
    # Create an Image widget to hold the rendered PNG.
    image_widget1 = widgets.Image(format='png', width=600, height=600)
    interactive1 = widgets.interactive(update_visualization, bone_index=dropdown, mesh=widgets.fixed(mesh), skinning_weights_np=widgets.fixed(skinning_weights_np), image_widget=widgets.fixed(image_widget1), side_view=side_view)
    # Create an Image widget to hold the rendered PNG.
    image_widget2 = widgets.Image(format='png', width=600, height=600)
    interactive2 = widgets.interactive(update_visualization, bone_index=dropdown, mesh=widgets.fixed(mesh), skinning_weights_np=widgets.fixed(predicted_skinning_weights_np), image_widget=widgets.fixed(image_widget2), side_view=side_view)
    # Extract the controls (which is the first child of the interactive container)
    controls1 = interactive1.children[0]
    controls2 = interactive2.children[0]

    # Now, arrange each set of controls and its corresponding image side by side.
    box1 = widgets.HBox([controls1, image_widget1])
    box2 = widgets.HBox([controls2, image_widget2])

    # Finally, arrange the two boxes side by side.
    final_box = widgets.HBox([box1, box2])
    display(final_box)