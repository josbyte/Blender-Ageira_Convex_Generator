# Precision Convex Decomposition (CoACD)

Blender add-on that performs high-precision convex decomposition using the CoACD library with optimized geometry (threaded rendering, bottom-left overlay, presets).

## Version
1.0.1

## Blender Compatibility
4.5.2

## Description
This add-on allows you to decompose selected mesh objects into convex hulls using the CoACD algorithm. It is useful for physics, games, and 3D modeling where convex shapes are required.

Key Features:
- Threaded rendering to avoid blocking the Blender interface.
- Progress overlay in the bottom-left corner.
- Predefined presets for different levels of detail.
- Blender Flatpak support.
- Automatic mesh validation and triangulation.
- Advanced options such as hull decimation and merging.

## Requirements
- Blender 4.5.2 or higher.
- Python libraries: trimesh, coacd, numpy (installed automatically if not present).

## Installation
1. Download the `GKPPVuJ.py` file.
2. In Blender, go to Edit > Preferences > Add-ons.
3. Click "Install..." and select the file.
4. Activate the "Precision Convex Decomposition (CoACD)" add-on.
5. The panel will appear in View3D > Sidebar > CoACD.

## Usage
1. Select one or more mesh objects in Object mode.
2. Go to the CoACD panel in the sidebar.
3. Choose a preset or adjust parameters manually.
4. Click "Convex Decomposition".
5. The progress will be shown in the lower left corner. Press ESC to cancel.
6. Convex hulls will be created in a new collection for each object.

## Presets
- **Low**: Fast, low detail (threshold 0.2, max hulls 8, decimate on).
- **Mid**: Balanced (threshold 0.1, max hulls 20).

These might take a while to proccess
- **High**: Detailed (threshold 0.03, max hulls 50).
- **Very High**: Maximum detail (threshold 0.01, max hulls 200, no blending).

## Parameters
- **Concavity Threshold**: Concavity threshold (lower = more pieces).
- **Max Hulls**: Maximum number of hulls per object.
- **Preprocess Mode**: Preprocessing mode (auto, on, off).
- **Preprocess Resolution**: Preprocessing resolution.
- **MCTS Iterations**: MCTS iterations for better quality.
- **Enable Decimate**: Apply decimation to hulls.
- **Decimate Ratio**: Ratio of faces to keep.
- **Merge Hulls**: Merge convex hulls.

## Notes
- The addon handles errors and validates meshes automatically.
- Compatible with Flatpak Blender by automatically detecting the site-packages path.
- Non-blocking processing thanks to threads.