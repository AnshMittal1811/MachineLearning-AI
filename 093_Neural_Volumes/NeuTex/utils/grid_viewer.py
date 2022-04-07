from sapien.core import Pose, Engine, OptifuserRenderer, OptifuserController

# , VulkanController, VulkanRenderer
import numpy as np


def visualize_grid(grid: np.ndarray):
    assert len(grid.shape) == 3
    assert grid.shape[0] == grid.shape[1] == grid.shape[2]
    resolution = grid.shape[0]
    grid = grid.astype(float)

    pts = np.linspace(-1, 1, resolution + 1)
    pts = (pts[:-1] + pts[1:]) / 2

    ref = np.stack(np.meshgrid(*([pts] * 3), indexing="ij"), axis=-1,)
    scale = 1 / resolution * 0.99
    indices = np.stack(np.where(grid != 0), 1)

    engine = Engine()
    renderer = OptifuserRenderer()
    controller = OptifuserController(renderer)
    engine.set_renderer(renderer)
    scene = engine.create_scene()
    controller.set_current_scene(scene)

    builder = scene.create_actor_builder()
    for idx in indices:
        p = ref[idx[0], idx[1], idx[2]]
        builder.add_box_visual(Pose(p), size=[scale, scale, scale])
    builder.build(True)
    scene.step()
    scene.update_render()
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0, -1], [0.5, 0.5, 0.5])

    controller.show_window()
    while not controller.should_quit:
        controller.render()
