from src.dataset.prepare import *
from src.visualize.render_mesh import *
from src.dataset.uv_face import uvm2mesh
import matplotlib.pyplot as plt


def test1():
    data_processor = DataProcessor(bbox_extend_rate=BBOX_EXTEND_RATE, marg_rate=MARG_RATE)
    data_processor.initialize('data/packs/AFLW2000/image00002.jpg', 'data/output')
    data_processor.bfm_info = sio.loadmat(data_processor.image_path.replace('.jpg', '.mat'))
    result = data_processor.run_offset_posmap()
    attention_mask, old_bbox, bbox, T_2d, T_2d_inv, T_3d, new_kpt, init_kpt, T_bfm, uv_position_map, uv_offset_map, cropped_image = result
    plt.imshow(cropped_image)
    plt.show()
    posmesh = uvm2mesh(uv_position_map / 128.0, False)
    ret = render_face_orthographic(posmesh, (cropped_image * 255).astype(np.int))
    plt.imshow(ret)
    plt.show()
    plt.imshow(attention_mask)
    plt.show()

    from src.dataset.uv_face import mean_shape_map_np
    uvm4d = np.concatenate((uv_position_map, np.ones((256, 256, 1))), axis=-1)
    t_all = (T_bfm.T.dot(T_3d.T)).T
    shape = (uv_offset_map + mean_shape_map_np) / OFFSET_FIX_RATE
    shape4d = np.concatenate((shape, np.ones((256, 256, 1))), axis=-1)
    print(shape4d.dot(t_all.T) - uvm4d)
    print(uvm4d.dot(inv(t_all).T) - shape4d)


data_processor = DataProcessor(bbox_extend_rate=BBOX_EXTEND_RATE, marg_rate=MARG_RATE)
data_processor.process_item('data/packs/AFLW2000/image00002.jpg', 'data/output')