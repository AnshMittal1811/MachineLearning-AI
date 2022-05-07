import tensorflow as tf
import h5py, os
import vis_utils, argparse



@tf.function
def orient_point_cloud(x_batch, rot_random_batch, rot_can):

    '''
    B, N, 3
    B, 3, 3
    '''

    rot_random = tf.tile(rot_random_batch, [x_batch.shape[0], 1, 1])
    # print(rot_random.shape)
    x_batch_rotated = tf.einsum("bij,bkj->bki", rot_random, x_batch)
    x_batch_canonical = tf.einsum("bij,bkj->bki", rot_can, x_batch_rotated)

    return x_batch_canonical, x_batch_rotated


def save_vis_point_cloud(x_file, rot_random_file, rot_can_file, batch_size = 32, num_rots = 120, output_directory = "./pointclouds", total_clouds = None):


    output_directory = os.path.join(output_directory,"")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    x = h5py.File(x_file, "r")["data"][:]
    rot_random = h5py.File(rot_random_file, "r")["data"][:]
    rot_can = h5py.File(rot_can_file, "r")["data"][:]

    num_rots_temp = min(rot_can.shape[1], rot_random.shape[0])
    num_rots = min(num_rots, num_rots_temp)
    total_shapes = x.shape[0]
    batch_num = 0

    while (batch_num * batch_size <= total_shapes):
        remaining = total_shapes - batch_num * batch_size

        for rot_num in range(num_rots):

            if remaining < batch_num * batch_size:
                input_shape = x[batch_num * batch_size: total_shapes]
                input_can_rot = rot_can[batch_num * batch_size: total_shapes]
            else:
                input_shape = x[batch_num * batch_size: (batch_num + 1)*batch_size]
                input_can_rot = rot_can[batch_num * batch_size: (batch_num + 1)*batch_size]

            input_rot_random = rot_random[rot_num: (rot_num + 1)] # 1 x 3 x 3
            result, result_rotated = orient_point_cloud(input_shape, input_rot_random, input_can_rot[:, rot_num])

            for save_id in range(result.shape[0]):
                idx = batch_num * batch_size + save_id
                vis_utils.save_pointcloud(result[save_id:save_id+1], output_directory + "%d_canonical_pointcloud_full_rot_%d.ply" % (idx, rot_num))
                vis_utils.save_pointcloud(result_rotated[save_id:save_id+1], output_directory + "%d_input_full_%d.ply" % (idx, rot_num))
            
            if total_clouds is not None:
                if batch_num * batch_size > total_clouds:
                    return
            

        batch_num += 1

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rot_random", required = True, type = str)
    parser.add_argument("--shape", required = True, type = str)
    parser.add_argument("--rot_can", required = True, type = str)
    parser.add_argument("--batch_size", type = int, default = 8)
    parser.add_argument("--output", type=str, required = True)
    parser.add_argument("--num_rots", type=int, default = 120)
    parser.add_argument("--total_clouds", type=int, default = None)


    args = parser.parse_args()

    save_vis_point_cloud(args.shape, args.rot_random, args.rot_can, args.batch_size, args.num_rots, args.output, args.total_clouds)