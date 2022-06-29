import numpy as np
import torch

def viz_distance(points, quads):
    pass

def calc_distance_vertically(_pc_scene, predicted_quads):

    # TODO: Use Tensor instead of np.ndarray

    pc_scene = _pc_scene.cuda()
    pc_center = torch.mean(pc_scene, dim=0)  # To find the inner side of the point clouds
    distance = 10.0 * torch.ones((pc_scene.shape[0],), dtype=torch.double).cuda()

    # Calculate distances
    for _predicted_quad in predicted_quads:
        predicted_quad = torch.tensor(_predicted_quad).cuda()
        # TODO: convert params predicted_quad to torch.Tensor before and after nms to form a complete compute map
        quad_center = torch.mean(predicted_quad, dim=0)
        predicted_quad_norm = torch.cross(predicted_quad[1] - predicted_quad[0], predicted_quad[2] - predicted_quad[0])
        predicted_quad_norm = predicted_quad_norm / torch.norm(predicted_quad_norm)

        if torch.dot(pc_center - quad_center, predicted_quad_norm) > 0:
            predicted_quad_norm = -predicted_quad_norm  # Make inner distance < 0 and outsider > 0

        vertical_distance = (pc_scene - quad_center) @ predicted_quad_norm
        # vertical_distance = torch.bmm((pc_scene - quad_center).view(num_points, 1, 3),
        #                               predicted_quad_norm.repeat(num_points, 1).view(num_points, 3, 1)) \
        #     .view(num_points, )

        # Use Dynamic programming to find the "nearest" quad with minimum absolute error :)
        mask = torch.abs(vertical_distance) < torch.abs(distance)
        distance[mask] = vertical_distance[mask]

    return distance


def calc_distance_from_center(_pc_scene, predicted_quads, lambda_l=0):
    pc_scene = _pc_scene.cuda()
    pc_center = torch.mean(pc_scene, dim=0)
    distance = 10.0 * torch.ones((pc_scene.shape[0],), dtype=torch.double).cuda()

    # Calculate distances per quad
    for _predicted_quad in predicted_quads:
        predicted_quad = torch.tensor(_predicted_quad).cuda()
        # TODO: convert params predicted_quad to torch.Tensor before and after nms to form a complete compute map
        quad_center = torch.mean(predicted_quad, dim=0)
        predicted_quad_norm = torch.cross(predicted_quad[1] - predicted_quad[0], predicted_quad[2] - predicted_quad[0])
        predicted_quad_norm = predicted_quad_norm / torch.norm(predicted_quad_norm)

        if torch.dot(pc_center - quad_center, predicted_quad_norm) > 0:
            predicted_quad_norm = -predicted_quad_norm  # Make inner distance < 0 and outsider > 0

        # Calc Vertical loss: [40000, 1, 3] x [40000, 3, 1]
        num_points = pc_scene.shape[0]
        # vertical_distance = torch.bmm((pc_scene - quad_center).view(num_points, 1, 3),
        #                               predicted_quad_norm.repeat(num_points, 1).view(num_points, 3, 1))\
        #                           .view(num_points,)
        vertical_distance = (pc_scene - quad_center) @ predicted_quad_norm

        # Calc parallel loss: penalty those points whose projection lying outside the quads
        # Norm: (x, y ,z) -> (-y, x, z)
        parallel_norm1 = predicted_quad[1] - predicted_quad[0]
        parallel_norm1 /= torch.norm(parallel_norm1)
        parallel_norm2 = torch.cross(predicted_quad_norm, parallel_norm1)
        parallel_norm2 /= torch.norm(parallel_norm2)
        limit1_p = torch.dot(predicted_quad[1] - quad_center, parallel_norm1)
        # limit1_n = torch.dot(predicted_quad[0] - quad_center, parallel_norm1)
        limit2_p = torch.dot(predicted_quad[2] - quad_center, parallel_norm2)
        # limit2_n = torch.dot(predicted_quad[0] - quad_center, parallel_norm2)
        # print(f"{limit1_p.data} {limit1_p.data} {limit2_p.data} {limit2_p.data}")
        parallel_distance1 = torch.relu(torch.abs((pc_scene - quad_center) @ parallel_norm1) - torch.abs(limit1_p))
        parallel_distance2 = torch.relu(torch.abs((pc_scene - quad_center) @ parallel_norm2) - torch.abs(limit2_p))
        cond = parallel_distance1 > parallel_distance2
        parallel_distance = torch.where(cond, parallel_distance1, parallel_distance2)

        # parallel_norm_vector = predicted_quad_norm[[1, 0, 2]] * torch.Tensor([-1, 1, 1]).cuda()
        # parallel_distance = torch.bmm((pc_scene - quad_center).view(num_points, 1, 3),
        #                               parallel_norm_vector.repeat(num_points, 1).view(num_points, 3, 1))\
        #                           .view(num_points,)

        # half_long_axis = torch.norm(predicted_quad[1] - predicted_quad[0]) * 0.5
        # parallel_distance = torch.relu(torch.abs(parallel_distance) - half_long_axis)

        new_distance = lambda_l * torch.abs(vertical_distance) + (1-lambda_l) * parallel_distance
        # new_distance = parallel_distance
        distance[torch.abs(new_distance) < torch.abs(distance)] = new_distance[torch.abs(new_distance) < torch.abs(distance)]

    return distance



def distance_loss(end_points, config, query_points_obj_topk, pc_loss, num_layer):

    distance_loss = 0.0

    points, semantic_labels = end_points['point_clouds'], end_points['semantic_labels']
    batch_size = points.shape[0]
    CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
                   'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
                   'per_class_proposal': True, 'conf_thresh': 0.0,'quad_thresh':0.5,
                   'dataset_config': None} 
        
    prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_layer-1)]

    for prefix in prefixes:
        from models.ap_helper_pq import parse_quad_predictions
        batch_pred_map_cls, pred_mask, batch_pred_corners_list = parse_quad_predictions(end_points, CONFIG_DICT, prefix)

        for b in range(batch_size):
    
            # Retrieve quads
            point_cloud, semantic_label = points[b], semantic_labels[b]
            import IPython
            IPython.embed(header="in training process, before calcing loss")
            pred_corner = end_points['proposal_batch_pred_corners_list_tensor'][b]


            # Calculate Distances

            # Filter out points

            # Calculate distance loss and accumulate

    lambda_distance = 0.50
    return lambda_ 
    pass
