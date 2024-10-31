import os
import numpy as np
import copy

def intersection_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersections = w * h
    return intersections

def box_area(bbox):
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return area

def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)                                              
    return(o)

def iou_batch_level(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    if (len(bboxes1) == 0 or len(bboxes2) == 0):
        return np.empty((0, len(bboxes1)), dtype=int)
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
              + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return (o)

def cal_score_dif_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    score2 = bboxes2[..., 4]
    score1 = bboxes1[..., 4]

    return (abs(score2 - score1))

def cal_score_dif_batch_two_score(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    score2 = bboxes2[..., 5]  # tracks最后一次匹配上的box的scores
    score1 = bboxes1[..., 4]

    return (abs(score2 - score1))

def hmiou(bboxes1, bboxes2):
    """
    Height_Modulated_IoU
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])  # 交集左上角点y坐标
    yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])  # 交集右下角点y坐标

    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])  # 并集左上角点y坐标
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])  # 并集右下角点y坐标
    o = (yy12 - yy11) / (yy22 - yy21)  # HIOU

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o *= wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)  # 按元素相乘 HIOU * IOU  高度弱信息 * 空间强信息
    return (o)


def giou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)  

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1 
    hc = yyc2 - yyc1 
    assert((wc > 0).all() and (hc > 0).all())
    area_enclose = wc * hc 
    giou = iou - (area_enclose - wh) / area_enclose
    giou = (giou + 1.)/2.0 # resize from (-1,1) to (0,1)
    return giou

def giou_batch_true(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    iou = wh / union

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1
    hc = yyc2 - yyc1
    assert((wc > 0).all() and (hc > 0).all())
    area_enclose = wc * hc
    giou = iou - (area_enclose - union) / area_enclose
    giou = (giou + 1.)/2.0 # resize from (-1,1) to (0,1)
    return giou

def diou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / outer_diag

    return (diou + 1) / 2.0 # resize from (-1,1) to (0,1)

def ciou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    # prevent dividing over zero. add one pixel shift
    h2 = h2 + 1.
    h1 = h1 + 1.
    arctan = np.arctan(w2/h2) - np.arctan(w1/h1)
    v = (4 / (np.pi ** 2)) * (arctan ** 2)
    S = 1 - iou 
    alpha = v / (S+v)
    ciou = iou - inner_diag / outer_diag - alpha * v
    
    return (ciou + 1) / 2.0 # resize from (-1,1) to (0,1)


def ct_dist(bboxes1, bboxes2):
    """
        Measure the center distance between two sets of bounding boxes,
        this is a coarse implementation, we don't recommend using it only
        for association, which can be unstable and sensitive to frame rate
        and object speed.
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    ct_dist = np.sqrt(ct_dist2)

    # The linear rescaling is a naive version and needs more study
    ct_dist = ct_dist / ct_dist.max()
    return ct_dist.max() - ct_dist # resize to (0,1)


def speed_direction_batch(dets, tracks):
    """
    batch formulation of function 'speed_direction', compute normalized speed from batch bboxes
    @param dets:
    @param tracks:
    @return: normalized speed in batch
    """
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:,0] + dets[:,2])/2.0, (dets[:,1]+dets[:,3])/2.0
    CX2, CY2 = (tracks[:,0] + tracks[:,2]) /2.0, (tracks[:,1]+tracks[:,3])/2.0
    dx = CX1 - CX2 
    dy = CY1 - CY2 
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm 
    dy = dy / norm
    return dy, dx # size: num_track x num_det


def linear_assignment(cost_matrix, thresh=0.):
    try:        # [hgx0411] goes here!
        import lap
        if thresh != 0:
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        else:
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate(detections, trackers, iou_threshold, velocities, previous_obs, vdc_weight):    
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:,4]<0)] = 0
    
    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix+angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def cost_vel(Y, X, trackers, velocities, detections, previous_obs, vdc_weight):
    # Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    scores = np.repeat(detections[:, -2][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight  # 0.05 * Δθ
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    return angle_diff_cost

def speed_direction_batch_lt(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,0], dets[:,1]  # 左上角xy
    CX2, CY2 = tracks[:,0], tracks[:,1]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def speed_direction_batch_rt(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,0], dets[:,3]
    CX2, CY2 = tracks[:,0], tracks[:,3]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def speed_direction_batch_lb(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,2], dets[:,1]
    CX2, CY2 = tracks[:,2], tracks[:,1]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def speed_direction_batch_rb(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,2], dets[:,3]
    CX2, CY2 = tracks[:,2], tracks[:,3]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def associate_4_points(detections, trackers, iou_threshold, lt, rt, lb, rb, previous_obs, vdc_weight, iou_type=None, args=None):
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    Y1, X1 = speed_direction_batch_lt(detections, previous_obs)
    Y2, X2 = speed_direction_batch_rt(detections, previous_obs)
    Y3, X3 = speed_direction_batch_lb(detections, previous_obs)
    Y4, X4 = speed_direction_batch_rb(detections, previous_obs)
    YC, XC = speed_direction_batch(detections, previous_obs)
    cost_lt = cost_vel(Y1, X1, trackers, lt, detections, previous_obs, vdc_weight)
    cost_rt = cost_vel(Y2, X2, trackers, rt, detections, previous_obs, vdc_weight)
    cost_lb = cost_vel(Y3, X3, trackers, lb, detections, previous_obs, vdc_weight)
    cost_rb = cost_vel(Y4, X4, trackers, rb, detections, previous_obs, vdc_weight)

    iou_matrix = iou_type(detections, trackers)
    angle_diff_cost = cost_lt + cost_rt + cost_lb + cost_rb

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def associate_4_points_with_score(detections, trackers, iou_threshold, lt, rt, lb, rb, previous_obs, vdc_weight, iou_type=None, args=None):
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    Y1, X1 = speed_direction_batch_lt(detections, previous_obs)
    Y2, X2 = speed_direction_batch_rt(detections, previous_obs)
    Y3, X3 = speed_direction_batch_lb(detections, previous_obs)
    Y4, X4 = speed_direction_batch_rb(detections, previous_obs)
    cost_lt = cost_vel(Y1, X1, trackers, lt, detections, previous_obs, vdc_weight)
    cost_rt = cost_vel(Y2, X2, trackers, rt, detections, previous_obs, vdc_weight)
    cost_lb = cost_vel(Y3, X3, trackers, lb, detections, previous_obs, vdc_weight)
    cost_rb = cost_vel(Y4, X4, trackers, rb, detections, previous_obs, vdc_weight)
    iou_matrix = iou_type(detections, trackers)
    score_dif = cal_score_dif_batch(detections, trackers)

    angle_diff_cost = cost_lt + cost_rt + cost_lb + cost_rb

    # TCM
    angle_diff_cost -= score_dif * args.TCM_first_step_weight

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def associate_4_points_with_score_with_reid(detections, trackers, iou_threshold, lt, rt, lb, rb, previous_obs, vdc_weight,
                                            iou_type=None, args=None,emb_cost=None, weights=(1.0, 0), thresh=0.8,
                                            long_emb_dists=None, with_longterm_reid=False,
                                            longterm_reid_weight=0.0, with_longterm_reid_correction=False,
                                            longterm_reid_correction_thresh=0.0, dataset="dancetrack"):
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    Y1, X1 = speed_direction_batch_lt(detections, previous_obs)
    Y2, X2 = speed_direction_batch_rt(detections, previous_obs)
    Y3, X3 = speed_direction_batch_lb(detections, previous_obs)
    Y4, X4 = speed_direction_batch_rb(detections, previous_obs)
    cost_lt = cost_vel(Y1, X1, trackers, lt, detections, previous_obs, vdc_weight)
    cost_rt = cost_vel(Y2, X2, trackers, rt, detections, previous_obs, vdc_weight)
    cost_lb = cost_vel(Y3, X3, trackers, lb, detections, previous_obs, vdc_weight)
    cost_rb = cost_vel(Y4, X4, trackers, rb, detections, previous_obs, vdc_weight)
    iou_matrix = iou_type(detections, trackers)  # HIOU * IOU
    score_dif = cal_score_dif_batch(detections, trackers)

    angle_diff_cost = cost_lt + cost_rt + cost_lb + cost_rb

    # TCM
    angle_diff_cost -= score_dif * args.TCM_first_step_weight

    if min(iou_matrix.shape) > 0:
        if emb_cost is None:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
        else:
            if not with_longterm_reid:
                matched_indices = linear_assignment(weights[0] * (-(iou_matrix + angle_diff_cost)) + weights[1] * emb_cost) # , thresh=thresh
            else:   # long-term reid feats
                matched_indices = linear_assignment(weights[0] * (-(iou_matrix + angle_diff_cost)) +
                                                    weights[1] * emb_cost + longterm_reid_weight * long_emb_dists)  # , thresh=thresh

        if matched_indices.size == 0:
            matched_indices = np.empty(shape=(0, 2))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU (and long-term ReID feats)
    matches = []
    # iou_matrix_thre = iou_matrix if dataset == "dancetrack" else iou_matrix - score_dif
    iou_matrix_thre = iou_matrix - score_dif
    if with_longterm_reid_correction:
        for m in matched_indices:
            if (emb_cost[m[0], m[1]] > longterm_reid_correction_thresh) and (iou_matrix_thre[m[0], m[1]] < iou_threshold):
                print("correction:", emb_cost[m[0], m[1]])
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
    else:
        for m in matched_indices:
            if (iou_matrix_thre[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate_kitti(detections, trackers, det_cates, iou_threshold, 
        velocities, previous_obs, vdc_weight):
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    """
        Cost from the velocity direction consistency
    """
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:,0], velocities[:,1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi /2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:,4]<0)]=0  
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    scores = np.repeat(detections[:,-1][:, np.newaxis], trackers.shape[0], axis=1)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    """
        Cost from IoU
    """
    iou_matrix = iou_batch(detections, trackers)
    

    """
        With multiple categories, generate the cost for catgory mismatch
    """
    num_dets = detections.shape[0]
    num_trk = trackers.shape[0]
    cate_matrix = np.zeros((num_dets, num_trk))
    for i in range(num_dets):
            for j in range(num_trk):
                if det_cates[i] != trackers[j, 4]:
                        cate_matrix[i][j] = -1e6
    
    cost_matrix = - iou_matrix -angle_diff_cost - cate_matrix

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# compute embedding distance and gating, borrowed and modified from FairMOT
from scipy.spatial.distance import cdist
def embedding_distance(tracks_feat, detections_feat, metric='cosine'):
    """
    :param tracks: list[KalmanBoxTracker]
    :param detections: list[KalmanBoxTracker]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks_feat), len(detections_feat)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    # det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)    # [detection_num, emd_dim]
    # #for i, track in enumerate(tracks):
    #     #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    # track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)    # [track_num, emd_dim]
    cost_matrix = np.maximum(0.0, cdist(tracks_feat, detections_feat, metric))  # Nomalized features, metric: cosine, [track_num, detection_num]
    return cost_matrix

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

# [hgx0411] compute embedding distance and gating, borrowed and modified from FairMOT
def fuse_motion(cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    for row, track in enumerate(tracks):
        gating_distance = track.kf.gating_distance(detections, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix

# [hgx0411] compute embedding distance and gating, borrowed and modified from FairMOT
import lap
def linear_assignment_appearance(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def fuse_score(cost_matrix, det_scores):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = - cost_matrix
    det_scores = np.expand_dims(det_scores, axis=1).repeat(cost_matrix.shape[1], axis=1)
    fuse_sim = iou_sim * det_scores
    fuse_cost = - fuse_sim
    return fuse_cost

def get_sub_mask(deep_range, col):
    mix_len=deep_range[0]
    max_len=deep_range[-1]
    if max_len == mix_len:
        lc = mix_len
    mask = []
    for d in deep_range:
        if d > deep_range[0] and d < deep_range[-1]:
            mask.append((col >= lc) & (col < d))
            lc = d
        elif d == deep_range[-1]:
            mask.append((col >= lc) & (col <= d))
            lc = d
        else:
            lc = d
            continue
    return mask

def get_deep_range(obj, step=2):
    col = []
    for t in obj:
        lend = t[-1]
        col.append(lend)
    max_len, mix_len = max(col), min(col)
    if max_len != mix_len:
        deep_range = np.arange(mix_len, max_len, (max_len - mix_len + 1) / step)
        if deep_range[-1] < max_len:
            deep_range = np.concatenate([deep_range, np.array([max_len],)])
            deep_range[0] = np.floor(deep_range[0])
            deep_range[-1] = np.ceil(deep_range[-1])
    else:
        deep_range = [mix_len,]
    mask = get_sub_mask(deep_range, col)
    return mask


def deep_vec(x):
    y2 = x[3]
    length = 2000 - y2
    return length

def get_deep_range_low(obj, step=6):
    col = []
    for t in obj:
        lend = t[-1]
        col.append(lend)
    max_len, mix_len = max(col), min(col)
    if max_len != mix_len:
        deep_range = np.arange(mix_len, max_len, (max_len - mix_len + 1) / step)
        if deep_range[-1] < max_len:
            deep_range = np.concatenate([deep_range, np.array([max_len],)])
            deep_range[0] = np.floor(deep_range[0])
            deep_range[-1] = np.ceil(deep_range[-1])
    else:
        deep_range = [mix_len,]
    mask = get_sub_mask(deep_range, col)
    return mask

def get_deep_range_DCM(obj, step=4):
    col = []
    for t in obj:
        lend = deep_vec(t)
        col.append(lend)
    max_len, mix_len = max(col), min(col)
    if max_len != mix_len:
        deep_range = np.arange(mix_len, max_len, (max_len - mix_len + 1) / step)
        if deep_range[-1] < max_len:
            deep_range = np.concatenate([deep_range, np.array([max_len],)])
            deep_range[0] = np.floor(deep_range[0])
            deep_range[-1] = np.ceil(deep_range[-1])
    else:
        deep_range = [mix_len,]
    mask = get_sub_mask(deep_range, col)
    return mask

def tb_dist(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    x_move_min = np.minimum(np.abs(bboxes1[..., 0] - bboxes2[..., 0]), np.abs(bboxes1[..., 2] - bboxes2[..., 2]))
    x_move_max = np.maximum(np.abs(bboxes1[..., 0] - bboxes2[..., 0]), np.abs(bboxes1[..., 2] - bboxes2[..., 2]))
    x_move_dist = x_move_max - x_move_min
    x_move_dist_max = x_move_dist.max()
    x_move_dist = x_move_dist / x_move_dist_max
    conf_x = x_move_dist_max - x_move_dist

    y_move_min = np.minimum(np.abs(bboxes1[..., 1] - bboxes2[..., 1]), np.abs(bboxes1[..., 3] - bboxes2[..., 3]))
    y_move_max = np.maximum(np.abs(bboxes1[..., 1] - bboxes2[..., 1]), np.abs(bboxes1[..., 3] - bboxes2[..., 3]))
    y_move_dist = y_move_max - y_move_min
    y_move_dist_max = y_move_dist.max()
    y_move_dist = y_move_dist / y_move_dist_max
    conf_y = y_move_dist_max - y_move_dist

    tb_dist = np.minimum(np.abs(bboxes1[..., 1] - bboxes2[..., 1]), np.abs(bboxes1[..., 3] - bboxes2[..., 3])) \
              + np.minimum(np.abs(bboxes1[..., 0] - bboxes2[..., 0]), np.abs(bboxes1[..., 2] - bboxes2[..., 2]))
    # tb_dist = np.abs(bboxes1[..., 1] - bboxes2[..., 1]) + np.abs(bboxes1[..., 3] - bboxes2[..., 3])

    tb_dist = tb_dist / tb_dist.max()  # 归一化
    return tb_dist.max() - tb_dist # resize to (0,1)
    # return tb_dist.max() - tb_dist

def tb_hmiou(bboxes1, bboxes2):
    """
    Height_Modulated_IoU
    """
    tb = tb_dist(bboxes1, bboxes2)
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    o = (yy12 - yy11) / (yy22 - yy21)  # HIOU

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    tb *= wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return (tb)

def Biou(bboxes1, bboxes2, level=1):
    """
    Height_Modulated_IoU
    """
    assert level in [1, 2], 'level must be 1 or 2'

    b = 0.3 if level == 1 else 0.4

    '''原始box'''
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    '''扩展后的box'''
    bboxes1_B = copy.deepcopy(bboxes1)
    bboxes2_B = copy.deepcopy(bboxes2)

    bboxes1_B[..., 0] = bboxes1[..., 0] - b * (bboxes1[..., 2] - bboxes1[..., 0])
    bboxes1_B[..., 1] = bboxes1[..., 1] - b * (bboxes1[..., 3] - bboxes1[..., 1])
    bboxes1_B[..., 2] = bboxes1[..., 2] + b * (bboxes1[..., 2] - bboxes1[..., 0])
    bboxes1_B[..., 3] = bboxes1[..., 3] + b * (bboxes1[..., 3] - bboxes1[..., 1])

    bboxes2_B[..., 0] = bboxes2[..., 0] - b * (bboxes2[..., 2] - bboxes2[..., 0])
    bboxes2_B[..., 1] = bboxes2[..., 1] - b * (bboxes2[..., 3] - bboxes2[..., 1])
    bboxes2_B[..., 2] = bboxes2[..., 2] + b * (bboxes2[..., 2] - bboxes2[..., 0])
    bboxes2_B[..., 3] = bboxes2[..., 3] + b * (bboxes2[..., 3] - bboxes2[..., 1])
    '''原始box的交集与并集'''
    yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    yy11_W = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy12_W = np.minimum(bboxes1[..., 2], bboxes2[..., 2])

    yy21_W = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yy22_W = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    '''扩展后的box的交集与并集'''
    yy11_B = np.maximum(bboxes1_B[..., 1], bboxes2_B[..., 1])
    yy12_B = np.minimum(bboxes1_B[..., 3], bboxes2_B[..., 3])

    yy21_B = np.minimum(bboxes1_B[..., 1], bboxes2_B[..., 1])
    yy22_B = np.maximum(bboxes1_B[..., 3], bboxes2_B[..., 3])

    yy11_WB = np.maximum(bboxes1_B[..., 0], bboxes2_B[..., 0])
    yy12_WB = np.minimum(bboxes1_B[..., 2], bboxes2_B[..., 2])

    yy21_WB = np.minimum(bboxes1_B[..., 0], bboxes2_B[..., 0])
    yy22_WB = np.maximum(bboxes1_B[..., 2], bboxes2_B[..., 2])
    '''原始box的HIOU'''
    o = (yy12 - yy11) / (yy22 - yy21)  # HIOU
    '''原始box的WIOU'''
    o_W = (yy12_W - yy11_W) / (yy22_W - yy21_W)
    '''扩展后的box的HIOU'''
    o_B = (yy12_B - yy11_B) / (yy22_B - yy21_B)
    '''扩展后的box的WIOU'''
    o_WB = (yy12_WB - yy11_WB) / (yy22_WB - yy21_WB)

    '''原始box的IOU'''
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    '''扩展后的box的IOU'''
    xx1_B = np.maximum(bboxes1_B[..., 0], bboxes2_B[..., 0])
    yy1_B = np.maximum(bboxes1_B[..., 1], bboxes2_B[..., 1])
    xx2_B = np.minimum(bboxes1_B[..., 2], bboxes2_B[..., 2])
    yy2_B = np.minimum(bboxes1_B[..., 3], bboxes2_B[..., 3])
    w_B = np.maximum(0., xx2_B - xx1_B)
    h_B = np.maximum(0., yy2_B - yy1_B)
    wh_B = w_B * h_B
    '''对扩展后的box的WIOU加以限制'''
    ans = wh_B / ((bboxes1_B[..., 2] - bboxes1_B[..., 0]) * (bboxes1_B[..., 3] - bboxes1_B[..., 1])
                     + (bboxes2_B[..., 2] - bboxes2_B[..., 0]) * (bboxes2_B[..., 3] - bboxes2_B[..., 1]) - wh_B)
    # o_HWB *= wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    #                  + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)

    return (ans)
