"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
import copy
import math
from .association import *
from collections import deque       # [hgx0418] deque for reid feature
np.random.seed(0)

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i  # 3，2，1
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    score = bbox[4]
    if score:
        return np.array([x, y, s, score, r]).reshape((5, 1))
    else:
        return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[4])
    h = x[2] / w
    score = x[3]
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_lt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_rt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[3]
    cx2, cy2 = bbox2[0], bbox2[3]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_lb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[1]
    cx2, cy2 = bbox2[2], bbox2[1]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_rb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[3]
    cx2, cy2 = bbox2[2], bbox2[3]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, temp_feat, delta_t=3, orig=False, buffer_size=30, args=None):     # 'temp_feat' and 'buffer_size' for reid feature
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        # if not orig and not args.kalman_GPR:
        if not orig:  # orig：原始的KF 7维
          from .kalmanfilter_score_new import KalmanFilterNew_score_new as KalmanFilter_score_new
          self.kf = KalmanFilter_score_new(dim_x=9, dim_z=5)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # u, v, s, c, r, ~u, ~v, ~s, ~c
        self.kf.F = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[5:, 5:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[-2, -2] *= 0.01
        self.kf.Q[5:, 5:] *= 0.01

        self.kf.x[:5] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.last_observation_save = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations = []
        self.velocity_lt = None
        self.velocity_rt = None
        self.velocity_lb = None
        self.velocity_rb = None
        self.delta_t = delta_t
        self.confidence_pre = None
        self.confidence = bbox[-1]
        self.args = args
        self.kf.args = args

        # add the following values and functions
        self.smooth_feat = None
        buffer_size = args.longterm_bank_length
        self.features = deque([], maxlen=buffer_size)
        self.update_features(temp_feat)

        # momentum of embedding update
        self.alpha = self.args.alpha

    # ReID. for update embeddings during tracking
    def update_features(self, feat, score=-1):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if self.args.adapfs:
                assert score > 0
                pre_w = self.alpha * (self.confidence / (self.confidence + score))
                cur_w = (1 - self.alpha) * (score / (self.confidence + score))
                sum_w = pre_w + cur_w
                pre_w = pre_w / sum_w
                cur_w = cur_w / sum_w
                self.smooth_feat = pre_w * self.smooth_feat + cur_w * feat
            else:
                self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat  # EMA
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)  # 正则化

    def camera_update(self, warp_matrix):
        """
        update 'self.mean' of current tracklet with ecc results.
        Parameters
        ----------
        warp_matrix: warp matrix computed by ECC.
        """
        x1, y1, x2, y2, s = convert_x_to_bbox(self.kf.x)[0]
        x1_, y1_, _ = warp_matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = warp_matrix @ np.array([x2, y2, 1]).T
        # w, h = x2_ - x1_, y2_ - y1_
        # cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:5] = convert_bbox_to_z([x1_, y1_, x2_, y2_, s])

    def update(self, bbox, id_feature, update_feature=True):
        """
        Updates the state vector with observed bbox.
        """
        velocity_lt = None
        velocity_rt = None
        velocity_lb = None
        velocity_rb = None
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    # dt = self.delta_t - i
                    if self.age - i - 1 in self.observations:
                        previous_box = self.observations[self.age - i - 1]
                        if velocity_lt is not None:
                            velocity_lt += speed_direction_lt(previous_box, bbox)
                            velocity_rt += speed_direction_rt(previous_box, bbox)
                            velocity_lb += speed_direction_lb(previous_box, bbox)
                            velocity_rb += speed_direction_rb(previous_box, bbox)
                        else:
                            velocity_lt = speed_direction_lt(previous_box, bbox)
                            velocity_rt = speed_direction_rt(previous_box, bbox)
                            velocity_lb = speed_direction_lb(previous_box, bbox)
                            velocity_rb = speed_direction_rb(previous_box, bbox)
                        # break
                if previous_box is None:
                    previous_box = self.last_observation
                    self.velocity_lt = speed_direction_lt(previous_box, bbox)
                    self.velocity_rt = speed_direction_rt(previous_box, bbox)
                    self.velocity_lb = speed_direction_lb(previous_box, bbox)
                    self.velocity_rb = speed_direction_rb(previous_box, bbox)
                else:
                    self.velocity_lt = velocity_lt
                    self.velocity_rt = velocity_rt
                    self.velocity_lb = velocity_lb
                    self.velocity_rb = velocity_rb
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.last_observation_save = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
            # add interface for update feature or not
            if update_feature:
                if self.args.adapfs:
                    self.update_features(id_feature, score=bbox[-1])
                else:
                    self.update_features(id_feature)
            self.confidence_pre = self.confidence
            self.confidence = bbox[-1]
        else:
            self.kf.update(bbox)
            self.confidence_pre = None

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[7]+self.kf.x[2]) <= 0):
            self.kf.x[7] *= 0.0

        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        if not self.confidence_pre:
            return self.history[-1], np.clip(self.kf.x[3], self.args.track_thresh, 1.0), np.clip(self.confidence, 0.1, self.args.track_thresh)
        else:
            return self.history[-1], np.clip(self.kf.x[3], self.args.track_thresh, 1.0), np.clip(self.confidence - (self.confidence_pre - self.confidence), 0.1, self.args.track_thresh)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist,
                "Height_Modulated_IoU": hmiou
                }


class DP_Sort_ReID(object):
    def __init__(self, args, det_thresh, max_age=30, min_hits=3,
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = args.use_byte
        self.args = args
        KalmanBoxTracker.count = 0

    # ECC for CMC
    def camera_update(self, trackers, warp_matrix):
        for tracker in trackers:
            tracker.camera_update(warp_matrix)

    def feature_change(self, fea, tracks):
        for i, track in enumerate(tracks):
            if track.time_since_update > 1:
                # fea_array = np.array(track.features)
                # fea_mean = np.mean(fea_array, axis=0, dtype=np.float)
                # fea[i] = fea_mean
                pass
            else:
                fea_last = np.array(track.curr_feat)
                fea[i] = fea_last
        return fea

    def crop_and_resize(self, track_xyxy, annos):
        search_area_factor = 7  # 搜索区域的尺寸
        x1 = track_xyxy[0]
        y1 = track_xyxy[1]
        x2 = track_xyxy[2]
        y2 = track_xyxy[3]
        w = x2 - x1
        h = y2 - y1
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
        x1_search = x1 + 0.5 * w - crop_sz * 0.5
        y1_search = y1 + 0.5 * h - crop_sz * 0.5

        transformed_coord_d = annos[:, 0:2] - np.array([x1_search, y1_search])
        annos_wh = annos[:, 2:4] - annos[:, 0:2]
        transformed_coord = np.concatenate((transformed_coord_d, annos_wh), axis=1)
        return transformed_coord, crop_sz

    def compute_mix_dist(self, stracks, dets, matrix):
        search_boxes = []
        search_sz = []
        for strack in stracks:
            s_box, t_search = self.crop_and_resize(strack, dets)
            search_sz.append(t_search)
        search_sz = np.array(search_sz)
        for i, boxes in enumerate(search_boxes):
            # correspond to strack[i]
            for j, (t, l, w, h) in enumerate(boxes):
                cx, cy = t + w / 2, l + h / 2
                # don't consider outsiders
                if cx > 0 and cy > 0 and cx < search_sz[i] and cy < search_sz[i]:
                    pass
                else:
                    matrix[j][i] = 1
        return matrix

    def compute_mix_dist_iou(self, stracks, dets, matrix):
        search_boxes = []  # 所有track对应的搜索区域内的box
        search_sz = []
        for strack in stracks:
            s_box, t_search = self.crop_and_resize(strack, dets)
            search_boxes.append(s_box)
            search_sz.append(t_search)
        search_sz = np.array(search_sz)
        for i, boxes in enumerate(search_boxes):
            # correspond to strack[i]
            for j, (t, l, w, h) in enumerate(boxes):
                cx, cy = t + w / 2, l + h / 2
                # don't consider outsiders
                if cx > 0 and cy > 0 and cx < search_sz[i] and cy < search_sz[i]:
                    pass
                else:
                    matrix[j][i] = 0
        return matrix

    def compute_mix_dist_cost(self, stracks, dets, matrix):
        search_boxes = []
        search_sz = []
        for strack in stracks:
            s_box, t_search = self.crop_and_resize(strack, dets)
            search_boxes.append(s_box)
            search_sz.append(t_search)
        search_sz = np.array(search_sz)
        for i, boxes in enumerate(search_boxes):
            # correspond to strack[i]
            for j, (t, l, w, h) in enumerate(boxes):
                cx, cy = t + w / 2, l + h / 2
                # don't consider outsiders
                if cx > 0 and cy > 0 and cx < search_sz[i] and cy < search_sz[i]:
                    pass
                else:
                    matrix[j][i] = 100
        return matrix

    def crop_and_resize_sec(self, track_xyxy, annos):
        search_area_factor = 6
        x1 = track_xyxy[0]
        y1 = track_xyxy[1]
        x2 = track_xyxy[2]
        y2 = track_xyxy[3]
        w = x2 - x1
        h = y2 - y1
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)
        x1_search = x1 + 0.5 * w - crop_sz * 0.5
        y1_search = y1 + 0.5 * h - crop_sz * 0.5

        transformed_coord_d = annos[:, 0:2] - np.array([x1_search, y1_search])
        annos_wh = annos[:, 2:4] - annos[:, 0:2]
        transformed_coord = np.concatenate((transformed_coord_d, annos_wh), axis=1)
        return transformed_coord, crop_sz

    def compute_mix_dist_sec(self, stracks, dets, matrix):
        search_boxes = []
        search_sz = []
        for strack in stracks:
            s_box, t_search = self.crop_and_resize_sec(strack, dets)
            search_boxes.append(s_box)
            search_sz.append(t_search)
        search_sz = np.array(search_sz)
        for i, boxes in enumerate(search_boxes):
            # correspond to strack[i]
            for j, (t, l, w, h) in enumerate(boxes):
                cx, cy = t + w / 2, l + h / 2
                # don't consider outsiders
                if cx > 0 and cy > 0 and cx < search_sz[i] and cy < search_sz[i]:
                    pass
                else:
                    matrix[j][i] = 1
        return matrix

    def compute_mix_dist_iou_sec(self, stracks, dets, matrix):
        search_boxes = []
        search_sz = []
        for strack in stracks:
            s_box, t_search = self.crop_and_resize_sec(strack, dets)
            search_sz.append(t_search)
        search_sz = np.array(search_sz)
        for i, boxes in enumerate(search_boxes):
            # correspond to strack[i]
            for j, (t, l, w, h) in enumerate(boxes):
                cx, cy = t + w / 2, l + h / 2  # 中心坐标
                # don't consider outsiders
                if cx > 0 and cy > 0 and cx < search_sz[i] and cy < search_sz[i]:
                    pass
                else:
                    matrix[j][i] = 0
        return matrix

    def DCM(self, dets, dets_feature, tracks, tracks_feature, tracks_information, last_box):
        det_second_mask = get_deep_range_DCM(dets)
        track_second_mask = get_deep_range_DCM(tracks)

        u_detection_sec, u_tracks_sec, u_det_emb_sec, u_trk_emb_sec, u_last_box_sec, u_track_information_sec, \
        res_det_sec, res_track_sec, res_track_information_sec, res_det_emb_sec, res_trk_emb_sec, \
        res_last_box_sec = [], [], [], [], [], [], [], [], [], [], [], []
        if len(track_second_mask) != 0:
            if len(track_second_mask) < len(det_second_mask):
                for i in range(len(det_second_mask) - len(track_second_mask)):
                    idx = np.argwhere(
                        det_second_mask[
                            len(track_second_mask) + i] == True)
                    for idd in idx:
                        res_det_sec.append(dets[idd[0]])
                    for idd_emb in idx:
                        res_det_emb_sec.append(dets_feature[idd_emb[0]])
            elif len(track_second_mask) > len(det_second_mask):
                for i in range(len(track_second_mask) - len(det_second_mask)):
                    idx = np.argwhere(track_second_mask[len(det_second_mask) + i] == True)
                    for idd in idx:
                        res_track_sec.append(tracks[idd[0]])
                    for idd_inf in idx:
                        res_track_information_sec.append(tracks_information[idd_inf[0]])
                    for idd_emb_t in idx:
                        res_trk_emb_sec.append(tracks_feature[idd_emb_t[0]])
                    for idd_lb in idx:
                        res_last_box_sec.append(last_box[idd_lb[0]])

            for i, (dm, tm) in enumerate(zip(det_second_mask, track_second_mask)):
                det_idx = np.argwhere(dm == True)
                trk_idx = np.argwhere(tm == True)

                # search det
                det_sec_ = []
                for idd in det_idx:
                    det_sec_.append(dets[idd[0]])
                det_sec_ = det_sec_ + u_detection_sec

                det_emb_sec_ = []
                for x in det_idx:
                    det_emb_sec_.append(dets_feature[x[0]])
                det_emb_sec_ = det_emb_sec_ + u_det_emb_sec

                # search trk
                track_sec_ = []
                for idt in trk_idx:
                    track_sec_.append(tracks[idt[0]])
                track_sec_ = track_sec_ + u_tracks_sec

                track_information_sec_ = []
                for idt_inf in trk_idx:
                    track_information_sec_.append(tracks_information[idt_inf[0]])
                track_information_sec_ = track_information_sec_ + u_track_information_sec

                trk_emb_sec_ = []
                for y in trk_idx:
                    trk_emb_sec_.append(tracks_feature[y[0]])
                trk_emb_sec_ = trk_emb_sec_ + u_trk_emb_sec

                last_box_sec_ = []
                for n in trk_idx:
                    last_box_sec_.append(last_box[n[0]])
                last_box_sec_ = last_box_sec_ + u_last_box_sec

                if len(det_sec_) != 0 and len(track_sec_) != 0:
                    iou_left = Biou(np.array(det_sec_), np.array(track_sec_), level=2)
                    iou_left = np.array(iou_left)
                    # iou_left = AUG_matrix(iou_left)
                    iou_area = Biou(np.array(det_sec_), np.array(track_sec_), level=2) * tb_dist(np.array(det_sec_),
                                                                                                 np.array(track_sec_))
                    # iou_area = AUG_matrix(iou_area)

                    # iou_left = self.compute_mix_dist_iou_sec(np.array(track_sec_), np.array(det_sec_), iou_left)

                    if iou_left.max() > self.iou_threshold:
                        """
                            NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                            get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                            uniform here for simplicity
                        """
                        if self.args.TCM_byte_step:
                            iou_left_ori = copy.deepcopy(iou_left)
                            iou_left -= np.array(cal_score_dif_batch_two_score(np.array(det_sec_), np.array(
                                track_sec_)) * self.args.TCM_byte_step_weight)
                            iou_left_thre = iou_left
                        if self.args.EG_weight_low_score > 0:
                            u_track_features = np.asarray(
                                [track.smooth_feat for track in track_information_sec_], dtype=np.float)
                            # u_track_features = self.feature_change(u_track_features, track_information_sec_)
                            emb_dists_low_score = embedding_distance(u_track_features,
                                                                     np.array(det_emb_sec_)).T

                            # emb_dists_low_score = self.compute_mix_dist_sec(np.array(track_sec_), np.array(det_sec_), emb_dists_low_score_original)

                            matched_indices = linear_assignment(
                                -iou_left - iou_area + self.args.EG_weight_low_score * emb_dists_low_score,
                            )
                        else:
                            matched_indices = linear_assignment(-iou_left)
                        # to_remove_trk_indices = []
                        for m in matched_indices:
                            # det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                            det_ind, trk_ind = m[0], m[1]
                            if self.args.with_longterm_reid_correction and self.args.EG_weight_low_score > 0:
                                if iou_left_thre[m[0], m[1]] < self.iou_threshold or emb_dists_low_score[m[0], m[1]] > self.args.longterm_reid_correction_thresh_low:
                                    print("correction 2nd:", emb_dists_low_score[m[0], m[1]])
                                    continue
                            else:
                                if iou_left_thre[m[0], m[1]] < self.iou_threshold:
                                    continue
                            d_sec = det_sec_[det_ind]
                            d_emb_sec = det_emb_sec_[det_ind]
                            t_inf = track_information_sec_[trk_ind]
                            t_inf.update(d_sec[0:5], d_emb_sec, update_feature=False)

                        # unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
                        unmatched_trackers_still_indices = []
                        unmatched_detections_still_indices = []
                        for t, trk in enumerate(track_sec_):
                            if (t not in matched_indices[:, 1]):
                                unmatched_trackers_still_indices.append(t)
                        for d, det in enumerate(det_sec_):
                            if (d not in matched_indices[:, 0]):
                                unmatched_detections_still_indices.append(d)

                        u_tracks_sec = [track_sec_[t] for t in unmatched_trackers_still_indices]
                        u_detection_sec = [det_sec_[t] for t in unmatched_detections_still_indices]
                        u_track_information_sec = [track_information_sec_[t] for t in unmatched_trackers_still_indices]
                        u_det_emb_sec = [det_emb_sec_[t] for t in unmatched_detections_still_indices]
                        u_trk_emb_sec = [trk_emb_sec_[t] for t in unmatched_trackers_still_indices]
                        u_last_box_sec = [last_box_sec_[t] for t in unmatched_trackers_still_indices]

                elif len(det_sec_) == 0:
                    u_tracks_sec = track_sec_
                    u_track_information_sec = track_information_sec_
                    u_trk_emb_sec = trk_emb_sec_
                    u_last_box_sec = last_box_sec_
                elif len(track_sec_) == 0:
                    u_detection_sec = det_sec_
                    u_det_emb_sec = det_emb_sec_

            unmatched_trks = u_tracks_sec + res_track_sec
            u_detection_sec = u_detection_sec + res_det_sec
            unmatched_trks_information = u_track_information_sec + res_track_information_sec
            u_det_emb_sec = u_det_emb_sec + res_det_emb_sec
            u_trk_emb_sec = u_trk_emb_sec + res_trk_emb_sec
            unmatched_trks_last_box = u_last_box_sec + res_last_box_sec

            # unmatched_trks = np.array(unmatched_trks)
            # unmatched_trks_last_box = np.array(unmatched_trks_last_box)

        return unmatched_trks, u_detection_sec, unmatched_trks_information, u_det_emb_sec, u_trk_emb_sec, unmatched_trks_last_box


    def associate_4_points_with_score_with_reid_change(self, detections, trackers, iou_threshold, lt, rt, lb, rb,
                                                       previous_obs, vdc_weight,
                                                       iou_type=None, args=None, trk_emb=None, det_emb=None, last_box=None,
                                                       weights=(1.0, 0), thresh=0.8,
                                                       with_longterm_reid=False,
                                                       longterm_reid_weight=0.0, with_longterm_reid_correction=False,
                                                       longterm_reid_correction_thresh=0.0, dataset="dancetrack"):

        if len(detections) > 0:
            det_mask = get_deep_range(detections)
        else:
            det_mask = []

        if len(trackers) != 0:
            track_mask = get_deep_range(trackers)
        else:
            track_mask = []

        u_detection, u_tracks, u_previous, u_lt, u_rt, u_lb, u_rb, u_det_emb, u_trk_emb, u_track_information, res_det, res_track, res_track_information, res_det_emb, res_trk_emb, res_last_box, u_last_box = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        if len(track_mask) != 0:
            if len(track_mask) < len(det_mask):
                for i in range(len(det_mask) - len(track_mask)):
                    idx = np.argwhere(
                        det_mask[len(track_mask) + i] == True)
                    for idd in idx:
                        res_det.append(detections[idd[0]])
                    for idd_emb in idx:
                        res_det_emb.append(det_emb[idd_emb[0]])
            elif len(track_mask) > len(det_mask):
                for i in range(len(track_mask) - len(det_mask)):
                    idx = np.argwhere(track_mask[len(det_mask) + i] == True)
                    for idd in idx:
                        res_track.append(trackers[idd[0]])
                    for idd_inf in idx:
                        res_track_information.append(self.trackers[idd_inf[0]])
                    for idd_emb_t in idx:
                        res_trk_emb.append(trk_emb[idd_emb_t[0]])
                    for idd_lb in idx:
                        res_last_box.append(last_box[idd_lb[0]])

            for dm, tm in zip(det_mask, track_mask):
                det_idx = np.argwhere(dm == True)
                trk_idx = np.argwhere(tm == True)

                # search det
                det_ = []
                for idd in det_idx:
                    det_.append(detections[idd[0]])
                det_ = det_ + u_detection

                det_emb_ = []
                for x in det_idx:
                    det_emb_.append(det_emb[x[0]])
                det_emb_ = det_emb_ + u_det_emb

                # search trk
                track_ = []
                for idt in trk_idx:
                    track_.append(trackers[idt[0]])
                track_ = track_ + u_tracks

                track_information_ = []
                for idt_inf in trk_idx:
                    track_information_.append(self.trackers[idt_inf[0]])
                track_information_ = track_information_ + u_track_information

                trk_emb_ = []
                for y in trk_idx:
                    trk_emb_.append(trk_emb[y[0]])
                trk_emb_ = trk_emb_ + u_trk_emb

                last_box_ = []
                for n in trk_idx:
                    last_box_.append(last_box[n[0]])
                last_box_ = last_box_ + u_last_box

                # search previous_obs
                pre_obs_ = []
                for idt_obs in trk_idx:
                    pre_obs_.append(previous_obs[idt_obs[0]])
                pre_obs_ = pre_obs_ + u_previous

                lt_, rt_, lb_, rb_ = [], [], [], []
                for i in trk_idx:
                    lt_.append(lt[i[0]])
                lt_ = lt_ + u_lt

                for j in trk_idx:
                    rt_.append(rt[j[0]])
                rt_ = rt_ + u_rt

                for k in trk_idx:
                    lb_.append(lb[k[0]])
                lb_ = lb_ + u_lb

                for l in trk_idx:
                    rb_.append(rb[l[0]])
                rb_ = rb_ + u_rb

                if len(det_) != 0 and len(track_) != 0:
                    Y1, X1 = speed_direction_batch_lt(np.array(det_), np.array(pre_obs_))
                    Y2, X2 = speed_direction_batch_rt(np.array(det_), np.array(pre_obs_))
                    Y3, X3 = speed_direction_batch_lb(np.array(det_), np.array(pre_obs_))
                    Y4, X4 = speed_direction_batch_rb(np.array(det_), np.array(pre_obs_))
                    cost_lt = cost_vel(Y1, X1, np.array(track_), np.array(lt_), np.array(det_), np.array(pre_obs_), vdc_weight)
                    cost_rt = cost_vel(Y2, X2, np.array(track_), np.array(rt_), np.array(det_), np.array(pre_obs_), vdc_weight)
                    cost_lb = cost_vel(Y3, X3, np.array(track_), np.array(lb_), np.array(det_), np.array(pre_obs_), vdc_weight)
                    cost_rb = cost_vel(Y4, X4, np.array(track_), np.array(rb_), np.array(det_), np.array(pre_obs_), vdc_weight)

                    iou_matrix = Biou(np.array(det_), np.array(track_), level=1)

                    score_dif = cal_score_dif_batch(np.array(det_), np.array(track_))
                    angle_diff_cost = cost_lt + cost_rt + cost_lb + cost_rb

                    # TCM
                    angle_diff_cost -= score_dif * args.TCM_first_step_weight

                    emb_cost = embedding_distance(np.array(trk_emb_), np.array(det_emb_)).T
                    emb_cost = self.compute_mix_dist(np.array(track_), np.array(det_), emb_cost)
                    iou_matrix = self.compute_mix_dist_iou(np.array(track_), np.array(det_), iou_matrix)

                    if min(iou_matrix.shape) > 0:
                        if emb_cost is None:
                            a = (iou_matrix > iou_threshold).astype(np.int32)
                            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                                matched_indices = np.stack(np.where(a), axis=1)
                            else:
                                matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
                        else:
                            if not with_longterm_reid:
                                matched_indices = linear_assignment(
                                    weights[0] * (-(iou_matrix + angle_diff_cost)) + weights[1] * emb_cost)  # , thresh=thresh
                                # matched_indices = linear_assignment(cost_matrix, thresh=10)
                            else:  # long-term reid feats
                                matched_indices = linear_assignment(weights[0] * (-(iou_matrix + angle_diff_cost)) +
                                                                    weights[1] * emb_cost + longterm_reid_weight * long_emb_dists)  # , thresh=thresh

                        if matched_indices.size == 0:
                            matched_indices = np.empty(shape=(0, 2))
                    else:
                        matched_indices = np.empty(shape=(0, 2))

                    unmatched_detections = []
                    for d, det in enumerate(np.array(det_)):
                        if (d not in matched_indices[:, 0]):
                            unmatched_detections.append(d)
                    unmatched_trackers = []
                    for t, trk in enumerate(np.array(track_)):
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

                    for idet, itracked in matches:
                        det = det_[idet]
                        track = track_information_[itracked]
                        det_embedding = det_emb_[idet]
                        track.update(det[0:5], det_embedding)

                    u_tracks = [track_[t] for t in unmatched_trackers]
                    u_detection = [det_[t] for t in unmatched_detections]
                    u_track_information = [track_information_[t] for t in unmatched_trackers]

                    u_previous = [pre_obs_[t] for t in unmatched_trackers]
                    u_lt = [lt_[t] for t in unmatched_trackers]
                    u_rt = [rt_[t] for t in unmatched_trackers]
                    u_lb = [lb_[t] for t in unmatched_trackers]
                    u_rb = [rb_[t] for t in unmatched_trackers]
                    u_det_emb = [det_emb_[t] for t in unmatched_detections]
                    u_trk_emb = [trk_emb_[t] for t in unmatched_trackers]
                    u_last_box = [last_box_[t] for t in unmatched_trackers]
                elif len(det_) == 0:
                    u_tracks = track_
                    u_track_information = track_information_
                    u_previous = pre_obs_
                    u_lt = lt_
                    u_rt = rt_
                    u_lb = lb_
                    u_rb = rb_
                    u_trk_emb = trk_emb_
                    u_last_box = last_box_
                elif len(track_) == 0:
                    u_detection = det_
                    u_det_emb = det_emb_

            u_tracks = u_tracks + res_track
            u_detection = u_detection + res_det
            u_track_information = u_track_information + res_track_information
            u_det_emb = u_det_emb + res_det_emb
            u_trk_emb = u_trk_emb + res_trk_emb
            u_last_box = u_last_box + res_last_box

        else:
            u_detection = detections
            u_det_emb = det_emb

        return np.array(u_detection), np.array(u_tracks), u_track_information, np.array(u_det_emb), np.array(u_trk_emb), np.array(u_last_box)


    def update(self, output_results, img_info, img_size, id_feature=None, warp_matrix=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))

        if self.args.ECC:
            # camera update for all stracks
            if warp_matrix is not None:
                self.camera_update(self.trackers, warp_matrix)

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        dets_original = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        all_dets = copy.deepcopy(bboxes)
        det_to_det_iou = iou_batch_level(all_dets, all_dets)
        max_iou_number = []
        for i in range(len(all_dets)):
            det_to_det_iou[i][i] = 0
            # max_iou_number.append(sum(det_to_det_iou[i] > 0.1))
            max_iou_number.append(det_to_det_iou[i].max())
        max_iou_number = np.array(max_iou_number)
        dets = np.concatenate((dets_original, np.expand_dims(max_iou_number, axis=-1)), axis=1)
        # dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        inds_low = scores > self.args.low_thresh
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        id_feature_keep = id_feature[remain_inds]
        id_feature_second = id_feature[inds_second]
        # max_iou_number_keep = max_iou_number[remain_inds]
        # max_iou_number_second = max_iou_number[inds_second]

        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos, kalman_score, simple_score = self.trackers[t].predict()
            try:
                trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score, simple_score[0]]
            except:
                trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score, simple_score]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        all_trks = copy.deepcopy(trks[..., 0:4])
        trk_to_trk_iou = iou_batch_level(all_trks, all_trks)
        max_iou_number_trk = []
        for i in range(len(all_trks)):
            trk_to_trk_iou[i][i] = 0
            # max_iou_number_trk.append(sum(trk_to_trk_iou[i] > 0.1))
            max_iou_number_trk.append(trk_to_trk_iou[i].max())
        max_iou_number_trk = np.array(max_iou_number_trk)
        trks = np.concatenate((trks, np.expand_dims(max_iou_number_trk, axis=-1)), axis=1)

        for t in reversed(to_del):
            self.trackers.pop(t)
        velocities_lt = np.array(
            [trk.velocity_lt if trk.velocity_lt is not None else np.array((0, 0)) for trk in self.trackers])
        velocities_rt = np.array(
            [trk.velocity_rt if trk.velocity_rt is not None else np.array((0, 0)) for trk in self.trackers])
        velocities_lb = np.array(
            [trk.velocity_lb if trk.velocity_lb is not None else np.array((0, 0)) for trk in self.trackers])
        velocities_rb = np.array(
            [trk.velocity_rb if trk.velocity_rb is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association  考虑IOU,height,vel,scores,emb
        """
        if self.args.EG_weight_high_score > 0 and self.args.TCM_first_step:
            track_features = np.asarray([track.smooth_feat for track in self.trackers],
                                        dtype=np.float)
            # track_features = self.feature_change(track_features, self.trackers)

            if self.args.with_longterm_reid or self.args.with_longterm_reid_correction:
                long_track_features = np.asarray([np.vstack(list(track.features)).mean(0) for track in self.trackers],
                                                 dtype=np.float)
                assert track_features.shape == long_track_features.shape
                # long_emb_dists = embedding_distance(long_track_features, id_feature_keep).T
                # assert emb_dists.shape == long_emb_dists.shape
                unmatched_dets, unmatched_trks, unmatched_trks_information, unmatched_dets_emb, unmatched_trks_emb, unmatched_trks_last_box = self.associate_4_points_with_score_with_reid_change(
                    dets, trks, self.iou_threshold, velocities_lt, velocities_rt, velocities_lb, velocities_rb,
                    k_observations, self.inertia, self.asso_func, self.args, trk_emb=track_features, det_emb=id_feature_keep, last_box=last_boxes,
                    weights=(1.0, self.args.EG_weight_high_score), thresh=self.args.high_score_matching_thresh,
                    with_longterm_reid=self.args.with_longterm_reid,
                    longterm_reid_weight=self.args.longterm_reid_weight,
                    with_longterm_reid_correction=self.args.with_longterm_reid_correction,
                    longterm_reid_correction_thresh=self.args.longterm_reid_correction_thresh,
                    dataset=self.args.dataset)

        """
            Second round of associaton by OCR  考虑IOU,height,scores,emb
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            det_second_mask = get_deep_range_low(dets_second)
            track_second_mask = get_deep_range_low(unmatched_trks)

            u_detection_sec, u_tracks_sec, u_det_emb_sec, u_trk_emb_sec, u_last_box_sec, u_track_information_sec, \
            res_det_sec, res_track_sec, res_track_information_sec, res_det_emb_sec, res_trk_emb_sec, \
            res_last_box_sec = [], [], [], [], [], [], [], [], [], [], [], []
            if len(track_second_mask) != 0:
                if len(track_second_mask) < len(det_second_mask):
                    for i in range(len(det_second_mask) - len(track_second_mask)):
                        idx = np.argwhere(
                            det_second_mask[len(track_second_mask) + i] == True)
                        for idd in idx:
                            res_det_sec.append(dets_second[idd[0]])
                        for idd_emb in idx:
                            res_det_emb_sec.append(id_feature_second[idd_emb[0]])
                elif len(track_second_mask) > len(det_second_mask):
                    for i in range(len(track_second_mask) - len(det_second_mask)):
                        idx = np.argwhere(track_second_mask[len(det_second_mask) + i] == True)
                        for idd in idx:
                            res_track_sec.append(unmatched_trks[idd[0]])
                        for idd_inf in idx:
                            res_track_information_sec.append(unmatched_trks_information[idd_inf[0]])
                        for idd_emb_t in idx:
                            res_trk_emb_sec.append(unmatched_trks_emb[idd_emb_t[0]])
                        for idd_lb in idx:
                            res_last_box_sec.append(unmatched_trks_last_box[idd_lb[0]])

                for i, (dm, tm) in enumerate(zip(det_second_mask, track_second_mask)):  # 遍历每一个伪深度层
                    det_idx = np.argwhere(dm == True)
                    trk_idx = np.argwhere(tm == True)

                    # search det
                    det_sec_ = []
                    for idd in det_idx:
                        det_sec_.append(dets_second[idd[0]])
                    det_sec_ = det_sec_ + u_detection_sec

                    det_emb_sec_ = []
                    for x in det_idx:
                        det_emb_sec_.append(id_feature_second[x[0]])
                    det_emb_sec_ = det_emb_sec_ + u_det_emb_sec

                    # search trk
                    track_sec_ = []
                    for idt in trk_idx:
                        track_sec_.append(unmatched_trks[idt[0]])
                    track_sec_ = track_sec_ + u_tracks_sec

                    track_information_sec_ = []
                    for idt_inf in trk_idx:
                        track_information_sec_.append(unmatched_trks_information[idt_inf[0]])
                    track_information_sec_ = track_information_sec_ + u_track_information_sec

                    trk_emb_sec_ = []
                    for y in trk_idx:
                        trk_emb_sec_.append(unmatched_trks_emb[y[0]])
                    trk_emb_sec_ = trk_emb_sec_ + u_trk_emb_sec

                    last_box_sec_ = []
                    for n in trk_idx:
                        last_box_sec_.append(unmatched_trks_last_box[n[0]])
                    last_box_sec_ = last_box_sec_ + u_last_box_sec

                    if len(det_sec_) != 0 and len(track_sec_) != 0:
                        iou_left = Biou(np.array(det_sec_), np.array(track_sec_), level=2)
                        iou_left = np.array(iou_left)
                        iou_area = Biou(np.array(det_sec_), np.array(track_sec_), level=2) * tb_dist(np.array(det_sec_), np.array(track_sec_))

                        iou_left = self.compute_mix_dist_iou_sec(np.array(track_sec_), np.array(det_sec_), iou_left)

                        if iou_left.max() > self.iou_threshold:
                            """
                                NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                                get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                                uniform here for simplicity
                            """
                            if self.args.TCM_byte_step:
                                iou_left_ori = copy.deepcopy(iou_left)
                                iou_left -= np.array(cal_score_dif_batch_two_score(np.array(det_sec_), np.array(track_sec_)) * self.args.TCM_byte_step_weight)  # HMIOU矩阵 - tracks最后一次匹配上的box的scores 与低分box之间的差值绝对值
                                iou_left_thre = iou_left

                            if self.args.EG_weight_low_score > 0:
                                u_track_features = np.asarray(
                                    [track.smooth_feat for track in track_information_sec_], dtype=np.float)
                                emb_dists_low_score = embedding_distance(u_track_features,
                                                                         np.array(det_emb_sec_)).T  # 计算cos距离矩阵，越小越好

                                emb_dists_low_score = self.compute_mix_dist_sec(np.array(track_sec_), np.array(det_sec_), emb_dists_low_score)

                                matched_indices = linear_assignment(
                                    -iou_left - iou_area + self.args.EG_weight_low_score * emb_dists_low_score,
                                    )
                            else:
                                matched_indices = linear_assignment(-iou_left)
                            # to_remove_trk_indices = []
                            for m in matched_indices:
                                # det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                                det_ind, trk_ind = m[0], m[1]
                                if self.args.with_longterm_reid_correction and self.args.EG_weight_low_score > 0:
                                    if iou_left_thre[m[0], m[1]] < self.iou_threshold or emb_dists_low_score[m[0], m[1]] > self.args.longterm_reid_correction_thresh_low:
                                        print("correction 2nd:", emb_dists_low_score[m[0], m[1]])
                                        continue
                                else:
                                    if iou_left_thre[m[0], m[1]] < self.iou_threshold:
                                        continue
                                d_sec = det_sec_[det_ind]
                                d_emb_sec = det_emb_sec_[det_ind]
                                t_inf = track_information_sec_[trk_ind]
                                t_inf.update(d_sec[0:5], d_emb_sec, update_feature=False)

                            # unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))
                            unmatched_trackers_still_indices = []
                            unmatched_detections_still_indices = []
                            for t, trk in enumerate(track_sec_):
                                if (t not in matched_indices[:, 1]):
                                    unmatched_trackers_still_indices.append(t)
                            for d, det in enumerate(det_sec_):
                                if (d not in matched_indices[:, 0]):
                                    unmatched_detections_still_indices.append(d)
                            u_tracks_sec = [track_sec_[t] for t in unmatched_trackers_still_indices]
                            u_detection_sec = [det_sec_[t] for t in unmatched_detections_still_indices]
                            u_track_information_sec = [track_information_sec_[t] for t in unmatched_trackers_still_indices]
                            u_det_emb_sec = [det_emb_sec_[t] for t in unmatched_detections_still_indices]
                            u_trk_emb_sec = [trk_emb_sec_[t] for t in unmatched_trackers_still_indices]
                            u_last_box_sec = [last_box_sec_[t] for t in unmatched_trackers_still_indices]

                    elif len(det_sec_) == 0:
                        u_tracks_sec = track_sec_
                        u_track_information_sec = track_information_sec_
                        u_trk_emb_sec = trk_emb_sec_
                        u_last_box_sec = last_box_sec_
                    elif len(track_sec_) == 0:
                        u_detection_sec = det_sec_
                        u_det_emb_sec = det_emb_sec_

                unmatched_trks = u_tracks_sec + res_track_sec
                u_detection_sec = u_detection_sec + res_det_sec
                unmatched_trks_information = u_track_information_sec + res_track_information_sec
                u_det_emb_sec = u_det_emb_sec + res_det_emb_sec
                u_trk_emb_sec = u_trk_emb_sec + res_trk_emb_sec
                unmatched_trks_last_box = u_last_box_sec + res_last_box_sec

                unmatched_trks = np.array(unmatched_trks)
                unmatched_trks_last_box = np.array(unmatched_trks_last_box)

        # 第三阶段 未匹配的tracks与高分框进行匹配
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:

            iou_left = Biou(unmatched_dets, unmatched_trks_last_box)
            iou_left = np.array(iou_left)

            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                for m in rematched_indices:
                    det_ind, trk_ind = m[0], m[1]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    unmatched_trks_information[trk_ind].update(unmatched_dets[det_ind, 0:5], unmatched_dets_emb[det_ind, :], update_feature=False)

                unmatched_trackers_still_indices_third = []
                unmatched_detections_still_indices_third = []
                for t, trk in enumerate(unmatched_trks_last_box):
                    if (t not in rematched_indices[:, 1]):
                        unmatched_trackers_still_indices_third.append(t)
                for d, det in enumerate(unmatched_dets):
                    if (d not in rematched_indices[:, 0]):
                        unmatched_detections_still_indices_third.append(d)

                unmatched_dets = unmatched_dets[unmatched_detections_still_indices_third]
                unmatched_dets_emb = unmatched_dets_emb[unmatched_detections_still_indices_third]
                unmatched_trks_information = [unmatched_trks_information[index] for index in
                                              unmatched_trackers_still_indices_third]

        for un_trk in unmatched_trks_information:
            un_trk.update(None, None)

        for i_det, i_emb in zip(unmatched_dets, unmatched_dets_emb):
            trk = KalmanBoxTracker(i_det[0:5], i_emb, delta_t=self.delta_t, args=self.args)
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0][:4]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):  # track未失配且为稳定态，或者track未失配且为前三帧
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti\
              (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
          
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))


