# strongsort_tracker.py
import numpy as np
import cv2
from basetrack import BaseTrack, TrackState, Counting
from kalman_filter import KalmanFilter
import matching

import numpy as np
import cv2
from basetrack import BaseTrack, TrackState, Counting
from kalman_filter import KalmanFilter
import matching

def cosine_distance(a, b):
    """计算余弦距离"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = np.clip(np.dot(a, b.T), -1.0, 1.0)
    return 1.0 - sim

class ColorHistReID:
    """颜色直方图特征提取器"""
    def __init__(self, bins=(8, 8, 8)):  # 减少bins数量，提高鲁棒性
        self.bins = bins
        self.feature_dim = sum(bins)

    def extract(self, frame, xyxy):
        """提取颜色直方图特征"""
        try:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            h, w = frame.shape[:2] if frame is not None else (0, 0)
            
            # 边界检查
            x1 = max(0, min(x1, w-1))
            x2 = max(x1+1, min(x2, w))
            y1 = max(0, min(y1, h-1))
            y2 = max(y1+1, min(y2, h))
            
            if x2 <= x1 or y2 <= y1 or frame is None or frame.size == 0:
                return np.zeros(self.feature_dim, dtype=np.float32)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return np.zeros(self.feature_dim, dtype=np.float32)
            
            # 转换到HSV空间
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # 计算直方图
            hist_h = cv2.calcHist([hsv], [0], None, [self.bins[0]], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [self.bins[1]], [0, 256]).flatten()
            hist_v = cv2.calcHist([hsv], [2], None, [self.bins[2]], [0, 256]).flatten()
            
            # 连接并归一化
            feat = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
            norm = np.linalg.norm(feat) + 1e-12
            feat = feat / norm
            
            return feat
            
        except Exception as e:
            return np.zeros(self.feature_dim, dtype=np.float32)

class STrack(BaseTrack):
    """StrongSORT 轨迹类"""
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat, ctInst):
        # 基本属性
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = float(score)
        self.tracklet_len = 0
        self.countInst = ctInst
        
        # 外观特征
        self.curr_feat = feat
        self.smooth_feat = feat.copy() if feat is not None else None
        self.alpha = 0.95  # 增加平滑系数，减少特征变化
        
        # 添加位置平滑
        self.smooth_tlwh = self._tlwh.copy()
        self.position_alpha = 0.8  # 位置平滑系数

    def update_feature(self, feat):
        """更新外观特征（EMA平滑）"""
        if feat is None:
            return
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat.copy()
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        # 重新归一化
        norm = np.linalg.norm(self.smooth_feat) + 1e-12
        self.smooth_feat = self.smooth_feat / norm

    def predict(self):
        """预测下一帧位置"""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """批量预测"""
        if len(stracks) == 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """激活新轨迹"""
        self.kalman_filter = kalman_filter
        self.track_id = self.countInst.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        
        # 初始化平滑位置
        self.smooth_tlwh = self._tlwh.copy()

    def re_activate(self, new_track, frame_id, new_id=False):
        """重新激活轨迹"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.update_feature(new_track.curr_feat)
        
        # 更新平滑位置
        self.smooth_tlwh = self.position_alpha * self.smooth_tlwh + (1 - self.position_alpha) * new_track.tlwh
        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.countInst.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """更新匹配的轨迹"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.update_feature(new_track.curr_feat)
        
        # 更新平滑位置
        self.smooth_tlwh = self.position_alpha * self.smooth_tlwh + (1 - self.position_alpha) * new_tlwh
        
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    @property
    def tlwh(self):
        """获取当前边界框 (top left x, top left y, width, height)"""
        if self.mean is None:
            return self._tlwh.copy()
        
        # 使用卡尔曼滤波预测的位置
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        
        # 可选：使用平滑位置（如果跳动严重可以启用）
        # return self.smooth_tlwh.copy()
        
        return ret

    @property
    def tlbr(self):
        """转换为 (min x, min y, max x, max y) 格式"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """转换为 (center x, center y, aspect ratio, height) 格式"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= (ret[3] + 1e-12)
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """从 tlbr 转换为 tlwh"""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """从 tlwh 转换为 tlbr"""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

class StrongSORTTracker:
    """StrongSORT 多目标跟踪器"""
    
    def __init__(self, args, frame_rate=30, appear_weight=0.3):  # 降低外观权重
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        
        self.countingInstance = Counting()
        self.frame_id = 0
        self.args = args
        
        # 修改卡尔曼滤波器参数
        self.kalman_filter = KalmanFilter()
        # 降低过程噪声，使预测更稳定
        self.kalman_filter.std_weight_position = 1.0 / 20  # 默认是 1/20
        self.kalman_filter.std_weight_velocity = 1.0 / 160  # 默认是 1/160
        
        self.reid = ColorHistReID()
        
        # 参数设置
        self.track_thresh = getattr(args, 'track_thresh', 0.5)
        self.det_thresh = self.track_thresh + 0.1
        self.match_thresh = getattr(args, 'match_thresh', 0.8)
        self.iou_match_thresh = 0.7  # IoU匹配阈值
        self.track_buffer = getattr(args, 'track_buffer', 30)
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        
        # 权重设置 - 降低外观权重，增加IoU权重
        self.appear_weight = float(appear_weight)
        self.iou_weight = float(1.0 - appear_weight)
        
        # 运动一致性检查
        self.max_iou_distance = 0.7  # 最大IoU距离阈值

    def update(self, output_results, frame):
        """
        更新跟踪器
        Args:
            output_results: 检测结果 (N, 6) [x1,y1,x2,y2,score,cls]
            frame: 当前帧图像
        Returns:
            跟踪结果 (M, 5) [x1,y1,w,h,track_id]
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # 处理检测结果
        if len(output_results) == 0:
            output_results = np.zeros((0, 6), dtype=np.float64)
        
        # 分离高低分检测
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]  # x1y1x2y2
        
        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        
        dets = bboxes[remain_inds]
        dets_second = bboxes[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        
        # 创建检测对象
        if len(dets) > 0:
            detections = []
            for i, (bbox, s) in enumerate(zip(dets, scores_keep)):
                tlwh = self.tlbr_to_tlwh(bbox)
                feat = self.reid.extract(frame, bbox)
                detections.append(STrack(tlwh, s, feat, self.countingInstance))
        else:
            detections = []
        
        # 分离未确认和已跟踪的轨迹
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # 合并跟踪和丢失的轨迹
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        
        # 预测当前位置
        STrack.multi_predict(strack_pool)
        
        # 第一次关联：优先使用IoU匹配
        iou_dists = matching.iou_distance(strack_pool, detections)
        iou_dists = matching.fuse_score(iou_dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(iou_dists, thresh=self.iou_match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # 第二次关联：对未匹配的使用外观+IoU
        remaining_tracked = [strack_pool[i] for i in u_track]
        remaining_dets = [detections[i] for i in u_detection]
        
        if len(remaining_tracked) > 0 and len(remaining_dets) > 0:
            dists = self._fuse_distance(remaining_tracked, remaining_dets)
            matches, u_track2, u_detection2 = matching.linear_assignment(dists, thresh=self.match_thresh)
            
            for itracked, idet in matches:
                track = remaining_tracked[itracked]
                det = remaining_dets[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            
            # 更新未匹配索引
            u_track = [u_track[i] for i in u_track2]
            u_detection = [u_detection[i] for i in u_detection2]
        
        # 第三次关联：低分检测
        if len(dets_second) > 0:
            detections_second = []
            for bbox, s in zip(dets_second, scores_second):
                tlwh = self.tlbr_to_tlwh(bbox)
                feat = self.reid.extract(frame, bbox)
                detections_second.append(STrack(tlwh, s, feat, self.countingInstance))
        else:
            detections_second = []
        
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # 处理未匹配的轨迹
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        # 处理未确认的轨迹
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        # 初始化新轨迹
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        
        # 移除长时间丢失的轨迹
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        
        # 更新轨迹列表
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        # 输出结果
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        # 转换为输出格式 [x1,y1,w,h,track_id]
        result = []
        for track in output_stracks:
            tlwh = track.tlwh
            track_id = track.track_id
            result.append(np.append(tlwh, track_id))
        
        return np.array(result, dtype=np.float64) if len(result) > 0 else np.zeros((0, 5), dtype=np.float64)

    def _fuse_distance(self, tracks, detections):
        """融合IoU距离和外观距离"""
        if len(tracks) == 0 or len(detections) == 0:
            return np.ones((len(tracks), len(detections)), dtype=np.float64) * 1e6
        
        # IoU距离
        iou_dists = matching.iou_distance(tracks, detections)
        
        # 外观距离
        track_features = []
        for track in tracks:
            if track.smooth_feat is not None:
                track_features.append(track.smooth_feat)
            else:
                track_features.append(np.zeros(self.reid.feature_dim, dtype=np.float32))
        
        det_features = []
        for det in detections:
            if det.curr_feat is not None:
                det_features.append(det.curr_feat)
            else:
                det_features.append(np.zeros(self.reid.feature_dim, dtype=np.float32))
        
        track_features = np.array(track_features)
        det_features = np.array(det_features)
        
        appearance_dists = cosine_distance(track_features, det_features)
        
        # 融合距离 - 降低外观权重
        dists = self.appear_weight * appearance_dists + self.iou_weight * iou_dists
        
        # 融合检测分数
        dists = matching.fuse_score(dists, detections)
        
        return dists

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """边界框格式转换"""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    def joint_stracks(self, tlista, tlistb):
        """合并两个轨迹列表"""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    def sub_stracks(self, tlista, tlistb):
        """从tlista中减去tlistb中的轨迹"""
        stracks = {}
        for t in tlista:
            stracks[t.track_id] = t
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())

    def remove_duplicate_stracks(self, stracksa, stracksb):
        """移除重复的轨迹"""
        pdist = matching.iou_distance(stracksa, stracksb)
        if pdist.size == 0:
            return stracksa, stracksb
        pairs = np.where(pdist < 0.15)
        dupa, dupb = list(), list()
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb

# def cosine_distance(a, b):
#     # a: NxD, b: MxD
#     if len(a) == 0 or len(b) == 0:
#         return np.zeros((len(a), len(b)), dtype=np.float64)
#     a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
#     b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
#     sim = np.clip(np.dot(a, b.T), -1.0, 1.0)
#     return 1.0 - sim

# class ColorHistReID:
#     def __init__(self, bins=(16, 16, 16)):
#         self.bins = bins

#     def extract(self, frame, xyxy):
#         x1, y1, x2, y2 = [int(v) for v in xyxy]
#         h, w = frame.shape[:2]
#         x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
#         y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
#         if x2 <= x1 or y2 <= y1:
#             return np.zeros(sum(self.bins), dtype=np.float32)
#         crop = frame[y1:y2, x1:x2]
#         if crop.size == 0:
#             return np.zeros(sum(self.bins), dtype=np.float32)
#         hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#         hist_h = cv2.calcHist([hsv], [0], None, [self.bins[0]], [0, 180]).flatten()
#         hist_s = cv2.calcHist([hsv], [1], None, [self.bins[1]], [0, 256]).flatten()
#         hist_v = cv2.calcHist([hsv], [2], None, [self.bins[2]], [0, 256]).flatten()
#         feat = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
#         feat /= (np.linalg.norm(feat) + 1e-12)
#         return feat

# class DetBox:
#     def __init__(self, tlwh, score, feat, xyxy=None):
#         self._tlwh = np.asarray(tlwh, dtype=np.float64)
#         self.score = float(score)
#         self.feat = feat  # np.ndarray
#         self._xyxy = xyxy

#     @property
#     def tlwh(self):
#         return self._tlwh

#     @property
#     def tlbr(self):
#         ret = self._tlwh.copy()
#         ret[2:] += ret[:2]
#         return ret

#     def to_xyah(self):
#         ret = self._tlwh.copy()
#         ret[:2] += ret[2:] / 2
#         ret[2] /= (ret[3] + 1e-12)
#         return ret

#     @staticmethod
#     def tlbr_to_tlwh(tlbr):
#         ret = np.asarray(tlbr).copy()
#         ret[2:] -= ret[:2]
#         return ret

# class STrack(BaseTrack):
#     shared_kalman = KalmanFilter()

#     def __init__(self, tlwh, score, feat, ctInst):
#         self._tlwh = np.asarray(tlwh, dtype=np.float64)
#         self.kalman_filter = None
#         self.mean, self.covariance = None, None
#         self.is_activated = False

#         self.score = float(score)
#         self.tracklet_len = 0
#         self.countInst = ctInst

#         # appearance
#         self.curr_feat = feat.astype(np.float32) if feat is not None else None
#         self.smooth_feat = self.curr_feat.copy() if self.curr_feat is not None else None
#         self.alpha = 0.9  # EMA for features
#         self.ema_box_alpha = 0.0  # output smoothing off by default

#     def update_feature(self, feat):
#         if feat is None:
#             return
#         if self.smooth_feat is None:
#             self.smooth_feat = feat.copy()
#         else:
#             self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat

#     def predict(self):
#         mean_state = self.mean.copy()
#         if self.state != TrackState.Tracked:
#             mean_state[7] = 0
#         self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

#     @staticmethod
#     def multi_predict(stracks):
#         if len(stracks) == 0:
#             return
#         multi_mean = np.asarray([st.mean.copy() for st in stracks])
#         multi_covariance = np.asarray([st.covariance for st in stracks])
#         for i, st in enumerate(stracks):
#             if st.state != TrackState.Tracked:
#                 multi_mean[i][7] = 0
#         multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
#         for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
#             stracks[i].mean = mean
#             stracks[i].covariance = cov

#     def activate(self, kalman_filter, frame_id):
#         self.kalman_filter = kalman_filter
#         self.track_id = self.countInst.next_id()
#         self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
#         self.tracklet_len = 0
#         self.state = TrackState.Tracked
#         self.is_activated = True if frame_id == 1 else True
#         self.frame_id = frame_id
#         self.start_frame = frame_id

#     def re_activate(self, det: DetBox, frame_id, new_id=False):
#         self.mean, self.covariance = self.kalman_filter.update(
#             self.mean, self.covariance, self.tlwh_to_xyah(det.tlwh)
#         )
#         self.tracklet_len = 0
#         self.state = TrackState.Tracked
#         self.is_activated = True
#         self.frame_id = frame_id
#         if new_id:
#             self.track_id = self.countInst.next_id()
#         self.score = det.score
#         self.update_feature(det.feat)

#     def update(self, det: DetBox, frame_id):
#         self.frame_id = frame_id
#         self.tracklet_len += 1
#         self.mean, self.covariance = self.kalman_filter.update(
#             self.mean, self.covariance, self.tlwh_to_xyah(det.tlwh)
#         )
#         self.state = TrackState.Tracked
#         self.is_activated = True
#         self.score = det.score
#         self.update_feature(det.feat)

#     @property
#     def tlwh(self):
#         if self.mean is None:
#             return self._tlwh.copy()
#         ret = self.mean[:4].copy()
#         ret[2] *= ret[3]
#         ret[:2] -= ret[2:] / 2
#         return ret

#     @property
#     def tlbr(self):
#         ret = self.tlwh.copy()
#         ret[2:] += ret[:2]
#         return ret

#     @staticmethod
#     def tlwh_to_xyah(tlwh):
#         ret = np.asarray(tlwh).copy()
#         ret[:2] += ret[2:] / 2
#         ret[2] /= (ret[3] + 1e-12)
#         return ret

# class StrongSORTTracker(object):
#     def __init__(self, args=None, frame_rate=30, appear_weight=0.6, det_thresh=None, iou_weight=None):
#         self.tracked_stracks = []
#         self.lost_stracks = []
#         self.removed_stracks = []

#         self.countingInstance = Counting()
#         self.frame_id = 0
#         self.args = args

#         self.kalman_filter = KalmanFilter()
#         self.reid = ColorHistReID()

#         self.track_thresh = args.track_thresh if (args and hasattr(args, 'track_thresh')) else 0.5
#         self.det_thresh = det_thresh if det_thresh is not None else self.track_thresh
#         self.match_thresh = args.match_thresh if (args and hasattr(args, 'match_thresh')) else 0.9
#         self.buffer_size = int(frame_rate / 30.0 * (args.track_buffer if (args and hasattr(args, 'track_buffer')) else 30))
#         self.max_time_lost = self.buffer_size

#         self.appear_weight = float(appear_weight)
#         self.iou_weight = float(1.0 - appear_weight) if iou_weight is None else float(iou_weight)

#     def _build_detections(self, frame, dets_np):
#         # dets_np: Nx6 -> [x1,y1,x2,y2,score,cls]
#         dets = []
#         for d in dets_np:
#             x1, y1, x2, y2, s, c = d
#             if s < self.track_thresh:
#                 continue
#             tlwh = DetBox.tlbr_to_tlwh([x1, y1, x2, y2])
#             feat = self.reid.extract(frame, [x1, y1, x2, y2])
#             dets.append(DetBox(tlwh, s, feat, xyxy=[x1, y1, x2, y2]))
#         return dets

#     def _fuse_cost(self, tracks, detections):
#         # IoU distance
#         iou_cost = matching.iou_distance(tracks, detections)
#         # appearance distance
#         track_feats = []
#         for t in tracks:
#             if t.smooth_feat is None:
#                 track_feats.append(np.zeros(48, dtype=np.float32))  # matches default ColorHistReID length
#             else:
#                 track_feats.append(t.smooth_feat.astype(np.float32))
#         det_feats = [d.feat.astype(np.float32) for d in detections]
#         feat_cost = cosine_distance(np.asarray(track_feats), np.asarray(det_feats)) if len(detections) > 0 and len(tracks) > 0 else iou_cost
#         # fuse
#         cost = self.appear_weight * feat_cost + self.iou_weight * iou_cost
#         # motion gating
#         cost = matching.gate_cost_matrix(self.kalman_filter, cost, tracks, detections, only_position=True)
#         return cost

#     def update(self, dets_np, frame):
#         # dets_np: (N, 6) -> [x1,y1,x2,y2,score,cls], frame: HxWx3 BGR
#         self.frame_id += 1

#         activated_starcks = []
#         refind_stracks = []
#         lost_stracks = []
#         removed_stracks = []

#         detections = self._build_detections(frame, dets_np)

#         unconfirmed = []
#         tracked_stracks = []
#         for track in self.tracked_stracks:
#             if not track.is_activated:
#                 unconfirmed.append(track)
#             else:
#                 tracked_stracks.append(track)

#         strack_pool = tracked_stracks + self.lost_stracks
#         STrack.multi_predict(strack_pool)

#         # First association: appearance + IoU
#         dists = self._fuse_cost(strack_pool, detections)
#         matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

#         for itracked, idet in matches:
#             track = strack_pool[itracked]
#             det = detections[idet]
#             if track.state == TrackState.Tracked:
#                 track.update(det, self.frame_id)
#                 activated_starcks.append(track)
#             else:
#                 track.re_activate(det, self.frame_id, new_id=False)
#                 refind_stracks.append(track)

#         # Unmatched tracked tracks are marked lost
#         for it in u_track:
#             track = strack_pool[it]
#             if track.state != TrackState.Lost:
#                 track.mark_lost()
#                 lost_stracks.append(track)

#         # Match unconfirmed with remaining detections (appearance + IoU)
#         detections_remain = [detections[i] for i in u_detection]
#         dists = self._fuse_cost(unconfirmed, detections_remain)
#         matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=max(0.7, self.match_thresh))

#         for itracked, idet in matches:
#             track = unconfirmed[itracked]
#             det = detections_remain[idet]
#             track.update(det, self.frame_id)
#             activated_starcks.append(track)

#         for it in u_unconfirmed:
#             track = unconfirmed[it]
#             track.mark_removed()
#             removed_stracks.append(track)

#         # Init new tracks
#         for inew in u_detection:
#             det = detections_remain[inew]
#             if det.score < self.det_thresh:
#                 continue
#             new_track = STrack(det.tlwh, det.score, det.feat, self.countingInstance)
#             new_track.activate(self.kalman_filter, self.frame_id)
#             activated_starcks.append(new_track)

#         # Remove too long lost
#         for track in self.lost_stracks:
#             if self.frame_id - track.end_frame > self.max_time_lost:
#                 track.mark_removed()
#                 removed_stracks.append(track)

#         # Final states collections
#         self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
#         self.tracked_stracks = self.tracked_stracks + activated_starcks + refind_stracks
#         self.lost_stracks = [t for t in self.lost_stracks if t.state != TrackState.Removed]
#         self.lost_stracks = [t for t in self.lost_stracks if t.state != TrackState.Tracked] + lost_stracks
#         self.removed_stracks.extend(removed_stracks)

#         # De-duplicate
#         self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

#         output_stracks = [track for track in self.tracked_stracks if track.is_activated]
#         result = []
#         for track in output_stracks:
#             tlwh = track.tlwh
#             track_id = track.track_id
#             result.append(np.append(tlwh, track_id))
#         return np.array(result, dtype=np.float64)

# def remove_duplicate_stracks(stracksa, stracksb):
#     pdist = matching.iou_distance(stracksa, stracksb)
#     if pdist.size == 0:
#         return stracksa, stracksb
#     pairs = np.where(pdist < 0.15)
#     dupa, dupb = list(), list()
#     for p, q in zip(*pairs):
#         timep = stracksa[p].frame_id - stracksa[p].start_frame
#         timeq = stracksb[q].frame_id - stracksb[q].start_frame
#         if timep > timeq:
#             dupb.append(q)
#         else:
#             dupa.append(p)
#     resa = [t for i, t in enumerate(stracksa) if i not in dupa]
#     resb = [t for i, t in enumerate(stracksb) if i not in dupb]
#     return resa, resb