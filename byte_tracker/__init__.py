import numpy as np
from byte_tracker import matching
from byte_tracker.base_track import BaseTrack, TrackState
from byte_tracker.kalman_filter import KalmanFilter
from utils.general import xywh2xyxy, xyxy2xywh


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()  # Tüm STrack nesneleri tarafından paylaşılan ortak bir Kalman filtresi örneği

    def __init__(self, tlwh, score, cls):
        self._tlwh = np.asarray(tlwh, dtype=np.float)  # Takip nesnesinin başlangıç konumu (top-left width height formatında)
        self.kalman_filter = None  # Nesneye özgü Kalman filtresi (başlangıçta None)
        self.mean, self.covariance = None, None  # Kalman filtresinin durum vektörü ve kovaryans matrisi
        self.is_activated = False  # İz aktif edildi mi?
        self.score = score  # İzlenen nesnenin algılama skoru
        self.tracklet_len = 0  # İzlenen nesnenin yaşam süresi (kaç kare göründüğü)
        self.cls = cls  # İzlenen nesnenin sınıfı

    def predict(self):
        # İz için bir sonraki durumu tahmin et
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0  # Eğer nesne "takip edilen" durumda değilse, hızını sıfırla
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        # Birden fazla iz için tahmin yap
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])  # Tüm izlerin durum vektörleri
            multi_covariance = np.asarray([st.covariance for st in stracks])  # Tüm izlerin kovaryans matrisleri
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0  # Takip edilmeyen izler için hızı sıfırla
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        # İz aktifleştirme işlemi
        self.kalman_filter = kalman_filter  # Nesnenin Kalman filtresini ayarla
        self.track_id = self.next_id()  # Yeni bir iz ID'si ata
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))  # Kalman filtresini başlat
        self.tracklet_len = 0  # İzlenen nesnenin yaşam süresini sıfırla
        self.state = TrackState.Tracked  # İzin durumunu "takip edilen" olarak güncelle
        if frame_id == 1:
            self.is_activated = True  # Eğer ilk kare ise, izi aktif olarak işaretle
        self.frame_id = frame_id  # Güncel kare ID'sini ayarla
        self.start_frame = frame_id  # İzin başlangıç karesini ayarla

    def re_activate(self, new_track, frame_id, new_id=False):
        # İz yeniden aktifleştirme işlemi
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.tracklet_len = 0  # İzlenen nesnenin yaşam süresini sıfırla
        self.state = TrackState.Tracked  # İzin durumunu "takip edilen" olarak güncelle
        self.is_activated = True  # İzi aktif olarak işaretle
        self.frame_id = frame_id  # Güncel kare ID'sini ayarla
        if new_id:
            self.track_id = self.next_id()  # Yeni bir ID atama opsiyonu
        self.score = new_track.score  # İzin skorunu güncelle
        self.cls = new_track.cls  # İzin sınıfını güncelle

    def update(self, new_track, frame_id):
        # İz güncelleme işlemi
        self.frame_id = frame_id  # Güncel kare ID'sini ayarla
        self.tracklet_len += 1  # İzlenen nesnenin yaşam süresini arttır
        new_tlwh = new_track.tlwh  # Yeni konum bilgisi
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked  # İzin durumunu "takip edilen" olarak güncelle
        self.is_activated = True  # İzi aktif olarak işaretle
        self.score = new_track.score  # İzin skorunu güncelle

    @property
    def tlwh(self):
        # İzin top-left width height formatında konumunu döndür
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # Genişliği yükseklikle çarp
        ret[:2] -= ret[2:] / 2  # Merkez konumdan top-left konuma dönüştür
        return ret

    @property
    def tlbr(self):
        # İzin top-left bottom-right formatında konumunu döndür
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]  # Width ve height değerlerini top-left değerlerine ekle
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        # top-left width height formatını merkez-x, merkez-y, oran ve yükseklik formatına dönüştür
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2  # top-left konumdan merkeze dönüştür
        ret[2] /= ret[3]  # Genişliği yüksekliğe bölerek oranı hesapla
        return ret

    def to_xyah(self):
        # İzin merkez-x, merkez-y, oran ve yükseklik formatında konumunu döndür
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        # top-left bottom-right formatını top-left width height formatına dönüştür
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]  # Bottom-right değerlerinden top-left değerlerini çıkar
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        # top-left width height formatını top-left bottom-right formatına dönüştür
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]  # Width ve height değerlerini top-left değerlerine ekle
        return ret

    def __repr__(self):
        # İzin temsili stringini döndür
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)  # Örnek: "OT_1_(0-10)"

class BYTETracker(object):
    def __init__(self, track_thresh=0.45, track_buffer=30, match_thresh=0.8, frame_rate=25):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.track_buffer = track_buffer
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / 25.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, dets):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        xyxys = dets[:, 0:4]
        xywh = xyxy2xywh(xyxys)
        confs = dets[:, 4]
        clss = dets[:, 5]

        classes = clss
        xyxys = xyxys
        confs = confs

        remain_inds = confs > self.track_thresh
        inds_low = confs > 0.1
        inds_high = confs < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = xywh[inds_second]
        dets = xywh[remain_inds]

        scores_keep = confs[remain_inds]
        scores_second = confs[inds_second]

        clss_keep = classes[remain_inds]
        clss_second = classes[remain_inds]

        if len(dets) > 0:
            detections = [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores_keep, clss_keep)]
        else:
            detections = []

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        if len(dets_second) > 0:
            detections_second = [STrack(xywh, s, c) for (xywh, s, c) in zip(dets_second, scores_second, clss_second)]
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

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

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

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        outputs = []

        for t in output_stracks:
            output = []
            tlwh = t.tlwh
            tid = t.track_id
            tlwh = np.expand_dims(tlwh, axis=0)
            xyxy = xywh2xyxy(tlwh)
            xyxy = np.squeeze(xyxy, axis=0)
            output.extend(xyxy)
            output.append(tid)
            output.append(t.cls)
            output.append(t.score)
            outputs.append(output)

        outputs = np.array(outputs)
        return outputs

def joint_stracks(tlista, tlistb):
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

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb