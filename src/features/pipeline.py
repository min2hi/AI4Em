"""One source-bound native session and one raw feature path for every consumer."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from numbers import Integral
from time import perf_counter

from src.config import validate_config
from src.contracts import FeatureSample, FramePacket
from src.features.eye import EyeFeatureExtractor
from src.features.head_pose import HeadPoseEstimator
from src.features.landmarks import FaceLandmarkDetector
from src.features.mouth import MouthFeatureExtractor
from src.features.quality import QualityGate, validate_quality_config


class FeaturePipeline:
    def __init__(self, config: dict):
        self._config = deepcopy(config)
        validate_config(self._config)
        if 'quality' not in self._config:
            raise ValueError('A measured, frozen quality profile is required')
        validate_quality_config(self._config['quality'], require_frozen=True)
        self._quality = QualityGate(self._config['quality'])
        self._pose = HeadPoseEstimator(self._config['canonical_model_path'],
            self._config['camera_model'],
            max_reprojection_error_norm=self._config['max_reprojection_error_norm'],
            epsilon=self._config['epsilon'])
        self._eyes = EyeFeatureExtractor(self._config['epsilon'])
        self._mouth = MouthFeatureExtractor(self._config['epsilon'])
        # All pure validation/assets above must succeed before allocating native state.
        self._detector = FaceLandmarkDetector(self._config)
        self._closed = False
        self._clear_session()

    def _clear_session(self):
        self._frame_index = None
        self._last_landmarks = self._last_quality = None
        self._processed_frames = 0
        self._reasons = Counter()
        self._valid = Counter()
        self._timings = {name: {'count':0,'total_ms':0.0} for name in
                         ('detect','eyes','mouth','pose','quality','process')}

    @property
    def last_landmarks(self):
        return self._last_landmarks

    @property
    def last_quality(self):
        return self._last_quality

    @property
    def pose_estimator(self):
        return self._pose

    @property
    def stats(self):
        return {'closed':self._closed,'processed_frames':self._processed_frames,
                'detector':dict(self._detector.stats),
                'valid_frames':dict(self._valid),'quality_reasons':dict(self._reasons),
                'timings_ms':{name:{**value,'mean_ms':value['total_ms']/value['count']
                                   if value['count'] else None}
                              for name,value in self._timings.items()}}

    def _measure(self, name, callback, *args):
        start = perf_counter()
        try:
            return callback(*args)
        finally:
            self._timings[name]['count'] += 1
            self._timings[name]['total_ms'] += (perf_counter()-start)*1000

    def process(self, packet: FramePacket) -> FeatureSample:
        self._last_landmarks = self._last_quality = None
        if self._closed:
            raise RuntimeError('FeaturePipeline is closed')
        index = packet.frame_index
        if isinstance(index,bool) or not isinstance(index,Integral) or index<0:
            raise ValueError('frame_index must be a nonnegative integer')
        if self._frame_index is not None and index<=self._frame_index:
            raise ValueError('frame_index must strictly increase within a pipeline session')
        started = perf_counter()
        try:
            result = self._measure('detect',self._detector.detect,packet)
            self._frame_index = int(index)
            eyes = self._measure('eyes',self._eyes.extract,result)
            mouth = self._measure('mouth',self._mouth.extract,result)
            pose = self._measure('pose',self._pose.estimate,result)
            quality = self._measure('quality',self._quality.evaluate,packet,result,eyes,mouth,pose)
            left = eyes.ear_left if quality.left_eye_valid else math.nan
            right = eyes.ear_right if quality.right_eye_valid else math.nan
            sample = FeatureSample(packet.timestamp_ms,packet.frame_index,packet.source_id,
                left,right,(left+right)/2 if quality.left_eye_valid and quality.right_eye_valid else math.nan,
                mouth.mar if quality.mouth_valid else math.nan,
                pose.pitch if quality.pose_valid else math.nan,
                pose.yaw if quality.pose_valid else math.nan,
                pose.roll if quality.pose_valid else math.nan,
                result.face_detected,quality.left_eye_valid,quality.right_eye_valid,
                quality.mouth_valid,quality.pose_valid,pose.reprojection_error_norm)
            self._last_landmarks,self._last_quality = result,quality
            self._processed_frames += 1
            self._reasons.update(quality.reasons)
            for name in ('face_detected','left_eye_valid','right_eye_valid','mouth_valid','pose_valid'):
                self._valid[name] += int(getattr(sample,name))
            return sample
        finally:
            self._timings['process']['count'] += 1
            self._timings['process']['total_ms'] += (perf_counter()-started)*1000

    def reset(self) -> None:
        if self._closed:
            raise RuntimeError('FeaturePipeline is closed')
        self._last_landmarks = self._last_quality = None
        try:
            self._detector.close()
            self._detector = FaceLandmarkDetector(self._config)
        except Exception:
            self._closed = True
            raise
        self._clear_session()

    def close(self) -> None:
        self._last_landmarks = self._last_quality = None
        if not self._closed:
            self._closed = True
            self._detector.close()

    def __enter__(self):
        if self._closed:
            raise RuntimeError('FeaturePipeline is closed')
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
