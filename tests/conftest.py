"""Independent geometry and external native model fixtures for pipeline consumers."""
from types import SimpleNamespace
import hashlib
import json
import cv2
import numpy as np
import pytest
from mediapipe.tasks.python import vision


@pytest.fixture
def feature_backend(tmp_path, monkeypatch):
    ids = (1,152,33,263,61,291)
    literal = np.array([(0,-1.126865,7.475604),(0,-9.403378,4.264492),
        (-4.445859,2.663991,3.173422),(4.445859,2.663991,3.173422),
        (-2.456206,-4.342621,4.283884),(2.456206,-4.342621,4.283884)])
    vertices = np.zeros((468,3))
    vertices[list(ids)] = literal
    obj = tmp_path/'canonical.obj'
    obj.write_text('\n'.join(f'v {x} {y} {z}' for x,y,z in vertices),encoding='utf-8')
    matrix = [[800,0,320],[0,800,240],[0,0,1]]
    xy,_ = cv2.projectPoints(literal*(1,-1,-1),np.zeros(3),np.array([0.,0.,60.]),
                             np.array(matrix,dtype=float),np.zeros(5))
    points = np.zeros((478,3),np.float32)
    points[list(ids),:2] = xy.reshape(6,2)/(640,480)
    for index,xy in ((234,(.25,.5)),(454,(.75,.5)),(10,(.5,.25)),
                    (78,(.46,.56)),(308,(.54,.56)),(82,(.48,.55)),(87,(.48,.57)),
                    (13,(.5,.55)),(14,(.5,.57)),(312,(.52,.55)),(317,(.52,.57))):
        points[index,:2] = xy
    # Horizontal endpoints33/263 remain the independent six-point projection.
    for outer,other,uppers,lowers,direction in ((33,133,(160,158),(144,153),1),
                                             (263,362,(385,387),(380,373),-1)):
        base = points[outer,:2].copy()
        points[other,:2] = base+(direction*.04,0)
        for i,upper,lower in zip((1,2),uppers,lowers):
            points[upper,:2] = base+(direction*.04*i/3,-.01)
            points[lower,:2] = base+(direction*.04*i/3,.01)
    quality = dict(policy_version='raw_quality_v1',reference_max_side=256,
        min_blur_variance=10.,brightness_min=40.,brightness_max=200.,
        min_eye_width_ratio=.01,max_abs_yaw_deg=35.,max_abs_pitch_deg=25.,eyes_occluded=False)
    report = tmp_path/'quality.json'
    report.write_text(json.dumps({'quality':quality}),encoding='utf-8')
    quality = {**quality,'frozen_report_path':report,
               'frozen_report_sha256':hashlib.sha256(report.read_bytes()).hexdigest()}
    asset = tmp_path/'face.task'
    asset.write_bytes(b'external native loader replaced by test fixture')
    state = SimpleNamespace(points=points,faces=True,closed=0,error=None,created=0)
    class Native:
        def detect_for_video(self,image,timestamp):
            if state.error is not None:
                raise state.error
            faces = [[SimpleNamespace(x=float(x),y=float(y),z=float(z))
                      for x,y,z in state.points]] if state.faces else []
            return vision.FaceLandmarkerResult(face_landmarks=faces,face_blendshapes=[],
                                               facial_transformation_matrixes=[])
        def close(self):
            state.closed += 1
    def create(options):
        state.created += 1
        return Native()
    monkeypatch.setattr(vision.FaceLandmarker,'create_from_options',create)
    state.config = dict(asset_path=asset,canonical_model_path=obj,epsilon=1e-6,
        max_reprojection_error_norm=.03,landmarker_mode='VIDEO',num_faces=1,
        min_face_detection_confidence=.5,min_face_presence_confidence=.5,
        min_tracking_confidence=.5,landmark_target_fps=20,quality=quality,
        camera_model=dict(mode='calibrated',reference_size=[640,480],matrix=matrix,
                          distortion=[0,0,0,0,0]))
    state.image = np.random.default_rng(7).integers(80,161,(480,640,3),dtype=np.uint8)
    return state
