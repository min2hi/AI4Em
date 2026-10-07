import math
import pytest
from src.contracts import FramePacket
from src.features.pipeline import FeaturePipeline


def packet(backend, index=0, source='a', timestamp=None):
    return FramePacket(backend.image,index*50 if timestamp is None else timestamp,index,source)


def test_missing_face_does_not_carry_previous_channels(feature_backend):
    b = feature_backend
    with FeaturePipeline(b.config) as p:
        first = p.process(packet(b))
        assert first.left_eye_valid and first.right_eye_valid and first.mouth_valid and first.pose_valid
        b.faces = False
        missing = p.process(packet(b,1))
        assert missing.timestamp_ms==50 and missing.frame_index==1 and missing.source_id=='a'
        assert not missing.face_detected and not missing.pose_valid
        assert all(math.isnan(getattr(missing,k)) for k in ('ear_left','ear_right','ear_mean','mar','pitch','yaw','roll'))
        b.faces = True
        recovered = p.process(packet(b,2))
        assert recovered.pose_valid and recovered.ear_mean==pytest.approx(first.ear_mean)
    assert b.closed==1


def test_one_invalid_eye_does_not_destroy_other_channels(feature_backend):
    b=feature_backend
    b.points[362,:2]=b.points[263,:2]
    with FeaturePipeline(b.config) as p:
        value=p.process(packet(b))
        assert not value.left_eye_valid and math.isnan(value.ear_left) and math.isnan(value.ear_mean)
        assert value.right_eye_valid and value.mouth_valid and value.pose_valid


def test_reset_changes_source_and_close_is_terminal(feature_backend):
    b=feature_backend
    p=FeaturePipeline(b.config)
    p.process(packet(b))
    with pytest.raises(ValueError):
        p.process(packet(b,1,'b'))
    assert p.last_landmarks is None and p.last_quality is None
    p.reset()
    assert b.closed==1
    assert p.process(packet(b,0,'b')).source_id=='b'
    p.close();p.close()
    assert b.closed==2
    with pytest.raises(RuntimeError):p.process(packet(b,1,'b'))
    with pytest.raises(RuntimeError):p.reset()


@pytest.mark.parametrize('index,timestamp',[(0,50),(1,0),(-1,50),(True,50)])
def test_order_errors_clear_stale_diagnostics(feature_backend,index,timestamp):
    b=feature_backend
    with FeaturePipeline(b.config) as p:
        p.process(packet(b))
        with pytest.raises(ValueError):p.process(packet(b,index,timestamp=timestamp))
        assert p.last_landmarks is None and p.last_quality is None


def test_manual_eye_occlusion_keeps_mouth_pose(feature_backend):
    import hashlib,json
    b=feature_backend
    quality=b.config['quality'];quality['eyes_occluded']=True
    path=quality['frozen_report_path']
    path.write_text(json.dumps({'quality':{k:v for k,v in quality.items() if not k.startswith('frozen_report_')}}),encoding='utf-8')
    quality['frozen_report_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with FeaturePipeline(b.config) as p:
        value=p.process(packet(b))
        assert not value.left_eye_valid and not value.right_eye_valid
        assert value.mouth_valid and value.pose_valid


def test_replay_same_packets_matches_independent_session(feature_backend):
    b=feature_backend
    packets=[packet(b,i) for i in range(3)]
    with FeaturePipeline(b.config) as first:
        expected=[first.process(x) for x in packets]
    with FeaturePipeline(b.config) as second:
        actual=[second.process(x) for x in packets]
    assert actual==expected


def test_bad_profile_fails_before_native_factory(feature_backend):
    b=feature_backend
    b.config['quality']['frozen_report_sha256']='0'*64
    with pytest.raises(ValueError):FeaturePipeline(b.config)
    assert b.created==0
