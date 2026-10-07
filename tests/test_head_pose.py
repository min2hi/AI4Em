import math
import cv2
import numpy as np
import pytest
from src.contracts import LandmarkResult
from src.features.head_pose import HeadPoseEstimator

POSE_IDS = (1,152,33,263,61,291)
CANONICAL_SIX = np.array([
    (0,-1.126865,7.475604), (0,-9.403378,4.264492),
    (-4.445859,2.663991,3.173422), (4.445859,2.663991,3.173422),
    (-2.456206,-4.342621,4.283884), (2.456206,-4.342621,4.283884),
], dtype=np.float64)
OBJECT_SIX = CANONICAL_SIX * (1,-1,-1)

@pytest.fixture
def temporary_canonical(tmp_path):
    vertices = np.zeros((468,3),dtype=float)
    vertices[list(POSE_IDS)] = CANONICAL_SIX
    lines = ['# fixture: only v records count']
    for x,y,z in vertices:
        lines.extend((f'v {x} {y} {z}','vt 0 0','vn 0 0 1'))
    path = tmp_path/'canonical.obj'
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return path

def signed_rotation(pitch, yaw_display, roll):
    p,y,r = np.deg2rad((pitch,-yaw_display,roll))
    cp,sp,cy,sy,cr,sr = np.cos(p),np.sin(p),np.cos(y),np.sin(y),np.cos(r),np.sin(r)
    rx = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    rz = np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]])
    return rz @ ry @ rx

def projected_result(angles, size=(640,480), K=None):
    w,h = size
    if K is None:
        K = np.array([[800.,0,w/2],[0,800.,h/2],[0,0,1]])
    rv,_ = cv2.Rodrigues(signed_rotation(*angles))
    xy,_ = cv2.projectPoints(OBJECT_SIX,rv,np.array([0.,0.,60.]),K,np.zeros(5))
    points = np.zeros((478,3),np.float32)
    points[list(POSE_IDS),:2] = xy.reshape(6,2) / (w,h)
    return LandmarkResult(points,0,True,size)

@pytest.mark.parametrize('angles', [
    (0,0,0),(10,0,0),(-10,0,0),(0,10,0),(0,-10,0),
    (0,0,10),(0,0,-10),(10,-10,10),
])
def test_signed_pose_and_neutral_not_180(temporary_canonical, angles):
    camera = {'mode':'calibrated','reference_size':[640,480],
              'matrix':[[800,0,320],[0,800,240],[0,0,1]],'distortion':[0,0,0,0,0]}
    estimator = HeadPoseEstimator(temporary_canonical,camera,
                    max_reprojection_error_norm=.03,epsilon=1e-6)
    result = estimator.estimate(projected_result(angles))
    assert result.pose_valid
    assert (result.pitch,result.yaw,result.roll) == pytest.approx(angles,abs=1)
    assert result.reprojection_error_norm < 1e-5


@pytest.fixture
def calibrated_camera():
    return {"mode": "calibrated", "reference_size": [640, 480],
            "matrix": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
            "distortion": [0, 0, 0, 0, 0]}


@pytest.fixture
def estimator(temporary_canonical, calibrated_camera):
    return HeadPoseEstimator(temporary_canonical, calibrated_camera,
                             max_reprojection_error_norm=.03, epsilon=1e-6)


def assert_invalid(pose, *, measured=False):
    assert not pose.pose_valid
    assert all(math.isnan(value) for value in (pose.pitch, pose.yaw, pose.roll))
    assert math.isfinite(pose.reprojection_error_norm) if measured else math.isnan(pose.reprojection_error_norm)


@pytest.mark.parametrize("size,K", [
    ((1280, 960), [[1600, 0, 640], [0, 1600, 480], [0, 0, 1]]),
    ((1280, 480), [[1600, 0, 640], [0, 800, 240], [0, 0, 1]]),
])
def test_scaled_intrinsics_preserve_pose(estimator, size, K):
    pose = estimator.estimate(projected_result((12, -8, 15), size, np.array(K, float)))
    assert pose.pose_valid
    assert (pose.pitch, pose.yaw, pose.roll) == pytest.approx((12, -8, 15), abs=1)
    metadata = estimator.camera_metadata(size)
    assert np.array(metadata["matrix"]) == pytest.approx(np.array(K))
    assert metadata["approximate"] is False


def test_approximate_camera_is_disclosed_and_uses_actual_size(temporary_canonical):
    estimator = HeadPoseEstimator(temporary_canonical, {"mode": "approximate"},
                                 max_reprojection_error_norm=.03, epsilon=1e-6)
    metadata = estimator.camera_metadata((1280, 480))
    assert metadata["approximate"] is True
    assert metadata["image_size"] == [1280, 480]
    assert metadata["matrix"] == [[1280., 0., 640.], [0., 1280., 240.], [0., 0., 1.]]
    pose = estimator.estimate(projected_result((10, 10, 10), (1280, 480),
                                             np.array(metadata["matrix"])))
    assert pose.pose_valid
    assert (pose.pitch, pose.yaw, pose.roll) == pytest.approx((10, 10, 10), abs=1)


def test_returned_angles_have_physical_forward_signs(estimator):
    pose = estimator.estimate(projected_result((15, 20, 0)))
    assert pose.pose_valid
    forward = signed_rotation(pose.pitch, pose.yaw, pose.roll) @ np.array([0., 0., -1.])
    assert forward[0] > 0  # displayed yaw+ points image-right
    assert forward[1] > 0  # pitch+ points image-down


@pytest.mark.parametrize("points,detected", [(None, True), (None, False), (np.zeros((478, 3)), False)])
def test_missing_face_has_no_invented_neutral_pose(estimator, points, detected):
    assert_invalid(estimator.estimate(LandmarkResult(points, 0, detected, (640, 480))))


def test_only_selected_xy_affects_pose_and_input_is_not_mutated(estimator):
    result = projected_result((10, 0, 0))
    result.points[:, 2] = np.nan
    result.points[0, :2] = np.inf
    before = result.points.copy()
    pose = estimator.estimate(result)
    assert pose.pose_valid
    assert pose.pitch == pytest.approx(10, abs=1)
    np.testing.assert_array_equal(result.points, before)


@pytest.mark.parametrize("value", [np.nan, np.inf, -.01, 1.01])
def test_invalid_selected_xy_is_rejected(estimator, value):
    result = projected_result((0, 0, 0))
    result.points[1, 0] = value
    assert_invalid(estimator.estimate(result))


@pytest.mark.parametrize("points,size", [
    (np.zeros((468, 3), float), (640, 480)),
    (np.zeros((478, 3), int), (640, 480)),
    ([[0., 0., 0.]] * 478, (640, 480)),
    (np.zeros((478, 3), float), (0, 480)),
    (np.zeros((478, 3), float), (True, 480)),
    (np.zeros((478, 3), float), [640, 480]),
])
def test_present_schema_errors_raise(estimator, points, size):
    with pytest.raises(ValueError):
        estimator.estimate(LandmarkResult(points, 0, True, size))


@pytest.mark.parametrize("collinear", [False, True])
def test_collapsed_and_collinear_image_geometry_invalid(estimator, collinear):
    result = projected_result((0, 0, 0))
    result.points[list(POSE_IDS), :2] = .5
    if collinear:
        result.points[list(POSE_IDS), 0] = np.linspace(.3, .7, 6)
    assert_invalid(estimator.estimate(result))


@pytest.mark.parametrize("angles", [(0, 0, 179), (0, 0, -179), (179, 0, 0), (-179, 0, 0)])
def test_wrapped_angles_reconstruct_rotation(estimator, angles):
    pose = estimator.estimate(projected_result(angles))
    assert pose.pose_valid
    assert all(-180 <= value < 180 for value in (pose.pitch, pose.yaw, pose.roll))
    assert signed_rotation(pose.pitch, pose.yaw, pose.roll) == pytest.approx(
        signed_rotation(*angles), abs=.02)


@pytest.mark.parametrize("outcome", ["failure", "exception", "negative_depth", "nonfinite", "singular"])
def test_solver_boundary_failures_never_create_pose(estimator, monkeypatch, outcome):
    def fit(*args, **kwargs):
        if outcome == "exception":
            raise cv2.error("forced fit failure")
        if outcome == "failure":
            return False, None, None
        rotation = signed_rotation(0, 90, 0) if outcome == "singular" else np.eye(3)
        vector, _ = cv2.Rodrigues(rotation)
        translation = np.array([0., 0., -60. if outcome == "negative_depth" else 60.])
        if outcome == "nonfinite":
            translation[0] = np.nan
        return True, vector, translation
    monkeypatch.setattr(cv2, "solvePnP", fit)
    assert_invalid(estimator.estimate(projected_result((0, 0, 0))))


def test_noisy_fit_retains_diagnostic_rms(temporary_canonical, calibrated_camera):
    result = projected_result((0, 0, 0))
    result.points[1, 0] += .06
    estimator = HeadPoseEstimator(temporary_canonical, calibrated_camera,
                                 max_reprojection_error_norm=1e-5, epsilon=1e-6)
    pose = estimator.estimate(result)
    assert_invalid(pose, measured=True)
    assert pose.reprojection_error_norm > 1e-5


def test_reprojection_threshold_equality_is_accepted(temporary_canonical, calibrated_camera):
    result = projected_result((0, 0, 0))
    result.points[1, 0] += .04
    loose = HeadPoseEstimator(temporary_canonical, calibrated_camera,
                             max_reprojection_error_norm=1., epsilon=1e-6)
    error = loose.estimate(result).reprojection_error_norm
    assert math.isfinite(error) and error > 0
    equal = HeadPoseEstimator(temporary_canonical, calibrated_camera,
                             max_reprojection_error_norm=error, epsilon=1e-6)
    strict = HeadPoseEstimator(temporary_canonical, calibrated_camera,
                              max_reprojection_error_norm=np.nextafter(error, 0.), epsilon=1e-6)
    assert equal.estimate(result).pose_valid
    assert_invalid(strict.estimate(result), measured=True)


@pytest.mark.parametrize("text", [
    "v 0 0 0\n", "v invalid 0 0\n" * 468, "v 0 0 0\n" * 468,
    "v nan 0 0\n" * 468, "v 0 0\n" * 468,
])
def test_invalid_obj_rejected_at_construction(tmp_path, text):
    path = tmp_path / "invalid.obj"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        HeadPoseEstimator(path, {"mode": "approximate"},
                          max_reprojection_error_norm=.03, epsilon=1e-6)


@pytest.mark.parametrize("epsilon,threshold", [
    (0, .03), (True, .03), (np.nan, .03), (1e-6, 0),
    (1e-6, 1.01), (1e-6, True), (1e-6, np.inf),
])
def test_invalid_estimator_parameters(temporary_canonical, epsilon, threshold):
    with pytest.raises(ValueError):
        HeadPoseEstimator(temporary_canonical, {"mode": "approximate"},
                          max_reprojection_error_norm=threshold, epsilon=epsilon)


@pytest.mark.parametrize("pixel_error,valid", [(24., True), (24.0001, False)])
def test_literal_three_percent_diagonal_rms_boundary(estimator, monkeypatch, pixel_error, valid):
    result = projected_result((0, 0, 0))
    observed = result.points[list(POSE_IDS), :2].astype(np.float64) * (640, 480)
    # Isolate projection residuals at the native boundary, leaving the real fit,
    # rotation and cheirality guards active. A 24px RMS / 800px diagonal is .03.
    def projection(*args, **kwargs):
        return (observed + [pixel_error, 0]).reshape(6, 1, 2), None
    monkeypatch.setattr(cv2, "projectPoints", projection)
    pose = estimator.estimate(result)
    assert pose.pose_valid is valid
    assert pose.reprojection_error_norm == pytest.approx(pixel_error / 800., abs=1e-12)
    if not valid:
        assert_invalid(pose, measured=True)


def test_nonzero_calibrated_distortion_and_copied_camera_inputs(temporary_canonical, calibrated_camera):
    calibrated_camera["distortion"] = [.15, -.05, .002, -.003, .01]
    estimator = HeadPoseEstimator(temporary_canonical, calibrated_camera,
                                 max_reprojection_error_norm=.03, epsilon=1e-6)
    matrix = np.array(calibrated_camera["matrix"], float)
    vector, _ = cv2.Rodrigues(signed_rotation(13, -17, 8))
    xy, _ = cv2.projectPoints(OBJECT_SIX, vector, np.array([0., 0., 60.]),
                              matrix, np.array(calibrated_camera["distortion"]))
    result = projected_result((0, 0, 0))
    result.points[list(POSE_IDS), :2] = xy.reshape(6, 2) / (640, 480)
    calibrated_camera["matrix"][0][0] = 1
    calibrated_camera["distortion"][0] = 99
    metadata = estimator.camera_metadata((640, 480))
    metadata["matrix"][0][0] = 2
    metadata["distortion"][0] = 98
    pose = estimator.estimate(result)
    assert pose.pose_valid
    assert (pose.pitch, pose.yaw, pose.roll) == pytest.approx((13, -17, 8), abs=1)
    assert pose.reprojection_error_norm < 1e-5


def test_rank_two_selected_object_geometry_is_rejected(tmp_path):
    vertices = np.zeros((468, 3))
    vertices[list(POSE_IDS)] = CANONICAL_SIX
    vertices[list(POSE_IDS), 2] = 0
    path = tmp_path / "planar.obj"
    path.write_text("\n".join(f"v {x} {y} {z}" for x, y, z in vertices), encoding="utf-8")
    with pytest.raises(ValueError):
        HeadPoseEstimator(path, {"mode": "approximate"},
                          max_reprojection_error_norm=.03, epsilon=1e-6)
