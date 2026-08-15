"""ALOHA and robomimic HDF5, behind the interface the LeRobot reader presents.

"LeRobot format" does not mean one thing — GR00T still ships a v3-to-v2
downconverter and Rerun could not load v3.0 until a patch this year — and a
large amount of real robot data lives in ALOHA and robomimic files instead.

The rule this file exists to enforce: an unfamiliar layout is REFUSED, never
guessed. Picking the wrong dataset as "the action" produces a panel full of
numbers that are all real and all about the wrong thing, which is worse than
a file that will not open.
"""

from __future__ import annotations

import pytest

h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

from modelmri.errors import BadRequest  # noqa: E402
from modelmri.hdf5_data import ALOHA, ROBOMIMIC, Hdf5Reader, detect  # noqa: E402


def _aloha(path, *, frames=30, cameras=("top", "wrist"), fps=50, action="action"):
    with h5py.File(path, "w") as f:
        if fps:
            f.attrs["fps"] = fps
        f.attrs["task"] = "transfer the cube"
        group = f.create_group("observations/images")
        for cam in cameras:
            group.create_dataset(
                cam,
                data=np.random.randint(0, 255, (frames, 48, 64, 3), dtype=np.uint8),
            )
        f.create_dataset("observations/qpos", data=np.random.randn(frames, 14))
        if action:
            f.create_dataset(action, data=np.random.randn(frames, 14))
    return path


def _robomimic(path, *, demos=(0, 1, 2, 10), frames=12, actions="actions"):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i in demos:
            ep = data.create_group(f"demo_{i}")
            ep.create_dataset(
                "obs/agentview_image",
                data=np.random.rand(frames, 32, 32, 3).astype(np.float32),
            )
            ep.create_dataset("obs/qpos", data=np.random.randn(frames, 7))
            if actions:
                ep.create_dataset(actions, data=np.random.randn(frames, 7))
    return path


# --------------------------------------------------------------- detection


def test_an_aloha_file_is_detected(tmp_path):
    reader = Hdf5Reader(_aloha(tmp_path / "ep.hdf5"))
    assert reader.layout.kind == ALOHA
    assert reader.layout.action == "action"
    reader.close()


def test_a_robomimic_file_is_detected(tmp_path):
    reader = Hdf5Reader(_robomimic(tmp_path / "d.hdf5"))
    assert reader.layout.kind == ROBOMIMIC
    assert len(reader.episodes()) == 4
    reader.close()


def test_an_unfamiliar_layout_is_refused_with_the_keys_it_found(tmp_path):
    """The refusal carries the top-level keys of the reader's OWN file, which
    is the one thing that lets them work out what to do next."""
    path = tmp_path / "mystery.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("frames", data=np.zeros((2, 2)))
        f.create_dataset("labels", data=np.zeros((2,)))
    with pytest.raises(BadRequest) as caught:
        Hdf5Reader(path)
    message = str(caught.value)
    assert "does not recognise this HDF5 layout" in message
    assert "'frames'" in message and "'labels'" in message
    assert "Refusing rather than guessing" in message


def test_an_aloha_file_with_no_action_is_refused_rather_than_guessed(tmp_path):
    """Guessing which dataset is the action would fill the panel with real
    numbers about the wrong thing."""
    with pytest.raises(BadRequest, match="no action dataset"):
        Hdf5Reader(_aloha(tmp_path / "ep.hdf5", action=""))


def test_a_robomimic_file_with_no_actions_is_refused(tmp_path):
    with pytest.raises(BadRequest, match="no action group"):
        Hdf5Reader(_robomimic(tmp_path / "d.hdf5", actions=""))


def test_an_empty_data_group_is_refused(tmp_path):
    path = tmp_path / "empty.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("data")
    with pytest.raises(BadRequest, match="no episodes in it"):
        Hdf5Reader(path)


def test_a_file_with_no_cameras_is_refused(tmp_path):
    with pytest.raises(BadRequest, match="nothing for the vision tower"):
        Hdf5Reader(_aloha(tmp_path / "ep.hdf5", cameras=()))


def test_a_file_that_is_not_hdf5_is_refused_without_leaking_its_path(tmp_path):
    path = tmp_path / "not_really.hdf5"
    path.write_text("this is not HDF5", encoding="utf-8")
    with pytest.raises(BadRequest) as caught:
        Hdf5Reader(path)
    assert "An .hdf5 extension is not a format" in str(caught.value)
    assert str(tmp_path) not in str(caught.value)


def test_a_missing_file_names_no_path():
    """`errors.py` names this exactly: a library talking about this machine to
    somebody who was not asking about this machine."""
    with pytest.raises(BadRequest) as caught:
        Hdf5Reader("nowhere/at/all.hdf5")
    assert "does not exist" in str(caught.value)
    assert "nowhere" not in str(caught.value)


# ------------------------------------------------------------- the ordering


def test_demo_10_sorts_after_demo_2(tmp_path):
    """Sorting demo names as strings puts demo_10 before demo_2, so the
    episode indices in the panel would not match the order in the file — and a
    reader comparing episode 2 here against episode 2 in their training log
    would be looking at different data."""
    reader = Hdf5Reader(_robomimic(tmp_path / "d.hdf5", demos=(0, 1, 2, 10)))
    names = [p.split("/")[-1] for p in reader.layout.episodes]
    assert names == ["demo_0", "demo_1", "demo_2", "demo_10"]
    reader.close()


# ------------------------------------------------------------- the cameras


def test_a_state_dataset_is_not_mistaken_for_a_camera(tmp_path):
    """`qpos` sits in robomimic's obs group beside the cameras, so this checks
    the SHAPE rather than the name."""
    reader = Hdf5Reader(_robomimic(tmp_path / "d.hdf5"))
    assert reader.cameras == ["agentview_image"]
    assert "qpos" not in reader.cameras
    reader.close()


def test_switching_cameras_works_and_an_unknown_one_is_refused(tmp_path):
    reader = Hdf5Reader(_aloha(tmp_path / "ep.hdf5"))
    assert reader.camera == "top"
    reader.use_camera("wrist")
    assert reader.camera == "wrist"
    with pytest.raises(BadRequest, match="is not a camera in this file"):
        reader.use_camera("nose")
    reader.close()


# ---------------------------------------------------------------- the frames


def test_a_frame_carries_its_state_and_action(tmp_path):
    reader = Hdf5Reader(_aloha(tmp_path / "ep.hdf5"))
    sample = reader.frame(0, 5)
    assert sample.width == 64 and sample.height == 48
    assert len(sample.state) == 14 and len(sample.action) == 14
    assert sample.image.startswith("data:image/png;base64,")
    assert sample.task == "transfer the cube"
    reader.close()


def test_float_frames_are_scaled_rather_than_cast(tmp_path):
    """A plain cast turns 0.87 into 0 and the panel shows black."""
    reader = Hdf5Reader(_robomimic(tmp_path / "d.hdf5"))
    rgb = reader.raw_frame(0, 0)
    assert rgb.dtype == np.uint8
    assert int(rgb.max()) > 200, "float frames in [0,1] were cast, not scaled"
    reader.close()


def test_a_greyscale_frame_becomes_three_channels(tmp_path):
    path = tmp_path / "grey.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("observations/images")
        g.create_dataset(
            "top", data=np.random.randint(0, 255, (5, 16, 16), dtype=np.uint8)
        )
        f.create_dataset("action", data=np.random.randn(5, 4))
    reader = Hdf5Reader(path)
    assert reader.raw_frame(0, 0).shape == (16, 16, 3)
    reader.close()


def test_an_rgba_frame_drops_its_alpha(tmp_path):
    path = tmp_path / "rgba.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("observations/images")
        g.create_dataset(
            "top", data=np.random.randint(0, 255, (5, 16, 16, 4), dtype=np.uint8)
        )
        f.create_dataset("action", data=np.random.randn(5, 4))
    reader = Hdf5Reader(path)
    assert reader.raw_frame(0, 0).shape == (16, 16, 3)
    reader.close()


def test_a_timestep_outside_the_episode_is_refused(tmp_path):
    reader = Hdf5Reader(_aloha(tmp_path / "ep.hdf5", frames=10))
    with pytest.raises(BadRequest, match=r"t must be in \[0,10\)"):
        reader.raw_frame(0, 10)
    reader.close()


def test_an_episode_outside_the_file_is_refused(tmp_path):
    reader = Hdf5Reader(_aloha(tmp_path / "ep.hdf5"))
    with pytest.raises(BadRequest, match=r"episode 7 not in \[0,1\)"):
        reader.raw_frame(7, 0)
    reader.close()


# ---------------------------------------------------- the unknown frequency


def test_a_file_that_states_no_frequency_reports_zero_not_a_guess(tmp_path):
    """0.0 means UNKNOWN. `vla_audit.check_action_lag` refuses outright on it
    rather than assuming 30, because a lag in frames means nothing without a
    real frequency."""
    reader = Hdf5Reader(_aloha(tmp_path / "ep.hdf5", fps=0))
    assert reader.fps == 0.0
    assert reader.info == {}
    reader.close()


def test_a_stated_frequency_is_read(tmp_path):
    reader = Hdf5Reader(_aloha(tmp_path / "ep.hdf5", fps=50))
    assert reader.fps == 50.0
    assert reader.info["fps"] == 50.0
    reader.close()


def test_the_audit_refuses_to_measure_lag_without_a_frequency(tmp_path):
    """The two modules agree by construction: this reader reports 0.0 and the
    audit reads that as absent."""
    from modelmri import vla_audit

    reader = Hdf5Reader(_aloha(tmp_path / "ep.hdf5", fps=0))
    try:
        check = vla_audit.check_action_lag(reader)
        assert check.verdict == vla_audit.UNCHECKED
        assert "does not state a control frequency" in check.detail
    finally:
        reader.close()


# ------------------------------------------- it matches the reader interface


def test_it_presents_the_same_surface_the_lerobot_reader_does(tmp_path):
    """`VLAPanel.tsx` and every `/api/vla/*` route is untouched by this
    reader, which is only true while the surfaces match."""
    from modelmri.vla_data import LeRobotV3Reader

    wanted = [
        name
        for name in ("episodes", "frame", "raw_frame", "use_camera", "summary", "close")
        if callable(getattr(LeRobotV3Reader, name, None))
    ]
    reader = Hdf5Reader(_aloha(tmp_path / "ep.hdf5"))
    try:
        for name in wanted:
            assert callable(getattr(reader, name, None)), f"missing {name}()"
        for name in ("cameras", "camera", "repo_id"):
            assert hasattr(reader, name), f"missing {name}"
    finally:
        reader.close()


def test_the_summary_names_which_convention_was_detected(tmp_path):
    """So a reader can see why the episode count is what it is."""
    reader = Hdf5Reader(_robomimic(tmp_path / "d.hdf5"))
    summary = reader.summary()
    assert summary["layout"]["kind"] == ROBOMIMIC
    assert summary["n_episodes"] == 4
    assert summary["image_shape"] == [32, 32, 3]
    reader.close()


def test_episodes_tile_the_file_the_way_the_audit_expects(tmp_path):
    """`data_from` accumulates, so `vla_audit.check_tiling` sees a clean tile
    rather than four episodes all starting at row zero."""
    from modelmri import vla_audit

    reader = Hdf5Reader(_robomimic(tmp_path / "d.hdf5", frames=12))
    try:
        starts = [e.data_from for e in reader.episodes()]
        assert starts == [0, 12, 24, 36]
        reader._frame_table = lambda: {"episode_index": [0] * 48}
        assert vla_audit.check_tiling(reader).verdict == vla_audit.OK
    finally:
        reader.close()


def test_closing_twice_is_not_an_error(tmp_path):
    reader = Hdf5Reader(_aloha(tmp_path / "ep.hdf5"))
    reader.close()
    reader.close()
