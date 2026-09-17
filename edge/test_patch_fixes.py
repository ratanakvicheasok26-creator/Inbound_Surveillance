import unittest
import time
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from launcher import (
    transform_point_hflip,
    transform_point_vflip,
    transform_point_rotate,
    transform_bay_geometry,
    LiveStreamEngine,
    CameraStreamPool,
    resolve_static_asset,
)


class TestPatchFixes(unittest.TestCase):
    def test_geometry_transforms(self):
        # Horizontal flip: x -> 1 - x
        self.assertEqual(transform_point_hflip(0.1, 0.2), (0.9, 0.2))
        self.assertEqual(transform_point_hflip(0.9, 0.2), (0.1, 0.2))

        # Vertical flip: y -> 1 - y
        self.assertEqual(transform_point_vflip(0.1, 0.2), (0.1, 0.8))
        self.assertEqual(transform_point_vflip(0.1, 0.8), (0.1, 0.2))

        # Rotate 90 CW: (x, y) -> (1 - y, x)
        self.assertEqual(transform_point_rotate(0.1, 0.2, 90), (0.8, 0.1))

        # Rotate 180: (x, y) -> (1 - x, 1 - y)
        self.assertEqual(transform_point_rotate(0.1, 0.2, 180), (0.9, 0.8))

        # Rotate 270 CW: (x, y) -> (y, 1 - x)
        self.assertEqual(transform_point_rotate(0.1, 0.2, 270), (0.2, 0.9))

    def test_bay_geometry_transform(self):
        bay = {
            "id": "bay_1",
            "name": "Lift Bay 1",
            "roi": [0.10, 0.20, 0.30, 0.40],
            "polygon": [[0.10, 0.20], [0.40, 0.20], [0.40, 0.60], [0.10, 0.60]],
        }
        # Horizontal flip
        tb = transform_bay_geometry(bay, transform_point_hflip)
        # Expected new xs: 1 - 0.4 = 0.6, 1 - 0.1 = 0.9. min_x = 0.6, width = 0.3
        self.assertAlmostEqual(tb["roi"][0], 0.60, places=3)
        self.assertAlmostEqual(tb["roi"][1], 0.20, places=3)
        self.assertAlmostEqual(tb["roi"][2], 0.30, places=3)
        self.assertAlmostEqual(tb["roi"][3], 0.40, places=3)

    def test_set_orient_transforms_bays(self):
        cfg = {
            "rotate": 0,
            "flip": "none",
            "active_camera_id": "cam_1",
            "cameras": [
                {
                    "id": "cam_1",
                    "name": "Cam 1",
                    "rotate": 0,
                    "flip": "none",
                    "roi": [0.10, 0.20, 0.30, 0.40],
                    "bays": [{"id": "bay_1", "roi": [0.10, 0.20, 0.30, 0.40]}],
                }
            ],
            "bays": [{"id": "bay_1", "roi": [0.10, 0.20, 0.30, 0.40]}],
            "roi": [0.10, 0.20, 0.30, 0.40],
        }
        with patch("launcher.read_config", return_value=cfg):
            engine = LiveStreamEngine()
            engine.cfg = cfg
            engine.bay_manager.set_bays(cfg["bays"])
        with patch("launcher.save_config"):
            # Change orientation: horizontal flip
            res = engine.set_orient(rotate=0, flip="h")
            self.assertTrue(res["success"])
            self.assertEqual(res["flip"], "h")
            bays = res["bays"]
            self.assertEqual(len(bays), 1)
            self.assertAlmostEqual(bays[0]["roi"][0], 0.60, places=3)

            # Change orientation: rotate 90
            res2 = engine.set_orient(rotate=90, flip="h")
            self.assertTrue(res2["success"])
            self.assertEqual(res2["rotate"], 90)

    def test_non_blocking_camera_pool_sync(self):
        pool = CameraStreamPool()
        slow_worker = MagicMock()
        def slow_stop():
            time.sleep(0.5)
        slow_worker.stop.side_effect = slow_stop
        slow_worker.camera_id = "slow_cam"

        with pool._lock:
            pool._workers["slow_cam"] = slow_worker

        start_time = time.time()
        # Syncing with empty list removes slow_cam
        pool.sync_cameras([])
        elapsed = time.time() - start_time
        # pool.sync_cameras must return almost instantaneously because teardown is async
        self.assertLess(elapsed, 0.2, f"sync_cameras took too long: {elapsed:.3f}s")
        self.assertNotIn("slow_cam", pool._workers)


class TestFrozenPathSplit(unittest.TestCase):
    def test_videos_live_in_writable_data_dir(self):
        from launcher import DATA_DIR, ROOT, VIDEOS_DIR, init_global_engine
        from paths import INBOUND_APP_VERSION, data_dir, resource_dir

        self.assertEqual(VIDEOS_DIR, DATA_DIR / "videos")
        self.assertEqual(DATA_DIR, data_dir())
        self.assertEqual(ROOT, resource_dir())
        self.assertEqual(INBOUND_APP_VERSION, "0.1.3")
        engine = init_global_engine()
        self.assertEqual(engine.ai_auditor.save_crops_dir, DATA_DIR / "proofs" / "ai_audits")

    def test_ffmpeg_candidates_are_os_specific(self):
        from media.go2rtc import ffmpeg_candidate_paths

        paths = ffmpeg_candidate_paths()
        blob = "\n".join(paths)
        if sys.platform == "win32":
            self.assertIn("ffmpeg.exe", blob)
            self.assertNotIn("/usr/bin/ffmpeg", blob)
        else:
            self.assertTrue(any(p.endswith("/ffmpeg") or p.endswith("ffmpeg") for p in paths))
            self.assertNotIn("ProgramFiles", blob)

    def test_yaml_quoted_path_strips_windows_escapes(self):
        from media.go2rtc import yaml_quoted_path

        quoted = yaml_quoted_path(r"C:\ffmpeg\bin\ffmpeg.exe")
        self.assertEqual(quoted, "C:/ffmpeg/bin/ffmpeg.exe")
        section = f'ffmpeg:\n  bin: "{quoted}"\n'
        self.assertNotIn("\\b", section)
        self.assertIn("C:/ffmpeg/bin/ffmpeg.exe", section)

    def test_static_pipeline_graph_resolves_and_blocks_traversal(self):
        asset = resolve_static_asset("/static/pipeline-graph.js")
        self.assertIsNotNone(asset)
        self.assertEqual(asset.name, "pipeline-graph.js")
        self.assertTrue(asset.is_file())
        self.assertIsNone(resolve_static_asset("/static/../hub.html"))
        escaped = resolve_static_asset("/static/..\\..\\launcher.py")
        if escaped is not None:
            self.assertEqual(escaped.parent.name, "static")


if __name__ == "__main__":
    unittest.main()
