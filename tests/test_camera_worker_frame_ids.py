import unittest
from types import SimpleNamespace

from camera_worker import CameraWorker


class _FakeStdin:
    def write(self, _payload):
        return None


class CameraWorkerFrameIdTests(unittest.TestCase):
    def test_recording_frame_id_is_assigned_before_packet_emission(self):
        worker = CameraWorker()
        worker.is_recording = True
        worker.frame_counter = 7
        worker.ffmpeg_process = SimpleNamespace(stdin=_FakeStdin())
        metadata = {}

        worker._assign_recording_frame_id(metadata)

        self.assertEqual(metadata["frame_id"], 7)


if __name__ == "__main__":
    unittest.main()
