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

    def test_virtual_camera_generates_marked_frame_packet(self):
        worker = CameraWorker()
        ok = worker.connect_camera(
            {
                "type": "virtual",
                "label": "SIMULATED: test",
                "serial": "sim-test",
                "width": 320,
                "height": 240,
                "fps": 30.0,
            }
        )

        self.assertTrue(ok)
        packet = worker._capture_virtual_frame_packet()

        self.assertIsNotNone(packet)
        self.assertEqual(packet.backend, "virtual")
        self.assertEqual(packet.frame.shape, (240, 320, 3))
        self.assertTrue(packet.metadata["simulated"])
        self.assertEqual(packet.metadata["camera_frame_id"], 0)
        worker.disconnect_camera()


if __name__ == "__main__":
    unittest.main()
