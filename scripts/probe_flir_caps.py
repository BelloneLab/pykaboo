"""Probe the connected FLIR Spinnaker camera for resolution/framerate ceilings.

Reports the max achievable AcquisitionFrameRate at full sensor, at 1920x1080,
and at a couple of intermediate ROIs, plus the DeviceLinkThroughputLimit.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch  # noqa: F401  (DLL order)
except Exception:
    pass

import PySpin


def _node_val(nodemap, name):
    try:
        node = nodemap.GetNode(name)
        if node is None:
            return None
        import PySpin as _ps
        if _ps.IsAvailable(node) and _ps.IsReadable(node):
            ntype = node.GetPrincipalInterfaceType()
            if ntype == _ps.intfIFloat:
                f = _ps.CFloatPtr(node)
                return float(f.GetValue())
            if ntype == _ps.intfIInteger:
                i = _ps.CIntegerPtr(node)
                return int(i.GetValue())
            if ntype == _ps.intfIBoolean:
                b = _ps.CBooleanPtr(node)
                return bool(b.GetValue())
            if ntype == _ps.intfIEnumeration:
                e = _ps.CEnumerationPtr(node)
                return e.GetCurrentEntry().GetSymbolic()
    except Exception as exc:
        return f"<err {exc}>"
    return None


def _set_int(nodemap, name, value):
    try:
        node = PySpin.CIntegerPtr(nodemap.GetNode(name))
        if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
            inc = node.GetInc() or 1
            value = (value // inc) * inc
            value = max(node.GetMin(), min(node.GetMax(), value))
            node.SetValue(value)
            return int(node.GetValue())
    except Exception as exc:
        return f"<err {exc}>"
    return None


def _max_framerate(nodemap):
    # Enable manual frame rate so the max reflects sensor/bandwidth ceiling.
    try:
        en = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
        if PySpin.IsAvailable(en) and PySpin.IsWritable(en):
            en.SetValue(True)
    except Exception:
        pass
    fr = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
    if PySpin.IsAvailable(fr) and PySpin.IsReadable(fr):
        return float(fr.GetMax())
    return None


def main() -> int:
    system = PySpin.System.GetInstance()
    cams = system.GetCameras()
    if cams.GetSize() == 0:
        print("No FLIR cameras found")
        cams.Clear()
        system.ReleaseInstance()
        return 1
    cam = cams.GetByIndex(0)
    cam.Init()
    nodemap = cam.GetNodeMap()

    print("Model:", _node_val(nodemap, "DeviceModelName"))
    print("SensorWidth :", _node_val(nodemap, "SensorWidth"))
    print("SensorHeight:", _node_val(nodemap, "SensorHeight"))
    print("Current WxH :", _node_val(nodemap, "Width"), "x", _node_val(nodemap, "Height"))
    print("PixelFormat :", _node_val(nodemap, "PixelFormat"))
    print("DeviceLinkThroughputLimit:", _node_val(nodemap, "DeviceLinkThroughputLimit"))
    print("DeviceLinkSpeed (bps?):", _node_val(nodemap, "DeviceLinkSpeed"))
    print("DeviceCurrentThroughput :", _node_val(nodemap, "DeviceLinkCurrentThroughput"))

    # Try to set BinningHorizontal/Vertical off and offsets to 0 first.
    for off in ("OffsetX", "OffsetY"):
        _set_int(nodemap, off, 0)

    rois = [
        ("full", _node_val(nodemap, "SensorWidth"), _node_val(nodemap, "SensorHeight")),
        ("3072x1600", 3072, 1600),
        ("1920x1080", 1920, 1080),
        ("1280x720", 1280, 720),
    ]
    print("\n=== Max AcquisitionFrameRate by ROI ===")
    for label, w, h in rois:
        try:
            if not isinstance(w, int) or not isinstance(h, int):
                continue
            # Set offsets to 0 then width/height.
            _set_int(nodemap, "OffsetX", 0)
            _set_int(nodemap, "OffsetY", 0)
            aw = _set_int(nodemap, "Width", w)
            ah = _set_int(nodemap, "Height", h)
            fr_max = _max_framerate(nodemap)
            thr = _node_val(nodemap, "DeviceLinkCurrentThroughput")
            print(f"{label:12s} set->{aw}x{ah}  max_fps={fr_max:.2f}  throughput={thr}")
        except Exception as exc:
            print(f"{label:12s} ERROR {exc}")

    cam.DeInit()
    del cam
    cams.Clear()
    system.ReleaseInstance()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
