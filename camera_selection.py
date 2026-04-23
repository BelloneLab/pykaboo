"""Helpers for restoring the last connected camera across app launches."""

from __future__ import annotations

from typing import Mapping


_SAVED_CAMERA_KEYS = (
    "last_camera_type",
    "last_camera_backend",
    "last_camera_index",
    "last_camera_serial",
    "last_camera_video_index",
    "last_camera_serial_port",
    "last_camera_label",
)


def saved_camera_settings_from_info(camera_info: Mapping[str, object] | None) -> dict[str, str]:
    if not camera_info:
        return {key: "" for key in _SAVED_CAMERA_KEYS}
    return {
        "last_camera_type": str(camera_info.get("type", "") or "").strip(),
        "last_camera_backend": str(camera_info.get("backend", "") or "").strip(),
        "last_camera_index": str(camera_info.get("index", "") or "").strip(),
        "last_camera_serial": str(camera_info.get("serial", "") or "").strip(),
        "last_camera_video_index": str(camera_info.get("video_index", "") or "").strip(),
        "last_camera_serial_port": str(camera_info.get("serial_port", "") or "").strip(),
        "last_camera_label": str(camera_info.get("label", "") or "").strip(),
    }


def saved_camera_settings_available(saved_settings: Mapping[str, object] | None) -> bool:
    if not saved_settings:
        return False
    camera_type = str(saved_settings.get("last_camera_type", "") or "").strip()
    if not camera_type:
        return False
    for key in (
        "last_camera_serial",
        "last_camera_video_index",
        "last_camera_serial_port",
        "last_camera_label",
        "last_camera_index",
    ):
        if str(saved_settings.get(key, "") or "").strip():
            return True
    return False


def saved_camera_match_score(
    camera_info: Mapping[str, object] | None,
    saved_settings: Mapping[str, object] | None,
) -> int:
    if not camera_info or not saved_camera_settings_available(saved_settings):
        return -1

    camera_type = str(camera_info.get("type", "") or "").strip()
    saved_type = str(saved_settings.get("last_camera_type", "") or "").strip()
    if camera_type != saved_type:
        return -1

    camera_backend = str(camera_info.get("backend", "") or "").strip()
    saved_backend = str(saved_settings.get("last_camera_backend", "") or "").strip()
    if saved_backend and camera_backend != saved_backend:
        return -1

    camera_serial = str(camera_info.get("serial", "") or "").strip()
    saved_serial = str(saved_settings.get("last_camera_serial", "") or "").strip()
    if camera_serial and saved_serial and camera_serial == saved_serial:
        return 5

    camera_video_index = str(camera_info.get("video_index", "") or "").strip()
    saved_video_index = str(saved_settings.get("last_camera_video_index", "") or "").strip()
    if camera_video_index and saved_video_index and camera_video_index == saved_video_index:
        return 4

    camera_serial_port = str(camera_info.get("serial_port", "") or "").strip()
    saved_serial_port = str(saved_settings.get("last_camera_serial_port", "") or "").strip()
    if camera_serial_port and saved_serial_port and camera_serial_port == saved_serial_port:
        return 3

    camera_label = str(camera_info.get("label", "") or "").strip()
    saved_label = str(saved_settings.get("last_camera_label", "") or "").strip()
    if camera_label and saved_label and camera_label == saved_label:
        return 2

    camera_index = str(camera_info.get("index", "") or "").strip()
    saved_index = str(saved_settings.get("last_camera_index", "") or "").strip()
    if camera_index and saved_index and camera_index == saved_index:
        return 1

    return -1


def camera_matches_saved_selection(
    camera_info: Mapping[str, object] | None,
    saved_settings: Mapping[str, object] | None,
) -> bool:
    return saved_camera_match_score(camera_info, saved_settings) >= 0
