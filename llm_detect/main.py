#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""llm_detect_rbnx — LLM-based object detection service.

Drop-in replacement for yolo_world_rbnx. Uses an OpenAI-compatible VLM
API (e.g. Gemini, Qwen-VL, GPT-4o) to detect objects in camera frames.
Owns ``robonix/service/perception/object_detect/*``.

Two parallel surfaces, sharing one detection function:

    1. Atlas-routed MCP   (the new path, what Pilot's LLM sees)
       robonix/service/perception/object_detect/detect_object

    2. Legacy ROS service (compat path, what pick.py + yolo_grasp.py
       still call)
       /yolo/detect_object  (graspnet_msgs/srv/ObjectDetectionRequest)

Both eventually return the highest-confidence match for the requested
name, with 2D bbox + 3D camera-frame centroid (median depth in bbox
back-projected through the camera_info K matrix).

Lifecycle:
    on_init      — parse config (LLM endpoint, model, prompts), resolve
                   atlas camera contracts, spawn rclpy background thread.
    on_deactivate — stop rclpy thread.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import toml
from openai import OpenAI

from robonix_api import ATLAS, Service, Ok, Err  # noqa: E402

# ── logging setup ───────────────────────────────────────────────────────────
# Use the same pattern as mid360_lidar_rbnx: simple basicConfig for stderr,
# then manually add a FileHandler. Use force=True (Python 3.8+) to ensure
# configuration takes effect even if robonix_api already configured the
# root logger during import.
_LOG_DIR = Path(os.environ.get(
    "LLM_DETECT_LOG_DIR",
    os.path.join(Path.home(), ".llm_detect", "logs"),
))
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "llm_detect.log"

# Directory for saving annotated detection images.
_DETECTION_IMG_DIR = Path(os.environ.get(
    "LLM_DETECT_IMG_DIR",
    os.path.join(Path.home(), ".llm_detect", "detections"),
))
_DETECTION_IMG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.environ.get("LLM_DETECT_LOG_LEVEL", "INFO"),
    format="[llm_detect] %(message)s",
    force=True,
)
log = logging.getLogger("llm_detect")
# Add file handler explicitly (append mode).
_file_handler = logging.FileHandler(str(_LOG_FILE), mode="a")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "[llm_detect] %(asctime)s %(levelname)s %(message)s"
))
log.addHandler(_file_handler)
log.info("log file: %s", _LOG_FILE)

# Provider id MUST match the deploy manifest's `service: - name: ...`.
llm_detect = Service(
    id=os.environ.get("ROBONIX_CAPABILITY_ID", "llm_detect"),
    namespace="robonix/service/perception/object_detect",
)

# ── shared state (between rclpy thread + MCP handlers) ──────────────────────
_state_lock = threading.Lock()
_initialized = False

# Cached cfg dict from on_init.
_LATEST_CFG: dict[str, Any] = {}

# Latest synchronized camera frame, captured by message_filters callback.
_latest_color_image = None       # numpy ndarray, BGR (from cv_bridge)
_latest_depth_image = None       # numpy ndarray, depth in mm (uint16)
_latest_camera_info = None       # sensor_msgs/CameraInfo

# LLM client (initialized in on_init).
_llm_client: Optional[OpenAI] = None
_llm_model: str = ""
_llm_temperature: float = 0.0
_prompts: dict[str, str] = {}
_rotation_cam2arm: bool = False

# cv_bridge instance.
_bridge = None

# ROS thread state.
_ros_node = None
_ros_thread: Optional[threading.Thread] = None
_ros_stop_evt = threading.Event()

# Synchronization between init() and _ros_thread_main.
_ros_ready_evt = threading.Event()
_ros_thread_error: Optional[BaseException] = None
_ROS_READY_TIMEOUT_S = 15.0


# ── upstream-resolution helpers ─────────────────────────────────────────────
_DEFAULT_TOPICS = {
    "rgb":         "/camera/color/image_raw",
    "depth":       "/camera/depth/image_raw",
    "camera_info": "/camera/color/camera_info",
}

_DEP_CONTRACTS = {
    "rgb":         "robonix/primitive/camera/rgb",
    "depth":       "robonix/primitive/camera/depth",
    "camera_info": "robonix/primitive/camera/camera_info",
}


def _resolve_topic(key: str, cfg: dict) -> str:
    """Resolve the ROS topic name to subscribe for `key`.

    Priority:
      1. cfg[f'{key}_topic'] — explicit override
      2. atlas find_capability(<contract>, transport=ros2) → endpoint
      3. _DEFAULT_TOPICS[key]
    """
    explicit = (cfg.get(f"{key}_topic") or "").strip()
    if explicit:
        log.info("topic[%s] explicit cfg override: %s", key, explicit)
        return explicit

    contract_id = _DEP_CONTRACTS[key]
    try:
        caps = ATLAS.find_capability(contract_id=contract_id, transport="ros2")
    except Exception as e:  # noqa: BLE001
        log.warning("atlas query %s failed: %s — falling back to default",
                    contract_id, e)
        caps = []

    if caps:
        try:
            ch = llm_detect.connect_capability(caps[0], contract_id, "ros2")
            ep = ch.endpoint
            try:
                ch.close()
            except Exception:  # noqa: BLE001
                pass
            if ep:
                log.info("topic[%s] resolved via atlas: %s (provider=%s)",
                         key, ep, caps[0].provider_id)
                return ep
        except Exception as e:  # noqa: BLE001
            log.warning("atlas connect %s failed: %s", contract_id, e)

    fallback = _DEFAULT_TOPICS[key]
    log.warning("topic[%s] no atlas provider; using default %s", key, fallback)
    return fallback


# ── LLM detection helpers ───────────────────────────────────────────────────
def _extract_json_from_markdown(text: str) -> str:
    """Extract JSON content from possible markdown code fences."""
    m = re.search(r"```(?:json|JSON)\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return text


def _call_llm_detect(image_bgr: np.ndarray, object_name: str) -> dict:
    """Call the LLM API with the image and object name, return parsed result.

    Returns dict with keys: success, class_name, box_center_x/y, box_width/height
    (all in normalized 0-1 coords), or success=False on failure.
    """
    global _llm_client, _llm_model, _llm_temperature, _prompts, _rotation_cam2arm

    if _llm_client is None:
        log.error("_call_llm_detect: LLM client is None")
        return {"success": False, "message": "LLM client not initialized"}

    # Rotate 180 degrees if camera is mounted opposite to arm.
    import cv2
    img = image_bgr.copy()
    if _rotation_cam2arm:
        log.debug("rotating image 180 degrees (rotation_cam2arm=True)")
        img = cv2.rotate(img, cv2.ROTATE_180)

    img_h, img_w = img.shape[:2]
    log.info("_call_llm_detect: object='%s', image_size=%dx%d, model=%s",
             object_name, img_w, img_h, _llm_model)

    # Encode image to base64 JPEG.
    _, img_encoded = cv2.imencode(".jpg", img)
    image_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
    log.debug("encoded image to base64 JPEG, length=%d chars", len(image_base64))

    # Build prompt from template.
    prompt_template = _prompts.get("single_detect_prompt", "")
    if not prompt_template:
        log.error("prompt template 'single_detect_prompt' not found in loaded prompts")
        return {"success": False, "message": "prompt template 'single_detect_prompt' not found"}
    prompt = prompt_template.replace("{object_name}", object_name)
    log.info("prompt (first 200 chars): %s", prompt[:200])

    # Call LLM API.
    log.info("calling LLM API: base_url=%s, model=%s, temperature=%s",
             _llm_client.base_url, _llm_model, _llm_temperature)
    t_api = time.time()
    try:
        completion = _llm_client.chat.completions.create(
            model=_llm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            temperature=_llm_temperature,
            response_format={"type": "json_object"},
        )
    except Exception as e:  # noqa: BLE001
        elapsed_api = time.time() - t_api
        log.error("LLM API call FAILED after %.2fs: %s: %s",
                  elapsed_api, type(e).__name__, e)
        return {"success": False, "message": f"LLM API call failed: {e}"}

    elapsed_api = time.time() - t_api
    log.info("LLM API responded in %.2fs", elapsed_api)

    if not completion.choices:
        log.error("LLM returned no choices. completion=%s", completion)
        return {"success": False, "message": "LLM returned no choices"}

    content = completion.choices[0].message.content
    if not content:
        log.error("LLM returned empty content. finish_reason=%s",
                  completion.choices[0].finish_reason)
        return {"success": False, "message": "LLM returned empty content"}

    log.info("LLM raw response (first 500 chars): %s", content[:500])

    # Parse JSON response.
    try:
        raw_json = _extract_json_from_markdown(content)
        result = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError) as e:
        log.error("JSON parse failed: %s. Raw content: %s", e, content[:1000])
        return {"success": False, "message": f"Failed to parse LLM response: {e}"}

    log.info("parsed LLM result: %s", result)

    # Check if detection failed.
    if result.get("failed", False):
        log.info("LLM reports object '%s' not found", object_name)
        return {
            "success": False,
            "message": f"object '{object_name}' not found by LLM",
        }

    # Validate normalized coordinates.
    try:
        cx = float(result["box_center_x"])
        cy = float(result["box_center_y"])
        w = float(result["box_width"])
        h = float(result["box_height"])
    except (KeyError, TypeError, ValueError) as e:
        log.error("invalid bbox fields in LLM response: %s. result=%s", e, result)
        return {"success": False, "message": f"Invalid bbox in LLM response: {e}"}

    log.info("normalized bbox: cx=%.4f cy=%.4f w=%.4f h=%.4f", cx, cy, w, h)

    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
            and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
        log.error("bbox out of [0,1] range: cx=%.4f cy=%.4f w=%.4f h=%.4f",
                  cx, cy, w, h)
        return {"success": False, "message": f"LLM bbox out of range: cx={cx}, cy={cy}, w={w}, h={h}"}

    # If rotation was applied, flip coordinates back.
    if _rotation_cam2arm:
        cx = 1.0 - cx
        cy = 1.0 - cy
        log.debug("flipped coords after rotation: cx=%.4f cy=%.4f", cx, cy)

    return {
        "success": True,
        "class_name": result.get("class_name", object_name),
        "box_center_x": cx,
        "box_center_y": cy,
        "box_width": w,
        "box_height": h,
    }


# ── detection core (shared between MCP + ROS service) ───────────────────────
def _detect_object(object_name: str) -> dict:
    """Detection core. Returns a dict with the same keys both surfaces fill.

    Returns:
      {
        "success":          bool,
        "message":          str,
        "bbox_2d":          [x_min, y_min, x_max, y_max] (pixels) or [],
        "object_center_3d": [x, y, z] (meters, camera optical) or [],
        "confidence":       float,
      }
    """
    global _latest_color_image, _latest_depth_image, _latest_camera_info

    with _state_lock:
        if (_latest_color_image is None or _latest_depth_image is None
                or _latest_camera_info is None):
            return {
                "success": False,
                "message": ("camera data not available "
                            "(waiting for synchronized RGB+depth+camera_info)"),
                "bbox_2d": [],
                "object_center_3d": [],
                "confidence": 0.0,
            }
        color_img = _latest_color_image.copy()
        depth_img = _latest_depth_image.copy()
        cam_info = _latest_camera_info

    # 1. LLM-based detection.
    img_h, img_w = color_img.shape[:2]
    log.info("_detect_object: object='%s', frame_size=%dx%d", object_name, img_w, img_h)
    t0 = time.time()
    llm_result = _call_llm_detect(color_img, object_name)
    elapsed = time.time() - t0
    log.info("LLM detection took %.2fs for '%s', success=%s",
             elapsed, object_name, llm_result.get("success", False))

    if not llm_result.get("success", False):
        log.warning("detection failed for '%s': %s",
                    object_name, llm_result.get("message", "unknown"))
        # Save the raw input image even on failure for debugging.
        _save_detection_image(color_img, object_name, None, success=False)
        return {
            "success": False,
            "message": llm_result.get("message", "LLM detection failed"),
            "bbox_2d": [],
            "object_center_3d": [],
            "confidence": 0.0,
        }

    # 2. Convert normalized coords to pixel bbox [x_min, y_min, x_max, y_max].
    cx = llm_result["box_center_x"]
    cy = llm_result["box_center_y"]
    w = llm_result["box_width"]
    h = llm_result["box_height"]

    x_min = int(max(0, (cx - w / 2) * img_w))
    y_min = int(max(0, (cy - h / 2) * img_h))
    x_max = int(min(img_w, (cx + w / 2) * img_w))
    y_max = int(min(img_h, (cy + h / 2) * img_h))
    bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]
    log.info("pixel bbox: [%d, %d, %d, %d]", x_min, y_min, x_max, y_max)

    # 3. 3D back-projection from depth + intrinsics.
    center_3d = _back_project_3d(bbox, depth_img, cam_info)
    log.info("3D center: %s", center_3d)

    # Heuristic confidence based on bbox area ratio.
    bbox_area = (x_max - x_min) * (y_max - y_min)
    img_area = img_w * img_h
    confidence = min(1.0, max(0.5, 0.5 + 0.5 * bbox_area / img_area))
    log.info("confidence=%.3f (bbox_area=%d, img_area=%d)",
             confidence, bbox_area, img_area)

    # 4. Save annotated image with bbox drawn.
    _save_detection_image(color_img, object_name, bbox, success=True,
                          class_name=llm_result.get("class_name", object_name),
                          confidence=confidence)

    return {
        "success":          True,
        "message":          f"detected '{object_name}' via LLM ({elapsed:.2f}s)",
        "bbox_2d":          bbox,
        "object_center_3d": center_3d if center_3d is not None else [],
        "confidence":       float(confidence),
    }


def _save_detection_image(image_bgr: np.ndarray, object_name: str,
                          bbox: "list | None", success: bool = True,
                          class_name: str = "", confidence: float = 0.0):
    """Save an annotated detection image to disk for debugging.

    On success: draws green bbox + label.
    On failure: saves raw image with red "FAILED" label.
    Images are saved to ~/.llm_detect/detections/ with timestamp.
    """
    try:
        import cv2
        img = image_bgr.copy()
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")

        if success and bbox is not None:
            x_min, y_min, x_max, y_max = [int(v) for v in bbox]
            # Draw green bounding box.
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                          (0, 255, 0), 2)
            # Draw label background.
            label = f"{class_name} ({confidence:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.6, 1)
            cv2.rectangle(img, (x_min, y_min - th - 8),
                          (x_min + tw + 4, y_min), (0, 255, 0), -1)
            cv2.putText(img, label, (x_min + 2, y_min - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                        cv2.LINE_AA)
            filename = f"{timestamp_str}_{object_name}_ok.jpg"
        else:
            # Draw red "FAILED" text.
            cv2.putText(img, f"FAILED: {object_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                        cv2.LINE_AA)
            filename = f"{timestamp_str}_{object_name}_fail.jpg"

        filepath = _DETECTION_IMG_DIR / filename
        cv2.imwrite(str(filepath), img)
        log.info("[save] annotated image saved: %s", filepath)

        # Also save the latest detection as a fixed-name file for quick access.
        latest_path = _DETECTION_IMG_DIR / "latest_detection.jpg"
        cv2.imwrite(str(latest_path), img)
    except Exception as e:  # noqa: BLE001
        log.warning("failed to save detection image: %s", e)


def _back_project_3d(bbox_2d, depth_img, cam_info):
    """Median-depth back-projection. Returns [x, y, z] meters, or None.

    Same algorithm as yolo_world_rbnx: median (not mean) on the bbox
    depth ROI to reject background/zero pixels. Depth in mm → meters.
    """
    try:
        x_min, y_min, x_max, y_max = [int(v) for v in bbox_2d]
        roi = depth_img[y_min:y_max, x_min:x_max]
        valid = roi[(roi > 0) & (roi < 3000)]   # mm, max 3m
        if len(valid) == 0:
            log.warning("back-project: no valid depth in bbox %s",
                        (x_min, y_min, x_max, y_max))
            return None
        z = float(np.median(valid)) / 1000.0    # mm → m

        cx_pix = (x_min + x_max) / 2.0
        cy_pix = (y_min + y_max) / 2.0
        K = cam_info.k
        fx, fy = float(K[0]), float(K[4])
        cx, cy = float(K[2]), float(K[5])

        return [
            (cx_pix - cx) * z / fx,
            (cy_pix - cy) * z / fy,
            z,
        ]
    except Exception as e:  # noqa: BLE001
        log.error("back-project failed: %s", e)
        return None


# ── ROS bring-up (background thread) ────────────────────────────────────────
def _ros_thread_main(rgb_topic: str, depth_topic: str, info_topic: str) -> None:
    """Subscribe + ROS service host + topic publishers, all in one rclpy
    node. Stays alive for the lifetime of the package.
    """
    global _ros_node, _bridge
    global _latest_color_image, _latest_depth_image, _latest_camera_info
    global _ros_thread_error

    node = None
    try:
        import rclpy                              # noqa: E402
        from rclpy.node import Node               # noqa: E402
        from sensor_msgs.msg import Image, CameraInfo  # noqa: E402
        from cv_bridge import CvBridge            # noqa: E402
        import message_filters                    # noqa: E402
        from graspnet_msgs.srv import ObjectDetectionRequest  # noqa: E402
        from graspnet_msgs.msg import DetectedObject, DetectedObjects  # noqa: E402

        rclpy.init(args=None)
        _bridge = CvBridge()
        node = Node("llm_detect_node")
        _ros_node = node

        # Synchronized RGB + depth + camera_info subscribers.
        sub_rgb   = message_filters.Subscriber(node, Image,      rgb_topic)
        sub_depth = message_filters.Subscriber(node, Image,      depth_topic)
        sub_info  = message_filters.Subscriber(node, CameraInfo, info_topic)
        sync = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_depth, sub_info], queue_size=10, slop=0.1)

        def _camera_cb(rgb_msg, depth_msg, info_msg):
            global _latest_color_image, _latest_depth_image, _latest_camera_info
            try:
                rgb   = _bridge.imgmsg_to_cv2(rgb_msg,   desired_encoding="passthrough")
                depth = _bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            except Exception as e:  # noqa: BLE001
                node.get_logger().error(f"camera_cb cv_bridge: {e}")
                return
            with _state_lock:
                _latest_color_image = rgb
                _latest_depth_image = depth
                _latest_camera_info = info_msg
        sync.registerCallback(_camera_cb)
        log.info("subscribed: rgb=%s  depth=%s  info=%s",
                 rgb_topic, depth_topic, info_topic)

        # Compat ROS service (pick.py, yolo_grasp.py both call this).
        def _ros_service_handler(request, response):
            result = _detect_object(request.object_name)
            response.success           = result["success"]
            response.message           = result["message"]
            response.bbox_2d           = list(result["bbox_2d"])
            response.object_center_3d  = list(result["object_center_3d"])
            response.confidence        = float(result["confidence"])
            return response
        node.create_service(ObjectDetectionRequest, "/yolo/detect_object",
                            _ros_service_handler)
        log.info("ROS service up: /yolo/detect_object")

        # Compat publishers (kept for subscribers like visualization tools).
        detection_image_pub = node.create_publisher(Image, "/yolo/detection_image", 10)
        detected_objects_pub = node.create_publisher(
            DetectedObjects, "/yolo/detect_objects", 10)
        globals()["_detection_image_pub"]  = detection_image_pub
        globals()["_detected_objects_pub"] = detected_objects_pub

    except BaseException as e:  # noqa: BLE001
        _ros_thread_error = e
        log.error("rclpy thread setup failed: %s: %s",
                  type(e).__name__, e, exc_info=True)
        try:
            if node is not None:
                node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            import rclpy as _rclpy_for_shutdown
            if _rclpy_for_shutdown.ok():
                _rclpy_for_shutdown.shutdown()
        except Exception:  # noqa: BLE001
            pass
        _ros_ready_evt.set()
        return

    # Setup OK — let init() proceed.
    _ros_ready_evt.set()

    # Spin until told to stop.
    import rclpy  # noqa: E402
    while not _ros_stop_evt.is_set():
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
        except Exception as e:  # noqa: BLE001
            log.warning("rclpy.spin_once raised: %s", e)
    try:
        node.destroy_node()
    except Exception:  # noqa: BLE001
        pass
    try:
        rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass
    log.info("rclpy thread exited")


# ── lifecycle ───────────────────────────────────────────────────────────────
@llm_detect.on_init
def init(cfg):
    """Driver(CMD_INIT). Steps:
      1. parse cfg + initialize LLM client
      2. load prompts from config/prompts.toml
      3. resolve atlas camera contracts → topic names
      4. spawn rclpy thread (subscribers + service + publishers)
    """
    global _initialized, _llm_client, _llm_model, _llm_temperature
    global _prompts, _rotation_cam2arm
    with _state_lock:
        if _initialized:
            return Ok()

    cfg = cfg or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg) if cfg else {}
        except json.JSONDecodeError as e:
            return Err(f"bad config_json: {e}")

    global _LATEST_CFG
    _LATEST_CFG = cfg

    # 1. Initialize LLM client.
    llm_base_url = cfg.get("llm_base_url", "").strip()
    llm_api_key = cfg.get("llm_api_key", "any").strip()
    _llm_model = cfg.get("llm_model", "").strip()
    _llm_temperature = float(cfg.get("temperature", 0.0))
    _rotation_cam2arm = bool(cfg.get("rotation_cam2arm", False))
    # API request timeout in seconds. Default 30s — much shorter than
    # OpenAI SDK's default of 600s, so we don't hang indefinitely if the
    # upstream API stalls.
    llm_timeout = float(cfg.get("llm_timeout_s", 30.0))
    llm_max_retries = int(cfg.get("llm_max_retries", 1))

    if not llm_base_url:
        return Err("config.llm_base_url is required (OpenAI-compatible endpoint)")
    if not _llm_model:
        return Err("config.llm_model is required (e.g. 'google/gemini-3-pro-preview')")

    log.info("initializing LLM client: base_url=%s model=%s temperature=%s timeout=%.1fs max_retries=%d",
             llm_base_url, _llm_model, _llm_temperature, llm_timeout, llm_max_retries)
    try:
        _llm_client = OpenAI(
            base_url=llm_base_url,
            api_key=llm_api_key,
            timeout=llm_timeout,
            max_retries=llm_max_retries,
        )
    except Exception as e:  # noqa: BLE001
        return Err(f"Failed to create OpenAI client: {e}")

    # 2. Load prompts.
    pkg_root = Path(os.environ.get(
        "RBNX_PACKAGE_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    ))
    prompts_path = cfg.get("prompts_file") or str(pkg_root / "config" / "prompts.toml")
    if not Path(prompts_path).is_file():
        return Err(f"Prompts file not found: {prompts_path}")
    try:
        prompts_data = toml.load(prompts_path)
        _prompts = prompts_data.get("prompts", {})
    except Exception as e:  # noqa: BLE001
        return Err(f"Failed to load prompts: {e}")
    log.info("loaded %d prompt templates from %s", len(_prompts), prompts_path)

    # 3. Resolve atlas camera contracts.
    rgb_topic   = _resolve_topic("rgb",         cfg)
    depth_topic = _resolve_topic("depth",       cfg)
    info_topic  = _resolve_topic("camera_info", cfg)

    # 4. Spawn rclpy thread.
    global _ros_thread, _ros_thread_error
    _ros_stop_evt.clear()
    _ros_ready_evt.clear()
    _ros_thread_error = None
    _ros_thread = threading.Thread(
        target=_ros_thread_main,
        args=(rgb_topic, depth_topic, info_topic),
        name="llm_detect-ros",
        daemon=True,
    )
    _ros_thread.start()

    if not _ros_ready_evt.wait(timeout=_ROS_READY_TIMEOUT_S):
        _ros_stop_evt.set()
        _ros_thread.join(timeout=2.0)
        return Err(
            f"rclpy thread did not become ready within "
            f"{_ROS_READY_TIMEOUT_S}s"
        )

    if _ros_thread_error is not None:
        err = _ros_thread_error
        _ros_stop_evt.set()
        _ros_thread.join(timeout=2.0)
        return Err(
            f"rclpy thread setup failed: {type(err).__name__}: {err}"
        )

    with _state_lock:
        _initialized = True
    log.info("init complete: object_detect MCP + /yolo/detect_object live")
    return Ok()


@llm_detect.on_deactivate
def deactivate():
    """ACTIVE → INACTIVE. Stop the rclpy thread."""
    log.info("CMD_DEACTIVATE: stopping rclpy thread")
    _ros_stop_evt.set()
    if _ros_thread is not None:
        _ros_thread.join(timeout=5.0)
    with _state_lock:
        global _initialized
        _initialized = False
    return Ok()


# ── atlas-routed MCP handler (Pilot's view) ─────────────────────────────────
from perception_mcp import (  # noqa: E402  pylint: disable=wrong-import-position
    DetectObject_Request, DetectObject_Response,
)


@llm_detect.mcp("robonix/service/perception/object_detect/detect_object")
def detect_object(req: DetectObject_Request) -> DetectObject_Response:
    """Detect a named object using LLM vision."""
    result = _detect_object(req.object_name)
    return DetectObject_Response(
        bbox_2d          = list(result["bbox_2d"]),
        object_center_3d = list(result["object_center_3d"]),
        confidence       = float(result["confidence"]),
        success          = bool(result["success"]),
        message          = str(result["message"]),
    )


def main() -> int:
    import signal
    def _on_signal(sig, _frame):
        log.info("signal %d — shutting down", sig)
        _ros_stop_evt.set()
        if _ros_thread is not None:
            _ros_thread.join(timeout=3.0)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)
    try:
        llm_detect.run()
    finally:
        _ros_stop_evt.set()
        if _ros_thread is not None:
            _ros_thread.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
