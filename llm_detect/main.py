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

    Returns dict with keys:
      - success:       bool, true if target object was found
      - message:       str, human-readable status
      - target:        dict | None, the matched target object (normalized coords)
      - other_objects: list[dict], other detected objects (normalized coords)
    Each object dict has: class_name, box_center_x/y, box_width/height (0-1).
    """
    global _llm_client, _llm_model, _llm_temperature, _prompts, _rotation_cam2arm

    if _llm_client is None:
        log.error("_call_llm_detect: LLM client is None")
        return {"success": False, "message": "LLM client not initialized",
                "target": None, "other_objects": []}

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
        return {"success": False, "message": "prompt template 'single_detect_prompt' not found",
                "target": None, "other_objects": []}
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
        return {"success": False, "message": f"LLM API call failed: {e}",
                "target": None, "other_objects": []}

    elapsed_api = time.time() - t_api
    log.info("LLM API responded in %.2fs", elapsed_api)

    if not completion.choices:
        log.error("LLM returned no choices. completion=%s", completion)
        return {"success": False, "message": "LLM returned no choices",
                "target": None, "other_objects": []}

    content = completion.choices[0].message.content
    if not content:
        log.error("LLM returned empty content. finish_reason=%s",
                  completion.choices[0].finish_reason)
        return {"success": False, "message": "LLM returned empty content",
                "target": None, "other_objects": []}

    log.info("LLM raw response (first 500 chars): %s", content[:500])

    # Parse JSON response.
    try:
        raw_json = _extract_json_from_markdown(content)
        result = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError) as e:
        log.error("JSON parse failed: %s. Raw content: %s", e, content[:1000])
        return {"success": False, "message": f"Failed to parse LLM response: {e}",
                "target": None, "other_objects": []}

    log.info("parsed LLM result keys=%s, full=%s",
             list(result.keys()), result)

    # Parse new schema: {target_found, target_index, objects: [...]}.
    objects_raw = result.get("objects", [])
    if not isinstance(objects_raw, list):
        log.warning("'objects' field is not a list, got %s; trying legacy schema",
                    type(objects_raw).__name__)
        objects_raw = []
    target_found = bool(result.get("target_found", False))
    target_index = int(result.get("target_index", -1))

    # Backward-compatibility fallback: if LLM returned the OLD single-bbox
    # schema (top-level box_center_x + optional `failed` flag).  We trigger
    # this whenever the top-level dict itself looks like a bbox object,
    # regardless of whether `objects` is present (some models return a
    # mixed/half-broken JSON).
    has_top_level_bbox = (
        "box_center_x" in result and "box_center_y" in result
        and "box_width" in result and "box_height" in result
    )
    if has_top_level_bbox and not objects_raw:
        log.info("detected legacy single-bbox schema (top-level bbox + empty objects), converting")
        legacy_failed = bool(result.get("failed", False))
        objects_raw = [{
            "class_name":   result.get("class_name", object_name),
            "box_center_x": result.get("box_center_x", 0.0),
            "box_center_y": result.get("box_center_y", 0.0),
            "box_width":    result.get("box_width", 0.0),
            "box_height":   result.get("box_height", 0.0),
            "is_target":    not legacy_failed,
        }]
        target_found = not legacy_failed
        target_index = 0 if target_found else -1
    elif has_top_level_bbox and objects_raw:
        # LLM returned BOTH top-level bbox AND objects[]. Treat the top-level
        # bbox as the target if no object in the list is marked as target.
        any_marked = any(bool(o.get("is_target")) for o in objects_raw
                         if isinstance(o, dict))
        if not any_marked:
            log.info("LLM returned top-level bbox + objects[] without is_target; "
                     "treating top-level bbox as target")
            legacy_failed = bool(result.get("failed", False))
            if not legacy_failed:
                # Prepend top-level bbox as is_target=True.
                objects_raw = [{
                    "class_name":   result.get("class_name", object_name),
                    "box_center_x": result.get("box_center_x", 0.0),
                    "box_center_y": result.get("box_center_y", 0.0),
                    "box_width":    result.get("box_width", 0.0),
                    "box_height":   result.get("box_height", 0.0),
                    "is_target":    True,
                }] + list(objects_raw)
                target_found = True
                target_index = 0

    # Validate + normalise each object.
    # If LLM returns out-of-range coords (e.g. raw pixel values like 628 instead
    # of 0.628), the object is BAD and we drop it — clamping silently produces
    # a bbox at the image border which is misleading.
    parsed_objects = []
    for i, obj in enumerate(objects_raw):
        try:
            cx = float(obj["box_center_x"])
            cy = float(obj["box_center_y"])
            w  = float(obj["box_width"])
            h  = float(obj["box_height"])
        except (KeyError, TypeError, ValueError) as e:
            log.warning("object[%d] has invalid bbox fields, skipping: %s (raw=%s)",
                        i, e, obj)
            continue
        # Read fuzzy-match score (added by new prompt). Tolerate missing/bad value.
        try:
            tms = float(obj.get("target_match_score", 0.0))
        except (TypeError, ValueError):
            tms = 0.0
        tms = max(0.0, min(1.0, tms))
        # Detect out-of-[0,1] coords (LLM occasionally outputs raw pixels) and
        # drop the object — clamping a value like 628 to 1.0 would silently
        # place the bbox at the image edge.
        oob = [v for v in (cx, cy, w, h) if not (0.0 <= v <= 1.0)]
        if oob:
            log.warning("object[%d] has out-of-range coords (cx=%.3f cy=%.3f w=%.3f h=%.3f); "
                        "dropping (LLM likely returned raw pixels). class_name='%s'",
                        i, cx, cy, w, h, obj.get("class_name", "?"))
            continue
        # Drop near-zero size boxes too.
        if w <= 1e-3 or h <= 1e-3:
            log.warning("object[%d] has degenerate size w=%.4f h=%.4f; dropping. class_name='%s'",
                        i, w, h, obj.get("class_name", "?"))
            continue
        cx_c, cy_c, w_c, h_c = cx, cy, w, h
        # If rotation was applied, flip coordinates back.
        if _rotation_cam2arm:
            cx_c = 1.0 - cx_c
            cy_c = 1.0 - cy_c
        parsed_objects.append({
            "class_name":         str(obj.get("class_name", "unknown")),
            "box_center_x":       cx_c,
            "box_center_y":       cy_c,
            "box_width":          w_c,
            "box_height":         h_c,
            "is_target":          bool(obj.get("is_target", False)),
            "target_match_score": tms,
        })

    log.info("parsed %d objects from LLM response (target_found=%s, target_index=%d)",
             len(parsed_objects), target_found, target_index)
    for i, o in enumerate(parsed_objects):
        log.info("  obj[%d] class='%s' is_target=%s target_match_score=%.2f bbox=(%.3f,%.3f,%.3f,%.3f)",
                 i, o["class_name"], o["is_target"], o["target_match_score"],
                 o["box_center_x"], o["box_center_y"], o["box_width"], o["box_height"])

    # ── Locate target ───────────────────────────────────────────────────
    # Strategy:
    #   (1) Strict match: any object with is_target=true.
    #   (2) target_index pointer (if valid).
    #   (3) target_found + single object.
    #   (4) Heuristic substring match on class_name (single-object case).
    #   (5) Fuzzy match: highest target_match_score above FUZZY_THRESHOLD.
    # Steps (1)-(4) yield a STRICT match (fuzzy_matched=False).
    # Step (5) yields a FUZZY match (fuzzy_matched=True) and is logged loudly.
    FUZZY_THRESHOLD = 0.60   # minimum score for fuzzy fallback acceptance

    target = None
    target_idx_in_parsed = -1
    fuzzy_matched = False
    match_reason = ""

    # (1) is_target flag.
    for i, o in enumerate(parsed_objects):
        if o.get("is_target"):
            target = o
            target_idx_in_parsed = i
            match_reason = "strict: is_target=true"
            break
    # (2) target_index pointer.
    if target is None and 0 <= target_index < len(parsed_objects):
        target = parsed_objects[target_index]
        target_idx_in_parsed = target_index
        target["is_target"] = True
        match_reason = f"strict: target_index={target_index}"
    # (3) target_found + single object.
    if target is None and target_found and len(parsed_objects) == 1:
        target = parsed_objects[0]
        target_idx_in_parsed = 0
        target["is_target"] = True
        match_reason = "strict: target_found=true with single object"
    # (4) Heuristic substring match on class_name (single-object case).
    if target is None and len(parsed_objects) == 1:
        only = parsed_objects[0]
        only_name = str(only.get("class_name", "")).lower()
        wanted = object_name.lower()
        if (wanted in only_name) or (only_name in wanted) or (wanted == only_name):
            target = only
            target_idx_in_parsed = 0
            target["is_target"] = True
            match_reason = (f"strict heuristic: single object class_name='{only_name}' "
                            f"matches '{wanted}'")

    # (5) Fuzzy match via target_match_score.
    if target is None and parsed_objects:
        # Sort candidates by score desc.
        scored = sorted(
            enumerate(parsed_objects),
            key=lambda kv: kv[1].get("target_match_score", 0.0),
            reverse=True,
        )
        log.info("[fuzzy] no strict target; candidates (sorted by score) for target='%s':",
                 object_name)
        for idx, o in scored:
            log.info("  cand[%d] class='%s' score=%.2f",
                     idx, o["class_name"], o.get("target_match_score", 0.0))
        best_idx, best_obj = scored[0]
        best_score = best_obj.get("target_match_score", 0.0)
        if best_score >= FUZZY_THRESHOLD:
            target = best_obj
            target_idx_in_parsed = best_idx
            target["is_target"] = True
            fuzzy_matched = True
            match_reason = (f"FUZZY: best score {best_score:.2f} >= threshold "
                            f"{FUZZY_THRESHOLD:.2f}; class_name='{best_obj['class_name']}'")
            log.info("[fuzzy] ACCEPTED: target='%s' -> '%s' (score=%.2f, threshold=%.2f)",
                     object_name, best_obj["class_name"], best_score, FUZZY_THRESHOLD)
        else:
            log.info("[fuzzy] REJECTED: best candidate '%s' score=%.2f < threshold %.2f",
                     best_obj["class_name"], best_score, FUZZY_THRESHOLD)

    other_objects = [o for i, o in enumerate(parsed_objects)
                     if i != target_idx_in_parsed]

    if target is None:
        return {
            "success":        False,
            "message":        (f"object '{object_name}' not found by LLM "
                                f"(but {len(other_objects)} other objects detected)"),
            "target":         None,
            "other_objects":  other_objects,
            "fuzzy_matched":  False,
            "match_reason":   "no target found",
        }

    log.info("target locked: class='%s' score=%.2f reason=%s",
             target["class_name"], target.get("target_match_score", 0.0),
             match_reason)

    return {
        "success":        True,
        "message":        "",
        "target":         target,
        "other_objects":  other_objects,
        "fuzzy_matched":  fuzzy_matched,
        "match_reason":   match_reason,
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

    target = llm_result.get("target")
    other_objects = llm_result.get("other_objects", [])
    fuzzy_matched = bool(llm_result.get("fuzzy_matched", False))
    match_reason = llm_result.get("match_reason", "")
    if fuzzy_matched and target is not None:
        log.warning("[fuzzy] using FUZZY match for '%s' -> '%s' (score=%.2f); reason=%s",
                    object_name, target.get("class_name", "?"),
                    target.get("target_match_score", 0.0), match_reason)

    # Helper: convert a normalised obj dict to pixel bbox.
    def _obj_to_pixel_bbox(o):
        cx, cy = o["box_center_x"], o["box_center_y"]
        w,  h  = o["box_width"],    o["box_height"]
        return [
            int(max(0,     (cx - w / 2) * img_w)),
            int(max(0,     (cy - h / 2) * img_h)),
            int(min(img_w, (cx + w / 2) * img_w)),
            int(min(img_h, (cy + h / 2) * img_h)),
        ]

    if not llm_result.get("success", False):
        log.warning("detection failed for '%s': %s",
                    object_name, llm_result.get("message", "unknown"))
        # Save annotated image with all OTHER objects drawn (blue) so we can
        # at least see what's in the frame.
        _save_detection_image(color_img, object_name,
                              target_bbox=None,
                              other_objects=[
                                  {**o, "bbox": _obj_to_pixel_bbox(o)}
                                  for o in other_objects
                              ],
                              success=False)
        return {
            "success": False,
            "message": llm_result.get("message", "LLM detection failed"),
            "bbox_2d": [],
            "object_center_3d": [],
            "confidence": 0.0,
        }

    # 2. Convert target's normalized coords to pixel bbox.
    bbox = [float(v) for v in _obj_to_pixel_bbox(target)]
    x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    log.info("target pixel bbox: [%d, %d, %d, %d]", x_min, y_min, x_max, y_max)

    # 3. 3D back-projection from depth + intrinsics.
    center_3d = _back_project_3d(bbox, depth_img, cam_info)
    log.info("3D center: %s", center_3d)

    # Heuristic confidence based on bbox area ratio.
    bbox_area = (x_max - x_min) * (y_max - y_min)
    img_area = img_w * img_h
    confidence = min(1.0, max(0.5, 0.5 + 0.5 * bbox_area / img_area))
    log.info("confidence=%.3f (bbox_area=%d, img_area=%d)",
             confidence, bbox_area, img_area)

    # 4. Save annotated image with target (green/yellow) + other objects (blue).
    target_class = target.get("class_name", object_name)
    target_score = float(target.get("target_match_score", 0.0))
    if fuzzy_matched:
        target_label = (f"{target_class} [FUZZY {target_score:.2f}] "
                        f"req='{object_name}' ({confidence:.2f})")
    else:
        target_label = f"{target_class} ({confidence:.2f})"
    _save_detection_image(color_img, object_name,
                          target_bbox={
                              "bbox": [x_min, y_min, x_max, y_max],
                              "class_name": target_class,
                              "confidence": confidence,
                              "label": target_label,
                              "fuzzy": fuzzy_matched,
                          },
                          other_objects=[
                              {**o, "bbox": _obj_to_pixel_bbox(o)}
                              for o in other_objects
                          ],
                          success=True)

    if fuzzy_matched:
        msg = (f"detected '{object_name}' via LLM (FUZZY match -> "
               f"'{target_class}', score={target_score:.2f}, {elapsed:.2f}s)")
    else:
        msg = f"detected '{object_name}' via LLM ({elapsed:.2f}s)"

    return {
        "success":          True,
        "message":          msg,
        "bbox_2d":          bbox,
        "object_center_3d": center_3d if center_3d is not None else [],
        "confidence":       float(confidence),
    }


def _draw_one_box(img, bbox, label, color, thickness=2):
    """Draw one bbox + label on `img` in-place."""
    import cv2
    x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Label background above the box (or below if no room).
        ly = max(y_min - 4, th + 4)
        cv2.rectangle(img, (x_min, ly - th - 4),
                      (x_min + tw + 4, ly), color, -1)
        cv2.putText(img, label, (x_min + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)


def _save_detection_image(image_bgr: np.ndarray, object_name: str,
                          target_bbox: "dict | None" = None,
                          other_objects: "list[dict] | None" = None,
                          success: bool = True):
    """Save an annotated detection image to disk for debugging.

    Always draws ALL detected objects so you can see what's in the frame:
      - target (strict success):  GREEN box, label "<class> (conf)"
      - target (FUZZY success):   YELLOW box, label "<class> [FUZZY <score>]"
      - target (failed):          none drawn (target not found)
      - other_objects:            BLUE boxes, label "<class> @<score>"

    Files use FIXED names (no timestamp) so each call overwrites:
      - latest_ok.jpg     when target was detected successfully
      - latest_fail.jpg   when target was NOT found
      - latest.jpg        always (most recent regardless of outcome)

    other_objects: list of dicts each containing keys:
      - "bbox":               [x_min, y_min, x_max, y_max] in pixels
      - "class_name":         string
      - "target_match_score": float (optional, drawn into label if present)
    target_bbox: dict with keys "bbox", "class_name", "confidence", and
      optional "label" (overrides default) and "fuzzy" (bool).
    """
    try:
        import cv2
        img = image_bgr.copy()
        other_objects = other_objects or []

        # 1. Draw all other objects in BLUE first (so target overlays them).
        for o in other_objects:
            bbox = o.get("bbox")
            if not bbox:
                continue
            cls = str(o.get("class_name", "?"))
            score = o.get("target_match_score")
            if score is not None:
                label = f"{cls} @{float(score):.2f}"
            else:
                label = cls
            _draw_one_box(img, bbox, label, color=(255, 128, 0), thickness=2)

        # 2. Draw target on top. Yellow if fuzzy, green if strict.
        if success and target_bbox is not None:
            is_fuzzy = bool(target_bbox.get("fuzzy", False))
            target_color = (0, 255, 255) if is_fuzzy else (0, 255, 0)  # BGR
            label = target_bbox.get("label") or (
                f"{target_bbox.get('class_name', object_name)} "
                f"({target_bbox.get('confidence', 0.0):.2f})"
            )
            _draw_one_box(img, target_bbox["bbox"], label,
                          color=target_color, thickness=3)
            # Overlay banner at top so the user immediately knows it was fuzzy.
            if is_fuzzy:
                cv2.putText(img, f"FUZZY MATCH: req='{object_name}'",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255), 2, cv2.LINE_AA)
            filename = "latest_ok.jpg"
        else:
            # Target not found — annotate top of image with red FAILED text.
            msg = f"FAILED: '{object_name}' not found ({len(other_objects)} other objs)"
            cv2.putText(img, msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        cv2.LINE_AA)
            filename = "latest_fail.jpg"

        filepath = _DETECTION_IMG_DIR / filename
        cv2.imwrite(str(filepath), img)
        # Also save a generic "latest" pointer that always reflects most recent.
        cv2.imwrite(str(_DETECTION_IMG_DIR / "latest.jpg"), img)
        is_fuzzy_log = (success and target_bbox is not None
                        and bool(target_bbox.get("fuzzy", False)))
        log.info("[save] annotated image saved: %s (target=%s%s, others=%d)",
                 filepath,
                 "OK" if success else "FAIL",
                 " FUZZY" if is_fuzzy_log else "",
                 len(other_objects))
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
