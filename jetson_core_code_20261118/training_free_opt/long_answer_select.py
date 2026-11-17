"""Build the long-answer pilot list. QA8 is never used as the accept-rate set."""
from __future__ import annotations

import json
from pathlib import Path

USED_VIDEOS = {
    "v_Ku76zinwWkg", "v_ghxiG0Dnrt0", "4567287291", "4025182492",
    "6709738709", "11606895084",
}
# Extra long-caption videos that exist on the server Test_Videos set.
EXTRA_NEW_CAPTION_IDS = (
    "v_dRF1C_sBE5M", "v_mY7QdIn51S8", "v_Ws12Dt0Ny5Q",
    "v_vMcXLYBcHsM", "v_889_kqLFM0E", "v_4mRkWNLE66E",
)
CAPTION_VIDEO_ROOT = "/data1/lgq2026/jev_spec/assets/datasets/videodetailcaption/videos/Test_Videos"
NEXTQA_VIDEO_ROOT = "/data1/lgq2026/jev_spec/assets/datasets/nextqa/videos/official_train"
DETAIL_PROMPT = (
    "Please provide a detailed description of the video, focusing on the main "
    "subjects, their actions, the order of events, and the background scenes. "
    "Cover visible facts only; do not invent unseen details."
)
MAX_NEW = 512


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _caption_row(video_id: str) -> dict:
    return {
        "request_id": f"pilot_long_{video_id}",
        "video_id": video_id,
        "video_path": f"{CAPTION_VIDEO_ROOT}/{video_id}.mp4",
        "video_source": "videodetailcaption",
        "split": "pilot",
        "task": "desc",
        "nframes": 16,
        "max_pixels": 200704,
    }


def select_pilot(manifest: Path, extra_caption_ids=EXTRA_NEW_CAPTION_IDS):
    rows = load_jsonl(manifest)
    anchors = [r for r in rows if r.get("video_id") in {"v_Ku76zinwWkg", "v_ghxiG0Dnrt0"} and r.get("task") == "desc"]
    new_desc = [r for r in rows if r.get("task") == "desc" and r.get("video_id") not in USED_VIDEOS]
    extra = [_caption_row(vid) for vid in extra_caption_ids]
    chosen = []
    seen = set()
    for row in anchors + extra + new_desc:
        vid = row["video_id"]
        if vid in seen:
            continue
        seen.add(vid)
        item = dict(row)
        item["question"] = DETAIL_PROMPT
        item["max_new_tokens"] = MAX_NEW
        item["natural_eos"] = True
        item["disable_eos"] = False
        item["role"] = "anchor" if vid in {"v_Ku76zinwWkg", "v_ghxiG0Dnrt0"} else "new_long"
        item["qa8_is_accept_eval"] = False
        chosen.append(item)
        if len(chosen) >= 12:
            break
    new_videos = {r["video_id"] for r in chosen if r["role"] == "new_long"}
    return {
        "max_new_tokens": MAX_NEW,
        "natural_eos": True,
        "n": len(chosen),
        "new_videos": sorted(new_videos),
        "new_video_count": len(new_videos),
        "requests": chosen,
        "note": "Pilot list only. Expand after existence check of additional Test_Videos. QA8 excluded.",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = select_pilot(Path(args.manifest))
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"n": payload["n"], "new_video_count": payload["new_video_count"],
                      "new_videos": payload["new_videos"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
