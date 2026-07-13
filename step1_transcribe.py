#!/usr/bin/env python3
"""Step 1: 使用 faster-whisper 将 .m4a 音频流式转录为 VTT 字幕文件。

流式策略:
  - 用 ffprobe 获取总时长，按 SEGMENT_DURATION (30 分钟) 分段
  - ffmpeg 输出 PCM 到 stdout → numpy 数组 → 直接喂给 faster-whisper
  - 零磁盘 I/O，全程内存管道
  - 模型只加载一次驻留 GPU

输出: data/transcripts/<同名>.vtt
"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

import config


def _format_vtt(segments, offset_seconds: float = 0.0) -> str:
    """将 faster-whisper 的 segments 转为 VTT 格式字符串，时间戳加偏移。"""
    lines = []
    for seg in segments:
        start = seg.start + offset_seconds
        end = seg.end + offset_seconds
        start_ts = (
            f"{int(start // 3600):02d}:{int(start % 3600 // 60):02d}:{start % 60:06.3f}"
        )
        end_ts = (
            f"{int(end // 3600):02d}:{int(end % 3600 // 60):02d}:{end % 60:06.3f}"
        )
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def _get_duration(m4a_path: str) -> float:
    """用 ffprobe 获取音频时长（秒）。"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", m4a_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def _read_audio_segment(m4a_path: str, start: float, duration: float) -> np.ndarray:
    """ffmpeg 流式输出 PCM → numpy 数组，不落地。

    返回 float32 数组，shape=(n_samples,)，采样率 16kHz 单声道。
    """
    cmd = [
        config.FFMPEG_PATH,
        "-ss", str(start),
        "-t", str(duration),
        "-i", m4a_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-f", "s16le",         # 原始 PCM，无容器头
        "pipe:1",              # 输出到 stdout
    ]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg failed (rc={proc.returncode}): "
            f"{proc.stderr.decode(errors='replace')[:500]}"
        )

    # s16le → float32 normalized to [-1, 1]
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def _transcribe_segment_stream(model, m4a_path: str, start: float,
                                duration: float, offset: float) -> str:
    """ffmpeg 流式读取 → numpy → faster-whisper 转录，全程不落盘。"""
    audio = _read_audio_segment(m4a_path, start, duration)
    segments, _ = model.transcribe(
        audio,
        beam_size=config.WHISPER_BEAM_SIZE,
        language=config.WHISPER_LANGUAGE,
        batch_size=config.WHISPER_BATCH_SIZE,
        vad_filter=config.WHISPER_VAD_FILTER,
    )
    return _format_vtt(segments, offset)


# --- 硬件自适应配置 ---

# VRAM 余量系数（留给 PyTorch 框架 + 推理临时分配）
_VRAM_HEADROOM = 0.4  # 40% 余量

# 模型 VRAM 需求（float16）
_MODEL_VRAM_MAP = {
    "tiny":      0.6,   # GB
    "base":      0.7,
    "small":     1.5,
    "medium":    2.5,
    "large-v3":  4.5,
}

# 模型排序（从强到弱，优选最强的能跑的）
_MODEL_RANK = ["large-v3", "medium", "small", "base", "tiny"]


def _auto_detect_hardware() -> dict:
    """根据 GPU 显存 / CPU 内存自动选择最佳模型和 batch_size。

    返回: {"model": str, "device": str, "device_index": ..., "batch_size": int}
    """
    import torch

    # 1. 检测 GPU
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if gpu_count > 0:
        # 取可用显存（考虑显存限制比例和余量）
        device_index = config.WHISPER_DEVICE_INDEX
        if device_index == -1:
            device_index = 0  # 自动检测取 GPU 0 判断
        if isinstance(device_index, list):
            device_index = device_index[0]

        total_vram_gb = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
        usable_vram = total_vram_gb * config.WHISPER_GPU_MEMORY_FRACTION * (1 - _VRAM_HEADROOM)

        # 选最强模型
        best_model = "tiny"
        for model_name in _MODEL_RANK:
            if _MODEL_VRAM_MAP.get(model_name, 99) <= usable_vram:
                best_model = model_name
                break

        # batch_size 按剩余显存估算（每 batch 约 0.1 GB）
        model_vram = _MODEL_VRAM_MAP[best_model]
        headroom = usable_vram - model_vram
        batch_size = max(4, min(32, int(headroom / 0.1)))

        # 显卡多的机器自动启用多卡
        final_device_index = config.WHISPER_DEVICE_INDEX
        if final_device_index == -1:
            final_device_index = list(range(gpu_count)) if gpu_count > 1 else 0

        print(f"[STEP1] 🖥️  GPU: {gpu_count}x, VRAM: {total_vram_gb:.1f} GB "
              f"(usable: {usable_vram:.1f} GB)")
        print(f"[STEP1] 🎯 Auto: model={best_model}, batch_size={batch_size}, "
              f"device=cuda, device_index={final_device_index}")

        return {
            "model": best_model,
            "device": "cuda",
            "device_index": final_device_index,
            "batch_size": batch_size,
        }

    # 2. 无 GPU → CPU 降级
    import os
    cpu_count = os.cpu_count() or 4
    cpu_threads = min(8, cpu_count // 2)  # 用一半核心

    print(f"[STEP1] ⚠️  No GPU detected. Using CPU with {cpu_threads} threads. "
          f"Model forced to 'base' for speed.")
    return {
        "model": "base",
        "device": "cpu",
        "device_index": 0,
        "batch_size": 1,  # CPU 不批处理
        "cpu_threads": cpu_threads,
    }


def _resolve_device_index() -> int | list[int]:
    """解析 GPU 设备索引，支持自动检测多卡。"""
    device_index = config.WHISPER_DEVICE_INDEX
    import torch
    gpu_count = torch.cuda.device_count()

    if device_index == -1:
        # 自动检测：全部 GPU 参与负载均衡
        print(f"[STEP1] Auto-detected {gpu_count} GPU(s)")
        return list(range(gpu_count)) if gpu_count > 1 else 0

    if isinstance(device_index, list):
        valid = [i for i in device_index if i < gpu_count]
        if not valid:
            print(f"[STEP1] WARNING: device_index={device_index} but only "
                  f"{gpu_count} GPU(s) available, falling back to GPU 0")
            return 0
        return valid

    if device_index >= gpu_count:
        print(f"[STEP1] WARNING: GPU {device_index} not available "
              f"({gpu_count} GPU(s) detected), falling back to GPU 0")
        return 0
    return device_index


def main():
    from faster_whisper import WhisperModel

    # --- 硬件自适应检测 ---
    hw = _auto_detect_hardware()
    model_name = hw["model"]
    device = hw["device"]
    device_index = hw["device_index"]
    batch_size = hw["batch_size"]
    cpu_threads = hw.get("cpu_threads", config.WHISPER_CPU_THREADS)

    # GPU 显存限制
    if device == "cuda":
        import torch
        indices = device_index if isinstance(device_index, list) else [device_index]
        for gpu_id in indices:
            torch.cuda.set_device(gpu_id)
            torch.cuda.set_per_process_memory_fraction(
                config.WHISPER_GPU_MEMORY_FRACTION, gpu_id
            )
            total_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            print(f"[STEP1] GPU {gpu_id}: memory capped at "
                  f"{config.WHISPER_GPU_MEMORY_FRACTION*100:.0f}% ({total_gb:.1f} GB)")

    # 覆盖为 auto-detected 值
    config.WHISPER_MODEL = model_name
    config.WHISPER_DEVICE = device
    config.WHISPER_BATCH_SIZE = batch_size
    if "cpu_threads" in hw:
        config.WHISPER_CPU_THREADS = cpu_threads

    m4a_files = sorted(
        [f for f in os.listdir(config.CHUNK_DIR) if f.endswith(".m4a")]
    )
    total_files = len(m4a_files)
    print(f"[STEP1] Found {total_files} .m4a file(s)")

    if total_files == 0:
        print("[STEP1] No .m4a files found. Run step0_download.py first.")
        return

    # --- 加载模型（只一次）---
    print(f"[STEP1] Loading model: {config.WHISPER_MODEL} "
          f"(device={config.WHISPER_DEVICE}, device_index={device_index}, "
          f"batch_size={config.WHISPER_BATCH_SIZE}, "
          f"compute={config.WHISPER_COMPUTE_TYPE})")
    t0 = time.time()
    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        device_index=device_index,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        cpu_threads=config.WHISPER_CPU_THREADS,
    )
    print(f"[STEP1] Model loaded in {time.time() - t0:.1f}s")

    for file_idx, m4a_name in enumerate(m4a_files, 1):
        m4a_path = os.path.join(config.CHUNK_DIR, m4a_name)
        vtt_name = Path(m4a_name).stem + ".vtt"
        vtt_path = config.TRANSCRIPT_DIR / vtt_name

        if vtt_path.exists():
            print(f"[STEP1] [{file_idx}/{total_files}] {m4a_name} → "
                  f"SKIP (already exists)")
            continue

        # --- 获取时长并分段 ---
        total_duration = _get_duration(m4a_path)
        total_min = total_duration / 60
        seg_dur = config.SEGMENT_DURATION
        num_segments = int(total_duration // seg_dur) + (
            1 if total_duration % seg_dur > 1 else 0
        )

        print(f"[STEP1] [{file_idx}/{total_files}] {m4a_name} "
              f"({total_min:.1f} min → {num_segments} segments of {seg_dur}s)")

        # --- 流式转录（零磁盘 I/O）---
        all_vtt_parts = []

        for seg_idx in range(num_segments):
            seg_start = seg_idx * seg_dur
            seg_actual_dur = min(seg_dur, total_duration - seg_start)

            if seg_actual_dur < 1.0:
                break

            mins = seg_start / 60
            print(f"[STEP1]   Segment {seg_idx + 1}/{num_segments} "
                  f"(offset {mins:.1f}min, {seg_actual_dur:.0f}s) ...",
                  end=" ", flush=True)

            t1 = time.time()
            try:
                vtt_text = _transcribe_segment_stream(
                    model, m4a_path, seg_start, seg_actual_dur, seg_start,
                )
                all_vtt_parts.append(vtt_text)
                elapsed = time.time() - t1
                ratio = seg_actual_dur / elapsed if elapsed > 0 else 0
                print(f"OK ({elapsed:.1f}s, {ratio:.1f}x realtime)")
            except Exception as e:
                print(f"FAIL ({e})")

        # --- 合并写出 VTT ---
        if all_vtt_parts:
            final_vtt = "WEBVTT\n" + "\n".join(all_vtt_parts)
            vtt_path.write_text(final_vtt, encoding="utf-8")
            vtt_mb = len(final_vtt) / 1024 / 1024
            print(f"[STEP1] [{file_idx}/{total_files}] ✅ {vtt_name} "
                  f"({vtt_mb:.1f} MB, {num_segments} segments)")
        else:
            print(f"[STEP1] [{file_idx}/{total_files}] ❌ {m4a_name} "
                  f"FAILED (no successful segments)")

    print(f"[STEP1] All done. VTT files in {config.TRANSCRIPT_DIR}")


if __name__ == "__main__":
    main()
