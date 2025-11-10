import os
import subprocess
from tqdm import tqdm
from faster_whisper import WhisperModel

# ==============================================================
# CONFIGURATION
# ==============================================================
BASE_PATH = r"E:\Downloads\Videos"   # Root folder
MODEL_SIZE = "base"        # Options: "tiny", "base", "small"
COMPUTE_TYPE = "int16"       # "int8" = fastest for CPU
LANGUAGE = "en"             # Set None for auto-detect
# ==============================================================

def extract_audio(video_path, audio_path):
    """Convert video to audio (WAV, 16kHz mono) using FFmpeg."""
    try:
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-q:a", "3", audio_path, "-y"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"❌ Failed to extract audio from {video_path}: {e}")
        return False


print(f"🔹 Loading Faster-Whisper model ({MODEL_SIZE}, {COMPUTE_TYPE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)

folders = [f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]
print(f"📁 Found {len(folders)} folders.\n")

for folder_name in folders:
    folder_path = os.path.join(BASE_PATH, folder_name)
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))]

    if not video_files:
        print(f"🚫 No video files in '{folder_name}', skipping...")
        continue

    print(f"\n🎬 Processing Folder: {folder_name}")

    for idx, video in enumerate(tqdm(video_files, desc=f"{folder_name}", ncols=100), start=1):
        video_path = os.path.join(folder_path, video)
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        transcript_path = os.path.join(folder_path, f"{folder_name}_{idx}.txt")

        # Step 1️⃣: Convert video to .wav if not already present
        if not os.path.exists(audio_path):
            print(f"🎧 Extracting audio: {video}")
            if not extract_audio(video_path, audio_path):
                continue
        else:
            print(f"⏩ Audio already exists for {video}")

        # Step 2️⃣: Transcribe .wav if not already done
        if os.path.exists(transcript_path):
            print(f"⏩ Transcript already exists: {transcript_path}")
            continue

        try:
            print(f"📝 Transcribing: {audio_path}")
            segments, info = model.transcribe(audio_path, beam_size=5, language=LANGUAGE)

            # Combine all segments
            text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])

            # Save transcript
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(f"# 🎧 {video}\n\n")
                f.write(text)

            print(f"✅ Transcript saved: {transcript_path}")

        except Exception as e:
            print(f"❌ Failed to transcribe {audio_path}: {e}")

print("\n🎉 All videos processed successfully — transcripts are ready!")
