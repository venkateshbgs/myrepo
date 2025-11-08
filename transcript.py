#pip install faster_whisper
#
from faster_whisper import WhisperModel
from pathlib import Path
from tqdm import tqdm

# 1️⃣ Folder containing your video/audio files
video_folder = Path(r"E:\Downloads\Python Videos\CollectedVideos\batch1")

# 2️⃣ Load faster-whisper model
# 'base' = good balance; 'tiny' = faster; 'small' = better accuracy
# compute_type="int8" makes CPU inference much faster

model = WhisperModel("base", device="cpu", compute_type="int8")

# 3️⃣ Supported file extensions
video_extensions = (".mp4", ".mkv", ".mov", ".avi", ".wav", ".mp3")

# 4️⃣ Loop over files with progress bar
for video_file in tqdm(video_folder.glob("*"), desc="Processing videos"):
    if video_file.suffix.lower() not in video_extensions:
        continue

    transcript_path = video_file.with_suffix(".txt")

    # Skip files already transcribed
    if transcript_path.exists():
        print(f"⏩ Skipping {video_file.name} (already has transcript)")
        continue

    try:
        print(f"\n Transcribing: {video_file.name}")

        # 5️⃣ Transcribe (returns tuple)
        segments, info = model.transcribe(str(video_file))

        # 6️⃣ Save transcript
        with open(transcript_path, "w", encoding="utf-8") as f:
            for segment in segments:
                f.write(segment.text.strip() + " ")

        print(f" Saved transcript: {transcript_path}")

    except Exception as e:
        print(f" Failed to process {video_file.name}: {e}")

print("\n All files processed successfully!")
