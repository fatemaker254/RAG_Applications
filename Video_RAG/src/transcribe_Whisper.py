import os
import json
import whisper
import subprocess
from config import VIDEO_DIR, TRANSCRIPT_DIR

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# Load local Whisper model (GPU enabled)
model = whisper.load_model("small.en")

def extract_audio_ffmpeg(video_path, audio_path):
    
    
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",  # wav format
        "-ar", "16000",  # sampling rate
        "-ac", "1",  # mono channel
        audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def transcribe(video_path):
    filename = os.path.splitext(os.path.basename(video_path))[0]
    out_json = os.path.join(TRANSCRIPT_DIR, f"{filename}.json")
    audio_path = os.path.join(TRANSCRIPT_DIR, f"{filename}.wav")

    print(f"\n🎬 Processing: {video_path}")
    print("🎧 Extracting audio using FFmpeg...")
    extract_audio_ffmpeg(video_path, audio_path)

    print("🧠 Transcribing with Whisper...")
    result = model.transcribe(audio_path, fp16=True)

    # Save transcript result
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Transcript saved: {out_json}")


if __name__ == "__main__":
    videos = [v for v in os.listdir(VIDEO_DIR) if v.endswith(".mp4")]

    if not videos:
        print("❌ No .mp4 videos found in /data/videos/")
    else:
        for v in videos:
            transcribe(os.path.join(VIDEO_DIR, v))
