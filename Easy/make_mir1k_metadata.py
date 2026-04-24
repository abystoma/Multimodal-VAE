import os
import re
import pandas as pd

ROOT = "data/MIR-1K"
LYRICS_DIR = os.path.join(ROOT, "Lyrics")
WAV_DIR = os.path.join(ROOT, "Wavfile")
OUT_CSV = os.path.join(ROOT, "metadata.csv")


def count_chinese_chars(text: str) -> int:
    return len(re.findall(r'[\u4e00-\u9fff]', text))


def count_latin_letters(text: str) -> int:
    return len(re.findall(r'[A-Za-z]', text))


def detect_language_from_lyrics(text: str) -> str:
    zh_count = count_chinese_chars(text)
    en_count = count_latin_letters(text)

    if zh_count == 0 and en_count == 0:
        return "unknown"

    if zh_count > en_count:
        return "zh"
    if en_count > zh_count:
        return "en"

    return "unknown"


def main():
    if not os.path.exists(LYRICS_DIR):
        raise FileNotFoundError(f"Missing folder: {LYRICS_DIR}")
    if not os.path.exists(WAV_DIR):
        raise FileNotFoundError(f"Missing folder: {WAV_DIR}")

    rows = []

    lyric_files = sorted([f for f in os.listdir(LYRICS_DIR) if f.lower().endswith(".txt")])

    for lyric_file in lyric_files:
        lyric_path = os.path.join(LYRICS_DIR, lyric_file)

        with open(lyric_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        language = detect_language_from_lyrics(text)

        base = os.path.splitext(lyric_file)[0]
        wav_name = base + ".wav"
        wav_path = os.path.join("Wavfile", wav_name)
        full_wav_path = os.path.join(WAV_DIR, wav_name)

        if os.path.exists(full_wav_path):
            rows.append({
                "file_path": wav_path,
                "language": language,
                "lyric_file": lyric_file,
            })

    df = pd.DataFrame(rows)

    print("Before filtering:")
    print(df["language"].value_counts(dropna=False))

    df = df[df["language"].isin(["en", "zh"])].copy()

    print("\nAfter filtering to en/zh:")
    print(df["language"].value_counts(dropna=False))

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()