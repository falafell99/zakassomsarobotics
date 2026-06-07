import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from speaker import speak_blocking


def main():
    parser = argparse.ArgumentParser(description="Speak one navigation output aloud.")
    parser.add_argument("text", nargs="*", help="Text to speak. Reads stdin when omitted.")
    args = parser.parse_args()

    text = " ".join(args.text).strip()
    if not text:
        text = sys.stdin.read().strip()
    if not text:
        raise SystemExit("No text provided")

    if not speak_blocking(text):
        raise SystemExit("TTS is disabled or the speaker queue is full")


if __name__ == "__main__":
    main()
