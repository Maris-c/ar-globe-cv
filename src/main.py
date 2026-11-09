"""
main.py — AR Globe CV: top-level launcher with text menu.

Run from the project root:
    python src/main.py
    python src/main.py --mode image --map assets/images/world_map.jpg
    python src/main.py --mode webcam --map assets/images/world_map.jpg
"""

from __future__ import annotations

import argparse
import sys
import pathlib

_SRC = pathlib.Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import CFG
from utils.file_utils import choose_image_file_dialog, load_image


BANNER = r"""
╔══════════════════════════════════════════════════════╗
║           AR Globe CV  —  Python + OpenCV            ║
║          Equirectangular Map AR Prototype             ║
╠══════════════════════════════════════════════════════╣
║  [1] Image Map   — click on a static map image       ║
║  [2] Webcam Map  — detect & click via live camera    ║
║  [Q] Quit                                            ║
╚══════════════════════════════════════════════════════╝
"""


def _select_image(cli_path: str | None) -> str | None:
    """Return a map image path from CLI arg or file dialog."""
    if cli_path:
        p = pathlib.Path(cli_path)
        if not p.exists():
            print(f"[ERROR] File not found: {p.resolve()}")
            return None
        return str(p)

    # Check assets/images for convenience.
    assets = pathlib.Path(__file__).parent.parent / "assets" / "images"
    candidates = list(assets.glob("*.jpg")) + list(assets.glob("*.png")) + list(assets.glob("*.jpeg"))
    if candidates:
        print(f"\nFound map image(s) in assets/images/:")
        for i, c in enumerate(candidates):
            print(f"  [{i+1}] {c.name}")
        print(f"  [0] Browse for a different file…")
        choice = input("Select (default=1): ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return str(candidates[int(choice) - 1])

    print("\nOpening file dialog…")
    return choose_image_file_dialog()


def run_image_mode(image_path: str) -> None:
    from image_map.image_map_app import ImageMapApp
    try:
        app = ImageMapApp(image_path)
        app.run()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")


def run_webcam_mode(image_path: str, device: int) -> None:
    from webcam.webcam_app import WebcamApp
    try:
        app = WebcamApp(image_path, device_index=device)
        app.run()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")


def interactive_menu(args: argparse.Namespace) -> None:
    print(BANNER)
    while True:
        choice = input("Select mode [1/2/Q]: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Goodbye.")
            sys.exit(0)
        elif choice in ("1", "image"):
            image_path = _select_image(args.map)
            if image_path:
                run_image_mode(image_path)
        elif choice in ("2", "webcam"):
            image_path = _select_image(args.map)
            if image_path:
                run_webcam_mode(image_path, device=args.device)
        else:
            print("Invalid choice. Enter 1, 2, or Q.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AR Globe CV — Prototype launcher",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["image", "webcam", "menu"],
        default="menu",
        help=(
            "image  → Image Map mode\n"
            "webcam → Webcam Map mode\n"
            "menu   → Interactive text menu (default)"
        ),
    )
    parser.add_argument("--map", "-m", type=str, default=None,
                        help="Path to the world map image.")
    parser.add_argument("--device", "-d", type=int,
                        default=CFG["webcam"]["device_index"],
                        help="Webcam device index (default: 0).")
    args = parser.parse_args()

    if args.mode == "image":
        image_path = _select_image(args.map)
        if image_path:
            run_image_mode(image_path)
    elif args.mode == "webcam":
        image_path = _select_image(args.map)
        if image_path:
            run_webcam_mode(image_path, device=args.device)
    else:
        interactive_menu(args)


if __name__ == "__main__":
    main()
