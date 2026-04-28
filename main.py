from __future__ import annotations
from ecg_forecast import build_parser, run_cli
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_cli(args)
if __name__ == "__main__":
    main()
