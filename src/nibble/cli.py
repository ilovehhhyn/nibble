"""Server and direct-launch entry points share exactly the same native job adapter."""

import argparse
import json
import uuid
from .runtime import Runtime
from .spec import TrainSpec


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    serve = subs.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    direct = subs.add_parser("direct")
    direct.add_argument("spec")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from .server import create_app

        uvicorn.run(create_app(), host="127.0.0.1", port=args.port, workers=1)
    else:
        from pathlib import Path
        from .runner import execute

        runtime = Runtime.load()
        spec = TrainSpec.from_dict(json.loads(Path(args.spec).read_text()))
        report = execute(
            runtime, runtime.root / "state" / "direct" / uuid.uuid4().hex, spec
        )
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["status"] == "succeeded" else 1)


if __name__ == "__main__":
    main()
