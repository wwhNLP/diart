import argparse
import os
from pathlib import Path

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def _apply_environment(hf_endpoint: str, offline: bool) -> None:
    """Configure model loading the same way as ``run_test.py``.

    huggingface_hub / pyannote.audio / matplotlib read these variables at
    import time, so this must run BEFORE ``from diart import ...`` below.
    """
    # Point the pyannote cache at the huggingface hub cache so downloaded
    # models are reusable, including offline.
    if "PYANNOTE_CACHE" not in os.environ:
        hf_hub_cache = os.environ.get("HF_HUB_CACHE")
        if hf_hub_cache is None:
            hf_home = Path(
                os.environ.get("HF_HOME", "~/.cache/huggingface")
            ).expanduser()
            hf_hub_cache = str(hf_home / "hub")
        os.environ["PYANNOTE_CACHE"] = hf_hub_cache
    # Avoid repeated matplotlib cache creation under a non-writable HOME.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/diart-matplotlib")
    os.environ["HF_ENDPOINT"] = hf_endpoint
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)


# Pre-scan this script's own flags: they must be applied to the environment
# before the libraries below are imported (they freeze env vars at import time).
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument(
    "--hf-endpoint",
    default=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT),
    help=f"Hugging Face endpoint (default: {DEFAULT_HF_ENDPOINT})",
)
_pre_connectivity = _pre_parser.add_mutually_exclusive_group()
_pre_connectivity.add_argument(
    "--offline",
    dest="offline",
    action="store_true",
    default=True,
    help="Only use models already present in the Hugging Face cache (default).",
)
_pre_connectivity.add_argument(
    "--online",
    dest="offline",
    action="store_false",
    help="Contact the Hugging Face endpoint to download or update models.",
)
_pre_args, _ = _pre_parser.parse_known_args()
_apply_environment(_pre_args.hf_endpoint, _pre_args.offline)

import torch

from diart import argdoc, utils
from diart import models as m
from diart import sources as src
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str, help="Server host")
    parser.add_argument("--port", default=7007, type=int, help="Server port")
    parser.add_argument(
        "--pipeline",
        default="SpeakerDiarization",
        type=str,
        help="Class of the pipeline to optimize. Defaults to 'SpeakerDiarization'",
    )
    parser.add_argument(
        "--segmentation",
        default="pyannote/segmentation-3.0",
        type=str,
        help=f"{argdoc.SEGMENTATION}. Defaults to pyannote/segmentation-3.0",
    )
    parser.add_argument(
        "--embedding",
        default="pyannote/embedding",
        type=str,
        help=f"{argdoc.EMBEDDING}. Defaults to pyannote/embedding",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5,
        help=f"{argdoc.DURATION}. Defaults to training segmentation duration",
    )
    parser.add_argument(
        "--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5"
    )
    parser.add_argument(
        "--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5"
    )
    parser.add_argument(
        "--tau-active", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5"
    )
    parser.add_argument(
        "--rho-update", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3"
    )
    parser.add_argument(
        "--delta-new", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1"
    )
    parser.add_argument(
        "--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3"
    )
    parser.add_argument(
        "--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10"
    )
    parser.add_argument(
        "--max-speakers",
        default=20,
        type=int,
        help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20",
    )
    parser.add_argument(
        "--cpu",
        dest="cpu",
        action="store_true",
        help=f"{argdoc.CPU}. Defaults to GPU if available, CPU otherwise",
    )
    parser.add_argument(
        "--output", type=Path, help=f"{argdoc.OUTPUT}. Defaults to no writing"
    )
    parser.add_argument(
        "--hf-token",
        default="true",
        type=str,
        help=f"{argdoc.HF_TOKEN}. Defaults to 'true' (required by pyannote)",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT),
        type=str,
        help=(
            "Hugging Face endpoint. Applied before model libraries are imported "
            f"(default: {DEFAULT_HF_ENDPOINT})"
        ),
    )
    connectivity = parser.add_mutually_exclusive_group()
    connectivity.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        default=True,
        help="Only use models already present in the Hugging Face cache (default).",
    )
    connectivity.add_argument(
        "--online",
        dest="offline",
        action="store_false",
        help="Contact the Hugging Face endpoint to download or update models.",
    )
    parser.add_argument(
        "--normalize-embedding-weights",
        action="store_true",
        help=f"{argdoc.NORMALIZE_EMBEDDING_WEIGHTS}. Defaults to False",
    )
    parser.add_argument(
        "--voiceprint-dir",
        default=None,
        type=str,
        help="声纹库目录 (<dir>/<说话人姓名>/*.wav)。提供后启用流式说话人确认",
    )
    parser.add_argument(
        "--verify-threshold",
        default=0.5,
        type=float,
        help="声纹确认余弦相似度阈值（EER 校准值）。Defaults to 0.5",
    )
    parser.add_argument(
        "--verify-min-chunks",
        default=3,
        type=int,
        help="确认所需连续命中 chunk 数。Defaults to 3",
    )
    parser.add_argument(
        "--verify-ema-alpha",
        default=0.3,
        type=float,
        help="相似度 EMA 平滑系数。Defaults to 0.3",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="禁用流式说话人确认",
    )
    args = parser.parse_args()
    if not args.verify:
        args.voiceprint_dir = None

    # Resolve device
    args.device = torch.device("cpu") if args.cpu else None

    # Resolve models
    hf_token = utils.parse_hf_token_arg(args.hf_token)
    args.segmentation = m.SegmentationModel.from_pretrained(args.segmentation, hf_token)
    args.embedding = m.EmbeddingModel.from_pretrained(args.embedding, hf_token)

    # Resolve pipeline
    pipeline_class = utils.get_pipeline_class(args.pipeline)
    config = pipeline_class.get_config_class()(**vars(args))
    pipeline = pipeline_class(config)

    # Create websocket audio source
    audio_source = src.WebSocketAudioSource(config.sample_rate, args.host, args.port)

    # Run online inference
    inference = StreamingInference(
        pipeline,
        audio_source,
        batch_size=1,
        do_profile=False,
        do_plot=False,
        show_progress=True,
    )

    # Write to disk if required
    if args.output is not None:
        inference.attach_observers(
            RTTMWriter(audio_source.uri, args.output / f"{audio_source.uri}.rttm")
        )

    # Send back responses as RTTM text lines
    inference.attach_hooks(lambda ann_wav: audio_source.send(ann_wav[0].to_rttm()))

    # Run server and pipeline
    inference()


if __name__ == "__main__":
    run()
