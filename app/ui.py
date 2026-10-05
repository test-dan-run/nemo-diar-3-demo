"""Gradio front end for the diarization service."""

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .diarizer import get_diarizer, load_audio, to_rttm

COLORS = plt.get_cmap("tab10").colors


def plot_timeline(result: dict):
    speakers = result["speakers"] or ["(none)"]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(speakers) + 1.2))
    for seg in result["segments"]:
        row = speakers.index(seg["speaker"])
        ax.broken_barh(
            [(seg["start"], seg["end"] - seg["start"])], (row - 0.4, 0.8),
            color=COLORS[row % len(COLORS)],
        )
    ax.set_yticks(range(len(speakers)), speakers)
    ax.invert_yaxis()
    ax.set_xlim(0, max(result["duration"], 0.1))
    ax.set_xlabel("Time (s)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def run(audio_path, threshold, min_duration):
    if not audio_path:
        raise gr.Error("Upload or record an audio clip first.")
    try:
        audio = load_audio(audio_path)
    except ValueError as e:
        raise gr.Error(str(e))
    result = get_diarizer().diarize(audio, threshold=threshold, min_duration=min_duration)
    table = pd.DataFrame(result["segments"], columns=["speaker", "start", "end"])
    summary = (
        f"**{result['num_speakers']} speaker(s)** · {len(result['segments'])} segments · "
        f"{result['duration']:.1f}s of audio processed in {result['processing_time']:.2f}s"
    )
    return summary, plot_timeline(result), table, to_rttm(result)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Nemotron 3 Diarization") as demo:
        gr.Markdown(
            "# Nemotron 3 Diarization demo\n"
            "Who spoke when? Upload or record a clip (up to 8 speakers). "
            "The same model is available as a REST API at `POST /diarize` — see [/docs](/docs)."
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
                threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Speech probability threshold")
                min_duration = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Drop segments shorter than (s)")
                btn = gr.Button("Diarize", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                plot = gr.Plot(label="Timeline")
                table = gr.Dataframe(label="Segments", interactive=False)
                rttm = gr.Code(label="RTTM", language=None)
        btn.click(run, [audio, threshold, min_duration], [summary, plot, table, rttm])
    return demo
