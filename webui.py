
from typing import Optional
from common.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
import torch, json, os, argparse, datetime, sys, yaml
from common.tts_model import ModelHolder
import gradio as gr
from multiprocessing import cpu_count
from common.log import logger
from common.subprocess_utils import run_script_with_log
from train import ( preprocess_all, initialize, resample, preprocess_text, bert_gen, style_gen, train )
from common.subprocess_utils import run_script_with_log, second_elem_of
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from umap import UMAP
from common.constants import DEFAULT_STYLE
from config import config
from style_vectors import ( load, save_style_vectors_from_clustering, save_style_vectors_from_files, representative_wav_files_gradio, do_clustering_gradio, do_dbscan_gradio )
from app import gr_util, make_non_interactive
from infer import InvalidToneError
from text.japanese import g2kata_tone, kata_tone2phone_tone, text_normalize
# Get path settings
with open(os.path.join("configs", "paths.yml"), "r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    dataset_root = path_config["dataset_root"]
    # assets_root = path_config["assets_root"]


def do_slice(
    model_name: str,
    min_sec: float,
    max_sec: float,
    min_silence_dur_ms: int,
    input_dir: str,
):
    if model_name == "":
        return "Error: Enter model name."
    logger.info("Start slicing...")
    output_dir = os.path.join(dataset_root, model_name, "raw")
    cmd = [
        "slice.py",
        "--output_dir",
        output_dir,
        "--min_sec",
        str(min_sec),
        "--max_sec",
        str(max_sec),
        "--min_silence_dur_ms",
        str(min_silence_dur_ms),
    ]
    if input_dir != "":
        cmd += ["--input_dir", input_dir]
    # ignore ONNX warning
    success, message = run_script_with_log(cmd, ignore_warning=True)
    if not success:
        return f"Error: {message}"
    return "Audio slicing has been completed."


def do_transcribe(
    model_name, 
    whisper_model,
    compute_type, 
    language, 
    #initial_prompt,
    input_dir, 
    device
):
    if model_name == "":
        return "Error: Enter model name."
    initial_prompt = ""
    if input_dir == "":
        input_dir = os.path.join(dataset_root, model_name, "raw")
    output_file = os.path.join(dataset_root, model_name, "esd.list")
    success, message = run_script_with_log(
        [
            "transcribe.py",
            "--input_dir",
            input_dir,
            "--output_file",
            output_file,
            "--speaker_name",
            model_name,
            "--model",
            whisper_model,
            "--compute_type",
            compute_type,
            "--device",
            device,
            "--language",
            language,
            "--initial_prompt",
            f'"{initial_prompt}"',
        ]
    )
    if not success:
        return f"Error: {message}"
    return """Audio transcription has been completed.

Tip (Optional): check the `Data/{model name}/esd.list` for any transcription errors. The more accurate the transcription, the better the training result.
"""

def tts_fn(
    text,
    language,
    reference_audio_path,
    sdp_ratio,
    noise_scale,
    noise_scale_w,
    length_scale,
    line_split,
    split_interval,
    assist_text,
    assist_text_weight,
    use_assist_text,
    style,
    style_weight,
    kata_tone_json_str,
    use_tone,
    speaker,
):
    assert model_holder.current_model is not None

    wrong_tone_message = ""
    kata_tone: Optional[list[tuple[str, int]]] = None
    if use_tone and kata_tone_json_str != "":
        if language != "JP":
            logger.warning("Only Japanese is supported for tone generation.")
            wrong_tone_message = "アクセント指定は現在日本語のみ対応しています。"
        if line_split:
            logger.warning("Tone generation is not supported for line split.")
            wrong_tone_message = "アクセント指定は改行で分けて生成を使わない場合のみ対応しています。"
        try:
            kata_tone = []
            json_data = json.loads(kata_tone_json_str)
            # tupleを使うように変換
            for kana, tone in json_data:
                assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
                kata_tone.append((kana, tone))
        except Exception as e:
            logger.warning(f"Error occurred when parsing kana_tone_json: {e}")
            wrong_tone_message = f"アクセント指定が不正です: {e}"
            kata_tone = None

    # toneは実際に音声合成に代入される際のみnot Noneになる
    tone: Optional[list[int]] = None
    if kata_tone is not None:
        phone_tone = kata_tone2phone_tone(kata_tone)
        tone = [t for _, t in phone_tone]

    speaker_id = model_holder.current_model.spk2id[speaker]

    start_time = datetime.datetime.now()

    try:
        sr, audio = model_holder.current_model.infer(
            text=text,
            language=language,
            reference_audio_path=reference_audio_path,
            sdp_ratio=sdp_ratio,
            noise=noise_scale,
            noisew=noise_scale_w,
            length=length_scale,
            line_split=line_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=use_assist_text,
            style=style,
            style_weight=style_weight,
            given_tone=tone,
            sid=speaker_id,
        )
    except InvalidToneError as e:
        logger.error(f"Tone error: {e}")
        return f"Error: アクセント指定が不正です:\n{e}", None, kata_tone_json_str

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    if tone is None and language == "JP":
        # アクセント指定に使えるようにアクセント情報を返す
        norm_text = text_normalize(text)
        kata_tone = g2kata_tone(norm_text)
        kata_tone_json_str = json.dumps(kata_tone, ensure_ascii=False)
    elif tone is None:
        kata_tone_json_str = ""
    message = f"Success, time: {duration} seconds."
    if wrong_tone_message != "":
        message = wrong_tone_message + "\n" + message
    return message, (sr, audio), kata_tone_json_str



initial_md = """


"""

dataset_md = """
## Before doing this step please prepare your target speaker audio.
## Speach only, no background music, no noise, no other people talking, no effects, no music, no nothing, just the voice of the speaker.

## 5 to 10 minutes of audio is enough, but you can use more if you want.
"""

train_md = """
## Training 4 batch size requires at least 8GB of GPU memory, Not tested on < 8GB GPUs.
## Training a 5 minutes dataset takes ~8 minutes for 100 epochs and 4 batch size tested on RTX 3080.
"""
# Get path settings
with open(os.path.join("configs", "paths.yml"), "r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    # dataset_root = path_config["dataset_root"]
    assets_root = path_config["assets_root"]

languages = [l.value for l in Languages]

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
parser.add_argument(
    "--dir", "-d", type=str, help="Model directory", default=assets_root
)
parser.add_argument(
    "--share", action="store_true", help="Share this app publicly", default=False
)
parser.add_argument(
    "--server-name",
    type=str,
    default=None,
    help="Server name for Gradio app",
)
parser.add_argument(
    "--no-autolaunch",
    action="store_true",
    default=False,
    help="Do not launch app automatically",
)
args = parser.parse_args()
model_dir = args.dir

if args.cpu:
    device = "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

model_holder = ModelHolder(model_dir, device)

model_names = model_holder.model_names
if len(model_names) == 0:
    logger.error(f"モデルが見つかりませんでした。{model_dir}にモデルを置いてください。")
    sys.exit(1)
initial_id = 0
initial_pth_files = model_holder.model_files_dict[model_names[initial_id]]
with gr.Blocks(theme="NoCrypt/miku") as app:
    gr.Markdown(initial_md)
    with gr.Tabs():
        with gr.TabItem("Inference"):
             with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column(scale=3):
                            model_name = gr.Dropdown(
                                label="Model List",
                                choices=model_names,
                                value=model_names[initial_id],
                            )
                            model_path = gr.Dropdown(
                                label="Model File",
                                choices=initial_pth_files,
                                value=initial_pth_files[0],
                            )
                        refresh_button = gr.Button("Refresh", scale=1, visible=True)
                        load_button = gr.Button("Load", scale=1, variant="primary")
                    text_input = gr.TextArea(label="Text", value="Hello world!")
                    
                    line_split = gr.Checkbox(label="Split by newline", value=DEFAULT_LINE_SPLIT)
                    split_interval = gr.Slider(
                        minimum=0.0,
                        maximum=2,
                        value=DEFAULT_SPLIT_INTERVAL,
                        step=0.1,
                        label="Silence duration to insert between new lines (in seconds)",
                    )
                    line_split.change(
                        lambda x: (gr.Slider(visible=x)),
                        inputs=[line_split],
                        outputs=[split_interval],
                    )
                    tone = gr.Textbox(
                        label="Tone adjustment (only numbers 0=low or 1=high are valid)",
                        info="Can only be used when not splitting by newline. It's not perfect.",
                    )
                    use_tone = gr.Checkbox(label="Use tone adjustment", value=False)
                    use_tone.change(
                        lambda x: (gr.Checkbox(value=False) if x else gr.Checkbox()),
                        inputs=[use_tone],
                        outputs=[line_split],
                    )
                    language = gr.Dropdown(choices=languages, value="EN", label="Language")
                    speaker = gr.Dropdown(label="Speaker")
                    with gr.Accordion(label="Detailed Settings", open=False):
                        sdp_ratio = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=DEFAULT_SDP_RATIO,
                            step=0.1,
                            label="SDP Ratio",
                        )
                        noise_scale = gr.Slider(
                            minimum=0.1,
                            maximum=2,
                            value=DEFAULT_NOISE,
                            step=0.1,
                            label="Noise",
                        )
                        noise_scale_w = gr.Slider(
                            minimum=0.1,
                            maximum=2,
                            value=DEFAULT_NOISEW,
                            step=0.1,
                            label="Noise_W",
                        )
                        length_scale = gr.Slider(
                            minimum=0.1,
                            maximum=2,
                            value=DEFAULT_LENGTH,
                            step=0.1,
                            label="Length",
                        )
                        use_assist_text = gr.Checkbox(label="Use assist text", value=False)
                        assist_text = gr.Textbox(
                            label="Assist text",
                            placeholder="どうして私の意見を無視するの？許せない、ムカつく！死ねばいいのに。",
                            info="This text will make the voice sound more like a reading of this text, but the tone and tempo may suffer.",
                            visible=False,
                        )
                        assist_text_weight = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=DEFAULT_ASSIST_TEXT_WEIGHT,
                            step=0.1,
                            label="Weight of assist text",
                            visible=False,
                        )
                        use_assist_text.change(
                            lambda x: (gr.Textbox(visible=x), gr.Slider(visible=x)),
                            inputs=[use_assist_text],
                            outputs=[assist_text, assist_text_weight],
                        )
                with gr.Column():
                    with gr.Accordion("About Styles", open=False):
                        gr.Markdown("About Styles.............")
                    style_mode = gr.Radio(
                        ["Choose from previews", "Enter audio file"],
                        label="How to specify style",
                        value="Choose from previews",
                    )
                    style = gr.Dropdown(
                        label=f"Style ({DEFAULT_STYLE} is average style)",
                        choices=["Load model first"],
                        value="Load model first",
                    )
                    style_weight = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=DEFAULT_STYLE_WEIGHT,
                        step=0.1,
                        label="Weight of style",
                    )
                    ref_audio_path = gr.Audio(label="Reference audio", type="filepath", visible=False)
                    tts_button = gr.Button(
                        "Text to Speech (Load model first)", variant="primary", interactive=False
                    )
                    text_output = gr.Textbox(label="Info")
                    audio_output = gr.Audio(label="Result")
                   # with gr.Accordion("Text Examples", open=False):
                       # gr.Examples(examples, inputs=[text_input, language])

                tts_button.click(
                    tts_fn,
                    inputs=[
                        text_input,
                        language,
                        ref_audio_path,
                        sdp_ratio,
                        noise_scale,
                        noise_scale_w,
                        length_scale,
                        line_split,
                        split_interval,
                        assist_text,
                        assist_text_weight,
                        use_assist_text,
                        style,
                        style_weight,
                        tone,
                        use_tone,
                        speaker,
                    ],
                    outputs=[text_output, audio_output, tone],
                )

                model_name.change(
                    model_holder.update_model_files_gr,
                    inputs=[model_name],
                    outputs=[model_path],
                )

                model_path.change(make_non_interactive, outputs=[tts_button])

                refresh_button.click(
                    model_holder.update_model_names_gr,
                    outputs=[model_name, model_path, tts_button],
                )

                load_button.click(
                    model_holder.load_model_gr,
                    inputs=[model_name, model_path],
                    outputs=[style, tts_button, speaker],
                )

                style_mode.change(
                    gr_util,
                    inputs=[style_mode],
                    outputs=[style, ref_audio_path],
                )

        with gr.TabItem("Voice Training"):
            with gr.Tabs():
                with gr.TabItem("Step 1: Dataset Processing"):
                    gr.Markdown(dataset_md)
                    model_name = gr.Textbox(label="Enter model name (will also be used as speaker name).")
                    gr.Markdown("## Audio slicing")
                    with gr.Accordion("Audio slicing"):
                        with gr.Row():
                            with gr.Column():
                                input_dir = gr.Textbox(
                                    label="Audio files folder",
                                    placeholder="inputs",
                                    info="Please put wav files in the input folder. or type the path to a folder containing your wav files.",
                                )
                                gr.Markdown("### Slicing settings")
                                min_sec = gr.Slider(
                                    minimum=0, maximum=10, value=2, step=0.5, label="Discard audio slices shorter than this duration (in seconds)"
                                )
                                max_sec = gr.Slider(
                                    minimum=0, maximum=15, value=12, step=0.5, label="Discard audio slices longer than this duration (in seconds)"
                                )
                                min_silence_dur_ms = gr.Slider(
                                    minimum=0,
                                    maximum=2000,
                                    value=700,
                                    step=100,
                                    label="Minimum duration of silence to consider as separator (in ms)",
                                )
                                slice_button = gr.Button("Slice audio", variant="primary")
                            result1 = gr.Textbox(label="Result")
                    gr.Markdown("## Dataset Transcription (IMPORTANT)")  
                    with gr.Accordion("Transcription"): 
                        with gr.Row():
                    
                            with gr.Column():
                                raw_dir = gr.Textbox(
                                    label="Folder containing the raw and sliced dataset `Default (Data/{model name}/raw)`",
                                    placeholder="Leave blank if you sclied the dataset in the previous step",
                                )
                                whisper_model = gr.Dropdown(
                                    ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                                    label="Whisper model",
                                    value="large-v3",
                                )
                                compute_type = gr.Dropdown(
                                    [
                                        "int8",
                                        "int8_float32",
                                        "int8_float16",
                                        "int8_bfloat16",
                                        "int16",
                                        "float16",
                                        "bfloat16",
                                        "float32",
                                    ],
                                    label="Computation precision",
                                    value="bfloat16",
                                )
                                device = gr.Radio(["cuda", "cpu"], label="Device", value="cuda")
                                language = gr.Dropdown(["ja", "en", "zh"], value="en", label="Language")
                                """ initial_prompt = gr.Textbox(
                                        label="Initial prompt",
                                        placeholder="こんにちは。元気、ですかー？ふふっ、私は…ちゃんと元気だよ！",
                                        info="For example, the sentence you want to be written like this, if it's in Japanese, you can omit it, if it's in English or Chinese, please write it",
                                    )"""
                            transcribe_button = gr.Button("Transcribe audio", variant="primary")
                            result2 = gr.Textbox(label="Result")
                            
                            slice_button.click(
                                do_slice,
                                inputs=[model_name, min_sec, max_sec, min_silence_dur_ms, input_dir],
                                outputs=[result1],
                            )
                            transcribe_button.click(
                                do_transcribe,
                                inputs=[
                                    model_name,
                                    whisper_model,
                                    compute_type,
                                    language,
                                    #initial_prompt,
                                    raw_dir,
                                    device,
                                ],
                                outputs=[result2],
                            )
                with gr.TabItem("Step 2: Training"):
                    gr.Markdown(train_md)
                    model_name = gr.Textbox(
                        label="Model name",
                        placeholder="Type the same model name you entered in step 1",
                    )
                    gr.Markdown("### Automatic preprocessing")
                    with gr.Row(variant="panel"):
                        with gr.Column():
                            batch_size = gr.Slider(
                                label="Batch size",
                                value=4,
                                minimum=1,
                                maximum=64,
                                step=1,
                            )
                            epochs = gr.Slider(
                                label="Number of epochs",
                                info="100 should be enough, but you can run more and the quality might improve",
                                value=100,
                                minimum=10,
                                maximum=1000,
                                step=10,
                            )
                            save_every_steps = gr.Slider(
                                label="How often to save the results (in steps)",
                                info="Different from the number of epochs",
                                value=1000,
                                minimum=100,
                                maximum=10000,
                                step=100,
                            )
                            bf16_run = gr.Checkbox(
                                label="Use bf16",
                                info="Might make training faster on new GPUs, but might not work on old GPUs",
                                value=True,
                            )
                            num_processes = gr.Slider(
                                label="Number of processes",
                                info="Number of parallel processes for preprocessing, can cause freezing if too large",
                                value=cpu_count() // 2,
                                minimum=1,
                                maximum=cpu_count(),
                                step=1,
                            )
                            normalize = gr.Checkbox(
                                label="Normalize audio volume (if volume is not consistent)",
                                value=False,
                            )
                            trim = gr.Checkbox(
                                label="Trim silence at the beginning and end of audio",
                                value=False,
                            )
                        with gr.Column():
                            preprocess_button = gr.Button(value="Run automatic preprocessing", variant="primary")
                            info_all = gr.Textbox(label="Status")
                    with gr.Accordion(open=False, label="Manual preprocessing (optional)"):
                        with gr.Row(variant="panel"):
                            with gr.Column():
                                gr.Markdown(value="#### Step 1: Generate configuration file")
                                batch_size_manual = gr.Slider(
                                    label="Batch size",
                                    value=4,
                                    minimum=1,
                                    maximum=64,
                                    step=1,
                                )
                                epochs_manual = gr.Slider(
                                    label="Number of epochs",
                                    value=100,
                                    minimum=1,
                                    maximum=1000,
                                    step=1,
                                )
                                save_every_steps_manual = gr.Slider(
                                    label="How often to save the results (in steps)",
                                    value=1000,
                                    minimum=100,
                                    maximum=10000,
                                    step=100,
                                )
                                bf16_run_manual = gr.Checkbox(
                                    label="Use bf16",
                                    value=True,
                                )
                            with gr.Column():
                                generate_config_btn = gr.Button(value="Run", variant="primary")
                                info_init = gr.Textbox(label="Status")
                        with gr.Row(variant="panel"):
                            with gr.Column():
                                gr.Markdown(value="#### Step 2: Resample audio files")
                                num_processes_resample = gr.Slider(
                                    label="Number of processes",
                                    value=cpu_count() // 2,
                                    minimum=1,
                                    maximum=cpu_count(),
                                    step=1,
                                )
                                normalize_resample = gr.Checkbox(
                                    label="Normalize audio volume",
                                    value=False,
                                )
                                trim_resample = gr.Checkbox(
                                    label="Trim silence at the beginning and end of audio",
                                    value=False,
                                )
                            with gr.Column():
                                resample_btn = gr.Button(value="Run", variant="primary")
                                info_resample = gr.Textbox(label="Status")
                        with gr.Row(variant="panel"):
                            with gr.Column():
                                gr.Markdown(value="#### Step 3: Preprocess transcript files")
                            with gr.Column():
                                preprocess_text_btn = gr.Button(value="Run", variant="primary")
                                info_preprocess_text = gr.Textbox(label="Status")
                        with gr.Row(variant="panel"):
                            with gr.Column():
                                gr.Markdown(value="#### Step 4: Generate BERT feature files")
                            with gr.Column():
                                bert_gen_btn = gr.Button(value="Run", variant="primary")
                                info_bert = gr.Textbox(label="Status")
                        with gr.Row(variant="panel"):
                            with gr.Column():
                                gr.Markdown(value="#### Step 5: Generate style feature files")
                                num_processes_style = gr.Slider(
                                    label="Number of processes",
                                    value=cpu_count() // 2,
                                    minimum=1,
                                    maximum=cpu_count(),
                                    step=1,
                                )
                            with gr.Column():
                                style_gen_btn = gr.Button(value="Run", variant="primary")
                                info_style = gr.Textbox(label="Status")
                    gr.Markdown("## Training")
                    with gr.Row(variant="panel"):
                        train_btn = gr.Button(value="Start training", variant="primary")
                        info_train = gr.Textbox(label="Status")
                        preprocess_button.click(
                            second_elem_of(preprocess_all),
                            inputs=[
                                model_name,
                                batch_size,
                                epochs,
                                save_every_steps,
                                bf16_run,
                                num_processes,
                                normalize,
                                trim,
                            ],
                            outputs=[info_all],
                        )
                        generate_config_btn.click(
                            second_elem_of(initialize),
                            inputs=[
                                model_name,
                                batch_size_manual,
                                epochs_manual,
                                save_every_steps_manual,
                                bf16_run_manual,
                            ],
                            outputs=[info_init],
                        )
                        resample_btn.click(
                            second_elem_of(resample),
                            inputs=[
                                model_name,
                                normalize_resample,
                                trim_resample,
                                num_processes_resample,
                            ],
                            outputs=[info_resample],
                        )
                        preprocess_text_btn.click(
                            second_elem_of(preprocess_text),
                            inputs=[model_name],
                            outputs=[info_preprocess_text],
                        )
                        bert_gen_btn.click(
                            second_elem_of(bert_gen),
                            inputs=[model_name],
                            outputs=[info_bert],
                        )
                        style_gen_btn.click(
                            second_elem_of(style_gen),
                            inputs=[model_name, num_processes_style],
                            outputs=[info_style],
                        )
                        train_btn.click(
                            second_elem_of(train), inputs=[model_name], outputs=[info_train]
                        )

                with gr.TabItem("Step 3: Generate Styles (Optional/Experimental)"):
                    with gr.Row():
                        model_name = gr.Textbox(placeholder="your_model_name", label="Model name")
                        reduction_method = gr.Radio(
                            choices=["UMAP", "t-SNE"],
                            label="Reduction method",
                            info="v 1.3 used t-SNE, but UMAP might have better possibilities.",
                            value="Umap",
                        )
                        load_button = gr.Button("Load style vectors", variant="primary")
                    output = gr.Plot(label="Visualization of audio styles")
                    load_button.click(load, inputs=[model_name, reduction_method], outputs=[output])
                    with gr.Tab("Method 1: Automatic style separation"):
                        with gr.Tab("Style separation 1"):
                            n_clusters = gr.Slider(
                                minimum=2,
                                maximum=10,
                                step=1,
                                value=4,
                                label="Number of styles to create (excluding average style)",
                                info="Please try different numbers of styles while looking at the above plot.",
                            )
                            c_method = gr.Radio(
                                choices=["Agglomerative after reduction", "KMeans after reduction", "Agglomerative", "KMeans"],
                                label="Algorithm",
                                info="You can try different clustering algorithms.",
                                value="Agglomerative after reduction",
                            )
                            c_button = gr.Button("Run clustering")
                        with gr.Tab("Style separation 2: DBSCAN"):
                            gr.Markdown("dbscan_md")
                            eps = gr.Slider(
                                minimum=0.1,
                                maximum=10,
                                step=0.01,
                                value=0.3,
                                label="eps",
                            )
                            min_samples = gr.Slider(
                                minimum=1,
                                maximum=50,
                                step=1,
                                value=15,
                                label="min_samples",
                            )
                            with gr.Row():
                                dbscan_button = gr.Button("Run clustering")
                                num_styles_result = gr.Textbox(label="Number of styles")
                        gr.Markdown("Clustering results")
                        gr.Markdown("Note: Since we are reducing it from 256 dimensions to 2 dimensions, the exact position relationship of the vectors cannot be guaranteed.")
                        with gr.Row():
                            gr_plot = gr.Plot()
                            with gr.Column():
                                with gr.Row():
                                    cluster_index = gr.Slider(
                                        minimum=1,
                                        maximum=10,
                                        step=1,
                                        value=1,
                                        label="Style number",
                                        info="The representative audio file of the selected style will be displayed.",
                                    )
                                    num_files = gr.Slider(
                                        minimum=1,
                                        maximum=10,
                                        step=1,
                                        value=5,
                                        label="Number of representative audio files to display",
                                    )
                                    get_audios_button = gr.Button("Get representative audio files")
                                with gr.Row():
                                    audio_list = []
                                    for i in range(10):
                                        audio_list.append(gr.Audio(visible=False, show_label=True))
                            c_button.click(
                                do_clustering_gradio,
                                inputs=[n_clusters, c_method],
                                outputs=[gr_plot, cluster_index] + audio_list,
                            )
                            dbscan_button.click(
                                do_dbscan_gradio,
                                inputs=[eps, min_samples],
                                outputs=[gr_plot, cluster_index, num_styles_result] + audio_list,
                            )
                            get_audios_button.click(
                                representative_wav_files_gradio,
                                inputs=[cluster_index, num_files],
                                outputs=audio_list,
                            )
                        gr.Markdown("If the results look good, let's save them.")
                        style_names = gr.Textbox(
                            "Angry, Sad, Happy",
                            label="Style names",
                            info="Please enter the names of the styles, separated by comma (you can use Japanese). Example: `Angry, Sad, Happy` or `怒り, 悲しみ, 喜び` etc. The average style will be automatically saved as `{DEFAULT_STYLE}`.",
                        )
                        with gr.Row():
                            save_button1 = gr.Button("Save style vectors", variant="primary")
                            info2 = gr.Textbox(label="Saving result")

                        save_button1.click(
                            save_style_vectors_from_clustering,
                            inputs=[model_name, style_names],
                            outputs=[info2],
                        )
                    with gr.Tab("Method 2: Manually select styles"):
                        gr.Markdown("Please enter the filenames of the representative audio files for each style, separated by comma, and the corresponding style names, also separated by comma, in the text box below.")
                        gr.Markdown("Example: `angry.wav, sad.wav, happy.wav` and `Angry, Sad, Happy`")
                        gr.Markdown(f"Note: The `{DEFAULT_STYLE}` style will be automatically saved, so please do not define a style with the name `{DEFAULT_STYLE}`.")
                        with gr.Row():
                            audio_files_text = gr.Textbox(
                                label="Audio file names", placeholder="angry.wav, sad.wav, happy.wav"
                            )
                            style_names_text = gr.Textbox(
                                label="Style names", placeholder="Angry, Sad, Happy"
                            )
                        with gr.Row():
                            save_button2 = gr.Button("Save style vectors", variant="primary")
                            info2 = gr.Textbox(label="Saving result")
                            save_button2.click(
                                save_style_vectors_from_files,
                                inputs=[model_name, audio_files_text, style_names_text],
                                outputs=[info2],
                            )

parser = argparse.ArgumentParser()
parser.add_argument(
    "--server-name",
    type=str,
    default=None,
    help="Server name for Gradio app",
)
parser.add_argument(
    "--no-autolaunch",
    action="store_true",
    default=False,
    help="Do not launch app automatically",
)
args = parser.parse_args()

app.launch(inbrowser=not args.no_autolaunch, server_name=args.server_name)
