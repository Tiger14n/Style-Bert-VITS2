import argparse
import datetime
import json
import os
import sys
from typing import Optional

import gradio as gr
import torch
import yaml

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
from common.log import logger
from common.tts_model import ModelHolder
from infer import InvalidToneError
from text.japanese import g2kata_tone, kata_tone2phone_tone, text_normalize

# Get path settings
with open(os.path.join("configs", "paths.yml"), "r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    # dataset_root = path_config["dataset_root"]
    assets_root = path_config["assets_root"]

languages = [l.value for l in Languages]


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
    model_holder
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


initial_text = "こんにちは、初めまして。あなたの名前はなんていうの？"

examples = [
    [initial_text, "JP"],
    [
        """あなたがそんなこと言うなんて、私はとっても嬉しい。
あなたがそんなこと言うなんて、私はとっても怒ってる。
あなたがそんなこと言うなんて、私はとっても驚いてる。
あなたがそんなこと言うなんて、私はとっても辛い。""",
        "JP",
    ],
    [  # ChatGPTに考えてもらった告白セリフ
        """私、ずっと前からあなたのことを見てきました。あなたの笑顔、優しさ、強さに、心惹かれていたんです。
友達として過ごす中で、あなたのことがだんだんと特別な存在になっていくのがわかりました。
えっと、私、あなたのことが好きです！もしよければ、私と付き合ってくれませんか？""",
        "JP",
    ],
    [  # 夏目漱石『吾輩は猫である』
        """吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。なんでも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
吾輩はここで初めて人間というものを見た。しかもあとで聞くと、それは書生という、人間中で一番獰悪な種族であったそうだ。
この書生というのは時々我々を捕まえて煮て食うという話である。""",
        "JP",
    ],
    [  # 梶井基次郎『桜の樹の下には』
        """桜の樹の下には屍体が埋まっている！これは信じていいことなんだよ。
何故って、桜の花があんなにも見事に咲くなんて信じられないことじゃないか。俺はあの美しさが信じられないので、このにさんにち不安だった。
しかしいま、やっとわかるときが来た。桜の樹の下には屍体が埋まっている。これは信じていいことだ。""",
        "JP",
    ],
    [  # ChatGPTと考えた、感情を表すセリフ
        """やったー！テストで満点取れた！私とっても嬉しいな！
どうして私の意見を無視するの？許せない！ムカつく！あんたなんか死ねばいいのに。
あはははっ！この漫画めっちゃ笑える、見てよこれ、ふふふ、あはは。
あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しい。""",
        "JP",
    ],
    [  # 上の丁寧語バージョン
        """やりました！テストで満点取れましたよ！私とっても嬉しいです！
どうして私の意見を無視するんですか？許せません！ムカつきます！あんたなんか死んでください。
あはははっ！この漫画めっちゃ笑えます、見てくださいこれ、ふふふ、あはは。
あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しいです。""",
        "JP",
    ],
    [  # ChatGPTに考えてもらった音声合成の説明文章
        """音声合成は、機械学習を活用して、テキストから人の声を再現する技術です。この技術は、言語の構造を解析し、それに基づいて音声を生成します。
この分野の最新の研究成果を使うと、より自然で表現豊かな音声の生成が可能である。深層学習の応用により、感情やアクセントを含む声質の微妙な変化も再現することが出来る。""",
        "JP",
    ],
    [
        "Speech synthesis is the artificial production of human speech. A computer system used for this purpose is called a speech synthesizer, and can be implemented in software or hardware products.",
        "EN",
    ],
    ["语音合成是人工制造人类语音。用于此目的的计算机系统称为语音合成器，可以通过软件或硬件产品实现。", "ZH"],
]

initial_md = """
# Style-Bert-VITS2 音声合成

注意: 初期からある[jvnvのモデル](https://huggingface.co/litagin/style_bert_vits2_jvnv)は、[JVNVコーパス（言語音声と非言語音声を持つ日本語感情音声コーパス）](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus)で学習されたモデルです。ライセンスは[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.ja)です。
"""

how_to_md = """
下のように`model_assets`ディレクトリの中にモデルファイルたちを置いてください。
```
model_assets
├── your_model
│   ├── config.json
│   ├── your_model_file1.safetensors
│   ├── your_model_file2.safetensors
│   ├── ...
│   └── style_vectors.npy
└── another_model
    ├── ...
```
各モデルにはファイルたちが必要です：
- `config.json`：学習時の設定ファイル
- `*.safetensors`：学習済みモデルファイル（1つ以上が必要、複数可）
- `style_vectors.npy`：スタイルベクトルファイル

上2つは`Train.bat`による学習で自動的に正しい位置に保存されます。`style_vectors.npy`は`Style.bat`を実行して指示に従って生成してください。

TODO: 現在のところはspeaker_id = 0に固定しており複数話者の合成には対応していません。
"""

style_md = f"""
- プリセットまたは音声ファイルから読み上げの声音・感情・スタイルのようなものを制御できます。
- デフォルトの{DEFAULT_STYLE}でも、十分に読み上げる文に応じた感情で感情豊かに読み上げられます。このスタイル制御は、それを重み付きで上書きするような感じです。
- 強さを大きくしすぎると発音が変になったり声にならなかったりと崩壊することがあります。
- どのくらいに強さがいいかはモデルやスタイルによって異なるようです。
- 音声ファイルを入力する場合は、学習データと似た声音の話者（特に同じ性別）でないとよい効果が出ないかもしれません。
"""


def make_interactive():
    return gr.update(interactive=True, value="音声合成")


def make_non_interactive():
    return gr.update(interactive=False, value="音声合成（モデルをロードしてください）")


def gr_util(item):
    if item == "プリセットから選ぶ":
        return (gr.update(visible=True), gr.Audio(visible=False, value=None))
    else:
        return (gr.update(visible=False), gr.update(visible=True))


