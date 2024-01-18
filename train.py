import argparse
import json
import os
import shutil
from datetime import datetime
from multiprocessing import cpu_count

import gradio as gr
import yaml

from common.log import logger
from common.subprocess_utils import run_script_with_log, second_elem_of

logger_handler = None

# Get path settings
with open(os.path.join("configs", "paths.yml"), "r", encoding="utf-8") as f:
    path_config: dict[str, str] = yaml.safe_load(f.read())
    dataset_root = path_config["dataset_root"]
    # assets_root = path_config["assets_root"]


def get_path(model_name):
    assert model_name != "", "モデル名は空にできません"
    dataset_path = os.path.join(dataset_root, model_name)
    lbl_path = os.path.join(dataset_path, "esd.list")
    train_path = os.path.join(dataset_path, "train.list")
    val_path = os.path.join(dataset_path, "val.list")
    config_path = os.path.join(dataset_path, "config.json")
    return dataset_path, lbl_path, train_path, val_path, config_path


def initialize(model_name, batch_size, epochs, save_every_steps, bf16_run):
    global logger_handler
    dataset_path, _, train_path, val_path, config_path = get_path(model_name)

    # 前処理のログをファイルに保存する
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"preprocess_{timestamp}.log"
    if logger_handler is not None:
        logger.remove(logger_handler)
    logger_handler = logger.add(os.path.join(dataset_path, file_name))

    logger.info(
        f"Step 1: start initialization...\nmodel_name: {model_name}, batch_size: {batch_size}, epochs: {epochs}, save_every_steps: {save_every_steps}, bf16_run: {bf16_run}"
    )
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        # Use default config
        with open("configs/config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    config["model_name"] = model_name
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["train"]["batch_size"] = batch_size
    config["train"]["epochs"] = epochs
    config["train"]["bf16_run"] = bf16_run
    config["train"]["eval_interval"] = save_every_steps

    model_path = os.path.join(dataset_path, "models")
    if os.path.exists(model_path):
        logger.warning(f"Step 1: {model_path} already exists, so copy it to backup.")
        shutil.copytree(
            src=model_path,
            dst=os.path.join(dataset_path, "models_backup"),
            dirs_exist_ok=True,
        )
        shutil.rmtree(model_path)
    try:
        shutil.copytree(
            src="pretrained",
            dst=model_path,
        )
    except FileNotFoundError:
        logger.error("Step 1: `pretrained` folder not found.")
        return False, "Step 1, Error: pretrainedフォルダが見つかりません。"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    if not os.path.exists("config.yml"):
        shutil.copy(src="default_config.yml", dst="config.yml")
    # yml_data = safe_load(open("config.yml", "r", encoding="utf-8"))
    with open("config.yml", "r", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = dataset_path
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)
    logger.success("Step 1: initialization finished.")
    return True, "Step 1, Success: 初期設定が完了しました"


def resample(model_name, normalize, trim, num_processes):
    logger.info("Step 2: start resampling...")
    dataset_path, _, _, _, _ = get_path(model_name)
    in_dir = os.path.join(dataset_path, "raw")
    out_dir = os.path.join(dataset_path, "wavs")
    cmd = [
        "resample.py",
        "--in_dir",
        in_dir,
        "--out_dir",
        out_dir,
        "--num_processes",
        str(num_processes),
        "--sr",
        "44100",
    ]
    if normalize:
        cmd.append("--normalize")
    if trim:
        cmd.append("--trim")
    success, message = run_script_with_log(cmd)
    if not success:
        logger.error(f"Step 2: resampling failed.")
        return False, f"Step 2, Error: 音声ファイルの前処理に失敗しました:\n{message}"
    elif message:
        logger.warning(f"Step 2: resampling finished with stderr.")
        return True, f"Step 2, Success: 音声ファイルの前処理が完了しました:\n{message}"
    logger.success("Step 2: resampling finished.")
    return True, "Step 2, Success: 音声ファイルの前処理が完了しました"


def preprocess_text(model_name):
    logger.info("Step 3: start preprocessing text...")
    dataset_path, lbl_path, train_path, val_path, config_path = get_path(model_name)
    try:
        lines = open(lbl_path, "r", encoding="utf-8").readlines()
    except FileNotFoundError:
        logger.error(f"Step 3: {lbl_path} not found.")
        return False, f"Step 3, Error: 書き起こしファイル {lbl_path} が見つかりません。"
    with open(lbl_path, "w", encoding="utf-8") as f:
        for line in lines:
            path, spk, language, text = line.strip().split("|")
            path = os.path.join(dataset_path, "wavs", os.path.basename(path)).replace(
                "\\", "/"
            )
            f.writelines(f"{path}|{spk}|{language}|{text}\n")
    success, message = run_script_with_log(
        [
            "preprocess_text.py",
            "--config-path",
            config_path,
            "--transcription-path",
            lbl_path,
            "--train-path",
            train_path,
            "--val-path",
            val_path,
        ]
    )
    if not success:
        logger.error(f"Step 3: preprocessing text failed.")
        return False, f"Step 3, Error: 書き起こしファイルの前処理に失敗しました:\n{message}"
    elif message:
        logger.warning(f"Step 3: preprocessing text finished with stderr.")
        return True, f"Step 3, Success: 書き起こしファイルの前処理が完了しました:\n{message}"
    logger.success("Step 3: preprocessing text finished.")
    return True, "Step 3, Success: 書き起こしファイルの前処理が完了しました"


def bert_gen(model_name):
    logger.info("Step 4: start bert_gen...")
    _, _, _, _, config_path = get_path(model_name)
    success, message = run_script_with_log(
        [
            "bert_gen.py",
            "--config",
            config_path,
            # "--num_processes",  # bert_genは重いのでプロセス数いじらない
            #  str(num_processes),
        ]
    )
    if not success:
        logger.error(f"Step 4: bert_gen failed.")
        return False, f"Step 4, Error: BERT特徴ファイルの生成に失敗しました:\n{message}"
    elif message:
        logger.warning(f"Step 4: bert_gen finished with stderr.")
        return True, f"Step 4, Success: BERT特徴ファイルの生成が完了しました:\n{message}"
    logger.success("Step 4: bert_gen finished.")
    return True, "Step 4, Success: BERT特徴ファイルの生成が完了しました"


def style_gen(model_name, num_processes):
    logger.info("Step 5: start style_gen...")
    _, _, _, _, config_path = get_path(model_name)
    success, message = run_script_with_log(
        [
            "style_gen.py",
            "--config",
            config_path,
            "--num_processes",
            str(num_processes),
        ]
    )
    if not success:
        logger.error(f"Step 5: style_gen failed.")
        return False, f"Step 5, Error: スタイル特徴ファイルの生成に失敗しました:\n{message}"
    elif message:
        logger.warning(f"Step 5: style_gen finished with stderr.")
        return True, f"Step 5, Success: スタイル特徴ファイルの生成が完了しました:\n{message}"
    logger.success("Step 5: style_gen finished.")
    return True, "Step 5, Success: スタイル特徴ファイルの生成が完了しました"


def preprocess_all(
    model_name,
    batch_size,
    epochs,
    save_every_steps,
    bf16_run,
    num_processes,
    normalize,
    trim,
):
    if model_name == "":
        return False, "Error: モデル名を入力してください"
    success, message = initialize(
        model_name, batch_size, epochs, save_every_steps, bf16_run
    )
    if not success:
        return False, message
    success, message = resample(model_name, normalize, trim, num_processes)
    if not success:
        return False, message
    success, message = preprocess_text(model_name)
    if not success:
        return False, message
    success, message = bert_gen(model_name)  # bert_genは重いのでプロセス数いじらない
    if not success:
        return False, message
    success, message = style_gen(model_name, num_processes)
    if not success:
        return False, message
    logger.success("Success: All preprocess finished!")
    return True, "Success: 全ての前処理が完了しました。ターミナルを確認しておかしいところがないか確認するのをおすすめします。"


def train(model_name):
    dataset_path, _, _, _, config_path = get_path(model_name)
    # 学習再開の場合は念のためconfig.ymlの名前等を更新
    with open("config.yml", "r", encoding="utf-8") as f:
        yml_data = yaml.safe_load(f)
    yml_data["model_name"] = model_name
    yml_data["dataset_path"] = dataset_path
    with open("config.yml", "w", encoding="utf-8") as f:
        yaml.dump(yml_data, f, allow_unicode=True)
    success, message = run_script_with_log(
        ["train_ms.py", "--config", config_path, "--model", dataset_path]
    )
    if not success:
        logger.error(f"Train failed.")
        return False, f"Error: 学習に失敗しました:\n{message}"
    elif message:
        logger.warning(f"Train finished with stderr.")
        return True, f"Success: 学習が完了しました:\n{message}"
    logger.success("Train finished.")
    return True, "Success: 学習が完了しました"

