# grokfilter/extractor/spec_extractor.py
import pandas as pd
import re
import yaml
from pathlib import Path
from typing import List
from ..models.spec import FilterSpec, RejectionBand


def extract_from_excel(path: str) -> FilterSpec:
    """
    从用户上传的Excel或文字表格中提取滤波器指标
    支持灵活格式（目前支持文档中的经典表格）
    """
    try:
        df = pd.read_excel(path, header=None, engine='openpyxl')
    except Exception as e:
        # 如果Excel打不开，尝试纯文本模式
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = " ".join(df.astype(str).values.flatten())

    # 默认值
    cf = bw = rl = 2300.0, 60.0, 18.0
    bands: List[RejectionBand] = []

    # 提取中心频率、通带宽度、回波损耗
    if match := re.search(r'中心频率.*?(\d+\.?\d*)', text, re.I):
        cf = float(match.group(1))
    if match := re.search(r'通带.*?[±\+](\d+\.?\d*)', text, re.I):
        half_bw = float(match.group(1))
        bw = half_bw * 2
    if match := re.search(r'RL.*?(\d+\.?\d*)', text, re.I):
        rl = float(match.group(1))

    # 提取所有抑制带
    rej_pattern = r'(\d{3,4})\s*[-~～]\s*(\d{3,4}).*?(\d+)'  # 支持 ~ ～ - 等符号
    for start, stop, db_str in re.findall(rej_pattern, text):
        try:
            bands.append(RejectionBand(
                start=float(start),
                stop=float(stop),
                required_db=float(db_str)
            ))
        except:
            continue

    return FilterSpec(
        cf=cf,
        bw=bw,
        rl_db=rl,
        rejection_bands=bands or None  # 触发默认值
    )


def extract_from_yaml(path: str) -> FilterSpec:
    """
    从YAML文件提取滤波器指标 —— 黄金标准输入方式
    完全对应用户提供的YAML格式，支持缺省值优雅降级
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML文件未找到: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 提取抑制带（最稳健的方式）
    bands: List[RejectionBand] = []
    for band in data.get('rejection_bands', []):
        # 支持两种写法：min_rejection_db 或 required_db
        rej_db = band.get('min_rejection_db') or band.get('required_db')
        if rej_db is not None:
            bands.append(RejectionBand(
                start=float(band['start']),
                stop=float(band['stop']),
                required_db=float(rej_db)
            ))

    # 提取偏好（带默认值）
    prefs = data.get('preferences', {})
    preferences_dict = {
        "max_order": prefs.get('max_order', 12),
        "prefer_symmetric_zeros": prefs.get('prefer_symmetric_zeros', True),
        "allow_n_plus_one": prefs.get('allow_n_plus_one', True),
    }

    spec = FilterSpec(
        cf=float(data.get('center_frequency_mhz', 2300)),
        bw=float(data.get('bandwidth_mhz', 60)),
        rl_db=float(data.get('return_loss_db', 18)),
        il_max_db=float(data.get('max_insert_loss_db', 1.15)),
        rejection_bands=bands or None,  # 触发默认值
        preferences=preferences_dict
    )
    spec.estimate_minimum_order()
    
    return spec
    