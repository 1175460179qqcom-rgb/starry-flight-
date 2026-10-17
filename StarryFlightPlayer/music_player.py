# -*- coding: utf-8 -*-
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
fly 音乐播放器 → Starry Flight
- 扫描本地音乐 / 播放 / 三种播放模式
- 自动识别歌名、艺术家，自动加载并滚动展示歌词
- 黑色背景 + 灰白色字体，简洁时尚
"""
import os
import re
import sys
import json
import random
import time
import hashlib
import datetime
import uuid
import difflib
import logging
import requests
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import sounddevice as sd
import librosa
from mutagen import File as MutagenFile
from mutagen.id3 import ID3NoHeaderError


class AudioEngine:
    """soundfile + sounddevice：数据不变，靠声卡采样率实现倍速（变调但零延迟）。
    pos 永远是 _orig 数组的样本索引；UI 时间轴一律用原曲时间 = pos/orig_sr。"""
    def __init__(self, volume=0.8):
        self._orig = None         # (n, 2) float32，原始数据
        self._orig_sr = 44100
        self._pos = 0             # 当前在 _orig 中的样本索引
        self._speed = 1.0
        self._volume = volume
        self._playing = False
        self._stream = None

    def load(self, path):
        import soundfile as sf
        self.stop()
        ext = os.path.splitext(path)[1].lower()
        data, sr = None, None
        # 先试 librosa
        try:
            import librosa
            data, sr = librosa.load(path, sr=None, mono=False)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            data = data.T
        except Exception:
            data = None
        # librosa 失败，用 pydub/ffmpeg 兜底
        if data is None:
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_file(path)
                import numpy as np
                data = np.array(seg.get_array_of_samples(), dtype='float32') / 32768.0
                if seg.channels == 2:
                    data = data.reshape(-1, 2)
                else:
                    data = data.reshape(-1, 1)
                sr = seg.frame_rate
            except Exception:
                # 最后用 soundfile
                data, sr = sf.read(path, dtype='float32', always_2d=True)
        self._orig = data.astype('float32')
        self._orig_sr = sr
        self._pos = 0

    def set_speed(self, s):
        self._speed = s
        # pos 是数据索引，不随速度变；只需重建流用新采样率
        if self._stream is not None:
            was_playing = self._playing
            self._open_stream()
            if was_playing:
                self._stream.start()

    def set_volume(self, v):
        self._volume = v

    @property
    def duration(self):
        """原曲总时长（秒）。"""
        return self._orig.shape[0] / self._orig_sr if self._orig is not None else 0.0

    @property
    def position(self):
        """当前原曲时间（秒），供进度条/歌词使用。"""
        return self._pos / self._orig_sr

    def play(self):
        if self._orig is None: return
        self._open_stream()
        self._stream.start()

    def _stream_sr(self):
        return int(self._orig_sr * self._speed)

    def _open_stream(self):
        if self._stream is not None:
            try: self._stream.stop(); self._stream.close()
            except: pass
        self._stream = sd.OutputStream(
            samplerate=self._stream_sr(), channels=2, dtype='float32',
            blocksize=512, callback=self._cb)
        self._playing = True

    def _cb(self, outdata, frames, time, status):
        if self._orig is None or not self._playing:
            outdata.fill(0); return
        # 声卡按 sr=orig_sr*speed 读，每帧对应原曲 1/orig_sr 秒
        end = self._pos + frames
        n = self._orig.shape[0]
        if end >= n:
            chunk = self._orig[self._pos:]
            outdata[:len(chunk)] = chunk * self._volume
            outdata[len(chunk):].fill(0)
            self._pos = n
            self._playing = False
        else:
            outdata[:] = self._orig[self._pos:end] * self._volume
            self._pos = end

    def pause(self):
        self._playing = False

    def unpause(self):
        self._playing = True

    def stop(self):
        self._playing = False
        if self._stream is not None:
            try: self._stream.stop(); self._stream.close()
            except: pass
            self._stream = None

    def seek(self, sec):
        """sec 是原曲时间（秒）。"""
        self._pos = int(sec * self._orig_sr)
        if self._orig is not None and self._pos > self._orig.shape[0]:
            self._pos = self._orig.shape[0]

    @property
    def at_end(self):
        return self._pos >= self._orig.shape[0] if self._orig is not None else False

    @property
    def playing(self):
        return self._playing
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QPoint, QPointF, QRectF, QPropertyAnimation, QEasingCurve, QSize, QTranslator, QLibraryInfo
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtGui import QPolygonF, QBrush, QPen, QLinearGradient, QPainterPath
from PyQt5.QtGui import QColor, QFont, QPainter, QIcon, QMovie, QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QListWidget, QListWidgetItem, QLabel, QPushButton, QSlider,
    QFileDialog, QSizePolicy, QFrame, QDialog, QDialogButtonBox, QComboBox, QMenu, QSplitter, QStyledItemDelegate, QStyle, QActionGroup, QCheckBox, QScrollArea, QGraphicsOpacityEffect, QLineEdit, QInputDialog,
    QSystemTrayIcon, QShortcut,
)
from PyQt5.QtGui import QKeySequence


THEMES = {
    "cyberpunk": {
        "name": "赛博朋克",
        "style": '''
QMainWindow, QDialog, QWidget { background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0a0a12,stop:1 #1a0a2e); color: #00ffc8; }
QListWidget { background: rgba(0,255,200,0.04); color: #c8ffe8; border: 1px solid rgba(0,255,200,0.2); border-radius: 10px; }
QListWidget::item { padding: 8px 12px; border-radius: 6px; }
QListWidget::item:hover { background: rgba(0,255,200,0.1); }
QListWidget::item:selected { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #ff006e,stop:1 #00f5d4); color: #fff; }
QSlider::groove:horizontal { height: 6px; background: rgba(0,255,200,0.15); border-radius: 3px; }
QSlider::sub-page:horizontal { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #ff006e,stop:1 #00f5d4); border-radius: 3px; }
QSlider::handle:horizontal { width: 16px; height: 16px; margin: -6px 0; background: #00f5d4; border-radius: 8px; border: 2px solid #ff006e; }
QPushButton { background: rgba(0,255,200,0.1); color: #00ffc8; border: 1px solid rgba(0,255,200,0.3); padding: 6px 14px; border-radius: 8px; }
QPushButton:hover { background: rgba(0,255,200,0.2); }
QPushButton:pressed { background: rgba(0,255,200,0.3); }
QLabel { color: #c8ffe8; }
QComboBox { background: rgba(0,255,200,0.1); color: #00ffc8; border: 1px solid rgba(0,255,200,0.3); padding: 6px; border-radius: 8px; }
QComboBox QAbstractItemView { background: #0a0a12; color: #00ffc8; selection-background-color: #ff006e; }
QMenu { background: #0a0a12; color: #00ffc8; border: 1px solid rgba(0,255,200,0.3); border-radius: 10px; padding: 6px; }
QMenu::item { padding: 8px 24px; border-radius: 6px; }
QMenu::item:selected { background: #ff006e; color: #fff; }
''',
        "accent": "#00f5d4", "list_bg": "rgba(0,255,200,0.04)",
        "lyric_bg": "qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #0a0a12,stop:1 #1a0a2e)",
        "trans_panel": "background: rgba(15,10,32,0.55); border: 1px solid rgba(0,245,212,0.30);",
    },
    "forest": {
        "name": "森林墨绿",
        "style": '''
QMainWindow, QDialog, QWidget { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #0f2027,stop:0.5 #203a43,stop:1 #2c5364); color: #d4f5e9; }
QListWidget { background: rgba(255,255,255,0.05); color: #b8e6d4; border: none; border-radius: 10px; }
QListWidget::item { padding: 8px 12px; border-radius: 6px; }
QListWidget::item:hover { background: rgba(255,255,255,0.08); }
QListWidget::item:selected { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #11998e,stop:1 #38ef7d); color: #fff; }
QSlider::groove:horizontal { height: 6px; background: rgba(255,255,255,0.12); border-radius: 3px; }
QSlider::sub-page:horizontal { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #11998e,stop:1 #38ef7d); border-radius: 3px; }
QSlider::handle:horizontal { width: 16px; height: 16px; margin: -6px 0; background: #fff; border-radius: 8px; border: 2px solid #11998e; }
QPushButton { background: rgba(255,255,255,0.08); color: #d4f5e9; border: none; padding: 6px 14px; border-radius: 8px; }
QPushButton:hover { background: rgba(255,255,255,0.16); }
QPushButton:pressed { background: rgba(255,255,255,0.22); }
QLabel { color: #b8e6d4; }
QComboBox { background: rgba(255,255,255,0.08); color: #d4f5e9; border: 1px solid rgba(255,255,255,0.12); padding: 6px; border-radius: 8px; }
QComboBox QAbstractItemView { background: #0f2027; color: #d4f5e9; selection-background-color: #11998e; }
QMenu { background: #0f2027; color: #d4f5e9; border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 6px; }
QMenu::item { padding: 8px 24px; border-radius: 6px; }
QMenu::item:selected { background: #11998e; }
''',
        "accent": "#11998e", "list_bg": "rgba(255,255,255,0.05)",
        "lyric_bg": "qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #0f2027,stop:0.5 #203a43,stop:1 #2c5364)",
        "trans_panel": "background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12);",
    },
}

def get_style(theme_key, corner=None):
    """corner: 界面圆滑度（px，基准 10）。None 时用默认基准 10。
    滑块槽/手柄等小控件半径固定，避免 radius 超过控件半宽导致圆角失效变方角。"""
    if corner is None:
        corner = 10
    s = THEMES.get(theme_key, THEMES["cyberpunk"])["style"]
    ratio = corner / 10.0
    out = []
    for line in s.splitlines():
        if "QSlider::" in line:
            out.append(line)   # 滑块等小控件半径固定
            continue
        out.append(re.sub(r"border-radius:\s*(\d+)px",
                          lambda m: f"border-radius: {max(0, min(round(int(m.group(1)) * ratio), 16))}px",
                          line))
    return "\n".join(out)

STYLE = get_style("cyberpunk")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
# 日志配置
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "player.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("player")


# =========================================================
# 界面多语言
# =========================================================
TEXTS = {
    "zh": {
        "app_title": "Starry Flight 音乐播放器",
        "brand": "♪  Starry Flight 音乐",
        "settings": "外观设置",
        "open_dir": "打开目录",
        "not_playing": "未在播放",
        "unknown_artist": "未知艺术家",
        "count_fmt": "共 {n} 首",
        "mode_order": "顺序播放",
        "mode_loop": "单曲循环",
        "mode_random": "随机播放",
        "no_lyric": "暂无歌词",
        "list_font_size": "歌曲列表字号",
        "lyric_font_size": "歌词字号",
        "lyric_line_spacing": "歌词行距",
        "corner_radius": "界面圆滑度",
        "lyric_translate": "歌词翻译",
        "trans_need_song": "请先播放歌曲，加载歌词后即可翻译",
        "trans_loading": "正在翻译歌词…",
        "trans_failed": "翻译失败：请检查网络或稍后重试",
        "lyric_color": "歌词高亮颜色",
        "lyric_font": "歌词字体",
        "language": "界面语言",
        "toggle_list": "显示/隐藏歌曲列表",
        "notice": "公告",
        "notice_empty": "暂无公告",
        "save": "保存",
        "cancel": "取消",
        "new_playlist": "新建歌单",
        "playlist_name": "歌单名称：",
        "menu_settings": "外观设置",
        "menu_open": "打开文件夹",
        "menu_toggle": "显示/隐藏歌曲列表",
        "menu_minimize": "最小化到托盘",
        "menu_quit": "退出应用",
        "menu_about": "关于 Starry Flight",
        "menu_privacy": "隐私政策",
        "menu_update": "检查更新",
        "menu_desktop_lrc": "桌面歌词",
        "menu_eq": "音频",
        "menu_spectrum": "频谱",
        "menu_history": "播放历史",
        "menu_sleep": "睡眠定时器",
        "menu_hotkey": "快捷键设置",
        "menu_check_update": "已是最新版本",
        "menu_audio": "音频",
        "menu_eq_settings": "均衡器",
        "menu_hotkey_settings": "快捷键设置",
        "menu_dlrc": "桌面歌词",
        "menu_dlrc_color": "桌面歌词颜色",
        "menu_dlrc_size": "桌面歌词字号",
        "menu_dlrc_font": "桌面歌词字体",
        "menu_spectrum": "频谱",
        "menu_spectrum_color": "频谱颜色",
        "menu_spectrum_block": "方块分段样式",
        "menu_notify": "桌面通知",
        "menu_global_hotkey": "全局快捷键",
        "menu_preload": "预加载下一首",
        "menu_search": "歌曲搜索框",
        "menu_sleep_timer": "睡眠定时器",
        "menu_history": "播放历史",
        "menu_off": "关闭",
        "menu_minutes_later": "分钟后停止",
        "menu_fullscreen": "全屏",
        "menu_optimize": "优化",
        "menu_import_skin": "导入皮肤(QSS)",
        "set_theme": "UI 主题",
        "set_lyric_bg_color": "歌词背景颜色",
        "set_lyric_bg": "歌词背景",
        "set_bg_opacity": "背景图透明度",
        "set_list_color": "歌曲列表颜色",
        "set_list_font": "歌曲列表字体",
        "set_lyric_offset": "歌词时间偏移(秒)",
        "set_list_size": "歌曲列表字号",
        "set_lyric_size": "歌词字号",
        "set_lyric_spacing": "歌词行距",
        "set_corner": "UI 圆角",
        "set_lyric_color": "歌词高亮颜色",
        "set_lyric_font": "歌词字体",
        "set_language": "界面语言",
        "set_volume": "音量",
        "set_not_playing": "未在播放",
        "set_songs_count": "共 0 首",
        "set_pick_image": "选择图片",
        "set_pick_video": "选择视频 (Pro)",
        "set_clear_bg": "清除背景",
        "set_fit_scale": "等比例缩放",
        "neon": "霓虹",
        "rainbow": "彩虹",
        "default_theme": "默认主题",
        "collapse_list": "收起歌曲栏",
        "dlrc_c0": "青绿色",
        "dlrc_c1": "红色",
        "dlrc_c2": "黄色",
        "dlrc_c3": "绿色",
        "dlrc_c4": "蓝色",
        "dlrc_c5": "橙色",
        "dlrc_c6": "紫色",
        "dlrc_c7": "白色(彩虹)",
        "dlrc_c8": "霓虹渐变",
        "sp_c0": "青绿色",
        "sp_c1": "霓虹渐变",
        "sp_c2": "彩虹",
        "sub_price": "¥5.00 / 月",
        "about_version": "版本 1.0.0",
        "about_desc": "本地音乐播放器，支持歌词、频谱、云账户。",
        "privacy_agree": "我已阅读并同意以上条款",
        "privacy_ok": "同意并继续",
        "privacy_cancel": "不同意并退出",
        "up_to_date": "当前已是最新版本 v1.0.0",
        "sub_life": "终身",
        "privacy_text": "<h3>隐私政策与用户协议</h3><p>欢迎使用 Starry Flight。</p><p><b>1. 数据收集</b></p><p>仅收集账户邮箱、昵称、头像。音乐文件和播放记录仅保存在本地。</p><p><b>2. 数据存储</b></p><p>账户数据通过 Supabase 云端存储，不出售给第三方。</p><p><b>3. 本地文件</b></p><p>仅播放本地音乐，不上传音频。</p><p><b>4. 歌词</b></p><p>歌词优先读取本地 .lrc 文件。联网歌词功能默认关闭，开启后仅向网易云/QQ音乐等接口请求歌词文本用于显示，不上传音乐文件。</p><p><b>5. 联系</b></p><p>support@starryflight.app</p>",
        "mode_order": "顺序播放",
        "mode_loop": "单曲循环",
        "mode_random": "随机播放",
        "save": "保存",
        "logout": "退出登录",
        "forgot_pwd": "忘记密码？",
        "send_code": "发送验证码",
        "fav": "★ 收藏",
        "unfav": "取消收藏",
        "tray_play": "播放/暂停",
        "tray_next": "下一曲",
        "tray_prev": "上一曲",
        "tray_quit": "退出",
        "tray_min": "最小化",
        "fullscreen": "全屏",
        "exit_fullscreen": "退出全屏",
        "profile_title": "个人主页",
        "pick_avatar": "选择头像",
        "account": "账号",
        "not_logged_in": "未登录",
        "set_nickname": "设置昵称",
        "fav_count": "收藏歌曲：{n} 首",
        "playlist_count": "歌单数：{n} 个",
        "change_pwd": "修改密码",
        "old_pwd": "旧密码：",
        "new_pwd": "新密码(至少6位)：",
        "confirm_pwd": "确认新密码：",
        "warn": "提示",
        "success": "成功",
        "fail": "失败",
        "pwd_mismatch": "两次新密码不一致",
        "logout": "退出登录",
        "close": "关闭",
        "login_reg": "登录 / 注册",
        "email": "邮箱：",
        "password": "密码：",
        "forgot_pwd": "忘记密码？",
        "set_pwd": "设置密码(至少6位)",
        "confirm_pwd2": "确认密码",
        "verify_code": "验证码",
        "send_code": "发送验证码",
        "login_tab": "登录",
        "reg_tab": "注册",
        "ok": "确定",
        "reset_pwd": "重置密码",
        "reg_email": "注册邮箱",
        "new_pwd2": "新密码(至少6位)",
        "confirm_new_pwd": "确认新密码",
        "about_title": "关于 Starry Flight",
        "privacy_title": "隐私政策",
        "checking_update": "正在检查更新...",
        "up_to_date": "已是最新版本",
        "no_update": "暂无更新",
        "sub_price": "¥5.00 / 月",
        "hotkey_tip": "点击按钮后，按下想要的按键即可设置",
        "hotkey_capture": "按任意键...",
        "show_main": "显示主窗口",
        "agree_terms": "我已阅读并同意以上条款",
        "unlock_features": "解锁全部功能：",
        "tip_press_key": "点击按钮后，按下想要的按键即可设置",
        "click_avatar": "点击头像更换图片",
        "nickname": "昵称：",
        "email": "邮箱：",
        "password": "密码：",
        "confirm_pwd": "确认密码：",
        "reg_email": "注册邮箱：",
        "verify_code": "验证码：",
        "new_password": "新密码：",
        "theme_cyberpunk": "赛博朋克",
        "theme_forest": "森林墨绿",
        "all_songs": "全部歌曲",
        "favorites": "★ 收藏",
        "error": "错误",
        "new_playlist_fail": "新建歌单失败：{err}",
        "history_title": "播放历史",
        "eq_title": "均衡器",
        "hotkey_title": "快捷键设置",
        "hotkey_play": "播放/暂停",
        "hotkey_next": "下一首",
        "hotkey_prev": "上一首",
        "key_space": "空格",
        "key_right": "右方向键",
        "key_left": "左方向键",
        "key_up": "上方向键",
        "key_down": "下方向键",
        "key_enter": "回车",
        "cancel_fav": "取消收藏",
        "add_to_playlist": "加入歌单",
        "no_custom_pl": "（暂无自定义歌单）",
        "new_pl_action": "＋ 新建歌单…",
        "skip_msg": "跳过: {name}",
        "select_music_dir": "选择音乐目录",
        "select_bg_image": "选择背景图片",
        "select_bg_video": "选择背景视频",
        "select_avatar": "选择头像",
        "select_skin": "选择皮肤QSS文件",
        "skin_too_large": "皮肤文件过大（上限 1MB）",
        "skin_utf8_required": "皮肤文件必须为 UTF-8 编码",
        "skin_load_fail": "加载皮肤失败: {err}",
        "nick_placeholder": "设置昵称",
        "pl_count": "歌单数：{n} 个",
        "search_hint": "搜索歌曲...",
        "new_playlist_tip": "新建歌单",
        "expand_list": "展开歌曲栏",
        "profile_click_avatar": "点击头像更换图片",
        "nickname_label": "昵称：",
        "spacing_val": "{v:.1f} 倍",
        "press_any_key": "按任意键...",
        "tray_minimized": "程序已缩到托盘，双击图标恢复",
        "dlrc_playing": "♪ 播放中...",
        "notice_default": "欢迎使用 Starry Flight！公告内容可在 config.json 的 announcements 中修改。",
    },
    "en": {
        "app_title": "Starry Flight Music Player",
        "brand": "♪  Starry Flight Music",
        "settings": "Settings",
        "open_dir": "Open Folder",
        "not_playing": "Not playing",
        "unknown_artist": "Unknown Artist",
        "count_fmt": "{n} songs",
        "mode_order": "Order",
        "mode_loop": "Loop One",
        "mode_random": "Shuffle",
        "no_lyric": "No lyrics",
        "lang_orig": "Original",
        "lang_trans": "Translation",
        "lang_both": "Bilingual",
        "list_font_size": "List Font Size",
        "lyric_font_size": "Lyrics Font Size",
        "lyric_line_spacing": "Lyrics Line Spacing",
        "corner_radius": "UI Corner Radius",
        "lyric_translate": "Lyrics Translate",
        "trans_need_song": "Play a song first, then translate lyrics",
        "trans_loading": "Translating lyrics…",
        "trans_failed": "Translation failed: check network or retry later",
        "lyric_color": "Highlight Color",
        "lyric_font": "Lyrics Font",
        "language": "Language",
        "toggle_list": "Show/Hide Playlist",
        "notice": "Notices",
        "notice_empty": "No notices",
        "save": "Save",
        "cancel": "Cancel",
        "new_playlist": "New Playlist",
        "playlist_name": "Playlist name:",
        "menu_settings": "Appearance",
        "menu_open": "Open Folder",
        "menu_toggle": "Show/Hide Playlist",
        "menu_minimize": "Minimize to Tray",
        "menu_quit": "Quit",
        "menu_about": "About Starry Flight",
        "menu_privacy": "Privacy Policy",
        "menu_update": "Check for Updates",
        "menu_desktop_lrc": "Desktop Lyrics",
        "menu_eq": "Audio",
        "menu_spectrum": "Spectrum",
        "menu_history": "History",
        "menu_sleep": "Sleep Timer",
        "menu_hotkey": "Hotkeys",
        "menu_check_update": "Up to date",
        "dlrc_c0": "Teal",
        "dlrc_c1": "Red",
        "dlrc_c2": "Yellow",
        "dlrc_c3": "Green",
        "dlrc_c4": "Blue",
        "dlrc_c5": "Orange",
        "dlrc_c6": "Purple",
        "dlrc_c7": "White (Rainbow)",
        "dlrc_c8": "Neon",
        "sp_c0": "Teal",
        "sp_c1": "Neon",
        "sp_c2": "Rainbow",
        "sub_price": "¥5.00 / month",
        "about_version": "Version 1.0.0",
        "about_desc": "Local music player with lyrics, spectrum, cloud account.",
        "privacy_agree": "I have read and agree to the terms",
        "privacy_ok": "Agree and Continue",
        "privacy_cancel": "Disagree and Exit",
        "up_to_date": "You are on the latest version v1.0.0",
        "sub_life": "Lifetime",
        "privacy_text": "<h3>Privacy Policy</h3><p>Welcome to Starry Flight.</p><p><b>1. Data Collection</b></p><p>Only collects account email, nickname, avatar. Music and history stay local.</p><p><b>2. Data Storage</b></p><p>Account data stored via Supabase cloud, not sold to third parties.</p><p><b>3. Local Files</b></p><p>Only plays local music, no audio upload.</p><p><b>4. Lyrics</b></p><p>Local .lrc files are preferred. Online lyrics are off by default; when enabled, only lyric text is fetched from services like NetEase/QQ Music for display, no music files are uploaded.</p><p><b>5. Contact</b></p><p>support@starryflight.app</p>",
        "menu_eq_settings": "Equalizer",
        "menu_hotkey_settings": "Hotkey Settings",
        "menu_dlrc": "Desktop Lyrics",
        "menu_dlrc_color": "Lyrics Color",
        "menu_dlrc_size": "Lyrics Size",
        "menu_dlrc_font": "Lyrics Font",
        "menu_spectrum_block": "Block Style",
        "menu_notify": "Desktop Notifications",
        "menu_global_hotkey": "Global Hotkeys",
        "menu_preload": "Preload Next",
        "menu_search": "Search Box",
        "menu_sleep_timer": "Sleep Timer",
        "menu_off": "Off",
        "menu_minutes_later": "min later",
        "menu_fullscreen": "Fullscreen",
        "menu_import_skin": "Import Skin (QSS)",
        "theme_cyberpunk": "Cyberpunk",
        "theme_forest": "Forest Green",
        "all_songs": "All Songs",
        "favorites": "★ Favorites",
        "error": "Error",
        "new_playlist_fail": "Failed to create playlist: {err}",
        "history_title": "Play History",
        "eq_title": "Equalizer",
        "hotkey_title": "Hotkey Settings",
        "hotkey_tip": "Click a button, then press the desired key",
        "hotkey_play": "Play/Pause",
        "hotkey_next": "Next",
        "hotkey_prev": "Previous",
        "key_space": "Space",
        "key_right": "Right Arrow",
        "key_left": "Left Arrow",
        "key_up": "Up Arrow",
        "key_down": "Down Arrow",
        "key_enter": "Enter",
        "cancel_fav": "Remove Favorite",
        "add_to_playlist": "Add to Playlist",
        "no_custom_pl": "（No custom playlists）",
        "new_pl_action": "＋ New Playlist…",
        "skip_msg": "Skipped: {name}",
        "select_music_dir": "Select Music Directory",
        "select_bg_image": "Select Background Image",
        "select_bg_video": "Select Background Video",
        "select_avatar": "Select Avatar",
        "select_skin": "Select Skin QSS File",
        "skin_too_large": "Skin file too large (max 1MB)",
        "skin_utf8_required": "Skin file must be UTF-8 encoded",
        "skin_load_fail": "Failed to load skin: {err}",
        "nick_placeholder": "Set Nickname",
        "fav_count": "Favorite Songs: {n}",
        "pl_count": "Playlists: {n}",
        "search_hint": "Search songs...",
        "new_playlist_tip": "New Playlist",
        "expand_list": "Expand Song List",
        "close": "Close",
        "profile_title": "Profile",
        "profile_click_avatar": "Click avatar to change image",
        "nickname_label": "Nickname: ",
        "spacing_val": "{v:.1f}x",
        "press_any_key": "Press any key...",
        "tray_minimized": "Minimized to tray, double-click to restore",
        "dlrc_playing": "♪ Playing...",
        "notice_default": "Welcome to Starry Flight! Announcements can be edited in the announcements field of config.json.",
    },
    "ja": {
        "app_title": "Starry Flight 音楽プレーヤー",
        "brand": "♪  Starry Flight 音楽",
        "settings": "設定",
        "open_dir": "フォルダを開く",
        "not_playing": "再生していません",
        "unknown_artist": "アーティスト不明",
        "count_fmt": "{n} 曲",
        "mode_order": "順番に再生",
        "mode_loop": "1曲リピート",
        "mode_random": "シャッフル",
        "no_lyric": "歌詞がありません",
        "lang_orig": "原文",
        "lang_trans": "訳文",
        "lang_both": "対訳",
        "list_font_size": "リスト文字サイズ",
        "lyric_font_size": "歌詞文字サイズ",
        "lyric_line_spacing": "歌詞行間",
        "corner_radius": "UI 角丸サイズ",
        "lyric_translate": "歌詞翻訳",
        "trans_need_song": "曲を再生して歌詞を読み込んでから翻訳してください",
        "trans_loading": "歌詞を翻訳中…",
        "trans_failed": "翻訳失敗：ネットワークを確認するか、後でもう一度お試しください",
        "lyric_color": "ハイライト色",
        "lyric_font": "歌詞フォント",
        "language": "表示言語",
        "toggle_list": "プレイリスト表示/非表示",
        "notice": "お知らせ",
        "notice_empty": "お知らせはありません",
        "save": "保存",
        "cancel": "キャンセル",
        "new_playlist": "新規プレイリスト",
        "playlist_name": "プレイリスト名：",
        "menu_minimize": "トレイに最小化",
        "menu_quit": "アプリを終了",
        "menu_about": "Starry Flight について",
        "menu_privacy": "プライバシーポリシー",
        "menu_update": "更新を確認",
        "menu_desktop_lrc": "デスクトップ歌詞",
        "menu_audio": "オーディオ",
        "menu_eq_settings": "イコライザー",
        "menu_hotkey_settings": "ショートカット設定",
        "menu_dlrc_color": "歌詞カラー",
        "menu_dlrc_size": "歌詞文字サイズ",
        "menu_dlrc_font": "歌詞フォント",
        "menu_spectrum": "スペクトラム",
        "menu_spectrum_block": "ブロックスタイル",
        "menu_notify": "デスクトップ通知",
        "menu_global_hotkey": "グローバルショートカット",
        "menu_preload": "次曲を事前読み込み",
        "menu_search": "曲検索ボックス",
        "menu_sleep_timer": "スリープタイマー",
        "menu_history": "再生履歴",
        "menu_off": "オフ",
        "menu_minutes_later": "分後に停止",
        "menu_fullscreen": "全画面",
        "menu_optimize": "最適化",
        "menu_import_skin": "スキン読込(QSS)",
        "set_volume": "音量",
        "set_not_playing": "再生していません",
        "set_songs_count": "全 {n} 曲",
        "show_main": "メインウィンドウを表示",
        "tray_play": "再生/一時停止",
        "tray_next": "次の曲",
        "tray_prev": "前の曲",
        "tray_quit": "終了",
        "exit_fullscreen": "全画面を終了",
        "menu_eq_settings": "イコライザー",
        "menu_hotkey_settings": "ショートカット設定",
        "menu_dlrc": "デスクトップ歌詞",
        "menu_dlrc_color": "歌詞カラー",
        "menu_dlrc_size": "歌詞文字サイズ",
        "menu_dlrc_font": "歌詞フォント",
        "dlrc_c0": "ティール",
        "dlrc_c1": "赤",
        "dlrc_c2": "黄",
        "dlrc_c3": "緑",
        "dlrc_c4": "青",
        "dlrc_c5": "オレンジ",
        "dlrc_c6": "紫",
        "dlrc_c7": "白(レインボー)",
        "dlrc_c8": "ネオン",
        "sp_c0": "ティール",
        "sp_c1": "ネオン",
        "sp_c2": "レインボー",
        "sub_price": "¥5.00 / 月",
        "about_version": "バージョン 1.0.0",
        "about_desc": "ローカル音楽プレーヤー。歌詞・スペクトラム・クラウドアカウント対応。",
        "privacy_agree": "上記条項を読み同意します",
        "privacy_ok": "同意して続行",
        "privacy_cancel": "同意せず終了",
        "up_to_date": "最新バージョン v1.0.0 です",
        "privacy_text": "<h3>プライバシーポリシー</h3><p>Starry Flight へようこそ。</p><p><b>1. 収集データ</b></p><p>アカウントメール、ニックネーム、アバターのみ収集。音楽と履歴はローカル保存。</p><p><b>2. 保存</b></p><p>アカウントデータは Supabase クラウドに保存。第三者に販売しません。</p><p><b>3. ローカルファイル</b></p><p>ローカル音楽のみ再生、音声はアップロードしません。</p><p><b>4. 歌詞</b></p><p>歌詞はローカル .lrc ファイルを優先。オンライン歌詞は初期状態でオフ。オンにすると表示用の歌詞テキストのみを取得し、音声ファイルはアップロードしません。</p><p><b>5. 連絡先</b></p><p>support@starryflight.app</p>",
        "theme_cyberpunk": "サイバーパンク",
        "theme_forest": "フォレストグリーン",
        "all_songs": "すべての曲",
        "favorites": "★ お気に入り",
        "error": "エラー",
        "new_playlist_fail": "プレイリスト作成失敗: {err}",
        "history_title": "再生履歴",
        "eq_title": "イコライザー",
        "hotkey_title": "ショートカット設定",
        "hotkey_tip": "ボタンをクリックしてからキーを押してください",
        "hotkey_play": "再生/一時停止",
        "hotkey_next": "次の曲",
        "hotkey_prev": "前の曲",
        "key_space": "スペース",
        "key_right": "右矢印",
        "key_left": "左矢印",
        "key_up": "上矢印",
        "key_down": "下矢印",
        "key_enter": "Enter",
        "cancel_fav": "お気に入り解除",
        "add_to_playlist": "プレイリストに追加",
        "no_custom_pl": "（カスタムプレイリストはありません）",
        "new_pl_action": "＋ 新規プレイリスト…",
        "skip_msg": "スキップ: {name}",
        "select_music_dir": "音楽フォルダを選択",
        "select_bg_image": "背景画像を選択",
        "select_bg_video": "背景動画を選択",
        "select_avatar": "アバターを選択",
        "select_skin": "スキンQSSファイルを選択",
        "skin_too_large": "スキンファイルが大きすぎます（最大1MB）",
        "skin_utf8_required": "スキンファイルはUTF-8でなければなりません",
        "skin_load_fail": "スキンの読み込み失敗: {err}",
        "nick_placeholder": "ニックネームを設定",
        "fav_count": "お気に入り曲：{n} 曲",
        "pl_count": "プレイリスト数：{n}",
        "search_hint": "曲を検索...",
        "new_playlist_tip": "新規プレイリスト",
        "expand_list": "曲リストを展開",
        "close": "閉じる",
        "profile_title": "プロフィール",
        "profile_click_avatar": "アバターをクリックして画像を変更",
        "nickname_label": "ニックネーム：",
        "spacing_val": "{v:.1f} 倍",
        "press_any_key": "任意のキーを押してください...",
        "tray_minimized": "トレイに最小化しました。ダブルクリックで復元",
        "dlrc_playing": "♪ 再生中...",
        "notice_default": "Starry Flight へようこそ！お知らせは config.json の announcements で変更できます。",
    },
    "fra": {
        "app_title": "Starry Flight Lecteur de musique",
        "brand": "♪  Starry Flight Musique",
        "settings": "Réglages d'apparence",
        "open_dir": "Ouvrir le dossier",
        "not_playing": "Aucune lecture",
        "unknown_artist": "Artiste inconnu",
        "count_fmt": "{n} chansons",
        "mode_order": "Lecture séquentielle",
        "mode_loop": "Répéter une",
        "mode_random": "Aléatoire",
        "no_lyric": "Pas de paroles",
        "lang_orig": "Original",
        "lang_trans": "Traduction",
        "lang_both": "Bilingue",
        "list_font_size": "Taille police liste",
        "lyric_font_size": "Taille police paroles",
        "lyric_line_spacing": "Interligne paroles",
        "corner_radius": "Arrondi de l'interface",
        "lyric_translate": "Traduire les paroles",
        "trans_need_song": "Lancez d'abord une chanson, chargez les paroles puis traduisez",
        "trans_loading": "Traduction des paroles…",
        "trans_failed": "Échec de la traduction : vérifiez le réseau ou réessayez plus tard",
        "lyric_color": "Couleur surlignée",
        "lyric_font": "Police des paroles",
        "language": "Langue de l'interface",
        "toggle_list": "Afficher/Masquer la liste",
        "notice": "Annonces",
        "notice_empty": "Aucune annonce",
        "save": "Enregistrer",
        "cancel": "Annuler",
        "new_playlist": "Nouvelle liste",
        "playlist_name": "Nom de la liste :",
        "menu_minimize": "Réduire dans la barre des tâches",
        "menu_quit": "Quitter",
        "menu_about": "À propos de Starry Flight",
        "menu_privacy": "Politique de confidentialité",
        "menu_update": "Vérifier les mises à jour",
        "menu_desktop_lrc": "Paroles bureau",
        "menu_audio": "Audio",
        "menu_eq_settings": "Égaliseur",
        "menu_hotkey_settings": "Raccourcis",
        "menu_spectrum": "Spectre",
        "menu_history": "Historique",
        "menu_sleep_timer": "Minuteur de sommeil",
        "menu_fullscreen": "Plein écran",
        "menu_optimize": "Optimisation",
        "set_volume": "Volume",
        "set_not_playing": "Aucune lecture",
        "set_songs_count": "{n} chansons",
        "show_main": "Afficher la fenêtre",
        "tray_play": "Lecture/Pause",
        "tray_next": "Suivant",
        "tray_prev": "Précédent",
        "tray_quit": "Quitter",
        "exit_fullscreen": "Quitter plein écran",
        "menu_eq_settings": "Égaliseur",
        "menu_hotkey_settings": "Raccourcis",
        "menu_dlrc": "Paroles bureau",
        "menu_dlrc_color": "Couleur des paroles",
        "menu_dlrc_size": "Taille des paroles",
        "menu_dlrc_font": "Police des paroles",
        "dlrc_c0": "Turquoise",
        "dlrc_c1": "Rouge",
        "dlrc_c2": "Jaune",
        "dlrc_c3": "Vert",
        "dlrc_c4": "Bleu",
        "dlrc_c5": "Orange",
        "dlrc_c6": "Violet",
        "dlrc_c7": "Blanc (Arc-en-ciel)",
        "dlrc_c8": "Néon",
        "sp_c0": "Turquoise",
        "sp_c1": "Néon",
        "sp_c2": "Arc-en-ciel",
        "sub_price": "¥5.00 / mois",
        "about_version": "Version 1.0.0",
        "about_desc": "Lecteur de musique local avec paroles, spectre, compte cloud.",
        "privacy_agree": "J'ai lu et j'accepte les conditions",
        "privacy_ok": "Accepter et continuer",
        "privacy_cancel": "Refuser et quitter",
        "up_to_date": "Vous avez la dernière version v1.0.0",
        "privacy_text": "<h3>Politique de confidentialité</h3><p>Bienvenue sur Starry Flight.</p><p><b>1. Données collectées</b></p><p>Uniquement email, pseudo, avatar. Musique et historique restent locaux.</p><p><b>2. Stockage</b></p><p>Données via Supabase cloud, non revendues.</p><p><b>3. Fichiers locaux</b></p><p>Lecture locale uniquement, pas d'upload audio.</p><p><b>4. Paroles</b></p><p>Fichiers .lrc locaux privilégiés. Paroles en ligne désactivées par défaut ; si activées, seul le texte des paroles est récupéré pour l'affichage, aucun fichier audio n'est envoyé.</p><p><b>5. Contact</b></p><p>support@starryflight.app</p>",
        "theme_cyberpunk": "Cyberpunk",
        "theme_forest": "Vert forêt",
        "all_songs": "Toutes les chansons",
        "favorites": "★ Favoris",
        "error": "Erreur",
        "new_playlist_fail": "Échec de création du playlist: {err}",
        "history_title": "Historique de lecture",
        "eq_title": "Égaliseur",
        "hotkey_title": "Paramètres de raccourcis",
        "hotkey_tip": "Cliquez sur un bouton puis appuyez sur la touche souhaitée",
        "hotkey_play": "Lecture/Pause",
        "hotkey_next": "Suivant",
        "hotkey_prev": "Précédent",
        "key_space": "Espace",
        "key_right": "Flèche droite",
        "key_left": "Flèche gauche",
        "key_up": "Flèche haute",
        "key_down": "Flèche basse",
        "key_enter": "Entrée",
        "cancel_fav": "Retirer des favoris",
        "add_to_playlist": "Ajouter à la playlist",
        "no_custom_pl": "（Aucune playlist personnalisée）",
        "new_pl_action": "＋ Nouvelle playlist…",
        "skip_msg": "Skippé: {name}",
        "select_music_dir": "Choisir le dossier musical",
        "select_bg_image": "Choisir une image d'arrière-plan",
        "select_bg_video": "Choisir une vidéo d'arrière-plan",
        "select_avatar": "Choisir un avatar",
        "select_skin": "Choisir un fichier skin QSS",
        "skin_too_large": "Fichier skin trop volumineux (max 1 Mo)",
        "skin_utf8_required": "Le fichier skin doit être encodé en UTF-8",
        "skin_load_fail": "Échec du chargement du skin: {err}",
        "nick_placeholder": "Définir le pseudo",
        "fav_count": "Chansons favorites : {n}",
        "pl_count": "Playlists : {n}",
        "search_hint": "Rechercher des chansons...",
        "new_playlist_tip": "Nouvelle playlist",
        "expand_list": "Déplier la liste de chansons",
        "close": "Fermer",
        "profile_title": "Profil",
        "profile_click_avatar": "Cliquez sur l'avatar pour changer d'image",
        "nickname_label": "Pseudo : ",
        "spacing_val": "{v:.1f}x",
        "press_any_key": "Appuyez sur n'importe quelle touche...",
        "tray_minimized": "Minimisé dans la barre, double-cliquez pour restaurer",
        "dlrc_playing": "♪ Lecture en cours...",
        "notice_default": "Bienvenue sur Starry Flight ! Les annonces peuvent être modifiées dans le champ announcements de config.json.",
    },
    "de": {
        "app_title": "Starry Flight Musikplayer",
        "brand": "♪  Starry Flight Musik",
        "settings": "Darstellung",
        "open_dir": "Ordner öffnen",
        "not_playing": "Keine Wiedergabe",
        "unknown_artist": "Unbekannter Künstler",
        "count_fmt": "{n} Songs",
        "mode_order": "Reihenfolge",
        "mode_loop": "Eins wiederholen",
        "mode_random": "Zufall",
        "no_lyric": "Keine Lyrics",
        "lang_orig": "Original",
        "lang_trans": "Übersetzung",
        "lang_both": "Zweisprachig",
        "list_font_size": "Listengröße",
        "lyric_font_size": "Lyrics-Größe",
        "lyric_line_spacing": "Lyrics-Zeilenabstand",
        "corner_radius": "UI-Rundung",
        "lyric_translate": "Lyrics übersetzen",
        "trans_need_song": "Erst einen Song abspielen, Lyrics laden und dann übersetzen",
        "trans_loading": "Lyrics werden übersetzt…",
        "trans_failed": "Übersetzung fehlgeschlagen: Netzwerk prüfen oder später erneut versuchen",
        "lyric_color": "Hervorhebungsfarbe",
        "lyric_font": "Lyrics-Schrift",
        "language": "Sprache",
        "toggle_list": "Liste anzeigen/ausblenden",
        "notice": "Ankündigungen",
        "notice_empty": "Keine Ankündigungen",
        "save": "Speichern",
        "cancel": "Abbrechen",
        "new_playlist": "Neue Playlist",
        "playlist_name": "Playlist-Name:",
        "menu_minimize": "In Taskleiste minimieren",
        "menu_quit": "Beenden",
        "menu_about": "Über Starry Flight",
        "menu_privacy": "Datenschutz",
        "menu_update": "Nach Updates suchen",
        "menu_desktop_lrc": "Desktop-Lyrics",
        "menu_audio": "Audio",
        "menu_eq_settings": "Equalizer",
        "menu_spectrum": "Spektrum",
        "menu_history": "Verlauf",
        "menu_fullscreen": "Vollbild",
        "menu_optimize": "Optimierung",
        "set_volume": "Lautstärke",
        "set_not_playing": "Keine Wiedergabe",
        "set_songs_count": "{n} Songs",
        "show_main": "Fenster anzeigen",
        "tray_play": "Wiedergabe/Pause",
        "tray_next": "Weiter",
        "tray_prev": "Zurück",
        "tray_quit": "Beenden",
        "exit_fullscreen": "Vollbild beenden",
        "menu_eq_settings": "Equalizer",
        "menu_hotkey_settings": "Tastenkürzel",
        "menu_dlrc": "Desktop-Lyrics",
        "menu_dlrc_color": "Lyrics-Farbe",
        "menu_dlrc_size": "Lyrics-Größe",
        "menu_dlrc_font": "Lyrics-Schriftart",
        "dlrc_c0": "Türkis",
        "dlrc_c1": "Rot",
        "dlrc_c2": "Gelb",
        "dlrc_c3": "Grün",
        "dlrc_c4": "Blau",
        "dlrc_c5": "Orange",
        "dlrc_c6": "Lila",
        "dlrc_c7": "Weiß (Regenbogen)",
        "dlrc_c8": "Neon",
        "sp_c0": "Türkis",
        "sp_c1": "Neon",
        "sp_c2": "Regenbogen",
        "sub_price": "¥5.00 / Monat",
        "about_version": "Version 1.0.0",
        "about_desc": "Lokaler Musikplayer mit Lyrics, Spektrum, Cloud-Konto.",
        "privacy_agree": "Ich habe die Bedingungen gelesen und stimme zu",
        "privacy_ok": "Zustimmen und fortfahren",
        "privacy_cancel": "Ablehnen und beenden",
        "up_to_date": "Sie sind auf dem neuesten Stand v1.0.0",
        "privacy_text": "<h3>Datenschutzerklärung</h3><p>Willkommen bei Starry Flight.</p><p><b>1. Datenerhebung</b></p><p>Nur E-Mail, Spitzname, Avatar. Musik und Verlauf bleiben lokal.</p><p><b>2. Speicherung</b></p><p>Kontodaten via Supabase-Cloud, nicht verkauft.</p><p><b>3. Lokale Dateien</b></p><p>Nur lokale Musik, kein Audio-Upload.</p><p><b>4. Lyrics</b></p><p>Lokale .lrc-Dateien werden bevorzugt. Online-Lyrics sind standardmäßig deaktiviert; wenn aktiviert, wird nur der Text für die Anzeige abgerufen, keine Audiodateien.</p><p><b>5. Kontakt</b></p><p>support@starryflight.app</p>",
        "theme_cyberpunk": "Cyberpunk",
        "theme_forest": "Waldgrün",
        "all_songs": "Alle Titel",
        "favorites": "★ Favoriten",
        "error": "Fehler",
        "new_playlist_fail": "Playlist-Erstellung fehlgeschlagen: {err}",
        "history_title": "Verlauf",
        "eq_title": "Equalizer",
        "hotkey_title": "Tastenkürzel-Einstellungen",
        "hotkey_tip": "Klicken Sie auf eine Schaltfläche und drücken Sie dann die gewünschte Taste",
        "hotkey_play": "Abspielen/Pause",
        "hotkey_next": "Nächster",
        "hotkey_prev": "Vorheriger",
        "key_space": "Leertaste",
        "key_right": "Rechtspfeil",
        "key_left": "Linkspfeil",
        "key_up": "Hochpfeil",
        "key_down": "Abwärtspfeil",
        "key_enter": "Eingabetaste",
        "cancel_fav": "Aus Favoriten entfernen",
        "add_to_playlist": "Zur Playlist hinzufügen",
        "no_custom_pl": "（Keine benutzerdefinierten Playlists）",
        "new_pl_action": "＋ Neue Playlist…",
        "skip_msg": "Übersprungen: {name}",
        "select_music_dir": "Musikordner wählen",
        "select_bg_image": "Hintergrundbild wählen",
        "select_bg_video": "Hintergrundvideo wählen",
        "select_avatar": "Avatar wählen",
        "select_skin": "QSS-Datei für Skin wählen",
        "skin_too_large": "Skin-Datei zu groß (max. 1 MB)",
        "skin_utf8_required": "Skin-Datei muss UTF-8-kodiert sein",
        "skin_load_fail": "Skin laden fehlgeschlagen: {err}",
        "nick_placeholder": "Nickname festlegen",
        "fav_count": "Favoriten: {n}",
        "pl_count": "Playlists: {n}",
        "search_hint": "Songs suchen...",
        "new_playlist_tip": "Neue Playlist",
        "expand_list": "Songliste erweitern",
        "close": "Schließen",
        "profile_title": "Profil",
        "profile_click_avatar": "Klicken Sie auf den Avatar, um das Bild zu ändern",
        "nickname_label": "Nickname: ",
        "spacing_val": "{v:.1f}x",
        "press_any_key": "Beliebige Taste drücken...",
        "tray_minimized": "In den Tray minimiert, Doppelklick zum Wiederherstellen",
        "dlrc_playing": "♪ Wiedergabe...",
        "notice_default": "Willkommen bei Starry Flight! Ankündigungen können im Feld announcements der config.json bearbeitet werden.",
    },
    "kor": {
        "app_title": "Starry Flight 음악 플레이어",
        "brand": "♪  Starry Flight 음악",
        "settings": "외관 설정",
        "open_dir": "폴더 열기",
        "not_playing": "재생 중이 아님",
        "unknown_artist": "알 수 없는 아티스트",
        "count_fmt": "{n}곡",
        "mode_order": "순서 재생",
        "mode_loop": "한 곡 반복",
        "mode_random": "랜덤 재생",
        "no_lyric": "가사 없음",
        "lang_orig": "원문",
        "lang_trans": "번역",
        "lang_both": "대역",
        "list_font_size": "목록 글자 크기",
        "lyric_font_size": "가사 글자 크기",
        "lyric_line_spacing": "가사 줄 간격",
        "corner_radius": "UI 라운드",
        "lyric_translate": "가사 번역",
        "trans_need_song": "먼저 노래를 재생하고 가사를 불러온 뒤 번역하세요",
        "trans_loading": "가사 번역 중…",
        "trans_failed": "번역 실패: 네트워크를 확인하거나 나중에 다시 시도하세요",
        "lyric_color": "하이라이트 색상",
        "lyric_font": "가사 글꼴",
        "language": "인터페이스 언어",
        "toggle_list": "노래 목록 표시/숨기기",
        "notice": "공지",
        "notice_empty": "공지 없음",
        "save": "저장",
        "cancel": "취소",
        "new_playlist": "새 재생목록",
        "playlist_name": "재생목록 이름:",
        "menu_minimize": "트레이로 최소화",
        "menu_quit": "종료",
        "menu_about": "Starry Flight 정보",
        "menu_privacy": "개인정보 처리방침",
        "menu_update": "업데이트 확인",
        "menu_desktop_lrc": "데스크톱 가사",
        "menu_audio": "오디오",
        "menu_eq_settings": "이퀄라이저",
        "menu_spectrum": "스펙트럼",
        "menu_history": "재생 기록",
        "menu_fullscreen": "전체 화면",
        "menu_optimize": "최적화",
        "set_volume": "음량",
        "set_not_playing": "재생 중이 아님",
        "set_songs_count": "{n}곡",
        "show_main": "메인 창 표시",
        "tray_play": "재생/일시정지",
        "tray_next": "다음 곡",
        "tray_prev": "이전 곡",
        "tray_quit": "종료",
        "exit_fullscreen": "전체 화면 종료",
        "menu_eq_settings": "이퀄라이저",
        "menu_hotkey_settings": "단축키 설정",
        "menu_dlrc": "데스크톱 가사",
        "menu_dlrc_color": "가사 색상",
        "menu_dlrc_size": "가사 크기",
        "menu_dlrc_font": "가사 글꼴",
        "dlrc_c0": "청록색",
        "dlrc_c1": "빨간색",
        "dlrc_c2": "노란색",
        "dlrc_c3": "초록색",
        "dlrc_c4": "파란색",
        "dlrc_c5": "주황색",
        "dlrc_c6": "보라색",
        "dlrc_c7": "흰색(무지개)",
        "dlrc_c8": "네온",
        "sp_c0": "청록색",
        "sp_c1": "네온",
        "sp_c2": "무지개",
        "sub_price": "¥5.00 / 월",
        "about_version": "버전 1.0.0",
        "about_desc": "로컬 음악 플레이어. 가사, 스펙트럼, 클라우드 계정 지원.",
        "privacy_agree": "위 약관을 읽고 동의합니다",
        "privacy_ok": "동의하고 계속",
        "privacy_cancel": "동의하지 않고 종료",
        "up_to_date": "최신 버전 v1.0.0입니다",
        "privacy_text": "<h3>개인정보 처리방침</h3><p>Starry Flight에 오신 것을 환영합니다.</p><p><b>1. 데이터 수집</b></p><p>계정 이메일, 닉네임, 아바타만 수집. 음악과 기록은 로컬 저장.</p><p><b>2. 저장</b></p><p>계정 데이터는 Supabase 클라우드에 저장, 제3자에게 판매 안 함.</p><p><b>3. 로컬 파일</b></p><p>로컬 음악만 재생, 오디오 업로드 안 함.</p><p><b>4. 가사</b></p><p>로컬 .lrc 파일 우선. 온라인 가사는 기본 꺼짐이며, 켜면 표시용 가사 텍스트만 가져오고 오디오 파일은 업로드하지 않습니다.</p><p><b>5. 연락처</b></p><p>support@starryflight.app</p>",
        "theme_cyberpunk": "사이버펑크",
        "theme_forest": "포레스트 그린",
        "all_songs": "모든 노래",
        "favorites": "★ 즐겨찾기",
        "error": "오류",
        "new_playlist_fail": "새 플레이리스트 만들기 실패: {err}",
        "history_title": "재생 기록",
        "eq_title": "이퀄라이저",
        "hotkey_title": "단축키 설정",
        "hotkey_tip": "버튼을 클릭한 후 원하는 키를 누르세요",
        "hotkey_play": "재생/일시정지",
        "hotkey_next": "다음 곡",
        "hotkey_prev": "이전 곡",
        "key_space": "스페이스",
        "key_right": "오른쪽 화살표",
        "key_left": "왼쪽 화살표",
        "key_up": "위쪽 화살표",
        "key_down": "아래쪽 화살표",
        "key_enter": "엔터",
        "cancel_fav": "즐겨찾기 해제",
        "add_to_playlist": "플레이리스트에 추가",
        "no_custom_pl": "（사용자 지정 플레이리스트 없음）",
        "new_pl_action": "＋ 새 플레이리스트…",
        "skip_msg": "건너뜀: {name}",
        "select_music_dir": "음악 폴더 선택",
        "select_bg_image": "배경 이미지 선택",
        "select_bg_video": "배경 동영상 선택",
        "select_avatar": "아바타 선택",
        "select_skin": "스킨 QSS 파일 선택",
        "skin_too_large": "스킨 파일이 너무 큽니다(최대 1MB)",
        "skin_utf8_required": "스킨 파일은 UTF-8 인코딩이어야 합니다",
        "skin_load_fail": "스킨 불러오기 실패: {err}",
        "nick_placeholder": "닉네임 설정",
        "fav_count": "즐겨찾기 곡: {n}곡",
        "pl_count": "플레이리스트 수: {n}",
        "search_hint": "노래 검색...",
        "new_playlist_tip": "새 플레이리스트",
        "expand_list": "노래 목록 펼치기",
        "close": "닫기",
        "profile_title": "프로필",
        "profile_click_avatar": "아바타를 클릭하여 이미지 변경",
        "nickname_label": "닉네임: ",
        "spacing_val": "{v:.1f}배",
        "press_any_key": "아무 키나 누르세요...",
        "tray_minimized": "트레이로 최소화됨. 더블클릭하여 복원",
        "dlrc_playing": "♪ 재생 중...",
        "notice_default": "Starry Flight에 오신 것을 환영합니다! 공지사항은 config.json의 announcements에서 수정할 수 있습니다.",
    },
    "ru": {
        "app_title": "Starry Flight Музыкальный плеер",
        "brand": "♪  Starry Flight Музыка",
        "settings": "Настройки внешнего вида",
        "open_dir": "Открыть папку",
        "not_playing": "Не воспроизводится",
        "unknown_artist": "Неизвестный исполнитель",
        "count_fmt": "{n} песен",
        "mode_order": "По порядку",
        "mode_loop": "Повтор одной",
        "mode_random": "Перемешать",
        "no_lyric": "Нет текста",
        "lang_orig": "Оригинал",
        "lang_trans": "Перевод",
        "lang_both": "Двуязычный",
        "list_font_size": "Размер шрифта списка",
        "lyric_font_size": "Размер шрифта текста",
        "lyric_line_spacing": "Межстрочный интервал",
        "corner_radius": "Скругление интерфейса",
        "lyric_translate": "Перевод текста",
        "trans_need_song": "Сначала включите песню, загрузите текст и затем переведите",
        "trans_loading": "Перевод текста…",
        "trans_failed": "Ошибка перевода: проверьте сеть или повторите позже",
        "lyric_color": "Цвет подсветки",
        "lyric_font": "Шрифт текста",
        "language": "Язык интерфейса",
        "toggle_list": "Показать/Скрыть список",
        "notice": "Объявления",
        "notice_empty": "Нет объявлений",
        "save": "Сохранить",
        "cancel": "Отмена",
        "new_playlist": "Новый плейлист",
        "playlist_name": "Название плейлиста:",
        "menu_minimize": "Свернуть в трей",
        "menu_quit": "Выход",
        "menu_about": "О Starry Flight",
        "menu_privacy": "Конфиденциальность",
        "menu_update": "Проверить обновления",
        "menu_desktop_lrc": "Десктоп-текст",
        "menu_audio": "Аудио",
        "menu_eq_settings": "Эквалайзер",
        "menu_spectrum": "Спектр",
        "menu_history": "История",
        "menu_fullscreen": "Полный экран",
        "menu_optimize": "Оптимизация",
        "set_volume": "Громкость",
        "set_not_playing": "Ничего не играет",
        "set_songs_count": "{n} треков",
        "show_main": "Показать окно",
        "tray_play": "Играть/Пауза",
        "tray_next": "Следующий",
        "tray_prev": "Предыдущий",
        "tray_quit": "Выход",
        "exit_fullscreen": "Выйти из полноэкранного",
        "menu_eq_settings": "Эквалайзер",
        "menu_hotkey_settings": "Горячие клавиши",
        "menu_dlrc": "Десктоп-текст",
        "menu_dlrc_color": "Цвет текста",
        "menu_dlrc_size": "Размер текста",
        "menu_dlrc_font": "Шрифт текста",
        "dlrc_c0": "Бирюзовый",
        "dlrc_c1": "Красный",
        "dlrc_c2": "Желтый",
        "dlrc_c3": "Зеленый",
        "dlrc_c4": "Синий",
        "dlrc_c5": "Оранжевый",
        "dlrc_c6": "Фиолетовый",
        "dlrc_c7": "Белый (Радуга)",
        "dlrc_c8": "Неон",
        "sp_c0": "Бирюзовый",
        "sp_c1": "Неон",
        "sp_c2": "Радуга",
        "sub_price": "¥5.00 / месяц",
        "about_version": "Версия 1.0.0",
        "about_desc": "Локальный музыкальный плеер с текстами, спектром, облачным аккаунтом.",
        "privacy_agree": "Я прочитал и согласен с условиями",
        "privacy_ok": "Согласен и продолжить",
        "privacy_cancel": "Не согласен и выйти",
        "up_to_date": "У вас последняя версия v1.0.0",
        "privacy_text": "<h3>Политика конфиденциальности</h3><p>Добро пожаловать в Starry Flight.</p><p><b>1. Сбор данных</b></p><p>Только email, ник, аватар. Музыка и история хранятся локально.</p><p><b>2. Хранение</b></p><p>Данные аккаунта в облаке Supabase, не продаются третьим лицам.</p><p><b>3. Локальные файлы</b></p><p>Только локальная музыка, аудио не загружается.</p><p><b>4. Тексты</b></p><p>Приоритет — локальные .lrc файлы. Онлайн-тексты отключены по умолчанию; при включении запрашивается только текст для отображения, аудиофайлы не загружаются.</p><p><b>5. Контакты</b></p><p>support@starryflight.app</p>",
        "theme_cyberpunk": "Киберпанк",
        "theme_forest": "Лесная зелень",
        "all_songs": "Все песни",
        "favorites": "★ Избранное",
        "error": "Ошибка",
        "new_playlist_fail": "Не удалось создать плейлист: {err}",
        "history_title": "История",
        "eq_title": "Эквалайзер",
        "hotkey_title": "Настройки горячих клавиш",
        "hotkey_tip": "Нажмите кнопку, затем клавишу",
        "hotkey_play": "Воспроизведение/Пауза",
        "hotkey_next": "Следующий",
        "hotkey_prev": "Предыдущий",
        "key_space": "Пробел",
        "key_right": "Стрелка вправо",
        "key_left": "Стрелка влево",
        "key_up": "Стрелка вверх",
        "key_down": "Стрелка вниз",
        "key_enter": "Enter",
        "cancel_fav": "Убрать из избранного",
        "add_to_playlist": "Добавить в плейлист",
        "no_custom_pl": "（Нет пользовательских плейлистов）",
        "new_pl_action": "＋ Новый плейлист…",
        "skip_msg": "Пропущено: {name}",
        "select_music_dir": "Выберите папку с музыкой",
        "select_bg_image": "Выбрать фоновое изображение",
        "select_bg_video": "Выбрать фоновое видео",
        "select_avatar": "Выбрать аватар",
        "select_skin": "Выберите файл скина QSS",
        "skin_too_large": "Файл скина слишком большой (макс. 1 МБ)",
        "skin_utf8_required": "Файл скина должен быть в кодировке UTF-8",
        "skin_load_fail": "Не удалось загрузить скин: {err}",
        "nick_placeholder": "Установить ник",
        "fav_count": "Избранные песни: {n}",
        "pl_count": "Плейлисты: {n}",
        "search_hint": "Поиск песен...",
        "new_playlist_tip": "Новый плейлист",
        "expand_list": "Развернуть список песен",
        "close": "Закрыть",
        "profile_title": "Профиль",
        "profile_click_avatar": "Нажмите на аватар, чтобы сменить изображение",
        "nickname_label": "Ник: ",
        "spacing_val": "{v:.1f}x",
        "press_any_key": "Нажмите любую клавишу...",
        "tray_minimized": "Свернуто в трей, двойной щелчок для восстановления",
        "dlrc_playing": "♪ Воспроизведение...",
        "notice_default": "Добро пожаловать в Starry Flight! Объявления можно изменить в поле announcements в config.json.",
    },
}


DEFAULT_NOTICE = TEXTS["zh"]["notice_default"]


def load_config() -> dict:
    default = {
        "list_font_size": 15,
        "list_font_family": "Microsoft YaHei",
        "list_color": "#ffffff",
        "lyric_font_size": 20,
        "trans_font_size": 16,
        "trans_line_spacing": 2.0,
        "lyric_line_spacing": 2.4,   # 行高 = 字号 * 该系数
        "lyric_color": "#ffffff",
        "lyric_font_family": "Microsoft YaHei",
        "lyric_bg_color": "#141414",
        "lyric_bg_theme": True,   # True=主题默认渐变背景；False=纯色/图片/视频
        "lyric_bg_fit": False,    # True=背景图等比例缩放；False=拉伸填满（随窗口）
        "lyric_bg_opacity": 1.0,  # 歌词背景（图片/视频/渐变）透明度 0.0~1.0
        "lyric_bg_image": "",
        "language": "zh",
        "window_width": 1080,
        "window_height": 680,
        "settings_win_w": 460,
        "settings_win_h": 640,
        "last_dir": "",
        "theme": "cyberpunk",
        "corner_radius": 10,
        "lyric_trans_lang": "off",
        "lyric_offset": 0.0,   # 歌词时间偏移（秒），正=歌词提前，负=歌词延后
        "online_lyrics_enabled": False,  # 联网歌词（商业化默认关闭）
        "trans_panel_width": 320,
        "volume": 80,
        "playback_speed": 1.0,
        "announcements": [DEFAULT_NOTICE],
        "playlists": {},           # {歌单名: [歌曲路径, ...]}
        "favorites": [],           # 收藏的歌曲路径列表
        "current_playlist": "all",
        "nickname": "",            # 用户昵称
        "avatar_path": "",        # 自定义头像路径
        "profile_w": 360,         # 个人主页窗口大小
        "profile_h": 480,
        "window_x": 100,          # 窗口位置
        "window_y": 80,
        "notify_enabled": True,    # 桌面通知
        "hotkey_enabled": True,    # 全局快捷键
        "preload_enabled": True,   # 预加载
        "play_history": [],        # 播放历史
        "history_w": 400,
        "history_h": 500,
        "spectrum_color": 0,      # 频谱颜色模式
        "spectrum_block": False,   # 方块分段样式
        "list_panel_w": 360,      # 歌曲栏宽度
        "privacy_accepted": False,  # 首次启动隐私政策确认
        "lang_selected": False,    # 首次启动语言选择标记
        "desktop_lyrics": False,    # 桌面歌词
        "dlrc_x": 200,
        "dlrc_y": 100,
        "dlrc_font_size": 24,
        "dlrc_color": "#00f5d4",
        "dlrc_color_mode": 0,
        "dlrc_font_family": "Microsoft YaHei",
        "hotkey_play": "Space",
        "hotkey_next": "Right",
        "hotkey_prev": "Left",
    }
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user = json.load(f)
        default.update({k: v for k, v in user.items() if k in default})
    except Exception:
        pass
    return default


def save_config(cfg: dict):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


@dataclass
class Song:
    path: str
    title: str = ""
    artist: str = ""
    album: str = ""
    duration: float = 0.0

    @property
    def display_name(self) -> str:
        if self.artist:
            return f"{self.title}  -  {self.artist}"
        return self.title or os.path.splitext(os.path.basename(self.path))[0]


AUDIO_EXTS = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".wma", ".ape"}


def _read_tag(path: str) -> Dict[str, str]:
    info = {"title": "", "artist": "", "album": "", "duration": 0.0}
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".mp3":
            try:
                audio = EasyID3(path)
                info["title"] = audio.get("title", [""])[0]
                info["artist"] = audio.get("artist", [""])[0]
                info["album"] = audio.get("album", [""])[0]
            except ID3NoHeaderError:
                pass
            try:
                info["duration"] = MP3(path).info.length
            except Exception:
                pass
        elif ext == ".flac":
            a = FLAC(path)
            info["title"] = a.get("title", [""])[0]
            info["artist"] = a.get("artist", [""])[0]
            info["album"] = a.get("album", [""])[0]
            info["duration"] = a.info.length
        else:
            f = MutagenFile(path)
            if f is not None:
                try:
                    info["duration"] = f.info.length
                except Exception:
                    pass
                tags = f.tags or {}
                for k, v in tags.items():
                    lk = k.lower()
                    if "title" in lk and not info["title"]:
                        info["title"] = str(v[0])
                    elif "artist" in lk and not info["artist"]:
                        info["artist"] = str(v[0])
                    elif "album" in lk and not info["album"]:
                        info["album"] = str(v[0])
    except Exception:
        pass
    return info


def scan_music(root: str) -> List[Song]:
    songs = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in AUDIO_EXTS:
                full = os.path.join(dirpath, fn)
                tag = _read_tag(full)
                title = tag["title"] or os.path.splitext(fn)[0]
                songs.append(Song(
                    path=full, title=title.strip(),
                    artist=tag["artist"].strip(), album=tag["album"].strip(),
                    duration=float(tag["duration"] or 0.0),
                ))
    songs.sort(key=lambda s: s.display_name.lower())
    return songs


LRC_TIME_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
BRACKET_RE = re.compile(r"[（(\[【][^）)\]】]*[）)\]】]")
NOISE_WORDS = re.compile(
    r"(伴奏|纯音乐|instrumental|karaoke|cover|翻自|"
    r"live|演唱会|现场版|demo|remix|重制版|高音质|hq|sq|"
    r"官方|mv|歌词版|歌词|原版|正式版|专辑版)",
    re.IGNORECASE,
)


def clean_name(s: str) -> str:
    if not s:
        return ""
    s = BRACKET_RE.sub(" ", s)
    s = NOISE_WORDS.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_lrc(text: str) -> List[tuple]:
    lines = []
    for raw in text.splitlines():
        stamps = LRC_TIME_RE.findall(raw)
        if not stamps:
            continue
        content = LRC_TIME_RE.sub("", raw).strip()
        for m, s in stamps:
            t = int(m) * 60 + float(s)
            lines.append((t, content))
    lines.sort(key=lambda x: x[0])
    return lines


def _read_lrc_file(p: str) -> Optional[str]:
    if not os.path.isfile(p):
        return None
    for enc in ("utf-8", "utf-8-sig", "gbk", "utf-16"):
        try:
            with open(p, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return None


def load_local_lrc(song_path: str):
    """返回 (原文lrc文本, 译文lrc文本或None)"""
    base_no_ext = os.path.splitext(song_path)[0]
    orig = _read_lrc_file(base_no_ext + ".lrc")
    trans = _read_lrc_file(base_no_ext + ".trans.lrc")
    return orig, trans


def _score_result(target_title: str, target_artist: str, hit: dict) -> float:
    score = 0.0
    h_name = clean_name(hit.get("name", ""))
    h_ar = clean_name(" ".join(a.get("name", "") for a in hit.get("artists", [])))
    t_name = clean_name(target_title)
    t_ar = clean_name(target_artist)
    if not t_name:
        return 0
    if t_name == h_name:
        score += 100
    elif t_name in h_name or h_name in t_name:
        score += 60
    else:
        score += difflib.SequenceMatcher(None, t_name, h_name).ratio() * 50
    if t_ar:
        if t_ar == h_ar:
            score += 40
        elif t_ar in h_ar or h_ar in t_ar:
            score += 25
        else:
            score += difflib.SequenceMatcher(None, t_ar, h_ar).ratio() * 20
    return score


def auto_timestamp(path: str, duration: float, n_lines: int) -> Optional[List[float]]:
    """自动时间戳：按音频能量分布把 n_lines 句歌词均衡分配到时间轴，
    使每句歌词间隔与音乐实际节奏/停顿同步（能量均分法，替代纯平均分配）。"""
    if n_lines <= 0 or not path or not os.path.isfile(path) or not duration or duration <= 0:
        return None
    try:
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            data, sr = librosa.load(path, sr=None, mono=True)
        if data.ndim > 1:
            mono = data.mean(axis=1)
        else:
            mono = data
        hop = 2048
        frames = max(1, int(len(mono) // hop))
        seg = mono[:frames * hop].reshape(frames, hop)
        energy = np.sqrt((seg ** 2).mean(axis=1) + 1e-9)
        # 平滑（移动平均），消除毛刺
        k = max(1, frames // 100)
        kern = np.ones(k, dtype=np.float32) / k
        energy_s = np.convolve(energy, kern, mode="same")
        cum = np.cumsum(energy_s + 1e-6)
        total = cum[-1]
        if total <= 0:
            return None
        n = int(n_lines)
        times = []
        for i in range(n):
            target = total * (i + 0.5) / n
            idx = int(np.searchsorted(cum, target))
            t = idx * hop / float(sr)
            times.append(max(0.0, min(t, duration)))
        return times
    except Exception:
        return None


def fetch_lyrics_lrclib(title: str, artist: str, duration: float = 0.0) -> Optional[str]:
    """lrclib.net 免费歌词库，无需token，返回同步LRC。"""
    try:
        params = {"track_name": title, "artist_name": artist}
        if duration:
            params["duration"] = round(duration, 1)
        r = requests.get("https://lrclib.net/api/get", params=params, timeout=8)
        if r.status_code == 200:
            data = r.json()
            lrc = data.get("syncedLyrics") or data.get("plainLyrics")
            if lrc:
                return lrc
        # 退化到搜索
        r = requests.get("https://lrclib.net/api/search",
                         params={"track_name": title, "artist_name": artist}, timeout=8)
        if r.status_code == 200:
            for item in r.json():
                lrc = item.get("syncedLyrics") or item.get("plainLyrics")
                if lrc:
                    return lrc
    except Exception:
        return None
    return None


def fetch_lyrics_online(title: str, artist: str, song_path: str = "") -> Optional[str]:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Referer": "https://music.163.com/",
    })
    queries = []
    if title and artist:
        queries += [f"{artist} {title}", f"{title} {artist}"]
    if title:
        queries.append(title)

    def _qq():
        try:
            qq = requests.Session()
            qq.headers.update({"User-Agent":"Mozilla/5.0","Referer":"https://y.qq.com/"})
            for kw in queries:
                rq = qq.get("https://c.y.qq.com/soso/fcgi-bin/client_search_cp", params={"w":kw,"format":"json","p":1,"n":10}, timeout=8)
                items = ((rq.json().get("data") or {}).get("song") or {}).get("list") or []
                for item in items[:8]:
                    mid = item.get("songmid")
                    if not mid: continue
                    rl = qq.get("https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_new.fcg", params={"songmid":mid,"format":"json","nobase64":1}, timeout=6)
                    lyc = (rl.json() or {}).get("lyric")
                    if lyc:
                        # QQ 歌词接口文本为 GBK 编码，json 解码后成乱码，需转回 UTF-8
                        for _enc in ("gbk", "utf-8"):
                            try:
                                lyc = lyc.encode("latin-1").decode(_enc)
                                if "[" in lyc:
                                    break
                            except Exception:
                                continue
                    if lyc and "[00:" in lyc:
                        if song_path:
                            try: open(os.path.splitext(song_path)[0]+".lrc","w",encoding="utf-8").write(lyc)
                            except: pass
                        return lyc, None
        except: pass
        return None, None

    if artist and ("周杰伦" in artist or "Jay Chou" in artist):
        return _qq()

    # 收集所有候选 (score, id)，按分数从高到低排序
    candidates = []
    seen = set()
    for kw in queries:
        try:
            r = s.post("https://music.163.com/api/search/get/web",
                       data={"s": kw, "type": 1, "offset": 0, "total": True, "limit": 10},
                       timeout=8)
            hits = (r.json().get("result") or {}).get("songs") or []
        except Exception:
            continue
        for h in hits:
            sid = h.get("id")
            if sid in seen:
                continue
            seen.add(sid)
            candidates.append((_score_result(title, artist, h), sid))
    candidates.sort(key=lambda x: -x[0])

    # 逐个试取歌词，第一个带时间轴的就用（原唱可能下架，翻唱可能反而有词）
    for sc, sid in candidates[:10]:
        try:
            r2 = s.get(f"https://music.163.com/api/song/lyric?id={sid}&lv=1&kv=1&tv=-1", timeout=6)
            j = r2.json() or {}
            lyc = (j.get("lrc") or {}).get("lyric")
            tlyc = (j.get("tlyric") or {}).get("lyric")
        except Exception:
            continue
        if lyc and "[00:" in lyc:
            if song_path:
                try:
                    with open(os.path.splitext(song_path)[0] + ".lrc", "w", encoding="utf-8") as f:
                        f.write(lyc)
                    if tlyc and "[00:" in tlyc:
                        with open(os.path.splitext(song_path)[0] + ".trans.lrc", "w", encoding="utf-8") as f:
                            f.write(tlyc)
                except Exception:
                    pass
            return lyc, (tlyc if tlyc and "[00:" in tlyc else None)
    return None, None


class PreloadWorker(QThread):
    """后台预加载下一首音频数据。"""
    done = pyqtSignal(object, object)  # (data, sr)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            import librosa
            data, sr = librosa.load(self.path, sr=None, mono=False)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            data = data.T.astype('float32')
            self.done.emit(data, sr)
        except Exception:
            self.done.emit(None, None)


class LyricsWorker(QThread):
    done = pyqtSignal(list, list)

    def __init__(self, song: Song):
        super().__init__()
        self.song = song

    def run(self):
        # 商业化版本：仅加载本地 .lrc 歌词文件，不联网抓取
        orig, trans = load_local_lrc(self.song.path)
        self.done.emit(parse_lrc(orig) if orig else [],
                       parse_lrc(trans) if trans else [])


class VideoBgThread(QThread):
    """OpenCV 解码视频帧 → QImage → 主线程 QLabel 显示。不经过 Qt 多媒体，避免 QMediaPlayer 崩溃。"""
    frame = pyqtSignal(object)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self._path = path
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            import cv2
            cap = cv2.VideoCapture(self._path)
            if not cap.isOpened():
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            delay_ms = max(10, int(1000.0 / max(1.0, min(fps, 30.0))))
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                self.frame.emit(qimg)
                self.msleep(delay_ms)
            cap.release()
        except Exception as e:
            print("video bg err:", e)


MODE_ORDER = "order"
MODE_LOOP = "loop"
MODE_RANDOM = "random"
MODE_LABELS = {MODE_ORDER: "顺序播放", MODE_LOOP: "单曲循环", MODE_RANDOM: "随机播放"}


class IconButton(QPushButton):
    """Apple Music 风格的播放控制按钮：自绘图标，无背景。
    kind: "prev" | "next" | "play" | "pause"
    """
    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.setFixedSize(70, 70)
        self.setCursor(Qt.PointingHandCursor)
        self._hover = False

    def enterEvent(self, e): self._hover = True; self.update()
    def leaveEvent(self, e): self._hover = False; self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = QColor("#ffffff") if self._hover else QColor("#d8d8dc")
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(color))
        w, h = self.width(), self.height()
        if self.kind == "prev":
            # 上一首（左）：竖线在左 + 三角朝左，贴紧
            lx = w*0.28
            p.drawRect(int(lx), int(h*0.30), 5, int(h*0.40))
            tri = QPolygonF([
                QPointF(lx+6+w*0.28, h*0.28),
                QPointF(lx+6+w*0.28, h*0.72),
                QPointF(lx+6, h*0.50),
            ])
            p.drawPolygon(tri)
        elif self.kind == "next":
            # 下一首（右）：三角朝右 + 竖线在右，贴紧
            rx = w*0.72
            p.drawRect(int(rx-5), int(h*0.30), 5, int(h*0.40))
            tri = QPolygonF([
                QPointF(rx-6-w*0.28, h*0.28),
                QPointF(rx-6-w*0.28, h*0.72),
                QPointF(rx-6, h*0.50),
            ])
            p.drawPolygon(tri)
        elif self.kind == "play":
            tri = QPolygonF([
                QPointF(w*0.34, h*0.24),
                QPointF(w*0.34, h*0.76),
                QPointF(w*0.72, h*0.50),
            ])
            p.drawPolygon(tri)
        elif self.kind == "pause":
            p.drawRect(int(w*0.34), int(h*0.26), 9, int(h*0.48))
            p.drawRect(int(w*0.56), int(h*0.26), 9, int(h*0.48))
        p.end()

class ClickSlider(QSlider):
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            ratio = e.x() / max(1, self.width())
            self.setValue(int(ratio * self.maximum()))
        super().mousePressEvent(e)


class NeonListDelegate(QStyledItemDelegate):
    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.mw = main_win
        self._phase = 0.0
    def tick(self):
        self._phase += 0.03
        lw = self.parent()
        if lw is not None:
            lw.viewport().update()
    def paint(self, painter, option, index):
        painter.save()
        text = index.data() or ''
        mode = self.mw.cfg.get('list_color', '#ffffff')
        if mode in ('neon', 'rainbow') and text:
            painter.setFont(QFont(option.font, option.font.pointSize(), QFont.Bold if option.state & QStyle.State_Selected else QFont.Normal))
            x = option.rect.x() + 10
            y = option.rect.center().y()
            fm = painter.fontMetrics()
            for ci, ch in enumerate(text):
                if mode == 'rainbow':
                    hue = (ci * 25 + self._phase * 60) % 360
                    c = QColor.fromHsv(int(hue), 230, 255)
                else:
                    t0 = (ci * 0.15 + self._phase) % 3.0
                    if t0 < 1.0:
                        c = QColor(0, int(240*(1-t0)), int(255-25*t0))
                    elif t0 < 2.0:
                        tt = t0-1.0
                        c = QColor(int(255*tt), int(252*tt), int(230*(1-tt)))
                    else:
                        tt = t0-2.0
                        c = QColor(int(255*(1-tt)), int(252-12*tt), int(255*tt))
                painter.setPen(c)
                painter.drawText(x, y + fm.ascent()//2 - fm.descent()//2, ch)
                x += fm.horizontalAdvance(ch)
        else:
            painter.restore()
            super().paint(painter, option, index)
            return
        painter.restore()


def make_notice_icon(size=26):
    """微信风格气泡图标（主体圆 + 左下尾巴，形似大写 Q）。"""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    body = QColor("#00f5d4")
    p.setPen(Qt.NoPen)
    p.setBrush(body)
    # 主体圆
    p.drawEllipse(QRectF(3.0, 1.5, size - 10.0, size - 9.0))
    # 左下尾巴（Q 的尾巴）
    tail = QPainterPath()
    tail.moveTo(8.5, 13.0)
    tail.lineTo(4.0, 19.0)
    tail.lineTo(11.5, 16.5)
    tail.closeSubpath()
    p.drawPath(tail)
    # 气泡内两点（眼睛）
    p.setBrush(QColor("#0a0a12"))
    p.drawEllipse(QRectF(8.0, 7.5, 3.4, 3.4))
    p.drawEllipse(QRectF(14.0, 7.5, 3.4, 3.4))
    p.end()
    return QIcon(pm)


class BgLabel(QLabel):
    """背景图 label：默认拉伸填满（随窗口缩放）；可选等比缩放居中（fit），拖动防抖。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._src = None
        self._use_movie = False
        self._fit = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._apply)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)   # 默认拉伸（同旧行为）

    def set_fit(self, on):
        self._fit = bool(on)
        self._apply()

    def set_opacity(self, v):
        """背景整体透明度（对图片/视频/GIF/渐变/纯色均生效）。"""
        v = max(0.0, min(1.0, float(v)))
        eff = self.graphicsEffect()
        if eff is None or not isinstance(eff, QGraphicsOpacityEffect):
            eff = QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(eff)
        eff.setOpacity(v)

    def set_source(self, pix):
        self._use_movie = False
        self._src = pix if not pix.isNull() else None
        self._apply()

    def set_video_frame(self, pix):
        self._use_movie = False
        self._src = pix if not pix.isNull() else None
        self._apply()

    def set_movie(self, mv):
        self._use_movie = True
        self._src = None
        super().setMovie(mv)
        self.setScaledContents(True)

    def clear_source(self):
        self._use_movie = False
        self._src = None
        self.clear()
        self.setScaledContents(True)

    def _apply(self):
        if self._use_movie:
            return
        if self._src is not None:
            if self._fit:
                # 等比缩放居中（不拉伸变形）
                self.setScaledContents(False)
                sz = self.size()
                if sz.width() > 0 and sz.height() > 0:
                    self.setPixmap(self._src.scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                # 拉伸填满（随窗口大小缩放）
                self.setScaledContents(True)
                self.setPixmap(self._src)
        else:
            self.clear()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._use_movie:
            if self._fit:
                self._debounce.start(80)   # 拖动时防抖，避免实时大图缩放卡顿
            else:
                self._apply()

    def sizeHint(self):
        return QSize(0, 0)

    def minimumSizeHint(self):
        return QSize(0, 0)


class SpectrumWidget(QWidget):
    """简易音频频谱可视化（从音频数据取FFT）。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self._bars = [0.0] * 128
        self._smooth = [0.0] * 128
        self._color_mode = 0  # 0=纯色 1=霓虹渐变 2=彩虹
        self._block_style = False  # 方块分段
        self._color1 = QColor(0, 245, 212)
        self._color2 = QColor(120, 0, 255)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(30)

    def set_levels(self, levels):
        self._bars = levels

    def set_color_mode(self, mode):
        self._color_mode = mode

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 0))
        w, h = self.width(), self.height()
        n = len(self._bars)
        bw = w / n
        for i in range(n):
            target = self._bars[i] if i < len(self._bars) else 0
            self._smooth[i] += (target - self._smooth[i]) * 0.25
            v = self._smooth[i]
            bh = max(2, int(v * h))
            # 颜色
            if self._color_mode == 1:
                t = i / n
                c = QColor(
                    int(self._color1.red() + (self._color2.red() - self._color1.red()) * t),
                    int(self._color1.green() + (self._color2.green() - self._color1.green()) * t),
                    int(self._color1.blue() + (self._color2.blue() - self._color1.blue()) * t),
                    220)
            elif self._color_mode == 2:
                import colorsys
                hue = (i / n) * 0.8
                r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                c = QColor(int(r * 255), int(g * 255), int(b * 255), 220)
            else:
                c = QColor(0, 245, 212, 220)
            if self._block_style:
                # 方块分段：每段6px，间隔2px
                seg = 8
                y = h
                filled = bh
                while filled > 0 and y > 0:
                    sh = min(seg - 2, filled)
                    p.fillRect(int(i * bw), y - sh, 5, sh, c)
                    y -= seg
                    filled -= seg
            else:
                p.fillRect(int(i * bw), h - bh, 5, bh, c)


class LyricView(QWidget):
    # show_mode: orig=只原文, trans=只译文, both=双语(原文+译文)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._orig = []
        self._trans = []
        self._current_idx = -1
        self.font_size = 20
        self.line_spacing = 2.0
        self.active_color = QColor("#ffffff")
        self.font_family = "Microsoft YaHei"
        self.inactive_color_base = (160, 160, 160)
        self.show_mode = "orig"
        self._phase = 0.0
        self._bg_color = QColor("#141414")
        self.empty_text = TEXTS["zh"]["no_lyric"]
        self._anim = QTimer(self)
        self._anim.timeout.connect(self._tick_anim)
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setObjectName("lyricView")
        self.setStyleSheet("#lyricView { background: transparent; border: none; }")
    def set_bg_color(self, color_hex):
        self._bg_color = QColor(color_hex)
        self.update()

    def _tick_anim(self):
        self._phase += 0.03
        self.update()

    def start_anim(self):
        if not self._anim.isActive():
            self._anim.start(100)

    def set_lines(self, orig, trans=None):
        self._orig = orig or []
        self._trans = trans or []
        self._current_idx = -1
        # 没译文时强制回原文
        if not self._trans and self.show_mode != "orig":
            self.show_mode = "orig"
        self.update()

    def has_translation(self):
        return bool(self._trans)

    def update_time(self, sec):
        if not self._orig:
            return
        idx = -1
        for i, (t, _) in enumerate(self._orig):
            if sec >= t:
                idx = i
            else:
                break
        if idx != self._current_idx:
            self._current_idx = idx
            self.update()

    def _current_text(self):
        if 0 <= self._current_idx < len(self._orig):
            return self._orig[self._current_idx][1]
        return ""

    def _trans_text_for(self, orig_time: float) -> str:
        if not self._trans:
            return ""
        best = ""
        best_dt = 999
        for t, txt in self._trans:
            if t <= orig_time:
                dt = orig_time - t
                if dt < best_dt:
                    best_dt = dt; best = txt
        return best

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        line_h = int(self.font_size * self.line_spacing)
        if not self._orig:
            p.setPen(QColor("#666"))
            p.setFont(QFont(self.font_family, 12))
            p.drawText(self.rect(), Qt.AlignCenter, self.empty_text)
            return

        center_y = h // 2
        visible = max(5, int(h / line_h) + 2)
        start = max(0, self._current_idx - visible // 2)
        end = min(len(self._orig), start + visible)
        start = max(0, end - visible)

        for i in range(start, end):
            t, orig_text = self._orig[i]
            y = center_y + (i - self._current_idx) * line_h
            if y + line_h < 0 or y - line_h > h:
                continue
            is_cur = (i == self._current_idx)

            # 决定显示什么文本
            if self.show_mode == "trans" and self._trans:
                main_text = self._trans_text_for(t) or orig_text
            elif self.show_mode == "both" and self._trans:
                main_text = orig_text
            else:
                main_text = orig_text

            if is_cur:
                p.setFont(QFont(self.font_family, self.font_size, QFont.Bold))
                if isinstance(self.active_color, str) and self.active_color in ("neon", "rainbow"):
                    fm = p.fontMetrics()
                    text = main_text or ""
                    tw = fm.horizontalAdvance(text)
                    tx = (w - tw) // 2
                    ty = y + line_h // 2 - fm.descent()
                    # 先画一层发光阴影
                    for glow_w, glow_a in [(6, 40), (3, 80)]:
                        g = QLinearGradient(tx, 0, tx + tw, 0)
                        if self.active_color == "rainbow":
                            for k in range(6):
                                hue = (k*60 + self._phase*60) % 360
                                c = QColor.fromHsv(int(hue), 230, 255)
                                c.setAlpha(glow_a)
                                g.setColorAt(k/5.0, c)
                        else:
                            off = (self._phase % 1.0) * 0.6
                            g.setColorAt(0.0, QColor(0, 240, 255, glow_a))
                            g.setColorAt(0.5, QColor(255, 0, 200, glow_a))
                            g.setColorAt(1.0, QColor(255, 230, 0, glow_a))
                            g.setStart(tx - off*tw, 0)
                            g.setFinalStop(tx + tw - off*tw, 0)
                        p.setPen(QPen(QBrush(g), glow_w))
                        p.drawText(tx, ty, text)
                    # 主文字
                    grad = QLinearGradient(tx, 0, tx + tw, 0)
                    if self.active_color == "rainbow":
                        for k in range(6):
                            hue = (k*60 + self._phase*60) % 360
                            grad.setColorAt(k/5.0, QColor.fromHsv(int(hue), 230, 255))
                    else:
                        off = (self._phase % 1.0) * 0.6
                        grad.setColorAt(0.0, QColor(0, 240, 255))
                        grad.setColorAt(0.5, QColor(255, 0, 200))
                        grad.setColorAt(1.0, QColor(255, 230, 0))
                        grad.setStart(tx - off*tw, 0)
                        grad.setFinalStop(tx + tw - off*tw, 0)
                    p.setPen(QPen(QBrush(grad), 1))
                    p.drawText(tx, ty, text)
                else:
                    p.setPen(self.active_color)
                    p.drawText(QRect(20, y - line_h // 2, w - 40, line_h), Qt.AlignCenter, main_text or "")
            else:
                r, g, b = self.inactive_color_base
                alpha = max(40, 210 - abs(i - self._current_idx) * 45)
                p.setPen(QColor(r, g, b, alpha))
                p.setFont(QFont(self.font_family, max(10, self.font_size - 3)))
                p.drawText(QRect(20, y - line_h // 2, w - 40, line_h), Qt.AlignCenter, main_text or "")

            # 双语模式下，在原文行下方画小一号译文
            if self.show_mode == "both" and self._trans:
                tr = self._trans_text_for(t)
                if tr:
                    ty = y + line_h // 2
                    if ty + line_h // 2 < h:
                        if is_cur:
                            p.setPen(QColor(200, 200, 200))
                        else:
                            r, g, b = self.inactive_color_base
                            alpha = max(30, 160 - abs(i - self._current_idx) * 40)
                            p.setPen(QColor(r, g, b, alpha))
                        p.setFont(QFont(self.font_family, max(9, self.font_size - 6)))
                        p.drawText(QRect(20, ty, w - 40, line_h // 2), Qt.AlignCenter, tr)
        p.end()


class SettingsDialog(QDialog):
    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.main_win = main_win
        self.setWindowTitle(main_win._tr("settings"))
        self.setAttribute(Qt.WA_StyledBackground, True)
        # 窗口可自由拉伸（不固定宽度），记住上次拖动后的窗口大小
        self.resize(main_win.cfg.get("settings_win_w", 460), main_win.cfg.get("settings_win_h", 640))
        self.setMinimumSize(380, 460)
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._save_size)
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 16)
        root.setSpacing(10)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { width: 6px; background: rgba(0,0,0,0); margin: 0; }"
            "QScrollBar::handle:vertical { background: rgba(0,255,200,0.45); border-radius: 3px; min-height: 32px; margin: 0; border: none; }"
            "QScrollBar::handle:vertical:hover { background: rgba(0,255,200,0.7); }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; width: 0; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }")
        # 滚动条右键菜单文案随界面语言（由 install_qt_translator 提供翻译）
        root.addWidget(scroll, 1)
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(12)
        scroll.setWidget(container)

        # --- 歌曲列表字号 ---
        lay.addWidget(QLabel(main_win._tr("list_font_size")))
        self.list_slider = QSlider(Qt.Horizontal)
        self.list_slider.setRange(11, 24)
        self.list_slider.setValue(main_win.cfg["list_font_size"])
        self.list_slider.valueChanged.connect(self._apply)
        lay.addWidget(self.list_slider)
        self.list_label = QLabel(f"{main_win.cfg['list_font_size']} px")
        self.list_label.setStyleSheet("color:#9a9a9a; font-size:12px;")
        lay.addWidget(self.list_label)

        # --- 歌词字号 ---
        lay.addWidget(QLabel(main_win._tr("lyric_font_size")))
        self.lf_slider = QSlider(Qt.Horizontal)
        self.lf_slider.setRange(12, 36)
        self.lf_slider.setValue(main_win.lyric_view.font_size)
        self.lf_slider.valueChanged.connect(self._apply)
        lay.addWidget(self.lf_slider)
        self.lf_label = QLabel(f"{main_win.lyric_view.font_size} px")
        self.lf_label.setStyleSheet("color:#9a9a9a; font-size:12px;")
        lay.addWidget(self.lf_label)

        # --- 歌词行距 ---
        lay.addWidget(QLabel(main_win._tr("lyric_line_spacing")))
        self.ls_slider = QSlider(Qt.Horizontal)
        self.ls_slider.setRange(15, 50)          # 1.5 ~ 4.0
        self.ls_slider.setValue(int(main_win.lyric_view.line_spacing * 10))
        self.ls_slider.valueChanged.connect(self._apply)
        lay.addWidget(self.ls_slider)
        self.ls_label = QLabel(main_win._tr("spacing_val").format(v=main_win.lyric_view.line_spacing))
        self.ls_label.setStyleSheet("color:#9a9a9a; font-size:12px;")
        lay.addWidget(self.ls_label)

        # --- 歌词时间偏移（保证歌词与声音同步）---
        lay.addWidget(QLabel(main_win._tr("set_lyric_offset")))
        self.off_slider = QSlider(Qt.Horizontal)
        self.off_slider.setRange(-50, 50)          # -5.0s ~ +5.0s（0.1s 步进）
        self.off_slider.setValue(int(main_win.cfg.get("lyric_offset", 0.0) * 10))
        self.off_slider.valueChanged.connect(self._on_lyric_offset)
        lay.addWidget(self.off_slider)
        self.off_label = QLabel(f"{self.off_slider.value() / 10:+.1f} s")
        self.off_label.setStyleSheet("color:#9a9a9a; font-size:12px;")
        lay.addWidget(self.off_label)

        # --- 界面圆滑度 ---
        lay.addWidget(QLabel(main_win._tr("corner_radius")))
        self.corner_slider = QSlider(Qt.Horizontal)
        self.corner_slider.setRange(0, 16)
        self.corner_slider.setValue(int(main_win.cfg.get("corner_radius", 10)))
        self.corner_slider.valueChanged.connect(self._apply_corner)
        lay.addWidget(self.corner_slider)
        self.corner_label = QLabel(f"{int(main_win.cfg.get('corner_radius', 10))} px")
        self.corner_label.setStyleSheet("color:#9a9a9a; font-size:12px;")
        lay.addWidget(self.corner_label)

        # --- 歌词字体 ---
        lay.addWidget(QLabel(main_win._tr("lyric_font")))
        self.font_combo = QComboBox()
        self.font_combo.setStyleSheet("QComboBox { background:#2a2a2a; color:#eee; padding:6px; border:1px solid #333; border-radius:6px; } QComboBox QAbstractItemView { background:#2a2a2a; color:#eee; selection-background-color:#444; }")
        # (显示名, 实际字体名)
        font_map = [
            ("微软雅黑", "Microsoft YaHei"),
            ("宋体", "SimSun"),
            ("黑体", "SimHei"),
            ("楷体", "KaiTi"),
            ("仿宋", "FangSong"),
            ("等线", "DengXian"),
            ("微软正黑体", "Microsoft JhengHei"),
            ("Arial", "Arial"),
            ("Times New Roman", "Times New Roman"),
            ("Courier New", "Courier New"),
            ("Georgia", "Georgia"),
            ("Verdana", "Verdana"),
        ]
        self._font_real_names = [real for _, real in font_map]
        for disp, real in font_map:
            self.font_combo.addItem(disp, real)
        cur = main_win.lyric_view.font_family
        idx = self.font_combo.findData(cur)
        if idx >= 0:
            self.font_combo.setCurrentIndex(idx)
        self.font_combo.currentTextChanged.connect(self._apply)
        lay.addWidget(self.font_combo)

        # --- 主题 ---
        lay.addWidget(QLabel(main_win._tr("set_theme")))
        self.theme_combo = QComboBox()
        for key, t in THEMES.items():
            self.theme_combo.addItem(main_win._tr("theme_" + key), key)
        cur_theme = main_win.cfg.get("theme", "neon_dark")
        ti = self.theme_combo.findData(cur_theme)
        if ti >= 0: self.theme_combo.setCurrentIndex(ti)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        lay.addWidget(self.theme_combo)

        # --- 界面语言 ---
        lay.addWidget(QLabel(main_win._tr("language")))
        self.lang_combo = QComboBox()
        self.lang_combo.setStyleSheet("QComboBox { background:#2a2a2a; color:#eee; padding:6px; border:1px solid #333; border-radius:6px; } QComboBox QAbstractItemView { background:#2a2a2a; color:#eee; selection-background-color:#444; }")
        self.lang_combo.addItem("中文", "zh")
        self.lang_combo.addItem("English", "en")
        self.lang_combo.addItem("日本語", "ja")
        self.lang_combo.addItem("Français", "fra")
        self.lang_combo.addItem("Deutsch", "de")
        self.lang_combo.addItem("한국어", "kor")
        self.lang_combo.addItem("Русский", "ru")
        li = self.lang_combo.findData(main_win.lang)
        if li >= 0:
            self.lang_combo.setCurrentIndex(li)
        self.lang_combo.currentIndexChanged.connect(self._on_lang_changed)
        lay.addWidget(self.lang_combo)

        # --- 颜色 ---
        lay.addWidget(QLabel(main_win._tr("lyric_color")))
        self._color_btns = []
        row = QHBoxLayout()
        for hexc in ["#ffffff", "#ffd166", "#8ecae6", "#95d5b2", "#ffb4a2", "#e0aaff"]:
            b = QPushButton()
            b.setFixedSize(32, 32)
            b.setStyleSheet(f"background-color:{hexc}; border-radius:16px; border:2px solid transparent;")
            b.clicked.connect(lambda _, c=hexc, btn=b: self._set_color(c, btn))
            row.addWidget(b)
            self._color_btns.append((hexc, b))
        # 霓虹渐变
        b_neon = QPushButton(main_win._tr("neon"))
        b_neon.setFixedSize(56, 32)
        b_neon.setStyleSheet("background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #00ffff,stop:0.5 #ff00ff,stop:1 #ffff00); color:white; border-radius:8px; font-weight:bold; border:2px solid transparent;")
        b_neon.clicked.connect(lambda: self._set_color("neon", b_neon))
        row.addWidget(b_neon)
        self._color_btns.append(("neon", b_neon))
        # 彩虹
        b_rb = QPushButton(main_win._tr("rainbow"))
        b_rb.setFixedSize(56, 32)
        b_rb.setStyleSheet("background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #f00,stop:0.2 #ff0,stop:0.4 #0f0,stop:0.6 #0ff,stop:0.8 #00f,stop:1 #f0f); color:white; border-radius:8px; font-weight:bold; border:2px solid transparent;")
        b_rb.clicked.connect(lambda: self._set_color("rainbow", b_rb))
        row.addWidget(b_rb)
        self._color_btns.append(("rainbow", b_rb))
        lay.addLayout(row)
        self._refresh_color_feedback()

        # --- 歌词背景 ---
        lay.addWidget(QLabel(main_win._tr("set_lyric_bg_color")))
        self._bg_colors = ["#141414", "#000000", "#1a1a2e", "#2d1b4e", "#1b3a2f", "#3d2817"]
        brow = QHBoxLayout()
        self._bg_btns = []
        for hexc in self._bg_colors:
            b = QPushButton()
            b.setFixedSize(28, 28)
            b.setStyleSheet(f"background-color:{hexc}; border-radius:14px; border:2px solid transparent;")
            b.clicked.connect(lambda _, c=hexc: self._set_bg(c))
            brow.addWidget(b)
            self._bg_btns.append((hexc, b))
        # 默认主题背景
        b_defbg = QPushButton(main_win._tr("default_theme"))
        b_defbg.setFixedHeight(28)
        b_defbg.setCursor(Qt.PointingHandCursor)
        b_defbg.clicked.connect(self._set_bg_theme)
        brow.addWidget(b_defbg)
        brow.addStretch()
        lay.addLayout(brow)

        lay.addWidget(QLabel(main_win._tr("set_lyric_bg")))
        brow2 = QHBoxLayout()
        self.btn_bg_img = QPushButton(main_win._tr("set_pick_image"))
        self.btn_bg_img.clicked.connect(self._pick_bg_image)
        brow2.addWidget(self.btn_bg_img)
        self.btn_bg_video = QPushButton(main_win._tr("set_pick_video"))
        self.btn_bg_video.clicked.connect(self._pick_bg_video)
        brow2.addWidget(self.btn_bg_video)
        self.btn_bg_clear = QPushButton(main_win._tr("set_clear_bg"))
        self.btn_bg_clear.clicked.connect(self._clear_bg_image)
        brow2.addWidget(self.btn_bg_clear)
        self.chk_bg_fit = QCheckBox(main_win._tr("set_fit_scale"))
        self.chk_bg_fit.setChecked(main_win.cfg.get("lyric_bg_fit", False))
        self.chk_bg_fit.toggled.connect(self._on_bg_fit)
        brow2.addWidget(self.chk_bg_fit)
        brow2.addStretch()
        lay.addLayout(brow2)

        # 背景图/视频透明度
        lay.addWidget(QLabel(main_win._tr("set_bg_opacity")))
        self.op_slider = QSlider(Qt.Horizontal)
        self.op_slider.setRange(10, 100)
        self.op_slider.setValue(int(main_win.cfg.get("lyric_bg_opacity", 1.0) * 100))
        self.op_slider.valueChanged.connect(self._on_bg_opacity)
        lay.addWidget(self.op_slider)
        self.op_label = QLabel(f"{self.op_slider.value()}%")
        self.op_label.setStyleSheet("color:#9a9a9a; font-size:12px;")
        lay.addWidget(self.op_label)

        # 歌曲列表颜色
        lay.addWidget(QLabel(main_win._tr("set_list_color")))
        self._list_color_btns = []
        row2 = QHBoxLayout()
        for hexc in ["#ffffff", "#ffd166", "#8ecae6", "#95d5b2", "#ffb4a2", "#e0aaff"]:
            b2 = QPushButton()
            b2.setFixedSize(32, 32)
            b2.setStyleSheet(f"background-color:{hexc}; border-radius:16px; border:2px solid transparent;")
            b2.clicked.connect(lambda _, c=hexc, btn=b2: self._set_list_color(c, btn))
            row2.addWidget(b2)
            self._list_color_btns.append((hexc, b2))
        lay.addLayout(row2)
        self._refresh_list_color_feedback()

        # 歌曲列表字体
        lay.addWidget(QLabel(main_win._tr("set_list_font")))
        self.lf_combo = QComboBox()
        self.lf_combo.setStyleSheet("QComboBox { background:#2a2a2a; color:#eee; padding:6px; border:1px solid #333; border-radius:6px; } QComboBox QAbstractItemView { background:#2a2a2a; color:#eee; selection-background-color:#444; }")
        fonts = [("微软雅黑","Microsoft YaHei"),("宋体","SimSun"),("楷体","KaiTi"),("仿宋","FangSong"),("Arial","Arial"),("Times New Roman","Times New Roman"),("Courier New","Courier New"),("Verdana","Verdana"),("Tahoma","Tahoma"),("Georgia","Georgia"),("Impact","Impact"),("Comic Sans MS","Comic Sans MS")]
        for cn, en in fonts:
            self.lf_combo.addItem(cn, en)
        lfi = self.lf_combo.findData(main_win.cfg.get("list_font_family","Microsoft YaHei"))
        if lfi >= 0: self.lf_combo.setCurrentIndex(lfi)
        self.lf_combo.currentTextChanged.connect(self._apply)
        lay.addWidget(self.lf_combo)

        btns = QDialogButtonBox(self)
        self.btn_save = btns.addButton(main_win._tr("save"), QDialogButtonBox.AcceptRole)
        btns.accepted.connect(self._on_ok)
        root.addWidget(btns)
        self.setStyleSheet(STYLE)

    def _on_theme_changed(self):
        key = self.theme_combo.currentData()
        if not key: return
        self.main_win.cfg["theme"] = key
        self.main_win.setStyleSheet(get_style(key, self.main_win.cfg.get("corner_radius", 10)))
        # 翻译面板背景跟随主题（该方法在部分版本不存在，需防御性调用）
        if hasattr(self.main_win, "_apply_trans_panel_style"):
            try:
                self.main_win._apply_trans_panel_style()
            except Exception:
                pass
        if hasattr(self.main_win, "_style_pl_row"):
            self.main_win._style_pl_row()
        # 歌词栏主题渐变背景跟随主题切换（图片/纯色模式不受影响）
        if self.main_win.cfg.get("lyric_bg_theme", True) and not self.main_win.cfg.get("lyric_bg_image", ""):
            self.main_win._apply_lyric_bg()
        save_config(self.main_win.cfg)

    def _apply_corner(self, v):
        self.corner_label.setText(f"{v} px")
        self.main_win.cfg["corner_radius"] = int(v)
        self.main_win.setStyleSheet(get_style(self.main_win.cfg.get("theme", "cyberpunk"), int(v)))
        if hasattr(self.main_win, "_style_pl_row"):
            self.main_win._style_pl_row()
        save_config(self.main_win.cfg)

    def _tr(self, key: str) -> str:
        return self.main_win._tr(key)

    def _refresh_texts(self):
        """语言切换后刷新本对话框内非 _tr 构建的文本。"""
        self.list_label.setText(f"{self.list_slider.value()} px")
        self.lf_label.setText(f"{self.lf_slider.value()} px")
        self.ls_label.setText(self._tr("spacing_val").format(v=self.ls_slider.value() / 10.0))
        self.off_label.setText(f"{self.off_slider.value() / 10:+.1f} s")
        self.corner_label.setText(f"{self.corner_slider.value()} px")
        self.op_label.setText(f"{self.op_slider.value()}%")
        for i in range(self.theme_combo.count()):
            key = self.theme_combo.itemData(i)
            if key:
                self.theme_combo.setItemText(i, self._tr("theme_" + key))

    def _on_lang_changed(self):
        new_lang = self.lang_combo.currentData()
        if new_lang and new_lang != self.main_win.lang:
            self.main_win.lang = new_lang
            self.main_win.cfg["language"] = new_lang
            self.main_win._retranslate_ui()
            self._refresh_texts()
            install_qt_translator(new_lang)   # Qt 内置菜单（滚动条右键等）跟随界面语言
            save_config(self.main_win.cfg)

    def _apply(self):
        v_list = self.list_slider.value()
        v_lf = self.lf_slider.value()
        v_ls = self.ls_slider.value() / 10.0
        v_font = self.font_combo.currentData() or self.font_combo.currentText()
        v_lfont = self.lf_combo.currentData() or self.lf_combo.currentText()
        self.main_win.cfg["list_font_family"] = v_lfont
        self.main_win.apply_ui_config(list_size=v_list, lyric_size=v_lf,
                                      lyric_spacing=v_ls, lyric_font=v_font)
        self.main_win.apply_list_style()
        save_config(self.main_win.cfg)
        self.list_label.setText(f"{v_list} px")
        self.lf_label.setText(f"{v_lf} px")
        self.ls_label.setText(self._tr("spacing_val").format(v=v_ls))

    def _set_color(self, hexc, btn=None):
        if hexc in ("neon", "rainbow"):
            self.main_win.lyric_view.active_color = hexc
            self.main_win.lyric_view.start_anim()
        else:
            self.main_win.lyric_view.active_color = QColor(hexc)
        self.main_win.lyric_view.update()
        self.main_win.cfg["lyric_color"] = hexc
        save_config(self.main_win.cfg)
        self._refresh_color_feedback(btn)

    def _refresh_color_feedback(self, active_btn=None):
        cur = self.main_win.cfg.get("lyric_color", "#ffffff")
        for hexc, b in self._color_btns:
            if b is active_btn or hexc == cur:
                b.setStyleSheet(f"background-color:{hexc}; border-radius:16px; border:2px solid #fff;")
            else:
                b.setStyleSheet(f"background-color:{hexc}; border-radius:16px; border:2px solid transparent;")

    def _set_bg(self, hexc):
        self.main_win.cfg["lyric_bg_color"] = hexc
        self.main_win.cfg["lyric_bg_theme"] = False   # 手动选色 → 关闭主题渐变
        save_config(self.main_win.cfg)
        self._apply_bg()
        for c, b in self._bg_btns:
            b.setStyleSheet(f"background-color:{c}; border-radius:14px; border:2px solid {'#fff' if c==hexc else 'transparent'};")

    def _set_bg_theme(self):
        """恢复主题默认渐变背景。"""
        self.main_win.cfg["lyric_bg_theme"] = True
        self.main_win.cfg["lyric_bg_image"] = ""
        save_config(self.main_win.cfg)
        self._apply_bg()
        for c, b in self._bg_btns:
            b.setStyleSheet(f"background-color:{c}; border-radius:14px; border:2px solid transparent;")

    def _apply_bg(self):
        self.main_win._apply_lyric_bg()

    def _pick_bg_image(self):
        try:
            from PyQt5.QtWidgets import QFileDialog
            path, _ = QFileDialog.getOpenFileName(
                self, self._tr("select_bg_image"), "",
                "图片 (*.png *.jpg *.jpeg *.bmp *.gif)")
            if path:
                self.main_win.cfg["lyric_bg_image"] = path
                self.main_win.cfg["lyric_bg_theme"] = False
                save_config(self.main_win.cfg)
                self._apply_bg()
        except Exception as e:
            print("bg img err:", e)

    def _pick_bg_video(self):
        from PyQt5.QtWidgets import QMessageBox, QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, self._tr("select_bg_video"), "",
            "视频 (*.mp4 *.mkv *.avi *.webm *.mov *.flv *.wmv *.m4v)")
        if path:
            self.main_win.cfg["lyric_bg_image"] = path
            self.main_win.cfg["lyric_bg_theme"] = False
            save_config(self.main_win.cfg)
            self._apply_bg()

    def _clear_bg_image(self):
        # 清除图片/视频 → 恢复主题默认渐变
        self.main_win.cfg["lyric_bg_image"] = ""
        self.main_win.cfg["lyric_bg_theme"] = True
        save_config(self.main_win.cfg)
        self._apply_bg()

    def _on_bg_fit(self, on):
        self.main_win.cfg["lyric_bg_fit"] = bool(on)
        save_config(self.main_win.cfg)
        if self.main_win.cfg.get("lyric_bg_image", ""):
            self.main_win._apply_lyric_bg()

    def _on_bg_opacity(self, v):
        self.main_win.cfg["lyric_bg_opacity"] = v / 100.0
        self.main_win.lyric_bg_label.set_opacity(v / 100.0)
        self.op_label.setText(f"{v}%")
        save_config(self.main_win.cfg)

    def _on_lyric_offset(self, v):
        self.main_win.cfg["lyric_offset"] = v / 10.0
        self.off_label.setText(f"{v / 10:+.1f} s")
        save_config(self.main_win.cfg)

    def _set_list_color(self, hexc, btn=None):
        self.main_win.cfg["list_color"] = hexc
        self.main_win.apply_list_style()
        save_config(self.main_win.cfg)
        self.main_win.list_widget.viewport().update()
        self._refresh_list_color_feedback(btn)

    def _refresh_list_color_feedback(self, active_btn=None):
        cur = self.main_win.cfg.get("list_color", "#ffffff")
        for hexc, b in self._list_color_btns:
            if b is active_btn or hexc == cur:
                b.setStyleSheet(f"background-color:{hexc}; border-radius:16px; border:2px solid #fff;")
            else:
                b.setStyleSheet(f"background-color:{hexc}; border-radius:16px; border:2px solid transparent;")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # 防抖：停止拖动 600ms 后再保存窗口大小
        self._resize_timer.start(600)

    def _save_size(self):
        self.main_win.cfg["settings_win_w"] = self.width()
        self.main_win.cfg["settings_win_h"] = self.height()
        save_config(self.main_win.cfg)

    def done(self, r):
        self._save_size()
        super().done(r)

    def _on_ok(self):
        self._apply()
        save_config(self.main_win.cfg)
        self.accept()


class MainWindow(QMainWindow):
    EDGE = 6  # 边缘缩放热区宽度
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Starry Flight 音乐播放器")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.cfg = load_config()
        self.resize(self.cfg.get("window_width", 1080), self.cfg.get("window_height", 680))
        self.move(self.cfg.get("window_x", 100), self.cfg.get("window_y", 80))
        self.setMinimumSize(920, 580)
        self.songs = []
        self._visible = []
        self._was_maximized = False
        self.current_index = -1
        self.mode = MODE_ORDER
        self._lyric_worker = None
        self._duration = 0.0
        self._seeking = False
        self._paused = False
        self._offset = 0.0
        self.audio = AudioEngine(self.cfg.get("volume", 80) / 100.0)
        self.lang = self.cfg.get("language", "zh")
        self._build_ui()
        # 应用已保存的配置
        self._apply_lyric_style()
        self._apply_lyric_bg()
        self._apply_list_font(self.cfg["list_font_size"])
        self._retranslate_ui()

        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        # 拖动进度条防抖：150ms 内没新动作才真正 seek，避免高频 stop/play
        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(150)
        self._seek_timer.timeout.connect(self._do_seek)
        # 睡眠定时器
        self._sleep_timer = QTimer(self)
        self._sleep_timer.setSingleShot(True)
        self._sleep_timer.timeout.connect(lambda: (self.audio.stop(), self.hide()))
        # 全局快捷键（可在菜单开关）
        if self.cfg.get("hotkey_enabled", True):
            self._setup_hotkeys()
        # 自动恢复桌面歌词
        if self.cfg.get("desktop_lyrics", False):
            QTimer.singleShot(300, lambda: self._toggle_desktop_lyrics(True))
        last = self.cfg.get("last_dir", "")
        if last and os.path.isdir(last):
            QTimer.singleShot(200, lambda: self.scan_dir(last))
        else:
            d = os.path.join(os.path.expanduser("~"), "Music")
            if os.path.isdir(d):
                QTimer.singleShot(200, lambda: self.scan_dir(d))

    def _apply_lyric_style(self):
        """统一应用歌词与译文面板的字体、字号、行距、颜色（含霓虹/彩虹渐变）。"""
        fs = self.cfg.get("lyric_font_size", 20)
        ls = self.cfg.get("lyric_line_spacing", 2.4)
        ff = self.cfg.get("lyric_font_family", "Microsoft YaHei")
        col = self.cfg.get("lyric_color", "#ffffff")
        active = col if col in ("rainbow", "neon") else QColor(col)
        # 歌词面板
        self.lyric_view.font_size = fs
        self.lyric_view.line_spacing = ls
        self.lyric_view.font_family = ff
        self.lyric_view.active_color = active
        self.lyric_view.update()

    def apply_list_style(self):
        """统一调度歌曲列表的字体、字号、颜色。"""
        ff = self.cfg.get("list_font_family", "Microsoft YaHei")
        fs = self.cfg.get("list_font_size", 15)
        lc = self.cfg.get("list_color", "#ffffff")
        if lc and not lc.startswith("#"):
            lc = "#ffffff"
        self.list_widget.setFont(QFont(ff, fs))
        self.list_widget.setStyleSheet(
            f"QListWidget {{ color: {lc}; background: rgba(255,255,255,0.04); border-radius:10px; "
            f"font-family: '{ff}'; font-size: {fs}px; }}")

    def _apply_list_font(self, size: int):
        self.cfg["list_font_size"] = size
        self.apply_list_style()

    def apply_ui_config(self, list_size, lyric_size, lyric_spacing, lyric_font=None):
        self._apply_list_font(list_size)
        self.cfg["lyric_font_size"] = lyric_size
        self.cfg["lyric_line_spacing"] = lyric_spacing
        if lyric_font:
            self.cfg["lyric_font_family"] = lyric_font
        self._apply_lyric_style()

    def _stop_video_bg(self):
        if getattr(self, "_video_bg", None) is not None:
            try:
                self._video_bg.stop()
                self._video_bg.wait(300)
            except Exception:
                pass
            self._video_bg = None

    def _style_pl_row(self):
        """歌单选择行样式：跟随全局主题圆角设置，只自定义背景色。"""
        tkey = self.cfg.get("theme", "cyberpunk")
        if tkey == "forest":
            accent = "#38ef7d"
            bg = "rgba(15,30,25,0.9)"
            popup_bg = "#0f2018"
            border = "rgba(56,239,125,0.35)"
            txt = "#d4f5e9"
        else:
            accent = "#00f5d4"
            bg = "rgba(10,10,18,0.9)"
            popup_bg = "#0a0a14"
            border = "rgba(0,245,212,0.35)"
            txt = "#c8ffe8"
        # 只设背景和颜色，border-radius 交给全局主题按 corner_radius 缩放
        cr = int(self.cfg.get("corner_radius", 10))
        ratio = cr / 10.0
        box_r = max(0, min(round(8 * ratio), 16))
        view_r = max(0, min(round(10 * ratio), 16))
        self.pl_combo.setStyleSheet(f"""
            QComboBox {{
                background: {bg};
                color: {txt};
                border: 1px solid {border};
                border-radius: {box_r}px;
                padding: 4px 12px;
                selection-background-color: {accent};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
                background: transparent;
            }}
            QComboBox QAbstractItemView {{
                background: {popup_bg};
                color: {txt};
                border: 1px solid {border};
                border-radius: {view_r}px;
                padding: 4px;
                selection-background-color: {accent};
                selection-color: #000;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 6px 12px;
                min-height: 20px;
            }}
        """)
        self.btn_pl_new.setStyleSheet(f"""
            QPushButton {{
                background: {bg};
                color: {accent};
                border: 1px solid {border};
                border-radius: {box_r}px;
                font-weight: bold;
                font-size: 18px;
            }}
            QPushButton:hover {{
                background: rgba(0,245,212,0.2);
            }}
        """)
        view = self.pl_combo.view()
        view.setAutoFillBackground(True)
        from PyQt5.QtGui import QPalette, QColor
        pal = view.palette()
        pal.setBrush(view.backgroundRole(), QColor(popup_bg))
        view.setPalette(pal)

    def _apply_lyric_bg(self):
        """背景：主题渐变 → 视频 → 图片 → 纯色。"""
        img = self.cfg.get("lyric_bg_image", "")
        theme = self.cfg.get("lyric_bg_theme", True)
        color = self.cfg.get("lyric_bg_color", "#141414")
        self.lyric_bg_label.set_opacity(self.cfg.get("lyric_bg_opacity", 1.0))
        vid_ext = (".mp4", ".mkv", ".avi", ".webm", ".mov", ".flv", ".wmv", ".m4v")
        if img and img.lower().endswith(vid_ext):
            self._stop_video_bg()
            try:
                t = VideoBgThread(img, self)
                t.frame.connect(self._on_video_bg_frame)
                self.lyric_bg_label.set_fit(self.cfg.get("lyric_bg_fit", False))
                self.lyric_bg_label.setStyleSheet("#lyricBgLabel { background: #000; }")
                self._video_bg = t
                t.start()
            except Exception as e:
                print("video bg start err:", e)
                self._video_bg = None
                self.lyric_bg_label.clear_source()
                self.lyric_bg_label.setStyleSheet(f"#lyricBgLabel {{ background: {color}; }}")
            return
        self._stop_video_bg()
        if img:
            try:
                if img.lower().endswith(".gif"):
                    mv = QMovie(img)
                    if mv.isValid():
                        self.lyric_bg_label.set_movie(mv)
                        mv.start()
                        self.lyric_bg_label.setStyleSheet("#lyricBgLabel { background: transparent; }")
                        return
                pix = QPixmap(img)
                if not pix.isNull():
                    self.lyric_bg_label.set_fit(self.cfg.get("lyric_bg_fit", False))
                    self.lyric_bg_label.set_source(pix)
                    self.lyric_bg_label.setStyleSheet("#lyricBgLabel { background: transparent; }")
                    return
            except Exception as e:
                print("bg apply err:", e)
        # 无图：主题默认渐变
        self._stop_video_bg()
        if theme:
            tkey = self.cfg.get("theme", "cyberpunk")
            style = THEMES.get(tkey, THEMES["cyberpunk"]).get("lyric_bg")
            if style:
                self.lyric_bg_label.clear_source()
                self.lyric_bg_label.setStyleSheet(f"#lyricBgLabel {{ background: {style}; }}")
                return
        # 纯色
        self.lyric_bg_label.clear_source()
        self.lyric_bg_label.setStyleSheet(f"#lyricBgLabel {{ background: {color}; }}")

    def _on_video_bg_frame(self, qimg):
        self.lyric_bg_label.set_video_frame(QPixmap.fromImage(qimg))

    def _build_menu(self):
        self._menu.clear()
        act_settings = self._menu.addAction(self._tr("settings"))
        act_settings.triggered.connect(self.open_settings)
        act_open = self._menu.addAction(self._tr("open_dir"))
        act_open.triggered.connect(self.choose_dir)
        act_toggle = self._menu.addAction(self._tr("toggle_list"))
        act_toggle.triggered.connect(self.toggle_list)
        self._act_full = self._menu.addAction(self._tr("menu_fullscreen"))
        self._act_full.triggered.connect(self.toggle_fullscreen)
        eq_menu = self._menu.addMenu(self._tr("menu_audio"))
        act_eq = eq_menu.addAction(self._tr("menu_eq_settings"))
        act_eq.triggered.connect(self._show_eq)
        act_hotkey = eq_menu.addAction(self._tr("menu_hotkey_settings"))
        act_hotkey.triggered.connect(self._show_hotkey_settings)
        act_desktop_lrc = eq_menu.addAction(self._tr("menu_dlrc"))
        act_desktop_lrc.setCheckable(True)
        act_desktop_lrc.setChecked(self.cfg.get("desktop_lyrics", False))
        act_desktop_lrc.triggered.connect(lambda c: (self.cfg.__setitem__("desktop_lyrics", c), save_config(self.cfg), self._toggle_desktop_lyrics(c)))
        dlrc_color_menu = eq_menu.addMenu(self._tr("menu_dlrc_color"))
        dlrc_colors = [self._tr(f"dlrc_c{i}") for i in range(9)]
        current_mode = self.cfg.get("dlrc_color_mode", 0)
        for i, name in enumerate(dlrc_colors):
            a = dlrc_color_menu.addAction(("✓ " if current_mode == i else "") + name)
            a.triggered.connect(lambda checked=False, m=i: (
                self.cfg.__setitem__("dlrc_color_mode", m),
                save_config(self.cfg),
                self._apply_dlrc_style(),
                self._menu.close()
            ))
        dlrc_size_menu = eq_menu.addMenu(self._tr("menu_dlrc_size"))
        for s in [16, 20, 24, 28, 32, 36, 42, 48]:
            a = dlrc_size_menu.addAction(f"{s}px")
            a.triggered.connect(lambda checked=False, v=s: (
                self.cfg.__setitem__("dlrc_font_size", v),
                save_config(self.cfg),
                self._apply_dlrc_style(),
                self._menu.close()
            ))
        dlrc_font_menu = eq_menu.addMenu(self._tr("menu_dlrc_font"))
        for f in ["Microsoft YaHei", "SimSun", "KaiTi", "Arial", "Times New Roman", "Courier New"]:
            a = dlrc_font_menu.addAction(f)
            a.triggered.connect(lambda checked=False, v=f: (
                self.cfg.__setitem__("dlrc_font_family", v),
                save_config(self.cfg),
                self._apply_dlrc_style(),
                self._menu.close()
            ))
        self._menu.addSeparator()
        opt_menu = self._menu.addMenu("⚡ " + self._tr("menu_optimize"))
        act_history = opt_menu.addAction(self._tr("menu_history"))
        act_history.triggered.connect(self._show_history)
        act_skin = opt_menu.addAction(self._tr("menu_import_skin"))
        act_skin.triggered.connect(self._import_skin)
        self._menu.addSeparator()
        sp_menu = self._menu.addMenu(self._tr("menu_spectrum"))
        current_color = self.cfg.get("spectrum_color", 0)
        for name, mode in [(self._tr("sp_c0"), 0), (self._tr("sp_c1"), 1), (self._tr("sp_c2"), 2)]:
            a = sp_menu.addAction(name)
            a.setCheckable(True)
            a.setChecked(current_color == mode)
            a.triggered.connect(lambda checked=False, m=mode: (self.spectrum.set_color_mode(m),
                                                                self.cfg.__setitem__("spectrum_color", m),
                                                                save_config(self.cfg)))
        a_block = sp_menu.addAction(self._tr("menu_spectrum_block"))
        a_block.setCheckable(True)
        a_block.setChecked(self.cfg.get("spectrum_block", False))
        def _toggle_block(checked):
            self.spectrum._block_style = checked
            self.cfg["spectrum_block"] = checked
            save_config(self.cfg)
        a_block.triggered.connect(_toggle_block)
        act_notify = opt_menu.addAction(self._tr("menu_notify"))
        act_notify.setCheckable(True)
        act_notify.setChecked(self.cfg.get("notify_enabled", True))
        act_notify.triggered.connect(lambda c: self.cfg.__setitem__("notify_enabled", c) or save_config(self.cfg))
        act_hotkey = opt_menu.addAction(self._tr("menu_global_hotkey"))
        act_hotkey.setCheckable(True)
        act_hotkey.setChecked(self.cfg.get("hotkey_enabled", True))
        act_hotkey.triggered.connect(lambda c: (self.cfg.__setitem__("hotkey_enabled", c), save_config(self.cfg)))
        act_preload = opt_menu.addAction(self._tr("menu_preload"))
        act_preload.setCheckable(True)
        act_preload.setChecked(self.cfg.get("preload_enabled", True))
        act_preload.triggered.connect(lambda checked: self.cfg.__setitem__("preload_enabled", checked) or save_config(self.cfg))
        act_search = opt_menu.addAction(self._tr("menu_search"))
        act_search.setCheckable(True)
        act_search.setChecked(True)
        act_search.triggered.connect(lambda checked: self.search_box.setVisible(checked))
        sleep_menu = opt_menu.addMenu(self._tr("menu_sleep_timer"))
        act_sleep_off = sleep_menu.addAction(self._tr("menu_off"))
        act_sleep_off.triggered.connect(lambda: self._sleep_timer.stop())
        for mins in (15, 30, 60):
            a = sleep_menu.addAction(f"{mins}{self._tr('menu_minutes_later')}")
            a.triggered.connect(lambda checked=False, m=mins: self._sleep_timer.start(m * 60 * 1000))
        self._menu.addSeparator()
        self._act_min = self._menu.addAction(self._tr("menu_minimize"))
        self._act_min.triggered.connect(self.hide)
        self._act_quit = self._menu.addAction(self._tr("menu_quit"))
        self._act_quit.triggered.connect(self._tray_quit)
        self._menu.addSeparator()
        act_desktop = self._menu.addAction(self._tr("menu_desktop_lrc"))
        act_desktop.setCheckable(True)
        act_desktop.setChecked(self.cfg.get("desktop_lyrics", False))
        act_desktop.triggered.connect(lambda c: (self.cfg.__setitem__("desktop_lyrics", c), save_config(self.cfg), self._toggle_desktop_lyrics(c)))
        self._menu.addSeparator()
        act_about = self._menu.addAction(self._tr("menu_about"))
        act_about.triggered.connect(self._show_about)
        act_privacy = self._menu.addAction(self._tr("menu_privacy"))
        act_privacy.triggered.connect(self._show_privacy)
        act_update = self._menu.addAction(self._tr("menu_update"))
        act_update.triggered.connect(self._check_update)
        self._menu_actions = (act_settings, act_open, act_toggle)

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        central.installEventFilter(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- 自定义标题栏（无边框窗口拖拽 + 最小化/最大化/关闭）----
        self.title_bar = QWidget()
        self.title_bar.setFixedHeight(34)
        title_bar = self.title_bar
        self.title_bar.installEventFilter(self)
        tb = QHBoxLayout(title_bar)
        tb.setContentsMargins(10, 0, 6, 0)
        tb.setSpacing(6)
        tb_title = QLabel("♪  Starry Flight")
        tb_title.setStyleSheet("color:#c8ffe8; font-size:13px; font-weight:bold;")
        self._tb_title = tb_title
        tb_title.installEventFilter(self)
        tb.addWidget(tb_title)
        tb.addStretch(1)
        self.btn_min = QPushButton("—")
        self.btn_max = QPushButton("▢")
        self.btn_close = QPushButton("✕")
        for b in (self.btn_min, self.btn_max, self.btn_close):
            b.setFixedSize(36, 26)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet("QPushButton { background: transparent; color:#c8ffe8; border:none; font-size:14px; } QPushButton:hover { background: rgba(0,245,212,0.2); }")
        self.btn_close.setStyleSheet("QPushButton { background: transparent; color:#c8ffe8; border:none; font-size:14px; } QPushButton:hover { background: #e63946; color:#fff; }")
        self.btn_min.clicked.connect(self.showMinimized)
        self.btn_max.clicked.connect(self._toggle_max)
        self.btn_close.clicked.connect(self.close)
        tb.addWidget(self.btn_min)
        tb.addWidget(self.btn_max)
        tb.addWidget(self.btn_close)
        root.addWidget(title_bar)

        # 内容区
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(20, 8, 20, 16)
        body_lay.setSpacing(14)
        root.addWidget(body, 1)

        top = QHBoxLayout()
        self.brand = QLabel(self._tr("brand"))
        self.brand.setObjectName("brand")
        top.addWidget(self.brand)
        top.addStretch(1)
        # 头像按钮（点击打开个人主页/设置）
        self.btn_avatar = QPushButton()
        self.btn_avatar.setFixedSize(30, 30)
        self.btn_avatar.setCursor(Qt.PointingHandCursor)
        avatar_path = self.cfg.get("avatar_path", "")
        if avatar_path:
            if avatar_path.startswith("data:image"):
                import base64 as b64mod
                pix = QPixmap()
                pix.loadFromData(b64mod.b64decode(avatar_path.split(",",1)[1]), "JPG")
                self.btn_avatar.setIcon(QIcon(pix))
            elif os.path.exists(avatar_path):
                self.btn_avatar.setIcon(QIcon(avatar_path))
            self.btn_avatar.setStyleSheet("QPushButton { background: transparent; border: 2px solid #00f5d4; border-radius: 15px; } QPushButton:hover { background: rgba(0,245,212,0.15); }")
        else:
            self.btn_avatar.setStyleSheet("QPushButton { background: rgba(0,245,212,0.2); border: 2px solid #00f5d4; border-radius: 15px; color: #00f5d4; font-size: 14px; font-weight: bold; } QPushButton:hover { background: rgba(0,245,212,0.35); }")
            self.btn_avatar.setText("♪")
        self.btn_avatar.setToolTip(self._tr("profile_title"))
        self.btn_avatar.clicked.connect(self.open_profile)
        top.addWidget(self.btn_avatar)
        top.addSpacing(6)
        # 公告栏（右上角信息栏，微信风格气泡图标，公告内容存 config.json 的 announcements）
        self.btn_notice = QPushButton()
        self.btn_notice.setIcon(make_notice_icon(26))
        self.btn_notice.setObjectName("menuBtn")
        self.btn_notice.setFixedSize(30, 26)
        self.btn_notice.setStyleSheet("padding:0;")
        self.btn_notice.setCursor(Qt.PointingHandCursor)
        self.btn_notice.clicked.connect(self.show_notice)
        top.addWidget(self.btn_notice)
        top.addSpacing(6)
        self.btn_menu = QPushButton("☰")
        self.btn_menu.setObjectName("menuBtn")
        self.btn_menu.setCursor(Qt.PointingHandCursor)
        self.btn_menu.clicked.connect(self.toggle_menu)
        top.addWidget(self.btn_menu)
        body_lay.addLayout(top)

        # 汉堡下拉菜单
        self._menu = QMenu(self)
        self._menu.setObjectName("appMenu")
        self._build_menu()

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(2)
        self.splitter.setStyleSheet("QSplitter::handle { background: rgba(255,255,255,0.04); } QSplitter::handle:hover { background: #00f5d4; }")
        self.splitter.setChildrenCollapsible(False)

        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(0)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._on_list_menu)
        # 搜索框
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText(self._tr("search_hint"))
        self.search_box.setStyleSheet("QLineEdit { background: rgba(255,255,255,0.08); border: 1px solid rgba(0,245,212,0.3); border-radius: 8px; padding: 4px 8px; color: #d4f5e9; }")
        self.search_box.textChanged.connect(self._filter_songs)
        self.list_container = QWidget()
        lc = QVBoxLayout(self.list_container)
        lc.setContentsMargins(0,0,0,0)
        lc.addWidget(self.search_box)
        lc.setSpacing(0)
        # 顶部：收起按钮 + 歌单选择行
        self.btn_toggle_list = QPushButton(self._tr("collapse_list"))
        self.btn_toggle_list.setFixedHeight(30)
        self.btn_toggle_list.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_list.setStyleSheet("QPushButton { background: rgba(0,245,212,0.1); color:#c8ffe8; border: 1px solid rgba(0,245,212,0.4); border-radius: 8px; font-size:13px; padding:0 12px; } QPushButton:hover { background: rgba(0,245,212,0.25); }")
        self.btn_toggle_list.clicked.connect(self.toggle_list)
        lc.addWidget(self.btn_toggle_list)
        # 歌单切换行
        pl_row = QHBoxLayout()
        pl_row.setContentsMargins(6, 4, 6, 4)
        pl_row.setSpacing(6)
        self.pl_combo = QComboBox()
        self.pl_combo.setObjectName("plCombo")
        self.pl_combo.setMinimumHeight(30)
        self.pl_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.pl_combo.currentIndexChanged.connect(self._on_pl_changed)
        self.btn_pl_new = QPushButton("+")
        self.btn_pl_new.setObjectName("plPlusBtn")
        self.btn_pl_new.setFixedSize(30, 30)
        self.btn_pl_new.setCursor(Qt.PointingHandCursor)
        self.btn_pl_new.setToolTip(self._tr("new_playlist_tip"))
        self.btn_pl_new.clicked.connect(self._create_playlist)
        pl_row.addWidget(self.pl_combo, 1)
        pl_row.addWidget(self.btn_pl_new)
        lc.addLayout(pl_row)
        lc.addWidget(self.list_widget, 1)
        self.splitter.addWidget(self.list_container)
        self._style_pl_row()

        right_w = QWidget()
        right_w.setMinimumWidth(200)
        right = QVBoxLayout(right_w)
        right.setContentsMargins(0,0,0,0)
        right.setSpacing(8)

        # 歌词主区（标题/歌词）与右侧翻译面板用分隔条分隔，可拖动、可收起
        self.right_splitter = QSplitter(Qt.Horizontal)
        self.right_splitter.setHandleWidth(2)
        self.right_splitter.setStyleSheet("QSplitter::handle { background: rgba(255,255,255,0.04); } QSplitter::handle:hover { background: #00f5d4; }")
        self.right_splitter.setChildrenCollapsible(False)

        lyric_area = QWidget()
        la = QVBoxLayout(lyric_area)
        la.setContentsMargins(0, 0, 0, 0)
        la.setSpacing(8)
        self.lbl_title = QLabel(self._tr("not_playing"))
        self.lbl_title.setObjectName("title")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_artist = QLabel("—")
        self.lbl_artist.setObjectName("artist")
        self.lbl_artist.setAlignment(Qt.AlignCenter)
        la.addWidget(self.lbl_title)
        la.addWidget(self.lbl_artist)
        dv = QFrame()
        dv.setFrameShape(QFrame.HLine)
        dv.setStyleSheet("color:#2a2a2a;")
        la.addWidget(dv)

        # 歌词容器：底层背景图 QLabel + 上层透明歌词 LyricView 重叠
        self.lyric_container = QWidget()
        self.lyric_container.setObjectName("lyricContainer")
        self.lyric_container.setStyleSheet("#lyricContainer { background: transparent; }")
        cl = QGridLayout(self.lyric_container)
        cl.setContentsMargins(0, 0, 0, 0)
        self.lyric_bg_label = BgLabel(self.lyric_container)
        self.lyric_bg_label.setObjectName("lyricBgLabel")
        self.lyric_bg_label.setStyleSheet("#lyricBgLabel { background: transparent; }")
        cl.addWidget(self.lyric_bg_label, 0, 0)
        self.lyric_view = LyricView(self.lyric_container)
        self.lyric_view.setObjectName("lyricViewOverlay")
        cl.addWidget(self.lyric_view, 0, 0)
        self.lyric_bg_label.lower()
        self.lyric_view.raise_()
        la.addWidget(self.lyric_container, 1)
        # 频谱可视化条
        self.spectrum = SpectrumWidget(lyric_area)
        self.spectrum.setMaximumHeight(30)
        self.spectrum.set_color_mode(self.cfg.get("spectrum_color", 0))
        self.spectrum._block_style = self.cfg.get("spectrum_block", False)
        la.addWidget(self.spectrum)
        self.right_splitter.addWidget(lyric_area)

        right.addWidget(self.right_splitter, 1)
        self.splitter.addWidget(right_w)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.list_container.setMinimumWidth(200)
        self.splitter.setCollapsible(0, False)
        QTimer.singleShot(100, lambda: self.splitter.setSizes(
            [self.cfg.get("list_panel_w", 360),
             max(100, self.splitter.width() - self.cfg.get("list_panel_w", 360))]))
        self.splitter.splitterMoved.connect(self._on_splitter_moved)
        body_lay.addWidget(self.splitter, 1)
        self._list_expanded = True

        # 系统托盘
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        app_icon = QIcon(icon_path) if os.path.exists(icon_path) else self.style().standardIcon(QStyle.SP_MediaPlay)
        self.setWindowIcon(app_icon)
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(app_icon)
        self.tray.setToolTip(self._tr("brand"))
        tray_menu = QMenu()
        act_show = tray_menu.addAction(self.style().standardIcon(QStyle.SP_TitleBarNormalButton), self._tr("show_main"))
        act_show.triggered.connect(self.showNormal)
        tray_menu.addSeparator()
        act_prev = tray_menu.addAction(self.style().standardIcon(QStyle.SP_MediaSkipBackward), self._tr("tray_prev"))
        act_prev.triggered.connect(self.play_prev)
        act_play = tray_menu.addAction(self.style().standardIcon(QStyle.SP_MediaPlay), self._tr("tray_play"))
        act_play.triggered.connect(self.toggle_play)
        act_next = tray_menu.addAction(self.style().standardIcon(QStyle.SP_MediaSkipForward), self._tr("tray_next"))
        act_next.triggered.connect(self.play_next)
        tray_menu.addSeparator()
        act_quit = tray_menu.addAction(self.style().standardIcon(QStyle.SP_MessageBoxCritical), self._tr("tray_quit"))
        act_quit.triggered.connect(self._tray_quit)
        self.tray.setContextMenu(tray_menu)
        self.tray.activated.connect(lambda r: self.showNormal() if r == QSystemTrayIcon.DoubleClick else None)
        self.tray.show()

        prog = QHBoxLayout()
        self.lbl_cur = QLabel("00:00"); self.lbl_cur.setObjectName("time")
        self.slider = ClickSlider(Qt.Horizontal)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self.slider.sliderReleased.connect(self._on_seek)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.lbl_total = QLabel("00:00"); self.lbl_total.setObjectName("time")
        prog.addWidget(self.lbl_cur); prog.addWidget(self.slider, 1); prog.addWidget(self.lbl_total)

        btns = QHBoxLayout()
        btns.setSpacing(10)
        self.btn_mode = QPushButton(self._tr("mode_order"))
        self.btn_mode.setObjectName("modeBtn")
        self.btn_mode.clicked.connect(self.cycle_mode)
        self.btn_speed = QPushButton("1.0x")
        self.btn_speed.setObjectName("modeBtn")
        self.btn_speed.clicked.connect(self._show_speed_menu)
        self._speed = self.cfg.get("playback_speed", 1.0)
        self.btn_speed.setText(f"{self._speed:g}x")
        bp = IconButton("prev"); bp.clicked.connect(self.play_prev)
        self.btn_play = IconButton("play")
        self.btn_play.clicked.connect(self.toggle_play)
        bn = IconButton("next"); bn.clicked.connect(self.play_next)
        btns.addWidget(self.btn_mode); btns.addWidget(self.btn_speed); btns.addStretch(1)
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setFixedWidth(100)
        self.vol_slider.setMinimum(0); self.vol_slider.setMaximum(100)
        self.vol_slider.setValue(int(self.cfg.get("volume", 80)))
        self.vol_slider.valueChanged.connect(self._set_volume)
        self.lbl_volume = QLabel(self._tr("set_volume"))
        btns.addWidget(self.lbl_volume)
        btns.addWidget(self.vol_slider)
        btns.addSpacing(12)
        btns.addWidget(bp); btns.addSpacing(18); btns.addWidget(self.btn_play); btns.addSpacing(18); btns.addWidget(bn)
        btns.addStretch(1)
        self.lbl_count = QLabel(self._tr("set_songs_count")); self.lbl_count.setObjectName("time")
        btns.addWidget(self.lbl_count)
        # 进度条与控制按钮放入歌词区底部（左右两个分隔条中间，宽度随歌词区变化）
        la.addLayout(prog)
        la.addLayout(btns)
        self.setStyleSheet(get_style(self.cfg.get("theme", "neon_dark"), self.cfg.get("corner_radius", 10)))
        # 全局样式表覆盖后，重新应用歌单栏专属圆角样式
        self._style_pl_row()
        self._apply_list_font(self.cfg["list_font_size"])
        # 标题栏背景
        tkey = self.cfg.get("theme", "cyberpunk")
        if tkey == "forest":
            title_bar.setStyleSheet("background: #0f2018;")
        else:
            title_bar.setStyleSheet("background: #0a0a14;")

    def _toggle_max(self):
        if getattr(self, "_maximized_manual", False):
            # 还原：恢复之前保存的窗口位置和大小
            if hasattr(self, "_geom_before_max"):
                self.setGeometry(self._geom_before_max)
            else:
                self.showNormal()
            self.title_bar.show()
            self._maximized_manual = False
            self.btn_max.setText("▢")
        else:
            self._geom_before_max = self.geometry()
            from PyQt5.QtWidgets import QDesktopWidget
            geo = QDesktopWidget().availableGeometry()
            self.setGeometry(geo)
            self._maximized_manual = True
            self.btn_max.setText("❐")

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.title_bar.setFixedHeight(34)
            self.title_bar.show()
            self._act_full.setText(self._tr("fullscreen"))
        else:
            self.showFullScreen()
            self.title_bar.setFixedHeight(0)
            self.title_bar.hide()
            self._act_full.setText(self._tr("exit_fullscreen"))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F11:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(e)

    EDGE = 8          # 边缘热区宽度(px)
    TITLE_H = 34      # 标题栏高度(px)

    def _edge_at(self, pos):
        """返回鼠标所在边缘方向；最大化/全屏时不响应。"""
        if getattr(self, "_maximized_manual", False) or self.isFullScreen():
            return None
        r = self.rect()
        l = pos.x() <= self.EDGE
        rt = pos.x() >= r.width() - self.EDGE
        t = pos.y() <= self.EDGE
        b = pos.y() >= r.height() - self.EDGE
        if t and l: return "tl"
        if t and rt: return "tr"
        if b and l: return "bl"
        if b and rt: return "br"
        if l: return "l"
        if rt: return "r"
        if t: return "t"
        if b: return "b"
        return None

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        # 全屏时禁止拖拽移动和边缘缩放
        if self.isFullScreen():
            return
        # 1) 标题栏 → 拖拽移动窗口
        if e.pos().y() <= self.TITLE_H:
            self._drag_offset = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()
            return
        # 2) 边缘 → 调整大小
        edge = self._edge_at(e.pos())
        if edge:
            self._resize_edge = edge
            self._resize_start = e.globalPos()
            self._geom_start = self.geometry()
            e.accept()

    def mouseMoveEvent(self, e):
        if self.isFullScreen():
            return
        try:
            # 拖拽移动窗口
            if hasattr(self, "_drag_offset") and e.buttons() & Qt.LeftButton:
                self.move(e.globalPos() - self._drag_offset)
                return
            # 边缘调整大小
            if hasattr(self, "_resize_edge") and e.buttons() & Qt.LeftButton:
                dx = e.globalPos().x() - self._resize_start.x()
                dy = e.globalPos().y() - self._resize_start.y()
                g = self._geom_start
                edge = self._resize_edge
                x, y, w, h = g.x(), g.y(), g.width(), g.height()
                if "l" in edge: x, w = g.x() + dx, g.width() - dx
                if "r" in edge: w = g.width() + dx
                if "t" in edge: y, h = g.y() + dy, g.height() - dy
                if "b" in edge: h = g.height() + dy
                if w < self.minimumWidth(): w = self.minimumWidth(); x = g.x()
                if h < self.minimumHeight(): h = self.minimumHeight(); y = g.y()
                self.setGeometry(x, y, w, h)
                return
            # 悬停光标
            edge = self._edge_at(e.pos())
            cur_map = {"l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
                       "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
                       "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                       "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor}
            self.setCursor(cur_map.get(edge, Qt.ArrowCursor))
        except Exception:
            pass

    def nativeEvent(self, eventType, message):
        """Windows原生边缘缩放处理，不干扰按钮点击。"""
        if eventType == "windows_generic_MSG":
            import ctypes
            import ctypes.wintypes
            msg = ctypes.wintypes.MSG.from_address(int(message))
            if msg.message == 0x0084:  # WM_NCHITTEST
                # 最大化/全屏时不缩放
                if getattr(self, "_maximized_manual", False) or self.isFullScreen():
                    return False, 1  # HTCLIENT
                x = msg.lParam & 0xFFFF
                y = (msg.lParam >> 16) & 0xFFFF
                # 处理坐标符号（多显示器）
                if x > 32767: x -= 65536
                if y > 32767: y -= 65536
                gp = self.mapFromGlobal(QPoint(x, y))
                r = self.rect()
                l = gp.x() <= self.EDGE
                rt = gp.x() >= r.width() - self.EDGE
                t = gp.y() <= self.EDGE
                b = gp.y() >= r.height() - self.EDGE
                border = 6
                if t and l: return True, 13  # HTTOPLEFT
                if t and rt: return True, 14  # HTTOPRIGHT
                if b and l: return True, 16  # HTBOTTOMLEFT
                if b and rt: return True, 17  # HTBOTTOMRIGHT
                if l: return True, 10  # HTLEFT
                if rt: return True, 11  # HTRIGHT
                if t: return True, 12  # HTTOP
                if b: return True, 15  # HTBOTTOM
        return super().nativeEvent(eventType, message)

    def mouseReleaseEvent(self, e):
        self._drag_offset = None
        self._resize_edge = None

    def eventFilter(self, obj, e):
        """只允许从logo区域拖拽移动窗口，全屏时禁用。"""
        try:
            if obj is not self._tb_title:
                return super().eventFilter(obj, e)
            if self.isFullScreen():
                return super().eventFilter(obj, e)
            if e.type() == QMouseEvent.MouseButtonPress and e.button() == Qt.LeftButton:
                self._drag_offset = e.globalPos() - self.frameGeometry().topLeft()
                return True
            if e.type() == QMouseEvent.MouseMove and (e.buttons() & Qt.LeftButton):
                self.move(e.globalPos() - self._drag_offset)
                return True
            if e.type() == QMouseEvent.MouseButtonRelease:
                self._drag_offset = None
                return True
        except Exception:
            pass
        return super().eventFilter(obj, e)

    def _tr(self, key: str) -> str:
        return TEXTS.get(self.lang, TEXTS["zh"]).get(key, key)

    def _retranslate_ui(self):
        self.setWindowTitle(self._tr("app_title"))
        self.brand.setText(self._tr("brand"))
        self.btn_notice.setIcon(make_notice_icon(26))
        self.lbl_title.setText(self._tr("not_playing"))
        self.lbl_artist.setText("—")
        self.btn_menu.setText("☰")
        self.btn_toggle_list.setText(self._tr("expand_list") if not self._list_expanded else self._tr("collapse_list"))
        self.search_box.setPlaceholderText(self._tr("search_hint"))
        self.btn_pl_new.setToolTip(self._tr("new_playlist_tip"))
        self.btn_avatar.setToolTip(self._tr("profile_title"))
        self._refresh_pl_combo()
        if hasattr(self, "lbl_volume"):
            self.lbl_volume.setText(self._tr("set_volume"))
        if hasattr(self, "lbl_count"):
            self.lbl_count.setText(self._tr("set_songs_count"))
        # 重建菜单
        if hasattr(self, "_menu"):
            self._build_menu()
        self._refresh_mode_label()
        self._refresh_count()
        if hasattr(self, "lyric_view"):
            self.lyric_view.empty_text = self._tr("no_lyric")
            self.lyric_view.update()

    def _refresh_mode_label(self):
        keys = {MODE_ORDER: "mode_order", MODE_LOOP: "mode_loop", MODE_RANDOM: "mode_random"}
        self.btn_mode.setText(self._tr(keys[self.mode]))

    def _refresh_count(self):
        self.lbl_count.setText(self._tr("count_fmt").format(n=len(self.songs)))
    def open_settings(self):
        SettingsDialog(self, self).exec_()


    def open_profile(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox, QFileDialog, QLineEdit, QHBoxLayout, QPushButton
        dlg = QDialog(self)
        dlg.setWindowTitle(self._tr("profile_title"))
        dlg.setMinimumSize(320, 400)
        pw = self.cfg.get("profile_w", 360)
        ph = self.cfg.get("profile_h", 480)
        dlg.resize(pw, ph)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(12)
        lay.addWidget(QLabel(f"<h2>{self._tr('profile_title')}</h2>"))
        # 头像
        big = QLabel()
        avatar_path = self.cfg.get("avatar_path", "")
        if avatar_path and os.path.exists(avatar_path):
            pix = QPixmap(avatar_path).scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            big.setPixmap(pix)
            big.setStyleSheet("border: 2px solid #00f5d4; border-radius: 40px;")
        else:
            big.setText("\u266a")
            big.setStyleSheet("background: rgba(0,245,212,0.2); border: 2px solid #00f5d4; border-radius: 40px; color: #00f5d4; font-size: 32px; font-weight: bold;")
        def change_avatar():
            path, _ = QFileDialog.getOpenFileName(dlg, self._tr("select_avatar"), "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif)")
            if path:
                pix = QPixmap(path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                local_avatar = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_avatar.jpg")
                pix.save(local_avatar, "JPG", quality=70)
                self.cfg["avatar_path"] = local_avatar
                save_config(self.cfg)
                big.setPixmap(pix)
                big.setStyleSheet("border: 2px solid #00f5d4; border-radius: 40px;")
                self.btn_avatar.setIcon(QIcon(local_avatar))
                self.btn_avatar.setText("")
        big.mousePressEvent = lambda e: change_avatar()
        lay.addWidget(big, 0, Qt.AlignCenter)
        tip = QLabel(self._tr("profile_click_avatar"))
        tip.setStyleSheet("color: rgba(200,255,232,0.5); font-size: 11px;")
        tip.setAlignment(Qt.AlignCenter)
        lay.addWidget(tip)
        # 昵称（本地保存）
        nick_row = QHBoxLayout()
        nick_row.addWidget(QLabel(self._tr("nickname_label")))
        nick_edit = QLineEdit(self.cfg.get("nickname", ""))
        nick_edit.setPlaceholderText(self._tr("nick_placeholder"))
        nick_row.addWidget(nick_edit)
        btn_save_nick = QPushButton(self._tr("save"))
        btn_save_nick.setCursor(Qt.PointingHandCursor)
        def do_save_nick():
            self.cfg["nickname"] = nick_edit.text().strip()
            save_config(self.cfg)
        btn_save_nick.clicked.connect(do_save_nick)
        nick_row.addWidget(btn_save_nick)
        lay.addLayout(nick_row)
        lay.addWidget(QLabel(self._tr("fav_count").format(n=len(self.cfg.get('favorites', [])))))
        lay.addWidget(QLabel(self._tr("pl_count").format(n=len(self.cfg.get('playlists', {})))))

        btns = QDialogButtonBox()
        btn_close = btns.addButton(self._tr("close"), QDialogButtonBox.AcceptRole)
        btn_close.clicked.connect(dlg.accept)
        lay.addWidget(btns)
        def on_finish():
            self.cfg["profile_w"] = dlg.width()
            self.cfg["profile_h"] = dlg.height()
            save_config(self.cfg)
        dlg.finished.connect(on_finish)
        tkey = self.cfg.get("theme", "cyberpunk")
        bg = "#0f2018" if tkey == "forest" else "#0a0a14"
        dlg.setStyleSheet(f"""
            QDialog {{ background: {bg}; color: #c8ffe8; border: 1px solid #00f5d4; border-radius: 12px; }}
            QLabel {{ color: #c8ffe8; }}
            QPushButton {{ background: rgba(255,255,255,0.08); color: #c8ffe8; border: 1px solid rgba(255,255,255,0.15); padding: 6px 20px; border-radius: 6px; }}
            QPushButton:hover {{ background: rgba(0,245,212,0.2); }}
        """)
        dlg.exec_()


    def toggle_menu(self, show=None):
        # 直接在按钮下方弹出
        gp = self.btn_menu.mapToGlobal(self.btn_menu.rect().bottomRight())
        self._menu.popup(gp)

    def show_notice(self):
        """右上角公告栏：弹出公告列表（每次点击实时读取 config.json 的 announcements）。"""
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                items = json.load(f).get("announcements", [])
        except Exception:
            items = self.cfg.get("announcements", [])
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background: #141422; color: #d8e6e0; border: 1px solid rgba(0,255,200,0.25); border-radius: 10px; padding: 8px; min-width: 300px; } QMenu::item { padding: 8px 14px; border-radius: 6px; }")
        if not items:
            a = menu.addAction(self._tr("notice_empty"))
            a.setEnabled(False)
        else:
            for i, text in enumerate(items):
                disp = self._tr("notice_default") if text == DEFAULT_NOTICE else text
                a = menu.addAction(f"{i + 1}.  {disp}")
                a.setEnabled(False)
        gp = self.btn_notice.mapToGlobal(self.btn_notice.rect().bottomLeft())
        menu.exec_(gp)

    def _set_lang(self, mode):
        self.lyric_view.show_mode = mode
        if hasattr(self, "btn_lang_orig"):
            self.btn_lang_orig.setChecked(mode == "orig")
            self.btn_lang_trans.setChecked(mode == "trans")
            self.btn_lang_both.setChecked(mode == "both")
        self.lyric_view.update()

    def _on_lyrics_ready(self, orig, trans):
        self.lyric_view.set_lines(orig, trans or [])

    def _show_speed_menu(self):
        if hasattr(self, "_speed_popup") and self._speed_popup.isVisible():
            self._speed_popup.hide()
            return
        from PyQt5.QtWidgets import QGraphicsDropShadowEffect
        pop = QFrame(self)
        pop.setObjectName("speedPopup")
        lay = QHBoxLayout(pop)
        lay.setContentsMargins(8,8,8,8)
        lay.setSpacing(4)
        speeds = [round(0.5 + i*0.1, 1) for i in range(16)]
        for s in speeds:
            b = QPushButton(f"{s:.1f}")
            b.setFixedHeight(32)
            b.setCursor(Qt.PointingHandCursor)
            is_cur = abs(s - self._speed) < 0.01
            bg = "#fa2d48" if is_cur else "transparent"
            fg = "#fff" if is_cur else "#ccc"
            b.setStyleSheet(f"QPushButton {{ background:{bg}; color:{fg}; border:none; border-radius:6px; padding:0 8px; font-size:12px; }} QPushButton:hover {{ background:#444; }}")
            b.clicked.connect(lambda _, v=s, p=pop: (self._set_speed(v), p.hide()))
            lay.addWidget(b)
        pop.adjustSize()
        # 弹出在按钮右侧（而不是右下方）
        gp = self.btn_speed.mapToGlobal(self.btn_speed.rect().topRight())
        pop.move(gp.x() + 6, gp.y())
        # 防超出屏幕右缘
        from PyQt5.QtWidgets import QApplication as _QA
        scr = _QA.desktop().availableGeometry(self)
        if pop.x() + pop.width() > scr.right():
            pop.move(gp.x() - pop.width() - 6, gp.y())
        pop.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        pop.setAttribute(Qt.WA_DeleteOnClose)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0,0,0,180))
        shadow.setOffset(0,4)
        pop.setGraphicsEffect(shadow)
        self._speed_popup = pop
        pop.show()

    def _set_volume(self, v):
        self.audio.set_volume(v / 100.0)
        self.cfg["volume"] = v

    def _set_speed(self, v):
        self._speed = v
        self.btn_speed.setText(f"{v:g}x")
        self.cfg["playback_speed"] = v
        save_config(self.cfg)
        try: self.audio.set_speed(v)
        except: pass
    def _on_splitter_moved(self, pos, index):
        # 收起状态下用户拖分隔条，自动展开
        if not self._list_expanded and pos > 20:
            self.toggle_list()
        # 记忆分隔条位置
        sizes = self.splitter.sizes()
        if len(sizes) == 2 and sizes[0] > 0:
            self.cfg["list_panel_w"] = sizes[0]
            save_config(self.cfg)

    def toggle_list(self):
        if self._list_expanded:
            # 收起：直接隐藏左栏，handle 消失，右侧完全贴左
            self._list_expanded = False
            self.btn_toggle_list.setText(self._tr("expand_list"))
            self.list_container.hide()
            self.splitter.setSizes([0, self.splitter.width()])
        else:
            # 展开：显示左栏并恢复到 360
            self._list_expanded = True
            self.btn_toggle_list.setText(self._tr("collapse_list"))
            self.list_container.setMinimumWidth(360)
            self.list_container.show()
            self.list_widget.show()
            self.splitter.setHandleWidth(2)
            w = self.splitter.width()
            self.splitter.setSizes([min(360, int(w * 0.4)), w - min(360, int(w * 0.4))])

    def closeEvent(self, e):
        try:
            self.cfg["window_width"] = self.width()
            self.cfg["window_height"] = self.height()
            self.cfg["window_x"] = self.x()
            self.cfg["window_y"] = self.y()
            save_config(self.cfg)
        except Exception:
            pass
        super().closeEvent(e)

    def choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, self._tr("select_music_dir"))
        if d:
            self.scan_dir(d)

    def _refresh_pl_combo(self):
        """填充歌单下拉框：全部歌曲 / 收藏 / 自定义歌单列表。"""
        self.pl_combo.blockSignals(True)
        self.pl_combo.clear()
        self.pl_combo.addItem(self._tr("all_songs"), "all")
        self.pl_combo.addItem(self._tr("favorites"), "fav")
        for name in self.cfg.get("playlists", {}).keys():
            self.pl_combo.addItem(name, name)
        # 恢复上次选中
        cur = self.cfg.get("current_playlist", "all")
        idx = 0
        for i in range(self.pl_combo.count()):
            if self.pl_combo.itemData(i) == cur:
                idx = i
                break
        self.pl_combo.setCurrentIndex(idx)
        self.pl_combo.blockSignals(False)

    def _current_pl_songs(self):
        """根据当前选中的歌单，返回该歌单包含的歌曲路径集合（None=全部）。"""
        key = self.pl_combo.currentData()
        if key == "fav":
            return set(self.cfg.get("favorites", []))
        if key and key != "all":
            return set(self.cfg.get("playlists", {}).get(key, []))
        return None   # 全部

    def _refresh_list(self):
        """按当前歌单和搜索词筛选显示歌曲。"""
        self.list_widget.clear()
        self._visible = []
        pl_set = self._current_pl_songs()
        kw = self.search_box.text().strip().lower() if hasattr(self, "search_box") else ""
        for i, s in enumerate(self.songs):
            if pl_set is not None and s.path not in pl_set:
                continue
            if kw and kw not in s.display_name.lower():
                continue
            mark = "★ " if s.path in self.cfg.get("favorites", []) else ""
            QListWidgetItem(mark + s.display_name, self.list_widget)
            self._visible.append(i)
        self._refresh_count()

    def _filter_songs(self, _):
        self._refresh_list()

    def scan_dir(self, root):
        self.songs = scan_music(root)
        self._refresh_pl_combo()
        self._refresh_list()
        self.cfg["last_dir"] = root
        save_config(self.cfg)

    def _on_pl_changed(self, _idx):
        self.cfg["current_playlist"] = self.pl_combo.currentData()
        save_config(self.cfg)
        self._refresh_list()

    def _create_playlist(self):
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle(self._tr("new_playlist"))
            dlg.setFixedWidth(320)
            dlg.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
            dlg.setAttribute(Qt.WA_StyledBackground, True)
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(20, 16, 20, 16)
            lay.setSpacing(10)
            # 标题行
            title_row = QHBoxLayout()
            title_row.addWidget(QLabel(self._tr("new_playlist")))
            title_row.addStretch()
            close_btn = QPushButton("✕")
            close_btn.setFixedSize(24, 24)
            close_btn.setCursor(Qt.PointingHandCursor)
            close_btn.clicked.connect(dlg.reject)
            title_row.addWidget(close_btn)
            lay.addLayout(title_row)
            lay.addWidget(QLabel(self._tr("playlist_name")))
            edit = QLineEdit()
            lay.addWidget(edit)
            btns = QDialogButtonBox()
            ok_btn = btns.addButton(self._tr("save"), QDialogButtonBox.AcceptRole)
            cancel_btn = btns.addButton(self._tr("cancel"), QDialogButtonBox.RejectRole)
            ok_btn.clicked.connect(dlg.accept)
            cancel_btn.clicked.connect(dlg.reject)
            lay.addWidget(btns)
            tkey = self.cfg.get("theme", "cyberpunk")
            if tkey == "forest":
                bg = "#0f2018"; fg = "#d4f5e9"; accent = "#38ef7d"
            else:
                bg = "#0a0a14"; fg = "#c8ffe8"; accent = "#00f5d4"
            dlg.setStyleSheet(f"""
                QDialog {{ background: {bg}; color: {fg}; border: 1px solid {accent}; border-radius: 10px; }}
                QLineEdit {{ background: rgba(255,255,255,0.08); color: {fg}; border: 1px solid rgba(255,255,255,0.15); padding: 6px; border-radius: 6px; }}
                QPushButton {{ background: rgba(255,255,255,0.08); color: {fg}; border: 1px solid rgba(255,255,255,0.15); padding: 6px 16px; border-radius: 6px; }}
                QPushButton:hover {{ background: rgba(0,245,212,0.2); }}
            """)
            if dlg.exec_() != QDialog.Accepted:
                return
            name = edit.text().strip()
            if not name:
                return
            pls = self.cfg.get("playlists", {})
            if name in pls:
                return
            pls[name] = []
            self.cfg["playlists"] = pls
            save_config(self.cfg)
            self._refresh_pl_combo()
            for i in range(self.pl_combo.count()):
                if self.pl_combo.itemData(i) == name:
                    self.pl_combo.setCurrentIndex(i)
                    break
        except Exception as e:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, self._tr("error"), self._tr("new_playlist_fail").format(err=e))

    def _on_list_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item or not self._visible:
            return
        row = self.list_widget.row(item)
        if row >= len(self._visible):
            return
        song_idx = self._visible[row]
        song = self.songs[song_idx]
        menu = QMenu(self)
        # 收藏切换
        favs = self.cfg.get("favorites", [])
        is_fav = song.path in favs
        act_fav = menu.addAction(self._tr("cancel_fav") if is_fav else self._tr("favorites"))
        # 加入歌单子菜单
        sub = menu.addMenu(self._tr("add_to_playlist"))
        pls = self.cfg.get("playlists", {})
        if not pls:
            sub.addAction(self._tr("no_custom_pl")).setEnabled(False)
        for name in pls.keys():
            in_pl = song.path in pls.get(name, [])
            sub.addAction(("✓ " if in_pl else "") + name)
        sub.addSeparator()
        act_new = sub.addAction(self._tr("new_pl_action"))
        chosen = menu.exec_(self.list_widget.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == act_fav:
            if is_fav:
                favs.remove(song.path)
            else:
                favs.append(song.path)
            self.cfg["favorites"] = favs
            save_config(self.cfg)
            self._refresh_list()
        elif chosen == act_new:
            self._create_playlist()
        elif sub.actions().count() > 0:
            # 找到选中的歌单
            for name, pl_songs in pls.items():
                if chosen.text().lstrip("✓ ") == name:
                    if song.path in pl_songs:
                        pl_songs.remove(song.path)
                    else:
                        pl_songs.append(song.path)
                    self.cfg["playlists"][name] = pl_songs
                    save_config(self.cfg)
                    self._refresh_list()
                    break

    def on_item_double(self, item):
        row = self.list_widget.row(item)
        if row < len(self._visible):
            self.play_index(self._visible[row])

    def play_index(self, idx):
        if not (0 <= idx < len(self.songs)):
            return
        self.current_index = idx
        song = self.songs[idx]
        ok = True
        # 使用预加载缓存
        cached = getattr(self, "_preload_cache", None)
        if cached and cached[0] == song.path:
            try:
                self.audio._orig = cached[1]
                self.audio._orig_sr = cached[2]
                self.audio._pos = 0
                self.audio.set_speed(self._speed)
                self.audio.play()
            except Exception:
                ok = False
        else:
            try:
                self.audio.load(song.path)
                self.audio.set_speed(self._speed)
                self.audio.play()
            except Exception:
                ok = False
        if not ok:
            self.lbl_title.setText(self._tr("skip_msg").format(name=os.path.basename(song.path)))
            QTimer.singleShot(600, self.play_next)
            return
        # 记录播放历史
        hist = self.cfg.get("play_history", [])
        if song.path in hist:
            hist.remove(song.path)
        hist.insert(0, song.path)
        self.cfg["play_history"] = hist[:50]  # 只留最近50首
        save_config(self.cfg)
        # 桌面通知
        if self.tray.isVisible() and self.cfg.get("notify_enabled", True):
            self.tray.showMessage(f"♪ {song.title or os.path.basename(song.path)}",
                                  song.artist or "", QSystemTrayIcon.Information, 2000)
        # 后台预加载下一首
        self._start_preload(idx + 1 if idx + 1 < len(self.songs) else 0)
        self._offset = 0.0
        self._paused = False
        self._duration = self.audio.duration
        self.slider.setRange(0, int(self._duration * 1000))
        self.lbl_total.setText(self._fmt_time(self._duration))
        self.lbl_title.setText(song.title or os.path.basename(song.path))
        self.lbl_artist.setText(song.artist or self._tr("unknown_artist"))
        self.list_widget.setCurrentRow(idx)
        self.btn_play.kind = "pause"; self.btn_play.update()
        self.lyric_view.set_lines([], [])
        if self._lyric_worker is not None:
            try:
                self._lyric_worker.done.disconnect(self._on_lyrics_ready)
            except Exception:
                pass
            self._lyric_worker.wait(50)
        self._lyric_worker = LyricsWorker(song)
        self._lyric_worker.done.connect(self._on_lyrics_ready)
        self._lyric_worker.start()

    def toggle_play(self):
        if self.current_index < 0:
            if self.songs:
                self.play_index(0)
            return
        if self._paused:
            self.audio.unpause(); self._paused = False; self.btn_play.kind = "pause"; self.btn_play.update()
        else:
            self.audio.pause(); self._paused = True; self.btn_play.kind = "play"; self.btn_play.update()

    def cycle_mode(self):
        order = [MODE_ORDER, MODE_LOOP, MODE_RANDOM]
        self.mode = order[(order.index(self.mode) + 1) % len(order)]
        self._refresh_mode_label()

    def _start_preload(self, idx):
        """后台预加载第idx首歌。"""
        if not (0 <= idx < len(self.songs)):
            return
        path = self.songs[idx].path
        if getattr(self, "_preload_path", "") == path:
            return
        self._preload_path = path
        self._preload_cache = None
        worker = PreloadWorker(path)
        def on_done(data, sr):
            if data is not None:
                self._preload_cache = (path, data, sr)
        worker.done.connect(on_done)
        worker.start()
        self._preload_worker = worker

    def play_next(self):
        if not self.songs:
            return
        if self.mode == MODE_RANDOM:
            self.play_index(random.randrange(len(self.songs)))
        else:
            self.play_index((self.current_index + 1) % len(self.songs))

    def play_prev(self):
        if not self.songs:
            return
        if self.mode == MODE_RANDOM:
            self.play_index(random.randrange(len(self.songs)))
        else:
            self.play_index((self.current_index - 1) % len(self.songs))

    def _on_slider_moved(self, value):
        # 拖动中：实时显示目标时间，并防抖触发真实 seek（边拖边播）
        self.lbl_cur.setText(self._fmt_time(value / 1000.0))
        self._seek_timer.start()

    def _on_seek(self):
        # 松手立即跳一次，不等防抖
        self._seek_timer.stop()
        self._do_seek()

    def _do_seek(self):
        if self.current_index < 0:
            self._seeking = False; return
        target = self.slider.value() / 1000.0
        try:
            self.audio.seek(target)
        except Exception:
            pass
        self._seeking = False

    def _tick(self):
        if self.current_index < 0:
            return
        if self.audio.at_end and not self._paused:
            if self.mode == MODE_LOOP:
                self.play_index(self.current_index)
            else:
                self.play_next()
            return
        real_sec = self.audio.position
        if not self._seeking:
            self.slider.setValue(int(real_sec * 1000))
        self.lbl_cur.setText(self._fmt_time(real_sec))
        self.lyric_view.update_time(real_sec - self.cfg.get("lyric_offset", 0.0))
        # 同步桌面歌词
        if self.cfg.get("desktop_lyrics", False) and hasattr(self, "_dlrc_label"):
            cur = self.lyric_view._current_text() if hasattr(self.lyric_view, "_current_text") else ""
            self._update_dlrc(cur or self._tr("dlrc_playing"))
        # 频谱可视化：从音频数据取一段采样算能量
        if hasattr(self, "spectrum") and self.cfg.get("spectrum_enabled", True):
            try:
                import numpy as np
                pos = int(self.audio._pos)
                data = self.audio._orig
                chunk = data[pos:pos + 4096] if data is not None and pos < len(data) else None
                if chunk is not None and len(chunk) > 100:
                    mono = chunk.mean(axis=1) if chunk.ndim > 1 else chunk
                    n = len(mono)
                    seg = n // 128
                    levels = []
                    for i in range(128):
                        s = mono[i * seg:(i + 1) * seg]
                        levels.append(min(1.0, float(np.abs(s).mean()) * 25))
                    self.spectrum.set_levels(levels)
            except Exception:
                pass

    @staticmethod
    def _fmt_time(sec):
        sec = max(0, int(sec))
        return f"{sec // 60:02d}:{sec % 60:02d}"

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_sizes_set", False) and self.splitter.width() > 400:
            self._sizes_set = True
            self.splitter.setHandleWidth(2)
            w = self.cfg.get("list_panel_w", 360)
            self.splitter.setSizes([w, self.splitter.width() - w])

    def _import_skin(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, self._tr("select_skin"), "", "QSS样式 (*.qss)")
        if path:
            try:
                if os.path.getsize(path) > 1024 * 1024:
                    QMessageBox.warning(self, self._tr("error"), self._tr("skin_too_large")); return
                with open(path, encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
            except UnicodeDecodeError:
                QMessageBox.warning(self, self._tr("error"), self._tr("skin_utf8_required"))
            except Exception as e:
                QMessageBox.warning(self, self._tr("error"), self._tr("skin_load_fail").format(err=e))

    def _show_history(self):
        from PyQt5.QtWidgets import QDialog, QListWidget, QVBoxLayout
        dlg = QDialog(self)
        dlg.setWindowTitle(self._tr("history_title"))
        dlg.setMinimumSize(350, 400)
        dw = self.cfg.get("history_w", 400)
        dh = self.cfg.get("history_h", 500)
        dlg.resize(dw, dh)
        lay = QVBoxLayout(dlg)
        lw = QListWidget()
        hist = self.cfg.get("play_history", [])
        song_map = {s.path: s.display_name for s in self.songs}
        for p in hist:
            name = song_map.get(p, os.path.basename(p))
            lw.addItem(name)
        lw.itemDoubleClicked.connect(lambda it: self.play_index(self._visible.index(next((i for i,s in enumerate(self.songs) if s.path == hist[lw.row(it)]), 0))))
        lay.addWidget(lw)
        dlg.finished.connect(lambda: (self.cfg.__setitem__("history_w", dlg.width()), self.cfg.__setitem__("history_h", dlg.height()), save_config(self.cfg)))
        dlg.exec_()

    def _show_eq(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QSlider, QLabel
        dlg = QDialog(self)
        dlg.setWindowTitle(self._tr("eq_title"))
        dlg.setMinimumSize(600, 350)
        lay = QVBoxLayout(dlg)
        freqs = ["60Hz", "250Hz", "1kHz", "4kHz", "16kHz"]
        for i, f in enumerate(freqs):
            row = QHBoxLayout()
            row.addWidget(QLabel(f))
            s = QSlider(Qt.Horizontal)
            s.setRange(-12, 12)
            s.setMinimumWidth(300)
            s.setValue(self.cfg.get(f"eq_{i}", 0))
            lbl = QLabel(f"{self.cfg.get(f'eq_{i}', 0):+d}dB")
            s.valueChanged.connect(lambda v, idx=i, l=lbl: (self.cfg.__setitem__(f"eq_{idx}", v), l.setText(f"{v:+d}dB"), save_config(self.cfg)))
            row.addWidget(s)
            row.addWidget(lbl)
            lay.addLayout(row)
        dlg.exec_()

    def _setup_hotkeys(self):
        for sc in getattr(self, "_hotkeys", []):
            sc.setEnabled(False)
            sc.deleteLater()
        self._hotkeys = []
        # 字母数字键
        key_map = {}
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            key_map[c] = getattr(Qt, "Key_" + c)
        key_map.update({"Space": Qt.Key_Space, "Right": Qt.Key_Right, "Left": Qt.Key_Left,
                        "Up": Qt.Key_Up, "Down": Qt.Key_Down, "Enter": Qt.Key_Return,
                        "F1": Qt.Key_F1, "F2": Qt.Key_F2, "F3": Qt.Key_F3})
        for action, key_name in [("play", self.cfg.get("hotkey_play", "Space")),
                                  ("next", self.cfg.get("hotkey_next", "Right")),
                                  ("prev", self.cfg.get("hotkey_prev", "Left"))]:
            k = key_map.get(key_name, Qt.Key_Space)
            cb = {"play": self.toggle_play, "next": self.play_next, "prev": self.play_prev}[action]
            sc = QShortcut(QKeySequence(k), self, activated=cb)
            self._hotkeys.append(sc)

    def _show_hotkey_settings(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
        from PyQt5.QtCore import Qt as _Qt
        dlg = QDialog(self)
        dlg.setWindowTitle(self._tr("hotkey_title"))
        dlg.setMinimumSize(450, 250)
        lay = QVBoxLayout(dlg)
        key_names = {"Space": self._tr("key_space"), "Right": self._tr("key_right"), "Left": self._tr("key_left"),
                     "Up": self._tr("key_up"), "Down": self._tr("key_down"), "Enter": self._tr("key_enter"),
                     "F1": "F1", "F2": "F2", "F3": "F3"}
        fields = [(self._tr("hotkey_play"), "hotkey_play"), (self._tr("hotkey_next"), "hotkey_next"), (self._tr("hotkey_prev"), "hotkey_prev")]
        buttons = {}
        capturing = {"field": None, "btn": None}

        class CaptureBtn(QPushButton):
            def __init__(self, text, parent=None):
                super().__init__(text, parent)
                self.capture_cb = None
            def keyPressEvent(self, e):
                if self.capture_cb:
                    self.capture_cb(e.key())

        def start_capture(field, btn):
            capturing["field"] = field
            capturing["btn"] = btn
            btn.setText(self._tr("press_any_key"))
            btn.setStyleSheet("QPushButton { background: #00f5d4; color: #000; font-weight: bold; border: 2px solid #fff; }")
            btn.setFocus()
            btn.setFocusPolicy(_Qt.StrongFocus)

        def on_key(key):
            # 支持字母数字键 A-Z, 0-9
            name = ""
            if 0x41 <= key <= 0x5A:  # A-Z
                name = chr(key)
            elif 0x30 <= key <= 0x39:  # 0-9
                name = chr(key)
            else:
                key_map = {0x01000020: "Space", 0x01000013: "Right", 0x01000012: "Left",
                           0x01000015: "Up", 0x01000014: "Down", 0x01000004: "Enter",
                           0x01000030: "F1", 0x01000031: "F2", 0x01000032: "F3"}
                name = key_map.get(key, "")
            if name and capturing["field"]:
                self.cfg[capturing["field"]] = name
                btn = capturing["btn"]
                btn.setText(name)
                btn.setStyleSheet("")
                capturing["field"] = None

        tip = QLabel(self._tr("hotkey_tip"))
        tip.setStyleSheet("color:#888;")
        lay.addWidget(tip)

        for label, key in fields:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addStretch(1)
            cur = self.cfg.get(key, "Space")
            btn = CaptureBtn(key_names.get(cur, cur))
            btn.setMinimumWidth(150)
            btn.setFocusPolicy(_Qt.StrongFocus)
            btn.capture_cb = on_key
            btn.clicked.connect(lambda checked=False, f=key, b=btn: start_capture(f, b))
            row.addWidget(btn)
            lay.addLayout(row)
            buttons[key] = btn

        save_btn = QPushButton(self._tr("save"))
        save_btn.setMinimumHeight(36)
        save_btn.clicked.connect(lambda: (save_config(self.cfg), self._setup_hotkeys(), dlg.accept()))
        lay.addWidget(save_btn)
        dlg.exec_()


    def _toggle_desktop_lyrics(self, show):
        if show:
            if not hasattr(self, "_desktop_lrc"):
                from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
                self._desktop_lrc = QWidget(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
                self._desktop_lrc.setAttribute(Qt.WA_TranslucentBackground)
                self._desktop_lrc.setGeometry(self.cfg.get("dlrc_x", 200), self.cfg.get("dlrc_y", 100), 800, 60)
                lay = QVBoxLayout(self._desktop_lrc)
                self._dlrc_label = QLabel("")
                self._apply_dlrc_style()
                self._dlrc_label.setAlignment(Qt.AlignCenter)
                lay.addWidget(self._dlrc_label)
                self._dlrc_drag = {"active": False, "x": 0, "y": 0}
                def mouse_press(e):
                    if e.button() == Qt.LeftButton:
                        self._dlrc_drag = {"active": True, "x": e.globalX() - self._desktop_lrc.x(), "y": e.globalY() - self._desktop_lrc.y()}
                def mouse_move(e):
                    if self._dlrc_drag["active"]:
                        self._desktop_lrc.move(e.globalX() - self._dlrc_drag["x"], e.globalY() - self._dlrc_drag["y"])
                def mouse_release(e):
                    self._dlrc_drag["active"] = False
                    self.cfg["dlrc_x"] = self._desktop_lrc.x()
                    self.cfg["dlrc_y"] = self._desktop_lrc.y()
                    save_config(self.cfg)
                self._dlrc_label.mousePressEvent = mouse_press
                self._dlrc_label.mouseMoveEvent = mouse_move
                self._dlrc_label.mouseReleaseEvent = mouse_release
            self._desktop_lrc.show()
        else:
            if hasattr(self, "_desktop_lrc"):
                self._desktop_lrc.hide()

    def _apply_dlrc_style(self):
        if not hasattr(self, "_dlrc_label"):
            return
        fs = self.cfg.get("dlrc_font_size", self.cfg.get("lyric_font_size", 20) + 4)
        ff = self.cfg.get("dlrc_font_family", "Microsoft YaHei")
        mode = self.cfg.get("dlrc_color_mode", 0)
        colors = ["#00f5d4", "#ff2d55", "#ffcc00", "#34c759", "#5ac8fa", "#ff9500", "#af52de", "#ffffff", "#00f5d4"]
        c = colors[mode] if mode < len(colors) else "#00f5d4"
        self._dlrc_label.setStyleSheet(f"color:{c}; font-size:{fs}px; font-family:'{ff}'; font-weight:bold; background:transparent;")
        self._dlrc_label.update()

    def _update_dlrc(self, text):
        if hasattr(self, "_dlrc_label"):
            self._dlrc_label.setText(text if text else "")

    def _show_about(self):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.about(self, self._tr("menu_about"),
            f"<h3>Starry Flight</h3>"
            f"<p>{self._tr('about_version')}</p>"
            f"<p>© 2026 Starry Flight</p>"
            f"<p>{self._tr('about_desc')}</p>")

    def _show_privacy(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QCheckBox, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle(self._tr("menu_privacy"))
        dlg.setMinimumSize(550, 450)
        lay = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setHtml(self._tr("privacy_text"))
        lay.addWidget(txt)
        chk = QCheckBox(self._tr("privacy_agree"))
        lay.addWidget(chk)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_btn = btns.button(QDialogButtonBox.Ok)
        ok_btn.setText(self._tr("privacy_ok"))
        ok_btn.setEnabled(False)
        chk.toggled.connect(ok_btn.setEnabled)
        btns.button(QDialogButtonBox.Cancel).setText(self._tr("privacy_cancel"))
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        result = dlg.exec_()
        if result != QDialog.Accepted:
            from PyQt5.QtWidgets import QApplication
            QApplication.quit()
        return result == QDialog.Accepted

    def _check_update(self):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, self._tr("menu_update"), self._tr("up_to_date"))

    def _tray_quit(self):
        self.tray.hide()
        QApplication.quit()

    def closeEvent(self, e):
        # 点关闭按钮时缩到托盘，不真正退出
        if hasattr(self, "tray") and self.tray.isVisible():
            e.ignore()
            self.hide()
            self.tray.showMessage("Starry Flight", self._tr("tray_minimized"), QSystemTrayIcon.Information, 2000)
            return
        try:
            self._stop_video_bg()
        except Exception:
            pass
        try:
            self.audio.stop()
        except Exception:
            pass
        try:
            if self._lyric_worker is not None:
                self._lyric_worker.wait(200)
        except Exception:
            pass
        try:
            self.cfg["window_width"] = self.width()
            self.cfg["window_height"] = self.height()
            self.cfg["window_x"] = self.x()
            self.cfg["window_y"] = self.y()
            save_config(self.cfg)
        except Exception:
            pass
        super().closeEvent(e)


QT_LANG_MAP = {"zh": "zh_CN", "ja": "ja", "fra": "fr", "de": "de", "kor": "ko", "ru": "ru"}

def install_qt_translator(lang):
    """让 Qt 内置组件（滚动条右键菜单等）文案跟随界面语言。
    加载 Qt 自带翻译文件（qt_zh_CN.qm / qt_ja.qm ...），无匹配语种时回退英文。"""
    app = QApplication.instance()
    if app is None:
        return
    for tr in getattr(app, "_qt_translators", []):
        app.removeTranslator(tr)
    app._qt_translators = []
    code = QT_LANG_MAP.get(lang)
    if not code:
        return
    try:
        tr = QTranslator()
        path = QLibraryInfo.location(QLibraryInfo.TranslationsPath)
        if tr.load(f"qt_{code}", path):
            app.installTranslator(tr)
            app._qt_translators.append(tr)
    except Exception:
        pass


class LanguageSelectDialog(QDialog):
    """首次启动语言选择窗口：列出七种语言单选，默认简体中文。
    确认后返回所选语言；取消/关闭兜底为简体中文并写入配置。
    风格与主程序一致（深色背景、灰白文字）。"""
    LANG_ITEMS = [
        ("zh", "简体中文"),
        ("en", "English"),
        ("ja", "日本語"),
        ("fra", "Français"),
        ("de", "Deutsch"),
        ("kor", "한국어"),
        ("ru", "Русский"),
    ]

    def __init__(self, cfg=None, parent=None):
        super().__init__(parent)
        self._selected = "zh"
        self._lang_codes = [code for code, _ in self.LANG_ITEMS]
        self.setWindowTitle("选择界面语言")
        self.setModal(True)
        self.setFixedSize(380, 340)
        self.setStyleSheet("""
QDialog { background: #141422; color: #d8e6e0; }
QLabel { color: #d8e6e0; }
QListWidget { background: #1b1b2e; color: #d8e6e0; border: 1px solid rgba(0,255,200,0.25); border-radius: 8px; outline: none; }
QListWidget::item { padding: 8px 12px; border-radius: 4px; }
QListWidget::item:hover { background: rgba(0,255,200,0.08); }
QListWidget::item:selected { background: #00f5d4; color: #141422; font-weight: bold; }
QPushButton { background: #1f1f36; color: #d8e6e0; border: 1px solid rgba(0,255,200,0.25); border-radius: 6px; padding: 6px 18px; }
QPushButton:hover { background: #00f5d4; color: #141422; }
""")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 16)
        lay.setSpacing(12)
        title = QLabel("选择界面语言")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        lay.addWidget(title)
        tip = QLabel("首次启动请选择界面语言，之后可在设置中随时更改")
        tip.setStyleSheet("color: #9aa7b8; font-size: 12px;")
        lay.addWidget(tip)
        self.list_widget = QListWidget()
        for _, name in self.LANG_ITEMS:
            self.list_widget.addItem(name)
        self.list_widget.setCurrentRow(0)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        lay.addWidget(self.list_widget)
        btns = QDialogButtonBox()
        ok_btn = btns.addButton("确定", QDialogButtonBox.AcceptRole)
        btns.addButton("取消", QDialogButtonBox.RejectRole)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self._ok_btn = ok_btn
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

    def _on_row_changed(self, row):
        if 0 <= row < len(self._lang_codes):
            self._selected = self._lang_codes[row]

    def selected_lang(self):
        """返回当前选择的语言代码；未确认（取消/关闭）时为默认 zh。"""
        return self._selected


def main():
    # 全局异常捕获，写日志不闪退
    def _excepthook(etype, evalue, tb):
        import traceback
        log.error("未捕获异常", exc_info=(etype, evalue, tb))
        traceback.print_exception(etype, evalue, tb)
    sys.excepthook = _excepthook

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # 首次启动语言选择：从未设置过界面语言（lang_selected 标记缺失/False）时弹窗
    cfg = load_config()
    if not cfg.get("lang_selected", False):
        dlg = LanguageSelectDialog(cfg)
        dlg.exec_()
        cfg["language"] = dlg.selected_lang()   # 取消/关闭兜底保持简体中文
        cfg["lang_selected"] = True
        save_config(cfg)
    w = MainWindow()
    install_qt_translator(w.cfg.get("language", "zh"))
    # 首次启动必须同意隐私政策才能使用
    if not w.cfg.get("privacy_accepted", False):
        if not w._show_privacy():
            sys.exit(0)
        w.cfg["privacy_accepted"] = True
        save_config(w.cfg)
    w.showNormal()
    w.raise_()
    w.activateWindow()
    w.setWindowState(Qt.WindowNoState)
    w.move(100, 100)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


