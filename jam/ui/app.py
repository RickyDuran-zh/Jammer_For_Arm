"""PyQt5 控制台：实时频谱、信道评分、跟踪轨迹、Agent 对话。"""
from __future__ import annotations

import html
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from ..agent.brain import JamAgent
from ..config import (POWER_MAX_DB, POWER_MIN_DB, WAVEFORM_NAMES, JamConfig)
from ..controller import JamController

BG = "#0b0f14"
PANEL = "#121821"
BORDER = "#1f2a37"
TEXT = "#d7e0ea"
MUTED = "#8593a5"
ACCENT = "#2dd4bf"
AMBER = "#f59e0b"
RED = "#ef4444"
GREEN = "#22c55e"

QSS = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: "Microsoft YaHei", "Segoe UI"; }}
QFrame#Card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
QLabel#CardTitle {{ color: {MUTED}; font-size: 12px; }}
QLabel#CardValue {{ color: {TEXT}; font-size: 20px; font-weight: 500; }}
QLabel#Title {{ color: {TEXT}; font-size: 20px; font-weight: 500; }}
QLabel#Subtitle {{ color: {MUTED}; font-size: 12px; }}
QLabel#Badge {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 9px;
                padding: 4px 10px; color: {ACCENT}; font-size: 12px; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
             margin-top: 12px; padding: 10px; font-size: 12px; color: {MUTED}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 6px 14px; }}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QTextBrowser {{ background: #0e141c; border: 1px solid {BORDER}; border-radius: 8px;
                color: {TEXT}; font-size: 13px; padding: 6px; }}
QDoubleSpinBox {{ background: #0e141c; border: 1px solid {BORDER}; border-radius: 6px;
                  padding: 4px; color: {TEXT}; font-size: 12px; }}
QLineEdit {{ background: #0e141c; border: 1px solid {BORDER}; border-radius: 8px;
             padding: 8px; color: {TEXT}; font-size: 13px; }}
QPushButton {{ background: #17202b; border: 1px solid {BORDER}; border-radius: 7px;
               padding: 6px 10px; color: {TEXT}; font-size: 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton#Primary {{ background: {ACCENT}; color: #06211d; border: none; font-weight: 500; }}
QTableWidget {{ background: #0e141c; alternate-background-color: #0e141c;
                border: 1px solid {BORDER}; gridline-color: {BORDER};
                color: {TEXT}; font-size: 12px; }}
QTableWidget::item {{ color: {TEXT}; padding: 3px; }}
QTableWidget::item:selected {{ background: #1c3a3a; color: {TEXT}; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none; padding: 5px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: #2a3542; border-radius: 4px; }}
"""


def _card(title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QLabel]:
    frame = QtWidgets.QFrame()
    frame.setObjectName("Card")
    frame.setMinimumHeight(64)
    lay = QtWidgets.QVBoxLayout(frame)
    lay.setContentsMargins(14, 10, 14, 10)
    lay.setSpacing(3)
    t = QtWidgets.QLabel(title)
    t.setObjectName("CardTitle")
    v = QtWidgets.QLabel("--")
    v.setObjectName("CardValue")
    lay.addWidget(t)
    lay.addWidget(v)
    return frame, v


class WaterfallView(QtWidgets.QWidget):
    """两层自绘：底层绿色信号，上层半透明红色干扰带。"""

    def __init__(self):
        super().__init__()
        self._sig = None
        self._jam = None
        self.setMinimumHeight(220)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

    def set_images(self, sig_img, jam_img):
        self._sig = sig_img
        self._jam = jam_img
        self.update()

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#1a2028"))
        if self._sig is not None:
            p.drawImage(self.rect(), self._sig)
        if self._jam is not None:
            p.drawImage(self.rect(), self._jam)   # alpha 叠加，绿色仍可见
        p.end()


class AgentWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(str)

    def __init__(self, agent: JamAgent, text: str):
        super().__init__()
        self.agent, self.text = agent, text

    def run(self):
        try:
            self.done.emit(self.agent.handle(self.text))
        except Exception as e:
            self.done.emit(f"指令执行失败：{e}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: JamConfig, controller: JamController):
        super().__init__()
        self.cfg = cfg
        self.ctrl = controller
        self.agent = JamAgent(controller)
        self._worker: AgentWorker | None = None
        self._trail: deque[tuple[int, float]] = deque(maxlen=600)
        self.wf_rows = 240
        self._wf_hist: deque = deque(maxlen=self.wf_rows)
        self._trend: deque = deque(maxlen=600)
        self._last_center = 0.0
        self._last_event_sig = None
        self._last_jam_mode = None

        self.setWindowTitle("感知·跟踪·干扰控制台")
        self.resize(1560, 960)
        self.setStyleSheet(QSS)
        self._build()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(120)

    # ── 布局 ─────────────────────────────────────────────────
    def _build(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(12)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)

        head = QtWidgets.QHBoxLayout()
        col = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("红方 · 跟踪干扰机")
        title.setObjectName("Title")
        sub = QtWidgets.QLabel("3 GHz · 1.6 MHz 观测 · 10×160 kHz 信道 · 感知→锁定→跟随")
        sub.setObjectName("Subtitle")
        col.addWidget(title)
        col.addWidget(sub)
        head.addLayout(col)
        head.addStretch(1)
        self.badge = QtWidgets.QLabel("初始化")
        self.badge.setObjectName("Badge")
        head.addWidget(self.badge)
        left.addLayout(head)

        cards = QtWidgets.QGridLayout()
        cards.setSpacing(8)
        self.c_state, self.v_state = _card("跟踪状态")
        self.c_target, self.v_target = _card("目标信道")
        self.c_hop, self.v_hop = _card("跟随跳频")
        self.c_center, self.v_center = _card("中心频率")
        self.c_param, self.v_param = _card("干扰参数")
        self.c_cycle, self.v_cycle = _card("控制周期")
        for i, w in enumerate([self.c_state, self.c_target, self.c_hop,
                               self.c_center, self.c_param, self.c_cycle]):
            cards.addWidget(w, 0, i)
        left.addLayout(cards)

        mid = QtWidgets.QHBoxLayout()
        mid.setSpacing(10)

        spec_box = QtWidgets.QGroupBox("实时频谱（相对噪声底）")
        spec_lay = QtWidgets.QVBoxLayout(spec_box)
        self.spec = pg.PlotWidget()
        self.spec.setBackground("#0e141c")
        self.spec.showGrid(x=True, y=True, alpha=0.15)
        self.spec.setLabel("bottom", "频率 (GHz)")
        self.spec.setLabel("left", "dB")
        self.spec.getAxis("bottom").setPen(pg.mkPen(MUTED))
        self.spec.getAxis("left").setPen(pg.mkPen(MUTED))
        self.spec.getAxis("bottom").setTextPen(pg.mkPen(MUTED))
        self.spec.getAxis("left").setTextPen(pg.mkPen(MUTED))
        self.spec_curve = self.spec.plot(pen=pg.mkPen(ACCENT, width=1.4))
        self.spec_target = pg.InfiniteLine(angle=90, pen=pg.mkPen(RED, width=1, style=QtCore.Qt.DashLine))
        self.spec_target.setVisible(False)
        self.spec.addItem(self.spec_target)
        spec_lay.addWidget(self.spec)
        mid.addWidget(spec_box, 3)

        wf_box = QtWidgets.QGroupBox("信道状态流水（顶部最新）")
        wf_lay = QtWidgets.QVBoxLayout(wf_box)
        wf_lay.setSpacing(4)
        legend = QtWidgets.QHBoxLayout()
        legend.setSpacing(10)
        for color, text in ((GREEN, "信号"), ("#ff6e6e", "干扰覆盖（半透明）"),
                            ("#1a2028", "空闲")):
            lb = QtWidgets.QLabel(f'<span style="color:{color}">■</span> {text}')
            lb.setStyleSheet(f"color:{MUTED};font-size:11px;")
            legend.addWidget(lb)
        legend.addStretch(1)
        wf_lay.addLayout(legend)

        self.wf_view = WaterfallView()
        self.wf_view.setStyleSheet(f"border:1px solid {BORDER};")
        wf_lay.addWidget(self.wf_view, 1)

        ch_row = QtWidgets.QHBoxLayout()
        ch_row.setSpacing(0)
        for i in range(1, 11):
            lb = QtWidgets.QLabel(str(i))
            lb.setAlignment(QtCore.Qt.AlignCenter)
            lb.setStyleSheet(f"color:{MUTED};font-size:11px;")
            ch_row.addWidget(lb, 1)
        wf_lay.addLayout(ch_row)
        mid.addWidget(wf_box, 2)
        left.addLayout(mid, 4)

        tabs = QtWidgets.QTabWidget()
        self.tabs = tabs
        log_tab = QtWidgets.QWidget()
        log_lay = QtWidgets.QVBoxLayout(log_tab)
        self.log = QtWidgets.QTableWidget(0, 4)
        self.log.setHorizontalHeaderLabels(["时间", "事件", "信道", "说明"])
        self.log.horizontalHeader().setStretchLastSection(True)
        self.log.verticalHeader().setVisible(False)
        self.log.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.log.setAlternatingRowColors(False)
        log_lay.addWidget(self.log)
        tabs.addTab(log_tab, "事件日志")

        trail_tab = QtWidgets.QWidget()
        trail_lay = QtWidgets.QVBoxLayout(trail_tab)
        self.trail = pg.PlotWidget()
        self.trail.setBackground("#0e141c")
        self.trail.setYRange(0.5, 10.5)
        self.trail.setLabel("left", "信道")
        self.trail.setLabel("bottom", "帧")
        self.trail.getAxis("left").setPen(pg.mkPen(MUTED))
        self.trail.getAxis("bottom").setPen(pg.mkPen(MUTED))
        self.trail.getAxis("left").setTextPen(pg.mkPen(MUTED))
        self.trail.getAxis("bottom").setTextPen(pg.mkPen(MUTED))
        self.trail_curve = self.trail.plot(pen=pg.mkPen(ACCENT, width=2), connect="finite")
        trail_lay.addWidget(self.trail)
        tabs.addTab(trail_tab, "跟踪轨迹")

        trend_tab = QtWidgets.QWidget()
        trend_lay = QtWidgets.QVBoxLayout(trend_tab)
        trend_lay.setSpacing(4)
        snr_title = QtWidgets.QLabel("全局 SNR (dB)")
        snr_title.setStyleSheet(f"color:{MUTED};font-size:11px;")
        trend_lay.addWidget(snr_title)
        self.trend_snr = pg.PlotWidget()
        self.trend_snr.setBackground("#0e141c")
        self.trend_snr.setYRange(0, 60)
        for ax in ("left", "bottom"):
            self.trend_snr.getAxis(ax).setPen(pg.mkPen(MUTED))
            self.trend_snr.getAxis(ax).setTextPen(pg.mkPen(MUTED))
        self.trend_snr_curve = self.trend_snr.plot(pen=pg.mkPen(ACCENT, width=1.6))
        trend_lay.addWidget(self.trend_snr)

        score_title = QtWidgets.QLabel("最高信道评分")
        score_title.setStyleSheet(f"color:{MUTED};font-size:11px;")
        trend_lay.addWidget(score_title)
        self.trend_score = pg.PlotWidget()
        self.trend_score.setBackground("#0e141c")
        self.trend_score.setLabel("bottom", "帧")
        self.trend_score.setYRange(0, 1.05)
        for ax in ("left", "bottom"):
            self.trend_score.getAxis(ax).setPen(pg.mkPen(MUTED))
            self.trend_score.getAxis(ax).setTextPen(pg.mkPen(MUTED))
        self.trend_score_curve = self.trend_score.plot(pen=pg.mkPen(AMBER, width=1.6))
        self.trend_score.addItem(pg.InfiniteLine(pos=self.cfg.on_threshold, angle=0,
                                                 pen=pg.mkPen("#6b7280", width=1,
                                                              style=QtCore.Qt.DashLine)))
        trend_lay.addWidget(self.trend_score)
        tabs.addTab(trend_tab, "运行趋势")

        wave_tab = QtWidgets.QWidget()
        wave_lay = QtWidgets.QVBoxLayout(wave_tab)
        self.wave_info = QtWidgets.QLabel("未发射")
        self.wave_info.setStyleSheet(f"color:{MUTED};font-size:12px;")
        wave_lay.addWidget(self.wave_info)
        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setBackground("#0e141c")
        self.wave_plot.setLabel("bottom", "时间 (ms)")
        self.wave_plot.setLabel("left", "幅度")
        self.wave_plot.getAxis("left").enableAutoSIPrefix(False)
        self.wave_plot.setYRange(0, 0.3)
        for ax in ("left", "bottom"):
            self.wave_plot.getAxis(ax).setPen(pg.mkPen(MUTED))
            self.wave_plot.getAxis(ax).setTextPen(pg.mkPen(MUTED))
        self.wave_curve = self.wave_plot.plot(pen=pg.mkPen("#8ab4f8", width=1.2))
        wave_lay.addWidget(self.wave_plot)
        tabs.addTab(wave_tab, "干扰波形")
        left.addWidget(tabs, 3)

        root.addLayout(left, 5)

        # ── 右侧 Agent ───────────────────────────────────────
        right = QtWidgets.QWidget()
        right.setFixedWidth(400)
        rlay = QtWidgets.QVBoxLayout(right)
        rlay.setContentsMargins(10, 12, 10, 12)
        rlay.setSpacing(8)

        at = QtWidgets.QLabel("频谱管控 Agent")
        at.setObjectName("Title")
        rlay.addWidget(at)
        hint = QtWidgets.QLabel("云端模型解析指令 · 跟踪与发射由本地闭环执行")
        hint.setObjectName("Subtitle")
        rlay.addWidget(hint)

        self.chat = QtWidgets.QTextBrowser()
        rlay.addWidget(self.chat, 1)

        quick = QtWidgets.QGridLayout()
        quick.setSpacing(6)
        for i, (label, fn) in enumerate([
            ("响应式", lambda: self._cmd("start")),
            ("停止干扰", lambda: self._cmd("stop")),
            ("多音", lambda: self._cmd("mode", 1)),
            ("限带噪声", lambda: self._cmd("mode", 0)),
            ("功率 +5", lambda: self._cmd("power", 5)),
            ("功率 -5", lambda: self._cmd("power", -5)),
            ("状态", lambda: self._cmd("status")),
            ("态势报告", lambda: self._cmd("situation")),
        ]):
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(fn)
            quick.addWidget(btn, i // 3, i % 3)
        rlay.addLayout(quick)

        fixed_row = QtWidgets.QHBoxLayout()
        fixed_row.setSpacing(6)
        flabel = QtWidgets.QLabel("固定信道")
        flabel.setStyleSheet(f"color:{MUTED};font-size:12px;")
        self.ch_combo = QtWidgets.QComboBox()
        self.ch_combo.addItems([str(i) for i in range(1, 11)])
        self.ch_combo.setCurrentText("6")      # 默认 3.000000 GHz
        self.ch_combo.setStyleSheet(f"background:#0e141c;border:1px solid {BORDER};"
                                    f"border-radius:6px;padding:4px;color:{TEXT};")
        fbtn = QtWidgets.QPushButton("固定干扰")
        fbtn.clicked.connect(self._fixed_jam)
        fixed_row.addWidget(flabel)
        fixed_row.addWidget(self.ch_combo, 1)
        fixed_row.addWidget(fbtn, 2)
        self.fixed_row = fixed_row
        rlay.addLayout(fixed_row)

        span_row = QtWidgets.QHBoxLayout()
        span_row.setSpacing(6)
        slabel = QtWidgets.QLabel("覆盖信道")
        slabel.setStyleSheet(f"color:{MUTED};font-size:12px;")
        self.span_spin = QtWidgets.QSpinBox()
        self.span_spin.setRange(1, 10)
        self.span_spin.setValue(1)
        self.span_spin.setSuffix(" 个")
        self.span_spin.setToolTip("从目标信道起，连续压住相邻的 N 个信道（带宽 = N×160 kHz）")
        self.span_spin.setStyleSheet(f"background:#0e141c;border:1px solid {BORDER};"
                                     f"border-radius:6px;padding:4px;color:{TEXT};")
        self.span_spin.valueChanged.connect(self._span_changed)
        self.span_hint = QtWidgets.QLabel("只压目标信道")
        self.span_hint.setStyleSheet(f"color:{MUTED};font-size:11px;")
        span_row.addWidget(slabel)
        span_row.addWidget(self.span_spin, 1)
        span_row.addWidget(self.span_hint, 2)
        rlay.addLayout(span_row)

        jamlen_row = QtWidgets.QHBoxLayout()
        jamlen_row.setSpacing(6)
        jlabel = QtWidgets.QLabel("干扰帧长")
        jlabel.setStyleSheet(f"color:{MUTED};font-size:12px;")
        self.jamlen_spin = QtWidgets.QDoubleSpinBox()
        self.jamlen_spin.setRange(5.0, 2000.0)
        self.jamlen_spin.setDecimals(0)
        self.jamlen_spin.setSingleStep(50.0)
        self.jamlen_spin.setValue(400.0)
        self.jamlen_spin.setSuffix(" ms")
        self.jamlen_spin.setToolTip("一帧压多久；越长占空比越高、看起来越连续，但跟踪越慢")
        self.jamlen_spin.setStyleSheet(f"background:#0e141c;border:1px solid {BORDER};"
                                       f"border-radius:6px;padding:4px;color:{TEXT};")
        self.jamlen_spin.valueChanged.connect(self._jamlen_changed)
        self.jamlen_hint = QtWidgets.QLabel("占空比 —")
        self.jamlen_hint.setStyleSheet(f"color:{MUTED};font-size:11px;")
        jamlen_row.addWidget(jlabel)
        jamlen_row.addWidget(self.jamlen_spin, 1)
        jamlen_row.addWidget(self.jamlen_hint, 2)
        rlay.addLayout(jamlen_row)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(6)
        mlabel = QtWidgets.QLabel("干扰模式")
        mlabel.setStyleSheet(f"color:{MUTED};font-size:12px;")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["固定干扰", "响应式干扰", "随机干扰"])
        self.mode_combo.setStyleSheet(f"background:#0e141c;border:1px solid {BORDER};"
                                      f"border-radius:6px;padding:4px;color:{TEXT};")
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.rand_spin = QtWidgets.QDoubleSpinBox()
        self.rand_spin.setRange(0.5, 60.0)
        self.rand_spin.setDecimals(1)
        self.rand_spin.setSingleStep(0.5)
        self.rand_spin.setValue(2.0)
        self.rand_spin.setSuffix(" s")
        self.rand_spin.setEnabled(False)
        self.rand_spin.setStyleSheet(f"background:#0e141c;border:1px solid {BORDER};"
                                     f"border-radius:6px;padding:4px;color:{TEXT};")
        self.rand_spin.valueChanged.connect(self._rand_period_changed)
        mode_row.addWidget(mlabel)
        mode_row.addWidget(self.mode_combo, 1)
        mode_row.addWidget(self.rand_spin, 1)
        rlay.addLayout(mode_row)

        react_row = QtWidgets.QHBoxLayout()
        react_row.setSpacing(6)
        rlabel = QtWidgets.QLabel("响应延迟")
        rlabel.setStyleSheet(f"color:{MUTED};font-size:12px;")
        self.react_spin = QtWidgets.QDoubleSpinBox()
        self.react_spin.setRange(0.0, 3000.0)
        self.react_spin.setDecimals(0)
        self.react_spin.setSingleStep(50.0)
        self.react_spin.setValue(0.0)
        self.react_spin.setSuffix(" ms")
        self.react_spin.setToolTip("判定切频后等多久才把干扰搬过去；0 = 立即跟随")
        self.react_spin.setStyleSheet(f"background:#0e141c;border:1px solid {BORDER};"
                                      f"border-radius:6px;padding:4px;color:{TEXT};")
        self.react_spin.valueChanged.connect(self._react_changed)
        self.react_hint = QtWidgets.QLabel("立即跟随")
        self.react_hint.setStyleSheet(f"color:{MUTED};font-size:11px;")
        react_row.addWidget(rlabel)
        react_row.addWidget(self.react_spin, 1)
        react_row.addWidget(self.react_hint, 2)
        rlay.addLayout(react_row)

        center_row = QtWidgets.QHBoxLayout()
        center_row.setSpacing(6)
        clabel = QtWidgets.QLabel("中心(MHz)")
        clabel.setStyleSheet(f"color:{MUTED};font-size:12px;")
        self.center_spin = QtWidgets.QDoubleSpinBox()
        self.center_spin.setRange(300.0, 6000.0)
        self.center_spin.setDecimals(3)
        self.center_spin.setSingleStep(1.0)
        self.center_spin.setKeyboardTracking(False)
        self.center_spin.setValue(self.cfg.center_freq_hz / 1e6)
        self.center_spin.lineEdit().returnPressed.connect(self._set_center)
        cbtn = QtWidgets.QPushButton("切换")
        cbtn.clicked.connect(self._set_center)
        center_row.addWidget(clabel)
        center_row.addWidget(self.center_spin, 1)
        center_row.addWidget(cbtn, 2)
        rlay.addLayout(center_row)

        self.victim_btn = QtWidgets.QPushButton("模拟信号 关")
        self.victim_btn.setCheckable(True)
        self.victim_btn.toggled.connect(self._toggle_victim)
        rlay.addWidget(self.victim_btn)


        row = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("例如：跟踪信道3，多音，功率15dB")
        self.input.returnPressed.connect(self._send)
        self.send_btn = QtWidgets.QPushButton("下发")
        self.send_btn.setObjectName("Primary")
        self.send_btn.clicked.connect(self._send)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_btn)
        rlay.addLayout(row)

        root.addWidget(right)

        self._append("系统", "控制台就绪。可直接说话，或点快捷指令。")

    # ── 交互 ─────────────────────────────────────────────────
    def _append(self, who: str, text: str):
        if who == "系统":
            color = ACCENT
        elif who == "我":
            color = "#93c5fd"
        else:
            color = AMBER          # Agent 用琥珀色，和正文白色区分
        body = html.escape(str(text)).replace("\n", "<br>")
        self.chat.append(f'<p style="margin:6px 0"><span style="color:{color}">'
                         f'<b>{who}</b></span>　{body}</p>')

    def _send(self):
        text = self.input.text().strip()
        if not text or self._worker is not None:
            return
        self.input.clear()
        self._append("我", text)
        self.send_btn.setEnabled(False)
        self._worker = AgentWorker(self.agent, text)
        self._worker.done.connect(self._on_reply)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

    @QtCore.pyqtSlot(str)
    def _on_reply(self, text: str):
        self._append("Agent", text)

    def _on_worker_done(self):
        self.send_btn.setEnabled(True)
        self._worker = None

    MODE_ITEMS = ("fixed", "track", "random")

    def _apply_mode(self, silent: bool = False):
        idx = self.mode_combo.currentIndex()
        mode = self.MODE_ITEMS[idx] if 0 <= idx < len(self.MODE_ITEMS) else "track"
        if mode == "fixed":
            ch = int(self.ch_combo.currentText())
            self.ctrl.start_fixed(ch)
            msg = f"固定干扰：锁死信道 {ch}，对方切频不跟随。"
        elif mode == "random":
            per = float(self.rand_spin.value())
            self.ctrl.start_random(period_s=per)
            msg = f"随机干扰：每 {per:.1f} 秒随机换信道/功率/带宽（不看对方动作）。"
        else:
            self.ctrl.start_tracking()
            msg = "响应式干扰：锁定并跟随对方切频。"
        if not silent:
            self._append("系统", msg)

    def _mode_changed(self, _idx: int):
        self.rand_spin.setEnabled(self.mode_combo.currentIndex() == 2)
        self._apply_mode()

    def _rand_period_changed(self, _v):
        if self.mode_combo.currentIndex() == 2:
            self.ctrl.start_random(period_s=float(self.rand_spin.value()))

    def _react_changed(self, v):
        # 拖动/连点会连续触发，这里只下发命令，提示文字交给 _refresh
        self.ctrl.set_params(react_delay_ms=float(v))

    def _span_changed(self, v):
        self.ctrl.set_params(span_channels=int(v))

    def _jamlen_changed(self, v):
        self.ctrl.set_params(jam_ms=float(v))

    def _fixed_jam(self):
        ch = int(self.ch_combo.currentText())
        self.ctrl.start_fixed(ch)
        self._append("系统", f"已在信道 {ch} 固定干扰（不自动跟随）。")

    def _toggle_victim(self, checked: bool):
        self.victim_btn.setText("模拟信号 开" if checked else "模拟信号 关")
        self.ctrl.set_victim(checked)
        self._append("系统", "子板2模拟信号已打开。" if checked
                     else "子板2模拟信号已关闭。")

    def _set_center(self):
        text = self.center_spin.lineEdit().text().replace(",", "").strip()
        try:
            mhz = float(text)
        except ValueError:
            mhz = float(self.center_spin.value())
        mhz = min(max(mhz, 300.0), 6000.0)
        self.center_spin.setValue(mhz)
        self.ctrl.set_center(mhz * 1e6)
        self._append("系统", f"中心频率切换到 {mhz:.3f} MHz，重新搜索。")

    def _cmd(self, kind: str, value=None):
        if kind == "start":
            self.ctrl.start_tracking()
            self._append("系统", "已切换到响应式干扰（锁定并跟随）。")
        elif kind == "stop":
            self.ctrl.stop_tracking()
            self._append("系统", "已停止跟踪与发射。")
        elif kind == "mode":
            self.ctrl.set_params(waveform_mode=value)
            self._append("系统", f"波形切换为 {WAVEFORM_NAMES.get(value)}。")
        elif kind == "power":
            cur = self.ctrl.status().get("power_db", 10.0)
            new = min(POWER_MAX_DB, max(POWER_MIN_DB, cur + value))
            self.ctrl.set_params(power_db=new)
            self._append("系统", f"功率调整为 {new:.0f} dB。")
        elif kind == "situation":
            self._append("系统", self.agent.tools.analyze_situation())
        elif kind == "status":
            st = self.ctrl.status()
            target = st.get("target_ch")
            self._append("系统", f"当前{'锁定信道 ' + str(target) if target else '搜索中'}，"
                                 f"中心 {st.get('center_mhz')} MHz，"
                                 f"已跟随跳频 {st.get('hop_count', 0)} 次，"
                                 f"周期 {st.get('cycle_ms')} ms。")

    # ── 刷新 ─────────────────────────────────────────────────
    def _refresh(self):
        st = self.ctrl.status()
        if not st:
            return

        if not st.get("tracking", True):
            state_text = "已停止"
        elif st.get("jam_mode") == "fixed":
            state_text = "固定"
        elif st.get("jam_mode") == "random":
            state_text = "随机"
        elif st.get("scanning"):
            state_text = "扫描"
        elif st.get("target_ch"):
            state_text = "锁定"
        else:
            state_text = "搜索"
        self.badge.setText(f"中心 {st.get('center_mhz', 0):.3f} MHz · {state_text}"
                           f" · TX {st.get('tx_frames', 0)}")
        self.v_state.setText(state_text)
        self.v_state.setStyleSheet(
            f"color:{RED if st.get('target_ch') else (MUTED if state_text == '已停止' else AMBER)};")
        target = st.get("target_ch")
        jam_ch = st.get("jam_target_ch")
        tx_mhz = st.get("tx_freq_mhz")
        if target and jam_ch and jam_ch != target:
            # 延迟窗口内：跟踪器已换目标，干扰还压在旧信道上
            self.v_target.setText(f"跟踪 {target}｜仍压 {jam_ch}")
        elif target and tx_mhz:
            self.v_target.setText(f"信道 {target} · {tx_mhz:.4f} MHz")
        else:
            self.v_target.setText(f"信道 {target}" if target else "—")

        # 反应延迟：值同步 + 剩余等待时间提示
        delay_ms = float(st.get("react_delay_ms") or 0.0)
        if not self.react_spin.hasFocus() and abs(self.react_spin.value() - delay_ms) > 0.5:
            self.react_spin.blockSignals(True)
            self.react_spin.setValue(delay_ms)
            self.react_spin.blockSignals(False)
        pend = st.get("react_pending_ms")
        if delay_ms <= 0:
            hint = "立即跟随"
        elif not st.get("react_active"):
            hint = f"延迟 {delay_ms:.0f} ms（仅响应式生效）"
        elif pend is not None:
            hint = f"延迟 {delay_ms:.0f} ms ｜ 等待中 {pend:.0f} ms"
        elif st.get("last_react_ms") is not None:
            hint = f"延迟 {delay_ms:.0f} ms ｜ 上次实延 {st['last_react_ms']:.0f} ms"
        else:
            hint = f"延迟 {delay_ms:.0f} ms"
        self.react_hint.setText(hint)

        # 覆盖信道数
        span = int(st.get("span_channels") or 1)
        if not self.span_spin.hasFocus() and self.span_spin.value() != span:
            self.span_spin.blockSignals(True)
            self.span_spin.setValue(span)
            self.span_spin.blockSignals(False)
        if span <= 1:
            self.span_hint.setText("只压目标信道")
        else:
            lo = st.get("jam_target_ch") or st.get("target_ch")
            hi = st.get("span_to_ch")
            bw = float(st.get("bw_khz") or 0)
            self.span_hint.setText(
                f"{lo}-{hi} 号连续 {span} 个 · {bw:.0f} kHz" if lo
                else f"连续 {span} 个 · {bw:.0f} kHz")

        # 干扰帧长 / 占空比
        jam_ms = float(st.get("jam_on_ms") or 0.0)
        if not self.jamlen_spin.hasFocus() and abs(self.jamlen_spin.value() - jam_ms) > 0.5:
            self.jamlen_spin.blockSignals(True)
            self.jamlen_spin.setValue(jam_ms)
            self.jamlen_spin.blockSignals(False)
        cyc = float(st.get("cycle_ms") or 0.0)
        self.jamlen_hint.setText(
            f"占空比 {st.get('duty_pct', 0):.0f}% · 周期 {cyc:.0f} ms" if cyc
            else f"{jam_ms:.0f} ms")
        self.v_hop.setText(str(st.get("hop_count", 0)))
        self.v_center.setText(f"{st.get('center_mhz', 0):.3f} MHz")
        von = bool(st.get("victim_on", False))
        if self.victim_btn.isChecked() != von:
            self.victim_btn.blockSignals(True)
            self.victim_btn.setChecked(von)
            self.victim_btn.setText("模拟信号 开" if von else "模拟信号 关")
            self.victim_btn.blockSignals(False)
        le = self.center_spin.lineEdit()
        if le is None or not le.hasFocus():
            self.center_spin.blockSignals(True)
            self.center_spin.setValue(float(st.get("center_mhz", 3000.0)))
            self.center_spin.blockSignals(False)
        wave = WAVEFORM_NAMES.get(st.get("waveform", 0), "-")
        self.v_param.setText(f"{st.get('power_db', 0):.0f} dB · {st.get('bw_khz', 0):.0f} kHz · {wave}")
        self.v_cycle.setText(f"{st.get('cycle_ms', 0):.1f} ms")

        # 干扰模式下拉与随机周期，跟随后端状态（阻塞信号，避免回环触发）
        jm = st.get("jam_mode", "track")
        want = {"fixed": 0, "track": 1, "random": 2}.get(jm, 1)
        if self.mode_combo.currentIndex() != want:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(want)
            self.mode_combo.blockSignals(False)
        self.rand_spin.setEnabled(jm == "random")
        # 只在模式刚变化时对齐一次信道下拉，避免周期性覆盖操作员刚选的信道
        if jm != self._last_jam_mode:
            if jm == "fixed" and st.get("target_ch"):
                self.ch_combo.blockSignals(True)
                self.ch_combo.setCurrentText(str(int(st["target_ch"])))
                self.ch_combo.blockSignals(False)
            self._last_jam_mode = jm
        if not self.rand_spin.hasFocus():
            per = float(st.get("random_period_s") or 2.0)
            if abs(self.rand_spin.value() - per) > 0.05:
                self.rand_spin.blockSignals(True)
                self.rand_spin.setValue(per)
                self.rand_spin.blockSignals(False)

        freqs = st.get("freqs_ghz") or []
        spec = st.get("spectrum") or []
        if freqs and spec and len(freqs) == len(spec):
            arr = np.asarray(spec, dtype=float)
            ref = np.median(arr) + 1e-12
            db = 20 * np.log10(arr / ref + 1e-9)
            self.spec_curve.setData(freqs, db)
            # 固定横轴到当前 1.6 MHz 窗口，避免隐藏项把 auto-range 拉偏
            self.spec.setXRange(freqs[0], freqs[-1], padding=0.01)
            if st.get("target_ch"):
                self.spec_target.setVisible(True)
                self.spec_target.setPos(self._target_freq(st["target_ch"]))
            else:
                self.spec_target.setVisible(False)

        center_now = float(st.get("center_mhz", 0) or 0)
        if abs(center_now - self._last_center) > 0.001:
            self._wf_hist.clear()          # 换频段后旧流水不再有效
            self._last_center = center_now

        scores = st.get("scores") or []
        if len(scores) == 10:
            target = st.get("target_ch")
            jam_on = bool(target)      # 负功率也是在发射，只是更弱
            # 干扰带：中心（0~10 的信道坐标）、宽度（信道数）+ 视觉外扩
            band = ((float(target) - 0.5,
                     float(st.get("bw_khz", 160) or 160) / 160.0,
                     True) if jam_on else (0.0, 0.0, False))
            self._wf_hist.appendleft((np.asarray(scores, dtype=float), band))
            self._render_waterfall()

        target = st.get("target_ch")
        frame = st.get("frames", 0)
        self._trail.append((frame, target if target else np.nan))
        xs = [p[0] for p in self._trail]
        ys = [p[1] for p in self._trail]
        self.trail_curve.setData(xs, ys)

        snr = float(st.get("snr_est", 0) or 0)
        mx = max(scores) if scores else 0.0
        self._trend.append((frame, snr, mx))
        tx = [p[0] for p in self._trend]
        self.trend_snr_curve.setData(tx, [p[1] for p in self._trend])
        self.trend_score_curve.setData(tx, [p[2] for p in self._trend])

        wave = st.get("wave_preview") or []
        if len(wave) >= 16:
            self.wave_curve.setData(
                np.linspace(0, self.cfg.jam_on_s * 1e3, len(wave)), wave)
            self.wave_info.setText(st.get("wave_meta", ""))
        else:
            self.wave_curve.setData([], [])
            self.wave_info.setText("未发射")

        ev = st.get("last_event")
        if ev:
            sig = (ev.get("kind"), ev.get("frame"), ev.get("channel"),
                   ev.get("prev_channel"))
            if sig != self._last_event_sig:
                self._last_event_sig = sig
                self._add_log(ev)

        err = st.get("error")
        if err:
            self.badge.setText("错误：" + err[:40])

    def _render_waterfall(self) -> None:
        """底层绿色信号 + 上层半透明淡红干扰带（比信号略宽），顶部最新。"""
        rows = self.wf_rows
        sub = 8
        width = 10 * sub

        sig_arr = np.zeros((rows, width, 3), dtype=np.uint8)
        sig_arr[:, :, 0] = 26
        sig_arr[:, :, 1] = 32
        sig_arr[:, :, 2] = 42
        jam_arr = np.zeros((rows, width, 4), dtype=np.uint8)   # alpha=0

        for i, (sig, band) in enumerate(self._wf_hist):
            if i >= rows:
                break
            # 噪声门限：低于 0.22 按空闲，避免底噪累积成竖条
            s_ = np.clip((sig - 0.22) / 0.78, 0, 1)
            for k in range(10):
                # 绿色块只画信道中间 75%，两侧留白让红色干扰层露出来，
                # 这样既不夸大干扰带宽，又能看出是两层
                c0, c1 = k * sub, (k + 1) * sub
                pad = max(1, sub // 8)
                a, b = c0 + pad, c1 - pad
                sig_arr[i, c0:c1, 0] = 26
                sig_arr[i, c0:c1, 1] = 32
                sig_arr[i, c0:c1, 2] = 42
                sig_arr[i, a:b, 0] = 26
                sig_arr[i, a:b, 1] = (32 + 210 * s_[k]).astype(np.uint8)
                sig_arr[i, a:b, 2] = (42 + 30 * s_[k]).astype(np.uint8)

            center, bw_ch, on = band
            if on:
                left = int(round((center - bw_ch / 2) * sub))
                right = int(round((center + bw_ch / 2) * sub))
                left = max(0, left)
                right = min(width, right)
                if right > left:
                    jam_arr[i, left:right] = (255, 120, 120, 115)

        sig_q = QtGui.QImage(sig_arr.data, width, rows, width * 3,
                             QtGui.QImage.Format_RGB888).copy()
        jam_q = QtGui.QImage(jam_arr.data, width, rows, width * 4,
                             QtGui.QImage.Format_RGBA8888).copy()
        self.wf_view.set_images(sig_q, jam_q)

    def _target_freq(self, ch: int) -> float:
        center = self.ctrl.status().get("center_mhz", self.cfg.center_freq_hz / 1e6) * 1e6
        return self.cfg.centers_at(center)[ch - 1] / 1e9

    def _add_log(self, ev: dict):
        names = {"acquired": "锁定", "hop": "跟随", "lost": "释放", "random": "随机"}
        row = self.log.rowCount()
        self.log.insertRow(row)
        vals = [time.strftime("%H:%M:%S"), names.get(ev.get("kind"), ev.get("kind")),
                ev.get("channel") or "-", ev.get("detail") or ""]
        for c, v in enumerate(vals):
            self.log.setItem(row, c, QtWidgets.QTableWidgetItem(str(v)))
        self.log.scrollToBottom()
        while self.log.rowCount() > 200:
            self.log.removeRow(0)

    def closeEvent(self, event):
        self.timer.stop()
        if self._worker is not None:
            self._worker.wait(3000)
        self.ctrl.stop()
        super().closeEvent(event)
