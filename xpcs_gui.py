#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xpcs_gui.py

XPCS(X-ray Photon Correlation Spectroscopy) 실습용 GUI.

Speckle_sim.py 에 있던 obj = {...} 설정 부분을 GUI 입력 폼으로 바꾸고,
Particle_Sim_module.py 의 함수(BC_conti, Speckle_calculator, ROI_matrix, cal_g2,
azimuthal_integration)를 xpcs_core.py 를 통해 그대로 호출합니다. 분석 알고리즘 자체는
바꾸지 않았고, Particle_Sim_module.py 에는 아래 2가지 fix만 직접 반영했습니다
(자세한 설명은 그 파일의 `[FIX ...]` 주석 참고):

  - SDD/파장이 speckle contrast에 반영되도록 scale2pix 계산을 물리식으로 교체
  - Python 3.10 미만에서도 import 되도록 타입힌트 평가를 지연시킴

같은 폴더 구성
--------------
    Particle_Sim_module.py   (원본 + 위 2가지 fix)
    xpcs_core.py              (원본 함수를 호출하는 wrapper + I(q)/Gamma(q) 피팅 함수)
    xpcs_gui.py                (이 파일 - PyQt5 UI)

실행 방법
--------
    pip install -r requirements.txt
    python xpcs_gui.py

탭 구성 (각 탭에는 그 단계에 필요한 파라미터만 있음)
--------------------------------------------------
1) Setup & Simulation : 물리 파라미터 입력 (아래 "실공간 단위" 참고) + Run Simulation
2) Preview             : Particle / Speckle 이미지 미리보기 (프레임 슬라이더)
3) I(q)                : 시간평균 speckle 패턴의 azimuthal 적분으로 I(q) 계산/플롯
4) g2(τ)               : Q-bin(ROI) 정의(+평균 이미지 위에 ROI 표시) + g2(τ) 계산/플롯
5) Gamma(q) & D fit     : 각 bin의 g2를 개별 피팅해 Gamma(q)를 얻고(1단계),
                          사용할 bin을 골라(outlier 제외) 선형피팅으로 D와
                          Stokes-Einstein 반경을 구함(2단계)

실공간 단위(pix2real)
---------------------
원본 코드는 dimx/dimy(샘플 격자 크기), r(입자 반경), jump(브라운 운동 스텝)를 전부
"픽셀" 단위의 임의의 숫자로만 다뤘습니다. 여기서는 `pix2real`(픽셀당 실제 크기, [m/px])
하나를 기준으로, 샘플 FOV/입자 반경/jump를 전부 실제 길이 단위([m])로 입력받고
내부적으로만 픽셀 단위로 환산해서 원본 함수에 넘깁니다.

기본값은 SDD=8m, pixelsize=75um, pix2real=10nm/px, FOV=10x10um, r=200nm,
jump=20nm/frame, seed=100, timestep=500ns, T_total=30us 로 설정되어 있습니다.
이 조건에서는 검출기 픽셀로 리샘플링된 speckle 이미지가 약 1463x1463 이라 시뮬레이션에
약 30초 정도 걸립니다. g2(τ) 탭의 기본 ROI(StartPixel=40, EndPixel=120, BinNum=5)는
이 조건에서 Gamma(q) decay가 관측 가능한 지연시간 범위 안에 clearly 들어오도록
확인해서 고른 값입니다.
"""

import sys
import traceback

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
import matplotlib.cm as mpl_cm

from PyQt5 import QtCore, QtWidgets

try:
    from xpcs_core import (
        build_default_obj,
        run_brownian_and_speckle,
        run_roi_and_g2,
        compute_iq,
        fit_gamma_per_bin,
        fit_D_from_gamma,
        stokes_einstein_radius_nm,
    )
except ImportError as e:
    raise ImportError(
        "xpcs_core.py / Particle_Sim_module.py 를 이 파일과 같은 폴더에 두세요."
    ) from e


# ----------------------------------------------------------------------
# matplotlib 버전에 상관없이 동작하는 discrete colormap 헬퍼
# (matplotlib.cm.get_cmap 은 최신 버전에서 deprecated/변경되었기 때문에
#  구버전/최신버전 모두에서 동작하도록 fallback 을 둔다 - 크로스플랫폼 호환성)
# ----------------------------------------------------------------------
def get_discrete_cmap(name, n):
    n = max(int(n), 1)
    try:
        return matplotlib.colormaps[name].resampled(n)
    except Exception:
        pass
    try:
        return matplotlib.colormaps.get_cmap(name).resampled(n)
    except Exception:
        pass
    return mpl_cm.get_cmap(name, n)


# ----------------------------------------------------------------------
# 숫자 파싱 유틸: "200e-6", "1/4.5e6" 같은 표현도 허용 (원본 스크립트 표기 스타일)
# ----------------------------------------------------------------------
def parse_number(text, cast=float):
    text = text.strip()
    try:
        return cast(float(text))
    except ValueError:
        pass
    # 아주 단순한 산술 표현("1/4.5e6" 등)만 허용하는 안전한 eval
    allowed = set("0123456789.eE+-/*() ")
    if set(text) <= allowed:
        try:
            return cast(eval(text, {"__builtins__": {}}, {}))
        except Exception:
            pass
    raise ValueError(f"숫자로 해석할 수 없습니다: {text!r}")


def px_from_real(real_value, pix2real, minimum=1):
    if pix2real <= 0:
        raise ValueError("pix2real 은 0보다 커야 합니다.")
    return max(minimum, int(round(real_value / pix2real)))


# ----------------------------------------------------------------------
# 백그라운드 작업 스레드
# ----------------------------------------------------------------------
class SimThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object, object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, obj):
        super().__init__()
        self.obj = obj

    def run(self):
        try:
            particle_img, spec_img, obj2 = run_brownian_and_speckle(
                self.obj, progress_cb=self.progress.emit
            )
            self.finished_ok.emit(particle_img, spec_img, obj2)
        except Exception:
            self.failed.emit(traceback.format_exc())


class G2Thread(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, spec_img, obj, baseline_subtract, n_lag_points):
        super().__init__()
        self.spec_img = spec_img
        self.obj = obj
        self.baseline_subtract = baseline_subtract
        self.n_lag_points = n_lag_points

    def run(self):
        try:
            res = run_roi_and_g2(
                self.spec_img,
                self.obj,
                baseline_subtract=self.baseline_subtract,
                n_lag_points=self.n_lag_points,
                progress_cb=self.progress.emit,
            )
            self.finished_ok.emit(res)
        except Exception:
            self.failed.emit(traceback.format_exc())


class IqThread(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, spec_img, obj, bins, r_min, r_max):
        super().__init__()
        self.spec_img = spec_img
        self.obj = obj
        self.bins = bins
        self.r_min = r_min
        self.r_max = r_max

    def run(self):
        try:
            res = compute_iq(
                self.spec_img, self.obj, bins=self.bins,
                r_min=self.r_min, r_max=self.r_max, progress_cb=self.progress.emit,
            )
            self.finished_ok.emit(res)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ----------------------------------------------------------------------
# 메인 윈도우
# ----------------------------------------------------------------------
class XPCSMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XPCS Tutorial GUI - Speckle Simulation & g2 Analysis")
        self.resize(1350, 900)

        self.particle_img = None
        self.spec_img = None
        self.avg_img = None
        self.obj = None
        self.roi_result = None
        self.iq_result = None
        self.gamma_per_bin = None
        self.sim_thread = None
        self.g2_thread = None
        self.iq_thread = None

        self._build_ui()
        self._load_defaults()

    # ------------------------------------------------------------------
    def _build_ui(self):
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_setup_tab(), "Setup && Simulation")
        self.tabs.addTab(self._build_preview_tab(), "Preview")
        self.tabs.addTab(self._build_iq_tab(), "I(q)")
        self.tabs.addTab(self._build_g2_tab(), "g2(τ)")
        self.tabs.addTab(self._build_fit_tab(), "Gamma(q) && D fit")

    # ---- Tab 1: Setup & Simulation ------------------------------------
    def _build_setup_tab(self):
        panel = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(panel)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)

        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        layout = QtWidgets.QVBoxLayout(inner)

        group = QtWidgets.QGroupBox("물리 / 시뮬레이션 파라미터")
        form = QtWidgets.QFormLayout()
        group.setLayout(form)

        self.fields = {}
        field_defs = [
            ("SDD", "SDD [m]"),
            ("Energy", "X-ray Energy [keV]"),
            ("pixelsize", "검출기 pixel size [m]"),
            ("pix2real", "pix2real (픽셀당 실제 크기) [m/px]"),
            ("dimx_real", "샘플 FOV 가로 [m]"),
            ("dimy_real", "샘플 FOV 세로 [m]"),
            ("r_real", "입자 반경 [m]"),
            ("jump_real", "Brownian jump [m/frame]"),
            ("seed", "입자 개수"),
            ("timestep", "timestep [s]"),
            ("T_total", "총 측정시간 T_total [s]"),
        ]
        for key, label in field_defs:
            edit = QtWidgets.QLineEdit()
            form.addRow(label, edit)
            self.fields[key] = edit

        self.bc_combo = QtWidgets.QComboBox()
        self.bc_combo.addItems(["unbound", "bound"])
        form.addRow("경계조건 (BC)", self.bc_combo)

        layout.addWidget(group)

        hint = QtWidgets.QLabel(
            "* dimx/dimy/입자반경/jump 는 실제 길이 단위([m])로 입력하면 pix2real 로 "
            "환산되어 시뮬레이션에 쓰입니다 (환산된 픽셀 값은 실행 후 상태 메시지에 표시).\n"
            "* Q-bin/ROI 는 g2(τ) 탭에서, I(q) binning 은 I(q) 탭에서 따로 설정합니다."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(hint)

        self.run_btn = QtWidgets.QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self.on_run_simulation)
        layout.addWidget(self.run_btn)

        self.reset_btn = QtWidgets.QPushButton("모든 탭 기본값으로 되돌리기")
        self.reset_btn.clicked.connect(self._load_defaults)
        layout.addWidget(self.reset_btn)

        self.sim_progress = QtWidgets.QProgressBar()
        self.sim_progress.setRange(0, 0)
        self.sim_progress.setVisible(False)
        layout.addWidget(self.sim_progress)

        self.status_label = QtWidgets.QLabel("대기 중.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch(1)

        note = QtWidgets.QLabel(
            "참고: 기본값 조건(FOV 10x10um, 1463x1463 speckle image)에서 시뮬레이션에\n"
            "약 30초 정도 걸립니다. FOV/측정시간을 더 늘리면 수 분 이상 걸릴 수 있습니다.\n\n"
            "beam center(ROI 중심)에 너무 가까운 StartPixel(g2(τ) 탭)을 쓰면 입자\n"
            "다분산성 재추출 노이즈가 섞여 g2가 비정상적으로 빨리 감쇠할 수 있습니다."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note)

        return panel

    # ---- Tab 2: Preview -------------------------------------------------
    def _build_preview_tab(self):
        panel = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(panel)

        self.fig1 = Figure(figsize=(7, 4.5))
        self.ax0 = self.fig1.add_subplot(1, 2, 1)
        self.ax1 = self.fig1.add_subplot(1, 2, 2)
        self.canvas1 = FigureCanvas(self.fig1)
        toolbar1 = NavigationToolbar(self.canvas1, panel)
        preview_layout.addWidget(toolbar1)
        preview_layout.addWidget(self.canvas1)

        slider_row = QtWidgets.QHBoxLayout()
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.update_frame_display)
        self.frame_label = QtWidgets.QLabel("Frame: - / -")
        slider_row.addWidget(self.frame_slider, 1)
        slider_row.addWidget(self.frame_label)
        preview_layout.addLayout(slider_row)

        return panel

    # ---- Tab 3: g2(tau) --------------------------------------------------
    def _build_g2_tab(self):
        panel = QtWidgets.QWidget()
        g2_layout = QtWidgets.QVBoxLayout(panel)

        roi_row = QtWidgets.QHBoxLayout()
        self.g2_fields = {}
        for key, label, default in [
            ("BinNum", "Q-bin 개수", "5"),
            ("StartPixel", "ROI 시작 pixel", "40"),
            ("EndPixel", "ROI 끝 pixel", "120"),
        ]:
            roi_row.addWidget(QtWidgets.QLabel(label + ":"))
            edit = QtWidgets.QLineEdit(default)
            edit.setFixedWidth(60)
            roi_row.addWidget(edit)
            self.g2_fields[key] = edit
        roi_row.addStretch(1)
        g2_layout.addLayout(roi_row)

        ctrl_row = QtWidgets.QHBoxLayout()
        self.baseline_check = QtWidgets.QCheckBox("Baseline 보정 (마지막 30pt median 빼기)")
        self.baseline_check.setChecked(True)
        ctrl_row.addWidget(self.baseline_check)

        ctrl_row.addWidget(QtWidgets.QLabel("Lag points:"))
        self.lagpoints_edit = QtWidgets.QLineEdit("100")
        self.lagpoints_edit.setFixedWidth(60)
        ctrl_row.addWidget(self.lagpoints_edit)

        self.g2_btn = QtWidgets.QPushButton("Compute ROI && g2")
        self.g2_btn.setEnabled(False)
        self.g2_btn.clicked.connect(self.on_compute_g2)
        ctrl_row.addWidget(self.g2_btn)
        ctrl_row.addStretch(1)
        g2_layout.addLayout(ctrl_row)

        self.g2_progress = QtWidgets.QProgressBar()
        self.g2_progress.setVisible(False)
        g2_layout.addWidget(self.g2_progress)

        self.fig2 = Figure(figsize=(10, 4.5))
        self.ax_g2_img = self.fig2.add_subplot(1, 2, 1)
        self.ax_g2 = self.fig2.add_subplot(1, 2, 2)
        self.canvas2 = FigureCanvas(self.fig2)
        toolbar2 = NavigationToolbar(self.canvas2, panel)
        g2_layout.addWidget(toolbar2)
        g2_layout.addWidget(self.canvas2)

        note = QtWidgets.QLabel(
            "왼쪽: 전체 프레임을 평균한 speckle 패턴 위에 선택된 Q-bin(ROI)이 표시됩니다.\n"
            "BinNum/StartPixel/EndPixel 은 [Compute ROI & g2] 를 누르는 시점의 값이 "
            "재시뮬레이션 없이 바로 반영됩니다."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        g2_layout.addWidget(note)

        return panel

    # ---- Tab 4: I(q) ------------------------------------------------------
    def _build_iq_tab(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.addWidget(QtWidgets.QLabel("Pixel binning (bins):"))
        self.iq_bins_edit = QtWidgets.QLineEdit("100")
        self.iq_bins_edit.setFixedWidth(60)
        ctrl_row.addWidget(self.iq_bins_edit)

        ctrl_row.addWidget(QtWidgets.QLabel("r_min [px]:"))
        self.iq_rmin_edit = QtWidgets.QLineEdit("0")
        self.iq_rmin_edit.setFixedWidth(50)
        ctrl_row.addWidget(self.iq_rmin_edit)

        ctrl_row.addWidget(QtWidgets.QLabel("r_max [px] (비우면 전체):"))
        self.iq_rmax_edit = QtWidgets.QLineEdit("")
        self.iq_rmax_edit.setFixedWidth(60)
        ctrl_row.addWidget(self.iq_rmax_edit)

        self.iq_btn = QtWidgets.QPushButton("Compute I(q)")
        self.iq_btn.setEnabled(False)
        self.iq_btn.clicked.connect(self.on_compute_iq)
        ctrl_row.addWidget(self.iq_btn)
        ctrl_row.addStretch(1)
        layout.addLayout(ctrl_row)

        self.iq_progress = QtWidgets.QProgressBar()
        self.iq_progress.setVisible(False)
        layout.addWidget(self.iq_progress)

        self.fig_iq = Figure(figsize=(7, 4.5))
        self.ax_iq = self.fig_iq.add_subplot(1, 1, 1)
        self.canvas_iq = FigureCanvas(self.fig_iq)
        toolbar_iq = NavigationToolbar(self.canvas_iq, panel)
        layout.addWidget(toolbar_iq)
        layout.addWidget(self.canvas_iq)

        note = QtWidgets.QLabel(
            "시뮬레이션에서 나온 모든 프레임을 시간 평균한 speckle 패턴에 "
            "azimuthal_integration() (Particle_Sim_module.py) 을 그대로 적용합니다."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(note)

        return panel

    # ---- Tab 5: Gamma(q) & D fit -------------------------------------------
    def _build_fit_tab(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.addWidget(QtWidgets.QLabel("Temperature T [K]:"))
        self.temp_edit = QtWidgets.QLineEdit("298")
        self.temp_edit.setFixedWidth(60)
        ctrl_row.addWidget(self.temp_edit)

        ctrl_row.addWidget(QtWidgets.QLabel("Viscosity η [Pa·s]:"))
        self.visc_edit = QtWidgets.QLineEdit("1.0e-3")
        self.visc_edit.setFixedWidth(70)
        ctrl_row.addWidget(self.visc_edit)

        self.fit_step1_btn = QtWidgets.QPushButton("1) Fit Γ per bin")
        self.fit_step1_btn.setEnabled(False)
        self.fit_step1_btn.clicked.connect(self.on_fit_step1)
        ctrl_row.addWidget(self.fit_step1_btn)

        self.fit_step2_btn = QtWidgets.QPushButton("2) Fit D (선택된 bin만)")
        self.fit_step2_btn.setEnabled(False)
        self.fit_step2_btn.clicked.connect(self.on_fit_step2)
        ctrl_row.addWidget(self.fit_step2_btn)
        ctrl_row.addStretch(1)
        layout.addLayout(ctrl_row)

        self.gamma_table = QtWidgets.QTableWidget(0, 6)
        self.gamma_table.setHorizontalHeaderLabels(
            ["D fit에 사용", "bin", "q [1/nm]", "q² [1/nm²]", "Γ [1/s]", "Γ 오차"]
        )
        self.gamma_table.horizontalHeader().setStretchLastSection(True)
        self.gamma_table.setMaximumHeight(180)
        layout.addWidget(self.gamma_table)

        self.fit_result_label = QtWidgets.QLabel(
            "g2(τ) 탭에서 g2를 먼저 계산한 뒤 '1) Fit Γ per bin'을 누르세요."
        )
        self.fit_result_label.setWordWrap(True)
        self.fit_result_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self.fit_result_label)

        self.fig3 = Figure(figsize=(7, 4.0))
        self.ax_fit = self.fig3.add_subplot(1, 1, 1)
        self.canvas3 = FigureCanvas(self.fig3)
        toolbar3 = NavigationToolbar(self.canvas3, panel)
        layout.addWidget(toolbar3)
        layout.addWidget(self.canvas3)

        note = QtWidgets.QLabel(
            "1단계: Siegert 관계 g2(τ)-baseline = beta*exp(-2*Gamma*τ) 로 각 bin의 g2를 "
            "개별 피팅해서 Gamma(q) 표를 만듭니다.\n"
            "2단계: 표에서 체크된(outlier 제외) bin만 사용해서 Gamma = D*q² 선형피팅으로 "
            "D를 구하고, Stokes-Einstein 관계식 D = kT/(6πηa) 로 유체역학적 반경 a를 계산합니다.\n"
            "주의: 이 시뮬레이션은 실공간 길이 단위(pix2real)를 사용자가 정하는 toy model이므로, "
            "여기서 나오는 반경 값을 실제 물리적 입자 크기와 직접 비교하려면 pix2real/jump가 "
            "실제 실험 조건과 일치하도록 맞춰야 합니다."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(note)

        return panel

    # ------------------------------------------------------------------
    def _load_defaults(self):
        obj = build_default_obj()
        pix2real = obj["sample_pixelsize"]

        defaults = {
            "SDD": str(obj["SDD"]),
            "Energy": str(obj["Energy"]),
            "pixelsize": str(obj["pixelsize"]),
            "pix2real": str(pix2real),
            "dimx_real": str(obj["dimx"] * pix2real),
            "dimy_real": str(obj["dimy"] * pix2real),
            "r_real": str(obj["r"] * pix2real),
            "jump_real": str(obj["jump"] * pix2real),
            "seed": str(obj["seed"]),
            "timestep": str(obj["timestep"]),
            "T_total": str((obj["dimt"] - 1) * obj["timestep"]),
        }
        for key, edit in self.fields.items():
            edit.setText(defaults[key])
        self.bc_combo.setCurrentText(obj["BC"])

        self.g2_fields["BinNum"].setText(str(obj["BinNum"]))
        self.g2_fields["StartPixel"].setText(str(obj["StartPixel"]))
        self.g2_fields["EndPixel"].setText(str(obj["EndPixel"]))
        self.baseline_check.setChecked(True)
        self.lagpoints_edit.setText("100")

        self.iq_bins_edit.setText("100")
        self.iq_rmin_edit.setText("0")
        self.iq_rmax_edit.setText("")

        self.temp_edit.setText("298")
        self.visc_edit.setText("1.0e-3")

    def _collect_obj_from_form(self):
        obj = {}
        obj["SDD"] = parse_number(self.fields["SDD"].text())
        obj["Energy"] = parse_number(self.fields["Energy"].text())
        obj["pixelsize"] = parse_number(self.fields["pixelsize"].text())

        pix2real = parse_number(self.fields["pix2real"].text())
        if pix2real <= 0:
            raise ValueError("pix2real 은 0보다 커야 합니다.")
        obj["sample_pixelsize"] = pix2real

        dimx_real = parse_number(self.fields["dimx_real"].text())
        dimy_real = parse_number(self.fields["dimy_real"].text())
        r_real = parse_number(self.fields["r_real"].text())
        jump_real = parse_number(self.fields["jump_real"].text())

        obj["dimx"] = px_from_real(dimx_real, pix2real, minimum=20)
        obj["dimy"] = px_from_real(dimy_real, pix2real, minimum=20)
        obj["r"] = px_from_real(r_real, pix2real, minimum=1)
        obj["jump"] = px_from_real(jump_real, pix2real, minimum=0)

        obj["seed"] = int(parse_number(self.fields["seed"].text(), cast=int))
        obj["BC"] = self.bc_combo.currentText()
        obj["timestep"] = parse_number(self.fields["timestep"].text())
        if obj["timestep"] <= 0:
            raise ValueError("timestep 은 0보다 커야 합니다.")

        T_total = parse_number(self.fields["T_total"].text())
        if T_total <= 0:
            raise ValueError("T_total 은 0보다 커야 합니다.")
        obj["dimt"] = max(3, int(round(T_total / obj["timestep"])) + 1)

        if obj["seed"] < 1:
            raise ValueError("입자 개수는 1 이상이어야 합니다.")

        return obj

    # ------------------------------------------------------------------
    def on_run_simulation(self):
        try:
            obj = self._collect_obj_from_form()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
            return

        if obj["jump"] == 0:
            QtWidgets.QMessageBox.information(
                self, "참고",
                "Brownian jump 가 pix2real 기준으로 반올림하면 0 픽셀이 됩니다.\n"
                "입자가 움직이지 않아 g2가 항상 1이 됩니다 - jump 값이나 pix2real을 조정해보세요."
            )

        self.run_btn.setEnabled(False)
        self.g2_btn.setEnabled(False)
        self.iq_btn.setEnabled(False)
        self.fit_step1_btn.setEnabled(False)
        self.fit_step2_btn.setEnabled(False)
        self.sim_progress.setVisible(True)
        self.status_label.setText("시뮬레이션 시작...")

        self.sim_thread = SimThread(obj)
        self.sim_thread.progress.connect(self.status_label.setText)
        self.sim_thread.finished_ok.connect(self.on_sim_finished)
        self.sim_thread.failed.connect(self.on_sim_failed)
        self.sim_thread.start()

    def on_sim_finished(self, particle_img, spec_img, obj2):
        self.particle_img = particle_img
        self.spec_img = spec_img
        self.avg_img = np.mean(spec_img, axis=0)
        self.obj = obj2
        self.roi_result = None
        self.iq_result = None
        self.gamma_per_bin = None
        self.gamma_table.setRowCount(0)

        n_frames = spec_img.shape[0]
        self.frame_slider.setEnabled(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(n_frames - 1)
        self.frame_slider.setValue(0)
        self.update_frame_display(0)
        self.update_g2_image_display()

        self.run_btn.setEnabled(True)
        self.g2_btn.setEnabled(True)
        self.iq_btn.setEnabled(True)
        self.fit_step1_btn.setEnabled(False)
        self.fit_step2_btn.setEnabled(False)
        self.sim_progress.setVisible(False)
        self.status_label.setText(
            f"시뮬레이션 완료: {n_frames} frames (dimt={obj2['dimt']}), "
            f"dimx={obj2['dimx']}px, dimy={obj2['dimy']}px, r={obj2['r']}px, "
            f"jump={obj2['jump']}px, speckle image shape={spec_img.shape[1:]}"
        )
        self.tabs.setCurrentIndex(1)  # Preview 탭으로 자동 이동

    def on_sim_failed(self, tb_text):
        self.run_btn.setEnabled(True)
        self.sim_progress.setVisible(False)
        self.status_label.setText("시뮬레이션 실패.")
        QtWidgets.QMessageBox.critical(self, "시뮬레이션 오류", tb_text)

    # ------------------------------------------------------------------
    def update_frame_display(self, idx):
        if self.spec_img is None:
            return
        self.frame_label.setText(f"Frame: {idx} / {self.spec_img.shape[0] - 1}")

        self.ax0.clear()
        self.ax1.clear()

        self.ax0.imshow(self.particle_img[idx], cmap="gray")
        self.ax0.set_title("Particle image")

        img = self.spec_img[idx]
        positive = img[img > 0]
        vmin = float(positive.min()) if positive.size else 1e-3
        vmax = float(img.max()) if img.max() > vmin else vmin * 10
        try:
            self.ax1.imshow(img, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
        except ValueError:
            self.ax1.imshow(img, cmap="viridis")
        self.ax1.set_title("Speckle pattern")

        if self.roi_result is not None:
            indexmat = self.roi_result["Indexmat"]
            colors = get_discrete_cmap("tab10", len(indexmat))
            for ii, idx_arr in enumerate(indexmat):
                if len(idx_arr) == 0:
                    continue
                self.ax1.plot(
                    idx_arr[:, 0], idx_arr[:, 1], ".",
                    markersize=1, alpha=0.35, color=colors(ii),
                )

        self.fig1.tight_layout()
        self.canvas1.draw_idle()

    # ------------------------------------------------------------------
    def update_g2_image_display(self):
        """g2(τ) 탭의 왼쪽 패널: 전체 프레임 평균 speckle 패턴 + 현재 ROI(Q-bin) 오버레이."""
        if self.avg_img is None:
            return
        self.ax_g2_img.clear()

        img = self.avg_img
        positive = img[img > 0]
        vmin = float(positive.min()) if positive.size else 1e-3
        vmax = float(img.max()) if img.max() > vmin else vmin * 10
        try:
            self.ax_g2_img.imshow(img, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
        except ValueError:
            self.ax_g2_img.imshow(img, cmap="viridis")
        self.ax_g2_img.set_title("Speckle pattern (전체 프레임 평균)")

        if self.roi_result is not None:
            indexmat = self.roi_result["Indexmat"]
            colors = get_discrete_cmap("tab10", len(indexmat))
            for ii, idx_arr in enumerate(indexmat):
                if len(idx_arr) == 0:
                    continue
                self.ax_g2_img.plot(
                    idx_arr[:, 0], idx_arr[:, 1], ".",
                    markersize=1, alpha=0.4, color=colors(ii),
                )

        self.fig2.tight_layout()
        self.canvas2.draw_idle()

    # ------------------------------------------------------------------
    def on_compute_g2(self):
        if self.spec_img is None or self.obj is None:
            return
        try:
            n_lag = int(parse_number(self.lagpoints_edit.text(), cast=int))
            if n_lag < 5:
                raise ValueError("Lag points 는 5 이상이어야 합니다.")

            # [FIX] BinNum/StartPixel/EndPixel 은 항상 이 탭의 "현재" 값을 다시
            # 읽어서 반영한다 (재시뮬레이션 없이 ROI만 바꿔서 여러 번 계산 가능).
            binnum = int(parse_number(self.g2_fields["BinNum"].text(), cast=int))
            startpix = int(parse_number(self.g2_fields["StartPixel"].text(), cast=int))
            endpix = int(parse_number(self.g2_fields["EndPixel"].text(), cast=int))
            if binnum < 1:
                raise ValueError("BinNum 은 1 이상이어야 합니다.")
            if endpix <= startpix:
                raise ValueError("EndPixel 은 StartPixel 보다 커야 합니다.")
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
            return

        roi_obj = dict(self.obj)
        roi_obj["BinNum"] = binnum
        roi_obj["StartPixel"] = startpix
        roi_obj["EndPixel"] = endpix

        self.g2_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.fit_step1_btn.setEnabled(False)
        self.fit_step2_btn.setEnabled(False)
        self.g2_progress.setVisible(True)
        self.g2_progress.setRange(0, 0)
        self.status_label.setText("ROI/g2 계산 시작...")

        self.g2_thread = G2Thread(
            self.spec_img, roi_obj, self.baseline_check.isChecked(), n_lag
        )
        self.g2_thread.progress.connect(self.status_label.setText)
        self.g2_thread.finished_ok.connect(self.on_g2_finished)
        self.g2_thread.failed.connect(self.on_g2_failed)
        self.g2_thread.start()

    def on_g2_finished(self, res):
        self.roi_result = res
        self.gamma_per_bin = None
        self.gamma_table.setRowCount(0)
        self.plot_g2(res)
        self.update_g2_image_display()               # 평균 이미지에 ROI 오버레이 갱신
        self.update_frame_display(self.frame_slider.value())  # Preview 탭 오버레이도 갱신

        self.g2_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        self.fit_step1_btn.setEnabled(True)
        self.fit_step2_btn.setEnabled(False)
        self.g2_progress.setVisible(False)

        n_zero = sum(1 for n in res["npix_list"] if n == 0)
        msg = "g2 계산 완료."
        if n_zero > 0:
            msg += f" (경고: {n_zero}개 bin에 픽셀이 없습니다 - StartPixel/EndPixel/BinNum을 확인하세요)"
        self.status_label.setText(msg)
        self.fit_result_label.setText("'1) Fit Γ per bin' 버튼을 눌러 Gamma(q) 표를 만드세요.")

    def on_g2_failed(self, tb_text):
        self.g2_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        self.g2_progress.setVisible(False)
        self.status_label.setText("g2 계산 실패.")
        QtWidgets.QMessageBox.critical(self, "g2 계산 오류", tb_text)

    def plot_g2(self, res):
        self.ax_g2.clear()
        colors = get_discrete_cmap("tab10", len(res["g2_list"]))
        for ii, g2_ in enumerate(res["g2_list"]):
            q_nm = res["q_centers_nm"][ii]
            npix = res["npix_list"][ii]
            self.ax_g2.plot(
                res["dtlist"], g2_, marker=".", markersize=4, linewidth=0.8,
                color=colors(ii),
                label=f"bin {ii}  q={q_nm:.4f} 1/nm  (n={npix})",
            )
        self.ax_g2.set_xscale("log")
        ylabel = "g2(τ)" + (" - baseline" if self.baseline_check.isChecked() else "")
        self.ax_g2.set_xlabel("τ (s)")
        self.ax_g2.set_ylabel(ylabel)
        self.ax_g2.legend(fontsize=7)
        self.ax_g2.grid(True, which="both", alpha=0.3)
        self.fig2.tight_layout()
        self.canvas2.draw_idle()

    # ------------------------------------------------------------------
    def on_compute_iq(self):
        if self.spec_img is None or self.obj is None:
            return
        try:
            bins = int(parse_number(self.iq_bins_edit.text(), cast=int))
            if bins < 4:
                raise ValueError("bins 는 4 이상이어야 합니다.")
            r_min = parse_number(self.iq_rmin_edit.text())
            r_max_text = self.iq_rmax_edit.text().strip()
            r_max = None if r_max_text == "" else parse_number(r_max_text)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
            return

        self.iq_btn.setEnabled(False)
        self.iq_progress.setVisible(True)
        self.iq_progress.setRange(0, 0)
        self.status_label.setText("I(q) 계산 시작...")

        self.iq_thread = IqThread(self.spec_img, self.obj, bins, r_min, r_max)
        self.iq_thread.progress.connect(self.status_label.setText)
        self.iq_thread.finished_ok.connect(self.on_iq_finished)
        self.iq_thread.failed.connect(self.on_iq_failed)
        self.iq_thread.start()

    def on_iq_finished(self, res):
        self.iq_result = res
        self.ax_iq.clear()
        self.ax_iq.errorbar(
            res["q_nm"], res["I_mean"], yerr=res["I_std"],
            fmt="-o", markersize=3, linewidth=0.8, capsize=2,
        )
        self.ax_iq.set_xscale("log")
        self.ax_iq.set_yscale("log")
        self.ax_iq.set_xlabel("q [1/nm]")
        self.ax_iq.set_ylabel("I(q)  (time-averaged, a.u.)")
        self.ax_iq.grid(True, which="both", alpha=0.3)
        self.fig_iq.tight_layout()
        self.canvas_iq.draw_idle()

        self.iq_btn.setEnabled(True)
        self.iq_progress.setVisible(False)
        self.status_label.setText("I(q) 계산 완료.")

    def on_iq_failed(self, tb_text):
        self.iq_btn.setEnabled(True)
        self.iq_progress.setVisible(False)
        self.status_label.setText("I(q) 계산 실패.")
        QtWidgets.QMessageBox.critical(self, "I(q) 계산 오류", tb_text)

    # ------------------------------------------------------------------
    def on_fit_step1(self):
        if self.roi_result is None:
            return
        try:
            per_bin = fit_gamma_per_bin(
                self.roi_result["dtlist"], self.roi_result["g2_list"],
                self.roi_result["q_centers_nm"], self.roi_result["npix_list"],
            )
        except Exception:
            QtWidgets.QMessageBox.critical(self, "피팅 오류", traceback.format_exc())
            return

        self.gamma_per_bin = per_bin
        n = len(per_bin["gamma"])
        self.gamma_table.setRowCount(n)
        for i in range(n):
            ok = bool(per_bin["fit_ok"][i])

            chk_item = QtWidgets.QTableWidgetItem()
            if ok:
                chk_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                chk_item.setCheckState(QtCore.Qt.Checked)
            else:
                chk_item.setFlags(QtCore.Qt.NoItemFlags)
                chk_item.setCheckState(QtCore.Qt.Unchecked)
            self.gamma_table.setItem(i, 0, chk_item)

            self.gamma_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(i)))
            self.gamma_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{per_bin['q_nm'][i]:.5f}"))
            self.gamma_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{per_bin['q2_nm2'][i]:.6f}"))
            gam_txt = f"{per_bin['gamma'][i]:.4g}" if np.isfinite(per_bin["gamma"][i]) else "fit 실패"
            self.gamma_table.setItem(i, 4, QtWidgets.QTableWidgetItem(gam_txt))
            err_txt = f"{per_bin['gamma_err'][i]:.2g}" if np.isfinite(per_bin["gamma_err"][i]) else "-"
            self.gamma_table.setItem(i, 5, QtWidgets.QTableWidgetItem(err_txt))

        self.fit_step2_btn.setEnabled(bool(np.any(per_bin["fit_ok"])))
        n_ok = int(np.sum(per_bin["fit_ok"]))
        self.fit_result_label.setText(
            f"{n_ok}/{n} bin 피팅 성공. 표에서 D 피팅에 쓸 bin(outlier 체크 해제)을 고른 뒤 "
            "'2) Fit D'를 누르세요."
        )
        self.plot_gamma_points(per_bin, include_mask=per_bin["fit_ok"])

    def on_fit_step2(self):
        if self.gamma_per_bin is None:
            return
        try:
            T_K = parse_number(self.temp_edit.text(), cast=float)
            eta = parse_number(self.visc_edit.text(), cast=float)
            if T_K <= 0 or eta <= 0:
                raise ValueError("Temperature 와 Viscosity 는 0보다 커야 합니다.")
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "입력 오류", str(e))
            return

        per_bin = self.gamma_per_bin
        n = self.gamma_table.rowCount()
        include_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            item = self.gamma_table.item(i, 0)
            include_mask[i] = item is not None and item.checkState() == QtCore.Qt.Checked

        d_fit = fit_D_from_gamma(per_bin["q2_nm2"], per_bin["gamma"], per_bin["gamma_err"], include_mask)
        self.plot_gamma_points(per_bin, include_mask=include_mask, d_fit=d_fit)

        if d_fit["n_used"] < 2:
            self.fit_result_label.setText(
                f"선택된 bin이 {d_fit['n_used']}개뿐입니다 (D 피팅에는 2개 이상 필요). "
                "표에서 체크박스를 더 선택해보세요."
            )
            return

        D_nm2_s = d_fit["D_fit_nm2_s"]
        D_err = d_fit["D_fit_err_nm2_s"]

        if not np.isfinite(D_nm2_s) or D_nm2_s <= 0:
            self.fit_result_label.setText(
                f"D fit = {D_nm2_s:.4g} nm^2/s (0 이하 또는 유효하지 않음) - "
                "신뢰할 수 없는 피팅입니다. 선택한 bin이나 g2 품질을 다시 확인해보세요."
            )
            return

        a_nm = stokes_einstein_radius_nm(D_nm2_s, T_K=T_K, eta_Pa_s=eta)

        text = (
            f"D (fit)        = {D_nm2_s:.4g} ± {D_err:.2g}  nm^2/s\n"
            f"                = {D_nm2_s*1e-6:.4g}  um^2/s\n"
            f"                = {D_nm2_s*1e-18:.4g}  m^2/s\n"
            f"Stokes-Einstein radius a = {a_nm:.4g} nm   (T={T_K:g} K, η={eta:g} Pa·s)\n"
            f"사용된 bin 수 = {d_fit['n_used']} / {n}"
        )
        self.fit_result_label.setText(text)

    def plot_gamma_points(self, per_bin, include_mask, d_fit=None):
        self.ax_fit.clear()
        q2 = per_bin["q2_nm2"]
        gamma = per_bin["gamma"]
        gamma_err = per_bin["gamma_err"]
        fit_ok = per_bin["fit_ok"]
        include_mask = np.asarray(include_mask, dtype=bool)

        used = include_mask & fit_ok
        excluded = fit_ok & ~include_mask

        if np.any(used):
            self.ax_fit.errorbar(
                q2[used], gamma[used], yerr=gamma_err[used],
                fmt="o", capsize=3, color="tab:blue", label="D fit에 사용",
            )
        if np.any(excluded):
            self.ax_fit.errorbar(
                q2[excluded], gamma[excluded], yerr=gamma_err[excluded],
                fmt="o", mfc="none", capsize=3, color="tab:red", label="제외됨",
            )

        if d_fit is not None and d_fit.get("q2_fit_line", np.array([])).size:
            self.ax_fit.plot(
                d_fit["q2_fit_line"], d_fit["gamma_fit_line"], "--", color="black",
                label=f"linear fit: D={d_fit['D_fit_nm2_s']:.3g} nm²/s",
            )

        self.ax_fit.set_xlabel("q² [1/nm²]")
        self.ax_fit.set_ylabel("Γ [1/s]")
        self.ax_fit.legend(fontsize=8)
        self.ax_fit.grid(True, alpha=0.3)
        self.fig3.tight_layout()
        self.canvas3.draw_idle()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = XPCSMainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
