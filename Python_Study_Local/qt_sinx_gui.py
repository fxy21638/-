import sys

import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("信号与系统 - sin(x)")
        self._ui_ready = False

        self._defaults = {
            "n_points": 100,
            "discrete_step": 0.5,
            "x_min": 0.0,
            "x_max": float(2 * np.pi),
            "y_min": -1.2,
            "y_max": 1.2,
            "function": "sin",
            "show_sampling": False,
        }

        central = QWidget(self)
        main_layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        # 左侧：固定宽度面板（避免交互时布局重排导致“漂移”）
        left_panel = QWidget(central)
        left_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        left_panel.setFixedWidth(320)
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel)

        # 右侧：绘图面板
        plot_panel = QWidget(central)
        plot_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot_layout = QVBoxLayout(plot_panel)
        main_layout.addWidget(plot_panel, 1)

        # Matplotlib 画布（嵌入 Qt 窗口）
        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)

        # Matplotlib 导航工具栏（放左侧，垂直显示）
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setOrientation(Qt.Orientation.Vertical)
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.toolbar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setContentsMargins(0, 0, 0, 0)
        if self.toolbar.layout() is not None:
            self.toolbar.layout().setContentsMargins(0, 0, 0, 0)
            self.toolbar.layout().setSpacing(0)
        # 某些平台/主题下 hover/focus 会改变按钮边框/内边距，导致工具栏“漂移”
        self.toolbar.setStyleSheet(
            "QToolBar { spacing: 0px; }"
            "QToolButton { padding: 2px; margin: 0px; border: 0px; }"
            "QToolButton:hover { padding: 2px; margin: 0px; border: 0px; }"
            "QToolButton:focus { padding: 2px; margin: 0px; border: 0px; }"
            "QToolButton:checked { padding: 2px; margin: 0px; border: 0px; }"
        )
        left_layout.addWidget(self.toolbar)

        # 参数区
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        left_layout.addLayout(form)

        self.n_points = QSpinBox()
        self.n_points.setRange(10, 5000)
        self.n_points.setValue(self._defaults["n_points"])
        self.n_points.setSingleStep(50)
        form.addRow("连续点数 N:", self.n_points)

        self.discrete_step = QDoubleSpinBox()
        self.discrete_step.setRange(0.01, 2.0)
        self.discrete_step.setDecimals(2)
        self.discrete_step.setSingleStep(0.05)
        self.discrete_step.setValue(self._defaults["discrete_step"])
        form.addRow("离散步长 Δx:", self.discrete_step)

        # 函数选择（正弦、余弦、冲激、阶跃）
        self.function_select = QComboBox()
        self.function_select.addItems(["sin(x)", "cos(x)", "impulse", "step"])
        # 内部映射到关键字
        self._function_map = {0: "sin", 1: "cos", 2: "impulse", 3: "step"}
        form.addRow("函数类型:", self.function_select)

        # 是否显示采样（离散点）
        self.show_sampling = QCheckBox("显示采样（离散点）")
        self.show_sampling.setChecked(self._defaults.get("show_sampling", False))
        form.addRow(self.show_sampling)

        # 新增：左侧函数切换栏目（单选按钮列） — 与上方的 function_select 保持同步
        func_group = QGroupBox("函数切换")
        func_group_layout = QVBoxLayout(func_group)
        self._func_radio_buttons = []
        labels = ["sin(x)", "cos(x)", "impulse", "step"]
        self._radio_btn_group = QButtonGroup(self)
        for i, lbl in enumerate(labels):
            rb = QRadioButton(lbl)
            func_group_layout.addWidget(rb)
            self._func_radio_buttons.append(rb)
            self._radio_btn_group.addButton(rb, i)
            rb.toggled.connect(lambda checked, idx=i: self._on_func_radio_toggled(idx, checked))

        # 根据默认值设置初始选中项
        inv_map = {val: key for key, val in self._function_map.items()}
        default_func = self._defaults.get("function", "sin")
        default_idx = inv_map.get(default_func, 0)
        if 0 <= default_idx < len(self._func_radio_buttons):
            self._func_radio_buttons[default_idx].setChecked(True)

        left_layout.addWidget(func_group)

        # 定义域范围（x 轴）
        self.x_min = QDoubleSpinBox()
        self.x_min.setRange(-1000.0, 1000.0)
        self.x_min.setDecimals(3)
        self.x_min.setSingleStep(0.5)
        self.x_min.setValue(self._defaults["x_min"])
        form.addRow("x_min:", self.x_min)

        self.x_max = QDoubleSpinBox()
        self.x_max.setRange(-1000.0, 1000.0)
        self.x_max.setDecimals(3)
        self.x_max.setSingleStep(0.5)
        self.x_max.setValue(self._defaults["x_max"])
        form.addRow("x_max:", self.x_max)

        # 值域范围（y 轴）
        self.y_min = QDoubleSpinBox()
        self.y_min.setRange(-1000.0, 1000.0)
        self.y_min.setDecimals(3)
        self.y_min.setSingleStep(0.1)
        self.y_min.setValue(self._defaults["y_min"])
        form.addRow("y_min:", self.y_min)

        self.y_max = QDoubleSpinBox()
        self.y_max.setRange(-1000.0, 1000.0)
        self.y_max.setDecimals(3)
        self.y_max.setSingleStep(0.1)
        self.y_max.setValue(self._defaults["y_max"])
        form.addRow("y_max:", self.y_max)

        buttons = QWidget(left_panel)
        buttons_layout = QHBoxLayout(buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        self.update_btn = QPushButton("更新绘图")
        self.reset_btn = QPushButton("重置参数")
        buttons_layout.addWidget(self.update_btn)
        buttons_layout.addWidget(self.reset_btn)
        left_layout.addWidget(buttons)

        left_layout.addWidget(QLabel("信息显示："))
        self.info_box = QPlainTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMinimumHeight(140)
        left_layout.addWidget(self.info_box)
        left_layout.addStretch(1)

        plot_layout.addWidget(self.canvas)

        axs = self.figure.subplots(2, 2)
        self.ax_time_cont = axs[0, 0]
        self.ax_time_disc = axs[0, 1]
        self.ax_freq_cont = axs[1, 0]
        self.ax_freq_disc = axs[1, 1]
        self._ui_ready = True
        self._plot()

        self.update_btn.clicked.connect(self._plot)
        self.reset_btn.clicked.connect(self._reset_defaults)
        self.n_points.valueChanged.connect(self._plot)
        self.discrete_step.valueChanged.connect(self._plot)
        self.x_min.valueChanged.connect(self._plot)
        self.x_max.valueChanged.connect(self._plot)
        self.y_min.valueChanged.connect(self._plot)
        self.y_max.valueChanged.connect(self._plot)
        self.function_select.currentIndexChanged.connect(self._plot)
        self.show_sampling.toggled.connect(self._plot)

        self._set_info("就绪：修改参数可实时更新")

    def _set_info(self, message: str) -> None:
        self.statusBar().showMessage(message)
        if hasattr(self, "info_box") and self.info_box is not None:
            self.info_box.setPlainText(message)

    def _reset_defaults(self) -> None:
        widgets = [
            (self.n_points, self._defaults["n_points"]),
            (self.discrete_step, self._defaults["discrete_step"]),
            (self.x_min, self._defaults["x_min"]),
            (self.x_max, self._defaults["x_max"]),
            (self.y_min, self._defaults["y_min"]),
            (self.y_max, self._defaults["y_max"]),
            (self.function_select, self._defaults.get("function", "sin")),
            (self.show_sampling, self._defaults.get("show_sampling", False)),
        ]

        for w, _ in widgets:
            w.blockSignals(True)
        try:
            for w, v in widgets:
                # QComboBox 和 QCheckBox 需要分别处理
                if isinstance(w, QComboBox):
                    # 将默认的关键字映射回索引
                    inv_map = {val: key for key, val in self._function_map.items()}
                    idx = inv_map.get(v, 0)
                    w.setCurrentIndex(idx)
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(v))
                else:
                    w.setValue(v)
        finally:
            for w, _ in widgets:
                w.blockSignals(False)

        self._plot()

    def _on_func_radio_toggled(self, idx: int, checked: bool) -> None:
        if not checked:
            return
        if not getattr(self, "_ui_ready", False):
            return
        # 将单选按钮的选择同步到组合框
        try:
            self.function_select.setCurrentIndex(idx)
        except Exception:
            pass
        # 触发更新
        self._plot()

    def _plot(self) -> None:
        if not getattr(self, "_ui_ready", False):
            return
        n = int(self.n_points.value())
        step = float(self.discrete_step.value())
        x_min = float(self.x_min.value())
        x_max = float(self.x_max.value())
        y_min = float(self.y_min.value())
        y_max = float(self.y_max.value())

        if x_min >= x_max:
            self._set_info("范围无效：需要 x_min < x_max")
            return
        if y_min >= y_max:
            self._set_info("范围无效：需要 y_min < y_max")
            return

        self.ax_time_cont.clear()
        self.ax_time_disc.clear()
        self.ax_freq_cont.clear()
        self.ax_freq_disc.clear()

        # 连续信号
        x = np.linspace(x_min, x_max, n)
        # 选择函数类型
        func_key = self._function_map.get(self.function_select.currentIndex(), "sin")
        title_map = {"sin": "y = sin(x)", "cos": "y = cos(x)", "impulse": "冲激（近似）", "step": "阶跃函数"}
        if func_key == "sin":
            y = np.sin(x)
        elif func_key == "cos":
            y = np.cos(x)
        elif func_key == "impulse":
            y = np.zeros_like(x)
            idx0 = int(np.argmin(np.abs(x)))
            y[idx0] = 1.0
        elif func_key == "step":
            y = np.where(x >= 0.0, 1.0, 0.0)
        else:
            y = np.sin(x)

        y_cont_min = float(np.min(y))
        y_cont_max = float(np.max(y))
        idx_cont_max = int(np.argmax(y))
        idx_cont_min = int(np.argmin(y))
        x_cont_at_max = float(x[idx_cont_max])
        x_cont_at_min = float(x[idx_cont_min])
        self.ax_time_cont.plot(x, y, label=None)
        self.ax_time_cont.set_title(f"{title_map.get(func_key, 'y = sin(x)')} (Continuous)")
        self.ax_time_cont.set_xlabel("x")
        self.ax_time_cont.set_ylabel("y")
        self.ax_time_cont.set_xlim(x_min, x_max)
        self.ax_time_cont.set_ylim(y_min, y_max)
        self.ax_time_cont.grid(True)
        self.ax_time_cont.legend([title_map.get(func_key, 'sin')])

        # 离散信号
        x_discrete = np.arange(x_min, x_max, step)
        if x_discrete.size == 0:
            self._set_info("离散点为空：请减小 Δx 或增大 x 范围")
            return
        if func_key == "sin":
            y_discrete = np.sin(x_discrete)
        elif func_key == "cos":
            y_discrete = np.cos(x_discrete)
        elif func_key == "impulse":
            y_discrete = np.zeros_like(x_discrete)
            idx0 = int(np.argmin(np.abs(x_discrete)))
            y_discrete[idx0] = 1.0
        elif func_key == "step":
            y_discrete = np.where(x_discrete >= 0.0, 1.0, 0.0)
        else:
            y_discrete = np.sin(x_discrete)
        y_disc_min = float(np.min(y_discrete))
        y_disc_max = float(np.max(y_discrete))
        idx_disc_max = int(np.argmax(y_discrete))
        idx_disc_min = int(np.argmin(y_discrete))
        x_disc_at_max = float(x_discrete[idx_disc_max])
        x_disc_at_min = float(x_discrete[idx_disc_min])
        x_disc_first = float(x_discrete[0])
        x_disc_last = float(x_discrete[-1])
        markerline, stemlines, baseline = self.ax_time_disc.stem(
            x_discrete,
            y_discrete,
            label=f"{title_map.get(func_key, 'sin')} (Discrete)",
        )
        plt_like_color = "C0"
        markerline.set_color(plt_like_color)
        stemlines.set_color(plt_like_color)
        baseline.set_color("0.5")

        self.ax_time_disc.set_title(f"{title_map.get(func_key, 'y = sin(x)')} (Discrete)")
        self.ax_time_disc.set_xlabel("x")
        self.ax_time_disc.set_ylabel("y")
        self.ax_time_disc.set_xlim(x_min, x_max)
        self.ax_time_disc.set_ylim(y_min, y_max)
        self.ax_time_disc.grid(True)
        self.ax_time_disc.legend()

        # 频谱（幅度谱）
        cont_dx = (x_max - x_min) / max(n - 1, 1)
        cont_freq = np.fft.rfftfreq(n, d=cont_dx)
        cont_mag = np.abs(np.fft.rfft(y)) / max(n, 1)
        self.ax_freq_cont.plot(cont_freq, cont_mag, color="C1")
        self.ax_freq_cont.set_title("Magnitude Spectrum (Continuous Sampled)")
        self.ax_freq_cont.set_xlabel("f")
        self.ax_freq_cont.set_ylabel("|X(f)|")
        self.ax_freq_cont.grid(True)

        disc_n = max(x_discrete.size, 1)
        disc_freq = np.fft.rfftfreq(disc_n, d=step)
        disc_mag = np.abs(np.fft.rfft(y_discrete)) / disc_n
        self.ax_freq_disc.plot(disc_freq, disc_mag, color="C2")
        self.ax_freq_disc.set_title("Magnitude Spectrum (Discrete)")
        self.ax_freq_disc.set_xlabel("f")
        self.ax_freq_disc.set_ylabel("|X(f)|")
        self.ax_freq_disc.grid(True)

        self.canvas.draw_idle()
        self._set_info(
            "已更新\n"
            f"- 函数 = {title_map.get(func_key, 'sin')}\n"
            f"- 连续点数 N = {n}\n"
            f"- 离散步长 Δx = {step:.2f}（离散点数 {x_discrete.size}）\n"
            f"- x 范围(显示) = [{x_min:.3f}, {x_max:.3f}]\n"
            f"- y 范围(显示) = [{y_min:.3f}, {y_max:.3f}]\n"
            f"- 连续 y 实际范围 = [{y_cont_min:.6f}, {y_cont_max:.6f}]\n"
            f"  - 连续峰值点 ≈ (x={x_cont_at_max:.6f}, y={y_cont_max:.6f})\n"
            f"  - 连续谷值点 ≈ (x={x_cont_at_min:.6f}, y={y_cont_min:.6f})\n"
            f"- 离散 y 实际范围 = [{y_disc_min:.6f}, {y_disc_max:.6f}]\n"
            f"  - 离散峰值点 = (x={x_disc_at_max:.6f}, y={y_disc_max:.6f})\n"
            f"  - 离散谷值点 = (x={x_disc_at_min:.6f}, y={y_disc_min:.6f})\n"
            f"- 离散 x 覆盖 = [{x_disc_first:.3f}, {x_disc_last:.3f}]"
        )


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(900, 700)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
