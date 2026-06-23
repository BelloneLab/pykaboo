"""
Application theme: global stylesheet and themed workspace widgets.

The stylesheet references small glyph PNGs (combo chevrons, checkbox
checkmarks, spin-button arrows) generated at launch by
branding.ensure_theme_assets(), because Qt stylesheets can only load such
glyphs from image files.

Public surface:
- build_app_stylesheet(): the full QSS string with glyph paths resolved.
- WorkspaceSplitter: a QSplitter whose handle paints a visible grip pill.
- ChipLabel: a QLabel-based status chip that can shrink below its text width.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy, QSplitter, QSplitterHandle

from branding import ensure_theme_assets


class ChipLabel(QLabel):
    """Status chip whose minimum width is zero.

    A plain QLabel reports its full text width as its minimum size, which makes
    a row of chips pin the whole window wide. This keeps the natural width as
    the preferred size (so chips look right when there is room) while letting
    the layout shrink them when space is tight.
    """

    def __init__(self, text: str = "", parent=None):
        """Create the chip with a shrinkable (Preferred/Fixed) size policy."""
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def minimumSizeHint(self) -> QSize:
        """Report a zero minimum width (full height) so layouts may shrink it."""
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())


class WorkspaceSplitterHandle(QSplitterHandle):
    """Splitter handle that paints a visible rounded grip pill."""

    def __init__(self, orientation, parent):
        """Enable hover tracking so the grip can brighten under the cursor."""
        super().__init__(orientation, parent)
        self._hovered = False
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def event(self, event):
        """Track hover enter/leave to toggle the grip's highlighted state."""
        if event.type() in (QEvent.Type.HoverEnter, QEvent.Type.HoverMove):
            if not self._hovered:
                self._hovered = True
                self.update()
        elif event.type() == QEvent.Type.HoverLeave:
            self._hovered = False
            self.update()
        return super().event(event)

    def paintEvent(self, _event):
        """Draw a centred rounded "pill" grip, brighter and thicker on hover."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#5fb2ff") if self._hovered else QColor("#2e4a6c"))
        rect = self.rect()
        if self.orientation() == Qt.Orientation.Vertical:
            pill_w = min(120, max(56, rect.width() // 8))
            pill_h = 5 if self._hovered else 4
            pill = QRectF(
                rect.center().x() - pill_w / 2,
                rect.center().y() - pill_h / 2,
                pill_w,
                pill_h,
            )
        else:
            pill_h = min(120, max(56, rect.height() // 8))
            pill_w = 5 if self._hovered else 4
            pill = QRectF(
                rect.center().x() - pill_w / 2,
                rect.center().y() - pill_h / 2,
                pill_w,
                pill_h,
            )
        painter.drawRoundedRect(pill, min(pill.width(), pill.height()) / 2, min(pill.width(), pill.height()) / 2)
        painter.end()


class WorkspaceSplitter(QSplitter):
    """QSplitter whose handles render the themed grip pill."""

    def createHandle(self):
        """Return a WorkspaceSplitterHandle so the divider shows the grip."""
        return WorkspaceSplitterHandle(self.orientation(), self)


_APP_STYLESHEET = """
    QMainWindow {
        background-color: #060c14;
        color: #e8f1fb;
    }
    QWidget {
        background-color: transparent;
        color: #e8f1fb;
        font-family: "Arial Narrow", Arial, "Segoe UI";
        font-size: 11px;
    }
    QDialog, QMessageBox {
        background-color: #0a131f;
    }
    QWidget#AppShell {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #050a12, stop:0.5 #08111c, stop:1 #04080f);
    }

    /* ===== Shells and cards ===== */
    QFrame#SideRail {
        background-color: #081220;
        border: 1px solid #182a40;
        border-radius: 20px;
    }
    QFrame#PanelShell {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #0b1522, stop:1 #0c1726);
        border: 1px solid #1f3148;
        border-radius: 20px;
    }
    QFrame#PanelHeader {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #0f1a2b, stop:1 #122134);
        border: 1px solid #24384f;
        border-radius: 14px;
    }
    QFrame#WorkspaceCard {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #0b1521, stop:1 #0c1725);
        border: 1px solid #1f3349;
        border-radius: 16px;
    }
    QFrame#WorkspaceSubCard {
        background-color: #0d1828;
        border: 1px solid #243a55;
        border-radius: 12px;
    }
    QFrame#MetricTile {
        background-color: #0d1827;
        border: 1px solid #223750;
        border-radius: 11px;
    }
    QGroupBox {
        border: 1px solid #213650;
        border-radius: 12px;
        margin-top: 15px;
        padding-top: 10px;
        background-color: #0c1622;
        font-weight: 700;
        color: #e3eefb;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 7px;
        background-color: #0c1622;
        color: #79bdf2;
        border-radius: 6px;
    }

    /* ===== Buttons ===== */
    QPushButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #3d8bf0, stop:0.5 #2f7fe6, stop:1 #2258bd);
        color: #f4f9ff;
        border: 1px solid rgba(124, 184, 255, 0.5);
        border-radius: 11px;
        padding: 6px 16px;
        font-weight: 700;
        font-size: 12px;
        min-height: 24px;
        outline: 0;
    }
    QPushButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4d99fb, stop:0.5 #3a8cf5, stop:1 #2b6cd8);
        border-color: rgba(160, 206, 255, 0.78);
    }
    QPushButton:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #1f59b8, stop:1 #16448c);
    }
    /* Focus: keep the 1px border width (no layout jiggle), jump to a bright
       near-white ring so keyboard focus is unmistakable over the gradient. */
    QPushButton:focus {
        border: 1px solid #cfe7ff;
    }
    QPushButton#successButton:focus { border: 1px solid #d6ffe6; }
    QPushButton#dangerButton:focus { border: 1px solid #ffd6de; }
    QPushButton#violetButton:focus { border: 1px solid #ecd9ff; }
    QPushButton#orangeButton:focus { border: 1px solid #ffe6cf; }
    QPushButton#ghostButton:focus, QPushButton#toggleButton:focus {
        border: 1px solid #8cc7ff;
    }
    QPushButton:disabled {
        background: #16202f;
        color: #5b6f87;
        border-color: #223349;
    }
    QPushButton#ghostButton {
        background: rgba(157, 196, 240, 0.06);
        border: 1px solid #2b4364;
        color: #d4e5f7;
        font-weight: 600;
    }
    QPushButton#ghostButton:hover {
        background: rgba(157, 196, 240, 0.13);
        border-color: #3f6491;
    }
    QPushButton#ghostButton:pressed {
        background: rgba(157, 196, 240, 0.03);
    }
    QPushButton#toggleButton {
        background: rgba(157, 196, 240, 0.05);
        border: 1px solid #2b4364;
        color: #9cc3e8;
        border-radius: 11px;
        padding: 6px 13px;
        font-weight: 600;
    }
    QPushButton#toggleButton:hover {
        background: rgba(157, 196, 240, 0.12);
        border-color: #46739f;
        color: #eef6ff;
    }
    QPushButton#toggleButton:checked {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #14365c, stop:1 #1b4f80);
        border: 1px solid #5cb2ff;
        color: #f0f8ff;
    }
    QPushButton#successButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #4cdd84, stop:0.5 #3ecf70, stop:1 #239b4d);
        border: 1px solid rgba(110, 235, 150, 0.6);
        color: #04120a;
    }
    QPushButton#successButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #5fe793, stop:0.5 #52dd82, stop:1 #2fbb61);
    }
    QPushButton#successButton:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #2bb15f, stop:1 #1d8a45);
    }
    QPushButton#dangerButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #ff6b7e, stop:0.5 #ff5b70, stop:1 #d6356a);
        border: 1px solid rgba(255, 140, 165, 0.55);
        color: #ffffff;
    }
    QPushButton#dangerButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #ff8194, stop:0.5 #ff7186, stop:1 #f44f86);
    }
    QPushButton#dangerButton:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #e0455c, stop:1 #c02a60);
    }
    QPushButton#violetButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #9a73ff, stop:0.5 #8a63ff, stop:1 #9b40e0);
        border: 1px solid rgba(206, 150, 255, 0.55);
    }
    QPushButton#violetButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #aa87ff, stop:0.5 #9a77ff, stop:1 #b964f5);
    }
    QPushButton#violetButton:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #7a53e6, stop:1 #8b34d0);
    }
    QPushButton#orangeButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #ffa057, stop:0.5 #ff9244, stop:1 #ef5733);
        border: 1px solid rgba(255, 185, 130, 0.55);
    }
    QPushButton#orangeButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #ffb16d, stop:0.5 #ffa35d, stop:1 #fb7252);
    }
    QPushButton#orangeButton:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #ef8038, stop:1 #df4f2d);
    }
    QPushButton#successButton:disabled, QPushButton#dangerButton:disabled,
    QPushButton#violetButton:disabled, QPushButton#orangeButton:disabled,
    QPushButton#ghostButton:disabled, QPushButton#toggleButton:disabled {
        background: #16202f;
        color: #5b6f87;
        border-color: #223349;
    }

    /* ===== Tool buttons ===== */
    QToolButton {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 3px;
        color: #cfe2f5;
    }
    QToolButton:hover {
        background: #13243a;
        border-color: #2e4a6d;
    }
    QToolButton#navButton {
        background: #0c1726;
        border: 1px solid #20334b;
        border-radius: 14px;
        padding: 8px;
        min-width: 44px;
        max-width: 44px;
        min-height: 44px;
        max-height: 44px;
        color: #9bb4d2;
        font-weight: 700;
    }
    QToolButton#navButton:hover {
        background: #122236;
        border-color: #3c6190;
        color: #eef6ff;
    }
    QToolButton#navButton:checked {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #133052, stop:1 #1a4470);
        border: 1px solid #66b7ff;
        border-left: 3px solid #66b7ff;
        color: #ffffff;
    }
    QToolButton#navButton:focus {
        border: 1px solid #66b7ff;
    }
    QToolButton#panelCloseButton {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 14px;
    }
    QToolButton#panelCloseButton:hover {
        background: rgba(255, 92, 116, 0.16);
        border-color: rgba(255, 92, 116, 0.5);
    }
    QToolButton#toolIconButton {
        background: rgba(157, 196, 240, 0.05);
        border: 1px solid #2b4364;
        border-radius: 9px;
    }
    QToolButton#toolIconButton:hover {
        background: rgba(157, 196, 240, 0.13);
        border-color: #3f6491;
    }
    QToolButton#toolIconButton:checked {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #14365c, stop:1 #1b4f80);
        border: 1px solid #5cb2ff;
    }
    QToolButton#toolIconButton:focus, QToolButton#ghostMenuButton:focus {
        border-color: #5cb2ff;
    }
    QToolButton#ghostMenuButton {
        background: rgba(157, 196, 240, 0.05);
        border: 1px solid #2b4364;
        border-radius: 11px;
        color: #d4e5f7;
        font-weight: 600;
        padding: 4px 10px 4px 9px;
    }
    QToolButton#ghostMenuButton:hover {
        background: rgba(157, 196, 240, 0.13);
        border-color: #3f6491;
    }
    QToolButton#ghostMenuButton::menu-indicator {
        image: url("%ASSETS%/chevron_down_sm.png");
        subcontrol-origin: padding;
        subcontrol-position: center right;
        right: 6px;
        width: 10px;
        height: 10px;
    }

    /* ===== Inputs ===== */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QTableWidget {
        background-color: #081223;
        border: 1px solid #233a55;
        border-radius: 9px;
        color: #e8f1fb;
        padding: 5px 9px;
        min-height: 20px;
        selection-background-color: #1d4a7a;
        selection-color: #ffffff;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QComboBox:focus, QTextEdit:focus {
        border-color: #4f9ff2;
        background-color: #0a1729;
    }
    QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
        border-color: #33506f;
    }
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
    QComboBox:disabled, QTextEdit:disabled {
        color: #56698f;
        background-color: #0a121d;
        border-color: #1b2b3f;
    }
    QLineEdit[readOnly="true"] {
        color: #a7bdd6;
        background-color: #0a141f;
    }

    /* Spin boxes: themed stepper buttons */
    QSpinBox, QDoubleSpinBox {
        padding-right: 24px;
    }
    QSpinBox::up-button, QDoubleSpinBox::up-button {
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 20px;
        border: none;
        border-left: 1px solid #1d3147;
        border-top-right-radius: 9px;
        background: transparent;
        margin: 1px 1px 0 0;
    }
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 20px;
        border: none;
        border-left: 1px solid #1d3147;
        border-bottom-right-radius: 9px;
        background: transparent;
        margin: 0 1px 1px 0;
    }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
        background: #16263c;
    }
    QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
    QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
        background: #1d3553;
    }
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        image: url("%ASSETS%/chevron_up_sm.png");
        width: 10px;
        height: 10px;
    }
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        image: url("%ASSETS%/chevron_down_sm.png");
        width: 10px;
        height: 10px;
    }
    QSpinBox::up-arrow:disabled, QSpinBox::up-arrow:off,
    QDoubleSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:off {
        image: url("%ASSETS%/chevron_up_sm_dim.png");
    }
    QSpinBox::down-arrow:disabled, QSpinBox::down-arrow:off,
    QDoubleSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:off {
        image: url("%ASSETS%/chevron_down_sm_dim.png");
    }

    /* Combo boxes: visible chevron so dropdowns read as dropdowns */
    QComboBox {
        padding-right: 28px;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 26px;
        border: none;
        border-left: 1px solid #1d3147;
        border-top-right-radius: 9px;
        border-bottom-right-radius: 9px;
    }
    QComboBox::down-arrow {
        image: url("%ASSETS%/chevron_down.png");
        width: 12px;
        height: 12px;
    }
    QComboBox::down-arrow:disabled {
        image: url("%ASSETS%/chevron_down_dim.png");
    }
    QComboBox QAbstractItemView {
        background-color: #0a1424;
        color: #e8f1fb;
        border: 1px solid #2a425f;
        border-radius: 10px;
        padding: 4px;
        selection-background-color: #1a3a61;
        selection-color: #ffffff;
        outline: 0;
    }
    QComboBox QAbstractItemView::item {
        min-height: 24px;
        padding: 5px 10px;
        border-radius: 6px;
        color: #e8f1fb;
        background: transparent;
    }
    QComboBox QAbstractItemView::item:selected {
        background-color: #1a3a61;
        color: #ffffff;
    }

    /* Check boxes: real checkmark glyph */
    QCheckBox {
        spacing: 8px;
        color: #dbe7f4;
    }
    QCheckBox:disabled {
        color: #5b6f87;
    }
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border-radius: 5px;
        border: 1px solid #33506f;
        background-color: #081223;
    }
    QCheckBox::indicator:hover {
        border-color: #4f9ff2;
    }
    QCheckBox::indicator:focus {
        border: 1px solid #5fb0ff;
    }
    QCheckBox::indicator:checked {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 #3094ff, stop:1 #2563eb);
        border-color: #5fb0ff;
        image: url("%ASSETS%/check.png");
    }
    QCheckBox::indicator:checked:disabled {
        background: #1b2940;
        border-color: #283a52;
        image: url("%ASSETS%/check_dim.png");
    }

    /* ===== Text, tooltips, menus ===== */
    QLabel {
        color: #d7e4f2;
        background: transparent;
    }
    QToolTip {
        background-color: #0e1d30;
        color: #dcebfb;
        border: 1px solid #35587f;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 11px;
    }
    QMenu {
        background-color: #0c1828;
        color: #e3eefb;
        border: 1px solid #2a425f;
        border-radius: 12px;
        padding: 6px;
    }
    QMenu::item {
        padding: 7px 24px 7px 14px;
        border-radius: 8px;
    }
    QMenu::item:selected {
        background-color: #173150;
        color: #ffffff;
    }
    QMenu::separator {
        height: 1px;
        background: #1f3551;
        margin: 5px 8px;
    }

    /* ===== Sliders and progress ===== */
    QSlider::groove:horizontal {
        height: 4px;
        background: #14233a;
        border-radius: 2px;
    }
    QSlider::sub-page:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #2f8dff, stop:1 #46c6ff);
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        width: 14px;
        height: 14px;
        margin: -6px 0;
        border-radius: 8px;
        background: #e8f4ff;
        border: 1px solid #5aa7ff;
    }
    QSlider::handle:horizontal:hover {
        background: #ffffff;
        border-color: #8cc7ff;
    }
    QProgressBar {
        background: #0a1220;
        border: 1px solid #1d2c42;
        border-radius: 5px;
        color: #d7e4f2;
        text-align: center;
    }
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #2f8dff, stop:1 #37d2ff);
        border-radius: 4px;
    }

    /* ===== Tabs: quiet underline style ===== */
    QTabWidget::pane {
        border: none;
        background: transparent;
    }
    QTabBar {
        background: transparent;
    }
    QTabBar::tab {
        background: transparent;
        color: #87a0bb;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 7px 14px 8px 14px;
        margin: 0 4px 0 0;
        font-weight: 700;
    }
    QTabBar::tab:hover {
        color: #cfe4fa;
    }
    QTabBar::tab:selected {
        color: #eef6ff;
        border-bottom: 2px solid #4da3ff;
    }
    QTabBar::tab:focus {
        color: #cfe4fa;
        border-bottom: 2px solid #4da3ff;
    }

    /* ===== Status bar ===== */
    QStatusBar {
        background-color: #07101a;
        color: #dbe7f3;
        border-top: 1px solid #16263a;
    }
    QStatusBar::item {
        border: none;
    }
    QLabel#statusMetric {
        background-color: #0d1a2b;
        border: 1px solid #1f3349;
        border-radius: 8px;
        padding: 3px 10px;
        margin: 2px;
        color: #b9cde3;
        font-weight: 600;
    }

    /* ===== Scrollbars ===== */
    QScrollBar:vertical {
        background-color: transparent;
        width: 8px;
        margin: 2px;
    }
    QScrollBar::handle:vertical {
        background-color: #24405e;
        min-height: 28px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #3a6190;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QScrollBar:horizontal {
        background-color: transparent;
        height: 8px;
        margin: 2px;
    }
    QScrollBar::handle:horizontal {
        background-color: #24405e;
        min-width: 28px;
        border-radius: 4px;
    }
    QScrollBar::handle:horizontal:hover {
        background-color: #3a6190;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    QScrollBar::add-page, QScrollBar::sub-page {
        background: transparent;
    }

    /* ===== Tables ===== */
    QHeaderView::section {
        background-color: #0e1a2b;
        color: #8fc6f5;
        border: none;
        border-right: 1px solid #1d3147;
        border-bottom: 1px solid #1d3147;
        padding: 6px 8px;
        font-weight: 700;
    }
    QTableWidget {
        gridline-color: #182c44;
        alternate-background-color: #0a1521;
        selection-background-color: #1d4f86;
        selection-color: #ffffff;
    }
    QTableWidget::item {
        padding: 3px 6px;
    }
    QTableWidget::item:hover {
        background-color: #102338;
    }
    QTableWidget::item:selected {
        background-color: #1d4f86;
        color: #ffffff;
    }
    QTableCornerButton::section {
        background-color: #0e1a2b;
        border: none;
    }

    /* ===== Misc containers ===== */
    QScrollArea {
        border: none;
        background: transparent;
    }
    QSplitter#workspaceSplitter::handle {
        background: transparent;
    }
"""


def build_app_stylesheet() -> str:
    """Return the global stylesheet with generated glyph assets resolved."""
    assets_url = ensure_theme_assets().as_posix()
    return _APP_STYLESHEET.replace("%ASSETS%", assets_url)
