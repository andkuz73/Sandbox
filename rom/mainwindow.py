# Sand:box ROM Generator PRO 5.1 by: Andkuz (or Andkuz73 or cuzhima); tg: @cuzhima
import sys
import struct
import math
import numpy as np
import random
import re
import importlib
import cmath
from typing import Dict, Any, Union

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QRadioButton, QGroupBox, QLineEdit, QProgressBar,
    QMessageBox, QFileDialog, QButtonGroup, QCheckBox, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QFont, QPalette, QColor

from .text import *
from .gentab import *

class RomGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sand:box ROM Generator 6.0 by Andkuz & KlasterK")
        self.setMinimumSize(1000, 700)  # Increased minimum size
        
        # Apply modern style
        self.setStyleSheet(ROM_GENERATOR_STYLESHEET)
        
        # Create main tab widget with scrolling tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setUsesScrollButtons(True)
        self.setCentralWidget(self.tab_widget)
        
        # Create info tab first as requested
        info_tab = QWidget()
        info_layout = QVBoxLayout()
        info_tab.setLayout(info_layout)
        
        info_label = QLabel(ROM_GENERATOR_INFO_TEXT)
        info_label.setWordWrap(True)
        info_label.setFont(QFont("Arial", 11))
        info_layout.addWidget(info_label)
        
        # Add examples section
        examples_group = QGroupBox("Usage Examples")
        examples_layout = QVBoxLayout()
        examples_group.setLayout(examples_layout)
        
        examples = [
            "Bit reverse: bit_reverse(a, 8)",
            "Rotate left: bit_rotate_left(a, 8, 2)",
            "Complex operation: cmath.exp(a * 1j)",
            "Vector dot product: np.dot([a, b], [b, a])",
            "Custom math: math.sin(a) + math.cos(b)"
        ]
        
        for example in examples:
            example_label = QLabel(f"• {example}")
            example_label.setFont(QFont("Courier New", 10))
            examples_layout.addWidget(example_label)
        
        info_layout.addWidget(examples_group)
        info_layout.addStretch()
        
        self.tab_widget.addTab(info_tab, "INFO")
        
        # Create configuration tabs
        self.config_tabs = {}
        for in_bits in (4, 8, 16, 32, 64):
            for out_bits in (4, 8, 16, 32, 64):
                id = f"{in_bits}to{out_bits}"
                title = f'{in_bits}-bit → {out_bits}-bit'
                
                tab = GeneratorTab(in_bits, out_bits)
                self.config_tabs[id] = tab
                self.tab_widget.addTab(tab, title)
