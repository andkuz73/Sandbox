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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QRadioButton, QGroupBox, QLineEdit, QProgressBar,
    QMessageBox, QFileDialog, QButtonGroup, QCheckBox, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor

# Constants
MAX_ROM_SIZE = 1 << 16  # 65536
SAFE_BUILTINS = {"__builtins__": {}}

class RomGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sand:box ROM Generator PRO 5.1 by Andkuz")
        self.setMinimumSize(1000, 700)  # Increased minimum size
        
        # Apply modern style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D30;
            }
            QTabWidget::pane {
                border: 0;
                background: #252526;
            }
            QTabBar::tab {
                background: #1E1E1E;
                color: #D4D4D4;
                padding: 8px 15px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 100px;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #007ACC;
                color: white;
            }
            QGroupBox {
                border: 1px solid #3F3F46;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
                color: #D4D4D4;
                background-color: #252526;
                font-size: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
            QLabel {
                color: #D4D4D4;
                font-size: 12px;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 15px;
                min-width: 100px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1C97EA;
            }
            QPushButton:pressed {
                background-color: #0062A3;
            }
            QComboBox, QLineEdit {
                background-color: #333337;
                color: #D4D4D4;
                border: 1px solid #3F3F46;
                border-radius: 3px;
                padding: 6px;
                font-size: 12px;
                min-height: 30px;
            }
            QProgressBar {
                border: 1px solid #3F3F46;
                border-radius: 3px;
                text-align: center;
                background: #252526;
                font-size: 12px;
                min-height: 25px;
            }
            QProgressBar::chunk {
                background-color: #007ACC;
            }
            QRadioButton {
                color: #D4D4D4;
                spacing: 5px;
                font-size: 12px;
            }
            QCheckBox {
                color: #D4D4D4;
                spacing: 5px;
                font-size: 12px;
            }
        """)
        
        # Create main tab widget with scrolling tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setUsesScrollButtons(True)
        self.setCentralWidget(self.tab_widget)
        
        # Create info tab first as requested
        info_tab = QWidget()
        info_layout = QVBoxLayout()
        info_tab.setLayout(info_layout)
        
        info_label = QLabel(self.get_info_text())
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
        configs = [
            ("4-bit → 4-bit", 4, 4),
            ("4-bit → 8-bit", 4, 8),
            ("4-bit → 16-bit", 4, 16),
            ("4-bit → 32-bit", 4, 32),
            ("4-bit → 64-bit", 4, 64),
            ("8-bit → 4-bit", 8, 4),
            ("8-bit → 8-bit", 8, 8),
            ("8-bit → 16-bit", 8, 16),
            ("8-bit → 32-bit", 8, 32),
            ("8-bit → 64-bit", 8, 64),
            ("16-bit → 4-bit", 16, 4),
            ("16-bit → 8-bit", 16, 8),
            ("16-bit → 16-bit", 16, 16),
            ("16-bit → 32-bit", 16, 32),
            ("16-bit → 64-bit", 16, 64),
            ("32-bit → 4-bit", 32, 4),
            ("32-bit → 8-bit", 32, 8),
            ("32-bit → 16-bit", 32, 16),
            ("32-bit → 32-bit", 32, 32),
            ("32-bit → 64-bit", 32, 64)
        ]
        
        for text, in_bits, out_bits in configs:
            tab = GeneratorTab(in_bits, out_bits)
            self.config_tabs[f"{in_bits}to{out_bits}"] = tab
            self.tab_widget.addTab(tab, text)
    
    def get_info_text(self):
        return """
        <h2>Sand:box ROM Generator PRO 5.1</h2>
        <p>This tool generates ROM files for the Sand:box game by evaluating mathematical expressions over all possible input combinations.</p>
        
        <h3>How to Use:</h3>
        <ol>
            <li>Select the appropriate input/output configuration tab</li>
            <li>Choose the operation type (predefined or custom)</li>
            <li>Select number type (int, float, complex)</li>
            <li>Configure output options</li>
            <li>Click "Generate" to create the ROM data</li>
            <li>Use "Copy ROM" or "Save ROM" to export the result</li>
        </ol>
        
        <h3>Features:</h3>
        <ul>
            <li><b>Predefined operations:</b> sum, subtract, multiply, divide, vector-matrix multiply</li>
            <li><b>Bit operations:</b> reverse, rotate, shift</li>
            <li><b>Custom expressions:</b> Use Python syntax with variables 'a' and 'b'</li>
            <li><b>Auto-imported modules:</b> math, cmath, random, numpy (as np)</li>
            <li><b>Number types:</b> int, float, complex</li>
            <li><b>Output options:</b> Full output or specific bytes</li>
            <li><b>New:</b> Support for 4-bit operands</li>
        </ul>
        
        <h3>Bit Manipulation Functions:</h3>
        <ul>
            <li><code>bit_reverse(value, bits)</code> - Reverse the bits of an integer</li>
            <li><code>bit_rotate_left(value, bits, n)</code> - Rotate left by n bits</li>
            <li><code>bit_rotate_right(value, bits, n)</code> - Rotate right by n bits</li>
            <li><code>bit_shift_left(value, bits, n)</code> - Logical shift left by n bits</li>
            <li><code>bit_shift_right(value, bits, n)</code> - Logical shift right by n bits</li>
        </ul>
        
        <p>For custom operations, you can import additional modules using standard Python import syntax.</p>
        """

class GeneratorTab(QWidget):
    def __init__(self, input_bits, output_bits):
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.rom_data = bytearray()
        self.compiled_expr = None
        
        # Main layout with better spacing
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        self.setLayout(main_layout)
        
        # Operation Settings
        operation_group = QGroupBox("Operation Settings")
        operation_layout = QGridLayout()
        operation_layout.setSpacing(10)
        operation_group.setLayout(operation_layout)
        main_layout.addWidget(operation_group)
        
        # Operation selection
        self.operation_group = QButtonGroup(self)
        operations = [
            ("Sum (a+b)", "sum"),
            ("Subtract (a-b)", "sub"),
            ("Multiply (a*b)", "mul"),
            ("Divide (a/b)", "div"),
            ("Bit Reverse", "bit_reverse"),
            ("Rotate Left", "rotate_left"),
            ("Shift Left", "shift_left"),
            ("Vector-Matrix Multiply", "vmmul"),
            ("Custom Expression", "custom")
        ]
        
        row, col = 0, 0
        for text, val in operations:
            radio = QRadioButton(text)
            radio.setProperty("value", val)  # Fixed: Set value property
            self.operation_group.addButton(radio, id=operations.index((text, val)))
            operation_layout.addWidget(radio, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        
        # Select first operation by default
        self.operation_group.button(0).setChecked(True)
        self.current_operation = "sum"
        self.operation_group.idClicked.connect(self.operation_changed)
        
        # Vector-Matrix settings
        self.vmmul_frame = QWidget()
        vmmul_layout = QHBoxLayout()
        self.vmmul_frame.setLayout(vmmul_layout)
        
        vmmul_layout.addWidget(QLabel("Vector size:"))
        self.vector_size = QComboBox()
        self.vector_size.addItems(["2", "4", "8"])
        self.vector_size.setCurrentIndex(1)
        vmmul_layout.addWidget(self.vector_size)
        vmmul_layout.addStretch()
        
        operation_layout.addWidget(self.vmmul_frame, row, 0, 1, 3)
        
        # Custom Expression
        self.custom_frame = QWidget()
        custom_layout = QVBoxLayout()
        self.custom_frame.setLayout(custom_layout)
        
        custom_layout.addWidget(QLabel("Python Expression (use a and b):"))
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("e.g., math.sin(a) + math.cos(b)")
        custom_layout.addWidget(self.text_input)
        
        operation_layout.addWidget(self.custom_frame, row+1, 0, 1, 3)
        
        # Debug option
        self.debug_checkbox = QCheckBox("Debug Mode (print values every 1024 addresses)")
        operation_layout.addWidget(self.debug_checkbox, row+2, 0, 1, 3)
        
        # Output Settings
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout()
        output_layout.setSpacing(10)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        # Number Type
        output_layout.addWidget(QLabel("Number type:"), 0, 0)
        self.num_type = QComboBox()
        self.num_type.addItems(["int", "float", "complex"])
        output_layout.addWidget(self.num_type, 0, 1)
        
        # Output Part
        self.part_group = QButtonGroup(self)
        output_layout.addWidget(QLabel("Output part:"), 1, 0)
        
        if output_bits == 64:
            self.full_radio = QRadioButton("Full output (8 bytes)")
            self.part_group.addButton(self.full_radio)
            output_layout.addWidget(self.full_radio, 1, 1)
            
            for i in range(8):
                radio = QRadioButton(f"Byte {i}")
                self.part_group.addButton(radio)
                output_layout.addWidget(radio, 2 + i//4, i%4)
            self.full_radio.setChecked(True)
            
        elif output_bits == 32:
            self.full_radio = QRadioButton("Full output (4 bytes)")
            self.part_group.addButton(self.full_radio)
            output_layout.addWidget(self.full_radio, 1, 1)
            
            for i in range(4):
                radio = QRadioButton(f"Byte {i}")
                self.part_group.addButton(radio)
                output_layout.addWidget(radio, 2 + i//2, i%2)
            self.full_radio.setChecked(True)
            
        elif output_bits == 16:
            self.full_radio = QRadioButton("Full output (2 bytes)")
            self.part_group.addButton(self.full_radio)
            output_layout.addWidget(self.full_radio, 1, 1)
            
            self.low_radio = QRadioButton("Low byte")
            self.part_group.addButton(self.low_radio)
            output_layout.addWidget(self.low_radio, 2, 0)
            
            self.high_radio = QRadioButton("High byte")
            self.part_group.addButton(self.high_radio)
            output_layout.addWidget(self.high_radio, 2, 1)
            
            self.full_radio.setChecked(True)
        elif output_bits == 4:
            self.full_radio = QRadioButton("Single nibble")
            self.part_group.addButton(self.full_radio)
            output_layout.addWidget(self.full_radio, 1, 1)
            self.full_radio.setChecked(True)
        else:  # 8-bit
            self.full_radio = QRadioButton("Single byte")
            self.part_group.addButton(self.full_radio)
            output_layout.addWidget(self.full_radio, 1, 1)
            self.full_radio.setChecked(True)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, MAX_ROM_SIZE)
        self.progress.setFormat("Generating: %p%")
        self.progress.setTextVisible(True)
        main_layout.addWidget(self.progress)
        
        # Buttons with better spacing
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        main_layout.addLayout(button_layout)
        
        self.generate_button = QPushButton("Generate ROM")
        self.generate_button.clicked.connect(self.generate)
        self.generate_button.setMinimumHeight(45)
        button_layout.addWidget(self.generate_button)
        
        self.copy_button = QPushButton("Copy ROM")
        self.copy_button.clicked.connect(self.copy)
        self.copy_button.setMinimumHeight(45)
        button_layout.addWidget(self.copy_button)
        
        self.save_button = QPushButton("Save ROM")
        self.save_button.clicked.connect(self.save_to_file)
        self.save_button.setMinimumHeight(45)
        button_layout.addWidget(self.save_button)
        
        # Initialize UI state
        self.operation_changed(0)
    
    def operation_changed(self, id):
        op = self.operation_group.button(id).text()
        # Fixed: Get value from property
        self.current_operation = self.operation_group.button(id).property("value")
        
        # Show/hide frames based on operation
        if self.current_operation == "vmmul":
            self.vmmul_frame.show()
            self.custom_frame.hide()
        elif self.current_operation == "custom":
            self.vmmul_frame.hide()
            self.custom_frame.show()
        else:
            self.vmmul_frame.hide()
            self.custom_frame.hide()
    
    def extract_used_modules(self, expr: str) -> Dict[str, Any]:
        """Extract modules used in the expression and import them"""
        module_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.\w+\b'
        modules_found = set(re.findall(module_pattern, expr))
        
        built_in_vars = {'a', 'b', 'math', 'random', 'np', 'cmath'}
        modules_to_import = {m for m in modules_found 
                            if m not in built_in_vars and m not in globals()}
        
        imported_modules = {}
        for module_name in modules_to_import:
            try:
                imported_modules[module_name] = importlib.import_module(module_name)
            except ImportError:
                print(f"Warning: Could not import module {module_name}")
                continue
        
        standard_modules = {
            'math': math,
            'random': random,
            'np': np,
            'cmath': cmath
        }
        
        return {**standard_modules, **imported_modules}
    
    # Bit manipulation functions
    @staticmethod
    def bit_reverse(value: int, bits: int) -> int:
        """Reverse the bits of an integer"""
        # Create a mask with the specified number of bits
        mask = (1 << bits) - 1
        # Reverse bits
        return int(bin(value & mask)[2:].zfill(bits)[::-1], 2)
    
    @staticmethod
    def bit_rotate_left(value: int, bits: int, n: int) -> int:
        """Rotate left by n bits"""
        # Normalize rotation count
        n = n % bits
        # Create bit mask
        mask = (1 << bits) - 1
        # Perform rotation
        return ((value << n) | (value >> (bits - n))) & mask
    
    @staticmethod
    def bit_rotate_right(value: int, bits: int, n: int) -> int:
        """Rotate right by n bits"""
        # Normalize rotation count
        n = n % bits
        # Create bit mask
        mask = (1 << bits) - 1
        # Perform rotation
        return ((value >> n) | (value << (bits - n))) & mask
    
    @staticmethod
    def bit_shift_left(value: int, bits: int, n: int) -> int:
        """Logical shift left by n bits"""
        # Create bit mask
        mask = (1 << bits) - 1
        # Perform shift
        return (value << n) & mask
    
    @staticmethod
    def bit_shift_right(value: int, bits: int, n: int) -> int:
        """Logical shift right by n bits"""
        # Create bit mask
        mask = (1 << bits) - 1
        # Perform shift
        return (value >> n) & mask
    
    def process_input(self, addr):
        """Process input based on input bits and selected number type"""
        num_type = self.num_type.currentText()
    
        if self.input_bits == 32:
            bytes_in = [(addr >> (8*i)) & 0xFF for i in range(4)]
            if num_type == "int":
                return bytes_in[0] | (bytes_in[1] << 8) | (bytes_in[2] << 16) | (bytes_in[3] << 24), 0
            elif num_type == "float":
                a = struct.unpack('!f', bytes(bytes_in))[0]
                return a, 0
            else:  # complex
                # Fixed complex decomposition
                real = self.float8_to_float32(addr & 0xFF)
                imag = self.float8_to_float32((addr >> 8) & 0xFF)
                return complex(real, imag), 0
            
        elif self.input_bits == 16:
            a = addr & 0xFF
            b = (addr >> 8) & 0xFF
            if num_type == "int":
                return a, b
            elif num_type == "float":
                return self.float8_to_float32(a), self.float8_to_float32(b)
            else:  # complex
                return complex(a, b), 0
        elif self.input_bits == 4:
            # For 4-bit input, split address into two 4-bit values
            a = addr & 0x0F
            b = (addr >> 4) & 0x0F
            if num_type == "int":
                return a, b
            elif num_type == "float":
                return float(a), float(b)
            else:  # complex
                return complex(a, b), 0
        else:  # 8-bit
            if num_type == "int":
                return addr, addr
            elif num_type == "float":
                val = self.float8_to_float32(addr)
                return val, val
            else:  # complex
                val = complex(addr, 0)
                return val, val
    
    def vector_matrix_multiply(self, vector, matrix_size):
        """SIMD vector-matrix multiplication"""
        num_type = self.num_type.currentText()
        
        if num_type == "int":
            vector = np.array([(vector >> (8*i)) & 0xFF for i in range(matrix_size)], dtype=np.int32)
            matrix = np.arange(1, matrix_size*matrix_size+1).reshape(matrix_size, matrix_size)
            result = np.dot(vector, matrix)
            return int(result[0])
        elif num_type == "float":
            vector = np.array([self.float8_to_float32((vector >> (8*i)) & 0xFF) for i in range(matrix_size)])
            matrix = np.arange(1, matrix_size*matrix_size+1).reshape(matrix_size, matrix_size).astype(np.float32)
            result = np.dot(vector, matrix)
            return float(result[0])
        else:  # complex
            vector = np.array([complex((vector >> (8*i)) & 0xFF, 0) for i in range(matrix_size)])
            matrix = np.arange(1, matrix_size*matrix_size+1).reshape(matrix_size, matrix_size).astype(np.complex64)
            result = np.dot(vector, matrix)
            return complex(result[0])
    
    def convert_output(self, value, num_type):
        """Convert result to output format"""
        if num_type == "int":
            if self.output_bits == 64:
                return value & 0xFFFFFFFFFFFFFFFF
            elif self.output_bits == 32:
                return value & 0xFFFFFFFF
            elif self.output_bits == 16:
                return value & 0xFFFF
            elif self.output_bits == 4:
                return value & 0xF  # Only 4 bits
            else:  # 8-bit
                return value & 0xFF
        elif num_type == "float":
            if self.output_bits == 64:
                return struct.unpack('!Q', struct.pack('!d', value))[0]
            elif self.output_bits == 32:
                return struct.unpack('!I', struct.pack('!f', value))[0]
            elif self.output_bits == 16:
                return struct.unpack('!H', struct.pack('!e', value))[0]
            elif self.output_bits == 4:
                # Simple scaling for 4-bit float
                return max(0, min(15, int(value)))
            else:  # 8-bit
                return self.float32_to_float8(value)
        else:  # complex
            if self.output_bits == 64:
                real = struct.pack('!f', value.real)
                imag = struct.pack('!f', value.imag)
                return (struct.unpack('!I', real)[0] << 32) | struct.unpack('!I', imag)[0]
            elif self.output_bits == 32:
                real = struct.pack('!e', value.real)
                imag = struct.pack('!e', value.imag)
                return (struct.unpack('!H', real)[0] << 16) | struct.unpack('!H', imag)[0]
            elif self.output_bits == 16:
                return ((int(value.real) & 0xFF) << 8) | (int(value.imag) & 0xFF)
            elif self.output_bits == 4:
                # Only take real part for 4-bit
                return int(value.real) & 0xF
            else:  # 8-bit
                return int(value.real) & 0xFF
    
    def float8_to_float32(self, x):
        """Convert custom 8-bit float to 32-bit float"""
        sign = -1 if (x & 0x80) else 1
        exponent = (x >> 4) & 0x07
        mantissa = x & 0x0F
        
        if exponent == 0:  # Denormal or zero
            if mantissa == 0:
                return 0.0
            return sign * (mantissa / 16.0) * (2.0 ** (-2))
        
        return sign * (1 + mantissa / 16.0) * (2.0 ** (exponent - 3))
    
    def float32_to_float8(self, f):
        """Convert 32-bit float to custom 8-bit float"""
        try:
            packed = struct.pack('!f', f)
            unpacked = struct.unpack('!I', packed)[0]
            
            sign = (unpacked >> 31) & 0x1
            exponent = ((unpacked >> 23) & 0xFF) - 127
            mantissa = unpacked & 0x7FFFFF
            
            if exponent < -2:
                return 0x00 if sign == 0 else 0x80
            
            exponent_8 = max(0, min(7, exponent + 3))
            mantissa_8 = (mantissa >> 19) & 0x0F
            
            return (sign << 7) | ((exponent_8 & 0x07) << 4) | (mantissa_8 & 0x0F)
        except:
            return 0
    
    def generate(self):
        num_type = self.num_type.currentText()
        operation = self.current_operation
        debug_mode = self.debug_checkbox.isChecked()
        
        # Get selected output part
        part_button = self.part_group.checkedButton()
        part = part_button.text().lower().replace(" ", "_") if part_button else "full"
        
        # Reset ROM data
        self.rom_data = bytearray()
        self.progress.setValue(0)
        
        # For bit operations, we need to know the bit size
        bit_size = self.input_bits if operation in ["bit_reverse", "rotate_left", 
                                                  "rotate_right", "shift_left", "shift_right"] else 0
        
        # Compile custom expression if needed
        if operation == "custom":
            expr = self.text_input.text()
            try:
                self.compiled_expr = compile(expr, "<string>", "eval")
            except Exception as e:
                QMessageBox.critical(self, "Compilation Error", f"Error compiling expression: {e}")
                return
        
        for addr in range(MAX_ROM_SIZE):
            try:
                if operation == "vmmul":
                    vec_size = int(self.vector_size.currentText())
                    result = self.vector_matrix_multiply(addr, vec_size)
                    a, b = None, None
                else:
                    a, b = self.process_input(addr)
                    
                    if operation == "sum":
                        result = a + b
                    elif operation == "sub":
                        result = a - b
                    elif operation == "mul":
                        result = a * b
                    elif operation == "div":
                        # Improved division by zero handling
                        if num_type == "int":
                            result = a // b if b != 0 else 0
                        elif num_type == "float":
                            if b == 0:
                                if a == 0:
                                    result = float('nan')
                                elif a > 0:
                                    result = float('inf')
                                else:
                                    result = float('-inf')
                            else:
                                result = a / b
                        else:  # complex
                            result = a / b if b != 0 else complex(0, 0)
                    elif operation == "bit_reverse":
                        result = self.bit_reverse(a, bit_size)
                    elif operation == "rotate_left":
                        result = self.bit_rotate_left(a, bit_size, b % bit_size)
                    elif operation == "rotate_right":
                        result = self.bit_rotate_right(a, bit_size, b % bit_size)
                    elif operation == "shift_left":
                        result = self.bit_shift_left(a, bit_size, b % bit_size)
                    else:  # custom
                        modules = self.extract_used_modules(self.text_input.text())
                        context = {
                            'a': a, 
                            'b': b, 
                            **modules,
                            # Add bit manipulation functions to context
                            'bit_reverse': self.bit_reverse,
                            'bit_rotate_left': self.bit_rotate_left,
                            'bit_rotate_right': self.bit_rotate_right,
                            'bit_shift_left': self.bit_shift_left,
                            'bit_shift_right': self.bit_shift_right
                        }
                        # Safe eval with restricted builtins
                        safe_context = {**SAFE_BUILTINS, **context}
                        result = eval(self.compiled_expr, safe_context)
                
                result_conv = self.convert_output(result, num_type)
                
                if "full" in part or "single" in part:
                    bytes_count = self.output_bits // 8
                    if self.output_bits == 4:  # Special handling for 4-bit
                        self.rom_data.append(result_conv & 0xF)
                    elif bytes_count > 0:
                        bytes_data = [(result_conv >> (8*i)) & 0xFF for i in range(bytes_count)]
                        self.rom_data.extend(bytes(bytes_data))
                elif "low" in part:
                    self.rom_data.append(result_conv & 0xFF)
                elif "high" in part:
                    self.rom_data.append((result_conv >> 8) & 0xFF)
                elif "byte" in part:
                    byte_num = int(part.split("_")[1])
                    byte = (result_conv >> (8*byte_num)) & 0xFF
                    self.rom_data.append(byte)
                elif "nibble" in part:  # For 4-bit output
                    self.rom_data.append(result_conv & 0xF)
                
                # Debug output
                if debug_mode and addr % 1024 == 0:
                    if operation == "vmmul":
                        print(f"[DEBUG] addr={addr:05X}, result={result}, result_conv={result_conv}")
                    else:
                        print(f"[DEBUG] addr={addr:05X}, a={a}, b={b}, result={result}, result_conv={result_conv}")
            
            except (ZeroDivisionError, ValueError, TypeError, NameError) as e:
                print(f"Error evaluating expression: {e}")
                if "full" in part or "single" in part:
                    bytes_count = self.output_bits // 8
                    if self.output_bits == 4:
                        self.rom_data.append(0)
                    elif bytes_count > 0:
                        self.rom_data.extend(bytes([0]*bytes_count))
                else:
                    self.rom_data.append(0)
            
            # Update progress bar every 256 addresses
            if addr % 256 == 0:
                self.progress.setValue(addr)
                QApplication.processEvents()
        
        self.progress.setValue(MAX_ROM_SIZE)
        QMessageBox.information(self, "Generation Complete", 
                               "ROM generation finished successfully!")
    
    def copy(self):
        if not self.rom_data:
            QMessageBox.warning(self, "No Data", "Generate ROM first!")
            return
            
        clipboard = QApplication.clipboard()
        clipboard.setText(self.rom_data.hex())
        QMessageBox.information(self, "Copied", "ROM data copied to clipboard!")
    
    def save_to_file(self):
        if not self.rom_data:
            QMessageBox.warning(self, "No Data", "Generate ROM first!")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROM File",
            "",
            "HEX Files (*.hex);;All Files (*)"
        )
        
        if file_path:
            if not file_path.endswith(".hex"):
                file_path += ".hex"
                
            try:
                with open(file_path, "w") as f:
                    f.write(self.rom_data.hex())
                QMessageBox.information(self, "Saved", f"ROM data saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving file: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))  # Set larger default font
    app.setStyle("Fusion")  # Modern style
    
    # Create a dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Highlight, QColor(142, 45, 197).lighter())
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)
    
    window = RomGeneratorApp()
    window.show()
    sys.exit(app.exec())
