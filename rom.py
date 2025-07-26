# Sand:box ROM Generator PRO 5.5 by: Andkuz (or Andkuz73 or cuzhima); tg: @cuzhima
import sys
import struct
import math
import numpy as np
import random
import re
import importlib
import cmath
import os
from typing import Dict, Any, Union, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QRadioButton, QGroupBox, QLineEdit, QProgressBar,
    QMessageBox, QFileDialog, QButtonGroup, QCheckBox, QSizePolicy, QTextEdit,
    QSplitter, QFileDialog, QStatusBar, QPlainTextEdit
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor, QTextCursor, QSyntaxHighlighter, QTextCharFormat, QBrush

# Constants
MAX_ROM_SIZE = 1 << 16  # 65536 (64 Kbytes)
SAFE_BUILTINS = {"__builtins__": {}}
HEX_VIEW_ROWS = 16  # 16 bytes for line in hex viewer
SHOW_BYTES = 4096  # First showed bytes of ROM in hex viewer

class HexHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.address_format = QTextCharFormat()
        self.address_format.setForeground(QBrush(QColor("#FFD700")))  # Gold
        
        self.data_format = QTextCharFormat()
        self.data_format.setForeground(QBrush(QColor("#87CEFA")))  # Light Sky Blue
        
        self.ascii_format = QTextCharFormat()
        self.ascii_format.setForeground(QBrush(QColor("#98FB98")))  # Pale Green

    def highlightBlock(self, text):
        if text.startswith("0x"):
            self.setFormat(0, 6, self.address_format)
            self.setFormat(7, 47, self.data_format)
            self.setFormat(55, 16, self.ascii_format)

class GeneratorThread(QThread):
    progress_updated = pyqtSignal(int)
    generation_complete = pyqtSignal(bytearray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, input_bits, output_bits, num_type, operation, expr, 
                 part, vector_size, debug_mode, max_address=MAX_ROM_SIZE, custom_script=None):
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.num_type = num_type
        self.operation = operation
        self.expr = expr
        self.part = part
        self.vector_size = vector_size
        self.debug_mode = debug_mode
        self.max_address = max_address  # Added for preview support
        self.custom_script = custom_script
        self.cancel_requested = False
        
    def run(self):
        try:
            rom_data = bytearray()
            
            # Compile custom expression if needed
            compiled_expr = None
            if self.operation == "custom":
                try:
                    compiled_expr = compile(self.expr, "<string>", "eval")
                except Exception as e:
                    self.error_occurred.emit(f"Error compiling expression: {e}")
                    return
            
            # For bit operations, we need to know the bit size
            bit_size = self.input_bits if self.operation in ["bit_reverse", "rotate_left", 
                                                          "rotate_right", "shift_left", "shift_right"] else 0
            
            # Use self.max_address instead of MAX_ROM_SIZE
            for addr in range(self.max_address):
                if self.cancel_requested:
                    return
                    
                try:
                    if self.operation == "vmmul":
                        vec_size = int(self.vector_size)
                        result = self.vector_matrix_multiply(addr, vec_size)
                        a, b = None, None
                    else:
                        a, b = self.process_input(addr)
                        
                        if self.operation == "sum":
                            result = a + b
                        elif self.operation == "sub":
                            result = a - b
                        elif self.operation == "mul":
                            result = a * b
                        elif self.operation == "div":
                            # Improved division by zero handling
                            if self.num_type == "int":
                                result = a // b if b != 0 else 0
                            elif self.num_type == "float":
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
                        elif self.operation == "bit_reverse":
                            result = self.bit_reverse(a, bit_size)
                        elif self.operation == "rotate_left":
                            result = self.bit_rotate_left(a, bit_size, b % bit_size)
                        elif self.operation == "rotate_right":
                            result = self.bit_rotate_right(a, bit_size, b % bit_size)
                        elif self.operation == "shift_left":
                            result = self.bit_shift_left(a, bit_size, b % bit_size)
                        else:  # custom
                            modules = self.extract_used_modules(self.expr)
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
                            result = eval(compiled_expr, safe_context)
                    
                    result_conv = self.convert_output(result, self.num_type)
                    
                    if "full" in self.part or "single" in self.part:
                        bytes_count = self.output_bits // 8
                        if self.output_bits == 4:  # Special handling for 4-bit
                            rom_data.append(result_conv & 0xF)
                        elif bytes_count > 0:
                            bytes_data = [(result_conv >> (8*i)) & 0xFF for i in range(bytes_count)]
                            rom_data.extend(bytes(bytes_data))
                    elif "low" in self.part:
                        rom_data.append(result_conv & 0xFF)
                    elif "high" in self.part:
                        rom_data.append((result_conv >> 8) & 0xFF)
                    elif "byte" in self.part:
                        byte_num = int(self.part.split("_")[1])
                        byte = (result_conv >> (8*byte_num)) & 0xFF
                        rom_data.append(byte)
                    elif "nibble" in self.part:  # For 4-bit output
                        rom_data.append(result_conv & 0xF)
                    
                    # Debug output
                    if self.debug_mode and addr % 1024 == 0:
                        if self.operation == "vmmul":
                            print(f"[DEBUG] addr={addr:05X}, result={result}, result_conv={result_conv}")
                        else:
                            print(f"[DEBUG] addr={addr:05X}, a={a}, b={b}, result={result}, result_conv={result_conv}")
                
                except (ZeroDivisionError, ValueError, TypeError, NameError) as e:
                    print(f"Error evaluating expression: {e}")
                    if "full" in self.part or "single" in self.part:
                        bytes_count = self.output_bits // 8
                        if self.output_bits == 4:
                            rom_data.append(0)
                        elif bytes_count > 0:
                            rom_data.extend(bytes([0]*bytes_count))
                    else:
                        rom_data.append(0)
                
                # Update progress bar every 256 addresses
                if addr % 256 == 0:
                    self.progress_updated.emit(addr)
            
            self.generation_complete.emit(rom_data)
        
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def cancel(self):
        self.cancel_requested = True
    
    # Bit manipulation functions
    @staticmethod
    def bit_reverse(value: int, bits: int) -> int:
        """Reverse the bits of an integer"""
        mask = (1 << bits) - 1
        return int(bin(value & mask)[2:].zfill(bits)[::-1], 2)
    
    @staticmethod
    def bit_rotate_left(value: int, bits: int, n: int) -> int:
        """Rotate left by n bits"""
        n = n % bits
        mask = (1 << bits) - 1
        return ((value << n) | (value >> (bits - n))) & mask
    
    @staticmethod
    def bit_rotate_right(value: int, bits: int, n: int) -> int:
        """Rotate right by n bits"""
        n = n % bits
        mask = (1 << bits) - 1
        return ((value >> n) | (value << (bits - n))) & mask
    
    @staticmethod
    def bit_shift_left(value: int, bits: int, n: int) -> int:
        """Logical shift left by n bits"""
        mask = (1 << bits) - 1
        return (value << n) & mask
    
    @staticmethod
    def bit_shift_right(value: int, bits: int, n: int) -> int:
        """Logical shift right by n bits"""
        mask = (1 << bits) - 1
        return (value >> n) & mask
    
    def process_input(self, addr):
        """Process input based on input bits and selected number type"""
        if self.input_bits == 32:
            bytes_in = [(addr >> (8*i)) & 0xFF for i in range(4)]
            if self.num_type == "int":
                return bytes_in[0] | (bytes_in[1] << 8) | (bytes_in[2] << 16) | (bytes_in[3] << 24), 0
            elif self.num_type == "float":
                a = struct.unpack('!f', bytes(bytes_in))[0]
                return a, 0
            else:  # complex
                real = self.float8_to_float32(addr & 0xFF)
                imag = self.float8_to_float32((addr >> 8) & 0xFF)
                return complex(real, imag), 0
            
        elif self.input_bits == 16:
            a = addr & 0xFF
            b = (addr >> 8) & 0xFF
            if self.num_type == "int":
                return a, b
            elif self.num_type == "float":
                return self.float8_to_float32(a), self.float8_to_float32(b)
            else:  # complex
                return complex(a, b), 0
        elif self.input_bits == 4:
            a = addr & 0x0F
            b = (addr >> 4) & 0x0F
            if self.num_type == "int":
                return a, b
            elif self.num_type == "float":
                return float(a), float(b)
            else:  # complex
                return complex(a, b), 0
        else:  # 8-bit
            if self.num_type == "int":
                return addr, addr
            elif self.num_type == "float":
                val = self.float8_to_float32(addr)
                return val, val
            else:  # complex
                val = complex(addr, 0)
                return val, val
    
    def vector_matrix_multiply(self, vector, matrix_size):
        """SIMD vector-matrix multiplication"""
        if self.num_type == "int":
            vector = np.array([(vector >> (8*i)) & 0xFF for i in range(matrix_size)], dtype=np.int32)
            matrix = np.arange(1, matrix_size*matrix_size+1).reshape(matrix_size, matrix_size)
            result = np.dot(vector, matrix)
            return int(result[0])
        elif self.num_type == "float":
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
                return value & 0xF
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
                return int(value.real) & 0xF
            else:  # 8-bit
                return int(value.real) & 0xFF
    
    def float8_to_float32(self, x):
        """Convert custom 8-bit float to 32-bit float"""
        sign = -1 if (x & 0x80) else 1
        exponent = (x >> 4) & 0x07
        mantissa = x & 0x0F
        
        if exponent == 0:
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

class RomGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sand:box ROM Generator PRO 5.2 by Andkuz")
        self.setMinimumSize(1200, 800)
        
        # Current ROM data
        self.rom_data = bytearray()
        self.generator_thread = None
        
        # Create main widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Settings
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        splitter.addWidget(settings_widget)
        
        # Right panel - Hex Viewer
        self.hex_viewer = QPlainTextEdit()
        self.hex_viewer.setReadOnly(True)
        self.hex_viewer.setFont(QFont("Courier New", 10))
        self.highlighter = HexHighlighter(self.hex_viewer.document())
        splitter.addWidget(self.hex_viewer)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 700])
        
        # Bits configuration
        bits_group = QGroupBox("Bits Configuration")
        bits_layout = QGridLayout()
        bits_group.setLayout(bits_layout)
        settings_layout.addWidget(bits_group)
        
        # Input bits
        bits_layout.addWidget(QLabel("Input Bits (each input number):"), 0, 0)
        self.input_bits = QComboBox()
        self.input_bits.addItems(["4", "8", "16", "32"])
        self.input_bits.setCurrentIndex(1)
        bits_layout.addWidget(self.input_bits, 0, 1)
        
        # Output bits
        bits_layout.addWidget(QLabel("Output Bits:"), 1, 0)
        self.output_bits = QComboBox()
        self.output_bits.addItems(["4", "8", "16", "32", "64"])
        self.output_bits.setCurrentIndex(1)
        bits_layout.addWidget(self.output_bits, 1, 1)
        
        # Operation Settings
        operation_group = QGroupBox("Operation Settings")
        operation_layout = QGridLayout()
        operation_layout.setSpacing(10)
        operation_group.setLayout(operation_layout)
        settings_layout.addWidget(operation_group)
        
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
            radio.setProperty("value", val)
            self.operation_group.addButton(radio)
            operation_layout.addWidget(radio, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        
        # Select first operation by default
        self.operation_group.buttons()[0].setChecked(True)
        self.current_operation = "sum"
        
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
        
        self.text_input = QPlainTextEdit()
        self.text_input.setPlaceholderText("Enter Python expression using a and b...")
        self.text_input.setFont(QFont("Consolas", 10))
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
        settings_layout.addWidget(output_group)
        
        # Number Type
        output_layout.addWidget(QLabel("Number type:"), 0, 0)
        self.num_type = QComboBox()
        self.num_type.addItems(["int", "float", "complex"])
        output_layout.addWidget(self.num_type, 0, 1)
        
        # Buttons
        button_layout = QHBoxLayout()
        settings_layout.addLayout(button_layout)
        
        # Formula file buttons
        file_button_layout = QHBoxLayout()
        settings_layout.addLayout(file_button_layout)
        
        self.load_button = QPushButton("Load Formula")
        self.load_button.clicked.connect(self.load_formula)
        file_button_layout.addWidget(self.load_button)
        
        self.save_button = QPushButton("Save Formula")
        self.save_button.clicked.connect(self.save_formula)
        file_button_layout.addWidget(self.save_button)
        
        # Action buttons
        self.preview_button = QPushButton("Preview ROM")
        self.preview_button.clicked.connect(self.preview_rom)
        button_layout.addWidget(self.preview_button)
        
        self.copy_button = QPushButton("Copy ROM")
        self.copy_button.clicked.connect(self.copy_rom)
        button_layout.addWidget(self.copy_button)
        
        self.save_rom_button = QPushButton("Save ROM")
        self.save_rom_button.clicked.connect(self.save_rom)
        button_layout.addWidget(self.save_rom_button)
        
        # Generate ROM button (large)
        self.generate_button = QPushButton("Generate ROM")
        self.generate_button.clicked.connect(self.generate_rom)
        self.generate_button.setMinimumHeight(60)  # Make button larger
        button_layout.addWidget(self.generate_button)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, MAX_ROM_SIZE)
        self.progress.setFormat("Ready")
        self.progress.setTextVisible(True)
        settings_layout.addWidget(self.progress)
        
        # Byte selection UI
        self.byte_group = QGroupBox("Byte Selection")
        byte_layout = QGridLayout()
        self.byte_group.setLayout(byte_layout)
        settings_layout.addWidget(self.byte_group)

        # Create byte selection radio buttons
        self.byte_var = "full"
        self.byte_buttons = []
        self.byte_labels = []
        
        # Full output radio
        full_radio = QRadioButton("Full Output")
        full_radio.setChecked(True)
        full_radio.toggled.connect(lambda: self.set_byte_var("full"))
        byte_layout.addWidget(full_radio, 0, 0)
        self.byte_buttons.append(full_radio)
        
        # Add byte-specific buttons
        for i in range(8):
            radio = QRadioButton(f"Byte {i}")
            radio.toggled.connect(lambda checked, idx=i: self.set_byte_var(f"byte{idx}") if checked else None)
            byte_layout.addWidget(radio, (i // 4) + 1, i % 4)
            self.byte_buttons.append(radio)
            
            # Add label for byte
            label = QLabel(f"Byte {i}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.byte_labels.append(label)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initialize UI state
        self.operation_changed()
        self.update_byte_buttons()
        
        # Connect signals
        self.operation_group.buttonClicked.connect(self.operation_changed)
        self.output_bits.currentIndexChanged.connect(self.update_byte_buttons)
    
    def operation_changed(self):
        op_button = self.operation_group.checkedButton()
        if op_button:
            self.current_operation = op_button.property("value")
        
        # Show/hide frames based on operation
        if self.current_operation == "vmmul":
            self.vmmul_frame.show()
            self.custom_frame.hide()
        elif self.current_operation == "custom":
            self.vmmul_frame.show()
            self.custom_frame.show()
        else:
            self.vmmul_frame.hide()
            self.custom_frame.hide()
            
    def set_byte_var(self, value):
        self.byte_var = value
        
    def update_byte_buttons(self):
        """Update byte selection buttons based on output bits"""
        output_bits = int(self.output_bits.currentText())
        bytes_count = (output_bits + 7) // 8  # Calculate number of bytes
        
        # Hide all byte buttons initially
        for i in range(1, len(self.byte_buttons)):
            self.byte_buttons[i].setVisible(False)
        
        # Show only needed byte buttons
        for i in range(min(bytes_count, 8)):
            self.byte_buttons[i+1].setVisible(True)
    
    def load_formula(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Formula Script",
            "",
            "Python Files (*.py);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, "r") as f:
                    formula = f.read()
                    self.text_input.setPlainText(formula)
                    self.status_bar.showMessage(f"Formula loaded from: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Error loading formula: {e}")
    
    def save_formula(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Formula Script",
            "",
            "Python Files (*.py);;All Files (*)"
        )
        
        if file_path:
            if not file_path.endswith(".py"):
                file_path += ".py"
                
            try:
                with open(file_path, "w") as f:
                    f.write(self.text_input.toPlainText())
                self.status_bar.showMessage(f"Formula saved to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving formula: {e}")
    
    def generate_rom(self):
        """Generate full ROM"""
        self._generate_rom(max_address=MAX_ROM_SIZE, preview=False)
    
    def generate_preview(self):
        """Generate ROM preview (first 256 addresses)"""
        self._generate_rom(max_address=256, preview=True)
    
    def _generate_rom(self, max_address, preview=False):
        """Internal method for ROM generation"""
        # Get current settings
        input_bits = int(self.input_bits.currentText())
        output_bits = int(self.output_bits.currentText())
        num_type = self.num_type.currentText()
        operation = self.current_operation
        expr = self.text_input.toPlainText()
        vector_size = self.vector_size.currentText()
        debug_mode = self.debug_checkbox.isChecked()
        part = self.byte_var  # Use selected byte option

        # Cancel any running generation
        if self.generator_thread and self.generator_thread.isRunning():
            self.generator_thread.cancel()
            self.generator_thread.wait()
        
        # Create and start generator thread
        self.generator_thread = GeneratorThread(
            input_bits, output_bits, num_type, operation, expr, 
            part, vector_size, debug_mode, max_address=max_address
        )
        
        # Connect signals
        self.generator_thread.progress_updated.connect(self.update_progress)
        
        if preview:
            self.generator_thread.generation_complete.connect(self.preview_generated)
        else:
            self.generator_thread.generation_complete.connect(self.rom_generated)
            
        self.generator_thread.error_occurred.connect(self.generation_error)
        
        # Update UI
        self.progress.setValue(0)
        if preview:
            self.progress.setFormat("Generating Preview: %p%")
            self.status_bar.showMessage("Generating preview...")
        else:
            self.progress.setFormat("Generating: %p%")
            self.status_bar.showMessage("Generating ROM...")
        
        # Start the thread
        self.generator_thread.start()
    
    def update_progress(self, value):
        self.progress.setValue(value)
    
    def rom_generated(self, rom_data):
        self.rom_data = rom_data
        self.progress.setValue(MAX_ROM_SIZE)
        self.progress.setFormat("Generation complete!")
        self.status_bar.showMessage(f"ROM generated successfully! Size: {len(rom_data)} bytes")
        self.display_hex_view()  # Update hex viewer
    
    def preview_generated(self, rom_data):
        self.rom_data = rom_data
        self.progress.setValue(256)
        self.progress.setFormat("Preview generated")
        self.status_bar.showMessage("Preview generated")
        self.display_hex_view()
    
    def generation_error(self, error_msg):
        self.progress.setFormat("Error")
        self.status_bar.showMessage(f"Error: {error_msg}")
        QMessageBox.critical(self, "Generation Error", error_msg)
    
    def display_hex_view(self):
        """Display ROM data in hex viewer with ASCII representation"""
        if not self.rom_data:
            self.hex_viewer.clear()
            return
            
        hex_text = ""
        bytes_per_line = 16
        
        # Only show the first SHOW_BYTES bytes
        for i in range(0, min(len(self.rom_data), SHOW_BYTES)):
            if i % bytes_per_line == 0:
                # Start of new line - show address
                hex_text += f"0x{i:04X}: "
            
            # Hex representation
            hex_text += f"{self.rom_data[i]:02X} "
            
            # ASCII representation at the end of the line
            if (i + 1) % bytes_per_line == 0 or i == min(len(self.rom_data), SHOW_BYTES) - 1:
                # Fill with spaces if not complete line
                if (i + 1) % bytes_per_line != 0:
                    spaces = (bytes_per_line - ((i + 1) % bytes_per_line)) * 3
                    hex_text += " " * spaces
                
                # ASCII representation
                hex_text += "  "
                start_idx = i - (i % bytes_per_line)
                end_idx = min(start_idx + bytes_per_line, len(self.rom_data), SHOW_BYTES)
                for j in range(start_idx, end_idx):
                    byte = self.rom_data[j]
                    # Show printable characters, others as dot
                    if 32 <= byte <= 126:
                        hex_text += chr(byte)
                    else:
                        hex_text += "."
                
                hex_text += "\n"
        
        self.hex_viewer.setPlainText(hex_text)
        self.hex_viewer.moveCursor(QTextCursor.MoveOperation.Start)
    
    def preview_rom(self):
        """Generate and display a preview of the ROM"""
        self.generate_preview()
    
    def copy_rom(self):
        """Generate and copy the full ROM to clipboard"""
        self.generate_rom()
        
        if self.rom_data:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.rom_data.hex())
            self.status_bar.showMessage("ROM data copied to clipboard!")
    
    def save_rom(self):
        """Generate and save the full ROM to a file"""
        self.generate_rom()
        
        if not self.rom_data:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROM File",
            "",
            "Sandbox ROM Files (*.sbrom);;Binary Files (*.bin);;HEX Files (*.hex);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, "wb") as f:
                    f.write(self.rom_data)
                self.status_bar.showMessage(f"ROM saved to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving ROM: {e}")
    
    def closeEvent(self, event):
        # Stop any running generation thread when closing
        if self.generator_thread and self.generator_thread.isRunning():
            self.generator_thread.cancel()
            self.generator_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    app.setStyle("Fusion")
    
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
