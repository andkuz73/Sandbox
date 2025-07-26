import cmath
import importlib
import math
import random
import re
import struct
from typing import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import numpy as np

MAX_ROM_SIZE = 1 << 16  # 65536
SAFE_BUILTINS = {"__builtins__": {}}

class Generator:
    def __init__(self, input_bits: int, output_bits: int):
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.rom_data = bytearray()
        self.compiled_expr = None
        self.current_operation = "sum"
        self.num_type = "int"
        self.debug_mode = False
        self.output_part = "full"
        self.vector_size = 4
        self.custom_expr = ""

    def set_operation(self, operation: str):
        self.current_operation = operation

    def set_num_type(self, num_type: str):
        self.num_type = num_type

    def set_debug_mode(self, debug: bool):
        self.debug_mode = debug

    def set_output_part(self, part: str):
        self.output_part = part

    def set_vector_size(self, size: int):
        self.vector_size = size

    def set_custom_expr(self, expr: str):
        self.custom_expr = expr
        if expr:
            self.compiled_expr = compile(expr, "<string>", "eval")

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

    def process_input(self, addr: int) -> Tuple[Any, Any]:
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

    def vector_matrix_multiply(self, vector: int, matrix_size: int) -> Any:
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

    def convert_output(self, value: Any) -> int:
        """Convert result to output format"""
        if self.num_type == "int":
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
        elif self.num_type == "float":
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

    @staticmethod
    def float8_to_float32(x: int) -> float:
        """Convert custom 8-bit float to 32-bit float"""
        sign = -1 if (x & 0x80) else 1
        exponent = (x >> 4) & 0x07
        mantissa = x & 0x0F
        
        if exponent == 0:  # Denormal or zero
            if mantissa == 0:
                return 0.0
            return sign * (mantissa / 16.0) * (2.0 ** (-2))
        
        return sign * (1 + mantissa / 16.0) * (2.0 ** (exponent - 3))

    @staticmethod
    def float32_to_float8(f: float) -> int:
        """Convert 32-bit float to custom 8-bit float"""
        try:
            packed = struct.pack('!f', f)
            unpacked = struct.unpack('!I', packed)[0]
        except struct.error:
            return 0
            
        sign = (unpacked >> 31) & 0x1
        exponent = ((unpacked >> 23) & 0xFF) - 127
        mantissa = unpacked & 0x7FFFFF
        
        if exponent < -2:
            return 0x00 if sign == 0 else 0x80
        
        exponent_8 = max(0, min(7, exponent + 3))
        mantissa_8 = (mantissa >> 19) & 0x0F
        
        return (sign << 7) | ((exponent_8 & 0x07) << 4) | (mantissa_8 & 0x0F)

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

    def generate_rom(self) -> bytearray:
        """Generate ROM data based on current settings"""
        self.rom_data = bytearray()
        bit_size = self.input_bits if self.current_operation in [
            "bit_reverse", "rotate_left", "rotate_right", "shift_left", "shift_right"
        ] else 0

        for addr in range(MAX_ROM_SIZE):
            try:
                if self.current_operation == "vmmul":
                    result = self.vector_matrix_multiply(addr, self.vector_size)
                    a, b = None, None
                else:
                    a, b = self.process_input(addr)
                    
                    if self.current_operation == "sum":
                        result = a + b
                    elif self.current_operation == "sub":
                        result = a - b
                    elif self.current_operation == "mul":
                        result = a * b
                    elif self.current_operation == "div":
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
                    elif self.current_operation == "bit_reverse":
                        result = self.bit_reverse(a, bit_size)
                    elif self.current_operation == "rotate_left":
                        result = self.bit_rotate_left(a, bit_size, b % bit_size)
                    elif self.current_operation == "rotate_right":
                        result = self.bit_rotate_right(a, bit_size, b % bit_size)
                    elif self.current_operation == "shift_left":
                        result = self.bit_shift_left(a, bit_size, b % bit_size)
                    else:  # custom
                        modules = self.extract_used_modules(self.custom_expr)
                        context = {
                            'a': a, 
                            'b': b, 
                            **modules,
                            'bit_reverse': self.bit_reverse,
                            'bit_rotate_left': self.bit_rotate_left,
                            'bit_rotate_right': self.bit_rotate_right,
                            'bit_shift_left': self.bit_shift_left,
                            'bit_shift_right': self.bit_shift_right
                        }
                        safe_context = {**SAFE_BUILTINS, **context}
                        result = eval(self.compiled_expr, safe_context)
                
                result_conv = self.convert_output(result)
                
                if "full" in self.output_part or "single" in self.output_part:
                    bytes_count = self.output_bits // 8
                    if self.output_bits == 4:
                        self.rom_data.append(result_conv & 0xF)
                    elif bytes_count > 0:
                        bytes_data = [(result_conv >> (8*i)) & 0xFF for i in range(bytes_count)]
                        self.rom_data.extend(bytes(bytes_data))
                elif "low" in self.output_part:
                    self.rom_data.append(result_conv & 0xFF)
                elif "high" in self.output_part:
                    self.rom_data.append((result_conv >> 8) & 0xFF)
                elif "byte" in self.output_part:
                    byte_num = int(self.output_part.split("_")[1])
                    byte = (result_conv >> (8*byte_num)) & 0xFF
                    self.rom_data.append(byte)
                elif "nibble" in self.output_part:
                    self.rom_data.append(result_conv & 0xF)
                
                if self.debug_mode and addr % 1024 == 0:
                    if self.current_operation == "vmmul":
                        print(f"[DEBUG] addr={addr:05X}, result={result}, result_conv={result_conv}")
                    else:
                        print(f"[DEBUG] addr={addr:05X}, a={a}, b={b}, result={result}, result_conv={result_conv}")
            
            except (ZeroDivisionError, ValueError, TypeError, NameError) as e:
                print(f"Error evaluating expression: {e}")
                if "full" in self.output_part or "single" in self.output_part:
                    bytes_count = self.output_bits // 8
                    if self.output_bits == 4:
                        self.rom_data.append(0)
                    elif bytes_count > 0:
                        self.rom_data.extend(bytes([0]*bytes_count))
                else:
                    self.rom_data.append(0)
        
        return self.rom_data

    def get_rom_data(self) -> bytearray:
        """Get generated ROM data"""
        return self.rom_data

    def get_rom_hex(self) -> str:
        """Get ROM data as hex string"""
        return self.rom_data.hex()

class GeneratorTab(QWidget):
    def __init__(self, input_bits, output_bits):
        super().__init__()
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.generator = Generator(input_bits, output_bits)
        
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
            radio.setProperty("value", val)
            self.operation_group.addButton(radio, id=operations.index((text, val)))
            operation_layout.addWidget(radio, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        
        # Select first operation by default
        self.operation_group.button(0).setChecked(True)
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
        operation_value = self.operation_group.button(id).property("value")
        self.generator.set_operation(operation_value)
        
        if operation_value == "vmmul":
            self.vmmul_frame.show()
            self.custom_frame.hide()
        elif operation_value == "custom":
            self.vmmul_frame.hide()
            self.custom_frame.show()
        else:
            self.vmmul_frame.hide()
            self.custom_frame.hide()
    
    def generate(self):
        # Update generator settings from UI
        self.generator.set_num_type(self.num_type.currentText())
        self.generator.set_debug_mode(self.debug_checkbox.isChecked())
        
        part_button = self.part_group.checkedButton()
        if part_button:
            self.generator.set_output_part(part_button.text().lower().replace(" ", "_"))
        
        if self.operation_group.checkedButton().property("value") == "vmmul":
            self.generator.set_vector_size(int(self.vector_size.currentText()))
        
        if self.operation_group.checkedButton().property("value") == "custom":
            self.generator.set_custom_expr(self.text_input.text())
        
        # Generate ROM
        self.progress.setValue(0)
        
        # Use QThread to prevent UI freezing
        self.worker = GeneratorWorker(self.generator)
        self.worker.progress_updated.connect(self.progress.setValue)
        self.worker.finished.connect(self.on_generation_complete)
        self.worker.start()
    
    def on_generation_complete(self):
        self.progress.setValue(MAX_ROM_SIZE)
        QMessageBox.information(self, "Generation Complete", 
                               "ROM generation finished successfully!")
    
    def copy(self):
        if not self.generator.get_rom_data():
            QMessageBox.warning(self, "No Data", "Generate ROM first!")
            return
            
        clipboard = QApplication.clipboard()
        clipboard.setText(self.generator.get_rom_hex())
        QMessageBox.information(self, "Copied", "ROM data copied to clipboard!")
    
    def save_to_file(self):
        if not self.generator.get_rom_data():
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
                with open(file_path, "w", encoding='ASCII') as f:
                    f.write(self.generator.get_rom_hex())
            except IOError as e:
                QMessageBox.critical(self, "Save Error", f"Error saving file: {e}")
            else:
                QMessageBox.information(self, "Saved", f"ROM data saved to:\n{file_path}")

class GeneratorWorker(QThread):
    progress_updated = Signal(int)
    finished = Signal()
    
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
    
    def run(self):
        self.generator.generate_rom()
        for addr in range(0, MAX_ROM_SIZE + 1, 256):
            self.progress_updated.emit(addr)
        self.finished.emit()