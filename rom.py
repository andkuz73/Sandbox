# Sand:box ROM Generator PRO 4.0 by: Andkuz (or Andkuz73 or cuzhima)
import tkinter as tk
from tkinter import ttk, messagebox
import struct
import math
import numpy as np
import random
import re
import importlib
import cmath
from typing import Dict, Any, Union

class RomGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sand:box ROM Generator PRO 4.0 by Andkuz")
        
        # Main menu using grid layout
        main_frame = tk.Frame(self)
        main_frame.pack(padx=20, pady=20)
        
        tk.Label(main_frame, text="Select input/output configuration:", 
                font=("Arial", 12)).grid(row=0, column=0, columnspan=3, pady=10)
        
        # Buttons for configuration selection
        configs = [
            ("8-bit → 8-bit", self.open_8to8),
            ("8-bit → 16-bit", self.open_8to16),
            ("8-bit → 32-bit", self.open_8to32),
            ("8-bit → 64-bit", self.open_8to64),
            ("16-bit → 8-bit", self.open_16to8),
            ("16-bit → 16-bit", self.open_16to16),
            ("16-bit → 32-bit", self.open_16to32),
            ("16-bit → 64-bit", self.open_16to64),
            ("32-bit → 8-bit", self.open_32to8),
            ("32-bit → 16-bit", self.open_32to16),
            ("32-bit → 32-bit", self.open_32to32),
            ("32-bit → 64-bit", self.open_32to64)
        ]
        
        # Place buttons in grid
        for i, (text, command) in enumerate(configs):
            btn = tk.Button(main_frame, text=text, width=15, command=command)
            btn.grid(row=1 + i//3, column=i%3, padx=5, pady=5)
        
        # Add Info button at the bottom
        info_button = tk.Button(main_frame, text="INFO", width=15, command=self.show_info)
        info_button.grid(row=5, column=1, pady=20)
    
    def show_info(self):
        """Show information about how to use the program"""
        info_text = """To generate a ROM for the Sand:box game, select the desired size for each operand and the result size. 
For example, 8 bits for each operand and 8 bits for the result (basic choice for Sand:box) is the 8-bit → 8-bit mode.

After selecting the mode, you can choose the operation (several basic ones or your own). 
For custom operations, use Python syntax. You can import modules/libraries. 
The following are automatically imported and don't need to be specified: math, cmath, random, numpy (as np).

Supported number types: int, float, complex.

'Generate' will create the ROM, and 'Copy ROM' will copy it to the clipboard."""
        
        messagebox.showinfo("How to use", info_text)

class BaseGenerator(tk.Toplevel):
    def __init__(self, master, input_bits, output_bits):
        super().__init__(master)
        self.input_bits = input_bits
        self.output_bits = output_bits
        self.title(f"Generator {input_bits}-bit → {output_bits}-bit")
        
        # Operation Settings
        operation_frame = tk.LabelFrame(self, text="Operation Settings", padx=5, pady=5)
        operation_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.operation_var = tk.StringVar(value="sum")
        operations = [
            ("Sum (a+b)", "sum"),
            ("Subtract (a-b)", "sub"),
            ("Multiply (a*b)", "mul"),
            ("Divide (a/b)", "div"),
            ("Vector-Matrix Multiply", "vmmul"),
            ("Custom", "custom")
        ]
        
        for i, (text, val) in enumerate(operations):
            tk.Radiobutton(operation_frame, text=text, variable=self.operation_var, 
                          value=val).grid(row=i//3, column=i%3, sticky=tk.W)
        self.operation_var.trace_add('write', self.toggle_custom_input)
        
        # Vector-Matrix settings
        self.vmmul_frame = tk.Frame(operation_frame)
        tk.Label(self.vmmul_frame, text="Vector size:").grid(row=0, column=0)
        self.vector_size = ttk.Combobox(self.vmmul_frame, values=[2, 4, 8], state="readonly")
        self.vector_size.current(1)
        self.vector_size.grid(row=0, column=1)
        
        # Custom Expression
        self.custom_frame = tk.Frame(operation_frame)
        self.expr_label = tk.Label(self.custom_frame, text="Expression (use a and b, supports complex numbers with cmath):")
        self.expr_label.pack(pady=2)
        self.text_input = tk.Entry(self.custom_frame, font=("Arial", 10))
        self.text_input.pack(fill=tk.X, pady=2)
        
        # Output Settings
        output_frame = tk.LabelFrame(self, text="Output Settings", padx=5, pady=5)
        output_frame.pack(padx=10, pady=5, fill=tk.X)
        
        # Number Type - now includes complex
        tk.Label(output_frame, text="Number type:").grid(row=0, column=0, sticky=tk.W)
        self.num_type = ttk.Combobox(output_frame, values=["int", "float", "complex"], state="readonly")
        self.num_type.current(0)
        self.num_type.grid(row=0, column=1, sticky=tk.W)
        
        # Output Part
        self.part_var = tk.StringVar(value="full")
        if output_bits == 64:
            tk.Radiobutton(output_frame, text="Full output (8 bytes)", variable=self.part_var, 
                          value="full").grid(row=1, column=0, columnspan=4, sticky=tk.W)
            for i in range(8):
                tk.Radiobutton(output_frame, text=f"Byte {i}", variable=self.part_var, 
                              value=f"byte{i}").grid(row=2+i//4, column=i%4, sticky=tk.W)
        elif output_bits == 32:
            tk.Radiobutton(output_frame, text="Full output (4 bytes)", variable=self.part_var, 
                          value="full").grid(row=1, column=0, columnspan=2, sticky=tk.W)
            for i in range(4):
                tk.Radiobutton(output_frame, text=f"Byte {i}", variable=self.part_var, 
                              value=f"byte{i}").grid(row=2+i//2, column=i%2, sticky=tk.W)
        elif output_bits == 16:
            tk.Radiobutton(output_frame, text="Full output (2 bytes)", variable=self.part_var, 
                          value="full").grid(row=1, column=0, sticky=tk.W)
            tk.Radiobutton(output_frame, text="Low byte", variable=self.part_var, 
                          value="low").grid(row=1, column=1, sticky=tk.W)
            tk.Radiobutton(output_frame, text="High byte", variable=self.part_var, 
                          value="high").grid(row=2, column=0, sticky=tk.W)
        else:  # 8-bit
            tk.Radiobutton(output_frame, text="Single byte", variable=self.part_var, 
                          value="full").grid(row=1, column=0, sticky=tk.W)
        
        # Buttons
        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)
        
        generate_button = tk.Button(button_frame, text="Generate", command=self.generate)
        generate_button.pack(side=tk.LEFT, padx=5)
        
        copy_button = tk.Button(button_frame, text="Copy ROM", command=self.copy)
        copy_button.pack(side=tk.LEFT, padx=5)
        
        self.output = []
        self.toggle_custom_input()
    
    def toggle_custom_input(self, *args):
        op = self.operation_var.get()
        if op == "custom":
            self.custom_frame.grid(row=2, column=0, columnspan=3, sticky="we")
            self.vmmul_frame.grid_forget()
        elif op == "vmmul":
            self.vmmul_frame.grid(row=2, column=0, columnspan=3, sticky="we")
            self.custom_frame.grid_forget()
        else:
            self.custom_frame.grid_forget()
            self.vmmul_frame.grid_forget()
    
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
    
    def process_input(self, addr):
        """Process input based on input bits and selected number type"""
        num_type = self.num_type.get()
        
        if self.input_bits == 32:
            bytes_in = [(addr >> (8*i)) & 0xFF for i in range(4)]
            if num_type == "int":
                return bytes_in[0] | (bytes_in[1] << 8) | (bytes_in[2] << 16) | (bytes_in[3] << 24), 0
            elif num_type == "float":
                a = struct.unpack('!f', bytes(bytes_in))[0]
                return a, 0
            else:  # complex
                real = struct.unpack('!f', bytes(bytes_in[:2] + [0, 0]))[0]
                imag = struct.unpack('!f', bytes(bytes_in[2:] + [0, 0]))[0]
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
        else:  # 8-bit
            if num_type == "int":
                return addr, 0
            elif num_type == "float":
                return self.float8_to_float32(addr), 0
            else:  # complex
                return complex(addr, 0), 0
    
    def vector_matrix_multiply(self, vector, matrix_size):
        """SIMD vector-matrix multiplication"""
        num_type = self.num_type.get()
        
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
            else:  # 8-bit
                return value & 0xFF
        elif num_type == "float":
            if self.output_bits == 64:
                return struct.unpack('!Q', struct.pack('!d', value))[0]
            elif self.output_bits == 32:
                return struct.unpack('!I', struct.pack('!f', value))[0]
            elif self.output_bits == 16:
                return struct.unpack('!H', struct.pack('!e', value))[0]
            else:  # 8-bit
                return self.float32_to_float8(value)
        else:  # complex
            if self.output_bits == 64:
                # Pack real and imaginary parts into 32 bits each
                real = struct.pack('!f', value.real)
                imag = struct.pack('!f', value.imag)
                return (struct.unpack('!I', real)[0] << 32) | struct.unpack('!I', imag)[0]
            elif self.output_bits == 32:
                # Use all 32 bits for complex (16 bits real, 16 bits imag)
                real = struct.pack('!e', value.real)
                imag = struct.pack('!e', value.imag)
                return (struct.unpack('!H', real)[0] << 16) | struct.unpack('!H', imag)[0]
            elif self.output_bits == 16:
                # Use all 16 bits (8 bits real, 8 bits imag)
                return ((int(value.real) & 0xFF) << 8) | (int(value.imag) & 0xFF)
            else:  # 8-bit
                # Just use real part for 8-bit
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
        num_type = self.num_type.get()
        operation = self.operation_var.get()
        part = self.part_var.get()
        
        self.output = []
        max_addr = 1 << 16  # Sand:box ROM has 16-bit address
        
        for addr in range(max_addr):
            try:
                if operation == "vmmul":
                    vec_size = int(self.vector_size.get())
                    result = self.vector_matrix_multiply(addr, vec_size)
                else:
                    a, b = self.process_input(addr)
                    
                    if operation == "sum":
                        result = a + b
                    elif operation == "sub":
                        result = a - b
                    elif operation == "mul":
                        result = a * b
                    elif operation == "div":
                        if num_type == "complex":
                            result = a / b if b != 0 else complex(0, 0)
                        else:
                            result = a / b if b != 0 else 0
                    else:  # custom
                        expr = self.text_input.get()
                        modules = self.extract_used_modules(expr)
                        context = {'a': a, 'b': b, **modules}
                        result = eval(expr, context)
                
                result_conv = self.convert_output(result, num_type)
                
                if part == "full":
                    bytes_count = self.output_bits // 8
                    bytes_data = [(result_conv >> (8*i)) & 0xFF for i in range(bytes_count)]
                    self.output.extend(f"{b:02x}" for b in bytes_data)
                else:
                    if part.startswith("byte"):
                        byte_num = int(part[4:])
                        byte = (result_conv >> (8*byte_num)) & 0xFF
                    elif part == "low":
                        byte = result_conv & 0xFF
                    elif part == "high":
                        byte = (result_conv >> 8) & 0xFF
                    self.output.append(f"{byte:02x}")
            
            except (ZeroDivisionError, ValueError, TypeError, NameError) as e:
                print(f"Error evaluating expression: {e}")
                if part == "full":
                    bytes_count = self.output_bits // 8
                    self.output.extend(["00"]*bytes_count)
                else:
                    self.output.append("00")
    
    def copy(self):
        self.clipboard_clear()
        self.clipboard_append("".join(self.output))
        self.update()

# Create classes for all combinations
class Generator8to8(BaseGenerator):
    def __init__(self, master): super().__init__(master, 8, 8)
class Generator8to16(BaseGenerator):
    def __init__(self, master): super().__init__(master, 8, 16)
class Generator8to32(BaseGenerator):
    def __init__(self, master): super().__init__(master, 8, 32)
class Generator8to64(BaseGenerator):
    def __init__(self, master): super().__init__(master, 8, 64)
class Generator16to8(BaseGenerator):
    def __init__(self, master): super().__init__(master, 16, 8)
class Generator16to16(BaseGenerator):
    def __init__(self, master): super().__init__(master, 16, 16)
class Generator16to32(BaseGenerator):
    def __init__(self, master): super().__init__(master, 16, 32)
class Generator16to64(BaseGenerator):
    def __init__(self, master): super().__init__(master, 16, 64)
class Generator32to8(BaseGenerator):
    def __init__(self, master): super().__init__(master, 32, 8)
class Generator32to16(BaseGenerator):
    def __init__(self, master): super().__init__(master, 32, 16)
class Generator32to32(BaseGenerator):
    def __init__(self, master): super().__init__(master, 32, 32)
class Generator32to64(BaseGenerator):
    def __init__(self, master): super().__init__(master, 32, 64)

# Add methods to main class
RomGeneratorApp.open_8to8 = lambda self: Generator8to8(self)
RomGeneratorApp.open_8to16 = lambda self: Generator8to16(self)
RomGeneratorApp.open_8to32 = lambda self: Generator8to32(self)
RomGeneratorApp.open_8to64 = lambda self: Generator8to64(self)
RomGeneratorApp.open_16to8 = lambda self: Generator16to8(self)
RomGeneratorApp.open_16to16 = lambda self: Generator16to16(self)
RomGeneratorApp.open_16to32 = lambda self: Generator16to32(self)
RomGeneratorApp.open_16to64 = lambda self: Generator16to64(self)
RomGeneratorApp.open_32to8 = lambda self: Generator32to8(self)
RomGeneratorApp.open_32to16 = lambda self: Generator32to16(self)
RomGeneratorApp.open_32to32 = lambda self: Generator32to32(self)
RomGeneratorApp.open_32to64 = lambda self: Generator32to64(self)

if __name__ == "__main__":
    app = RomGeneratorApp()
    app.mainloop()