ROM_GENERATOR_STYLESHEET = """
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
        """

ROM_GENERATOR_INFO_TEXT = """
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