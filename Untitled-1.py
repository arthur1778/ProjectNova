**Full runnable script**

```python
#!/usr/bin/env python3
import tkinter as tk
import ast
import operator

# ---------- Safe evaluator ----------
def safe_eval(expr: str) -> float:
    """
    Evaluate a simple arithmetic expression containing
    +, -, *, /, parentheses and decimal numbers.
    Raises ValueError on any other node.
    """
    # mapping of AST operators to actual functions
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.UAdd: lambda x: +x,
        ast.USub: lambda x: -x,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type not in ops:
                raise ValueError(f"Unsupported operator {op_type}")
            return ops[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in ops:
                raise ValueError(f"Unsupported unary operator {op_type}")
            return ops[op_type](operand)
        elif isinstance(node, ast.Num):          # Python <3.8
            return node.n
        elif isinstance(node, ast.Constant):     # Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant {node.value}")
        else:
            raise ValueError(f"Unsupported expression {node}")

    try:
        parsed = ast.parse(expr, mode='eval')
        return _eval(parsed)
    except Exception as e:
        raise ValueError(f"Invalid expression: {expr}") from e


# ---------- Calculator ----------
class Calculator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calculator")
        self.geometry("340x460")
        self.resizable(False, False)

        # colours
        self.bg_color   = "#2b2b2b"
        self.btn_color  = "#4d4d4d"
        self.op_color   = "#ff9500"
        self.txt_color  = "#ffffff"

        self.configure(bg=self.bg_color)

        self.expression = ""

        # display
        self.display = tk.Entry(
            self,
            font=("Helvetica", 28),
            bg=self.bg_color,
            fg=self.txt_color,
            borderwidth=0,
            justify="right"
        )
        self.display.pack(fill="both", padx=12, pady=12, ipady=10)

        # button layout (text, row, column, columnspan)
        buttons = [
            ("C", 0, 0, 1), ("±", 0, 1, 1), ("⌫", 0, 2, 1), ("/", 0, 3, 1),
            ("7", 1, 0, 1), ("8", 1, 1, 1), ("9", 1, 2, 1), ("*", 1, 3, 1),
            ("4", 2, 0, 1), ("5", 2, 1, 1), ("6", 2, 2, 1), ("-", 2, 3, 1),
            ("1", 3, 0, 1), ("2", 3, 1, 1), ("3", 3, 2, 1), ("+", 3, 3, 1),
            ("0", 4, 0, 2), (".", 4, 2, 1), ("=", 4, 3, 1