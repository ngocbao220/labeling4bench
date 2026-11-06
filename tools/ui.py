"""Thanh tiến trình và bảng dùng chung cho các tool (rich). Ghi ra stderr để stdout vẫn dùng được cho dữ liệu."""
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

__all__ = ["Console", "console", "progress", "table"]

console = Console(stderr=True)


def progress():
    return Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(), TextColumn("{task.completed:g}/{task.total:g}"),
                    TextColumn("{task.percentage:>3.0f}%"), TimeElapsedColumn(), TextColumn("còn"), TimeRemainingColumn(),
                    console=console, transient=False)


def table(title, cols, rows, out=None):
    """cols: tên cột; cột nào bắt đầu bằng '>' thì căn phải (số)."""
    t = Table(title=title, title_justify="left", header_style="bold cyan")
    for c in cols:
        t.add_column(c.lstrip(">"), justify="right" if c.startswith(">") else "left")
    for r in rows:
        t.add_row(*["" if x is None else str(x) for x in r])
    (out or console).print(t)
