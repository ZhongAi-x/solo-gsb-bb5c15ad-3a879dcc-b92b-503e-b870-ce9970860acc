"""水平扫描线填充区间（只读即时分析）。

对一份 SVG path 即时解析并完成轮廓包含层级分类后，在有效闭合轮廓
（``outer`` / ``hole``）的**整体包围盒**内，从最小 y 起每隔
``spacing`` 取一条水平扫描线（最大 y 不含），求该线与填充区域的
交集区间，供逐行加工（如激光/刀具路径规划）使用。

- 开放轮廓（未以 ``Z`` 闭合）与退化轮廓（面积为 0 或顶点不足）跳过；
- 交点采用**半开边规则**：``min(y1,y2) <= y < max(y1,y2)`` 的边才
  与扫描线求交（含下端点、不含上端点），水平边跳过——扫描线经过
  顶点时该顶点恰好被两条邻边中的一条计数，局部极值点计 0 或 2 次，
  不会重复计数；
- 所有有效轮廓的交点统一按 x 排序后两两配对，开关状态每次穿越切换
  一次：点在有效轮廓中的数量为奇数时即填充，等价于按包含层级
  **交替合并**外环（深度 0、2 …）、孔与岛中孔（深度 1、3 …）；
- 每行输出按 x 升序、互不重叠的 ``[x_start, x_end]`` 区间，
  轮廓相切/尖点产生的零宽区间丢弃。

交点基于曲线细分后的多边形（与 normalize / metrics / clearance
同一份细分折线，见 :func:`app.svggeo.flatten_subpath`）。单次最多
生成 :data:`MAX_SCANLINES` 条扫描线，超出抛出
:class:`ScanlineLimitError`（API 层转为 422）。整个计算只读、
不写库。
"""

from __future__ import annotations

import math
from typing import Any

from .svggeo import Contour

# 单次请求允许的最大扫描线条数
MAX_SCANLINES = 1000
# 输出量化位数（仅影响 JSON 展示，不改变几何判定）
_DECIMALS = 9


class ScanlineLimitError(ValueError):
    """扫描线条数超过单次上限。"""


def compute_scanlines(contours: list[Contour],
                      spacing: float,
                      max_lines: int = MAX_SCANLINES) -> dict[str, Any]:
    """对已分类的轮廓列表生成水平扫描线填充区间。

    参数:
        contours: :func:`app.svggeo.analyze_contours` 返回的轮廓
            （已带 ``role`` / ``depth``）
        spacing: 扫描线纵向间距（必须 > 0，且为有限数）
        max_lines: 单次最多生成的扫描线条数

    返回:
        ``{spacing, line_count, lines, skipped_contours}``；
        ``lines`` 每项为 ``{"y": ..., "intervals": [[x_start, x_end], ...]}``，
        按 y 升序，区间按 x 升序且互不重叠。无有效闭合轮廓（空路径、
        仅开放/退化轮廓）时 ``lines`` 为空列表。

    异常:
        ValueError: ``spacing`` 非法
        ScanlineLimitError: 按当前间距生成的扫描线超过 ``max_lines``
    """
    if not (isinstance(spacing, (int, float))
            and math.isfinite(spacing) and spacing > 0):
        raise ValueError("spacing 必须是大于 0 的有限数")

    skipped = [
        {"original_index": c.original_index, "role": c.role,
         "reason": ("open" if c.role == "open" else "degenerate")}
        for c in sorted(contours, key=lambda c: c.original_index)
        if c.role not in ("outer", "hole")
    ]

    valid = [c for c in contours if c.role in ("outer", "hole")]
    if not valid:
        return {"spacing": spacing, "line_count": 0, "lines": [],
                "skipped_contours": skipped}

    # 有效闭合轮廓的整体包围盒
    min_y = min(c.bounds[1] for c in valid)
    max_y = max(c.bounds[3] for c in valid)

    lines: list[dict[str, Any]] = []
    k = 0
    while True:
        y = min_y + k * spacing
        if not y < max_y:  # 最大 y（不含）；NaN 也在此终止
            break
        if len(lines) >= max_lines:
            raise ScanlineLimitError(
                f"扫描线条数超过单次上限 {max_lines}：请增大 spacing")
        lines.append({"y": _num(y), "intervals": _fill_intervals(valid, y)})
        k += 1

    return {"spacing": spacing, "line_count": len(lines), "lines": lines,
            "skipped_contours": skipped}


def _fill_intervals(contours: list[Contour],
                    y: float) -> list[list[float]]:
    """扫描线 y 与所有闭合轮廓的填充区间（升序、互不重叠）。"""
    xs: list[float] = []
    for c in contours:
        pts = c.points
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if y1 == y2:
                continue  # 水平边不产生交点
            # 半开边规则：含下端点、不含上端点 -> 顶点不重复计数
            if (y1 <= y < y2) or (y2 <= y < y1):
                t = (y - y1) / (y2 - y1)
                xs.append(x1 + t * (x2 - x1))

    xs.sort()
    intervals: list[list[float]] = []
    # 每条有效轮廓交点成对，全体按 x 排序后两两配对即为交替填充段
    for i in range(0, len(xs) - 1, 2):
        x_start, x_end = xs[i], xs[i + 1]
        if x_end > x_start:  # 丢弃相切/尖点产生的零宽区间
            intervals.append([_num(x_start), _num(x_end)])
    return intervals


def _num(v: float) -> float:
    """量化展示值并消除 -0.0。"""
    r = round(v, _DECIMALS)
    return r + 0.0 if r == 0 else r
