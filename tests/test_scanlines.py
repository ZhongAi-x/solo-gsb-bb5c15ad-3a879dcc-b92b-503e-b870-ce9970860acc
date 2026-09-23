"""scanlines：水平扫描线填充区间单元测试。"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.scanlines import (  # noqa: E402
    MAX_SCANLINES,
    ScanlineLimitError,
    compute_scanlines,
)
from app.svggeo import analyze_contours  # noqa: E402


def _scan(d, spacing, fill_rule="nonzero", **kw):
    contours, _ = analyze_contours(d, fill_rule)
    return compute_scanlines(contours, spacing, **kw)


def _ys(res):
    return [line["y"] for line in res["lines"]]


def _intervals_at(res, y):
    for line in res["lines"]:
        if math.isclose(line["y"], y, abs_tol=1e-12):
            return line["intervals"]
    raise AssertionError(f"扫描线 y={y} 不存在")


# ---------------------------------------------------------------------------
# 取线规则：整体包围盒最小 y 起、等间距、最大 y 不含
# ---------------------------------------------------------------------------

def test_lines_start_at_bbox_min_y_and_exclude_max_y():
    # 方形 y 0..10，间距 2 -> y = 0,2,4,6,8（10 不含），共 5 条
    res = _scan("M0,0h10v10h-10Z", 2)
    assert res["line_count"] == 5
    assert _ys(res) == [0.0, 2.0, 4.0, 6.0, 8.0]
    # 每行都是整个外方
    for line in res["lines"]:
        assert line["intervals"] == [[0.0, 10.0]]


def test_lines_use_translated_overall_bbox():
    # 方形 y 5..15：最小 y 从 5 起算
    res = _scan("M0,5h10v10h-10Z", 2)
    assert _ys(res) == [5.0, 7.0, 9.0, 11.0, 13.0]


def test_non_aligned_spacing_count():
    # 高度 10、间距 3 -> y = 0,3,6,9，共 4 条（ceil(10/3)）
    res = _scan("M0,0h10v10h-10Z", 3)
    assert res["line_count"] == 4
    assert _ys(res) == [0.0, 3.0, 6.0, 9.0]


def test_line_between_vertically_disjoint_contours_is_empty():
    # 两个上下分离的外环：y 0..10 与 y 20..30，整体包围盒 0..30
    d = "M0,0h10v10h-10Z M0,20h10v10h-10Z"
    res = _scan(d, 5)
    assert _ys(res) == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    assert _intervals_at(res, 15.0) == []  # 中间空带仍输出一行
    assert _intervals_at(res, 5.0) == [[0.0, 10.0]]
    assert _intervals_at(res, 25.0) == [[0.0, 10.0]]


# ---------------------------------------------------------------------------
# 半开边规则 / 交替合并
# ---------------------------------------------------------------------------

def test_half_open_rule_bottom_edge_included_top_excluded():
    # 首线恰在最小 y：底边与两侧边交点成对，得到完整区间（顶点不重复）
    res = _scan("M0,0h10v10h-10Z", 5)
    assert _intervals_at(res, 0.0) == [[0.0, 10.0]]
    # y=10（最大 y）不含，故不存在
    assert all(not math.isclose(y, 10.0) for y in _ys(res))


def test_vertex_aligned_scanline_not_double_counted():
    # 顶点恰在扫描线上的三角形：每线交点仍成对
    # M0,0 L10,0 L5,10 Z，y=0 底边完整，y=5 截宽 5（x 2.5..7.5）
    res = _scan("M0,0 L10,0 L5,10 Z", 5)
    assert _intervals_at(res, 0.0) == [[0.0, 10.0]]
    assert _intervals_at(res, 5.0) == [[2.5, 7.5]]


def test_zero_width_interval_at_tangent_apex_dropped():
    # 菱形，y=0 仅触及下顶点：两个交点重合为零宽区间，应丢弃
    res = _scan("M5,0 L10,5 L5,10 L0,5 Z", 0.5)
    assert _intervals_at(res, 0.0) == []
    # 中线截宽最大（x 0..10）
    assert _intervals_at(res, 5.0) == [[0.0, 10.0]]


def test_outer_hole_alternating_merge():
    # 外方 0..10，内孔 2..8：y=5 填充 [0,2] 与 [8,10]
    res = _scan("M0,0h10v10h-10Z M2,2h6v6h-6Z", 2)
    for y in (2.0, 4.0, 6.0):
        assert _intervals_at(res, y) == [[0.0, 2.0], [8.0, 10.0]]


def test_outer_hole_island_alternating_merge():
    # 外 0..10 / 孔 1..9 / 岛 6..8：y=7 填充 [0,1]、[6,8]、[9,10]
    d = "M0,0h10v10h-10Z M1,1h8v8h-8Z M6,6h2v2h-2Z"
    res = _scan(d, 1)
    assert _intervals_at(res, 7.0) == [[0.0, 1.0], [6.0, 8.0], [9.0, 10.0]]
    # 岛外的孔层 y=1.5（岛在 y 6..8 之外）只剩两侧
    assert _intervals_at(res, 1.0) == [[0.0, 1.0], [9.0, 10.0]]


def test_four_level_nesting_merge():
    # 外 0..10 / 孔 1..9 / 岛 3..7 / 岛中孔 4..6
    d = ("M0,0h10v10h-10Z M1,1h8v8h-8Z "
         "M3,3h4v4h-4Z M4,4h2v2h-2Z")
    res = _scan(d, 1)
    assert _intervals_at(res, 5.0) == [
        [0.0, 1.0], [3.0, 4.0], [6.0, 7.0], [9.0, 10.0]]


def test_disjoint_outers_produce_separate_intervals():
    # 两个左右分离的外环：同一行上得到两段升序、不重叠区间
    d = "M0,0h10v10h-10Z M30,0h10v10h-10Z"
    res = _scan(d, 5)
    intervals = _intervals_at(res, 0.0)
    assert intervals == [[0.0, 10.0], [30.0, 40.0]]
    # 全部区间升序、互不重叠
    for line in res["lines"]:
        xs = [x for iv in line["intervals"] for x in iv]
        assert xs == sorted(xs)
        for a, b in zip(line["intervals"], line["intervals"][1:]):
            assert a[1] <= b[0]


# ---------------------------------------------------------------------------
# 曲线细分 / 开放与退化跳过 / 空路径
# ---------------------------------------------------------------------------

def test_curves_flattened_before_intersection():
    # 单位圆（两段半弧细分）：y=0 截宽近似 2
    d = "M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z"
    res = _scan(d, 0.5)
    x0, x1 = _intervals_at(res, 0.0)[0]
    assert 1.98 < x1 - x0 <= 2.0 + 1e-9
    # 越靠近上下顶点截宽越窄
    w_mid = _intervals_at(res, 0.0)[0][1] - _intervals_at(res, 0.0)[0][0]
    w_off = _intervals_at(res, 0.5)[0][1] - _intervals_at(res, 0.5)[0][0]
    assert w_off < w_mid


def test_open_and_degenerate_contours_skipped():
    d = ("M0,0h10v10h-10Z "        # 有效外环
         "M5,5 L8,5 L5,5 Z "       # 退化闭合
         "M20,0 L30,0 L30,10")     # 开放折线
    res = _scan(d, 2)
    assert res["line_count"] == 5
    assert _intervals_at(res, 4.0) == [[0.0, 10.0]]
    skipped = {(s["original_index"], s["reason"])
               for s in res["skipped_contours"]}
    assert skipped == {(1, "degenerate"), (2, "open")}


def test_empty_path_returns_empty_lines():
    res = _scan("", 1)
    assert res["lines"] == []
    assert res["line_count"] == 0
    assert res["skipped_contours"] == []


def test_only_open_contours_returns_empty_lines():
    res = _scan("M0,0 L10,0 L10,10", 1)
    assert res["lines"] == [] and res["line_count"] == 0
    assert res["skipped_contours"] == [
        {"original_index": 0, "role": "open", "reason": "open"}]


def test_only_degenerate_contours_returns_empty_lines():
    res = _scan("M0,0 L10,0 Z", 1)
    assert res["lines"] == [] and res["line_count"] == 0
    assert res["skipped_contours"][0]["reason"] == "degenerate"


# ---------------------------------------------------------------------------
# 上限 / 参数校验 / 确定性
# ---------------------------------------------------------------------------

def test_exceeding_line_limit_raises():
    # 高度 10、间距 2 共 5 行：上限 4 即超限
    with pytest.raises(ScanlineLimitError):
        _scan("M0,0h10v10h-10Z", 2, max_lines=4)


def test_line_limit_boundary_accepted():
    # 恰好 5 行：上限 5 接受
    res = _scan("M0,0h10v10h-10Z", 2, max_lines=5)
    assert res["line_count"] == 5


def test_default_limit_is_1000():
    assert MAX_SCANLINES == 1000
    # 高度 10、间距 0.01 恰 1000 行 -> 不超限
    res = _scan("M0,0h10v10h-10Z", 0.01)
    assert res["line_count"] == 1000
    # 间距更小 -> 超限
    with pytest.raises(ScanlineLimitError):
        _scan("M0,0h10v10h-10Z", 0.009)


def test_invalid_spacing_raises():
    contours, _ = analyze_contours("M0,0h10v10h-10Z", "nonzero")
    with pytest.raises(ValueError):
        compute_scanlines(contours, 0)
    with pytest.raises(ValueError):
        compute_scanlines(contours, -1)
    with pytest.raises(ValueError):
        compute_scanlines(contours, float("nan"))
    with pytest.raises(ValueError):
        compute_scanlines(contours, float("inf"))


def test_results_deterministic():
    d = "M0,0h10v10h-10Z M1,1h8v8h-8Z M6,6h2v2h-2Z"
    outs = [_scan(d, 1) for _ in range(3)]
    assert outs[0] == outs[1] == outs[2]
