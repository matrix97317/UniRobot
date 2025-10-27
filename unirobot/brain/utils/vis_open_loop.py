# -*- coding: utf-8 -*-
"""Brain."""
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_dimension_comparison(
    data_list, figsize=(20, 25), dims=6, save_path=None, dpi=300
):
    """
    绘制14个维度的GT和Model预测值对比曲线并保存为JPG

    参数:
    data_list: 包含字典的列表，每个字典有'gt'和'model'键，值为14维numpy数组
    figsize: 图像大小
    save_path: 保存路径，如果为None则不保存
    dpi: 图片分辨率
    """
    # 确定数据点数量
    n_points = len(data_list)
    x = range(n_points)

    # 创建14个子图
    fig, axes = plt.subplots(int(dims / 2), 2, figsize=figsize)
    axes = axes.flatten()

    # 为每个维度绘制对比图
    for dim in range(dims):
        ax = axes[dim]

        # 提取该维度的GT和Model数据
        gt_data = [data["gt"][dim] for data in data_list]
        model_data = [data["model"][dim] for data in data_list]

        # 绘制曲线
        ax.plot(x, gt_data, label="GT", linewidth=1.5, alpha=0.8, color="blue")
        ax.plot(x, model_data, label="Model", linewidth=1.5, alpha=0.8, color="red")

        # 设置标题和标签
        ax.set_title(f"Dimension {dim + 1} Comparison", fontsize=12, fontweight="bold")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel(f"Value - Dim {dim + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加一些统计信息
        gt_mean = np.mean(gt_data)
        model_mean = np.mean(model_data)
        correlation = np.corrcoef(gt_data, model_data)[0, 1]

        ax.text(
            0.02,
            0.98,
            f"GT Mean: {gt_mean:.3f}\nModel Mean: {model_mean:.3f}\nCorr: {correlation:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if save_path is not None:
        # 如果目录不存在则创建
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format="jpg")
        print(f"图片已保存至: {save_path}")

    # 显示图片
    plt.show()

    # 关闭图形以释放内存
    plt.close()
