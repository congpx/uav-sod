import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


def add_box(ax, center, width, height, text,
            facecolor="#f8f9fa", edgecolor="black",
            fontsize=11, lw=1.4, linestyle="-", zorder=2):
    x = center[0] - width / 2
    y = center[1] - height / 2
    rect = Rectangle(
        (x, y), width, height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder
    )
    ax.add_patch(rect)
    ax.text(
        center[0], center[1], text,
        ha="center", va="center",
        fontsize=fontsize,
        family="sans-serif",
        zorder=zorder + 1
    )
    return rect


def add_two_line_box(ax, center, width, height,
                     line1, line2,
                     facecolor="#f8f9fa", edgecolor="black",
                     fontsize1=11, fontsize2=8,
                     italic2=True, lw=1.4, linestyle="-", zorder=2):
    x = center[0] - width / 2
    y = center[1] - height / 2
    rect = Rectangle(
        (x, y), width, height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder
    )
    ax.add_patch(rect)

    # line 1
    ax.text(
        center[0], center[1] + 0.18,
        line1,
        ha="center", va="center",
        fontsize=fontsize1,
        family="sans-serif",
        zorder=zorder + 1
    )

    # line 2
    ax.text(
        center[0], center[1] - 0.18,
        line2,
        ha="center", va="center",
        fontsize=fontsize2,
        style="italic" if italic2 else "normal",
        family="sans-serif",
        zorder=zorder + 1
    )
    return rect


def box_anchor(center, width, height, side):
    x, y = center
    if side == "top":
        return (x, y + height / 2)
    if side == "bottom":
        return (x, y - height / 2)
    if side == "left":
        return (x - width / 2, y)
    if side == "right":
        return (x + width / 2, y)
    raise ValueError("side must be top/bottom/left/right")


def add_arrow(ax, p1, p2, style="-", lw=1.4, color="black",
              connectionstyle="arc3,rad=0.0",
              arrowstyle="-|>", mutation_scale=18, zorder=3):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=connectionstyle,
        zorder=zorder
    )
    ax.add_patch(arr)
    return arr


def add_elbow_arrow(ax, p1, p2, mid_x=None, mid_y=None,
                    style="-", lw=1.4, color="black",
                    arrowstyle="-|>", mutation_scale=18, zorder=3):
    """
    Vẽ mũi tên gấp khúc:
    - nếu có mid_x: p1 -> (mid_x, y1) -> (mid_x, y2) -> p2
    - nếu có mid_y: p1 -> (x1, mid_y) -> (x2, mid_y) -> p2
    """
    x1, y1 = p1
    x2, y2 = p2

    if mid_x is not None:
        ax.plot([x1, mid_x], [y1, y1],
                linestyle=style, linewidth=lw, color=color, zorder=zorder)
        ax.plot([mid_x, mid_x], [y1, y2],
                linestyle=style, linewidth=lw, color=color, zorder=zorder)
        add_arrow(
            ax,
            (mid_x, y2),
            p2,
            style=style,
            lw=lw,
            color=color,
            connectionstyle="arc3,rad=0.0",
            arrowstyle=arrowstyle,
            mutation_scale=mutation_scale,
            zorder=zorder
        )
    elif mid_y is not None:
        ax.plot([x1, x1], [y1, mid_y],
                linestyle=style, linewidth=lw, color=color, zorder=zorder)
        ax.plot([x1, x2], [mid_y, mid_y],
                linestyle=style, linewidth=lw, color=color, zorder=zorder)
        add_arrow(
            ax,
            (x2, mid_y),
            p2,
            style=style,
            lw=lw,
            color=color,
            connectionstyle="arc3,rad=0.0",
            arrowstyle=arrowstyle,
            mutation_scale=mutation_scale,
            zorder=zorder
        )
    else:
        raise ValueError("Either mid_x or mid_y must be provided.")


def main():
    fig, ax = plt.subplots(figsize=(8.0, 10.8))
    ax.set_xlim(0, 12.4)   # co bề ngang lại cho compact hơn
    ax.set_ylim(0, 18)
    ax.axis("off")

    # ===== Colors =====
    c_input = "#F2F2F2"
    c_backbone = "#E8F1FB"
    c_neck = "#E8F1FB"
    c_p2 = "#FFD966"
    c_head = "#D9EAD3"
    c_pred = "#D9EAF7"
    c_output = "#EDEDED"

    c_gt = "#FCE5CD"
    c_area = "#F4CCCC"
    c_assign = "#D9D2E9"
    c_pos = "#F9CB9C"
    c_loss = "#CFE2F3"

    container_face = "#FBFBFB"
    container_edge = "#666666"

    # ===== Containers =====
    detector_container = Rectangle(
        (0.8, 7.9), 10.8, 9.1,
        facecolor=container_face,
        edgecolor=container_edge,
        linewidth=1.5,
        linestyle="-",
        zorder=0
    )
    ax.add_patch(detector_container)
    ax.text(1.0, 16.65, "Final Detector", fontsize=13, fontweight="bold", va="center")

    train_container = Rectangle(
        (0.8, 0.7), 10.8, 5.9,   # hạ xuống thấp hơn
        facecolor=container_face,
        edgecolor=container_edge,
        linewidth=1.5,
        linestyle="--",
        zorder=0
    )
    ax.add_patch(train_container)
    ax.text(1.0, 6.35, "Training-Time Assignment Refinement",
            fontsize=13, fontweight="bold", va="center")

    ax.text(5.0, 17.45, "Dashed arrows: training-only path", fontsize=10)
    ax.text(5.0, 17.05, "Solid arrows: inference + training path", fontsize=10)

    # ===== Top section =====
    input_c = (6.2, 15.6)
    backbone_c = (6.2, 14.0)
    neck_c = (6.2, 12.3)

    # P2 rộng hơn, P3/P4/P5 hẹp hơn
    p2_c = (2.8, 10.8)
    p3_c = (5.4, 10.8)
    p4_c = (7.3, 10.8)
    p5_c = (9.2, 10.8)

    pred_c = (6.2, 8.95)
    final_det_c = (6.2, 7.65)

    w_big, h_big = 4.2, 0.9
    w_pred, h_pred = 4.2, 0.95
    w_p2, h_head = 3.1, 0.9
    w_head_small = 1.35

    add_box(
        ax, input_c, w_big, h_big,
        "Input UAV Image\n(e.g., 640×640 or 800×800)",
        facecolor=c_input, fontsize=12
    )

    add_box(
        ax, backbone_c, w_big, h_big,
        "YOLOv8n Backbone\nConv / C2f / SPPF",
        facecolor=c_backbone, fontsize=12
    )

    add_box(
        ax, neck_c, w_big, h_big,
        "FPN/PAN Neck\nmulti-scale feature fusion",
        facecolor=c_neck, fontsize=12
    )

    add_two_line_box(
        ax, p2_c, w_p2, h_head,
        "P2", "(added shallow head)",
        facecolor=c_p2, fontsize1=12, fontsize2=8, italic2=True, lw=1.8
    )

    add_box(ax, p3_c, w_head_small, h_head, "P3", facecolor=c_head, fontsize=11)
    add_box(ax, p4_c, w_head_small, h_head, "P4", facecolor=c_head, fontsize=11)
    add_box(ax, p5_c, w_head_small, h_head, "P5", facecolor=c_head, fontsize=11)

    add_box(
        ax, pred_c, w_pred, h_pred,
        "Detection Predictions\n(boxes + class scores)",
        facecolor=c_pred, fontsize=12
    )

    add_two_line_box(
        ax, final_det_c, w_big, h_big,
        "Final Detections", "(after NMS)",
        facecolor=c_output, fontsize1=12, fontsize2=8, italic2=True
    )

    # Arrows top
    add_arrow(ax, box_anchor(input_c, w_big, h_big, "bottom"),
              box_anchor(backbone_c, w_big, h_big, "top"))

    add_arrow(ax, box_anchor(backbone_c, w_big, h_big, "bottom"),
              box_anchor(neck_c, w_big, h_big, "top"))

    add_arrow(ax, box_anchor(neck_c, w_big, h_big, "bottom"),
              box_anchor(p2_c, w_p2, h_head, "top"))
    add_arrow(ax, box_anchor(neck_c, w_big, h_big, "bottom"),
              box_anchor(p3_c, w_head_small, h_head, "top"))
    add_arrow(ax, box_anchor(neck_c, w_big, h_big, "bottom"),
              box_anchor(p4_c, w_head_small, h_head, "top"))
    add_arrow(ax, box_anchor(neck_c, w_big, h_big, "bottom"),
              box_anchor(p5_c, w_head_small, h_head, "top"))

    add_arrow(ax, box_anchor(p2_c, w_p2, h_head, "bottom"),
              box_anchor(pred_c, w_pred, h_pred, "top"))
    add_arrow(ax, box_anchor(p3_c, w_head_small, h_head, "bottom"),
              box_anchor(pred_c, w_pred, h_pred, "top"))
    add_arrow(ax, box_anchor(p4_c, w_head_small, h_head, "bottom"),
              box_anchor(pred_c, w_pred, h_pred, "top"))
    add_arrow(ax, box_anchor(p5_c, w_head_small, h_head, "bottom"),
              box_anchor(pred_c, w_pred, h_pred, "top"))

    # Detection Predictions -> Final Detections
    add_arrow(ax, box_anchor(pred_c, w_pred, h_pred, "bottom"),
              box_anchor(final_det_c, w_big, h_big, "top"))

    # ===== Bottom section =====
    gt_c = (2.2, 4.9)
    gt_area_c = (2.2, 3.45)
    area_weight_c = (2.2, 1.95)

    assign_c = (6.0, 5.0)
    positive_c = (6.0, 3.55)
    loss_c = (6.0, 1.95)

    w_gt, h_gt = 2.6, 0.9
    w_small, h_small = 2.8, 0.9
    w_assign, h_assign = 4.3, 1.0
    w_loss, h_loss = 3.0, 0.95

    add_box(ax, gt_c, w_gt, h_gt,
            "Ground-Truth Boxes", facecolor=c_gt, fontsize=11.5)

    add_box(ax, gt_area_c, w_small, h_small,
            "GT Area Computation\nA = w × h", facecolor=c_area, fontsize=11)

    add_two_line_box(
        ax, area_weight_c, w_small, h_small,
        "Area Weight", "(smaller GT → larger weight)",
        facecolor=c_area, fontsize1=11, fontsize2=8, italic2=True
    )

    add_box(ax, assign_c, w_assign, h_assign,
            "Assignment Module\n(TAL / Area-aware TAL)",
            facecolor=c_assign, fontsize=12)

    add_two_line_box(
        ax, positive_c, w_small + 0.2, h_small,
        "Positive Assignment", "(selected positives)",
        facecolor=c_pos, fontsize1=11.5, fontsize2=8, italic2=True
    )

    add_box(ax, loss_c, w_loss, h_loss,
            "Detection Losses\n(box / cls / DFL)",
            facecolor=c_loss, fontsize=12)

    # Bottom arrows
    add_arrow(ax, box_anchor(gt_c, w_gt, h_gt, "right"),
              box_anchor(assign_c, w_assign, h_assign, "left"))

    add_arrow(ax, box_anchor(gt_c, w_gt, h_gt, "bottom"),
              box_anchor(gt_area_c, w_small, h_small, "top"))

    add_arrow(ax, box_anchor(gt_area_c, w_small, h_small, "bottom"),
              box_anchor(area_weight_c, w_small, h_small, "top"))

    # Area Weight -> Assignment Module (gấp khúc, có mũi tên rõ)
    add_elbow_arrow(
        ax,
        box_anchor(area_weight_c, w_small, h_small, "right"),
        box_anchor(assign_c, w_assign, h_assign, "bottom"),
        mid_x=4.2
    )

    # Detection Predictions -> Assignment Module (training only, dashed)
    # Vẽ tường minh, xuất phát từ Detection Predictions chứ không đi qua Final Detections
    add_elbow_arrow(
        ax,
        box_anchor(pred_c, w_pred, h_pred, "bottom"),
        box_anchor(assign_c, w_assign, h_assign, "top"),
        mid_x=6.2,
        style="--"
    )
    ax.text(6.45, 6.7, "(training only)", fontsize=9, style="italic")

    add_arrow(ax, box_anchor(assign_c, w_assign, h_assign, "bottom"),
              box_anchor(positive_c, w_small + 0.2, h_small, "top"))

    add_arrow(ax, box_anchor(positive_c, w_small + 0.2, h_small, "bottom"),
              box_anchor(loss_c, w_loss, h_loss, "top"))

    # Detection Predictions -> Detection Losses (gấp khúc thẳng)
    add_elbow_arrow(
        ax,
        box_anchor(pred_c, w_pred, h_pred, "right"),
        box_anchor(loss_c, w_loss, h_loss, "right"),
        mid_x=8.9
    )

    ax.text(8.95, 3.05, "loss computation", fontsize=10, rotation=90, alpha=0.85)

    ax.set_title("Overall Framework of the Proposed Study",
                 fontsize=15, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig("figure1_compact_fixed.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()