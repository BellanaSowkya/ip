"""
leaf_apex_fft_colab.py
=======================

Colab-friendly version. Two changes from the command-line version:

  1. No argparse (Colab's launcher injects its own args, which breaks
     argparse). Instead you call preview_first_frame(...) and run(...)
     directly from notebook cells.

  2. No cv2.imshow()/cv2.waitKey() popup (Colab has no display for that),
     no plt.ginput() (Colab's inline backend is non-interactive, so
     ginput() just hangs), and no input() prompt either (Colab's input
     box can be unreliable depending on output/scroll state). Instead,
     point selection is a fully explicit two-step, no-prompt workflow.

--------------------------------------------------------------------
HOW TO USE IN COLAB (paste this code into a cell, then run these)
--------------------------------------------------------------------
!pip install opencv-python-headless numpy matplotlib --quiet

from google.colab import files
uploaded = files.upload()          # pick your ~30s leaf video

# Cell 2: just look at the frame, no input needed
preview_first_frame("leaf_video.mp4")
# -> shows frame 0 with a yellow pixel grid. Read off the apex's
#    approximate (col, row) with your eyes, e.g. col=340, row=210.

# Cell 3: pass the coordinates you read as a normal argument
run(video_path="leaf_video.mp4", apex_xy=(340, 210))
# -> this now also displays (and saves) the leaf frame itself with the
#    tracked apex path drawn on it, in addition to the FFT plots.
--------------------------------------------------------------------
"""

import csv
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# 1. POINT SELECTION (first frame)
# --------------------------------------------------------------------------
# NOTE: plt.ginput() needs an interactive matplotlib backend with a live
# event loop -- Colab's '%matplotlib inline' backend doesn't have one, so
# ginput() just hangs. And Colab's input() box can also be unreliable
# depending on output/scroll state. So selection is now a fully explicit,
# two-step, no-prompt workflow:
#
#   Step 1: preview_first_frame(video_path)
#           -> just displays frame 0 with a pixel grid overlay so you can
#              read off the apex coordinates with your eyes.
#   Step 2: run(video_path, apex_xy=(col, row))
#           -> you pass in the coordinates you read, as a normal Python
#              argument. No input()/ginput() involved at all.
def preview_first_frame(video_path):
    """Display frame 0 with a pixel grid so you can read off coordinates."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        sys.exit("Could not read first frame.")

    frame_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    step_x = max(1, w // 15)
    step_y = max(1, h // 15)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame_rgb)
    ax.set_xticks(np.arange(0, w, step_x))
    ax.set_yticks(np.arange(0, h, step_y))
    ax.grid(True, color="yellow", alpha=0.4, linewidth=0.5)
    ax.set_xlabel("column (x)")
    ax.set_ylabel("row (y)")
    ax.set_title("Read off the leaf apex (col, row) from this grid,\n"
                  "then pass it to run(video_path, apex_xy=(col, row))")
    plt.show()
    print(f"Frame size: width={w} px, height={h} px")


# --------------------------------------------------------------------------
# 2. TRACK THE POINT ACROSS THE VIDEO (Lucas-Kanade optical flow)
# --------------------------------------------------------------------------
def track_apex(video_path, apex_xy):
    """apex_xy: (col, row) pixel coordinates of the apex in frame 0,
    e.g. as read off the grid from preview_first_frame()."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ok, frame0 = cap.read()
    if not ok:
        sys.exit("Could not read first frame.")

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    x0, y0 = float(apex_xy[0]), float(apex_xy[1])

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    p_prev = np.array([[[x0, y0]]], dtype=np.float32)  # shape (1,1,2)

    rows = [y0]
    cols = [x0]
    frame_idx = [0]

    gray_prev = gray0
    idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p_next, status, err = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray, p_prev, None, **lk_params
        )

        if status[0][0] == 0:
            print(f"Warning: lost track at frame {idx}; "
                  f"holding last known position.")
            p_next = p_prev

        x, y = p_next[0, 0]
        rows.append(float(y))
        cols.append(float(x))
        frame_idx.append(idx)

        gray_prev = gray
        p_prev = p_next
        idx += 1

    cap.release()
    return np.array(frame_idx), np.array(rows), np.array(cols), fps


# --------------------------------------------------------------------------
# 3. SAVE TRAJECTORY TO CSV
# --------------------------------------------------------------------------
def save_trajectory_csv(path, frame_idx, rows, cols, fps):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "row_y", "col_x"])
        for i, r, c in zip(frame_idx, rows, cols):
            w.writerow([i, i / fps, r, c])
    print(f"Trajectory saved to {path}")


# --------------------------------------------------------------------------
# 3b. SHOW THE LEAF WITH THE TRACKED PATH DRAWN ON IT
# --------------------------------------------------------------------------
def show_tracked_overlay(video_path, rows, cols, out_path="leaf_apex_overlay.png"):
    """Draw the full tracked apex path on top of the actual leaf frame
    (first frame as background), so you can visually check the tracking
    quality -- not just the pixel-coordinate plots."""
    cap = cv2.VideoCapture(video_path)
    ok, frame0 = cap.read()
    cap.release()
    if not ok:
        print("Could not re-read first frame for overlay; skipping.")
        return

    frame_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB).copy()
    h, w = frame_rgb.shape[:2]

    pts = np.stack([cols, rows], axis=1).astype(np.int32)  # (x, y) pairs
    for i in range(1, len(pts)):
        # frame_rgb is in RGB order, so (0,0,255) = blue
        cv2.line(frame_rgb, tuple(pts[i - 1]), tuple(pts[i]), (0, 0, 255), 2)
    # start = green, end = red (RGB order)
    cv2.circle(frame_rgb, tuple(pts[0]), 8, (0, 255, 0), -1)
    cv2.circle(frame_rgb, tuple(pts[-1]), 8, (255, 0, 0), -1)

    step_x = max(1, w // 15)
    step_y = max(1, h // 15)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(frame_rgb)
    ax.set_xticks(np.arange(0, w, step_x))
    ax.set_yticks(np.arange(0, h, step_y))
    ax.grid(True, color="yellow", alpha=0.4, linewidth=0.5)
    ax.set_xlabel("column (x)")
    ax.set_ylabel("row (y)")
    ax.set_title("Leaf with tracked apex path\n"
                  "(green = start, blue line = path, red = end)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Overlay image saved to {out_path}")
    plt.show()


# --------------------------------------------------------------------------
# 4. FFT ANALYSIS
# --------------------------------------------------------------------------
def analyze_and_plot(frame_idx, rows, cols, fps, out_prefix="leaf_apex"):
    n = len(frame_idx)
    t = frame_idx / fps

    row_sig = rows - np.mean(rows)
    col_sig = cols - np.mean(cols)

    # ---- (a) Separate 1D FFTs -------------------------------------------
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    ROW_F = np.fft.rfft(row_sig)
    COL_F = np.fft.rfft(col_sig)

    # ---- (b) Complex-signal FFT (captures 2D rotational motion) --------
    z = col_sig + 1j * row_sig
    freqs_full = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fps))
    Z_F = np.fft.fftshift(np.fft.fft(z))

    # ---- (c) Literal 2D FFT of the (N x 2) trajectory matrix -----------
    traj_matrix = np.stack([row_sig, col_sig], axis=1)  # shape (N, 2)
    FFT2D = np.fft.fft2(traj_matrix)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    axes[0, 0].plot(t, rows, label="row (y)")
    axes[0, 0].plot(t, cols, label="col (x)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Pixel coordinate")
    axes[0, 0].set_title("Tracked apex trajectory")
    axes[0, 0].legend()

    axes[0, 1].plot(cols, rows, ".-", ms=2)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("col (x)")
    axes[0, 1].set_ylabel("row (y)")
    axes[0, 1].set_title("Apex path in image plane")

    axes[1, 0].plot(freqs, np.abs(ROW_F))
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("|FFT|")
    axes[1, 0].set_title("1D FFT of row(t)")

    axes[1, 1].plot(freqs, np.abs(COL_F))
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("|FFT|")
    axes[1, 1].set_title("1D FFT of col(t)")

    axes[2, 0].plot(freqs_full, np.abs(Z_F))
    axes[2, 0].set_xlabel("Frequency (Hz)")
    axes[2, 0].set_ylabel("|FFT|")
    axes[2, 0].set_title("Complex FFT of z(t)=col+i*row\n"
                          "(sign of freq = direction of rotation)")

    im = axes[2, 1].imshow(
        np.abs(np.fft.fftshift(FFT2D)),
        aspect="auto",
        cmap="viridis",
        extent=[0, 1, freqs_full.min(), freqs_full.max()],
        origin="lower",
    )
    axes[2, 1].set_xlabel("column index (0=row-signal, 1=col-signal)")
    axes[2, 1].set_ylabel("Frequency (Hz)")
    axes[2, 1].set_title("Literal 2D FFT of [row,col] matrix\n"
                          "(np.fft.fft2, magnitude)")
    fig.colorbar(im, ax=axes[2, 1])

    fig.tight_layout()
    out_png = f"{out_prefix}_fft_analysis.png"
    fig.savefig(out_png, dpi=150)
    print(f"Figure saved to {out_png}")
    plt.show()

    return {
        "freqs_rfft": freqs,
        "row_fft": ROW_F,
        "col_fft": COL_F,
        "freqs_full": freqs_full,
        "complex_fft": Z_F,
        "fft2d": FFT2D,
    }


# --------------------------------------------------------------------------
# ENTRY POINT — call this from a Colab cell
# --------------------------------------------------------------------------
def run(video_path, apex_xy, fps_override=None, csv_path="leaf_apex_trajectory.csv"):
    """apex_xy: (col, row) of the leaf apex in frame 0 -- read this off the
    image shown by preview_first_frame(video_path) first."""
    frame_idx, rows, cols, fps_meta = track_apex(video_path, apex_xy)
    fps = fps_override if fps_override else fps_meta
    print(f"Tracked {len(frame_idx)} frames at {fps:.2f} fps "
          f"({len(frame_idx) / fps:.1f} s of video)")

    save_trajectory_csv(csv_path, frame_idx, rows, cols, fps)
    show_tracked_overlay(video_path, rows, cols)
    return analyze_and_plot(frame_idx, rows, cols, fps)


if __name__ == "__main__":
    # For a plain (non-notebook) run:
    #   1. python -c "from leaf_apex_fft_colab import preview_first_frame; preview_first_frame('leaf.mp4')"
    #      -> read off the apex (col, row) from the saved/shown grid image
    #   2. Edit APEX_XY below to match, then: python leaf_apex_fft_colab.py
    VIDEO_PATH = r"C:\Users\HP\Downloads\leaf_video.mp4"
    APEX_XY = (310, 630)   # <-- replace with your apex (col, row)
    FPS_OVERRIDE = None
    run(VIDEO_PATH, apex_xy=APEX_XY, fps_override=FPS_OVERRIDE)
