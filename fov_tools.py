import numpy as np

# this contains tools for computing various parameters
# related to field of view. As a check, it uses these
# tools to compute GSD at camera center


def dfov(width_px, height_px, pitch_um, f_mm):
    half_diag_px = 0.5 * np.sqrt(width_px**2 + height_px**2)
    half_diag_mm = half_diag_px * pitch_um * 1e-3
    theta = 2 * np.arctan(half_diag_mm / f_mm)

    return theta


def hfov(width_px, height_px, pitch_um, f_mm):
    return dfov(width_px, 0, pitch_um, f_mm)


def vfov(width_px, height_px, pitch_um, f_mm):
    return dfov(0, height_px, pitch_um, f_mm)


def gsd(pitch_um, f_mm, distance):
    return distance * np.tan(hfov(1, 1, pitch_um, f_mm))


def summary(identifier, width_px, height_px, pitch_um, f_mm, distance_m):
    theta_d = dfov(width_px, height_px, pitch_um, f_mm)
    theta_h = hfov(width_px, height_px, pitch_um, f_mm) * 180.0 / np.pi
    theta_v = vfov(width_px, height_px, pitch_um, f_mm) * 180.0 / np.pi
    gsd_mm = gsd(pitch_um, f_mm, distance_m * 1e3)
    print(
        f"{identifier}\n"
        f" focal length   = {f_mm:.2f} mm\n"
        f" FOV   Diagonal = {theta_d * 180. / np.pi:.1f} degrees\n"
        f"     Horizontal = {theta_h:.1f} degrees\n"
        f"       Vertical = {theta_v:.1f} degrees\n"
        f"  GSD at {distance_m:.2f} m = {gsd_mm:.1f} mm\n"
    )


def compute_pitch_um(dfov_rad, width_px, height_px, f_mm):
    half_diag_px = 0.5 * np.sqrt(width_px**2 + height_px**2)
    pitch_um = np.tan(dfov_rad) * f_mm / half_diag_px * 1e3
    return pitch_um


def compute_focal_mm(dfov_rad, width_px, height_px, pitch_um):
    half_diag_mm = 0.5 * np.sqrt(width_px**2 + height_px**2) * pitch_um * 1e-3
    f_mm = half_diag_mm / np.tan(dfov_rad * 0.5)
    return f_mm
