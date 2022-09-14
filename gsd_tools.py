import matplotlib.pyplot as plt
import numpy as np


def gsd(
    u: float,
    v: float,
    pitch_um_per_px: float,
    focal_mm: float,
    camera_height_m: float,
    declination_degrees: float,
):
    dec = np.pi * 0.5 + declination_degrees * np.pi / 180.0
    x = u * pitch_um_per_px * 1e-3
    y = v * pitch_um_per_px * 1e-3
    xp = x
    yp = y * np.cos(dec) + focal_mm * np.sin(dec)
    zp = -y * np.sin(dec) + focal_mm * np.cos(dec)
    return (
        pitch_um_per_px
        * camera_height_m
        / focal_mm
        / (np.cos(np.arctan(np.sqrt(xp**2 + yp**2) / zp)))
    )


def groundPoint(
    u: float,
    v: float,
    sensor_width_pixels: int,
    sensor_height_pixels: int,
    pitch_um_per_px: float,
    focal_mm: float,
    camera_height_m: float,
    declination_degrees: float,
):
    """
    returns (x,y,0) coordinate that corresponds to pixel coordinate (u,v)
    this assumes that (u,v) are relative to the coordinate system with the
    origin at the center of the sensor
    """
    dec = np.pi * 0.5 + declination_degrees * np.pi / 180.0
    x = (u - sensor_width_pixels * 0.5) * pitch_um_per_px * 1e-3
    y = (v - sensor_height_pixels * 0.5) * pitch_um_per_px * 1e-3
    xp = x
    yp = y * np.cos(dec) + focal_mm * np.sin(dec)
    zp = -y * np.sin(dec) + focal_mm * np.cos(dec)
    hmm = camera_height_m * 1e3
    t = hmm / (hmm + zp)
    print(f"xp: {xp.shape} yp: {yp.shape} t: {t.shape} hmm: {hmm}")
    xg = xp * t
    yg = yp * t
    return (xg, yg, zp)


def sensorGSD(
    sensor_width_pixels: int,
    sensor_height_pixels: int,
    pitch_um_per_px: float,
    focal_mm: float,
    declination_deg: float,
    camera_height_m: 2,
) -> tuple:
    """
    compute the GSD over the entire sensor
    assuming the camera is aimed along the
    optical axis and that the ground is planar
    parameters:
    sensor_width_pixels: sensor width in pixels
    sensor_height_pixels: sensor height in pixels
    pitch_um_per_px: size of pixels (assumed square)
    focal_mm: lens focal length (mm)
    declination_deg: camera declination, degrees down from horizon
    camera_height_m: height of camera above ground
    """
    plot_resoluton_pixels = 10
    xl = np.arange(
        -sensor_width_pixels // 2, sensor_width_pixels // 2, plot_resoluton_pixels
    )
    yl = np.arange(
        -sensor_height_pixels // 2, sensor_height_pixels // 2, plot_resoluton_pixels
    )
    (x, y) = np.meshgrid(xl, yl)
    dp = gsd(x, y, pitch_um_per_px, focal_mm, camera_height_m, declination_deg)

    # now you need to flip the sensor coordinates
    # and re-center it at top left corner
    xsens = np.arange(0, sensor_width_pixels, plot_resoluton_pixels)
    ysens = np.arange(sensor_height_pixels, 0, -plot_resoluton_pixels)
    (xs, ys) = np.meshgrid(xsens, ysens)

    # now compute the ground coordinates
    (xg, yg, t) = groundPoint(
        xs,
        ys,
        sensor_width_pixels,
        sensor_height_pixels,
        pitch_um_per_px,
        focal_mm,
        camera_height_m,
        declination_deg,
    )
    # xg, yg = np.meshgrid(xground, yground)
    return (xs, ys, xg, yg, dp, t)


def plot(
    sensor_width_pixels: int,
    sensor_height_pixels: int,
    pitch_um_px: float,
    focal_mm: float,
    declination_deg: float,
    camera_height_m: float,
    levels=[0.5, 1.0, 1.25, 1.5, 2.5, 3.5],
) -> None:
    """
    plot the GSD over the entire sensor
    assuming the camera is aimed along the
    optical axis and that the ground is planar
    parameters:
    sensor_width_pixels: sensor width in pixels
    sensor_height_pixels: sensor height in pixels
    pitch_um_per_px: size of pixels (assumed square)
    focal_mm: lens focal length (mm)
    declination_deg: camera declination, degrees down from horizon
    camera_height_m: height of camera above ground
    """

    (xs, ys, xg, yg, z, t) = sensorGSD(
        sensor_width_pixels,
        sensor_height_pixels,
        pitch_um_px,
        focal_mm,
        declination_deg,
        camera_height_m,
    )

    fig, axs = plt.subplots(1,2)
    fig.set_figwidth(fig.get_figwidth() * 2)
    ctf = axs[0].contourf(xs, ys, z, levels=levels)
    plt.colorbar(ctf)
    axs[0].set_xlabel("x-pixel coordinate")
    axs[0].set_ylabel("y-pixel coordinate")
    axs[0].set_title("GSD (mm) displayed in sensor coordinates")
    axs[0].grid()

    print(f"t: {min(t.ravel())} : {max(t.ravel())} {max(t.ravel()) - min(t.ravel())}")
    print(f"camera height mm {1e3 * camera_height_m}")
    # ctg = axs[1].contourf(xg, yg, z, levels=levels)
    ctg = axs[1].contourf(xg, yg, t)
    axs[1].set_xlabel("x coordinate (m?)")
    axs[1].set_ylabel("y coordinate (m?)")
    axs[1].set_title("GSD (mm) displayed in ground coordinates")
    axs[1].grid()
    plt.show()
    return


def computeGsdCenter(
    pitch_um_px: float,
    focal_mm: float,
    declination_deg: float,
    camera_height_m: float,
) -> tuple:
    """
    compute the distance between the camera center
    and the ground point at which the center is
    projected and also calculate the GSD at this point
    """
    center_distance = camera_height_m / np.sin(declination_deg * np.pi / 180)
    center_GSD_mm = pitch_um_px / focal_mm * center_distance
    return center_distance, center_GSD_mm


def summary(
    sensor_width_pixels: int,
    sensor_height_pixels: int,
    pitch_um_px: float,
    focal_mm: float,
    declination_deg: float,
    camera_height_m: float,
) -> None:
    """
    sensor_width_pixels: sensor width in pixels
    sensor_height_pixels: sensor height in pixels
    pitch_um_per_px: size of pixels (assumed square)
    focal_mm: lens focal length (mm)
    declination_deg: camera declination, degrees down from horizon
    camera_height_m: height of camera above ground
    """

    center_distance, center_GSD_mm = computeGsdCenter(
        pitch_um_px,
        focal_mm,
        declination_deg,
        camera_height_m,
    )
    # summary:
    print(
        f"Sensor shape: [{sensor_width_pixels}, {sensor_height_pixels}] pixels\n"
        f"focal length {focal_mm:0.2e} mm, pitch {1e-3*pitch_um_px:0.3e} mm/pixel\n"
        f"camera height {camera_height_m} m\n"
        f"declination {declination_deg} degrees down from horizontal\n"
        f"estimated GSD at center {center_GSD_mm:0.1f} mm\n"
        f"distance to projected camera center {center_distance:0.2f}"
    )
    return


if __name__ == "__main__":
    print("Demo run for the Lucid Trident 230s with a 12mm lens")
    sensor_width_pixels = 5164
    sensor_height_pixels = 3873
    pitch_um_px = 1.47  # microns/pixel
    focal_mm = 6.83  # mm
    declination_deg = 30  # degrees down from horizontal
    camera_height_m = 2  # meters

    summary(
        sensor_width_pixels,
        sensor_height_pixels,
        pitch_um_px,
        focal_mm,
        declination_deg,
        camera_height_m,
    )

    plot(
        sensor_width_pixels,
        sensor_height_pixels,
        pitch_um_px,
        focal_mm,
        declination_deg,
        camera_height_m,
        levels=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    )
