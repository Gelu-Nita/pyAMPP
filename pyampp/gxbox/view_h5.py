#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import warnings
import tempfile
try:
    from pyvista import PyVistaDeprecationWarning
except Exception:
    PyVistaDeprecationWarning = DeprecationWarning
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import Heliocentric, get_earth
from sunpy.sun import constants as sun_consts

from pyampp.gxbox.boxutils import read_b3d_h5
from pyampp.gxbox.magfield_viewer import MagFieldViewer
from PyQt5.QtWidgets import QApplication, QFileDialog


def _decode_meta_text(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "ignore")
    if isinstance(value, np.ndarray) and value.shape == ():
        item = value.item()
        if isinstance(item, (bytes, bytearray)):
            return item.decode("utf-8", "ignore")
        return str(item)
    return str(value)


def _to_xyz_if_zyx(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return np.transpose(arr, (2, 1, 0))
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return np.transpose(arr, (2, 1, 0, 3))
    if arr.ndim == 4 and arr.shape[0] == 3:
        # (c, z, y, x) -> (x, y, z, c)
        return np.transpose(arr, (3, 2, 1, 0))
    return arr


def normalize_viewer_axis_order(b3d: dict) -> dict:
    """
    Convert canonical H5 zyx cubes into viewer xyz cubes.
    MagFieldViewer expects (x, y, z).
    """
    meta = b3d.get("metadata", {}) if isinstance(b3d, dict) else {}
    axis_order = _decode_meta_text(meta.get("axis_order_3d", "")).strip().lower()
    if axis_order != "zyx":
        return b3d

    for model_key in ("corona", "chromo", "nlfff", "pot"):
        if model_key not in b3d or not isinstance(b3d[model_key], dict):
            continue
        for comp in ("bx", "by", "bz", "bcube", "chromo_bcube"):
            if comp in b3d[model_key]:
                b3d[model_key][comp] = _to_xyz_if_zyx(np.asarray(b3d[model_key][comp]))
    return b3d


@dataclass
class SimpleBox:
    dims_pix: np.ndarray
    res: u.Quantity
    b3d: dict
    _frame_obs: object
    _center: SkyCoord

    @property
    def grid_coords(self):
        dims = self.dims_pix
        dx = self.res
        x = np.linspace(-dims[0] / 2, dims[0] / 2, dims[0]) * dx
        y = np.linspace(-dims[1] / 2, dims[1] / 2, dims[1]) * dx
        z = np.linspace(-dims[2] / 2, dims[2] / 2, dims[2]) * dx
        return {"x": x, "y": y, "z": z, "frame": self._frame_obs}


def infer_dims(b3d: dict) -> np.ndarray:
    for key in ("corona", "nlfff", "pot"):
        if key in b3d and "bx" in b3d[key]:
            return np.array(b3d[key]["bx"].shape, dtype=int)
    if "chromo" in b3d:
        if "bx" in b3d["chromo"]:
            return np.array(b3d["chromo"]["bx"].shape, dtype=int)
        if "bcube" in b3d["chromo"]:
            return np.array(b3d["chromo"]["bcube"].shape[:3], dtype=int)
        if "chromo_bcube" in b3d["chromo"]:
            return np.array(b3d["chromo"]["chromo_bcube"].shape[1:4], dtype=int)
    raise ValueError("Unable to infer dimensions from HDF5.")


def infer_time(b3d: dict) -> Time:
    if "chromo" in b3d and "attrs" in b3d["chromo"]:
        attrs = b3d["chromo"]["attrs"]
        if "obs_time" in attrs:
            try:
                return Time(attrs["obs_time"])
            except Exception:
                pass
    return Time.now()


def infer_res(b3d: dict) -> u.Quantity:
    if "corona" in b3d and "dr" in b3d["corona"]:
        dr = b3d["corona"]["dr"]
        if dr is not None and np.size(dr) >= 1:
            return (dr[0] * sun_consts.radius.to(u.km).value) * u.km
    if "chromo" in b3d and "dr" in b3d["chromo"]:
        dr = b3d["chromo"]["dr"]
        if dr is not None and np.size(dr) >= 1:
            return (dr[0] * sun_consts.radius.to(u.km).value) * u.km
    return 1.0 * u.Mm


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a saved HDF5 model in the 3D viewer without recomputing.")
    parser.add_argument("h5_path", nargs="?", help="Path to the HDF5 model file (positional).")
    parser.add_argument("--h5", dest="h5_opt", help="Path to the HDF5 model file.")
    parser.add_argument("--dir", dest="start_dir", help="Initial directory for file picker when no model path is given.")
    parser.add_argument("--pick", action="store_true", help="Open file picker even when model path is provided.")
    args = parser.parse_args()

    h5_arg = args.h5_opt or args.h5_path
    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication([])
        owns_app = True

    if args.pick or not h5_arg:
        start_dir = Path(args.start_dir).expanduser() if args.start_dir else Path.cwd()
        if not start_dir.exists() or not start_dir.is_dir():
            start_dir = Path.cwd()
        dialog = QFileDialog(None, "Open Model (HDF5 or SAV)")
        # Native macOS picker may ignore selectFile(); use Qt dialog for reliable preselection.
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Model Files (*.h5 *.sav);;HDF5 Files (*.h5);;SAV Files (*.sav);;All Files (*)")
        if h5_arg:
            candidate = Path(h5_arg).expanduser()
            dialog.setDirectory(str(candidate.parent if candidate.parent.exists() else start_dir))
            dialog.selectFile(str(candidate.name))
        else:
            dialog.setDirectory(str(start_dir))
        if not dialog.exec_():
            return 0
        selected = dialog.selectedFiles()
        if not selected:
            return 0
        h5_arg = selected[0]

    model_path = Path(h5_arg).expanduser().resolve()
    temp_h5_path = None
    if model_path.suffix.lower() == ".sav":
        try:
            from pyampp.tests.build_h5_from_sav import build_h5_from_sav
        except Exception as exc:
            raise RuntimeError(
                "SAV input requires converter module 'pyampp.tests.build_h5_from_sav'. "
                "Run conversion manually to H5, then reopen."
            ) from exc
        tmp_dir = Path(tempfile.mkdtemp(prefix="pyampp_view_h5_"))
        temp_h5_path = tmp_dir / f"{model_path.stem}.viewer.h5"
        build_h5_from_sav(sav_path=model_path, out_h5=temp_h5_path, template_h5=None)
        h5_path = temp_h5_path
        print(f"Converted SAV to temporary HDF5: {h5_path}")
    else:
        h5_path = model_path
    b3d = read_b3d_h5(str(h5_path))
    b3d = normalize_viewer_axis_order(b3d)

    dims = infer_dims(b3d)
    obs_time = infer_time(b3d)
    res = infer_res(b3d)

    frame = Heliocentric(observer=get_earth(obs_time), obstime=obs_time)
    center = SkyCoord(0 * u.Mm, 0 * u.Mm, 0 * u.Mm, frame=frame)
    box = SimpleBox(dims_pix=dims, res=res.to(u.Mm), b3d=b3d, _frame_obs=frame, _center=center)

    if "corona" in b3d:
        b3dtype = "corona"
    elif "nlfff" in b3d:
        b3dtype = "corona"
        b3d["corona"] = b3d.pop("nlfff")
    elif "pot" in b3d:
        b3dtype = "corona"
        b3d["corona"] = b3d.pop("pot")
    elif "chromo" in b3d:
        b3dtype = "chromo"
        chromo = b3d.get("chromo", {})
        if "bx" not in chromo and "bcube" in chromo:
            bcube = chromo["bcube"]
            if bcube.ndim == 4 and bcube.shape[-1] == 3:
                chromo["bx"] = bcube[:, :, :, 0]
                chromo["by"] = bcube[:, :, :, 1]
                chromo["bz"] = bcube[:, :, :, 2]
                b3d["chromo"] = chromo
    else:
        raise ValueError("No known model types found in HDF5 (expected corona/chromo).")

    warnings.filterwarnings("ignore", category=PyVistaDeprecationWarning)
    viewer = MagFieldViewer(box, time=obs_time, b3dtype=b3dtype, parent=None)
    if hasattr(viewer, "app_window"):
        viewer.app_window.setWindowTitle(f"GxBox 3D viewer - {h5_path}")
    viewer.show()
    if owns_app:
        app.exec_()
    # Temporary conversion artifact can be removed after viewer exits.
    if temp_h5_path is not None:
        try:
            temp_h5_path.unlink(missing_ok=True)
            temp_h5_path.parent.rmdir()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
