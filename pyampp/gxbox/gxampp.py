import sys
import os
import select
import re
import shlex
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QComboBox,
                             QRadioButton,
                             QCheckBox, QGridLayout, QGroupBox, QButtonGroup, QVBoxLayout, QHBoxLayout, QDateTimeEdit,
                             QCalendarWidget, QTextEdit, QMessageBox, QDockWidget, QToolButton, QMenu,
                             QFileDialog, QStyle)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QDateTime, Qt, QTimer, QSettings
from datetime import datetime
from PyQt5 import uic

from pyampp.util.config import *
import pyampp
from pathlib import Path
from pyampp.gxbox.boxutils import read_b3d_h5, validate_number
from pyampp.gxbox.gx_fov2box import (
    _load_entry_box_any,
    _entry_stage_from_loaded,
    _extract_execute_paths,
)
from pyampp.util.idl_execute_to_gxfov2box import _parse_idl_call, _build_gx_fov2box_command
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from sunpy.coordinates import get_earth, HeliographicStonyhurst, HeliographicCarrington, Helioprojective
from sunpy.sun import constants as sun_consts
import numpy as np
import typer
import subprocess

app = typer.Typer(help="Launch the PyAmpp application.")

base_dir = Path(pyampp.__file__).parent


class CustomQLineEdit(QLineEdit):
    def setTextL(self, text):
        """
        Sets the text of the QLineEdit and moves the cursor to the beginning.

        :param text: str
            The text to set.
        """
        self.setText(text)
        self.setCursorPosition(0)


class PyAmppGUI(QMainWindow):
    """
    Main application GUI for the Solar Data Model.

    This class creates the main window and sets up the user interface for managing solar data and model configurations.

    Attributes
    ----------
    central_widget : QWidget
        The central widget of the main window.
    main_layout : QVBoxLayout
        The main layout for the central widget.

    Methods
    -------
    initUI():
        Initializes the user interface.
    add_data_repository_section():
        Adds the data repository section to the UI.
    update_sdo_data_dir():
        Updates the SDO data directory path.
    update_gxmodel_dir():
        Updates the GX model directory path.
    update_external_box_dir():
        Updates the external box directory path.
    update_dir(new_path, default_path):
        Updates the specified directory path.
    open_sdo_file_dialog():
        Opens a file dialog for selecting the SDO data directory.
    open_gx_file_dialog():
        Opens a file dialog for selecting the GX model directory.
    open_external_file_dialog():
        Opens a file dialog for selecting the external box directory.
    add_model_configuration_section():
        Adds the model configuration section to the UI.
    add_options_section():
        Adds the options section to the UI.
    add_cmd_display():
        Adds the command display section to the UI.
    add_cmd_buttons():
        Adds command buttons to the UI.
    add_status_log():
        Adds the status log section to the UI.
    update_command_display():
        Updates the command display with the current command.
    update_hpc_state(checked):
        Updates the UI when Helioprojective coordinates are selected.
    update_hgc_state(checked):
        Updates the UI when Heliographic Carrington coordinates are selected.
    get_command():
        Constructs the command based on the current UI settings.
    execute_command():
        Executes the constructed command.
    save_command():
        Saves the current command.
    refresh_command():
        Refreshes the current session.
    clear_command():
        Clears the status log.
    """

    def __init__(self):
        """
        Initializes the PyAmppGUI class.
        """
        super().__init__()
        self._gxbox_proc = None
        self._proc_partial_line = ""
        self.info_only_box = None
        self._last_model_path = None
        self._last_valid_entry_box = ""
        self._entry_stage_detected = None
        self._entry_type_detected = None
        self._hydrating_entry = False
        self._proc_timer = QTimer(self)
        self._proc_timer.setInterval(500)
        self._proc_timer.timeout.connect(self._check_gxbox_process)
        self._settings = QSettings("SUNCAST", "pyAMPP")
        self.model_time_orig = None
        # self.rotate_to_time_button = None
        self.rotate_revert_button = None
        self.coords_center = None
        self.coords_center_orig = None
        self.initUI()

    def _has_entry_box(self) -> bool:
        return bool(self.external_box_edit.text().strip())

    def _stage_to_jump_action(self, stage: str) -> str:
        s = (stage or "").upper()
        return {
            "NONE": "none",
            "POT": "potential",
            "BND": "bounds",
            "NAS": "nlfff",
            "GEN": "lines",
            "CHR": "chromo",
        }.get(s, "none")

    def _set_model_params_enabled(self, enabled: bool) -> None:
        widgets = [
            self.model_time_edit,
            self.coord_x_edit,
            self.coord_y_edit,
            self.hpc_radio_button,
            self.hgc_radio_button,
            self.hgs_radio_button,
            self.proj_cea_radio,
            self.proj_top_radio,
            self.grid_x_edit,
            self.grid_y_edit,
            self.grid_z_edit,
            self.res_edit,
            self.padding_size_edit,
            self.disambig_hmi_radio,
            self.disambig_sfq_radio,
        ]
        for w in widgets:
            w.setEnabled(enabled)

    def initUI(self):
        """
        Sets up the initial user interface for the main window.
        """
        # Main widget and layout
        uic.loadUi(Path(__file__).parent / "UI" / "gxampp.ui", self)
        self.setWindowTitle("GX Automatic Production Pipeline Interface")

        # Adding different sections
        self.add_data_repository_section()
        self.add_model_configuration_section()
        self.add_options_section()
        self.add_cmd_display()
        self.add_cmd_buttons()
        self.add_status_log()
        self.update_coords_center()

        self._sync_pipeline_options()
        self.update_command_display()
        self.show()

    def add_data_repository_section(self):
        layout = self.data_repository_section.layout()
        if layout is not None:
            layout.setColumnStretch(0, 0)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(2, 0)
        self.sdo_data_edit.setText(self._settings.value("paths/data_dir", DOWNLOAD_DIR, type=str))
        self.sdo_data_edit.setMinimumWidth(520)
        self.sdo_data_edit.returnPressed.connect(self.update_sdo_data_dir)
        self.sdo_data_edit.textChanged.connect(self._persist_data_dir)
        self.sdo_browse_button.clicked.connect(self.open_sdo_file_dialog)
        self.sdo_browse_button.setText("")
        self.sdo_browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.sdo_browse_button.setToolTip("Select SDO data repository")
        self.sdo_browse_button.setFixedWidth(28)

        self.gx_model_edit.setText(self._settings.value("paths/gxmodel_dir", GXMODEL_DIR, type=str))
        self.gx_model_edit.setMinimumWidth(520)
        self.gx_model_edit.returnPressed.connect(self.update_gxmodel_dir)
        self.gx_model_edit.textChanged.connect(self._persist_gxmodel_dir)
        self.gx_browse_button.clicked.connect(self.open_gx_file_dialog)
        self.gx_browse_button.setText("")
        self.gx_browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.gx_browse_button.setToolTip("Select GX model repository")
        self.gx_browse_button.setFixedWidth(28)

        notify_email = os.environ.get("PYAMPP_JSOC_NOTIFY_EMAIL", JSOC_NOTIFY_EMAIL)
        self.jsoc_notify_email_edit.setText(notify_email)
        self.jsoc_notify_email_edit.returnPressed.connect(self.update_jsoc_notify_email)

        self.external_box_edit.setMinimumWidth(520)
        self.external_box_edit.returnPressed.connect(self.update_external_box_dir)
        self.external_browse_button.clicked.connect(self.open_external_file_dialog)
        self.external_browse_button.setText("")
        self.external_browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.external_browse_button.setToolTip("Select entry model file (.h5/.sav)")
        self.external_browse_button.setFixedWidth(28)
        self.entry_stage_label = QLabel("Detected Entry Type:")
        self.entry_stage_edit = QLineEdit("N/A")
        self.entry_stage_edit.setReadOnly(True)
        self.entry_stage_edit.setFixedWidth(130)
        self.entry_stage_edit.setToolTip("Detected entry type, e.g. POT.GEN.SAV or NAS.CHR.H5")
        self.continue_radio = QRadioButton("Continue")
        self.rebuild_none_radio = QRadioButton("Rebuild from NONE")
        self.rebuild_obs_radio = QRadioButton("Rebuild from OBS")
        self.continue_radio.setChecked(True)
        self.continue_radio.toggled.connect(self._sync_pipeline_options)
        self.rebuild_none_radio.toggled.connect(self._sync_pipeline_options)
        self.rebuild_obs_radio.toggled.connect(self._sync_pipeline_options)
        self.entry_mode_widget = QWidget()
        mode_layout = QHBoxLayout(self.entry_mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)
        mode_layout.addWidget(self.entry_stage_edit)
        mode_layout.addWidget(self.continue_radio)
        mode_layout.addWidget(self.rebuild_none_radio)
        mode_layout.addWidget(self.rebuild_obs_radio)
        mode_layout.addStretch()
        if layout is not None:
            layout.addWidget(self.entry_stage_label, 4, 0)
            layout.addWidget(self.entry_mode_widget, 4, 1, 1, 2)

    def update_sdo_data_dir(self):
        """
        Updates the SDO data directory path based on the user input.
        """
        new_path = self.sdo_data_edit.text()
        self.update_dir(new_path, DOWNLOAD_DIR, self.sdo_data_edit)
        self._persist_data_dir(self.sdo_data_edit.text())
        self.update_command_display()

    def update_gxmodel_dir(self):
        """
        Updates the GX model directory path based on the user input.
        """
        new_path = self.gx_model_edit.text()
        self.update_dir(new_path, GXMODEL_DIR, self.gx_model_edit)
        self._persist_gxmodel_dir(self.gx_model_edit.text())
        self.update_command_display()

    def _persist_data_dir(self, text):
        self._settings.setValue("paths/data_dir", (text or "").strip())

    def _persist_gxmodel_dir(self, text):
        self._settings.setValue("paths/gxmodel_dir", (text or "").strip())

    def update_jsoc_notify_email(self):
        """
        Updates the JSOC notify email via the PYAMPP_JSOC_NOTIFY_EMAIL environment variable.
        """
        new_email = self.jsoc_notify_email_edit.text().strip()
        if new_email:
            os.environ["PYAMPP_JSOC_NOTIFY_EMAIL"] = new_email
        else:
            os.environ.pop("PYAMPP_JSOC_NOTIFY_EMAIL", None)
            self.jsoc_notify_email_edit.setText(JSOC_NOTIFY_EMAIL)

    def read_external_box(self):
        """
        Reads the external box path based on the user input.
        """
        boxfile = self.external_box_edit.text()
        self._hydrating_entry = True
        try:
            boxdata = _load_entry_box_any(Path(boxfile))
            entry_stage = _entry_stage_from_loaded(boxdata, Path(boxfile))
            self._entry_stage_detected = entry_stage
            entry_type = self._derive_entry_type(boxdata, Path(boxfile), entry_stage)
            self._entry_type_detected = entry_type
            self.entry_stage_edit.setText(entry_type)

            execute_text = self._decode_meta_value(boxdata.get("metadata", {}).get("execute", ""))
            self._apply_execute_defaults(execute_text, boxdata)
            # Canonical time comes from entry model identity (id/path), not stale execute text.
            # This guarantees "Rebuild from OBS" uses the uploaded model timestamp.
            entry_dt = self._infer_entry_datetime(boxdata, Path(boxfile))
            if entry_dt is not None:
                self.model_time_edit.setDateTime(QDateTime(entry_dt))
            exec_data_dir, exec_model_dir = _extract_execute_paths(execute_text)
            warnings = []
            if exec_data_dir:
                p = Path(exec_data_dir).expanduser()
                if p.exists():
                    self.sdo_data_edit.setText(str(p))
                else:
                    self.sdo_data_edit.setText(DOWNLOAD_DIR)
                    warnings.append(f"Invalid execute data-dir on this system, using default: {DOWNLOAD_DIR}")
            if exec_model_dir:
                p = Path(exec_model_dir).expanduser()
                if p.exists():
                    self.gx_model_edit.setText(str(p))
                else:
                    self.gx_model_edit.setText(GXMODEL_DIR)
                    warnings.append(f"Invalid execute gxmodel-dir on this system, using default: {GXMODEL_DIR}")
            self._persist_data_dir(self.sdo_data_edit.text())
            self._persist_gxmodel_dir(self.gx_model_edit.text())

            # Keep command state predictable when importing an entry box.
            self._reset_pipeline_checks_for_entry()
            self._set_jump_action("continue")
            self._last_valid_entry_box = boxfile
            if warnings:
                QMessageBox.warning(self, "Entry Box Path Warnings", "\n".join(warnings))
        finally:
            self._hydrating_entry = False
        self._set_model_params_enabled(False)
        self._sync_pipeline_options()
        self.update_command_display()

    def _derive_entry_type(self, boxdata: dict, entry_path: Path, entry_stage: str) -> str:
        meta_id = self._decode_meta_value(boxdata.get("metadata", {}).get("id", "")).strip()
        stage_path = entry_stage
        if ".CEA." in meta_id:
            stage_path = meta_id.split(".CEA.", 1)[1]
        elif ".TOP." in meta_id:
            stage_path = meta_id.split(".TOP.", 1)[1]
        stage_path = self._infer_stage_path_from_content(boxdata, stage_path)
        suffix = entry_path.suffix.lower()
        file_tag = "SAV" if suffix == ".sav" else "H5"
        return f"{stage_path}.{file_tag}".upper()

    @staticmethod
    def _infer_entry_datetime(boxdata: dict, entry_path: Path):
        """
        Infer observation datetime from metadata id or filename token YYYYMMDD_HHMMSS.
        """
        meta = boxdata.get("metadata", {}) if isinstance(boxdata, dict) else {}
        meta_id = PyAmppGUI._decode_meta_value(meta.get("id", "")).strip()
        candidates = [meta_id, entry_path.stem, entry_path.name]
        for text in candidates:
            m = re.search(r"(\d{8}_\d{6})", str(text))
            if not m:
                continue
            try:
                return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            except Exception:
                continue
        return None

    @staticmethod
    def _has_lines_metadata(boxdata: dict) -> bool:
        required = {"codes", "apex_idx", "start_idx", "end_idx", "seed_idx", "av_field", "phys_length", "voxel_status"}
        lines = boxdata.get("lines")
        if isinstance(lines, dict) and required.issubset(set(lines.keys())):
            return True
        # Backward compatibility: old files stored line arrays under chromo.
        chromo = boxdata.get("chromo")
        if isinstance(chromo, dict) and required.issubset(set(chromo.keys())):
            return True
        return False

    def _infer_stage_path_from_content(self, boxdata: dict, stage_path_hint: str) -> str:
        hint = (stage_path_hint or "").upper()
        corona = boxdata.get("corona", {}) if isinstance(boxdata, dict) else {}
        model_type = ""
        if isinstance(corona, dict):
            attrs = corona.get("attrs", {})
            if isinstance(attrs, dict):
                model_type = str(attrs.get("model_type", "")).strip().lower()
        has_lines = self._has_lines_metadata(boxdata)
        has_chromo = isinstance(boxdata.get("chromo"), dict)

        # First derive POT/NAS branch from data, fallback to hint.
        if model_type == "pot" or hint.startswith("POT"):
            prefix = "POT"
        elif model_type in ("nlfff", "nas") or hint.startswith("NAS"):
            prefix = "NAS"
        elif hint.startswith("BND") or model_type in ("bnd", "bounds"):
            return "BND"
        elif hint.startswith("NONE") or model_type == "none":
            return "NONE"
        else:
            prefix = "NAS"

        # Then derive stage detail from content (not ID text).
        if has_chromo:
            return f"{prefix}.GEN.CHR" if has_lines else f"{prefix}.CHR"
        if has_lines:
            return f"{prefix}.GEN"
        if prefix == "POT":
            return "POT"
        return "NAS"

    @staticmethod
    def _decode_meta_value(v):
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", errors="ignore")
        if hasattr(v, "item"):
            try:
                vv = v.item()
                if isinstance(vv, (bytes, bytearray)):
                    return vv.decode("utf-8", errors="ignore")
                return str(vv)
            except Exception:
                pass
        return str(v or "")

    def _apply_execute_defaults(self, execute_text: str, boxdata: dict):
        """
        Populate GUI fields from entry model metadata/execute.
        """
        parsed_cmd = []
        text = (execute_text or "").strip()
        if text:
            # Native Python execute string stored in metadata.
            if text.startswith("gx-fov2box") or " --" in text:
                try:
                    parsed_cmd = shlex.split(text)
                except Exception:
                    parsed_cmd = []
            # IDL execute string fallback.
            if not parsed_cmd:
                try:
                    _, kw = _parse_idl_call(text)
                    parsed = _build_gx_fov2box_command(kw)
                    parsed_cmd = parsed.command
                except Exception:
                    parsed_cmd = []

        # Parse translated command tokens into a simple option map.
        flag_arity = {
            "--time": 1,
            "--coords": 2,
            "--box-dims": 3,
            "--dx-km": 1,
            "--pad-frac": 1,
            "--data-dir": 1,
            "--gxmodel-dir": 1,
        }
        opts = {}
        i = 0
        while i < len(parsed_cmd):
            tok = parsed_cmd[i]
            if tok in flag_arity:
                n = flag_arity[tok]
                vals = parsed_cmd[i + 1:i + 1 + n]
                if len(vals) == n:
                    opts[tok] = vals
                i += 1 + n
                continue
            opts[tok] = True
            i += 1

        # Fallback extraction for robustness if parser misses.
        if "--time" not in opts:
            m = re.search(r"'(\d{1,2}-[A-Za-z]{3}-\d{2,4}\s+\d{2}:\d{2}:\d{2})'", text)
            if m:
                try:
                    _, kw = _parse_idl_call(f"gx_fov2box, '{m.group(1)}'")
                    parsed = _build_gx_fov2box_command(kw)
                    if "--time" in parsed.command:
                        ti = parsed.command.index("--time")
                        if ti + 1 < len(parsed.command):
                            opts["--time"] = [parsed.command[ti + 1]]
                except Exception:
                    pass
        if "--coords" not in opts:
            m = re.search(r"CENTER_ARCSEC\s*=\s*\[\s*([^\],]+)\s*,\s*([^\]]+)\s*\]", text, flags=re.IGNORECASE)
            if m:
                opts["--coords"] = [m.group(1).strip(), m.group(2).strip()]
                opts["--hpc"] = True
        if "--box-dims" not in opts:
            m = re.search(r"SIZE_PIX\s*=\s*\[\s*([^\],]+)\s*,\s*([^\],]+)\s*,\s*([^\]]+)\s*\]", text, flags=re.IGNORECASE)
            if m:
                opts["--box-dims"] = [m.group(1).strip(), m.group(2).strip(), m.group(3).strip()]

        # Time
        if "--time" in opts:
            try:
                iso = str(opts["--time"][0]).strip()
                dt = datetime.fromisoformat(iso)
                self.model_time_edit.setDateTime(QDateTime(dt))
            except Exception:
                pass
        # Frame first (without firing conversion handlers that could overwrite imported coords).
        if "--hgc" in opts:
            target_frame = "hgc"
        elif "--hgs" in opts:
            target_frame = "hgs"
        else:
            # default to HPC if execute didn't specify.
            target_frame = "hpc"

        for rb in (self.hpc_radio_button, self.hgc_radio_button, self.hgs_radio_button):
            rb.blockSignals(True)
        self.hpc_radio_button.setChecked(target_frame == "hpc")
        self.hgc_radio_button.setChecked(target_frame == "hgc")
        self.hgs_radio_button.setChecked(target_frame == "hgs")
        for rb in (self.hpc_radio_button, self.hgc_radio_button, self.hgs_radio_button):
            rb.blockSignals(False)

        # Coordinates (apply after frame selection so imported values are not overwritten).
        if "--coords" in opts:
            try:
                cx, cy = opts["--coords"]
                self.coord_x_edit.setText(str(cx))
                self.coord_y_edit.setText(str(cy))
            except Exception:
                pass

        # Projection
        if "--top" in opts:
            self.proj_top_radio.setChecked(True)
        else:
            self.proj_cea_radio.setChecked(True)

        # Box dimensions
        if "--box-dims" in opts:
            try:
                nx, ny, nz = opts["--box-dims"]
                self.grid_x_edit.setText(str(nx))
                self.grid_y_edit.setText(str(ny))
                self.grid_z_edit.setText(str(nz))
            except Exception:
                pass
        else:
            corona = boxdata.get("corona", {})
            if isinstance(corona, dict) and "bx" in corona:
                try:
                    nz, ny, nx = corona["bx"].shape
                    self.grid_x_edit.setText(str(nx))
                    self.grid_y_edit.setText(str(ny))
                    self.grid_z_edit.setText(str(nz))
                except Exception:
                    pass

        # Resolution (dx_km)
        if "--dx-km" in opts:
            try:
                self.res_edit.setText(f"{float(opts['--dx-km'][0]):.3f}")
            except Exception:
                pass
        else:
            corona = boxdata.get("corona", {})
            if isinstance(corona, dict) and "dr" in corona:
                try:
                    dr0 = float(corona["dr"][0])
                    rsun_km = sun_consts.radius.to(u.km).value
                    self.res_edit.setText(f"{dr0 * rsun_km:.3f}")
                except Exception:
                    pass

        # Padding fraction
        if "--pad-frac" in opts:
            try:
                pad_frac = float(opts["--pad-frac"][0])
                self.padding_size_edit.setText(f"{pad_frac * 100:.1f}")
            except Exception:
                pass

        # Context map toggles
        self.download_aia_euv.setChecked("--euv" in opts and "--no-euv" not in opts)
        self.download_aia_uv.setChecked("--uv" in opts and "--no-uv" not in opts)

        # Disambiguation
        disambig = self._decode_meta_value(boxdata.get("metadata", {}).get("disambiguation", "")).strip().upper()
        if "--sfq" in opts or disambig == "SFQ":
            self.disambig_sfq_radio.setChecked(True)
        else:
            self.disambig_hmi_radio.setChecked(True)

    def update_external_box_dir(self):
        """
        Updates the external box directory path based on the user input.
        """
        new_path = self.external_box_edit.text()
        if not new_path.strip():
            self._entry_stage_detected = None
            self._entry_type_detected = None
            self.entry_stage_edit.setText("N/A")
            self._set_jump_action("continue")
            self._set_model_params_enabled(True)
            self._sync_pipeline_options()
            self.update_command_display()
            return
        if not os.path.isfile(new_path):
            QMessageBox.critical(self, "Invalid Entry Box", f"Path is not a file:\n{new_path}")
            self._restore_last_valid_entry_box()
            return
        try:
            self.read_external_box()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Entry Box", f"Could not read entry box:\n{exc}")
            self._restore_last_valid_entry_box()
            return
        self.update_command_display()

    def _restore_last_valid_entry_box(self):
        self.external_box_edit.blockSignals(True)
        self.external_box_edit.setText(self._last_valid_entry_box)
        self.external_box_edit.blockSignals(False)
        self.update_command_display()

    def update_dir(self, new_path, default_path, target_edit=None):
        """
        Updates the specified directory path.

        :param new_path: The new directory path.
        :type new_path: str
        :param default_path: The default directory path.
        :type default_path: str
        """
        if new_path != default_path:
            # Normalize the path whether it's absolute or relative
            if not os.path.isabs(new_path):
                new_path = os.path.abspath(new_path)

            if not os.path.exists(new_path):  # Checks if the path does not exist
                # Ask user if they want to create the directory
                reply = QMessageBox.question(self, 'Create Directory?',
                                             "The directory does not exist. Do you want to create it?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                if reply == QMessageBox.Yes:
                    try:
                        os.makedirs(new_path)
                        # QMessageBox.information(self, "Directory Created", "The directory was successfully created.")
                    except PermissionError:
                        QMessageBox.critical(self, "Permission Denied",
                                             "You do not have permission to create this directory.")
                    except OSError as e:
                        QMessageBox.critical(self, "Error", f"Failed to create directory: {str(e)}")
                else:
                    # User chose not to create the directory, revert to the original path
                    if target_edit is not None:
                        target_edit.setText(default_path)
        # else:
        #     QMessageBox.warning(self, "Invalid Path", "The specified path is not a valid absolute path.")

    def open_sdo_file_dialog(self):
        """
        Opens a file dialog for selecting the SDO data directory.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        start_dir = self.sdo_data_edit.text().strip() or DOWNLOAD_DIR
        file_name = QFileDialog.getExistingDirectory(self, "Select Directory", start_dir)
        if file_name:
            self.sdo_data_edit.setText(file_name)
            self.update_sdo_data_dir()

    def open_gx_file_dialog(self):
        """
        Opens a file dialog for selecting the GX model directory.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        start_dir = self.gx_model_edit.text().strip() or GXMODEL_DIR
        file_name = QFileDialog.getExistingDirectory(self, "Select Directory", start_dir)
        if file_name:
            self.gx_model_edit.setText(file_name)
            self.update_gxmodel_dir()

    def open_external_file_dialog(self):
        """
        Opens a file dialog for selecting the external box directory.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File", os.getcwd(), "Model Files (*.h5 *.sav)")
        # file_name = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        if file_name:
            self.external_box_edit.setText(file_name)
            self.update_external_box_dir()

    def add_model_configuration_section(self):
        # Hide legacy jump controls; workflow is linear with optional rebuild.
        self.jump_to_action_combo.setVisible(False)
        self.label_jumpToAction.setVisible(False)
        self.model_time_edit.setDateTime(QDateTime.currentDateTimeUtc())
        self.model_time_edit.setDateTimeRange(QDateTime(2010, 1, 1, 0, 0, 0), QDateTime(QDateTime.currentDateTimeUtc()))
        self.model_time_edit.dateTimeChanged.connect(self.on_time_input_changed)

        self.coord_x_edit.returnPressed.connect(lambda: self.on_coord_x_input_return_pressed(self.coord_x_edit))
        self.coord_y_edit.returnPressed.connect(lambda: self.on_coord_y_input_return_pressed(self.coord_y_edit))

        self.hpc_radio_button.toggled.connect(self.update_hpc_state)
        self.hgc_radio_button.toggled.connect(self.update_hgc_state)
        self.hgs_radio_button.toggled.connect(self.update_hgs_state)

        self.grid_x_edit.returnPressed.connect(lambda: self.on_grid_x_input_return_pressed(self.grid_x_edit))
        self.grid_y_edit.returnPressed.connect(lambda: self.on_grid_y_input_return_pressed(self.grid_y_edit))
        self.grid_z_edit.returnPressed.connect(lambda: self.on_grid_z_input_return_pressed(self.grid_z_edit))
        self.res_edit.returnPressed.connect(lambda: self.on_res_input_return_pressed(self.res_edit))
        self.padding_size_edit.returnPressed.connect(
            lambda: self.on_padding_size_input_return_pressed(self.padding_size_edit))
        self.hpc_radio_button.setText("Heliocentric")
        self.hgc_radio_button.setText("Carrington")
        self.hgs_radio_button.setText("Stonyhurst")

        self.proj_group = QGroupBox("Geometrical Projection")
        self.proj_cea_radio = QRadioButton("CEA")
        self.proj_top_radio = QRadioButton("TOP")
        self.proj_cea_radio.setChecked(True)
        self.proj_button_group = QButtonGroup(self.proj_group)
        self.proj_button_group.addButton(self.proj_cea_radio)
        self.proj_button_group.addButton(self.proj_top_radio)
        proj_layout = QHBoxLayout()
        proj_layout.addWidget(self.proj_cea_radio)
        proj_layout.addWidget(self.proj_top_radio)
        proj_layout.addStretch()
        self.proj_group.setLayout(proj_layout)
        self.proj_cea_radio.toggled.connect(self.update_command_display)
        self.proj_top_radio.toggled.connect(self.update_command_display)

        # Standalone disambiguation group in model configuration (not part of workflow options).
        self.disambig_group = QGroupBox("Pi-disambiguation")
        self.disambig_hmi_radio = QRadioButton("HMI")
        self.disambig_sfq_radio = QRadioButton("SFQ")
        self.disambig_hmi_radio.setChecked(True)
        self.disambig_button_group = QButtonGroup(self.disambig_group)
        self.disambig_button_group.addButton(self.disambig_hmi_radio)
        self.disambig_button_group.addButton(self.disambig_sfq_radio)
        disambig_layout = QHBoxLayout()
        disambig_layout.addWidget(self.disambig_hmi_radio)
        disambig_layout.addWidget(self.disambig_sfq_radio)
        disambig_layout.addStretch()
        self.disambig_group.setLayout(disambig_layout)
        self.disambig_hmi_radio.toggled.connect(self.update_command_display)
        self.disambig_sfq_radio.toggled.connect(self.update_command_display)

        # Keep projection and disambiguation on one row.
        proj_disambig_row = QHBoxLayout()
        proj_disambig_row.addWidget(self.proj_group)
        proj_disambig_row.addWidget(self.disambig_group)
        self.verticalLayout_2.addLayout(proj_disambig_row)

    def _get_jump_action(self):
        if self.rebuild_obs_radio.isChecked():
            return "rebuild_obs"
        if self.rebuild_none_radio.isChecked():
            return "rebuild_none"
        return "continue"

    def _set_jump_action(self, action):
        mode = (action or "continue").lower()
        self.continue_radio.blockSignals(True)
        self.rebuild_none_radio.blockSignals(True)
        self.rebuild_obs_radio.blockSignals(True)
        self.continue_radio.setChecked(mode == "continue")
        self.rebuild_none_radio.setChecked(mode == "rebuild_none")
        self.rebuild_obs_radio.setChecked(mode == "rebuild_obs")
        self.continue_radio.blockSignals(False)
        self.rebuild_none_radio.blockSignals(False)
        self.rebuild_obs_radio.blockSignals(False)

    def _reset_pipeline_checks_for_entry(self):
        boxes = [
            self.download_hmi_box,
            self.download_aia_uv,
            self.download_aia_euv,
            self.stop_early_box,
            self.save_empty_box,
            self.save_potential_box,
            self.save_bounds_box,
            self.save_nas_box,
            self.save_gen_box,
            self.empty_box_only_box,
            self.stop_after_bnd_box,
            self.potential_only_box,
            self.stop_after_potential_box,
            self.nlfff_only_box,
            self.generic_only_box,
            self.add_save_chromo_box,
            self.skip_nlfff_extrapolation,
            self.skip_line_computation_box,
            self.center_vox_box,
        ]
        for b in boxes:
            b.blockSignals(True)
            b.setChecked(False)
            b.blockSignals(False)

    def add_options_section(self):
        """
        Adds the options section to the main layout.
        """
        self.optionsGroupBox.setTitle("Pipeline Workflow")
        self.download_hmi_box = QCheckBox("Download HMI Vector Magnetograms")
        self.download_hmi_box.setChecked(True)
        self.download_hmi_box.setEnabled(False)
        self.stop_early_box = QCheckBox("Stop")
        self.download_aia_euv.setChecked(True)
        self.download_aia_uv.setChecked(True)
        self.save_empty_box.setChecked(False)
        self.save_potential_box.setChecked(False)
        self.save_bounds_box.setChecked(False)
        self.skip_nlfff_extrapolation.setChecked(False)
        self.stop_after_potential_box.setChecked(False)
        self.stop_after_potential_box.setVisible(True)
        self.stop_after_potential_box.setEnabled(True)
        self.stop_after_potential_box.setText("Stop after POT")
        self.skip_nlfff_extrapolation.setText("Skip NLFFF stage")
        self.download_aia_uv.setText("Download AIA/UV")
        self.download_aia_euv.setText("Download AIA/EUV")
        self.save_empty_box.setText("Save Empty Box (NONE)")
        self.save_potential_box.setText("Save Potential Box (POT)")
        self.save_bounds_box.setText("Save Bounds Box (BND)")

        # Additional CLI parity controls (added programmatically to preserve .ui compatibility)
        options_layout = self.optionsGroupBox.layout()
        while options_layout.count():
            item = options_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        self.save_nas_box = QCheckBox("Save NLFFF Box (NAS)")
        self.save_gen_box = QCheckBox("Save Lines (GEN)")
        self.empty_box_only_box = QCheckBox("Stop after NONE")
        self.stop_after_bnd_box = QCheckBox("Stop after BND")
        self.potential_only_box = QCheckBox("Potential Only")
        self.nlfff_only_box = QCheckBox("Stop after NAS")
        self.generic_only_box = QCheckBox("Stop after GEN")
        self.skip_line_computation_box = QCheckBox("Skip Line Computation")
        self.center_vox_box = QCheckBox("Center Box Tracing")
        self.add_save_chromo_box = QCheckBox("Add and save Chromo Model (CHR)")
        self.add_save_chromo_box.setChecked(True)

        # Stage-ordered two-column workflow layout.
        # Two columns, 8 rows each, strict requested order (fill column 1 then column 2).
        options_layout.addWidget(self.download_hmi_box, 0, 0)           # 1
        options_layout.addWidget(self.download_aia_uv, 1, 0)            # 2
        options_layout.addWidget(self.download_aia_euv, 2, 0)           # 3
        options_layout.addWidget(self.stop_early_box, 3, 0)             # 4
        options_layout.addWidget(self.save_empty_box, 4, 0)             # 5
        options_layout.addWidget(self.empty_box_only_box, 5, 0)         # 6
        options_layout.addWidget(self.save_potential_box, 6, 0)         # 7
        options_layout.addWidget(self.save_bounds_box, 7, 0)            # 8
        options_layout.addWidget(self.stop_after_potential_box, 0, 1)   # 9
        options_layout.addWidget(self.skip_nlfff_extrapolation, 1, 1)   # 10
        options_layout.addWidget(self.save_nas_box, 2, 1)               # 11
        options_layout.addWidget(self.nlfff_only_box, 3, 1)             # 12
        options_layout.addWidget(self.skip_line_computation_box, 4, 1)  # 13
        options_layout.addWidget(self.save_gen_box, 5, 1)               # 14
        options_layout.addWidget(self.generic_only_box, 6, 1)           # 15
        options_layout.addWidget(self.add_save_chromo_box, 7, 1)        # 16
        self.center_vox_box.setToolTip("Line tracing mode: Fast is default; enable for center-voxel tracing.")
        self.center_vox_box.setVisible(False)

        # Keep only for CLI backward compatibility.
        self.potential_only_box.setVisible(False)

        # Update command/state when options change
        dynamic_widgets = [
            self.download_aia_euv,
            self.download_aia_uv,
            self.download_hmi_box,
            self.stop_early_box,
            self.save_empty_box,
            self.save_potential_box,
            self.save_bounds_box,
            self.stop_after_potential_box,
            self.skip_nlfff_extrapolation,
            self.save_nas_box,
            self.save_gen_box,
            self.empty_box_only_box,
            self.stop_after_bnd_box,
            self.nlfff_only_box,
            self.generic_only_box,
            self.skip_line_computation_box,
            self.add_save_chromo_box,
        ]
        for w in dynamic_widgets:
            w.toggled.connect(self._sync_pipeline_options)
        self.external_box_edit.textChanged.connect(self.update_command_display)
        self.sdo_data_edit.textChanged.connect(self.update_command_display)
        self.gx_model_edit.textChanged.connect(self.update_command_display)
        self._sync_pipeline_options()

    def _set_checkbox_state(self, box, enabled):
        box.blockSignals(True)
        if not enabled and box.isChecked():
            box.setChecked(False)
        box.setEnabled(enabled)
        box.blockSignals(False)

    def _sync_pipeline_options(self, *_):
        """
        Enforce linear stage workflow:
        - from scratch (no entry): full pipeline options
        - entry selected: forward-only from detected stage
        - rebuild selected: restart from NONE using restored parameters
        """
        try:
            stage_rank = {"NONE": 0, "POT": 1, "BND": 2, "NAS": 3, "GEN": 4, "CHR": 5}
            save_stage_boxes = [
                (self.save_empty_box, 0),
                (self.save_potential_box, 1),
                (self.save_bounds_box, 2),
                (self.save_nas_box, 3),
                (self.save_gen_box, 4),
            ]
            stop_stage_boxes = [
                (self.stop_early_box, -1, None),
                (self.empty_box_only_box, 0, self.save_empty_box),
                (self.stop_after_potential_box, 1, self.save_potential_box),
                (self.nlfff_only_box, 3, self.save_nas_box),
                (self.generic_only_box, 4, self.save_gen_box),
            ]

            has_entry = self._has_entry_box()
            mode = self._get_jump_action()
            rebuild_obs = mode == "rebuild_obs"
            rebuild_none = mode == "rebuild_none"
            start_stage = 0 if (not has_entry or rebuild_obs or rebuild_none) else stage_rank.get(self._entry_stage_detected or "NONE", 0)

            if not has_entry and mode != "continue":
                self._set_jump_action("continue")
                mode = "continue"
                rebuild_obs = False
                rebuild_none = False
            self.rebuild_none_radio.setEnabled(has_entry)
            self.rebuild_obs_radio.setEnabled(has_entry)
            self._set_model_params_enabled((not has_entry) or rebuild_obs)

            # Strict no-going-back for map downloads when resuming from entry box.
            if has_entry and not rebuild_obs:
                for box in (self.download_hmi_box, self.download_aia_uv, self.download_aia_euv):
                    box.blockSignals(True)
                    box.setChecked(False)
                    box.setEnabled(False)
                    box.blockSignals(False)
            else:
                self.download_hmi_box.blockSignals(True)
                self.download_hmi_box.setChecked(True)
                self.download_hmi_box.setEnabled(False)
                self.download_hmi_box.blockSignals(False)
                self.download_aia_uv.setEnabled(True)
                self.download_aia_euv.setEnabled(True)

            # Keep stop controls mutually exclusive in stage order.
            stop_stage = None
            chosen_box = None
            for box, stage, _save in stop_stage_boxes:
                if box.isChecked():
                    stop_stage = stage
                    chosen_box = box
                    break
            if stop_stage is not None:
                for box, _stage, _save in stop_stage_boxes:
                    if box is not chosen_box and box.isChecked():
                        box.blockSignals(True)
                        box.setChecked(False)
                        box.blockSignals(False)

            use_potential = self.skip_nlfff_extrapolation.isChecked()
            skip_lines = self.skip_line_computation_box.isChecked()
            # Allow skip-NLFFF from NONE/POT/BND starts.
            if start_stage > 2:
                self._set_checkbox_state(self.skip_nlfff_extrapolation, False)
                use_potential = False
            else:
                # Skip-NLFFF is only meaningful if pipeline can proceed beyond POT.
                skip_enabled = stop_stage is None or stop_stage >= 2
                self._set_checkbox_state(self.skip_nlfff_extrapolation, skip_enabled)
                use_potential = self.skip_nlfff_extrapolation.isChecked()

            skip_lines_enabled = start_stage <= 4 and (stop_stage is None or stop_stage >= 4)
            self._set_checkbox_state(self.skip_line_computation_box, skip_lines_enabled)
            skip_lines = self.skip_line_computation_box.isChecked()

            for box, stage, _save in stop_stage_boxes:
                enabled = (stage == -1 and start_stage == 0) or (stage >= start_stage)
                if stop_stage is not None and stage > stop_stage:
                    enabled = False
                if use_potential and stage in (2, 3):
                    enabled = False
                if skip_lines and stage == 4:
                    enabled = False
                self._set_checkbox_state(box, enabled)

            for box, stage in save_stage_boxes:
                enabled = stage >= start_stage
                if stop_stage is not None and stage > stop_stage:
                    enabled = False
                if use_potential and stage in (2, 3):
                    enabled = False
                if skip_lines and stage == 4:
                    enabled = False
                # Requested behavior: when stopping after POT, POT save is default
                # while Save BND stays available as an optional bonus.
                if stop_stage == 1 and stage == 2:
                    enabled = True
                self._set_checkbox_state(box, enabled)

            # When a stop is selected, corresponding save is automatic.
            if stop_stage is not None:
                for _stop_box, stage, save_box in stop_stage_boxes:
                    if stage == stop_stage and save_box is not None:
                        save_box.blockSignals(True)
                        save_box.setChecked(True)
                        save_box.setEnabled(False)
                        save_box.blockSignals(False)
                        break

            # CHR control is explicit and final-stage save is implicit.
            chr_enabled = start_stage <= 5 and (stop_stage is None or stop_stage >= 4)
            self._set_checkbox_state(self.add_save_chromo_box, chr_enabled)
            if not chr_enabled:
                self.add_save_chromo_box.blockSignals(True)
                self.add_save_chromo_box.setChecked(False)
                self.add_save_chromo_box.blockSignals(False)
            elif not self.add_save_chromo_box.isChecked():
                # keep unchecked by user; handled as stop-after-gen.
                pass

            # center-vox matters only when lines are computed.
            center_vox_enabled = True
            if start_stage > 4:
                center_vox_enabled = False
            if stop_stage is not None and stop_stage < 4:
                center_vox_enabled = False
            if skip_lines:
                center_vox_enabled = False
            self._set_checkbox_state(self.center_vox_box, center_vox_enabled)
            self.update_command_display()
        except Exception as exc:
            self.status_log_edit.append(f"GUI workflow sync error: {exc}")

    def add_cmd_display(self):
        """
        Adds the command display section to the main layout.
        """
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(11)
        self.cmd_display_edit.setFont(mono)

    def add_cmd_buttons(self):
        """
        Adds the command buttons to the main layout.
        """
        self.info_only_box = QCheckBox("Info Only")
        self.info_only_box.toggled.connect(self.update_command_display)
        # Place utility toggle with execution controls, not pipeline-flow options.
        if self.cmd_button_layout is not None:
            spacer_idx = max(0, self.cmd_button_layout.count() - 1)
            self.cmd_button_layout.insertWidget(spacer_idx, self.info_only_box)
        self.execute_button.clicked.connect(self.execute_command)
        self.stop_button.clicked.connect(self.stop_command)
        self.stop_button.setEnabled(False)
        self.save_button.setVisible(False)
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_command)
        self.send_to_viewer_button = QPushButton("Send to gxbox-view")
        self.send_to_viewer_button.setToolTip("Open latest generated model in gxbox-view")
        self.send_to_viewer_button.setEnabled(False)
        if self.cmd_button_layout is not None:
            spacer_idx = max(0, self.cmd_button_layout.count() - 1)
            self.cmd_button_layout.insertWidget(spacer_idx, self.send_to_viewer_button)
        self.send_to_viewer_button.clicked.connect(self.send_to_gxbox_view)
        self.clear_button_refresh.clicked.connect(self.refresh_command)
        self.clear_button_clear.setVisible(False)
        self.clear_button_clear.setEnabled(False)
        self.clear_button_clear.clicked.connect(self.clear_command)

    def add_status_log(self):
        """
        Adds the status log section to the main layout.
        """
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(10)
        self.status_log_edit.setFont(mono)
        # Move console to a right-side dock panel to keep the main workflow visible.
        for i in range(self.main_layout.count()):
            item = self.main_layout.itemAt(i)
            if item is not None and item.widget() is self.status_log_edit:
                self.main_layout.takeAt(i)
                break
        self.console_dock = QDockWidget("Console", self)
        self.console_dock.setObjectName("consoleDock")
        self.console_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.console_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.console_dock.setMinimumWidth(340)
        self.console_dock.setMaximumWidth(520)
        self.console_dock.setWidget(self.status_log_edit)
        title_bar = QWidget(self.console_dock)
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(6, 2, 6, 2)
        title_layout.setSpacing(4)
        title_label = QLabel("Console", title_bar)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        self.console_menu_button = QToolButton(title_bar)
        self.console_menu_button.setText("⋮")
        self.console_menu_button.setToolTip("Console options")
        self.console_menu_button.setPopupMode(QToolButton.InstantPopup)
        console_menu = QMenu(self.console_menu_button)
        console_menu.addAction("Clear", self.clear_console)
        console_menu.addAction("Copy All", self.copy_console)
        console_menu.addAction("Save As...", self.save_console)
        self.console_menu_button.setMenu(console_menu)
        title_layout.addWidget(self.console_menu_button)
        self.console_dock.setTitleBarWidget(title_bar)
        self.addDockWidget(Qt.RightDockWidgetArea, self.console_dock)

    @validate_number
    def on_coord_x_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_coord_y_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_grid_x_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_grid_y_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_grid_z_input_return_pressed(self, widget):
        self.update_command_display(widget)

    @validate_number
    def on_res_input_return_pressed(self, widget):
        self.update_command_display(widget)

    def on_padding_size_input_return_pressed(self, widget):
        self.update_command_display(widget)

    def _remove_stretch_from_layout(self, layout):
        """
        Removes the last stretch item from the given layout if it exists.

        This method checks the last item in the layout and removes it if it is a spacer item.
        It is useful for dynamically managing layout items, especially when adding or removing widgets.

        Parameters
        ----------
        layout : QLayout
            The layout from which the stretch item should be removed.
        """
        count = layout.count()
        if count > 0 and layout.itemAt(count - 1).spacerItem():
            layout.takeAt(count - 1)

    def on_time_input_changed(self):
        self.coords_center = self._coords_center
        if self.model_time_orig is not None:
            time = Time(self.model_time_edit.dateTime().toPyDateTime()).mjd
            model_time_orig = self.model_time_orig.mjd
            time_sec_diff = (time - model_time_orig) * 24 * 3600
            print(time_sec_diff)
            if np.abs(time_sec_diff) >= 0.5:
                self.on_rotate_model_to_time()
                if self.rotate_revert_button is None:
                    self._remove_stretch_from_layout(self.model_time_layout)
                    self.rotate_revert_button = QPushButton("Revert")
                    self.rotate_revert_button.setToolTip("Revert the model to the original time")
                    self.rotate_revert_button.clicked.connect(self.on_rotate_revert_button_clicked)
                    self.model_time_layout.addWidget(self.rotate_revert_button)
                    self.model_time_layout.addStretch()
            else:
                if self.rotate_revert_button is not None:
                    self.update_coords_center(revert=True)
                    self.rotate_revert()
                    self._remove_stretch_from_layout(self.model_time_layout)
                    self.model_time_layout.removeWidget(self.rotate_revert_button)
                    self.rotate_revert_button.deleteLater()
                    self.rotate_revert_button = None
                    self.model_time_layout.addStretch()
        self.update_command_display()

    def on_rotate_revert_button_clicked(self):
        self.model_time_edit.setDateTime(QDateTime(self.model_time_orig.to_datetime()))
        self.rotate_revert()

    def rotate_revert(self):
        if self.hpc_radio_button.isChecked():
            self.update_hpc_state(True)
        elif self.hgc_radio_button.isChecked():
            self.update_hgc_state(True)
        elif self.hgs_radio_button.isChecked():
            self.update_hgs_state(True)

    def on_rotate_model_to_time(self):
        """
        Rotates the model to the specified time.
        """
        from sunpy.coordinates import RotatedSunFrame
        point = self.coords_center_orig
        time = Time(self.model_time_edit.dateTime().toPyDateTime()).mjd
        model_time_orig = self.model_time_orig.mjd
        time_sec_diff = (time - model_time_orig) * 24 * 3600
        diffrot_point = SkyCoord(RotatedSunFrame(base=point, duration=time_sec_diff * u.s))
        self.coords_center = diffrot_point.transform_to(self._coords_center.frame)
        print(self.coords_center_orig, self.coords_center)
        # self.status_log_edit.append("Model rotated to the specified time")
        if self.hpc_radio_button.isChecked():
            self.update_hpc_state(True, self.coords_center)
        elif self.hgc_radio_button.isChecked():
            self.update_hgc_state(True, self.coords_center)
        elif self.hgs_radio_button.isChecked():
            self.update_hgs_state(True, self.coords_center)
        self.update_command_display()

    def update_command_display(self, widget=None):
        """
        Updates the command display with the current command.
        """
        try:
            if getattr(self, "_hydrating_entry", False):
                return
            self.coords_center = self._coords_center
            command = self.get_command()
            self.cmd_display_edit.setPlainText(" ".join(command))
        except Exception as exc:
            self.status_log_edit.append(f"GUI update error: {exc}")

    def update_hpc_state(self, checked, coords_center=None):
        """
        Updates the UI when Helioprojective coordinates are selected.

        :param checked: Whether the Helioprojective radio button is checked.
        :type checked: bool
        """
        if checked:
            self.coord_x_edit.setToolTip("Solar X coordinate of the model center in arcsec")
            self.coord_y_edit.setToolTip("Solar Y coordinate of the model center in arcsec")
            self.coord_label.setText("Center Coords  in arcsec")
            self.coord_x_label.setText("X:")
            self.coord_y_label.setText("Y:")
            if coords_center is None:
                obstime = Time(self.model_time_edit.dateTime().toPyDateTime())
                observer = get_earth(obstime)
                coords_center = self.coords_center.transform_to(Helioprojective(obstime=obstime, observer=observer))
            self.coord_x_edit.setText(f'{coords_center.Tx.to(u.arcsec).value}')
            self.coord_y_edit.setText(f'{coords_center.Ty.to(u.arcsec).value}')
            self.update_command_display()

    def update_hgc_state(self, checked, coords_center=None):
        """
        Updates the UI when Heliographic Carrington coordinates are selected.

        :param checked: Whether the Heliographic Carrington radio button is checked.
        :type checked: bool
        """
        if checked:
            self.coord_x_edit.setToolTip("Heliographic Carrington Longitude of the model center in deg")
            self.coord_y_edit.setToolTip("Heliographic Carrington Latitude of the model center in deg")
            self.coord_label.setText("Center Coords in deg")
            self.coord_x_label.setText("lon:")
            self.coord_y_label.setText("lat:")
            if coords_center is None:
                print(f'coords_center: {self.coords_center}')
                obstime = Time(self.model_time_edit.dateTime().toPyDateTime())
                observer = get_earth(obstime)
                coords_center = self.coords_center.transform_to(
                    HeliographicCarrington(obstime=obstime, observer=observer))
            print(f'new coords_center: {coords_center}')
            self.coord_x_edit.setText(f'{coords_center.lon.to(u.deg).value}')
            self.coord_y_edit.setText(f'{coords_center.lat.to(u.deg).value}')
            self.update_command_display()

    def update_hgs_state(self, checked, coords_center=None):
        """
        Updates the UI when Heliographic Stonyhurst coordinates are selected.

        :param checked: Whether the Heliographic Stonyhurst radio button is checked.
        :type checked: bool
        """
        if checked:
            self.coord_x_edit.setToolTip("Heliographic Stonyhurst Longitude of the model center in deg")
            self.coord_y_edit.setToolTip("Heliographic Stonyhurst Latitude of the model center in deg")
            self.coord_label.setText("Center Coords in deg")
            self.coord_x_label.setText("lon:")
            self.coord_y_label.setText("lat:")
            if coords_center is None:
                obstime = Time(self.model_time_edit.dateTime().toPyDateTime())
                # observer = get_earth(obstime)
                coords_center = self.coords_center.transform_to(
                    HeliographicStonyhurst(obstime=obstime))
            self.coord_x_edit.setText(f'{coords_center.lon.to(u.deg).value}')
            self.coord_y_edit.setText(f'{coords_center.lat.to(u.deg).value}')
            self.update_command_display()

    def update_coords_center(self, revert=False):
        if revert:
            self.coords_center = self.coords_center_orig
        else:
            self.coords_center = self._coords_center

    @property
    def _coords_center(self):
        time = Time(self.model_time_edit.dateTime().toPyDateTime())
        coords = [float(self.coord_x_edit.text()), float(self.coord_y_edit.text())]
        observer = get_earth(time)
        if self.hpc_radio_button.isChecked():
            coords_center = SkyCoord(coords[0] * u.arcsec, coords[1] * u.arcsec, obstime=time, observer=observer,
                                     rsun=696 * u.Mm, frame='helioprojective')
        elif self.hgc_radio_button.isChecked():
            coords_center = SkyCoord(lon=coords[0] * u.deg, lat=coords[1] * u.deg, obstime=time, observer=observer,
                                     radius=696 * u.Mm,
                                     frame='heliographic_carrington')
        elif self.hgs_radio_button.isChecked():
            coords_center = SkyCoord(lon=coords[0] * u.deg, lat=coords[1] * u.deg, obstime=time, observer=observer,
                                     radius=696 * u.Mm,
                                     frame='heliographic_stonyhurst')
        return coords_center

    def get_command(self):
        """
        Constructs the command based on the current UI settings.

        Returns
        -------
        list
            The command as a list of strings.
        """
        import astropy.time
        import astropy.units as u

        command = ['gx-fov2box']
        has_entry = self._has_entry_box()
        jump_action = self._get_jump_action()

        # In entry continue/rebuild-none modes keep CLI minimal: entry-box + restored/selected paths + workflow flags.
        if has_entry and jump_action != "rebuild_obs":
            command += ['--data-dir', self.sdo_data_edit.text()]
            command += ['--gxmodel-dir', self.gx_model_edit.text()]
            command += ['--entry-box', self.external_box_edit.text()]
        else:
            time = astropy.time.Time(self.model_time_edit.dateTime().toPyDateTime())
            command += ['--time', time.to_datetime().strftime('%Y-%m-%dT%H:%M:%S')]
            command += ['--coords', self.coord_x_edit.text(), self.coord_y_edit.text()]
            if self.hpc_radio_button.isChecked():
                command += ['--hpc']
            elif self.hgc_radio_button.isChecked():
                command += ['--hgc']
            else:
                command += ['--hgs']
            if self.proj_top_radio.isChecked():
                command += ['--top']
            else:
                command += ['--cea']

            command += ['--box-dims', self.grid_x_edit.text(), self.grid_y_edit.text(), self.grid_z_edit.text()]
            command += ['--dx-km', f'{float(self.res_edit.text()):.3f}']
            command += ['--pad-frac', f'{float(self.padding_size_edit.text()) / 100:.2f}']
            command += ['--data-dir', self.sdo_data_edit.text()]
            command += ['--gxmodel-dir', self.gx_model_edit.text()]
            if has_entry:
                command += ['--entry-box', self.external_box_edit.text()]

        if self.download_aia_euv.isChecked():
            command += ['--euv']
        if self.download_aia_uv.isChecked():
            command += ['--uv']

        if self.save_empty_box.isChecked():
            command += ['--save-empty-box']
        if self.save_potential_box.isChecked():
            command += ['--save-potential']
        if self.save_bounds_box.isChecked():
            command += ['--save-bounds']
        if self.save_nas_box.isChecked():
            command += ['--save-nas']
        if self.save_gen_box.isChecked():
            command += ['--save-gen']

        if self.stop_early_box.isChecked():
            command += ['--stop-after', 'dl']
        elif self.empty_box_only_box.isChecked():
            command += ['--stop-after', 'none']
        elif self.stop_after_potential_box.isChecked():
            command += ['--stop-after', 'pot']
        elif self.nlfff_only_box.isChecked():
            command += ['--stop-after', 'nas']
        elif self.generic_only_box.isChecked() or not self.add_save_chromo_box.isChecked():
            command += ['--stop-after', 'gen']

        if self.skip_nlfff_extrapolation.isChecked():
            command += ['--use-potential']
        if self.skip_line_computation_box.isChecked():
            command += ['--skip-lines']
        if self.center_vox_box.isChecked():
            command += ['--center-vox']

        if jump_action == 'rebuild_obs':
            command += ['--rebuild']
        elif jump_action == 'rebuild_none':
            command += ['--rebuild-from-none']

        # Disambiguation affects rebuild/new-box computation only.
        if (not has_entry or jump_action == "rebuild_obs") and self.disambig_sfq_radio.isChecked():
            command += ['--sfq']

        if self.info_only_box is not None and self.info_only_box.isChecked():
            command += ['--info']

        return command

    def execute_command(self):
        """
        Executes the constructed command.
        """
        if self._gxbox_proc is not None and self._gxbox_proc.poll() is None:
            QMessageBox.warning(self, "GXbox Running", "A GXbox process is already running.")
            return

        command = self.get_command()
        self._last_model_path = None
        self.send_to_viewer_button.setEnabled(False)
        try:
            self._gxbox_proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            self._proc_partial_line = ""
            if self._gxbox_proc.stdout is not None:
                os.set_blocking(self._gxbox_proc.stdout.fileno(), False)
            self.execute_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_log_edit.append("Command started: " + " ".join(command))
            self._proc_timer.start()
        except Exception as e:
            QMessageBox.critical(self, "Execution Error", f"Failed to start command: {e}")
            self.status_log_edit.append("Command failed to start")

    def _drain_process_output(self):
        if self._gxbox_proc is None or self._gxbox_proc.stdout is None:
            return
        stdout = self._gxbox_proc.stdout
        fd = stdout.fileno()
        chunks = []
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            chunk = stdout.read()
            if not chunk:
                break
            chunks.append(chunk)
        if not chunks:
            return
        text = self._proc_partial_line + "".join(chunks).replace("\r\n", "\n").replace("\r", "\n")
        if text.endswith("\n"):
            complete_lines = text.split("\n")[:-1]
            self._proc_partial_line = ""
        else:
            parts = text.split("\n")
            complete_lines = parts[:-1]
            self._proc_partial_line = parts[-1]
        for line in complete_lines:
            if line.strip():
                self.status_log_edit.append(line)

    def stop_command(self):
        """
        Stops the running GXbox process if any.
        """
        if self._gxbox_proc is None or self._gxbox_proc.poll() is not None:
            self.status_log_edit.append("No running command to stop")
            self.stop_button.setEnabled(False)
            self.execute_button.setEnabled(True)
            return

        self.status_log_edit.append("Stopping command...")
        try:
            self._gxbox_proc.terminate()
            self._gxbox_proc.wait(timeout=5)
            self._drain_process_output()
            self.status_log_edit.append("Command stopped")
        except subprocess.TimeoutExpired:
            self._gxbox_proc.kill()
            self._drain_process_output()
            self.status_log_edit.append("Command killed")
        finally:
            self._gxbox_proc = None
            self.stop_button.setEnabled(False)
            self.execute_button.setEnabled(True)
            self._proc_timer.stop()

    def _check_gxbox_process(self):
        try:
            if self._gxbox_proc is None:
                self._proc_timer.stop()
                return

            self._drain_process_output()
            if self._gxbox_proc.poll() is None:
                return

            self._drain_process_output()
            if self._proc_partial_line.strip():
                self.status_log_edit.append(self._proc_partial_line)
                self._proc_partial_line = ""
            exit_code = self._gxbox_proc.returncode
            if exit_code == 0:
                self.status_log_edit.append("Command finished successfully")
                self._update_last_model_path()
            else:
                self.status_log_edit.append(f"Command exited with code {exit_code}")
        except Exception as exc:
            self.status_log_edit.append(f"GUI process-monitor error: {exc}")
        finally:
            if self._gxbox_proc is not None and self._gxbox_proc.poll() is not None:
                self._gxbox_proc = None
                self.stop_button.setEnabled(False)
                self.execute_button.setEnabled(True)
                self._proc_timer.stop()

    def save_command(self):
        """
        Saves the current command.
        """
        # Placeholder for saving command
        self.status_log_edit.append("Command saved")

    def refresh_command(self):
        """
        Refreshes the current session.
        """
        # Placeholder for refreshing command
        self.status_log_edit.append("Command refreshed")

    def clear_command(self):
        """
        Clears the status log.
        """
        # Placeholder for clearing command
        self.status_log_edit.clear()

    def clear_console(self):
        """
        Clears the console panel.
        """
        self.status_log_edit.clear()
        self._last_model_path = None
        self.send_to_viewer_button.setEnabled(False)

    def copy_console(self):
        """
        Copies the full console text to clipboard.
        """
        QApplication.clipboard().setText(self.status_log_edit.toPlainText())

    def save_console(self):
        """
        Saves the console output to a text file.
        """
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Console Output",
            str(Path.cwd() / "pyampp_console.txt"),
            "Text Files (*.txt);;Log Files (*.log);;All Files (*)",
        )
        if not file_name:
            return
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(self.status_log_edit.toPlainText())
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not save console output:\n{exc}")

    def _update_last_model_path(self):
        text = self.status_log_edit.toPlainText()
        candidates = re.findall(r"^- (.+\.h5)\s*$", text, flags=re.MULTILINE)
        for raw in reversed(candidates):
            p = Path(raw).expanduser()
            if p.exists():
                self._last_model_path = str(p)
                self.send_to_viewer_button.setEnabled(True)
                return
        root = Path(self.gx_model_edit.text()).expanduser()
        if not root.exists():
            return
        newest = None
        newest_mtime = -1.0
        for p in root.rglob("*.h5"):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest = p
        if newest is not None:
            self._last_model_path = str(newest)
            self.send_to_viewer_button.setEnabled(True)

    def send_to_gxbox_view(self):
        if not self._last_model_path:
            QMessageBox.information(self, "No Model", "No generated model was found to send.")
            return
        model_path = Path(self._last_model_path).expanduser()
        if not model_path.exists():
            QMessageBox.warning(self, "Missing Model", f"Model file not found:\n{model_path}")
            return
        try:
            start_dir = model_path.parent
            subprocess.Popen([
                "gxbox-view",
                "--pick",
                "--dir",
                str(start_dir),
                "--h5",
                str(model_path),
            ])
            self.status_log_edit.append(f"Launched gxbox-view with: {model_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Launch Failed", f"Could not launch gxbox-view:\n{exc}")


@app.command()
def main(
        debug: bool = typer.Option(
            False,
            "--debug",
            help="Enable debug mode with an interactive IPython session."
        )
):
    """
    Entry point for the PyAmppGUI application.

    This function initializes the PyQt application, sets up and displays the main
    GUI window for the Solar Data Model. It pre-configures some GUI elements with default
    values for model time and coordinates. Default values are set programmatically
    before the event loop starts.

    :param debug: Enable debug mode with an interactive IPython session, defaults to False
    :type debug: bool, optional
    :raises SystemExit: Exits the application loop when the GUI is closed
    :return: None
    :rtype: NoneType

    Examples
    --------
    .. code-block:: bash

        pyampp
    """

    app_qt = QApplication([])
    pyampp = PyAmppGUI()
    pyampp.model_time_edit.setDateTime(QDateTime(2024, 5, 12, 0, 0))
    pyampp.coord_x_edit.setText('0')
    pyampp.coord_y_edit.setText('0')
    pyampp.update_coords_center()
    pyampp.update_command_display()

    if debug:
        # Start an interactive IPython session for debugging
        import IPython
        IPython.embed()

        # If any matplotlib plots are created, show them
        import matplotlib.pyplot as plt
        plt.show()
    sys.exit(app_qt.exec_())

if __name__ == '__main__':
    app()
