# house_price_gui_ny_with_charts.py
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# Optional background image
try:
    from PIL import Image, ImageTk  # pip install pillow
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ML stack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Matplotlib for charts (embedded inside Tk)
import matplotlib
matplotlib.use("TkAgg")  # important for Tkinter backends
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# CONFIG — UPDATE THESE
# =========================
CSV_PATH = r"C:\Users\allur\OneDrive\Documents\Job Applications 2025\Github_Project\NY-House-Dataset.csv"
BG_IMAGE_PATH = None  # e.g., r"C:\path\to\HouseBG.jpg" or leave as None

TARGET = "PRICE"
NUMERIC_COLS = ["BEDS", "BATH", "PROPERTYSQFT", "LATITUDE", "LONGITUDE"]
CATEGORICAL_COLS = ["TYPE", "STATE", "ZIPCODE"]
ZIP_REGEX = re.compile(r"(\d{5})(?:[- ]\d{4})?$")
RANDOM_STATE = 40

# ============== Data utils ==============
def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at: {p}")
    df = pd.read_csv(p, low_memory=False, encoding="utf-8", encoding_errors="replace")
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Columns: {list(df.columns)}")
    return df

def extract_zipcode(df: pd.DataFrame) -> pd.DataFrame:
    z = pd.Series([np.nan] * len(df), index=df.index, dtype="object")
    for col in ["MAIN_ADDRESS", "ADDRESS", "FORMATTED_ADDRESS"]:
        if col in df.columns:
            z = z.fillna(df[col].astype(str).str.extract(ZIP_REGEX, expand=False))
    df["ZIPCODE"] = z.astype("string")
    return df

def build_working_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [TARGET] + [c for c in NUMERIC_COLS if c in df.columns] + [c for c in CATEGORICAL_COLS if c in df.columns]
    work = df[cols].copy()
    # coerce numeric
    for c in [TARGET] + [k for k in NUMERIC_COLS if k in work.columns]:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[TARGET]).reset_index(drop=True)
    work = work[(work[TARGET] >= 5_000) & (work[TARGET] <= 300_000_000)].reset_index(drop=True)
    return work

# ============== Training & evaluation ==============
def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in NUMERIC_COLS if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", MinMaxScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre

def train_three_models(work: pd.DataFrame):
    X = work.drop(columns=[TARGET])
    y = work[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

    pre = get_preprocessor(X_train)

    candidates = [
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ]

    results = []
    best_name, best_pipe, best_rmse = None, None, float("inf")
    y_pred_best = None

    for name, model in candidates:
        pipe = Pipeline([("prep", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        pred_val = pipe.predict(X_valid)
        rmse_val = float(np.sqrt(mean_squared_error(y_valid, pred_val)))
        results.append({"model": name, "rmse_val": rmse_val})

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_name = name
            best_pipe = pipe

    # Evaluate best on test
    y_pred_best = best_pipe.predict(X_test)
    rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_best)))

    metrics = {
        "best_model": best_name,
        "validation": results,
        "test": {"rmse_test": rmse_test}
    }
    return best_pipe, metrics, (X_test, y_test, y_pred_best)

# ============== Chart creation ==============
def fig_model_bar(results):
    names = [d["model"] for d in results]
    rmses = [d["rmse_val"] for d in results]
    fig = plt.figure(figsize=(6, 4))
    plt.bar(names, rmses)
    plt.ylabel("Validation RMSE (lower is better)")
    plt.title("Model Comparison")
    plt.tight_layout()
    return fig

def fig_residuals(y_true, y_pred):
    resid = y_true - y_pred
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, resid, s=10)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (y - ŷ)")
    plt.title("Residuals (Best Model, Test Set)")
    plt.tight_layout()
    return fig

def fig_pred_vs_actual(y_true, y_pred):
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, s=10)
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual (Test Set)")
    plt.tight_layout()
    return fig

def save_fig(fig: plt.Figure, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)

# ============== Tkinter App ==============
class HousePriceApp:
    def __init__(self, master: tk.Tk, pipeline: Pipeline, metrics: dict,
                 test_pack: tuple[pd.DataFrame, pd.Series, np.ndarray],
                 defaults: dict | None = None, bg_path: str | None = None):
        self.master = master
        self.pipeline = pipeline
        self.metrics = metrics
        self.X_test, self.y_test, self.y_pred_best = test_pack
        master.title("House Price Prediction (NY)")

        # Canvas-based full-window background
        self.canvas = tk.Canvas(master, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.bg_img_orig = None
        self.bg_img_tk = None
        self.bg_img_id = None

        if bg_path and PIL_AVAILABLE:
            try:
                self.bg_img_orig = Image.open(bg_path).convert("RGB")
            except Exception as e:
                print("Background image failed to load:", e)

        self.frm = ttk.Frame(self.canvas, padding=12)
        self.frm_id = self.canvas.create_window(16, 16, anchor="nw", window=self.frm)

        ttk.Label(self.frm, text="Enter feature values (leave empty if unknown)", foreground="gray").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        self.entries = {}
        row = 1

        # Numeric inputs
        for field in NUMERIC_COLS:
            ttk.Label(self.frm, text=field).grid(row=row, column=0, padx=(0, 10), pady=6, sticky="e")
            e = ttk.Entry(self.frm, width=28)
            e.grid(row=row, column=1, pady=6, sticky="w")
            if defaults and field in defaults and pd.notna(defaults[field]):
                e.insert(0, str(defaults[field]))
            self.entries[field] = e
            row += 1

        # Categorical inputs
        for field in CATEGORICAL_COLS:
            ttk.Label(self.frm, text=field).grid(row=row, column=0, padx=(0, 10), pady=6, sticky="e")
            e = ttk.Entry(self.frm, width=28)
            e.grid(row=row, column=1, pady=6, sticky="w")
            if defaults and field in defaults and pd.notna(defaults[field]):
                e.insert(0, str(defaults[field]))
            self.entries[field] = e
            row += 1

        # Buttons & result
        self.result_var = tk.StringVar(value="Prediction will appear here")
        ttk.Button(self.frm, text="Predict", command=self.on_predict).grid(row=row, column=0, pady=(10, 0), sticky="e")
        ttk.Label(self.frm, textvariable=self.result_var, font=("Segoe UI", 11, "bold")).grid(
            row=row, column=1, pady=(10, 0), sticky="w"
        )
        row += 1

        # Charts button
        ttk.Button(self.frm, text="Show Charts", command=self.show_charts).grid(row=row, column=0, columnspan=2, pady=(10, 0))

        master.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        self.canvas.config(width=event.width, height=event.height)
        self.canvas.coords(self.frm_id, 16, 16)
        if self.bg_img_orig is not None:
            win_w, win_h = max(event.width, 1), max(event.height, 1)
            img_w, img_h = self.bg_img_orig.size
            scale = max(win_w / img_w, win_h / img_h)  # cover
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            img_resized = self.bg_img_orig.resize((new_w, new_h), Image.LANCZOS)
            offset_x = (win_w - new_w) // 2
            offset_y = (win_h - new_h) // 2
            self.bg_img_tk = ImageTk.PhotoImage(img_resized)
            if self.bg_img_id is None:
                self.bg_img_id = self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.bg_img_tk)
            else:
                self.canvas.itemconfig(self.bg_img_id, image=self.bg_img_tk)
                self.canvas.coords(self.bg_img_id, offset_x, offset_y)

    def on_predict(self):
        try:
            payload = {}
            for f in NUMERIC_COLS:
                val = self.entries[f].get().strip()
                payload[f] = float(val) if val not in ("", None) else None
            for f in CATEGORICAL_COLS:
                val = self.entries[f].get().strip()
                payload[f] = val if val != "" else None

            Xnew = pd.DataFrame([payload])
            yhat = float(self.pipeline.predict(Xnew)[0])
            self.result_var.set(f"Predicted Price: ${yhat:,.0f}")

            # Optional: (re)render charts right after prediction if you prefer
            # self.show_charts()
        except ValueError as ve:
            messagebox.showerror("Invalid input", f"Please check your inputs.\n{ve}")
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))

    def show_charts(self):
        """Render a window with three charts and also save them as PNGs."""
        # Build figures
        bar = fig_model_bar(self.metrics["validation"])
        resid = fig_residuals(self.y_test, self.y_pred_best)
        pva = fig_pred_vs_actual(self.y_test, self.y_pred_best)

        # Save to PNGs next to this script
        base = Path(__file__).with_suffix("")
        save_fig(bar,   str(base) + "_model_scores_bar.png")
        save_fig(resid, str(base) + "_residuals_best.png")
        save_fig(pva,   str(base) + "_pred_vs_actual.png")

        # Recreate for display (since we closed them when saving)
        bar = fig_model_bar(self.metrics["validation"])
        resid = fig_residuals(self.y_test, self.y_pred_best)
        pva = fig_pred_vs_actual(self.y_test, self.y_pred_best)

        # Tk window with embedded canvases
        win = tk.Toplevel(self.master)
        win.title("Model Performance Charts")
        nb = ttk.Notebook(win)
        nb.pack(fill="both", expand=True)

        def add_tab(fig: plt.Figure, title: str):
            frame = ttk.Frame(nb)
            nb.add(frame, text=title)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        add_tab(bar, "RMSE Comparison")
        add_tab(resid, "Residuals (Best)")
        add_tab(pva, "Predicted vs Actual")

# ============== App bootstrap ==============
def main():
    # Prepare data
    df = load_dataset(CSV_PATH)
    df = extract_zipcode(df)
    work = build_working_df(df)

    # Train
    pipe, metrics, test_pack = train_three_models(work)
    print("Validation metrics:", metrics["validation"])
    print("Best model:", metrics["best_model"], "| Test RMSE:", f'{metrics["test"]["rmse_test"]:.2f}')

    # Defaults for UI
    defaults = {}
    for c in NUMERIC_COLS:
        if c in work.columns:
            defaults[c] = work[c].median(skipna=True)
    for c in CATEGORICAL_COLS:
        if c in work.columns:
            m = work[c].mode(dropna=True)
            defaults[c] = m.iloc[0] if not m.empty else ""

    # Launch GUI
    root = tk.Tk()
    _ = HousePriceApp(root, pipe, metrics, test_pack, defaults=defaults, bg_path=BG_IMAGE_PATH)
    root.mainloop()

if __name__ == "__main__":
    main()
