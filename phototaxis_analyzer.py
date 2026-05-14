"""
Phototaxis Analyzer
====================
Auswertungsprogramm für KIN-Dateien aus Phototaxis-Experimenten.
Vergleicht mehrere Proben anhand von Up/Down-Verhältnis, Zeitverlauf,
Kreisdiagrammen und statistischen Tests.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.facecolor"] = "white"
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Farben & Stil ─────────────────────────────────────────────────────────────
COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#8BC34A", "#E91E63", "#607D8B"
]
BG      = "#F5F7FA"
SIDEBAR = "#1E2A3A"
ACCENT  = "#2196F3"
TEXT_LIGHT = "#ECEFF1"
TEXT_DARK  = "#263238"

# ── KIN Parser ────────────────────────────────────────────────────────────────

def parse_kin(filepath):
    """Liest eine KIN-Datei und gibt (meta, df) zurück."""
    with open(filepath, encoding="iso-8859-1") as f:
        lines = f.readlines()

    # Zeile 0: Einstellungsname, Zeile 1: Werte, Zeile 2: Meta (Zeit, Datum)
    meta_line = lines[2].strip().split(",")
    meta = {
        "time"    : meta_line[2].strip() if len(meta_line) > 2 else "",
        "date"    : meta_line[3].strip() if len(meta_line) > 3 else "",
        "filepath": filepath,
        "filename": os.path.basename(filepath),
    }

    # Zeile 3: Spaltenköpfe
    col_names = lines[3].strip().split(",")
    col_names = [c.strip() for c in col_names]

    # Daten ab Zeile 4
    data_str = "".join(lines[4:])
    df = pd.read_csv(io.StringIO(data_str), header=None, names=col_names)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=["time [s]"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return meta, df


def direction_col(df):
    """Gibt den Namen der Direction-Spalte zurück (robust)."""
    for c in df.columns:
        if "direction" in c.lower():
            return c
    return None


def up_down_ratio(df):
    """Mittleres Up/Down-Verhältnis (direction [%]) über die Zeit."""
    col = direction_col(df)
    if col is None:
        return np.nan
    return df[col].mean()


# ── Probe (Sample) ────────────────────────────────────────────────────────────

class Sample:
    def __init__(self, name, color):
        self.name  = name
        self.color = color
        self.files = []   # list of (meta, df)

    def add_file(self, filepath):
        meta, df = parse_kin(filepath)
        self.files.append((meta, df))

    def mean_direction_series(self):
        """Gibt mittlere direction-Zeitreihe (interpoliert auf gemeinsames Grid)."""
        if not self.files:
            return None, None
        all_series = []
        for _, df in self.files:
            col = direction_col(df)
            if col is None:
                continue
            all_series.append(df[["time [s]", col]].copy())
        if not all_series:
            return None, None
        # Gemeinsames Zeitgitter (0–120 s, 1 s Schritte)
        t_grid = np.arange(0, 121, 0.5)
        interp_vals = []
        for s in all_series:
            t = s["time [s]"].values
            v = s.iloc[:, 1].values
            interp = np.interp(t_grid, t, v, left=np.nan, right=np.nan)
            interp_vals.append(interp)
        mat = np.array(interp_vals)
        mean = np.nanmean(mat, axis=0)
        sem  = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
        return t_grid, mean, sem

    def all_direction_values(self):
        """Alle direction-Werte als flaches Array (für Statistik)."""
        vals = []
        for _, df in self.files:
            col = direction_col(df)
            if col is not None:
                vals.extend(df[col].dropna().tolist())
        return np.array(vals)

    def mean_direction(self):
        v = self.all_direction_values()
        return np.nanmean(v) if len(v) else np.nan

    def sem_direction(self):
        v = self.all_direction_values()
        return stats.sem(v, nan_policy="omit") if len(v) > 1 else 0.0

    def polar_data(self):
        """Theta- und r-Werte aller Dateien zusammen (für Kreisdiagramm)."""
        thetas, rs = [], []
        for _, df in self.files:
            if "theta [°]" in df.columns and "r-value" in df.columns:
                thetas.extend(df["theta [°]"].dropna().tolist())
                rs.extend(df["r-value"].dropna().tolist())
        return np.array(thetas), np.array(rs)


# ── Haupt-App ─────────────────────────────────────────────────────────────────

class PhototaxisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phototaxis Analyzer")
        self.geometry("1400x860")
        self.configure(bg=BG)
        self.minsize(1100, 700)

        self.samples: list[Sample] = []
        self._color_idx = 0
        self._build_ui()

    # ── UI Aufbau ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── linke Sidebar ──
        self.sidebar = tk.Frame(self, bg=SIDEBAR, width=280)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        tk.Label(self.sidebar, text="🔬 Phototaxis Analyzer",
                 bg=SIDEBAR, fg=TEXT_LIGHT,
                 font=("Helvetica", 13, "bold")).pack(pady=(18, 4), padx=10)
        tk.Label(self.sidebar, text="Proben & Dateien verwalten",
                 bg=SIDEBAR, fg="#90A4AE", font=("Helvetica", 9)).pack(pady=(0, 14))

        # Proben-Buttons
        btn_frame = tk.Frame(self.sidebar, bg=SIDEBAR)
        btn_frame.pack(fill="x", padx=10, pady=4)
        self._btn(btn_frame, "＋ Neue Probe", self._add_sample).pack(side="left", fill="x", expand=True)
        self._btn(btn_frame, "🗑 Probe löschen", self._delete_sample, danger=True).pack(side="left", fill="x", expand=True, padx=(4, 0))

        # Probenliste
        self.sample_listbox = tk.Listbox(
            self.sidebar, bg="#263238", fg=TEXT_LIGHT,
            selectbackground=ACCENT, selectforeground="white",
            font=("Helvetica", 10), bd=0, highlightthickness=0,
            activestyle="none", height=8
        )
        self.sample_listbox.pack(fill="x", padx=10, pady=4)
        self.sample_listbox.bind("<<ListboxSelect>>", self._on_sample_select)

        ttk.Separator(self.sidebar, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # Datei-Buttons
        self._btn(self.sidebar, "📂 KIN-Dateien hinzufügen", self._add_files).pack(fill="x", padx=10, pady=2)
        self._btn(self.sidebar, "❌ Datei entfernen", self._remove_file).pack(fill="x", padx=10, pady=2)

        # Dateiliste
        self.file_listbox = tk.Listbox(
            self.sidebar, bg="#263238", fg="#B0BEC5",
            selectbackground="#37474F", font=("Courier", 8),
            bd=0, highlightthickness=0, height=10
        )
        self.file_listbox.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        ttk.Separator(self.sidebar, orient="horizontal").pack(fill="x", padx=10, pady=4)

        # Auswertungs-Buttons
        tk.Label(self.sidebar, text="AUSWERTUNG", bg=SIDEBAR, fg="#607D8B",
                 font=("Helvetica", 8, "bold")).pack(pady=(4, 2))
        self._btn(self.sidebar, "📊 Alle Analysen zeigen", self._run_all, primary=True).pack(fill="x", padx=10, pady=3)
        self._btn(self.sidebar, "📈 Zeitverlauf",       lambda: self._show_tab(0)).pack(fill="x", padx=10, pady=2)
        self._btn(self.sidebar, "🎯 Kreisdiagramme",    lambda: self._show_tab(1)).pack(fill="x", padx=10, pady=2)
        self._btn(self.sidebar, "📦 Probenvergleich",   lambda: self._show_tab(2)).pack(fill="x", padx=10, pady=2)
        self._btn(self.sidebar, "📐 Statistik",         lambda: self._show_tab(3)).pack(fill="x", padx=10, pady=2)

        # ── rechter Hauptbereich ──
        self.main_frame = tk.Frame(self, bg=BG)
        self.main_frame.pack(side="left", fill="both", expand=True)

        # Notebook (Tabs)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background="#CFD8DC", foreground=TEXT_DARK,
                        padding=[12, 5], font=("Helvetica", 10))
        style.map("TNotebook.Tab", background=[("selected", ACCENT)],
                  foreground=[("selected", "white")])

        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_time   = tk.Frame(self.notebook, bg=BG)
        self.tab_polar  = tk.Frame(self.notebook, bg=BG)
        self.tab_bar    = tk.Frame(self.notebook, bg=BG)
        self.tab_stats  = tk.Frame(self.notebook, bg=BG)

        self.notebook.add(self.tab_time,  text="📈 Zeitverlauf")
        self.notebook.add(self.tab_polar, text="🎯 Kreisdiagramme")
        self.notebook.add(self.tab_bar,   text="📦 Probenvergleich")
        self.notebook.add(self.tab_stats, text="📐 Statistik")

        self._show_welcome()

    def _btn(self, parent, text, cmd, primary=False, danger=False):
        if danger:
            bg, fg, abg = "#B71C1C", "white", "#C62828"
        elif primary:
            bg, fg, abg = "#1565C0", "white", "#1976D2"
        else:
            bg, fg, abg = "#37474F", TEXT_LIGHT, "#455A64"
        b = tk.Button(parent, text=text, command=cmd,
                      bg=bg, fg=fg, activebackground=abg,
                      activeforeground="white", relief="flat",
                      font=("Helvetica", 9), cursor="hand2", pady=5)
        return b

    def _show_welcome(self):
        for tab in [self.tab_time, self.tab_polar, self.tab_bar, self.tab_stats]:
            for w in tab.winfo_children():
                w.destroy()
        msg = tk.Label(self.tab_time,
                       text="👈  Neue Probe anlegen, KIN-Dateien hinzufügen\n"
                            "und dann 'Alle Analysen zeigen' klicken.",
                       bg=BG, fg="#607D8B", font=("Helvetica", 13))
        msg.pack(expand=True)

    # ── Probenverwaltung ──────────────────────────────────────────────────────

    def _add_sample(self):
        dlg = SampleNameDialog(self)
        name = dlg.result
        if not name:
            return
        color = COLORS[self._color_idx % len(COLORS)]
        self._color_idx += 1
        s = Sample(name, color)
        self.samples.append(s)
        self._refresh_sample_list()
        self.sample_listbox.selection_clear(0, "end")
        self.sample_listbox.selection_set(len(self.samples) - 1)
        self._on_sample_select()

    def _delete_sample(self):
        sel = self.sample_listbox.curselection()
        if not sel:
            messagebox.showinfo("Hinweis", "Bitte eine Probe auswählen.")
            return
        idx = sel[0]
        name = self.samples[idx].name
        if messagebox.askyesno("Löschen", f"Probe '{name}' wirklich löschen?"):
            del self.samples[idx]
            self._refresh_sample_list()
            self.file_listbox.delete(0, "end")

    def _refresh_sample_list(self):
        self.sample_listbox.delete(0, "end")
        for s in self.samples:
            n_files = len(s.files)
            self.sample_listbox.insert("end", f"● {s.name}  ({n_files} Dateien)")
        # Farben
        for i, s in enumerate(self.samples):
            self.sample_listbox.itemconfig(i, fg=s.color)

    def _on_sample_select(self, event=None):
        sel = self.sample_listbox.curselection()
        if not sel:
            return
        s = self.samples[sel[0]]
        self.file_listbox.delete(0, "end")
        for meta, _ in s.files:
            self.file_listbox.insert("end", "  " + meta["filename"])

    def _selected_sample(self):
        sel = self.sample_listbox.curselection()
        if not sel:
            messagebox.showinfo("Hinweis", "Bitte zuerst eine Probe auswählen.")
            return None
        return self.samples[sel[0]]

    def _add_files(self):
        s = self._selected_sample()
        if s is None:
            return
        paths = filedialog.askopenfilenames(
            title="KIN-Dateien auswählen",
            filetypes=[("KIN-Dateien", "*.KIN *.kin"), ("Alle", "*.*")]
        )
        for p in paths:
            try:
                s.add_file(p)
            except Exception as e:
                messagebox.showerror("Fehler", f"Datei konnte nicht geladen werden:\n{p}\n\n{e}")
        self._refresh_sample_list()
        self._on_sample_select()

    def _remove_file(self):
        s = self._selected_sample()
        if s is None:
            return
        sel = self.file_listbox.curselection()
        if not sel:
            messagebox.showinfo("Hinweis", "Bitte eine Datei auswählen.")
            return
        del s.files[sel[0]]
        self._refresh_sample_list()
        self._on_sample_select()

    def _show_tab(self, idx):
        self.notebook.select(idx)
        self._run_all()

    # ── Analyse ───────────────────────────────────────────────────────────────

    def _run_all(self):
        if not self.samples or all(len(s.files) == 0 for s in self.samples):
            messagebox.showinfo("Keine Daten", "Bitte zuerst Proben mit KIN-Dateien anlegen.")
            return
        samples = [s for s in self.samples if s.files]
        self._plot_time(samples)
        self._plot_polar(samples)
        self._plot_bar(samples)
        self._plot_stats(samples)

    # ── Wissenschaftlicher Stil ──────────────────────────────────────────────

    def _sci_ax(self, ax):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(direction="out", length=4, width=1.0, labelsize=10)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#CCCCCC", alpha=0.8)
        ax.set_axisbelow(True)

    def _sci_fig(self, *args, **kwargs):
        kwargs.setdefault("facecolor", "white")
        return Figure(*args, **kwargs)

    # ── Tab 1: Zeitverlauf ────────────────────────────────────────────────────

    def _plot_time(self, samples):
        for w in self.tab_time.winfo_children():
            w.destroy()

        fig = self._sci_fig(figsize=(11, 5), tight_layout=True)
        ax  = fig.add_subplot(111)
        self._sci_ax(ax)
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.fill_between([-5, 125], 0, 100, alpha=0.05, color="#4CAF50", zorder=0)
        ax.fill_between([-5, 125], -100, 0, alpha=0.05, color="#F44336", zorder=0)

        for s in samples:
            result = s.mean_direction_series()
            if result[0] is None:
                continue
            t, mean, sem = result
            mask = ~np.isnan(mean)
            ax.plot(t[mask], mean[mask], color=s.color, linewidth=2, label=s.name)
            ax.fill_between(t[mask],
                            (mean - sem)[mask], (mean + sem)[mask],
                            color=s.color, alpha=0.15)

        ax.set_xlabel("Zeit [s]", fontsize=11)
        ax.set_ylabel("Direction [%]", fontsize=11)
        ax.set_title("Phototaxis-Zeitverlauf (Mittelwert ± SEM)", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlim(0, 122)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=9)
        ax.text(1, 94, "▲ positive Phototaxis", color="#2E7D32", alpha=0.7, fontsize=8, va="top")
        ax.text(1, -94, "▼ negative Phototaxis", color="#B71C1C", alpha=0.7, fontsize=8, va="bottom")

        self._embed_figure(fig, self.tab_time)

    # ── Tab 2: Kreisdiagramme ─────────────────────────────────────────────────

    def _plot_polar(self, samples):
        for w in self.tab_polar.winfo_children():
            w.destroy()

        n = len(samples)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig = self._sci_fig(figsize=(4.5 * cols, 4.2 * rows))
        fig.suptitle("Bewegungsrichtungen der Zellen (Kreisdiagramme)",
                     fontsize=12, fontweight="bold", y=1.01)

        for i, s in enumerate(samples):
            ax = fig.add_subplot(rows, cols, i + 1, projection="polar")
            thetas, rs = s.polar_data()
            if len(thetas) == 0:
                ax.set_title(s.name, color=s.color)
                continue

            theta_rad = np.deg2rad(thetas)

            # Histogramm der Richtungen
            bins = np.linspace(0, 2 * np.pi, 37)
            counts, bin_edges = np.histogram(theta_rad, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bar_width   = 2 * np.pi / 36

            bars = ax.bar(bin_centers, counts, width=bar_width, bottom=0,
                          color=s.color, alpha=0.7, edgecolor="white", linewidth=0.5)

            # Mittlerer Vektor (Rayleigh)
            mean_dir = np.angle(np.mean(np.exp(1j * theta_rad)))
            r_mean   = np.abs(np.mean(np.exp(1j * theta_rad)))
            ax.annotate("", xy=(mean_dir, r_mean * max(counts)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2))

            # Lichtrichtung markieren (oben = 90° = "up")
            ax.annotate("☀", xy=(np.pi / 2, max(counts) * 1.15), ha="center",
                        fontsize=14, color="#FFA000")

            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_title(f"{s.name}\n(n={len(thetas)} Messpunkte)",
                         color=s.color, fontweight="bold", pad=12)
            ax.set_yticklabels([])

        fig.tight_layout()
        self._embed_figure(fig, self.tab_polar)

    # ── Tab 3: Probenvergleich ────────────────────────────────────────────────

    def _plot_bar(self, samples):
        for w in self.tab_bar.winfo_children():
            w.destroy()

        fig = self._sci_fig(figsize=(11, 6.5))
        fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.22, wspace=0.35)

        # Links: Barplot Direction
        ax1 = fig.add_subplot(121)
        self._sci_ax(ax1)
        ax1.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)

        names  = [s.name for s in samples]
        means  = [s.mean_direction() for s in samples]
        sems   = [s.sem_direction()  for s in samples]
        colors = [s.color for s in samples]
        x      = np.arange(len(samples))

        ax1.bar(x, means, yerr=sems, capsize=7,
                color=colors, alpha=0.75, edgecolor="white",
                linewidth=1.5, error_kw=dict(elinewidth=2, ecolor="black", zorder=6))

        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax1.set_ylabel("Direction [%]  (Mittelwert \u00b1 SEM)", fontsize=10)
        ax1.set_title("Up/Down-Verh\u00e4ltnis pro Probe", fontsize=12, fontweight="bold")

        # Rechts: Up vs Down Geschwindigkeit
        ax2 = fig.add_subplot(122)
        self._sci_ax(ax2)
        bw  = 0.35
        x2  = np.arange(len(names))
        up_means_v, down_means_v = [], []
        up_sems_v,  down_sems_v  = [], []
        for s in samples:
            uvals = np.concatenate([df["up velocity"].dropna().values   for _, df in s.files]) if s.files else np.array([])
            dvals = np.concatenate([df["down velocity"].dropna().values for _, df in s.files]) if s.files else np.array([])
            up_means_v.append(np.nanmean(uvals)  if len(uvals) else 0)
            down_means_v.append(np.nanmean(dvals) if len(dvals) else 0)
            up_sems_v.append(stats.sem(uvals, nan_policy="omit") if len(uvals) > 1 else 0)
            down_sems_v.append(stats.sem(dvals, nan_policy="omit") if len(dvals) > 1 else 0)

        ax2.bar(x2 - bw/2, up_means_v,   bw, yerr=up_sems_v,   capsize=5,
                color="#42A5F5", alpha=0.85, label="Up velocity", edgecolor="white")
        ax2.bar(x2 + bw/2, down_means_v, bw, yerr=down_sems_v, capsize=5,
                color="#EF5350", alpha=0.85, label="Down velocity", edgecolor="white")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax2.set_ylabel("Geschwindigkeit [\u00b5m/s]", fontsize=10)
        ax2.set_title("Up- vs. Down-Geschwindigkeit", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.25)

        self._embed_figure(fig, self.tab_bar)

    def _add_significance_brackets(self, ax, samples, x, means):
        """Signifikanz-Brackets (ns / * / ** / ***) zwischen allen Probenpaaren."""
        n = len(samples)
        if n < 2:
            return

        # Paarweise Mann-Whitney-U auf Datei-Mittelwerten (biologische Replikate)
        def file_means(s):
            vals = []
            for _, df in s.files:
                col = direction_col(df)
                if col is not None:
                    m = df[col].mean()
                    if not np.isnan(m):
                        vals.append(m)
            return np.array(vals)

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                a = file_means(samples[i])
                b = file_means(samples[j])
                if len(a) < 2 or len(b) < 2:
                    # Zu wenige Replikate: trotzdem anzeigen, aber als "n.d."
                    pairs.append((i, j, None))
                    continue
                try:
                    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                    pairs.append((i, j, p))
                except Exception:
                    pass

        if not pairs:
            return

        # Sortierung: erst kurze Abstände (weniger Ebenen nötig)
        pairs.sort(key=lambda t: t[1] - t[0])

        # Startpunkt der Brackets: oberhalb aller Datenpunkte
        all_file_means = []
        for s in samples:
            all_file_means.extend(file_means(s).tolist())
        all_file_means.extend(means)
        data_top  = max(all_file_means) if all_file_means else 0
        data_bot  = min(all_file_means) if all_file_means else 0
        data_span = max(abs(data_top), abs(data_bot), 10)
        step      = data_span * 0.13

        bracket_y = data_top + step * 0.6

        for k, (i, j, p) in enumerate(pairs):
            if p is None:
                label = "n.d."
                col   = "#AAAAAA"
            elif p < 0.001:
                label, col = "***", "black"
            elif p < 0.01:
                label, col = "**",  "black"
            elif p < 0.05:
                label, col = "*",   "black"
            else:
                label, col = "ns",  "#888888"

            y = bracket_y + k * step
            tick = step * 0.18

            ax.plot([x[i], x[i], x[j], x[j]],
                    [y - tick, y, y, y - tick],
                    color="black", linewidth=0.9, clip_on=False)
            ax.text((x[i] + x[j]) / 2, y + step * 0.08, label,
                    ha="center", va="bottom", fontsize=9,
                    color=col, fontweight="bold" if label != "ns" else "normal")

        ax.set_ylim(ax.get_ylim()[0],
                    bracket_y + len(pairs) * step + step * 0.5)

    # ── Tab 4: Statistik ──────────────────────────────────────────────────────

    def _plot_stats(self, samples):
        for w in self.tab_stats.winfo_children():
            w.destroy()

        # Statistik auf Datei-Mittelwerten (= biologische Replikate, nicht alle Zeitpunkte)
        def file_means(s):
            ms = []
            for _, df in s.files:
                col = direction_col(df)
                if col is not None:
                    m = df[col].mean()
                    if not np.isnan(m):
                        ms.append(m)
            return np.array(ms)

        groups_all = [file_means(s) for s in samples]

        def stars(p):
            return ("***" if p < 0.001 else "**" if p < 0.01
                    else "*" if p < 0.05 else "ns")

        def p_fmt(p):
            return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"

        # ── scrollbarer Inhaltsbereich ──────────────────────────────────────
        canvas_outer = tk.Canvas(self.tab_stats, bg=BG, highlightthickness=0)
        scrollbar    = ttk.Scrollbar(self.tab_stats, orient="vertical",
                                     command=canvas_outer.yview)
        canvas_outer.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas_outer.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas_outer, bg=BG)
        inner_win = canvas_outer.create_window((0, 0), window=inner, anchor="nw")

        def _on_resize(event):
            canvas_outer.itemconfig(inner_win, width=event.width)
        def _on_frame(event):
            canvas_outer.configure(scrollregion=canvas_outer.bbox("all"))
        canvas_outer.bind("<Configure>", _on_resize)
        inner.bind("<Configure>", _on_frame)

        def section(text):
            tk.Label(inner, text=text, bg=BG, fg=TEXT_DARK,
                     font=("Helvetica", 10, "bold")).pack(
                anchor="w", padx=12, pady=(14, 3))

        def sep():
            ttk.Separator(inner, orient="horizontal").pack(
                fill="x", padx=12, pady=4)

        def legend():
            tk.Label(inner,
                     text="  Signifikanzniveaus:  ns  p ≥ 0.05  |"
                          "  * p < 0.05  |  ** p < 0.01  |  *** p < 0.001",
                     bg=BG, fg="#666666", font=("Helvetica", 8)
                     ).pack(anchor="w", padx=12, pady=(0, 6))

        def make_tree(parent, columns, widths, height=6):
            frame = tk.Frame(parent, bg=BG)
            frame.pack(fill="x", padx=12, pady=(0, 6))
            tv = ttk.Treeview(frame, columns=columns, show="headings", height=height)
            for col, w in zip(columns, widths):
                tv.heading(col, text=col)
                tv.column(col, width=w, anchor="center", minwidth=w)
            sb = ttk.Scrollbar(frame, orient="vertical", command=tv.yview)
            tv.configure(yscrollcommand=sb.set)
            tv.pack(side="left", fill="x", expand=True)
            sb.pack(side="left", fill="y")
            return tv

        # ══════════════════════════════════════════════════════════════════
        # 1. t-Test jeder Probe gegen 0
        # ══════════════════════════════════════════════════════════════════
        section("1.  Einzel-Probe: t-Test gegen 0  (kein Phototaxis-Effekt)")

        cols_t  = ("Probe", "n", "Mittelwert [%]", "SEM", "t-Wert", "p-Wert", "Signifikanz", "Interpretation")
        widths_t = [130, 60, 120, 80, 80, 90, 90, 140]
        tree_t  = make_tree(inner, cols_t, widths_t,
                            height=min(len(samples) + 1, 8))
        tree_t.column("Probe",          anchor="w")
        tree_t.column("Interpretation", anchor="w")
        tree_t.tag_configure("pos", background="#E8F5E9", foreground="#1B5E20")
        tree_t.tag_configure("neg", background="#FFEBEE", foreground="#B71C1C")
        tree_t.tag_configure("neu", background="#FFFDE7", foreground="#5D4037")

        for s, grp in zip(samples, groups_all):
            if len(grp) < 2:
                continue
            mean = grp.mean()
            sem  = stats.sem(grp)
            t_s, p = stats.ttest_1samp(grp, 0)
            interp = ("pos. Phototaxis" if mean > 5
                      else "neg. Phototaxis" if mean < -5 else "kein Effekt")
            tag = "pos" if mean > 5 else "neg" if mean < -5 else "neu"
            tree_t.insert("", "end",
                          values=(s.name, len(grp), f"{mean:+.2f}",
                                  f"±{sem:.2f}", f"{t_s:+.2f}",
                                  p_fmt(p), stars(p), interp),
                          tags=(tag,))
        tk.Label(inner,
                 text="  Hinweis: n = Anzahl KIN-Dateien (Replikate) je Probe, nicht Zeitpunkte.",
                 bg=BG, fg="#888888", font=("Helvetica", 8)).pack(anchor="w", padx=12)
        legend()
        sep()

        # ══════════════════════════════════════════════════════════════════
        # 2. Einfaktorielle ANOVA (alle Proben)
        # ══════════════════════════════════════════════════════════════════
        section("2.  Einfaktorielle ANOVA  (alle Proben gemeinsam)")

        valid_grps = [g for g in groups_all if len(g) >= 2]
        anova_frame = tk.Frame(inner, bg=BG)
        anova_frame.pack(fill="x", padx=12, pady=(0, 6))

        if len(valid_grps) >= 2:
            try:
                f_stat, p_anova = stats.f_oneway(*valid_grps)
                # Eta-Quadrat (Effektgröße)
                grand_mean = np.concatenate(valid_grps).mean()
                ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in valid_grps)
                ss_total   = sum(((v - grand_mean)**2).sum() for v in valid_grps)
                eta2 = ss_between / ss_total if ss_total > 0 else float("nan")

                anova_cols   = ("F-Wert", "df (Gruppen)", "df (Fehler)", "p-Wert", "Signifikanz", "Eta² (Effekt)")
                anova_widths = [100, 110, 110, 100, 100, 130]
                tv_an = make_tree(anova_frame, anova_cols, anova_widths, height=2)
                df_between = len(valid_grps) - 1
                df_within  = sum(len(g) for g in valid_grps) - len(valid_grps)
                sig_an = stars(p_anova)
                col_tag = "sig" if p_anova < 0.05 else "ns_row"
                tv_an.tag_configure("sig",    background="#E3F2FD", foreground="#0D47A1")
                tv_an.tag_configure("ns_row", background="#F5F5F5", foreground="#555555")
                tv_an.insert("", "end",
                             values=(f"{f_stat:.3f}", df_between, df_within,
                                     p_fmt(p_anova), sig_an, f"{eta2:.3f}"),
                             tags=(col_tag,))

                # Interpretation Eta²
                if eta2 < 0.01:
                    eff = "vernachlässigbar"
                elif eta2 < 0.06:
                    eff = "klein"
                elif eta2 < 0.14:
                    eff = "mittel"
                else:
                    eff = "groß"
                tk.Label(anova_frame,
                         text=f"  η² = {eta2:.3f} → Effektgröße: {eff}"
                              "   (η² < 0.01 vernachl., < 0.06 klein, < 0.14 mittel, ≥ 0.14 groß)",
                         bg=BG, fg="#555555", font=("Helvetica", 8)
                         ).pack(anchor="w", pady=(2, 0))
            except Exception as e:
                tk.Label(anova_frame, text=f"ANOVA nicht berechenbar: {e}",
                         bg=BG, fg="red").pack(anchor="w")
        else:
            tk.Label(anova_frame,
                     text="  Mindestens 2 Proben mit je ≥2 Werten erforderlich.",
                     bg=BG, fg="#888888", font=("Helvetica", 9)).pack(anchor="w")

        legend()
        sep()

        # ══════════════════════════════════════════════════════════════════
        # 3. Post-hoc: paarweiser Tukey-HSD
        # ══════════════════════════════════════════════════════════════════
        section("3.  Post-hoc-Test: paarweiser Tukey-HSD")

        posthoc_frame = tk.Frame(inner, bg=BG)
        posthoc_frame.pack(fill="x", padx=12, pady=(0, 6))

        n_pairs = len(samples) * (len(samples) - 1) // 2
        cols_ph  = ("Probe 1", "Probe 2", "Mitteldiff. [%]", "p-Wert (adj.)", "Signifikanz")
        widths_ph = [150, 150, 130, 130, 100]
        tv_ph = ttk.Treeview(posthoc_frame, columns=cols_ph, show="headings",
                              height=min(max(n_pairs, 2), 8))
        for col, w in zip(cols_ph, widths_ph):
            tv_ph.heading(col, text=col)
            tv_ph.column(col, width=w, anchor="center", minwidth=w)
        tv_ph.column("Probe 1", anchor="w")
        tv_ph.column("Probe 2", anchor="w")
        tv_ph.tag_configure("sig", background="#E3F2FD", foreground="#0D47A1")
        tv_ph.tag_configure("ns",  background="#F5F5F5", foreground="#555555")

        if len(valid_grps) >= 2:
            try:
                # Tukey-HSD: manuelle Implementierung (keine externen Abhängigkeiten)
                from scipy.stats import studentized_range
                n_groups  = len(valid_grps)
                N_total   = sum(len(g) for g in valid_grps)
                df_err    = N_total - n_groups
                ms_within = sum(((g - g.mean())**2).sum() for g in valid_grps) / df_err

                for i in range(n_groups):
                    for j in range(i + 1, n_groups):
                        gi = [g for s2, g in zip(samples, groups_all) if len(g) >= 2][i]
                        gj = [g for s2, g in zip(samples, groups_all) if len(g) >= 2][j]
                        si = [s2 for s2, g in zip(samples, groups_all) if len(g) >= 2][i]
                        sj = [s2 for s2, g in zip(samples, groups_all) if len(g) >= 2][j]
                        meandiff = gi.mean() - gj.mean()
                        se = np.sqrt(ms_within / 2 * (1/len(gi) + 1/len(gj)))
                        if se == 0:
                            continue
                        q_stat = abs(meandiff) / se
                        p_adj  = 1 - studentized_range.cdf(q_stat, n_groups, df_err)
                        p_adj  = min(max(p_adj, 0.0), 1.0)
                        tag = "sig" if p_adj < 0.05 else "ns"
                        tv_ph.insert("", "end",
                                     values=(si.name, sj.name,
                                             f"{meandiff:+.2f}",
                                             p_fmt(p_adj), stars(p_adj)),
                                     tags=(tag,))
            except Exception as e:
                tk.Label(posthoc_frame, text=f"  Tukey-HSD Fehler: {e}",
                         bg=BG, fg="red", font=("Helvetica", 9)).pack(anchor="w")
        else:
            tk.Label(posthoc_frame,
                     text="  Mindestens 2 Proben mit je ≥2 Werten erforderlich.",
                     bg=BG, fg="#888888", font=("Helvetica", 9)).pack(anchor="w")

        sb_ph = ttk.Scrollbar(posthoc_frame, orient="vertical", command=tv_ph.yview)
        tv_ph.configure(yscrollcommand=sb_ph.set)
        tv_ph.pack(side="left", fill="x", expand=True)
        sb_ph.pack(side="left", fill="y")
        legend()
        sep()

        # ══════════════════════════════════════════════════════════════════
        # 4. Boxplot
        # ══════════════════════════════════════════════════════════════════
        section("4.  Verteilung: Boxplot  (Median, IQR, Whisker = 1.5×IQR, Ausreißer = ○)")

        fig_frame = tk.Frame(inner, bg="white")
        fig_frame.pack(fill="x", padx=12, pady=(0, 12))

        fig = self._sci_fig(figsize=(10, 3.8), tight_layout=True)
        ax  = fig.add_subplot(111)
        self._sci_ax(ax)
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)

        data_box, labels_box, colors_box = [], [], []
        for s, grp in zip(samples, groups_all):
            if len(grp):
                data_box.append(grp)
                labels_box.append(s.name)
                colors_box.append(s.color)

        if data_box:
            bp = ax.boxplot(data_box, patch_artist=True,
                            medianprops=dict(color="#111111", linewidth=2.0),
                            whiskerprops=dict(linewidth=1.2, color="#444444"),
                            capprops=dict(linewidth=1.2, color="#444444"),
                            flierprops=dict(marker="o", markersize=4,
                                            markerfacecolor="#999999",
                                            markeredgewidth=0.5,
                                            markeredgecolor="#555555", alpha=0.6))
            for patch, color in zip(bp["boxes"], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.60)
                patch.set_linewidth(1.2)

        ax.set_xticks(range(1, len(labels_box) + 1))
        ax.set_xticklabels(labels_box, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Direction [%]", fontsize=10)
        ax.set_title("Verteilung der Phototaxis-Werte",
                     fontsize=11, fontweight="bold", pad=8)

        self._embed_figure(fig, fig_frame)


    # ── Hilfsfunktionen ───────────────────────────────────────────────────────

    def _embed_figure(self, fig, parent):
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()

        toolbar_frame = tk.Frame(parent, bg="#EEEEEE")
        toolbar_frame.pack(side="bottom", fill="x")

        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        # Speichern-Button mit 300 dpi
        def save_hq():
            from tkinter import filedialog
            path = filedialog.asksaveasfilename(
                title="Abbildung speichern (300 dpi)",
                defaultextension=".png",
                filetypes=[
                    ("PNG (300 dpi, empfohlen)", "*.png"),
                    ("SVG (Vektorgrafik)",       "*.svg"),
                    ("PDF (Vektorgrafik)",       "*.pdf"),
                    ("TIFF (300 dpi)",           "*.tiff"),
                ]
            )
            if not path:
                return
            ext = path.rsplit(".", 1)[-1].lower()
            dpi = 72 if ext in ("svg", "pdf") else 300
            fig.savefig(path, dpi=dpi, bbox_inches="tight",
                        facecolor="white", transparent=False)
            if ext in ('svg', 'pdf'):
                detail = 'Format: Vektorgrafik (verlustfrei)'
            else:
                detail = f'Aufloesung: {dpi} dpi'
            messagebox.showinfo('Gespeichert',
                'Abbildung gespeichert:\n' + path + '\n' + detail)

        tk.Button(toolbar_frame, text="💾 Speichern (300 dpi)",
                  command=save_hq, bg="#1565C0", fg="white",
                  relief="flat", font=("Helvetica", 9), cursor="hand2",
                  padx=8, pady=3).pack(side="right", padx=6, pady=3)

        canvas.get_tk_widget().pack(fill="both", expand=True)


# ── Dialog: Probenname ─────────────────────────────────────────────────────────

class SampleNameDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Neue Probe")
        self.geometry("320x140")
        self.resizable(False, False)
        self.configure(bg=BG)
        self.grab_set()
        self.result = None

        tk.Label(self, text="Probenname:", bg=BG,
                 font=("Helvetica", 11)).pack(pady=(20, 5))
        self.entry = ttk.Entry(self, font=("Helvetica", 11), width=28)
        self.entry.pack()
        self.entry.insert(0, f"Probe {len(parent.samples) + 1}")
        self.entry.select_range(0, "end")
        self.entry.focus()
        self.entry.bind("<Return>", lambda e: self._ok())

        btn_f = tk.Frame(self, bg=BG)
        btn_f.pack(pady=12)
        tk.Button(btn_f, text="OK", command=self._ok, bg=ACCENT, fg="white",
                  relief="flat", width=10, cursor="hand2").pack(side="left", padx=4)
        tk.Button(btn_f, text="Abbrechen", command=self.destroy,
                  bg="#607D8B", fg="white", relief="flat", width=10,
                  cursor="hand2").pack(side="left", padx=4)
        self.wait_window()

    def _ok(self):
        name = self.entry.get().strip()
        if name:
            self.result = name
        self.destroy()


# ── Start ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = PhototaxisApp()
    app.mainloop()
