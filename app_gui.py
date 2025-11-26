import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import sys
import threading
import os
import shlex
from PIL import Image, ImageTk
from lab2_features import init_detector, process_pair_return_imgs
import time
import cv2


class LabLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторные — Меню")
        self.geometry("620x355")
        self.resizable(False, False)

        self.proc = None
        self.proc_lock = threading.Lock()

        self.create_widgets()

    def create_widgets(self):
        pad = 10
        frame = ttk.Frame(self, padding=pad)
        frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(frame, text="Выберите лабораторную", font=(None, 14))
        title.pack(pady=(0, 8))

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(0, 8))

        # Lab 1 button
        self.lab1_btn = ttk.Button(
            btn_frame, text="Лаб. 1 — Обнаружение лиц/глаз", command=self.on_lab1
        )
        self.lab1_btn.pack(fill=tk.X, padx=pad, pady=4)

        # Lab 2 UI
        lab2_container = ttk.LabelFrame(frame, text="Лаб.2 — Поиск шаблона (в одном окне)")
        lab2_container.pack(fill=tk.BOTH, padx=pad, pady=4, expand=False)

        l2_select_row = ttk.Frame(lab2_container)
        l2_select_row.pack(fill=tk.X, padx=6, pady=6)
        self.lab2_template_var = tk.StringVar()
        ttk.Entry(l2_select_row, textvariable=self.lab2_template_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(l2_select_row, text="Шаблон", command=self.choose_lab2_template).pack(side=tk.LEFT, padx=4)
        self.lab2_scene_var = tk.StringVar()
        ttk.Entry(l2_select_row, textvariable=self.lab2_scene_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8,0))
        ttk.Button(l2_select_row, text="Сцена (файл)", command=self.choose_lab2_scene).pack(side=tk.LEFT, padx=4)

        opts = ttk.Frame(lab2_container)
        opts.pack(fill=tk.X, padx=6)
        self.lab2_show_kp = tk.BooleanVar(value=True)
        self.lab2_show_matches = tk.BooleanVar(value=True)
        self.lab2_show_contour = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="KP", variable=self.lab2_show_kp).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(opts, text="Matches", variable=self.lab2_show_matches).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(opts, text="Contour", variable=self.lab2_show_contour).pack(side=tk.LEFT, padx=4)

        btns = ttk.Frame(lab2_container)
        btns.pack(fill=tk.X, padx=6, pady=(6,8))
        ttk.Button(btns, text="Start Lab2 (single-window)", command=self.start_lab2_in_window).pack(side=tk.LEFT)
        ttk.Button(btns, text="Stop Lab2", command=self.stop_lab2_in_window).pack(side=tk.LEFT, padx=8)

        # ---- Отдельный фрейм для canvas отображения (ниже секции) ----
        display_frame = ttk.Frame(self)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # left column (scene + matches)
        self.left_col = ttk.Frame(display_frame)
        self.left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scene_label = ttk.Label(self.left_col)
        self.scene_label.pack(fill=tk.BOTH, expand=True)

        self.matches_label = ttk.Label(self.left_col)
        self.matches_label.pack(fill=tk.BOTH, expand=True)

        # right column (template)
        self.right_col = ttk.Frame(display_frame, width=220)
        self.right_col.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(self.right_col, text="Template (справа)").pack()
        self.template_label = ttk.Label(self.right_col)
        self.template_label.pack(pady=8)

        # ----- Переменные для режима Lab2 -----
        self._lab2_running = False
        self._lab2_thread = None
        self._lab2_stop_event = threading.Event()
        self._lab2_capture = None
        self._lab2_detector = None

        # Lab 3 UI
        lab3_container = ttk.LabelFrame(
            frame, text="Лаб. 3 — Контурный анализ / поиск по шаблонам"
        )
        lab3_container.pack(fill=tk.X, padx=pad, pady=4)
        l3_row = ttk.Frame(lab3_container)
        l3_row.pack(fill=tk.X, pady=6, padx=6)
        self.lab3_dir_var = tk.StringVar()
        self.lab3_entry = ttk.Entry(l3_row, textvariable=self.lab3_dir_var)
        self.lab3_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lab3_browse = ttk.Button(
            l3_row, text="Выбрать папку", command=self.choose_lab3_dir
        )
        self.lab3_browse.pack(side=tk.LEFT, padx=(8, 0))
        self.lab3_btn = ttk.Button(
            lab3_container, text="Запустить Лаб.3", command=self.on_lab3
        )
        self.lab3_btn.pack(fill=tk.X, padx=6, pady=(0, 8))

        # Bottom controls: camera selection, run/stop
        bottom = ttk.Frame(frame)
        bottom.pack(fill=tk.X, pady=(6, 0), padx=pad)

        cam_label = ttk.Label(bottom, text="Камера (index):")
        cam_label.pack(side=tk.LEFT)
        self.camera_var = tk.IntVar(value=0)
        self.cam_spin = ttk.Spinbox(
            bottom, from_=0, to=10, width=4, textvariable=self.camera_var
        )
        self.cam_spin.pack(side=tk.LEFT, padx=(6, 12))

        self.run_btn = ttk.Button(
            bottom, text="Запустить выбранную", command=self.on_run_current
        )
        # NOTE: the run button will call specific handlers; we keep it disabled to avoid ambiguity
        self.run_btn.pack_forget()

        self.stop_btn = ttk.Button(
            bottom, text="Остановить процесс", command=self.on_stop
        )
        self.stop_btn.pack(side=tk.RIGHT)

        # status box
        status_frame = ttk.Frame(frame)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=pad, pady=(8, 0))
        self.status = tk.Text(status_frame, height=5, state=tk.DISABLED)
        self.status.pack(fill=tk.BOTH, expand=True)

        # Menu bar
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Выход", command=self.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Инструкция", command=self.show_help)
        menubar.add_cascade(label="Помощь", menu=help_menu)
        self.config(menu=menubar)

    def log(self, text):
        self.status.config(state=tk.NORMAL)
        self.status.insert(tk.END, text + "\n")
        self.status.see(tk.END)
        self.status.config(state=tk.DISABLED)

    def choose_lab2_scene(self):
        p = filedialog.askopenfilename(
            title="Выберите файл сцены",
            filetypes=[("Изображения", "*.png *.jpg *.bmp *.jpeg"), ("All", "*.*")],
        )
        if p:
            self.lab2_scene_var.set(p)
            self.log(f"Выбрана сцена для Лаб.2: {p}")

    def choose_lab2_template(self):
        p = filedialog.askopenfilename(
            title="Выберите файл шаблона",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp"), ("Все файлы", "*")],
        )
        if p:
            self.lab2_template_var.set(p)
            self.log(f"Выбран шаблон для Лаб.2: {p}")

    def choose_lab3_dir(self):
        d = filedialog.askdirectory(title="Выберите папку с шаблонами для Лаб.3")
        if d:
            self.lab3_dir_var.set(d)
            self.log(f"Выбрана папка для Лаб.3: {d}")

    def show_help(self):
        msg = (
            "Приложение запускает лабораторные модули.\n"
            "- Для Лаб.1 нажмите соответствующую кнопку.\n"
            "- Для Лаб.2 выберите файл шаблона и (опционально) файл сцены или используйте камеру.\n"
            "- Для Лаб.3 выберите папку с шаблонами и нажмите Запустить.\n"
            "Визуализация Лаб.2 происходит в окне приложения (справа — шаблон, слева — сцена/совпадения)."
        )
        messagebox.showinfo("Инструкция", msg)

    def build_command(self, lab: int):
        # Используем текущий интерпретатор для запуска main.py
        py = sys.executable or "python"
        cmd = None
        cam = str(self.camera_var.get())
        if lab == 1:
            cmd = [py, "main.py", "lab1", "--camera", cam]
        elif lab == 2:
            tpl = self.lab2_template_var.get().strip()
            if not tpl:
                messagebox.showwarning(
                    "Шаблон не выбран", "Выберите файл шаблона для Лаб.2"
                )
                return None
            cmd = [py, "main.py", "lab2", "--template", tpl, "--camera", cam]
        elif lab == 3:
            d = self.lab3_dir_var.get().strip()
            if not d:
                messagebox.showwarning(
                    "Папка не выбрана", "Выберите папку шаблонов для Лаб.3"
                )
                return None
            cmd = [py, "main.py", "lab3", "--templates_dir", d, "--camera", cam]
        return cmd

    def start_process(self, cmd):
        # Start subprocess and keep reference
        try:
            self.log(f"Запускаю: {' '.join(shlex.quote(p) for p in cmd)}")
            with self.proc_lock:
                if self.proc is not None:
                    self.log(
                        "Процесс уже запущен. Остановите его перед запуском нового."
                    )
                    return
                # start with stdout/stderr visible in console; we don't capture to avoid blocking
                self.proc = subprocess.Popen(cmd, cwd=os.getcwd())
        except Exception as e:
            self.log(f"Ошибка при запуске: {e}")

    def _run_in_thread(self, cmd):
        t = threading.Thread(target=self.start_process, args=(cmd,), daemon=True)
        t.start()

    def on_lab1(self):
        cmd = self.build_command(1)
        if cmd:
            self._run_in_thread(cmd)

    def on_lab2(self):
        # forward to single-window runner
        self.start_lab2_in_window()

    def start_lab2_in_window(self):
        if self._lab2_running:
            self.log("Lab2 уже запущена.")
            return

        tpl = self.lab2_template_var.get().strip()
        if not tpl:
            messagebox.showwarning("Шаблон не выбран", "Выберите шаблон")
            return
        # подготовка детектора
        det, norm, name = init_detector()
        self._lab2_detector = (det, norm)
        # подготовка источника: файл сцены или камера
        scene_file = self.lab2_scene_var.get().strip()
        if scene_file:
            # статический режим (один кадр)
            img_template = cv2.imread(tpl)
            img_scene = cv2.imread(scene_file)
            if img_template is None or img_scene is None:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображения")
                return
            self._lab2_running = True
            self._lab2_stop_event.clear()
            # сразу отобразим один кадр
            out = process_pair_return_imgs(det, norm, img_template, img_scene,
                                        do_keypoints=self.lab2_show_kp.get(),
                                        do_matches=self.lab2_show_matches.get(),
                                        do_contour=self.lab2_show_contour.get())
            self._display_lab2_outputs(img_template, out)
            self._lab2_running = False
            return

        # camera mode
        try:
            cam_idx = int(self.camera_var.get())
        except Exception:
            cam_idx = 0
        self._lab2_capture = cv2.VideoCapture(cam_idx)
        if not self._lab2_capture.isOpened():
            messagebox.showerror("Ошибка", f"Не удалось открыть камеру {cam_idx}")
            return

        self._lab2_running = True
        self._lab2_stop_event.clear()

        def loop():
            img_template = cv2.imread(tpl)
            while not self._lab2_stop_event.is_set():
                ret, frame = self._lab2_capture.read()
                if not ret:
                    break
                out = process_pair_return_imgs(det, norm, img_template, frame,
                                            do_keypoints=self.lab2_show_kp.get(),
                                            do_matches=self.lab2_show_matches.get(),
                                            do_contour=self.lab2_show_contour.get())
                # schedule GUI update in main thread via after
                self.after(0, lambda img_t=img_template.copy(), res=out: self._display_lab2_outputs(img_t, res))
                time.sleep(0.03)  # ~30 FPS limit
            # cleanup
            self._lab2_running = False
            if self._lab2_capture:
                try:
                    self._lab2_capture.release()
                except Exception:
                    pass
                self._lab2_capture = None

        self._lab2_thread = threading.Thread(target=loop, daemon=True)
        self._lab2_thread.start()
        self.log("Lab2 (single-window) started")

    def stop_lab2_in_window(self):
        if not self._lab2_running:
            self.log("Lab2 не запущена")
            return
        self._lab2_stop_event.set()
        # подождём немного, чтобы поток освободил ресурс камеры
        if self._lab2_thread and self._lab2_thread.is_alive():
            self._lab2_thread.join(timeout=1.0)
        # дополнительная очистка
        if self._lab2_capture:
            try:
                self._lab2_capture.release()
            except Exception:
                pass
            self._lab2_capture = None
        self._lab2_running = False
        self.log("Lab2 stopped")

    def _display_lab2_outputs(self, img_template, outputs):
        # convert BGR->RGB->PIL Image->PhotoImage and set to labels
        def to_photo(cv_img, max_size=None):
            if cv_img is None:
                return None
            img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img)
            if max_size is not None:
                pil.thumbnail(max_size, Image.ANTIALIAS)
            return ImageTk.PhotoImage(pil)

        # template (right) - keep reasonable size
        tpl_photo = to_photo(img_template, max_size=(320,480))
        if tpl_photo:
            self.template_label.img = tpl_photo
            self.template_label.config(image=tpl_photo)

        # scene with box or keypoints shown in top-left
        scene_img = outputs.get('scene_box_img') or outputs.get('kp_scene_img') or outputs.get('matches_img')
        scene_photo = to_photo(scene_img, max_size=(640,360))
        if scene_photo:
            self.scene_label.img = scene_photo
            self.scene_label.config(image=scene_photo)

        # matches image shown bottom-left (if available); otherwise duplicate scene
        matches_img = outputs.get('matches_img') or scene_img
        matches_photo = to_photo(matches_img, max_size=(640,240))
        if matches_photo:
            self.matches_label.img = matches_photo
            self.matches_label.config(image=matches_photo)

        # update log with counts
        kp1,kp2 = outputs.get('kp_counts', (0,0))
        good = outputs.get('good_matches_count', 0)
        self.log(f"kp tpl:{kp1} scene:{kp2} | good matches: {good}")

    def on_lab3(self):
        cmd = self.build_command(3)
        if cmd:
            self._run_in_thread(cmd)

    def on_run_current(self):
        # reserved for generic run button (not used currently)
        pass

    def on_stop(self):
        with self.proc_lock:
            if self.proc is None:
                self.log("Процесс не запущен.")
                return

            try:
                self.log("Завершаю процесс...")
                self.proc.terminate()
                self.proc.wait(timeout=3)
                self.log("Процесс остановлен.")
            except Exception as e:
                self.log(f"Не удалось корректно завершить процесс: {e}")
            finally:
                self.proc = None


if __name__ == "__main__":
    app = LabLauncher()
    app.mainloop()
