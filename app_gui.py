import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import sys
import threading
import os
import shlex

class LabLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторные — Меню")
        self.geometry("520x320")
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
        btn_frame.pack(fill=tk.X, pady=(0,8))

        # Lab 1 button
        self.lab1_btn = ttk.Button(btn_frame, text="Лаб. 1 — Обнаружение лиц/глаз", command=self.on_lab1)
        self.lab1_btn.pack(fill=tk.X, padx=pad, pady=4)

        # Lab 2 UI
        lab2_container = ttk.LabelFrame(frame, text="Лаб. 2 — Устойчивые признаки / поиск шаблона")
        lab2_container.pack(fill=tk.X, padx=pad, pady=4)
        l2_row = ttk.Frame(lab2_container)
        l2_row.pack(fill=tk.X, pady=6, padx=6)
        self.lab2_template_var = tk.StringVar()
        self.lab2_entry = ttk.Entry(l2_row, textvariable=self.lab2_template_var)
        self.lab2_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lab2_browse = ttk.Button(l2_row, text="Выбрать шаблон", command=self.choose_lab2_template)
        self.lab2_browse.pack(side=tk.LEFT, padx=(8,0))
        self.lab2_btn = ttk.Button(lab2_container, text="Запустить Лаб.2", command=self.on_lab2)
        self.lab2_btn.pack(fill=tk.X, padx=6, pady=(0,8))

        # Lab 3 UI
        lab3_container = ttk.LabelFrame(frame, text="Лаб. 3 — Контурный анализ / поиск по шаблонам")
        lab3_container.pack(fill=tk.X, padx=pad, pady=4)
        l3_row = ttk.Frame(lab3_container)
        l3_row.pack(fill=tk.X, pady=6, padx=6)
        self.lab3_dir_var = tk.StringVar()
        self.lab3_entry = ttk.Entry(l3_row, textvariable=self.lab3_dir_var)
        self.lab3_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.lab3_browse = ttk.Button(l3_row, text="Выбрать папку", command=self.choose_lab3_dir)
        self.lab3_browse.pack(side=tk.LEFT, padx=(8,0))
        self.lab3_btn = ttk.Button(lab3_container, text="Запустить Лаб.3", command=self.on_lab3)
        self.lab3_btn.pack(fill=tk.X, padx=6, pady=(0,8))

        # Bottom controls: camera selection, run/stop
        bottom = ttk.Frame(frame)
        bottom.pack(fill=tk.X, pady=(6,0), padx=pad)

        cam_label = ttk.Label(bottom, text="Камера (index):")
        cam_label.pack(side=tk.LEFT)
        self.camera_var = tk.IntVar(value=0)
        self.cam_spin = ttk.Spinbox(bottom, from_=0, to=10, width=4, textvariable=self.camera_var)
        self.cam_spin.pack(side=tk.LEFT, padx=(6,12))

        self.run_btn = ttk.Button(bottom, text="Запустить выбранную», command=self.on_run_current")
        # NOTE: the run button will call specific handlers; we keep it disabled to avoid ambiguity
        self.run_btn.pack_forget()

        self.stop_btn = ttk.Button(bottom, text="Остановить процесс", command=self.on_stop)
        self.stop_btn.pack(side=tk.RIGHT)

        # status box
        status_frame = ttk.Frame(frame)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=pad, pady=(8,0))
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

    def choose_lab2_template(self):
        p = filedialog.askopenfilename(title="Выберите файл шаблона", filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp"), ("Все файлы","*")])
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
            "Приложение запускает лабораторные через файл main.py или соответствующие модули.\n"
            "- Для Лаб.1 нажмите соответствующую кнопку.\n"
            "- Для Лаб.2 выберите файл шаблона и нажмите Запустить.\n"
            "- Для Лаб.3 выберите папку с шаблонами и нажмите Запустить.\n"
            "Во время работы лабораторных будут открыты окна OpenCV; чтобы остановить их, нажмите Остановить процесс." 
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
                messagebox.showwarning("Шаблон не выбран", "Выберите файл шаблона для Лаб.2")
                return None
            cmd = [py, "main.py", "lab2", "--template", tpl, "--camera", cam]
        elif lab == 3:
            d = self.lab3_dir_var.get().strip()
            if not d:
                messagebox.showwarning("Папка не выбрана", "Выберите папку шаблонов для Лаб.3")
                return None
            cmd = [py, "main.py", "lab3", "--templates_dir", d, "--camera", cam]
        return cmd

    def start_process(self, cmd):
        # Start subprocess and keep reference
        try:
            self.log(f"Запускаю: {' '.join(shlex.quote(p) for p in cmd)}")
            with self.proc_lock:
                if self.proc is not None:
                    self.log("Процесс уже запущен. Остановите его перед запуском нового.")
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
        cmd = self.build_command(2)
        if cmd:
            self._run_in_thread(cmd)

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


if __name__ == '__main__':
    app = LabLauncher()
    app.mainloop()
