import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import librosa
import numpy as np
import pretty_midi
import os
from pathlib import Path
from scipy import signal
from scipy.ndimage import median_filter, gaussian_filter1d
import tempfile
import noisereduce as nr
import crepe


class AdvancedVocalToMIDIConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("ᗰᑭᗪI")
        self.root.geometry("750x700")
        self.root.resizable(True, True)

        # Переменные
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.progress = tk.DoubleVar()
        self.status = tk.StringVar(value="Готов к работе")
        self.is_processing = False

        # Параметры алгоритма
        self.pitch_detection_method = tk.StringVar(value="crepe")
        self.use_noise_reduction = tk.BooleanVar(value=True)
        self.use_harmonic_percussive = tk.BooleanVar(value=True)
        self.min_note_duration = tk.DoubleVar(value=0.08)
        self.sensitivity = tk.DoubleVar(value=0.7)
        self.volume_threshold = tk.DoubleVar(value=0.02)

        self.setup_ui()

    def setup_ui(self):
        # Стиль
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 10))
        style.configure("Title.TLabel", background="#f0f0f0", font=("Arial", 14, "bold"))
        style.configure("TButton", font=("Arial", 10))
        style.configure("Status.TLabel", background="#e0e0e0", font=("Arial", 9))
        style.configure("TCheckbutton", background="#f0f0f0")
        style.configure("TRadiobutton", background="#f0f0f0")

        # Главный фрейм
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        title_label = ttk.Label(main_frame, text="ᗰᑭᗪI",
                                style="Title.TLabel")
        title_label.pack(pady=(0, 20))

        # Фрейм выбора файла
        file_frame = ttk.LabelFrame(main_frame, text="Файлы", padding="15")
        file_frame.pack(fill=tk.X, pady=(0, 15))

        # Входной файл
        ttk.Label(file_frame, text="Входной аудио файл:").grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        input_entry = ttk.Entry(file_frame, textvariable=self.input_file, width=50)
        input_entry.grid(row=0, column=1, padx=(10, 5), pady=(0, 10), sticky=tk.EW)
        ttk.Button(file_frame, text="Обзор...", command=self.browse_input_file).grid(row=0, column=2, padx=(5, 0),
                                                                                     pady=(0, 10))

        # Выходной файл
        ttk.Label(file_frame, text="Выходной MIDI файл:").grid(row=1, column=0, sticky=tk.W)
        output_entry = ttk.Entry(file_frame, textvariable=self.output_file, width=50)
        output_entry.grid(row=1, column=1, padx=(10, 5), sticky=tk.EW)
        ttk.Button(file_frame, text="Обзор...", command=self.browse_output_file).grid(row=1, column=2, padx=(5, 0))

        file_frame.columnconfigure(1, weight=1)

        # Фрейм метода определения высоты тона
        method_frame = ttk.LabelFrame(main_frame, text="Метод определения высоты тона", padding="15")
        method_frame.pack(fill=tk.X, pady=(0, 15))

        # Радиокнопки для выбора метода
        ttk.Radiobutton(method_frame, text="CREPE (рекомендуется - высокая точность)",
                        variable=self.pitch_detection_method, value="crepe").grid(row=0, column=0, sticky=tk.W,
                                                                                  pady=(0, 5))
        ttk.Radiobutton(method_frame, text="PYIN (хорошо для чистых записей)",
                        variable=self.pitch_detection_method, value="pyin").grid(row=1, column=0, sticky=tk.W,
                                                                                 pady=(0, 5))
        ttk.Radiobutton(method_frame, text="Комбинированный метод (CREPE + PYIN)",
                        variable=self.pitch_detection_method, value="combined").grid(row=2, column=0, sticky=tk.W)

        # Фрейм основных настроек
        basic_settings_frame = ttk.LabelFrame(main_frame, text="Основные настройки", padding="15")
        basic_settings_frame.pack(fill=tk.X, pady=(0, 15))

        # Диапазон нот
        ttk.Label(basic_settings_frame, text="Минимальная нота:").grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        self.min_note_combo = ttk.Combobox(basic_settings_frame, values=[
            "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G#2", "A2", "A#2", "B2",
            "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
            "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4"
        ], state="readonly", width=8)
        self.min_note_combo.set("C3")
        self.min_note_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=(0, 10))

        ttk.Label(basic_settings_frame, text="Максимальная нота:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0),
                                                                        pady=(0, 10))
        self.max_note_combo = ttk.Combobox(basic_settings_frame, values=[
            "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
            "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5",
            "C6", "C#6", "D6", "D#6", "E6", "F6", "F#6", "G6", "G#6", "A6", "A#6", "B6",
            "C7"
        ], state="readonly", width=8)
        self.max_note_combo.set("C6")
        self.max_note_combo.grid(row=0, column=3, sticky=tk.W, padx=(10, 0), pady=(0, 10))

        # Инструмент MIDI
        ttk.Label(basic_settings_frame, text="Инструмент MIDI:").grid(row=1, column=0, sticky=tk.W)
        self.instrument_combo = ttk.Combobox(basic_settings_frame, values=[
            "Acoustic Grand Piano (0)",
            "Bright Acoustic Piano (1)",
            "Electric Grand Piano (2)",
            "Honky-tonk Piano (3)",
            "Electric Piano 1 (4)",
            "Electric Piano 2 (5)",
            "Harpsichord (6)",
            "Clavinet (7)",
            "Celesta (8)",
            "Glockenspiel (9)",
            "Music Box (10)",
            "Vibraphone (11)",
            "Marimba (12)",
            "Xylophone (13)",
            "Tubular Bells (14)",
            "Dulcimer (15)",
            "Violin (40)",
            "Viola (41)",
            "Cello (42)",
            "Contrabass (43)",
            "Trumpet (56)",
            "Trombone (57)",
            "Tuba (58)",
            "Saxophone (66)",
            "Oboe (68)",
            "English Horn (69)",
            "Bassoon (70)",
            "Clarinet (71)",
            "Piccolo (72)",
            "Flute (73)",
            "Recorder (74)",
            "Pan Flute (75)",
            "Blown Bottle (76)",
            "Shakuhachi (77)",
            "Whistle (78)",
            "Ocarina (79)",
            "Voice Oohs (53)",
            "Voice Aahs (54)",
            "Synth Voice (85)"
        ], state="readonly", width=25)
        self.instrument_combo.set("Acoustic Grand Piano (0)")
        self.instrument_combo.grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=(10, 0))

        basic_settings_frame.columnconfigure(1, weight=1)
        basic_settings_frame.columnconfigure(3, weight=1)

        # Фрейм продвинутых настроек
        advanced_settings_frame = ttk.LabelFrame(main_frame, text="Продвинутые настройки обработки", padding="15")
        advanced_settings_frame.pack(fill=tk.X, pady=(0, 15))

        # Чекбоксы алгоритмов
        ttk.Checkbutton(advanced_settings_frame, text="Подавление шума",
                        variable=self.use_noise_reduction).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        ttk.Checkbutton(advanced_settings_frame, text="Разделение гармоник/перкуссии",
                        variable=self.use_harmonic_percussive).grid(row=0, column=1, sticky=tk.W, pady=(0, 5))

        # Чувствительность
        ttk.Label(advanced_settings_frame, text="Чувствительность:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        sensitivity_scale = ttk.Scale(advanced_settings_frame, from_=0.1, to=1.0,
                                      variable=self.sensitivity, orient=tk.HORIZONTAL)
        sensitivity_scale.grid(row=1, column=1, sticky=tk.EW, padx=(10, 0), pady=(10, 0))

        # Порог громкости
        ttk.Label(advanced_settings_frame, text="Порог громкости:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        volume_scale = ttk.Scale(advanced_settings_frame, from_=0.001, to=0.1,
                                 variable=self.volume_threshold, orient=tk.HORIZONTAL)
        volume_scale.grid(row=2, column=1, sticky=tk.EW, padx=(10, 0), pady=(10, 0))

        # Минимальная длительность ноты
        ttk.Label(advanced_settings_frame, text="Мин. длительность ноты (сек):").grid(row=3, column=0, sticky=tk.W,
                                                                                      pady=(10, 0))
        min_duration_spin = ttk.Spinbox(advanced_settings_frame, from_=0.05, to=0.5,
                                        increment=0.05, textvariable=self.min_note_duration, width=8)
        min_duration_spin.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))

        advanced_settings_frame.columnconfigure(1, weight=1)

        # Фрейм прогресса
        progress_frame = ttk.LabelFrame(main_frame, text="Процесс конвертации", padding="15")
        progress_frame.pack(fill=tk.X, pady=(0, 15))

        # Прогресс-бар
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Статус
        status_label = ttk.Label(progress_frame, textvariable=self.status, style="Status.TLabel")
        status_label.pack(fill=tk.X, ipady=5)

        # Фрейм кнопок
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        # Кнопки
        self.convert_button = ttk.Button(button_frame, text="Конвертировать", command=self.start_conversion)
        self.convert_button.pack(side=tk.RIGHT, padx=(10, 0))

        ttk.Button(button_frame, text="Очистить", command=self.clear_all).pack(side=tk.RIGHT)

        # Информация
        info_frame = ttk.LabelFrame(main_frame, text="Улучшенные алгоритмы", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        info_text = tk.Text(info_frame, height=8, wrap=tk.WORD, font=("Arial", 9), bg="#f9f9f9")
        info_text.pack(fill=tk.BOTH, expand=True)

        tips = """
СОВЕТЫ:

• CREPE: Лучшая точность, но требует больше ресурсов
• PYIN: Быстрее, хорошо для чистых записей
• Комбинированный: Наиболее устойчивый к артефактам
• Увеличьте порог громкости для записей с шумом
• Для мужских голосов: C2-C5, для женских: A3-C6
• Минимальная длительность 0.08-0.1с для естественных результатов
"""
        info_text.insert(tk.END, tips)
        info_text.config(state=tk.DISABLED)

    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[("Audio files", "*.mp3 *.wav *.ogg *.m4a *.flac"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            # Автоматически генерируем имя выходного файла
            if not self.output_file.get():
                output_path = Path(filename).with_suffix('.mid')
                self.output_file.set(str(output_path))

    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            title="Сохранить MIDI файл как",
            defaultextension=".mid",
            filetypes=[("MIDI files", "*.mid"), ("All files", "*.*")]
        )
        if filename:
            self.output_file.set(filename)

    def clear_all(self):
        self.input_file.set("")
        self.output_file.set("")
        self.progress.set(0)
        self.status.set("Готов к работе")

    def start_conversion(self):
        if not self.input_file.get():
            messagebox.showerror("Ошибка", "Выберите входной аудио файл")
            return

        if not self.output_file.get():
            messagebox.showerror("Ошибка", "Укажите выходной MIDI файл")
            return

        self.is_processing = True
        self.convert_button.config(state=tk.DISABLED)
        self.progress.set(0)
        self.status.set("Начинаю обработку...")

        # Запуск конвертации в отдельном потоке
        thread = threading.Thread(target=self.convert_audio_to_midi)
        thread.daemon = True
        thread.start()

        # Мониторинг прогресса
        self.monitor_conversion(thread)

    def monitor_conversion(self, thread):
        if thread.is_alive():
            self.root.after(100, lambda: self.monitor_conversion(thread))
        else:
            self.convert_button.config(state=tk.NORMAL)
            self.is_processing = False

    def update_progress(self, value, message):
        self.progress.set(value)
        self.status.set(message)
        self.root.update_idletasks()

    def preprocess_audio(self, y, sr):
        """Предварительная обработка аудио"""
        self.update_progress(10, "Предварительная обработка аудио...")

        # Нормализация
        y = librosa.util.normalize(y)

        # Подавление шума
        if self.use_noise_reduction.get():
            try:
                # Используем первую секунду как образец шума
                noise_sample = y[:min(sr, len(y))]
                y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample, prop_decrease=0.75)
            except Exception as e:
                print(f"Шумоподавление не удалось: {e}")

        # Разделение гармоник и перкуссии
        if self.use_harmonic_percussive.get():
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                y = y_harmonic  # Используем гармоническую составляющую
            except Exception as e:
                print(f"Разделение HPSS не удалось: {e}")

        return y

    def detect_pitch_crepe(self, y, sr):
        """Определение высоты тона с помощью CREPE"""
        self.update_progress(30, "Анализ CREPE (нейросеть)...")

        try:
            # Используем CREPE для определения высоты тона
            time, frequency, confidence, activation = crepe.predict(
                y, sr,
                viterbi=True,  # Сглаживание Витерби
                model_capacity="full"  # Полная модель для лучшей точности
            )

            # Фильтруем по уверенности
            confidence_threshold = 0.4 + 0.4 * self.sensitivity.get()
            valid_mask = confidence > confidence_threshold
            frequency[~valid_mask] = 0

            # Конвертируем в MIDI ноты
            midi_notes = np.zeros_like(frequency)
            valid_freqs = frequency > 0
            midi_notes[valid_freqs] = librosa.hz_to_midi(frequency[valid_freqs])

            return midi_notes, time, confidence

        except Exception as e:
            print(f"CREPE не удался: {e}")
            return None, None, None

    def detect_pitch_pyin(self, y, sr):
        """Определение высоты тона с помощью PYIN"""
        self.update_progress(30, "Анализ PYIN...")

        fmin = librosa.note_to_hz(self.min_note_combo.get())
        fmax = librosa.note_to_hz(self.max_note_combo.get())

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=fmin * 0.9,  # Немного расширяем диапазон
            fmax=fmax * 1.1,
            sr=sr,
            frame_length=2048,
            hop_length=512,
            fill_na=0
        )

        # Конвертируем в MIDI ноты
        midi_notes = np.zeros_like(f0)
        valid_mask = (voiced_flag) & (~np.isnan(f0)) & (f0 > 0)
        midi_notes[valid_mask] = librosa.hz_to_midi(f0[valid_mask])

        return midi_notes, librosa.times_like(f0, sr=sr, hop_length=512), voiced_probs

    def combined_pitch_detection(self, y, sr):
        """Комбинированный метод CREPE + PYIN"""
        self.update_progress(25, "Комбинированный анализ...")

        # Получаем результаты обоих методов
        crepe_notes, crepe_times, crepe_conf = self.detect_pitch_crepe(y, sr)
        pyin_notes, pyin_times, pyin_conf = self.detect_pitch_pyin(y, sr)

        if crepe_notes is None:
            return pyin_notes, pyin_times, pyin_conf

        # Интерполируем CREPE результаты к временной сетке PYIN
        if len(crepe_notes) != len(pyin_notes):
            from scipy.interpolate import interp1d
            try:
                f = interp1d(crepe_times, crepe_notes, kind='linear',
                             bounds_error=False, fill_value=0)
                crepe_notes_interp = f(pyin_times)
            except:
                crepe_notes_interp = np.zeros_like(pyin_notes)
        else:
            crepe_notes_interp = crepe_notes

        # Объединяем результаты
        combined_notes = np.zeros_like(pyin_notes)

        for i in range(len(combined_notes)):
            crepe_note = crepe_notes_interp[i]
            pyin_note = pyin_notes[i]

            if crepe_note > 0 and pyin_note > 0:
                # Оба метода дали результат - берем среднее
                combined_notes[i] = (crepe_note + pyin_note) / 2
            elif crepe_note > 0:
                # Только CREPE
                combined_notes[i] = crepe_note
            elif pyin_note > 0:
                # Только PYIN
                combined_notes[i] = pyin_note

        return combined_notes, pyin_times, np.maximum(crepe_conf if crepe_conf is not None else 0,
                                                      pyin_conf if pyin_conf is not None else 0)

    def advanced_note_processing(self, midi_notes, times, sr, y):
        """Продвинутая обработка нот"""
        self.update_progress(60, "Обработка и сглаживание нот...")

        if len(midi_notes) == 0:
            return []

        # Вычисляем энергию сигнала для фильтрации тихих участков
        hop_length = 512
        frame_length = 2048
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Интерполируем RMS к временной сетке нот
        if len(rms) != len(midi_notes):
            rms_times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
            rms_interp = np.interp(times, rms_times, rms)
        else:
            rms_interp = rms

        # Фильтрация по громкости
        volume_threshold = self.volume_threshold.get()
        loud_enough = rms_interp > volume_threshold

        # Медианная фильтрация для удаления выбросов
        notes_clean = median_filter(midi_notes, size=5)

        # Гауссово сглаживание
        notes_smooth = gaussian_filter1d(notes_clean, sigma=2)

        # Квантование к полутонам
        notes_quantized = np.round(notes_smooth)

        # Применяем фильтр громкости
        notes_quantized[~loud_enough] = 0

        # Создаем ноты
        min_duration = self.min_note_duration.get()
        notes = []
        current_note = None
        note_start = 0
        note_pitches = []

        for i, (note_num, time_val) in enumerate(zip(notes_quantized, times)):
            if note_num > 0:
                if current_note is None:
                    current_note = note_num
                    note_start = time_val
                    note_pitches = [note_num]
                elif note_num != current_note:
                    # Проверяем, является ли изменение значительным
                    if abs(note_num - current_note) < 2 and len(note_pitches) < 10:
                        # Небольшое изменение - продолжаем ноту
                        note_pitches.append(note_num)
                        current_note = int(np.median(note_pitches))
                    else:
                        # Значительное изменение - начинаем новую ноту
                        note_end = time_val
                        duration = note_end - note_start

                        if duration >= min_duration:
                            final_pitch = int(np.median(note_pitches))
                            notes.append((final_pitch, note_start, note_end))

                        current_note = note_num
                        note_start = time_val
                        note_pitches = [note_num]
                else:
                    note_pitches.append(note_num)
            else:
                if current_note is not None:
                    note_end = time_val
                    duration = note_end - note_start

                    if duration >= min_duration:
                        final_pitch = int(np.median(note_pitches))
                        notes.append((final_pitch, note_start, note_end))

                    current_note = None
                    note_pitches = []

        # Добавляем последнюю ноту
        if current_note is not None:
            note_end = times[-1]
            duration = note_end - note_start

            if duration >= min_duration:
                final_pitch = int(np.median(note_pitches))
                notes.append((final_pitch, note_start, note_end))

        return notes

    def convert_audio_to_midi(self):
        try:
            self.update_progress(5, "Загрузка аудио...")

            # Загрузка аудиофайла
            y, sr = librosa.load(self.input_file.get(), sr=22050, mono=True)

            # Предварительная обработка
            y_processed = self.preprocess_audio(y, sr)

            self.update_progress(20, "Определение высоты тона...")

            # Выбор метода определения высоты тона
            method = self.pitch_detection_method.get()

            if method == "crepe":
                midi_notes, times, confidence = self.detect_pitch_crepe(y_processed, sr)
                if midi_notes is None:
                    self.update_progress(20, "CREPE не сработал, использую PYIN...")
                    midi_notes, times, confidence = self.detect_pitch_pyin(y_processed, sr)
            elif method == "pyin":
                midi_notes, times, confidence = self.detect_pitch_pyin(y_processed, sr)
            else:  # combined
                midi_notes, times, confidence = self.combined_pitch_detection(y_processed, sr)

            if midi_notes is None or len(midi_notes) == 0:
                raise Exception("Не удалось определить высоту тона в аудио")

            self.update_progress(50, "Обработка нот...")

            # Продвинутая обработка нот
            notes = self.advanced_note_processing(midi_notes, times, sr, y_processed)

            if len(notes) == 0:
                raise Exception("Не удалось извлечь ноты из аудио")

            self.update_progress(80, "Создание MIDI...")

            # Создание объекта MIDI
            midi_data = pretty_midi.PrettyMIDI()
            instrument_program = int(self.instrument_combo.get().split("(")[1].split(")")[0])
            instrument = pretty_midi.Instrument(program=instrument_program)

            # Добавление нот в MIDI
            min_midi = librosa.note_to_midi(self.min_note_combo.get())
            max_midi = librosa.note_to_midi(self.max_note_combo.get())

            for pitch, start, end in notes:
                if min_midi <= pitch <= max_midi:
                    # Вычисляем velocity на основе громкости
                    velocity = 60 + int(40 * self.sensitivity.get())
                    note = pretty_midi.Note(
                        velocity=min(velocity, 127),
                        pitch=int(pitch),
                        start=max(0, start - 0.02),  # Небольшая коррекция времени
                        end=end
                    )
                    instrument.notes.append(note)

            midi_data.instruments.append(instrument)

            self.update_progress(95, "Сохранение файла...")

            # Сохраняем MIDI-файл
            midi_data.write(self.output_file.get())

            self.update_progress(100, "Конвертация завершена!")
            messagebox.showinfo("Успех",
                                f"MIDI-файл успешно сохранен:\n{self.output_file.get()}\n\nИзвлечено нот: {len(notes)}")

        except Exception as e:
            self.update_progress(0, f"Ошибка: {str(e)}")
            messagebox.showerror("Ошибка", f"Произошла ошибка при конвертации:\n{str(e)}")


def main():
    root = tk.Tk()
    app = AdvancedVocalToMIDIConverter(root)
    root.mainloop()


if __name__ == "__main__":
    main()