"""
Airport Security Detection - Desktop UI

A standalone desktop application for detecting prohibited items in X-ray baggage images.

REQUIREMENTS:
- Python 3.8+
- ultralytics
- pillow
- tkinter (comes with Python)

INSTALLATION:
pip install ultralytics pillow

USAGE:
python airport_security_ui.py

Make sure 'best.pt' model file is in the same directory or update MODEL_PATH
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
import threading

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed!")
    print("Run: pip install ultralytics")
    exit(1)


class AirportSecurityUI:
    """Desktop UI for prohibited item detection in X-ray images"""

    def __init__(self, root):
        self.root = root
        self.root.title("Airport Security Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2C3E50')

        # Configuration
        self.MODEL_PATH = "best.pt"
        self.CLASS_NAMES = [
            "Folding_Knife",
            "Straight_Knife",
            "Scissor",
            "Utility_Knife",
            "Multi-tool_Knife"
        ]

        # State variables
        self.model = None
        self.current_image = None
        self.current_image_path = None
        self.result_image = None

        # Initialize UI
        self.setup_ui()

        # Load model
        self.load_model()

    def setup_ui(self):
        """Create the user interface"""

        # ============================================================
        # HEADER
        # ============================================================
        header_frame = tk.Frame(self.root, bg='#34495E', height=80)
        header_frame.pack(fill='x', padx=10, pady=10)

        title_label = tk.Label(
            header_frame,
            text="🔍 Airport Security Detection System",
            font=('Arial', 24, 'bold'),
            bg='#34495E',
            fg='white'
        )
        title_label.pack(pady=20)

        subtitle_label = tk.Label(
            header_frame,
            text="YOLOv8 Prohibited Item Detection | mAP: 83.9%",
            font=('Arial', 11),
            bg='#34495E',
            fg='#BDC3C7'
        )
        subtitle_label.pack()

        # ============================================================
        # MAIN CONTENT AREA
        # ============================================================
        content_frame = tk.Frame(self.root, bg='#2C3E50')
        content_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Left Panel - Original Image
        left_panel = tk.Frame(content_frame, bg='#34495E', relief='raised', borderwidth=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))

        left_title = tk.Label(
            left_panel,
            text="Original X-ray Image",
            font=('Arial', 14, 'bold'),
            bg='#34495E',
            fg='white'
        )
        left_title.pack(pady=10)

        self.original_image_label = tk.Label(
            left_panel,
            bg='#2C3E50',
            text="No image loaded\n\nClick 'Upload Image' to start",
            font=('Arial', 12),
            fg='#95A5A6'
        )
        self.original_image_label.pack(padx=20, pady=20, fill='both', expand=True)

        # Right Panel - Detection Results
        right_panel = tk.Frame(content_frame, bg='#34495E', relief='raised', borderwidth=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))

        right_title = tk.Label(
            right_panel,
            text="Detection Results",
            font=('Arial', 14, 'bold'),
            bg='#34495E',
            fg='white'
        )
        right_title.pack(pady=10)

        self.result_image_label = tk.Label(
            right_panel,
            bg='#2C3E50',
            text="Detection results will appear here",
            font=('Arial', 12),
            fg='#95A5A6'
        )
        self.result_image_label.pack(padx=20, pady=20, fill='both', expand=True)

        # ============================================================
        # CONTROL PANEL
        # ============================================================
        control_frame = tk.Frame(self.root, bg='#34495E', height=150)
        control_frame.pack(fill='x', padx=10, pady=10)

        # Buttons Row
        button_frame = tk.Frame(control_frame, bg='#34495E')
        button_frame.pack(pady=15)

        self.upload_btn = tk.Button(
            button_frame, text="📁 Upload Image", command=self.upload_image,
            font=('Arial', 12, 'bold'), bg='#3498DB', fg='white',
            activebackground='#2980B9', activeforeground='white',
            cursor='hand2', width=15, height=2
        )
        self.upload_btn.pack(side='left', padx=10)

        self.detect_btn = tk.Button(
            button_frame, text="🔍 Detect Items", command=self.detect_items,
            font=('Arial', 12, 'bold'), bg='#E74C3C', fg='white',
            activebackground='#C0392B', activeforeground='white',
            cursor='hand2', width=15, height=2, state='disabled'
        )
        self.detect_btn.pack(side='left', padx=10)

        self.clear_btn = tk.Button(
            button_frame, text="🗑️ Clear", command=self.clear_all,
            font=('Arial', 12, 'bold'), bg='#95A5A6', fg='white',
            activebackground='#7F8C8D', activeforeground='white',
            cursor='hand2', width=15, height=2
        )
        self.clear_btn.pack(side='left', padx=10)

        self.save_btn = tk.Button(
            button_frame, text="💾 Save Results", command=self.save_results,
            font=('Arial', 12, 'bold'), bg='#27AE60', fg='white',
            activebackground='#229954', activeforeground='white',
            cursor='hand2', width=15, height=2, state='disabled'
        )
        self.save_btn.pack(side='left', padx=10)

        # ============================================================
        # STATUS PANEL
        # ============================================================
        status_frame = tk.Frame(control_frame, bg='#34495E')
        status_frame.pack(fill='x', pady=10)

        self.status_label = tk.Label(
            status_frame,
            text="Status: Ready | Model: YOLOv8n | Waiting for image...",
            font=('Arial', 10), bg='#34495E', fg='#BDC3C7', anchor='w'
        )
        self.status_label.pack(fill='x', padx=20)

        self.results_text = tk.Text(
            control_frame, height=4, font=('Courier', 10),
            bg='#2C3E50', fg='white', relief='sunken', borderwidth=2
        )
        self.results_text.pack(fill='x', padx=20, pady=10)
        self.results_text.insert('1.0', 'Detection results will appear here...')
        self.results_text.config(state='disabled')

    def load_model(self):
        """Load YOLOv8 model"""
        try:
            self.update_status("Loading model...")
            self.model = YOLO(self.MODEL_PATH)
            self.update_status("Status: Model loaded successfully ✓ | Ready to detect")
            messagebox.showinfo(
                "Success",
                f"Model loaded successfully!\n\nModel: YOLOv8n\nClasses: {len(self.CLASS_NAMES)}\nPath: {self.MODEL_PATH}"
            )
        except Exception as e:
            self.update_status("Status: ERROR - Model not loaded ✗")
            messagebox.showerror("Model Error", f"Failed to load model!\n\nError: {str(e)}\n\nMake sure 'best.pt' is in the same folder.")

    def upload_image(self):
        """Open file dialog and load image"""
        file_path = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            self.current_image_path = file_path
            image = Image.open(file_path)
            self.current_image = image
            self.display_image(image, self.original_image_label)
            self.detect_btn.config(state='normal')
            self.update_status(f"Status: Image loaded ✓ | File: {Path(file_path).name} | Click 'Detect Items'")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image!\n\n{str(e)}")

    def detect_items(self):
        """Run detection on current image"""
        if self.current_image is None or self.model is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        self.update_status("Status: Running detection... Please wait")
        self.detect_btn.config(state='disabled')
        thread = threading.Thread(target=self._run_detection)
        thread.daemon = True
        thread.start()

    def _run_detection(self):
        """Actual detection logic (runs in thread)"""
        try:
            results = self.model.predict(source=self.current_image_path, conf=0.25, iou=0.45, verbose=False)
            result = results[0]
            annotated_img = result.plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            annotated_img = Image.fromarray(annotated_img)
            self.result_image = annotated_img

            detections = []
            if len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.CLASS_NAMES[class_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({'class': class_name, 'confidence': confidence, 'box': (int(x1), int(y1), int(x2), int(y2))})

            self.root.after(0, self._update_detection_ui, annotated_img, detections)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Detection Error", f"Error during detection!\n\n{str(e)}"))
            self.root.after(0, lambda: self.detect_btn.config(state='normal'))

    def _update_detection_ui(self, annotated_img, detections):
        """Update UI with detection results"""
        self.display_image(annotated_img, self.result_image_label)
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', 'end')

        if detections:
            msg = f"⚠️  THREAT DETECTED! Found {len(detections)} prohibited item(s)\n" + "=" * 60 + "\n\n"
            for i, det in enumerate(detections, 1):
                msg += f"{i}. {det['class']:<20} Confidence: {det['confidence']:.1%}\n"
                msg += f"   Location: ({det['box'][0]}, {det['box'][1]}) to ({det['box'][2]}, {det['box'][3]})\n\n"
            self.results_text.insert('1.0', msg)
            self.results_text.tag_add("threat", "1.0", "1.end")
            self.results_text.tag_config("threat", foreground="#E74C3C", font=('Arial', 11, 'bold'))
            self.update_status(f"Status: ⚠️ THREAT DETECTED | {len(detections)} item(s) found")
        else:
            msg = "✅ CLEAR - No prohibited items detected\n" + "=" * 60 + "\n\nThis baggage appears safe for clearance.\nNo knives, scissors, or other prohibited items found."
            self.results_text.insert('1.0', msg)
            self.results_text.tag_add("safe", "1.0", "1.end")
            self.results_text.tag_config("safe", foreground="#27AE60", font=('Arial', 11, 'bold'))
            self.update_status("Status: ✅ CLEAR | No threats detected")

        self.results_text.config(state='disabled')
        self.detect_btn.config(state='normal')
        self.save_btn.config(state='normal')

    def display_image(self, image, label):
        """Display PIL image in tkinter label"""
        label.update()
        max_width = label.winfo_width() - 40
        max_height = label.winfo_height() - 40
        if max_width <= 1 or max_height <= 1:
            max_width, max_height = 500, 500
        image_copy = image.copy()
        image_copy.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image_copy)
        label.config(image=photo, text='')
        label.image = photo

    def save_results(self):
        """Save detection results image"""
        if self.result_image is None:
            messagebox.showwarning("Warning", "No results to save!")
            return
        file_path = filedialog.asksaveasfilename(
            title="Save Detection Results", defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.result_image.save(file_path)
                messagebox.showinfo("Success", f"Results saved successfully!\n\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results!\n\n{str(e)}")

    def clear_all(self):
        """Clear all images and results"""
        self.current_image = None
        self.current_image_path = None
        self.result_image = None
        self.original_image_label.config(image='', text="No image loaded\n\nClick 'Upload Image' to start", fg='#95A5A6')
        self.result_image_label.config(image='', text="Detection results will appear here", fg='#95A5A6')
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', 'end')
        self.results_text.insert('1.0', 'Detection results will appear here...')
        self.results_text.config(state='disabled')
        self.detect_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        self.update_status("Status: Ready | Cleared | Waiting for image...")

    def update_status(self, message):
        """Update status bar message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()


def main():
    root = tk.Tk()
    app = AirportSecurityUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
