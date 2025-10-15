#!/usr/bin/env python3
"""
Linear Algebra - Image Deblurring Application
A GUI application for image deblurring using MPRNet with Linear Algebra concepts demonstration
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

# เพิ่ม path สำหรับ MPRNet และ utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from MPRNet import MPRNet
from math_explanation import LinearAlgebraExplainer


class LinearAlgebraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Algebra - Image Deblurring Application")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # ตัวแปรสำหรับโมเดล
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.current_image_path = None
        self.processed_image = None
        self.original_image = None
        
        # Mathematical explainer
        self.math_explainer = LinearAlgebraExplainer()
        
        # สร้าง GUI
        self.create_widgets()
        self.load_model()
        
    def create_widgets(self):
        """สร้าง GUI components"""
        # Header
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="Linear Algebra - Image Deblurring", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(header_frame, text="MPRNet with Cosine Similarity Analysis", 
                                 font=('Arial', 12), fg='#ecf0f1', bg='#2c3e50')
        subtitle_label.pack()
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg='#ecf0f1', width=300)
        left_panel.pack(side='left', fill='y', padx=(0, 5))
        left_panel.pack_propagate(False)
        
        # Image selection
        select_frame = tk.LabelFrame(left_panel, text="Image Selection", 
                                   font=('Arial', 12, 'bold'), bg='#ecf0f1')
        select_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(select_frame, text="Browse Image", command=self.browse_image,
                 font=('Arial', 10), bg='#3498db', fg='white').pack(fill='x', padx=5, pady=5)
        
        tk.Button(select_frame, text="Use Sample Image", command=self.use_sample_image,
                 font=('Arial', 10), bg='#27ae60', fg='white').pack(fill='x', padx=5, pady=5)
        
        # Sample image selection
        sample_frame = tk.LabelFrame(left_panel, text="Sample Images (1-300)", 
                                   font=('Arial', 10), bg='#ecf0f1')
        sample_frame.pack(fill='x', padx=10, pady=5)
        
        self.sample_var = tk.StringVar(value="1")
        sample_entry = tk.Entry(sample_frame, textvariable=self.sample_var, font=('Arial', 10))
        sample_entry.pack(fill='x', padx=5, pady=5)
        
        tk.Button(sample_frame, text="Load Sample", command=self.load_sample,
                 font=('Arial', 10), bg='#e74c3c', fg='white').pack(fill='x', padx=5, pady=5)
        
        # Processing controls
        process_frame = tk.LabelFrame(left_panel, text="Processing", 
                                    font=('Arial', 12, 'bold'), bg='#ecf0f1')
        process_frame.pack(fill='x', padx=10, pady=10)
        
        self.process_btn = tk.Button(process_frame, text="Process Image", 
                                   command=self.process_image, font=('Arial', 12, 'bold'),
                                   bg='#9b59b6', fg='white', state='disabled')
        self.process_btn.pack(fill='x', padx=5, pady=5)
        
        # Learning resources (training removed)
        learn_frame = tk.LabelFrame(left_panel, text="Learning Resources", 
                                   font=('Arial', 12, 'bold'), bg='#ecf0f1')
        learn_frame.pack(fill='x', padx=10, pady=10)
        tk.Button(learn_frame, text="Mathematical Concepts", 
                 command=self.show_math_concepts, font=('Arial', 10),
                 bg='#8e44ad', fg='white').pack(fill='x', padx=5, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(process_frame, textvariable=self.status_var, 
                               font=('Arial', 10), bg='#ecf0f1', fg='#7f8c8d')
        status_label.pack(pady=5)
        
        # Results
        results_frame = tk.LabelFrame(left_panel, text="Results", 
                                    font=('Arial', 12, 'bold'), bg='#ecf0f1')
        results_frame.pack(fill='x', padx=10, pady=10)
        
        self.cos_sim_var = tk.StringVar(value="Cosine Similarity: --")
        cos_sim_label = tk.Label(results_frame, textvariable=self.cos_sim_var, 
                               font=('Arial', 10), bg='#ecf0f1', fg='#2c3e50')
        cos_sim_label.pack(pady=5)
        
        # Right panel - Image display
        right_panel = tk.Frame(main_frame, bg='#f0f0f0')
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Image display frame
        self.image_frame = tk.Frame(right_panel, bg='#f0f0f0')
        self.image_frame.pack(fill='both', expand=True)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.patch.set_facecolor('#f0f0f0')
        
        self.ax1.set_title("Original (Blurred)", fontsize=12, fontweight='bold')
        self.ax1.axis('off')
        self.ax2.set_title("Restored (AI Deblur)", fontsize=12, fontweight='bold')
        self.ax2.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.image_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Footer
        footer_frame = tk.Frame(self.root, bg='#34495e', height=40)
        footer_frame.pack(fill='x', padx=10, pady=5)
        footer_frame.pack_propagate(False)
        
        footer_label = tk.Label(footer_frame, text="Linear Algebra Applications: CNN Architecture • Matrix Convolution • Vector Similarity", 
                               font=('Arial', 10), fg='white', bg='#34495e')
        footer_label.pack(expand=True)
        
    def load_model(self):
        """โหลดโมเดล MPRNet"""
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'model_deblurring.pth')
            
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return False
                
            self.model = MPRNet().to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
                
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            self.status_var.set("Model loaded successfully")
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set("Model loading failed")
            return False
    
    def browse_image(self):
        """เลือกภาพจากไฟล์"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def use_sample_image(self):
        """ใช้ภาพตัวอย่าง"""
        if self.current_image_path:
            self.process_image()
        else:
            messagebox.showwarning("Warning", "Please select an image first")
    
    def load_sample(self):
        """โหลดภาพตัวอย่างตามหมายเลข"""
        try:
            sample_num = self.sample_var.get()
            sample_path = os.path.join(os.path.dirname(__file__), 'datasets', 'blurred_images', f"{sample_num}.jpg")
            
            if not os.path.exists(sample_path):
                # ลองหาไฟล์ที่มีนามสกุลอื่น
                for ext in ['JPG', 'png', 'PNG']:
                    alt_path = os.path.join(os.path.dirname(__file__), 'datasets', 'blurred_images', f"{sample_num}.{ext}")
                    if os.path.exists(alt_path):
                        sample_path = alt_path
                        break
                else:
                    messagebox.showerror("Error", f"Sample image {sample_num} not found")
                    return
            
            self.load_image(sample_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample: {str(e)}")
    
    def load_image(self, image_path):
        """โหลดและแสดงภาพ"""
        try:
            self.current_image_path = image_path
            self.original_image = cv2.imread(image_path)
            
            if self.original_image is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            
            # แปลงสีและปรับขนาด
            self.original_image = cv2.resize(self.original_image, (512, 512))
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # แสดงภาพต้นฉบับ
            self.display_images()
            
            # เปิดใช้งานปุ่มประมวลผล
            self.process_btn.config(state='normal')
            self.status_var.set("Image loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def process_image(self):
        """ประมวลผลภาพด้วย MPRNet"""
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded")
            return
        
        if self.original_image is None:
            messagebox.showerror("Error", "No image loaded")
            return
        
        # เรียกใช้ใน thread แยกเพื่อไม่ให้ GUI ค้าง
        threading.Thread(target=self._process_image_thread, daemon=True).start()
    
    def _process_image_thread(self):
        """Thread สำหรับประมวลผลภาพ"""
        try:
            self.root.after(0, lambda: self.status_var.set("Processing..."))
            self.root.after(0, lambda: self.process_btn.config(state='disabled'))
            
            # แปลงเป็น tensor
            input_tensor = transforms.ToTensor()(self.original_image).unsqueeze(0).to(self.device)
            
            # ประมวลผลด้วยโมเดล
            with torch.no_grad():
                restored = self.model(input_tensor)
            
            restored = torch.clamp(restored[0], 0, 1)
            self.processed_image = restored.squeeze().permute(1, 2, 0).cpu().numpy()
            
            # คำนวณ cosine similarity
            cos_sim = self.calculate_cosine_similarity()
            
            # อัปเดต GUI
            self.root.after(0, lambda: self.display_images())
            self.root.after(0, lambda: self.cos_sim_var.set(f"Cosine Similarity: {cos_sim:.6f}"))
            self.root.after(0, lambda: self.status_var.set("Processing completed"))
            self.root.after(0, lambda: self.process_btn.config(state='normal'))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Processing failed"))
            self.root.after(0, lambda: self.process_btn.config(state='normal'))
    
    def calculate_cosine_similarity(self):
        """คำนวณ cosine similarity"""
        try:
            original_tensor = torch.tensor(self.original_image.transpose(2, 0, 1)).float().flatten()
            restored_tensor = torch.tensor(self.processed_image.transpose(2, 0, 1)).float().flatten()
            
            cos_sim = F.cosine_similarity(original_tensor, restored_tensor, dim=0)
            return cos_sim.item()
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def display_images(self):
        """แสดงภาพต้นฉบับและผลลัพธ์"""
        self.ax1.clear()
        self.ax1.set_title("Original (Blurred)", fontsize=12, fontweight='bold')
        self.ax1.axis('off')
        
        if self.original_image is not None:
            self.ax1.imshow(self.original_image)
        
        self.ax2.clear()
        self.ax2.set_title("Restored (AI Deblur)", fontsize=12, fontweight='bold')
        self.ax2.axis('off')
        
        if self.processed_image is not None:
            self.ax2.imshow(self.processed_image)
        else:
            self.ax2.text(0.5, 0.5, 'Click "Process Image" to see result', 
                         ha='center', va='center', transform=self.ax2.transAxes,
                         fontsize=12, color='gray')
        
        self.canvas.draw()
    
    def select_blurred_folder(self):
        """เลือกโฟลเดอร์รูปเบลอ"""
        folder_path = filedialog.askdirectory(title="Select Blurred Images Folder")
        if folder_path:
            self.blur_path_var.set(folder_path)
            self.status_var.set(f"Blurred folder selected: {os.path.basename(folder_path)}")
            self.update_pairing_info()
    
    def select_sharp_folder(self):
        """เลือกโฟลเดอร์รูปชัด"""
        folder_path = filedialog.askdirectory(title="Select Sharp Images Folder")
        if folder_path:
            self.sharp_path_var.set(folder_path)
            self.status_var.set(f"Sharp folder selected: {os.path.basename(folder_path)}")
            self.update_pairing_info()

    def update_pairing_info(self):
        """ตรวจสอบและแสดงจำนวนคู่รูปที่จับคู่กันได้โดยอิงชื่อไฟล์"""
        try:
            blur_dir = self.blur_path_var.get()
            sharp_dir = self.sharp_path_var.get()
            if not blur_dir or not sharp_dir or blur_dir == "Not selected" or sharp_dir == "Not selected":
                return
            import glob
            blur_files = []
            for ext in ['*.jpg','*.jpeg','*.png','*.bmp','*.JPG','*.JPEG','*.PNG']:
                blur_files.extend(glob.glob(os.path.join(blur_dir, ext)))
            sharp_files = []
            for ext in ['*.jpg','*.jpeg','*.png','*.bmp','*.JPG','*.JPEG','*.PNG']:
                sharp_files.extend(glob.glob(os.path.join(sharp_dir, ext)))
            sharp_basenames = {os.path.splitext(os.path.basename(f))[0] for f in sharp_files}
            matched = 0
            for bf in blur_files:
                base = os.path.splitext(os.path.basename(bf))[0]
                if base in sharp_basenames:
                    matched += 1
            messagebox.showinfo("Dataset Ready", f"Matched image pairs: {matched}\nBlurred: {len(blur_files)} | Sharp: {len(sharp_files)}\nPairing uses filenames without extensions.")
        except Exception as e:
            messagebox.showerror("Pairing Error", f"Failed to analyze folders: {str(e)}")

    def toggle_advanced_params(self):
        """แสดง/ซ่อนการตั้งค่าขั้นสูง"""
        if self.show_advanced_var.get():
            self.advanced_frame.pack(fill='x', pady=2)
        else:
            self.advanced_frame.pack_forget()
    
    def start_gui_training(self):
        """เริ่มต้นการฝึกโมเดลผ่าน GUI"""
        try:
            # ตรวจสอบว่ามีการเลือกโฟลเดอร์หรือไม่
            if self.blur_path_var.get() == "Not selected":
                messagebox.showerror("Error", "Please select blurred images folder first!")
                return
            
            if self.sharp_path_var.get() == "Not selected":
                messagebox.showerror("Error", "Please select sharp images folder first!")
                return
            
            # กำหนดพารามิเตอร์จากโหมดง่ายหรือขั้นสูง
            if self.show_advanced_var.get():
                try:
                    epochs = int(self.epochs_var.get())
                    batch_size = int(self.batch_size_var.get())
                    learning_rate = float(self.lr_var.get())
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numbers for advanced settings!")
                    return
            else:
                # Map friendly length to epochs and defaults
                length = self.length_var.get()
                if length == 'Quick':
                    epochs = 2
                elif length == 'Standard':
                    epochs = 5
                else:
                    epochs = 10
                batch_size = 4
                learning_rate = 0.0001
            
            # เริ่มการฝึกใน thread แยก
            training_thread = threading.Thread(
                target=self._gui_training_thread, 
                args=(epochs, batch_size, learning_rate),
                daemon=True
            )
            training_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")
    
    def _gui_training_thread(self, epochs, batch_size, learning_rate):
        """Thread สำหรับการฝึกโมเดลผ่าน GUI"""
        try:
            self.root.after(0, lambda: self.status_var.set("Starting training..."))
            
            # Import training modules
            sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
            from gui_trainer import MPRNetTrainer
            
            # สร้าง trainer
            trainer = MPRNetTrainer()
            
            # ตั้งค่า parameters
            trainer.setup_training(
                blurred_folder=self.blur_path_var.get(),
                sharp_folder=self.sharp_path_var.get(),
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                use_gpu=self.use_gpu_var.get()
            )
            
            # เริ่มการฝึก
            self.root.after(0, lambda: self.status_var.set("Training in progress..."))
            
            # สร้างหน้าต่างแสดงความคืบหน้า
            self.create_training_progress_window()
            
            # ฝึกโมเดล
            model_path = trainer.train_with_progress_callback(self.update_training_progress)
            
            self.root.after(0, lambda: self.status_var.set("Training completed!"))
            self.root.after(0, lambda: messagebox.showinfo("Training Complete", 
                                f"Model training completed successfully!\nSaved to: {model_path}"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Training Error", f"Training failed: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Training failed"))
    
    def create_training_progress_window(self):
        """สร้างหน้าต่างแสดงความคืบหน้าในการฝึก"""
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Training Progress")
        self.progress_window.geometry("600x400")
        self.progress_window.configure(bg='#f0f0f0')
        
        # Title
        title_label = tk.Label(self.progress_window, text="Model Training Progress", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_window, variable=self.progress_var, 
                                          maximum=100, length=500)
        self.progress_bar.pack(pady=10)
        
        # Progress text
        self.progress_text = tk.StringVar(value="Preparing training...")
        progress_label = tk.Label(self.progress_window, textvariable=self.progress_text,
                                font=('Arial', 10), bg='#f0f0f0')
        progress_label.pack(pady=5)
        
        # Epoch info
        self.epoch_text = tk.StringVar(value="Epoch: 0/0")
        epoch_label = tk.Label(self.progress_window, textvariable=self.epoch_text,
                              font=('Arial', 10), bg='#f0f0f0')
        epoch_label.pack(pady=5)
        
        # Loss info
        self.loss_text = tk.StringVar(value="Loss: --")
        loss_label = tk.Label(self.progress_window, textvariable=self.loss_text,
                             font=('Arial', 10), bg='#f0f0f0')
        loss_label.pack(pady=5)
        
        # Mathematical concepts
        concepts_frame = tk.LabelFrame(self.progress_window, text="Mathematical Concepts in Action", 
                                     font=('Arial', 12, 'bold'), bg='#f0f0f0')
        concepts_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.concepts_text = tk.Text(concepts_frame, wrap='word', font=('Courier', 9), 
                                   bg='white', fg='black', height=10)
        self.concepts_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Initial mathematical concepts
        initial_concepts = """
MATHEMATICAL CONCEPTS BEING APPLIED:

1. MATRIX OPERATIONS:
   • Convolution: y[m,n] = Σ Σ x[i,j] * k[m-i, n-j]
   • Matrix Multiplication: C_ij = Σ(A_ik * B_kj)
   • Feature Extraction: Applying learned filters

2. VECTOR OPERATIONS:
   • Loss Computation: L = ||y_true - y_pred||²₂ / n
   • Gradient Calculation: ∂L/∂θ = ∂L/∂y * ∂y/∂θ
   • Parameter Updates: θ = θ - α∇L(θ)

3. OPTIMIZATION:
   • Adam Optimizer with momentum and adaptive learning rate
   • Backpropagation using chain rule
   • Gradient descent in parameter space

Training will demonstrate these concepts in real-time...
        """
        self.concepts_text.insert('1.0', initial_concepts)
        self.concepts_text.config(state='disabled')
    
    def update_training_progress(self, epoch, total_epochs, batch, total_batches, loss, lr):
        """อัปเดตความคืบหน้าในการฝึก"""
        try:
            # คำนวณเปอร์เซ็นต์
            epoch_progress = (epoch - 1) / total_epochs * 100
            batch_progress = batch / total_batches * (100 / total_epochs)
            total_progress = epoch_progress + batch_progress
            
            # อัปเดต GUI
            self.root.after(0, lambda: self.progress_var.set(total_progress))
            self.root.after(0, lambda: self.progress_text.set(
                f"Epoch {epoch}/{total_epochs} - Batch {batch}/{total_batches}"))
            self.root.after(0, lambda: self.epoch_text.set(
                f"Epoch: {epoch}/{total_epochs} | Learning Rate: {lr:.6f}"))
            self.root.after(0, lambda: self.loss_text.set(f"Loss: {loss:.6f}"))
            
            # อัปเดต mathematical concepts
            concepts_update = f"""
CURRENT TRAINING STATUS:

Epoch {epoch}/{total_epochs} - Batch {batch}/{total_batches}

MATHEMATICAL OPERATIONS IN PROGRESS:

1. FORWARD PASS:
   • Input: Blurred image tensor (batch_size, 3, H, W)
   • Convolution: Applying learned filters to extract features
   • Activation: ReLU(x) = max(0, x) for non-linearity
   • Output: Restored image prediction

2. LOSS COMPUTATION:
   • MSE Loss: L = (1/n) * Σ(y_true - y_pred)²
   • Current Loss: {loss:.6f}
   • Vector Norm: ||y_true - y_pred||²₂

3. BACKPROPAGATION:
   • Gradient Computation: ∂L/∂θ using chain rule
   • Learning Rate: {lr:.6f}
   • Adam Update: θ = θ - α * m_t / (√v_t + ε)

4. OPTIMIZATION:
   • Parameter Update: θ_{epoch} = θ_{epoch-1} - α∇L(θ_{epoch-1})
   • Momentum: m_t = β₁ m_{{t-1}} + (1-β₁) g_t
   • Adaptive LR: v_t = β₂ v_{{t-1}} + (1-β₂) g_t²

GPU ACCELERATION: {'Enabled' if self.use_gpu_var.get() else 'Disabled'}
            """
            
            self.root.after(0, lambda: self.update_concepts_text(concepts_update))
            
        except Exception as e:
            print(f"Error updating progress: {e}")
    
    def update_concepts_text(self, text):
        """อัปเดตข้อความ mathematical concepts"""
        self.concepts_text.config(state='normal')
        self.concepts_text.delete('1.0', 'end')
        self.concepts_text.insert('1.0', text)
        self.concepts_text.config(state='disabled')
        self.concepts_text.see('end')
    
    def start_training(self):
        """เริ่มต้นการ training model (legacy method)"""
        try:
            # เรียกใช้ training script
            import subprocess
            import sys
            
            training_script = os.path.join(os.path.dirname(__file__), 'scripts', 'train_model.py')
            
            # เปิด terminal ใหม่สำหรับ training
            if sys.platform.startswith('win'):
                subprocess.Popen(['cmd', '/c', 'python', training_script, '--create_data', '--epochs', '10'])
            else:
                subprocess.Popen(['python3', training_script, '--create_data', '--epochs', '10'])
            
            messagebox.showinfo("Training Started", 
                              "Model training has been started in a new window.\n" +
                              "This will demonstrate mathematical concepts including:\n" +
                              "• Vector operations (dot product, norms)\n" +
                              "• Matrix operations (convolution, multiplication)\n" +
                              "• Optimization algorithms (Adam, gradient descent)\n" +
                              "• Loss functions (MSE, L1, perceptual)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")
    
    def show_math_concepts(self):
        """แสดง mathematical concepts"""
        try:
            # สร้างหน้าต่างใหม่สำหรับแสดง mathematical concepts
            math_window = tk.Toplevel(self.root)
            math_window.title("Linear Algebra Mathematical Concepts")
            math_window.geometry("800x600")
            math_window.configure(bg='#f0f0f0')
            
            # สร้าง text widget สำหรับแสดงข้อมูล
            text_frame = tk.Frame(math_window, bg='#f0f0f0')
            text_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, wrap='word', font=('Courier', 10), 
                                bg='white', fg='black')
            scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')
            
            # แสดง mathematical concepts
            concepts_text = self.get_mathematical_concepts_text()
            text_widget.insert('1.0', concepts_text)
            text_widget.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show mathematical concepts: {str(e)}")
    
    def get_mathematical_concepts_text(self):
        """สร้างข้อความสำหรับแสดง mathematical concepts"""
        text = []
        text.append("LINEAR ALGEBRA MATHEMATICAL CONCEPTS")
        text.append("=" * 60)
        
        # Vector operations
        text.append("\n1. VECTOR OPERATIONS:")
        text.append("-" * 30)
        text.append("Dot Product: a · b = Σ(a_i * b_i)")
        text.append("  • Used in: Attention mechanisms, similarity calculations")
        text.append("  • Neural Network: Fully connected layers, cosine similarity")
        
        text.append("\nVector Norm: ||v||_p = (Σ|v_i|^p)^(1/p)")
        text.append("  • L2 norm: Euclidean distance")
        text.append("  • L1 norm: Manhattan distance")
        text.append("  • Used in: Regularization, normalization")
        
        text.append("\nCosine Similarity: cos(θ) = (a·b) / (||a|| * ||b||)")
        text.append("  • Measures angle between vectors")
        text.append("  • Range: [-1, 1]")
        text.append("  • Used in: Loss functions, attention mechanisms")
        
        # Matrix operations
        text.append("\n\n2. MATRIX OPERATIONS:")
        text.append("-" * 30)
        text.append("Matrix Multiplication: C = AB where C_ij = Σ(A_ik * B_kj)")
        text.append("  • Fundamental operation in neural networks")
        text.append("  • Used in: Fully connected layers, attention mechanisms")
        
        text.append("\nConvolution: y[m,n] = Σ Σ x[i,j] * k[m-i, n-j]")
        text.append("  • Sliding window operation")
        text.append("  • Used in: CNN feature extraction")
        
        text.append("\nTranspose: A^T where A^T_ij = A_ji")
        text.append("  • Essential for backpropagation")
        text.append("  • Used in: Gradient computation")
        
        # Optimization
        text.append("\n\n3. OPTIMIZATION MATHEMATICS:")
        text.append("-" * 30)
        text.append("Gradient Descent: θ_{t+1} = θ_t - α∇L(θ_t)")
        text.append("  • Updates parameters in steepest descent direction")
        text.append("  • Uses first-order Taylor approximation")
        
        text.append("\nAdam Optimizer:")
        text.append("  • m_t = β₁m_{t-1} + (1-β₁)g_t  (momentum)")
        text.append("  • v_t = β₂v_{t-1} + (1-β₂)g_t² (adaptive learning rate)")
        text.append("  • θ_t = θ_{t-1} - α*m_t/√(v_t + ε)")
        
        # Loss functions
        text.append("\n\n4. LOSS FUNCTIONS:")
        text.append("-" * 30)
        text.append("MSE Loss: L = (1/n) * Σ(y_true - y_pred)²")
        text.append("  • L2 norm: ||y_true - y_pred||²₂ / n")
        text.append("  • Penalizes large errors more than small ones")
        
        text.append("\nL1 Loss: L = (1/n) * Σ|y_true - y_pred|")
        text.append("  • L1 norm: ||y_true - y_pred||₁ / n")
        text.append("  • More robust to outliers")
        
        text.append("\nPerceptual Loss: L = ||φ(y_true) - φ(y_pred)||²₂")
        text.append("  • Uses feature space distance")
        text.append("  • φ is feature extraction function (e.g., VGG)")
        
        # CNN specific
        text.append("\n\n5. CNN LINEAR ALGEBRA:")
        text.append("-" * 30)
        text.append("Convolution as Matrix Multiplication:")
        text.append("  • y = Conv2D(x, k) = reshape(Toeplitz(x) @ flatten(k))")
        
        text.append("\nBatch Normalization:")
        text.append("  • y = γ * (x - μ) / √(σ² + ε) + β")
        text.append("  • Normalizes inputs for stable training")
        
        text.append("\nAttention Mechanism:")
        text.append("  • Attention(Q,K,V) = softmax(QK^T/√d_k)V")
        text.append("  • Uses matrix multiplication for attention weights")
        
        return "\n".join(text)
    
    def show_training_progress(self):
        """แสดง training progress"""
        try:
            # ตรวจสอบว่ามีไฟล์ training progress หรือไม่
            results_dir = os.path.join(os.path.dirname(__file__), 'trained_models')
            
            if not os.path.exists(results_dir):
                messagebox.showinfo("No Training Data", 
                                  "No training has been performed yet.\n" +
                                  "Click 'Start Training' to begin training the model.")
                return
            
            # แสดง training curves ถ้ามี
            curve_file = os.path.join(results_dir, 'training_curves.png')
            if os.path.exists(curve_file):
                # เปิดไฟล์ภาพ
                import subprocess
                import sys
                
                if sys.platform.startswith('win'):
                    os.startfile(curve_file)
                else:
                    subprocess.Popen(['xdg-open', curve_file])
                
                messagebox.showinfo("Training Progress", 
                                  f"Training curves have been opened.\n" +
                                  f"File location: {curve_file}")
            else:
                messagebox.showinfo("No Progress Data", 
                                  "No training progress data found.\n" +
                                  "Start training to generate progress plots.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show training progress: {str(e)}")


def main():
    """ฟังก์ชันหลัก"""
    root = tk.Tk()
    app = LinearAlgebraApp(root)
    
    # ตั้งค่า icon และ style
    try:
        root.iconbitmap('icon.ico')  # ถ้ามี icon file
    except:
        pass
    
    # เริ่มต้นแอปพลิเคชัน
    root.mainloop()


if __name__ == "__main__":
    main()
