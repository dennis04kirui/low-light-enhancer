import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from scripts.inference import enhance_image
import os

class LowLightGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Low-Light Image Enhancer")
        self.root.geometry("800x500")
        self.root.resizable(True, True)

        # Frames
        self.frame_top = ttk.Frame(root)
        self.frame_top.pack(pady=10)

        self.frame_images = ttk.Frame(root)
        self.frame_images.pack(pady=10, fill="both", expand=True)

        self.frame_buttons = ttk.Frame(root)
        self.frame_buttons.pack(pady=10)

        # Browse button
        self.btn_browse = ttk.Button(self.frame_top, text="Browse Image", command=self.browse_image)
        self.btn_browse.pack()

        # Original Image Label
        self.label_original = ttk.Label(self.frame_images, text="Original Image")
        self.label_original.grid(row=0, column=0, padx=10, pady=5)

        # Enhanced Image Label
        self.label_enhanced = ttk.Label(self.frame_images, text="Enhanced Image")
        self.label_enhanced.grid(row=0, column=1, padx=10, pady=5)

        # Canvas for images
        self.canvas_original = tk.Label(self.frame_images)
        self.canvas_original.grid(row=1, column=0, padx=10, pady=5)

        self.canvas_enhanced = tk.Label(self.frame_images)
        self.canvas_enhanced.grid(row=1, column=1, padx=10, pady=5)

        # Save button
        self.btn_save = ttk.Button(self.frame_buttons, text="Save Enhanced Image", command=self.save_image, state="disabled")
        self.btn_save.pack()

        self.original_image = None
        self.enhanced_image = None

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        try:
            # Load original image
            self.original_image = Image.open(file_path).convert("RGB")
            img_display = self.original_image.resize((350, 350))
            self.tk_original = ImageTk.PhotoImage(img_display)
            self.canvas_original.config(image=self.tk_original)
            self.canvas_original.image = self.tk_original

            # Enhance image
            self.enhanced_image = enhance_image(file_path)
            img_display_enhanced = self.enhanced_image.resize((350, 350))
            self.tk_enhanced = ImageTk.PhotoImage(img_display_enhanced)
            self.canvas_enhanced.config(image=self.tk_enhanced)
            self.canvas_enhanced.image = self.tk_enhanced

            # Enable save button
            self.btn_save.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to enhance image:\n{e}")

    def save_image(self):
        if self.enhanced_image is None:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG Image", "*.jpg"),
                                                            ("PNG Image", "*.png")])
        if not save_path:
            return
        try:
            self.enhanced_image.save(save_path)
            messagebox.showinfo("Success", f"Enhanced image saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LowLightGUI(root)
    root.mainloop()
