import rawpy
import numpy as np
import os
import csv
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import subprocess
import platform

def extract_2x2_rgb_average(raw_file_path, x, y):
    """
    从RAW图片中提取指定坐标处2x2区域的RGB平均值（使用debayer后的RGB图像）
    
    参数:
        raw_file_path: DNG文件路径
        x, y: 坐标（左上角位置，在debayer后的RGB图像坐标系中）
    
    返回:
        dict: 包含R, G, B平均值和统计信息的字典
    """
    # 使用 context manager 方式读取，这是 rawpy 推荐的方式
    try:
        # 确保路径是字符串格式
        file_path_str = str(Path(raw_file_path).resolve())
        
        # 使用 context manager 读取 RAW 文件
        with rawpy.imread(file_path_str) as raw:
            # 将RAW图像转换为RGB图像（debayer处理）
            # 使用默认参数进行转换，保持原始尺寸
            rgb_array = raw.postprocess(
                use_camera_wb=True,      # 使用相机白平衡
                half_size=False,         # 不缩小尺寸
                no_auto_bright=True,     # 不自动调整亮度
                output_bps=16,           # 16位输出以获得更高精度
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD  # 使用AHD算法
            )
            
            # rgb_array 的形状是 (height, width, 3)，3个通道分别是R, G, B
            height, width, channels = rgb_array.shape
            
            # 检查坐标是否在有效范围内（考虑2x2区域）
            if x < 0 or y < 0 or x + 2 > width or y + 2 > height:
                raise ValueError(f"坐标 ({x}, {y}) 超出图像范围 ({width}x{height})")
            
            # 提取2x2区域的RGB值
            # region的形状是 (2, 2, 3)，包含4个像素，每个像素有R、G、B三个通道
            region = rgb_array[y:y+2, x:x+2, :]
            
            # 提取每个像素的RGB值
            # 像素位置：
            # [0,0] 左上角
            # [0,1] 右上角
            # [1,0] 左下角
            # [1,1] 右下角
            pixel_00 = region[0, 0, :]  # 左上角像素 (R, G, B)
            pixel_01 = region[0, 1, :]  # 右上角像素 (R, G, B)
            pixel_10 = region[1, 0, :]  # 左下角像素 (R, G, B)
            pixel_11 = region[1, 1, :]  # 右下角像素 (R, G, B)
            
            # 提取所有像素的R、G、B值
            r_values = [int(pixel_00[0]), int(pixel_01[0]), int(pixel_10[0]), int(pixel_11[0])]
            g_values = [int(pixel_00[1]), int(pixel_01[1]), int(pixel_10[1]), int(pixel_11[1])]
            b_values = [int(pixel_00[2]), int(pixel_01[2]), int(pixel_10[2]), int(pixel_11[2])]
            
            # 计算平均值
            r_avg = float(np.mean(r_values))
            g_avg = float(np.mean(g_values))
            b_avg = float(np.mean(b_values))
            
            # 计算统计信息
            r_min = float(np.min(r_values))
            r_max = float(np.max(r_values))
            g_min = float(np.min(g_values))
            g_max = float(np.max(g_values))
            b_min = float(np.min(b_values))
            b_max = float(np.max(b_values))
            
            r_std = float(np.std(r_values))
            g_std = float(np.std(g_values))
            b_std = float(np.std(b_values))
            
            result = {
                'file': os.path.basename(raw_file_path),
                'x': x,
                'y': y,
                'R': r_avg,
                'G': g_avg,
                'B': b_avg,
                'G2': g_avg,  # G2保持与G相同（因为debayer后每个像素都有完整的RGB）
                'R_min': r_min,
                'R_max': r_max,
                'G_min': g_min,
                'G_max': g_max,
                'B_min': b_min,
                'B_max': b_max,
                'G2_min': g_min,  # G2统计与G相同
                'G2_max': g_max,
                'R_std': r_std,
                'G_std': g_std,
                'B_std': b_std,
                'G2_std': g_std,  # G2标准差与G相同
                'pixel_values': {
                    'pixel_00': {'R': int(pixel_00[0]), 'G': int(pixel_00[1]), 'B': int(pixel_00[2])},
                    'pixel_01': {'R': int(pixel_01[0]), 'G': int(pixel_01[1]), 'B': int(pixel_01[2])},
                    'pixel_10': {'R': int(pixel_10[0]), 'G': int(pixel_10[1]), 'B': int(pixel_10[2])},
                    'pixel_11': {'R': int(pixel_11[0]), 'G': int(pixel_11[1]), 'B': int(pixel_11[2])}
                }
            }
            
            return result
                    
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        print(f"处理文件 {raw_file_path} 时出错:")
        print(f"  错误类型: {error_type}")
        print(f"  错误信息: {error_msg}")
        print(f"  详细堆栈:\n{error_traceback}")
        return None

def process_image_folder(folder_path, x, y, output_csv='output.csv', progress_callback=None):
    """
    处理文件夹中的所有DNG图片
    
    参数:
        folder_path: 包含DNG文件的文件夹路径
        x, y: 要提取的坐标
        output_csv: 输出CSV文件路径（绝对路径）
        progress_callback: 进度回调函数，接收(current, total, filename)参数
    
    返回:
        str: 保存的文件完整路径
    """
    folder = Path(folder_path)
    # 使用不区分大小写的方式查找DNG文件，避免重复
    dng_files = sorted(set(folder.glob('*.DNG')) | set(folder.glob('*.dng')))
    
    if not dng_files:
        print(f"在文件夹 {folder_path} 中未找到DNG文件")
        return None
    
    print(f"找到 {len(dng_files)} 个DNG文件")
    print(f"处理坐标 ({x}, {y}) 处的2x2区域...")
    print("-" * 60)
    
    results = []
    failed_files = []
    total_files = len(dng_files)
    
    for idx, dng_file in enumerate(dng_files, 1):
        print(f"[{idx}/{total_files}] 正在处理: {dng_file.name}")
        
        # 更新进度回调
        if progress_callback:
            progress_callback(idx, total_files, dng_file.name)
        
        result = extract_2x2_rgb_average(dng_file, x, y)
        if result:
            results.append(result)
            print(f"  ✓ 成功")
        else:
            failed_files.append(dng_file.name)
            print(f"  ✗ 失败")
    
    print("-" * 60)
    print(f"处理完成: 成功 {len(results)} 个, 失败 {len(failed_files)} 个")
    
    if failed_files:
        print(f"失败的文件列表:")
        for fname in failed_files[:10]:  # 只显示前10个
            print(f"  - {fname}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个文件失败")
    
    if not results:
        print("\n错误: 没有成功处理任何文件！")
        print("请检查:")
        print("1. DNG文件是否损坏")
        print("2. 坐标是否在图像范围内")
        print("3. rawpy库是否正确安装")
        return None
    
    # 确保使用绝对路径
    output_path = Path(output_csv).resolve()
    
    # 导出到CSV
    if results:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ['file', 'x', 'y', 'R', 'G', 'B', 'G2', 
                         'R_min', 'R_max', 'G_min', 'G_max', 'B_min', 'B_max', 
                         'G2_min', 'G2_max', 'R_std', 'G_std', 'B_std', 'G2_std']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                # 只写入CSV字段，不包括pixel_values
                row = {k: v for k, v in result.items() if k != 'pixel_values'}
                writer.writerow(row)
        
        print(f"\n数据已导出到: {output_path}")
        
        # 计算所有图片的平均值
        if len(results) > 1:
            r_values = [r['R'] for r in results]
            g_values = [r['G'] for r in results]
            b_values = [r['B'] for r in results]
            g2_values = [r['G2'] for r in results]
            
            print(f"\n所有图片的平均值统计:")
            print(f"R: 平均值={np.mean(r_values):.2f}, 最小值={np.min(r_values):.2f}, 最大值={np.max(r_values):.2f}, 标准差={np.std(r_values):.2f}")
            print(f"G: 平均值={np.mean(g_values):.2f}, 最小值={np.min(g_values):.2f}, 最大值={np.max(g_values):.2f}, 标准差={np.std(g_values):.2f}")
            print(f"B: 平均值={np.mean(b_values):.2f}, 最小值={np.min(b_values):.2f}, 最大值={np.max(b_values):.2f}, 标准差={np.std(b_values):.2f}")
            print(f"G2: 平均值={np.mean(g2_values):.2f}, 最小值={np.min(g2_values):.2f}, 最大值={np.max(g2_values):.2f}, 标准差={np.std(g2_values):.2f}")
    
    return str(output_path)

class CoordinateSelector:
    def __init__(self, folder_path, output_csv='rgb_average_results.csv'):
        self.folder_path = folder_path
        # 使用绝对路径，保存在脚本所在目录
        script_dir = Path(__file__).parent.resolve()
        self.output_csv = str(script_dir / output_csv)
        self.output_file_path = None  # 保存实际生成的文件路径
        self.selected_x = None
        self.selected_y = None
        self.original_image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.base_scale_factor = 1.0  # 初始缩放比例
        self.current_zoom = 1.0  # 当前缩放倍数
        self.original_width = 0
        self.original_height = 0
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("选择坐标 - DNG图片处理")
        self.root.geometry("1200x800")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)  # 画布所在行
        
        # 信息标签和文件夹显示
        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, pady=(5, 2), sticky=(tk.W, tk.E))
        top_frame.columnconfigure(1, weight=1)
        
        info_label = ttk.Label(top_frame, text="点击图片选择坐标（2x2区域的左上角）", 
                              font=('Arial', 12))
        info_label.grid(row=0, column=0, columnspan=2, pady=(0, 2))
        
        folder_label_frame = ttk.Frame(top_frame)
        folder_label_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=0)
        folder_label_frame.columnconfigure(1, weight=1)
        
        ttk.Label(folder_label_frame, text="当前文件夹:", font=('Arial', 9)).grid(row=0, column=0, padx=2)
        self.folder_path_label = ttk.Label(folder_label_frame, text=folder_path, 
                                          font=('Arial', 9), foreground='blue')
        self.folder_path_label.grid(row=0, column=1, sticky=tk.W, padx=2)
        
        # 创建画布和滚动条
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=2)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(canvas_frame, bg='gray', cursor='crosshair')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        # 绑定鼠标滚轮缩放
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel)     # Linux
        self.canvas.bind("<Button-5>", self.on_mousewheel)     # Linux
        self.canvas.focus_set()  # 使画布可以接收焦点以响应滚轮事件
        
        # 坐标输入和显示框架
        coord_frame = ttk.Frame(main_frame)
        coord_frame.grid(row=2, column=0, pady=(5, 2))
        
        ttk.Label(coord_frame, text="坐标:", font=('Arial', 10)).pack(side=tk.LEFT, padx=2)
        ttk.Label(coord_frame, text="X:", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        self.x_entry = ttk.Entry(coord_frame, width=8, font=('Arial', 9))
        self.x_entry.pack(side=tk.LEFT, padx=2)
        self.x_entry.bind('<Return>', self.on_coord_entry)
        self.x_entry.bind('<KP_Enter>', self.on_coord_entry)
        
        ttk.Label(coord_frame, text="Y:", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        self.y_entry = ttk.Entry(coord_frame, width=8, font=('Arial', 9))
        self.y_entry.pack(side=tk.LEFT, padx=2)
        self.y_entry.bind('<Return>', self.on_coord_entry)
        self.y_entry.bind('<KP_Enter>', self.on_coord_entry)
        
        ttk.Button(coord_frame, text="定位", command=self.on_coord_entry, width=6).pack(side=tk.LEFT, padx=5)
        
        # 坐标显示标签
        self.coord_label = ttk.Label(main_frame, text="坐标: 未选择", font=('Arial', 10))
        self.coord_label.grid(row=3, column=0, pady=(2, 2))
        
        # 缩放控制框架
        zoom_frame = ttk.Frame(main_frame)
        zoom_frame.grid(row=4, column=0, pady=2)
        ttk.Label(zoom_frame, text="缩放:", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="放大 (+)", command=lambda: self.zoom_image(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="缩小 (-)", command=lambda: self.zoom_image(0.8)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="重置", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(zoom_frame, text="100%", font=('Arial', 9))
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, pady=5)
        
        self.select_button = ttk.Button(button_frame, text="确认选择并处理", 
                                        command=self.confirm_and_process, state='disabled')
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="重新选择坐标", command=self.reset_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="选择文件夹", command=self.select_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
        # 进度条（使用确定模式显示实际进度）
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # 状态标签
        self.status_label = ttk.Label(main_frame, text="正在加载第一张图片...", font=('Arial', 9))
        self.status_label.grid(row=7, column=0, pady=2)
        
        # 文件路径显示标签（可点击复制）
        self.file_path_label = ttk.Label(main_frame, text="", font=('Arial', 8), 
                                         foreground='blue', cursor='hand2')
        self.file_path_label.grid(row=8, column=0, pady=1)
        self.file_path_label.bind("<Button-1>", self.copy_path_to_clipboard)
        
        # 打开文件按钮（初始隐藏）
        self.open_file_button = ttk.Button(main_frame, text="打开文件位置", 
                                          command=self.open_file_location, state='disabled')
        self.open_file_button.grid(row=9, column=0, pady=2)
        
        # 加载第一张图片
        self.load_first_image()
    
    def load_first_image(self):
        """加载第一张DNG图片用于选择坐标"""
        folder = Path(self.folder_path)
        dng_files = sorted(set(folder.glob('*.DNG')) | set(folder.glob('*.dng')))
        
        if not dng_files:
            messagebox.showerror("错误", f"在文件夹 {self.folder_path} 中未找到DNG文件")
            self.root.quit()
            return
        
        try:
            self.status_label.config(text=f"正在加载: {dng_files[0].name}...")
            self.root.update()
            
            # 读取RAW图片
            with rawpy.imread(str(dng_files[0])) as raw:
                # 先获取16位RGB图像以获取准确尺寸（用于坐标系统）
                rgb_16bit = raw.postprocess(
                    use_camera_wb=True,      # 使用相机白平衡
                    half_size=False,         # 不缩小尺寸
                    no_auto_bright=True,     # 不自动调整亮度
                    output_bps=16,           # 16位用于获取尺寸
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD  # 使用AHD算法
                )
                
                # 获取RGB图像的尺寸（这是实际使用的坐标系）
                self.original_height, self.original_width = rgb_16bit.shape[:2]
                
                # 转换为8位用于显示（PIL Image需要8位或特殊格式）
                # 将16位值缩放到8位范围 (0-65535 -> 0-255)
                rgb_8bit = (rgb_16bit / 256).astype(np.uint8)
                
                # 转换为PIL Image
                self.original_image = Image.fromarray(rgb_8bit)
                
                # 计算初始缩放比例以适应窗口
                max_display_width = 1000
                max_display_height = 700
                
                img_width, img_height = self.original_image.size
                scale_w = max_display_width / img_width
                scale_h = max_display_height / img_height
                self.base_scale_factor = min(scale_w, scale_h, 1.0)  # 初始缩放比例
                self.scale_factor = self.base_scale_factor
                self.current_zoom = 1.0
                
                # 显示图片
                self.update_display_image()
                
                self.status_label.config(text=f"已加载: {dng_files[0].name} ({img_width}x{img_height}) | "
                                             f"使用鼠标滚轮或按钮可以缩放")
                
        except Exception as e:
            messagebox.showerror("错误", f"加载图片时出错: {str(e)}")
            self.root.quit()
    
    def on_canvas_click(self, event):
        """处理画布点击事件"""
        # 获取画布坐标
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # 转换为原始图片坐标
        original_x = int(canvas_x / self.scale_factor)
        original_y = int(canvas_y / self.scale_factor)
        
        # 确保坐标在有效范围内（考虑2x2区域）
        if original_x < 0 or original_y < 0 or \
           original_x + 2 > self.original_width or original_y + 2 > self.original_height:
            messagebox.showwarning("警告", 
                f"坐标 ({original_x}, {original_y}) 超出图像范围或无法容纳2x2区域\n"
                f"图像尺寸: {self.original_width}x{self.original_height}")
            return
        
        self.selected_x = original_x
        self.selected_y = original_y
        
        # 更新输入框
        self.x_entry.delete(0, tk.END)
        self.x_entry.insert(0, str(original_x))
        self.y_entry.delete(0, tk.END)
        self.y_entry.insert(0, str(original_y))
        
        # 更新显示
        self.update_coord_display()
        self.draw_selection_marker(canvas_x, canvas_y)
        self.select_button.config(state='normal')
    
    def on_canvas_motion(self, event):
        """鼠标移动时显示坐标"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        original_x = int(canvas_x / self.scale_factor)
        original_y = int(canvas_y / self.scale_factor)
        
        if 0 <= original_x < self.original_width and 0 <= original_y < self.original_height:
            self.coord_label.config(text=f"鼠标位置: ({original_x}, {original_y}) | "
                                        f"已选择: ({self.selected_x if self.selected_x is not None else '未选择'}, "
                                        f"{self.selected_y if self.selected_y is not None else '未选择'})")
    
    def update_display_image(self):
        """更新显示的图片"""
        if self.original_image is None:
            return
        
        img_width, img_height = self.original_image.size
        display_width = int(img_width * self.scale_factor)
        display_height = int(img_height * self.scale_factor)
        
        # 缩放图片用于显示
        self.display_image = self.original_image.resize(
            (display_width, display_height), Image.Resampling.LANCZOS)
        
        # 转换为PhotoImage
        self.photo = ImageTk.PhotoImage(self.display_image)
        
        # 清除画布并重新显示
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        # 如果有选择标记，重新绘制
        if self.selected_x is not None and self.selected_y is not None:
            canvas_x = self.selected_x * self.scale_factor
            canvas_y = self.selected_y * self.scale_factor
            self.draw_selection_marker(canvas_x, canvas_y)
        
        # 更新缩放标签
        zoom_percent = int(self.current_zoom * 100)
        self.zoom_label.config(text=f"{zoom_percent}%")
    
    def zoom_image(self, zoom_factor, mouse_x=None, mouse_y=None):
        """缩放图片，以鼠标位置为中心"""
        if self.original_image is None:
            return
        
        # 限制缩放范围（增加最大缩放倍数）
        min_zoom = 0.1
        max_zoom = 20.0  # 从5.0增加到20.0
        
        new_zoom = self.current_zoom * zoom_factor
        new_zoom = max(min_zoom, min(max_zoom, new_zoom))
        
        if new_zoom == self.current_zoom:
            return  # 没有变化，不需要更新
        
        # 如果提供了鼠标位置，以鼠标位置为中心缩放
        if mouse_x is not None and mouse_y is not None:
            # 保存旧的缩放比例
            old_scale_factor = self.scale_factor
            
            # 获取鼠标在画布上的位置（考虑当前滚动）
            canvas_x = self.canvas.canvasx(mouse_x)
            canvas_y = self.canvas.canvasy(mouse_y)
            
            # 计算鼠标位置对应的原始图片坐标（使用旧的缩放比例）
            image_x = canvas_x / old_scale_factor
            image_y = canvas_y / old_scale_factor
            
            # 更新缩放比例
            self.current_zoom = new_zoom
            self.scale_factor = self.base_scale_factor * self.current_zoom
            
            # 更新显示（这会重新创建图片，可能会重置滚动）
            self.update_display_image()
            
            # 等待画布更新
            self.root.update_idletasks()
            
            # 计算缩放后该图片坐标对应的新画布位置
            new_canvas_x = image_x * self.scale_factor
            new_canvas_y = image_y * self.scale_factor
            
            # 获取画布内容的总尺寸
            bbox = self.canvas.bbox("all")
            if bbox:
                total_width = bbox[2] - bbox[0]
                total_height = bbox[3] - bbox[1]
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if total_width > 0 and total_height > 0 and canvas_width > 0 and canvas_height > 0:
                    # 计算需要滚动到的位置
                    # 我们希望 new_canvas_x 显示在 mouse_x 位置
                    # 滚动位置 = new_canvas_x - mouse_x
                    target_scroll_x = new_canvas_x - mouse_x
                    target_scroll_y = new_canvas_y - mouse_y
                    
                    # 限制滚动范围
                    max_scroll_x = max(0, total_width - canvas_width)
                    max_scroll_y = max(0, total_height - canvas_height)
                    target_scroll_x = max(0, min(target_scroll_x, max_scroll_x))
                    target_scroll_y = max(0, min(target_scroll_y, max_scroll_y))
                    
                    # 应用新的滚动位置（转换为0-1的比例）
                    if total_width > canvas_width:
                        self.canvas.xview_moveto(target_scroll_x / total_width)
                    if total_height > canvas_height:
                        self.canvas.yview_moveto(target_scroll_y / total_height)
        else:
            # 没有鼠标位置，以画布中心缩放
            self.current_zoom = new_zoom
            self.scale_factor = self.base_scale_factor * self.current_zoom
            self.update_display_image()
    
    def reset_zoom(self):
        """重置缩放"""
        if self.original_image is None:
            return
        
        self.current_zoom = 1.0
        self.scale_factor = self.base_scale_factor
        self.update_display_image()
    
    def on_mousewheel(self, event):
        """处理鼠标滚轮缩放，以鼠标位置为中心"""
        if self.original_image is None:
            return
        
        # 获取鼠标在画布上的位置
        mouse_x = event.x
        mouse_y = event.y
        
        # Windows 使用 delta，Linux 使用 num
        if event.num == 4 or event.delta > 0:
            self.zoom_image(1.1, mouse_x, mouse_y)  # 放大，以鼠标位置为中心
        elif event.num == 5 or event.delta < 0:
            self.zoom_image(0.9, mouse_x, mouse_y)  # 缩小，以鼠标位置为中心
    
    def draw_selection_marker(self, canvas_x, canvas_y):
        """在画布上绘制选择标记"""
        # 清除之前的标记
        self.canvas.delete("selection_marker")
        
        # 绘制2x2区域的标记
        marker_size = 2 * self.scale_factor
        self.canvas.create_rectangle(
            canvas_x, canvas_y,
            canvas_x + marker_size, canvas_y + marker_size,
            outline='red', width=2, tags="selection_marker"
        )
        # 绘制中心点
        self.canvas.create_oval(
            canvas_x + marker_size/2 - 3, canvas_y + marker_size/2 - 3,
            canvas_x + marker_size/2 + 3, canvas_y + marker_size/2 + 3,
            fill='red', outline='red', tags="selection_marker"
        )
    
    def update_coord_display(self):
        """更新坐标显示"""
        if self.selected_x is not None and self.selected_y is not None:
            self.coord_label.config(
                text=f"已选择坐标: ({self.selected_x}, {self.selected_y}) | "
                     f"2x2区域: ({self.selected_x}, {self.selected_y}) 到 "
                     f"({self.selected_x+1}, {self.selected_y+1})"
            )
    
    def reset_selection(self):
        """重置选择"""
        self.selected_x = None
        self.selected_y = None
        self.canvas.delete("selection_marker")
        self.coord_label.config(text="坐标: 未选择")
        self.select_button.config(state='disabled')
        # 清空输入框
        self.x_entry.delete(0, tk.END)
        self.y_entry.delete(0, tk.END)
    
    def on_coord_entry(self, event=None):
        """处理手动输入坐标"""
        if self.original_image is None:
            messagebox.showinfo("提示", "请先加载图片")
            return
        
        try:
            x_str = self.x_entry.get().strip()
            y_str = self.y_entry.get().strip()
            
            if not x_str or not y_str:
                messagebox.showwarning("警告", "请输入X和Y坐标")
                return
            
            x = int(x_str)
            y = int(y_str)
            
            # 检查坐标是否在有效范围内（考虑2x2区域）
            if x < 0 or y < 0 or x + 2 > self.original_width or y + 2 > self.original_height:
                messagebox.showwarning("警告", 
                    f"坐标 ({x}, {y}) 超出图像范围或无法容纳2x2区域\n"
                    f"图像尺寸: {self.original_width}x{self.original_height}\n"
                    f"有效范围: X: 0-{self.original_width-2}, Y: 0-{self.original_height-2}")
                return
            
            # 设置选择的坐标
            self.selected_x = x
            self.selected_y = y
            
            # 更新显示
            self.update_coord_display()
            
            # 计算画布坐标并绘制标记
            canvas_x = x * self.scale_factor
            canvas_y = y * self.scale_factor
            self.draw_selection_marker(canvas_x, canvas_y)
            
            # 滚动到选择的位置（如果不在可见区域）
            self.scroll_to_position(canvas_x, canvas_y)
            
            self.select_button.config(state='normal')
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字坐标")
    
    def scroll_to_position(self, canvas_x, canvas_y):
        """滚动画布到指定位置"""
        # 获取画布的可见区域
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return  # 画布还没有初始化
        
        # 获取画布内容的总尺寸
        bbox = self.canvas.bbox("all")
        if bbox is None:
            return
        
        total_width = bbox[2] - bbox[0]
        total_height = bbox[3] - bbox[1]
        
        if total_width <= 0 or total_height <= 0:
            return
        
        # 计算需要滚动的距离（将目标位置放在中心）
        scroll_x = canvas_x - canvas_width / 2
        scroll_y = canvas_y - canvas_height / 2
        
        # 限制滚动范围
        scroll_x = max(0, min(scroll_x, total_width - canvas_width))
        scroll_y = max(0, min(scroll_y, total_height - canvas_height))
        
        # 滚动到位置（转换为0-1之间的比例）
        if total_width > canvas_width:
            self.canvas.xview_moveto(scroll_x / total_width)
        if total_height > canvas_height:
            self.canvas.yview_moveto(scroll_y / total_height)
    
    def select_folder(self):
        """选择包含DNG文件的文件夹"""
        folder = filedialog.askdirectory(
            title="选择包含DNG文件的文件夹",
            initialdir=self.folder_path if Path(self.folder_path).exists() else None
        )
        
        if folder:
            # 检查文件夹中是否有DNG文件
            folder_path = Path(folder)
            dng_files = sorted(set(folder_path.glob('*.DNG')) | set(folder_path.glob('*.dng')))
            
            if not dng_files:
                messagebox.showwarning("警告", f"在文件夹 {folder} 中未找到DNG文件")
                return
            
            # 更新文件夹路径
            self.folder_path = folder
            self.folder_path_label.config(text=folder)
            
            # 重置选择
            self.reset_selection()
            self.reset_zoom()
            
            # 重新加载第一张图片
            self.load_first_image()
    
    def confirm_and_process(self):
        """确认选择并开始处理"""
        if self.selected_x is None or self.selected_y is None:
            messagebox.showwarning("警告", "请先选择坐标")
            return
        
        # 禁用按钮
        self.select_button.config(state='disabled')
        self.progress['value'] = 0
        self.progress['maximum'] = 100
        self.status_label.config(text="正在处理所有图片，请稍候...")
        
        # 获取文件总数以设置进度条最大值
        folder = Path(self.folder_path)
        dng_files = sorted(set(folder.glob('*.DNG')) | set(folder.glob('*.dng')))
        total_files = len(dng_files)
        
        if total_files == 0:
            messagebox.showerror("错误", "未找到DNG文件")
            self.select_button.config(state='normal')
            return
        
        # 进度更新回调函数
        def update_progress(current, total, filename):
            """更新进度条"""
            progress_value = int((current / total) * 100)
            self.root.after(0, lambda: self.progress.config(value=progress_value))
            self.root.after(0, lambda: self.status_label.config(
                text=f"正在处理: {filename} ({current}/{total})"
            ))
        
        # 在新线程中处理，避免阻塞UI
        def process_thread():
            try:
                output_path = process_image_folder(
                    self.folder_path, 
                    self.selected_x, 
                    self.selected_y, 
                    self.output_csv,
                    progress_callback=update_progress
                )
                self.root.after(0, lambda: self.on_processing_complete(output_path))
            except Exception as e:
                self.root.after(0, lambda: self.on_processing_error(str(e)))
        
        thread = threading.Thread(target=process_thread, daemon=True)
        thread.start()
    
    def on_processing_complete(self, output_path):
        """处理完成"""
        self.progress['value'] = 100  # 设置为100%完成
        self.output_file_path = output_path
        
        if output_path and Path(output_path).exists():
            self.status_label.config(text=f"处理完成！文件已保存")
            self.file_path_label.config(text=f"文件位置: {output_path} (点击复制路径)")
            self.open_file_button.config(state='normal')
            
            messagebox.showinfo("完成", 
                f"所有图片处理完成！\n\n"
                f"文件已保存到:\n{output_path}\n\n"
                f"点击'打开文件位置'按钮可以查看文件")
        else:
            self.status_label.config(text="处理完成，但文件路径未找到")
            messagebox.showwarning("警告", "处理完成，但无法找到保存的文件路径")
    
    def on_processing_error(self, error_msg):
        """处理出错"""
        self.progress['value'] = 0  # 重置进度条
        self.status_label.config(text=f"处理出错: {error_msg}")
        messagebox.showerror("错误", f"处理图片时出错:\n{error_msg}")
        self.select_button.config(state='normal')
    
    def copy_path_to_clipboard(self, event=None):
        """复制文件路径到剪贴板"""
        if self.output_file_path:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.output_file_path)
            self.status_label.config(text="路径已复制到剪贴板！")
    
    def open_file_location(self):
        """打开文件所在位置"""
        if not self.output_file_path or not Path(self.output_file_path).exists():
            messagebox.showerror("错误", "文件不存在或路径无效")
            return
        
        file_path = Path(self.output_file_path)
        system = platform.system()
        
        try:
            if system == "Windows":
                # Windows: 在资源管理器中打开并选中文件
                subprocess.run(f'explorer /select,"{file_path}"', shell=True)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", "-R", str(file_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(file_path.parent)])
        except Exception as e:
            messagebox.showerror("错误", f"无法打开文件位置:\n{str(e)}")
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    # 配置参数
    DEFAULT_FOLDER = "处理"  # 默认文件夹
    OUTPUT_CSV = "rgb_average_results.csv"  # 输出文件名
    
    # 如果默认文件夹不存在，让用户选择
    if not Path(DEFAULT_FOLDER).exists():
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        folder = filedialog.askdirectory(
            title="选择包含DNG文件的文件夹",
            initialdir=Path.cwd()
        )
        
        root.destroy()
        
        if not folder:
            print("未选择文件夹，程序退出")
            exit(0)
        
        FOLDER_PATH = folder
    else:
        FOLDER_PATH = DEFAULT_FOLDER
    
    # 启动GUI
    app = CoordinateSelector(FOLDER_PATH, OUTPUT_CSV)
    app.run()

