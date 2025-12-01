# -*- coding: utf-8 -*-
"""
深度情報から3D点群データを生成し、RANSACで床面を特定後、
その床面の境界（崖）を検出することで、下り階段を認識するプログラム。

"""

import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

# --- 型エイリアス定義 ---
PcdPair = Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]
# (is_floor, (inlier, outlier), plane_model)
FloorDetectionResult = Tuple[bool, PcdPair, np.ndarray | None]

# (all_cliffs_mask, combined_cliff_mask, median_distance, morphology_masks_dict, valid_z_values)
CliffDetectionResult = Tuple[np.ndarray, np.ndarray, float, Dict[str, np.ndarray], np.ndarray]

# (all_cliffs, combined_cliff, dist, inlier, outlier, morph_masks, valid_z_values)
FrameProcessingResult = Tuple[np.ndarray, np.ndarray, float, o3d.geometry.PointCloud, o3d.geometry.PointCloud, Dict[str, np.ndarray], np.ndarray]


@dataclass
class AppConfig:
    """アプリケーションの全設定を保持するデータクラス"""
    # 1. カメラ設定
    WIDTH: int = 640
    HEIGHT: int = 480
    FPS: int = 30

    # 位置合わせ（Align）を有効にするか
    ENABLE_ALIGNMENT: bool = True

    # 2. 深度フィルター設定 (ノイズ除去)
    USE_POST_PROCESSING: bool = True
    SPATIAL_FILTER_STRENGTH: int = 2
    TEMPORAL_FILTER_SMOOTHNESS: float = 0.5

    # 3. 3D点群設定 (床面検出のRANSACでのみ使用)
    DOWNSAMPLE_VOXEL_SIZE: float = 0.03
    MAX_VIEW_DISTANCE: float = 6.0

    # 4. RANSAC床面検出設定
    RANSAC_DISTANCE_THRESHOLD: float = 0.1
    RANSAC_N: int = 3
    RANSAC_ITERATIONS: int = 100
    
    RANSAC_CANDIDATE_Y_START_PERCENT: float = 0.2 
    RANSAC_CANDIDATE_X_WIDTH_PERCENT: float = 0.5
    RANSAC_MIN_CANDIDATE_INLIERS: int = 100 

    # 5. 崖検出（下り階段の縁）設定
    CLIFF_DEPTH_THRESHOLD: float = 0.3
    CLIFF_EDGE_COLOR: Tuple[int, int, int] = (0, 165, 255)  # BGR: Orange
    COMBINED_CLIFF_COLOR: Tuple[int, int, int] = (0, 0, 255)  # BGR: Red
    AVERAGING_KERNEL_SIZE: int = 7
    
    # 2D面積フィルタ設定
    MIN_CLIFF_AREA_PIXELS: int = 30  # 小さすぎるノイズは無視
    
    # クラスタリング統合用設定
    CLUSTERING_CONNECT_KERNEL_SIZE: Tuple[int, int] = (6, 6) 
    CLUSTERING_CONNECT_ITERATIONS: int = 2 
    
    # 内側（床側）計測用の設定
    INNER_MEASURE_KERNEL_SIZE: int = 5
    INNER_MEASURE_ITERATIONS: int = 2
    
    FILL_ALPHA: float = 0.4 

    # 6. モルフォロジー処理設定
    MORPH_KERNEL_SIZE: Tuple[int, int] = (5, 5)
    MORPH_CLOSE_ITERATIONS: int = 3
    MORPH_OPEN_ITERATIONS: int = 1
    MORPH_DILATE_ITERATIONS: int = 1

    # 7. 可視化・デバッグ設定
    VISUALIZE_MORPHOLOGY: bool = True
    CLIFF_DRAW_KERNEL_SIZE: Tuple[int, int] = (3, 3)
    CLIFF_DRAW_ITERATIONS: int = 1
    
    # 生データの無効領域をマゼンタで表示するかどうか
    SHOW_RAW_INVALID_REGIONS: bool = True
    RAW_INVALID_COLOR: Tuple[int, int, int] = (255, 0, 255) # Magenta

    # 8. 表示調整
    BRIGHTNESS_ADJUSTMENT_ENABLED: bool = True
    BRIGHTNESS_OFFSET: int = 30
    CONTRAST_FACTOR: float = 1.0
    GAMMA_CORRECTION_ENABLED: bool = False
    GAMMA_VALUE: float = 1.2


class StairDetector:
    """
    深度カメラからの情報を用いて、下り階段をリアルタイムで検出し、可視化するクラス。
    """
    COLOR_WINDOW = "Color View"
    DEPTH_WINDOW = "Depth View"
    MORPHOLOGY_WINDOW = "Morphology Steps"

    def __init__(self, config: AppConfig):
        self.config = config
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.is_paused = False
        
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, config.SPATIAL_FILTER_STRENGTH)
        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, config.TEMPORAL_FILTER_SMOOTHNESS)
        
        self.morph_kernel = np.ones(self.config.MORPH_KERNEL_SIZE, np.uint8)
        self.draw_kernel = np.ones(self.config.CLIFF_DRAW_KERNEL_SIZE, np.uint8)
        
        self.clustering_connect_kernel = np.ones(self.config.CLUSTERING_CONNECT_KERNEL_SIZE, np.uint8)
        self.inner_measure_kernel = np.ones((self.config.INNER_MEASURE_KERNEL_SIZE, self.config.INNER_MEASURE_KERNEL_SIZE), np.uint8)
        
        self.total_frames = 0
        self.detected_frames = 0
        self.is_bag_playback = False

    # --- 1. 初期化・セットアップ関連 ---
    def _select_bag_file(self) -> str | None:
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(title=".bagファイルを選択してください", filetypes=[("BAG files", "*.bag")])
        root.destroy()
        return file_path if file_path else None

    def _setup_realsense(self) -> rs.pipeline_profile | None:
        rs_config = rs.config()
        print(".bagファイルを選択してください。キャンセルするとライブカメラが起動します。")
        bag_file_path = self._select_bag_file()
        if bag_file_path:
            print(f"'{bag_file_path}' を読み込んでいます...")
            try:
                rs_config.enable_device_from_file(bag_file_path, repeat_playback=False)
                self.is_bag_playback = True 
            except RuntimeError as e:
                print(f"エラー: .bagファイルの読み込みに失敗しました: {e}")
                return None
        else:
            print("ライブカメラを起動します...")
            self.is_bag_playback = False
            try:
                rs_config.enable_stream(rs.stream.depth, self.config.WIDTH, self.config.HEIGHT, rs.format.z16, self.config.FPS)
                rs_config.enable_stream(rs.stream.color, self.config.WIDTH, self.config.HEIGHT, rs.format.bgr8, self.config.FPS)
            except RuntimeError as e:
                print(f"エラー: ライブカメラのストリーム設定に失敗しました: {e}")
                return None
        try:
            profile = self.pipeline.start(rs_config)
            if bag_file_path:
                playback = profile.get_device().as_playback()
                if playback: playback.set_real_time(False)
            return profile
        except RuntimeError as e:
            print(f"エラー: RealSenseの起動に失敗しました: {e}")
            return None

    def _setup_windows(self):
        cv2.namedWindow(self.COLOR_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty(self.COLOR_WINDOW, cv2.WND_PROP_TOPMOST, 1)
        if self.config.VISUALIZE_MORPHOLOGY:
            cv2.namedWindow(self.MORPHOLOGY_WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setWindowProperty(self.MORPHOLOGY_WINDOW, cv2.WND_PROP_TOPMOST, 1)
        else:
            cv2.namedWindow(self.DEPTH_WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setWindowProperty(self.DEPTH_WINDOW, cv2.WND_PROP_TOPMOST, 1)

    # --- 2. メインループ ---
    def run(self):
        profile = self._setup_realsense()
        if not profile: return
        
        # [v3.27] 常にカラーカメラのストリームプロファイル（＝位置合わせ先）を使用
        stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            
        intrinsics = stream_profile.get_intrinsics()
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        )
        
        depth_sensor = profile.get_device().first_depth_sensor()
        if not depth_sensor: return
        depth_scale = depth_sensor.get_depth_scale()
        
        self._setup_windows()
        
        print("検出を開始します。'q'キーで終了、スペースキーで一時停止します。")
        try:
            while True:
                if self._check_exit_conditions(): break
                
                if self.is_paused:
                    if not self._handle_keyboard_input(): break
                    continue
                
                success, frames = self.pipeline.try_wait_for_frames(100)
                if not success:
                    if self.is_bag_playback: 
                        print(".bagファイルの再生が終了しました。")
                        break
                    else: continue
                
                # 常に位置合わせ(Align)を実行
                processed_frames = self.align.process(frames)

                depth_frame = processed_frames.get_depth_frame()
                color_frame = processed_frames.get_color_frame()
                
                if not depth_frame or not color_frame: continue
                
                depth_image_raw = np.asanyarray(depth_frame.get_data()).copy()
                self.total_frames += 1
                
                if self.config.USE_POST_PROCESSING:
                    depth_frame = self.spatial_filter.process(depth_frame)
                    depth_frame = self.temporal_filter.process(depth_frame)
                
                depth_image_filtered = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                results = self._process_frame(depth_image_filtered, depth_scale, pinhole_camera_intrinsic, intrinsics)
                
                combined_cliff_mask = results[1]
                distance_val = results[2]
                valid_z_values = results[6]

                is_detected = np.any(combined_cliff_mask)

                if is_detected:
                    self.detected_frames += 1
                    print(f"\n[Frame {self.total_frames:04d}] Stair Detected | Mean Dist: {distance_val:.4f} m")
                    if len(valid_z_values) > 0:
                        if len(valid_z_values) > 10:
                            print(f"  Z-Depths (First 10 of {len(valid_z_values)}): {valid_z_values[:10]}")
                        else:
                            print(f"  Z-Depths: {valid_z_values}")
                    else:
                         print(f"  Z-Depths: No valid data in inner region.")
                else:
                    pass
                
                self._update_display(color_image, depth_image_filtered, depth_image_raw, results)
                
                if self.config.VISUALIZE_MORPHOLOGY: self._update_morphology_display(results[5])
                
                if not self._handle_keyboard_input(): break

        finally: self._cleanup()

    def _process_frame(self, depth_image: np.ndarray, depth_scale: float, 
                       pinhole_camera_intrinsic: o3d.camera.PinholeCameraIntrinsic, 
                       intrinsics: rs.intrinsics) -> FrameProcessingResult:
        is_floor, (inlier_cloud, outlier_cloud), plane_model = self._detect_floor_plane_from_depth(
            depth_image, depth_scale, pinhole_camera_intrinsic
        )
        
        empty_mask = np.zeros((self.config.HEIGHT, self.config.WIDTH), dtype=np.uint8)
        morphology_masks: Dict[str, np.ndarray] = {}
        empty_z_values = np.array([])
        
        if not is_floor or plane_model is None:
            return empty_mask, empty_mask, 0.0, inlier_cloud, outlier_cloud, morphology_masks, empty_z_values
        
        all_cliffs, largest_cliff, dist, morph_masks, valid_z_values = self._detect_cliff_from_floor(
            plane_model, depth_image, intrinsics, depth_scale
        )
        return all_cliffs, largest_cliff, dist, inlier_cloud, outlier_cloud, morph_masks, valid_z_values

    # --- 3. コア検出ロジック ---

    def _detect_floor_plane_from_depth(self, depth_image: np.ndarray, depth_scale: float, 
                                       pinhole_camera_intrinsic: o3d.camera.PinholeCameraIntrinsic) -> FloorDetectionResult:
        # 床面検出
        empty_pcd = o3d.geometry.PointCloud()
        
        y_start_pixel = int(self.config.HEIGHT * self.config.RANSAC_CANDIDATE_Y_START_PERCENT)
        x_width_pixels = int(self.config.WIDTH * self.config.RANSAC_CANDIDATE_X_WIDTH_PERCENT)
        x_margin = (self.config.WIDTH - x_width_pixels) // 2
        x_start_pixel = x_margin
        x_end_pixel = self.config.WIDTH - x_margin
        
        depth_image_candidate = np.zeros_like(depth_image)
        depth_image_candidate[y_start_pixel:self.config.HEIGHT, x_start_pixel:x_end_pixel] = \
            depth_image[y_start_pixel:self.config.HEIGHT, x_start_pixel:x_end_pixel]
        
        o3d_depth_candidate = o3d.geometry.Image(depth_image_candidate)
        pcd_candidate_full = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth_candidate, pinhole_camera_intrinsic, depth_scale=1.0 / depth_scale,
            depth_trunc=self.config.MAX_VIEW_DISTANCE
        )
        pcd_candidates = pcd_candidate_full.voxel_down_sample(self.config.DOWNSAMPLE_VOXEL_SIZE)
        
        o3d_depth_all = o3d.geometry.Image(depth_image)
        pcd_all_full = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth_all, pinhole_camera_intrinsic, depth_scale=1.0 / depth_scale, depth_trunc=self.config.MAX_VIEW_DISTANCE
        )
        pcd_all_downsampled = pcd_all_full.voxel_down_sample(self.config.DOWNSAMPLE_VOXEL_SIZE)
        
        if not pcd_candidates.has_points():
            return False, (empty_pcd, pcd_all_downsampled), None

        plane_model, inliers_in_candidates = pcd_candidates.segment_plane(
            distance_threshold=self.config.RANSAC_DISTANCE_THRESHOLD, ransac_n=self.config.RANSAC_N, num_iterations=self.config.RANSAC_ITERATIONS
        )

        outlier_cloud = pcd_all_downsampled
        
        if not pcd_all_downsampled.has_points(): 
            return False, (empty_pcd, empty_pcd), None
        
        if len(inliers_in_candidates) < self.config.RANSAC_MIN_CANDIDATE_INLIERS:
             return False, (pcd_candidates, outlier_cloud), None

        a, b, c, d = plane_model
        all_points = np.asarray(pcd_all_downsampled.points)
        if all_points.shape[0] == 0: return False, (empty_pcd, outlier_cloud), None
        
        distances = np.abs(all_points @ np.array([a, b, c]) + d)
        final_inlier_indices = np.where(distances < self.config.RANSAC_DISTANCE_THRESHOLD)[0]
        
        inlier_cloud = pcd_all_downsampled.select_by_index(final_inlier_indices)
        outlier_cloud = pcd_all_downsampled.select_by_index(final_inlier_indices, invert=True)
        
        return True, (inlier_cloud, outlier_cloud), plane_model

    def _detect_cliff_from_floor(self, plane_model: np.ndarray, depth_image: np.ndarray, 
                                  intrinsics: rs.intrinsics, depth_scale: float) -> CliffDetectionResult:
        floor_mask = self._create_mask_from_plane_equation(depth_image, intrinsics, depth_scale, plane_model)

        floor_mask_closed = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=self.config.MORPH_CLOSE_ITERATIONS)
        floor_mask_opened = cv2.morphologyEx(floor_mask_closed, cv2.MORPH_OPEN, self.morph_kernel, iterations=self.config.MORPH_OPEN_ITERATIONS)
        floor_mask_largest_component = self._extract_largest_connected_component(floor_mask_opened)
        floor_mask_filled = self._fill_holes_in_mask(floor_mask_largest_component)
        dilated_floor = cv2.dilate(floor_mask_filled, self.morph_kernel, iterations=self.config.MORPH_DILATE_ITERATIONS)
        boundary = dilated_floor - floor_mask_filled
        
        morphology_masks = {
            "Initial Mask": floor_mask, "After Closing": floor_mask_closed, "After Opening": floor_mask_opened, 
            "Largest Comp.": floor_mask_largest_component, "After Hole Fill": floor_mask_filled, "Boundary": boundary
        }
        empty_mask = np.zeros_like(floor_mask)
        empty_z_values = np.array([])
        
        if not np.any(boundary): return empty_mask, empty_mask, 0.0, morphology_masks, empty_z_values

        cliff_mask_initial = self._compare_boundary_depths(boundary, depth_image, floor_mask_filled, dilated_floor, depth_scale)
        morphology_masks["Raw Cliff"] = cliff_mask_initial

        all_cliffs, largest_cliff, median_dist, connect_mask, valid_z_values = self._filter_cliffs_on_2d_image(
            cliff_mask_initial, depth_image, depth_scale, floor_mask_filled
        )
        
        morphology_masks["Connect Mask"] = connect_mask 
        morphology_masks["All Cliffs"] = all_cliffs
        morphology_masks["Combined Cliff"] = largest_cliff

        return all_cliffs, largest_cliff, median_dist, morphology_masks, valid_z_values

    def _filter_cliffs_on_2d_image(self, cliff_mask: np.ndarray, depth_image: np.ndarray, depth_scale: float, floor_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        """
        [v3.26/v3.27 改良] 
        - 距離計算時、崖マスクを膨張させ、床面マスク(floor_mask)との共通領域（内側）を使って計算する。
        """
        # 1. 統合用マスクの作成
        connection_mask = cv2.dilate(cliff_mask, self.clustering_connect_kernel, iterations=self.config.CLUSTERING_CONNECT_ITERATIONS)
        
        # 2. ラベリング
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connection_mask, connectivity=8)
        
        all_cliffs_mask = np.zeros_like(cliff_mask)
        largest_cliff_mask = np.zeros_like(cliff_mask)
        median_distance = 0.0
        valid_z_values = np.array([])
        
        max_area = 0
        largest_label_idx = -1
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.config.MIN_CLIFF_AREA_PIXELS: continue
            
            group_mask_dilated = (labels == i).astype(np.uint8) * 255
            group_mask_original = cv2.bitwise_and(cliff_mask, group_mask_dilated)
            
            all_cliffs_mask = cv2.bitwise_or(all_cliffs_mask, group_mask_original)
            
            if area > max_area:
                max_area = area
                largest_label_idx = i
        
        if largest_label_idx != -1:
            largest_group_dilated = (labels == largest_label_idx).astype(np.uint8) * 255
            largest_cliff_mask = cv2.bitwise_and(cliff_mask, largest_group_dilated)
            
            # [v3.26/v3.27] 内側計測: 崖マスクを膨張 -> 床マスクとAND -> その領域で距離計算
            # 1. 崖マスクを膨張させて「周辺」を含める
            dilated_cliff = cv2.dilate(largest_cliff_mask, self.inner_measure_kernel, iterations=self.config.INNER_MEASURE_ITERATIONS)
            # 2. 床面（内側）との共通部分だけ残す ＝ 「崖のすぐ手前の床」
            inner_region_mask = cv2.bitwise_and(dilated_cliff, floor_mask)
            
            # 距離計算 (Z深度平均 + 全データ取得)
            median_distance, valid_z_values = self._calculate_distance(inner_region_mask, depth_image, depth_scale)
        
        return all_cliffs_mask, largest_cliff_mask, median_distance, connection_mask, valid_z_values

    def _calculate_distance(self, mask: np.ndarray, depth_image: np.ndarray, depth_scale: float) -> Tuple[float, np.ndarray]:
        """
        マスク領域の距離と、その元データ全件を返す。
        戻り値: (平均距離, 全有効深度データの配列)
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0: return 0.0, np.array([])
        
        # RealSenseの値をメートル単位に変換
        z_depths = depth_image[ys, xs].astype(np.float32) * depth_scale
        
        # 有効な深度のみ抽出
        valid_mask = (z_depths > 0) & (z_depths <= self.config.MAX_VIEW_DISTANCE)
        valid_z = z_depths[valid_mask]
        
        if len(valid_z) > 0:
            return float(np.mean(valid_z)), valid_z
        return 0.0, np.array([])

    # --- その他ヘルパー ---
    def _extract_largest_connected_component(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1: return mask
        largest_label_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        largest_component_mask = np.zeros_like(mask)
        largest_component_mask[labels == largest_label_index] = 255
        return largest_component_mask

    def _fill_holes_in_mask(self, mask: np.ndarray) -> np.ndarray:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = mask.copy()
        if hierarchy is None: return filled_mask
        for i, hier in enumerate(hierarchy[0]):
            if hier[3] != -1: cv2.drawContours(filled_mask, contours, i, 255, -1)
        return filled_mask

    def _create_mask_from_plane_equation(self, depth_image: np.ndarray, intrinsics: rs.intrinsics,
                                         depth_scale: float, plane_model: np.ndarray) -> np.ndarray:
        h, w = self.config.HEIGHT, self.config.WIDTH
        depth_in_meters = depth_image.astype(np.float32) * depth_scale
        pixel_y, pixel_x = np.mgrid[0:h, 0:w]
        z = depth_in_meters
        x = (pixel_x - intrinsics.ppx) * z / intrinsics.fx
        y = (pixel_y - intrinsics.ppy) * z / intrinsics.fy
        points_flat = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        a, b, c, d = plane_model
        distances = np.abs(points_flat @ np.array([a, b, c]) + d)
        z_flat = z.flatten()
        valid_depth_mask = (z_flat > 0) & (z_flat <= self.config.MAX_VIEW_DISTANCE)
        floor_mask_flat = (distances < self.config.RANSAC_DISTANCE_THRESHOLD) & valid_depth_mask
        return (floor_mask_flat.reshape(h, w) * 255).astype(np.uint8)

    def _compare_boundary_depths(self, boundary: np.ndarray, depth_image: np.ndarray,
                                 inner_mask: np.ndarray, dilated_mask: np.ndarray,
                                 depth_scale: float) -> np.ndarray:
        boundary_ys, boundary_xs = np.where(boundary > 0)
        valid_depth_indices = depth_image[boundary_ys, boundary_xs] > 0
        valid_xs = boundary_xs[valid_depth_indices]
        valid_ys = boundary_ys[valid_depth_indices]
        cliff_mask = np.zeros_like(boundary)
        if len(valid_xs) == 0: return cliff_mask

        depth_in_meters = depth_image.astype(np.float32) * depth_scale
        range_mask = (depth_in_meters > 0) & (depth_in_meters <= self.config.MAX_VIEW_DISTANCE)
        outer_region_mask = dilated_mask - inner_mask
        k_size = (self.config.AVERAGING_KERNEL_SIZE, self.config.AVERAGING_KERNEL_SIZE)

        def calculate_avg_and_count(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            combined_mask = (mask / 255.0) * range_mask
            depth_map = depth_in_meters * combined_mask
            counts_map = combined_mask.astype(np.float32)
            sum_of_depths = cv2.boxFilter(depth_map, -1, k_size, normalize=False)
            count_of_pixels = cv2.boxFilter(counts_map, -1, k_size, normalize=False)
            avg_depth_map = sum_of_depths / (count_of_pixels + 1e-6)
            return avg_depth_map, count_of_pixels

        avg_inner_depth_map, count_inner_pixels_map = calculate_avg_and_count(inner_mask)
        avg_outer_depth_map, count_outer_pixels_map = calculate_avg_and_count(outer_region_mask)

        avg_inner_depths = avg_inner_depth_map[valid_ys, valid_xs]
        avg_outer_depths = avg_outer_depth_map[valid_ys, valid_xs]
        count_inner_pixels = count_inner_pixels_map[valid_ys, valid_xs]
        count_outer_pixels = count_outer_pixels_map[valid_ys, valid_xs]

        is_cliff_type1 = (count_outer_pixels > 0) & \
                         (avg_outer_depths > avg_inner_depths) & \
                         ((avg_outer_depths - avg_inner_depths) > self.config.CLIFF_DEPTH_THRESHOLD)
        is_cliff_type2 = (count_outer_pixels == 0) & (count_inner_pixels > 0)
        is_cliff = is_cliff_type1 | is_cliff_type2

        cliff_mask[valid_ys[is_cliff], valid_xs[is_cliff]] = 255
        return cliff_mask

    # --- 4. 可視化・表示関連 ---

    def _apply_brightness_contrast(self, img):
        if self.config.BRIGHTNESS_ADJUSTMENT_ENABLED:
            img = cv2.convertScaleAbs(img, alpha=self.config.CONTRAST_FACTOR, beta=self.config.BRIGHTNESS_OFFSET)
        return img

    def _apply_gamma_correction(self, img):
        if self.config.GAMMA_CORRECTION_ENABLED and self.config.GAMMA_VALUE != 1.0:
            inv_gamma = 1.0 / self.config.GAMMA_VALUE
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)
        return img

    def _update_display(self, color_image: np.ndarray, depth_image: np.ndarray, depth_image_raw: np.ndarray, results: FrameProcessingResult):
        all_cliffs_mask, combined_cliff_mask, distance, *_ = results
        display_image = color_image.copy()
        display_image = self._apply_brightness_contrast(display_image)
        display_image = self._apply_gamma_correction(display_image)
        
        # 1. フィルタ済み深度の黒塗り
        display_image[depth_image == 0] = (0, 0, 0)
        
        # 2. 生データの無効領域をマゼンタで上書き
        if self.config.SHOW_RAW_INVALID_REGIONS:
            display_image[depth_image_raw == 0] = self.config.RAW_INVALID_COLOR
        
        # [v3.29] 階段検出のオーバーレイ表示（赤・オレンジ）を削除
        # if np.any(combined_cliff_mask): ... (削除)
        # if np.any(all_cliffs_mask): ... (削除)
        
        self._draw_text_with_bg(display_image, "Press 'q' to quit (Space to pause)", (10, 30))
        
        # [v3.29] 距離表示を削除
        # dist_text = ... (削除)
        # self._draw_text_with_bg(display_image, dist_text, (10, 60)) (削除)
        
        if self.is_paused: self._draw_text_with_bg(display_image, "PAUSED", (self.config.WIDTH // 2 - 50, self.config.HEIGHT // 2), scale=1.2)
        
        cv2.imshow(self.COLOR_WINDOW, display_image)
        
        if not self.config.VISUALIZE_MORPHOLOGY:
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap[depth_image == 0] = 0
            cv2.imshow(self.DEPTH_WINDOW, depth_colormap)

    def _update_morphology_display(self, morph_masks: Dict[str, np.ndarray]):
        if not morph_masks: return
        h, w = self.config.HEIGHT // 2, self.config.WIDTH // 3
        masks_to_display = [
            ("Initial", morph_masks.get("Initial Mask")),
            ("Opened", morph_masks.get("After Opening")),
            ("Boundary", morph_masks.get("Boundary")),
            ("Raw Cliff", morph_masks.get("Raw Cliff")), 
            ("Connect(Debug)", morph_masks.get("Connect Mask")), 
            ("Largest(Red)", morph_masks.get("Combined Cliff")) 
        ]
        display_rows = []
        for i in range(0, len(masks_to_display), 3):
            row_images = []
            for j in range(3):
                if i + j < len(masks_to_display):
                    name, mask = masks_to_display[i + j]
                    if mask is None:
                        img = np.zeros((h, w, 3), dtype=np.uint8)
                        cv2.putText(img, f"{name} (N/A)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        row_images.append(img)
                        continue
                    resized_mask = cv2.resize(mask, (w, h))
                    bgr_mask = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
                    cv2.putText(bgr_mask, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    row_images.append(bgr_mask)
                else: row_images.append(np.zeros((h, w, 3), dtype=np.uint8))
            display_rows.append(np.hstack(row_images))
        try: cv2.imshow(self.MORPHOLOGY_WINDOW, np.vstack(display_rows))
        except cv2.error: pass

    def _draw_text_with_bg(self, img, text, pos, scale=0.8, color=(0, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), base = cv2.getTextSize(text, font, scale, 2)
        top_left = (max(0, pos[0]-5), max(0, pos[1]-text_h-5))
        bottom_right = (min(img.shape[1], pos[0]+text_w+5), min(img.shape[0], pos[1]+base+5))
        if top_left[1] < bottom_right[1] and top_left[0] < bottom_right[0]:
            sub = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = \
                cv2.addWeighted(sub, 0.5, np.full(sub.shape, 0, dtype=np.uint8), 0.5, 1.0)
        cv2.putText(img, text, pos, font, scale, color, 2, cv2.LINE_AA)

    def _handle_keyboard_input(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): return False
        if key == ord(' '): self.is_paused = not self.is_paused
        return True

    def _check_exit_conditions(self) -> bool:
        if not self._handle_keyboard_input(): return True
        try:
            if cv2.getWindowProperty(self.COLOR_WINDOW, cv2.WND_PROP_VISIBLE) < 1: return True
        except: return True
        return False

    def _cleanup(self):
        print("クリーンアップ処理を行い、終了します...")
        print("\n--- 処理結果 ---")
        if self.total_frames > 0:
            detection_rate = (self.detected_frames / self.total_frames) * 100
            print(f"総処理フレーム数: {self.total_frames} フレーム")
            print(f"階段検出フレーム数: {self.detected_frames} フレーム")
            print(f"階段検出率: {detection_rate:.2f} %")
        print("----------------\n")
        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app_config = AppConfig()
    detector = StairDetector(app_config)
    detector.run()
