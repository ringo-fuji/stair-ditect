import pyrealsense2 as rs
import numpy as np
import cv2
import simpleaudio as sa
import time

# --- 設定 (ここを調整できます) ---
WIDTH = 640  # カメラの解像度 (幅)
HEIGHT = 480 # カメラの解像度 (高さ)
FPS = 30     # カメラのフレームレート

MIN_DIST = 0.2  # 音が鳴り始める最短距離 (メートル)
MAX_DIST = 3.0  # 音が鳴り止む（最も低くなる）最長距離 (メートル)

MIN_FREQ = 300  # 遠い時の音の高さ (Hz)
MAX_FREQ = 2000 # 近い時の音の高さ (Hz)

BEEP_DURATION_MS = 50 # ビープ音の長さ (ミリ秒)
SAMPLE_RATE = 44100   # オーディオのサンプルレート
# ------------------------------

def generate_tone(frequency, duration_ms, sample_rate):
    """
    指定された周波数とデュレーションのサイン波（ビープ音）を生成する
    """
    duration_sec = duration_ms / 1000.0
    t = np.linspace(0., duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = np.sin(2 * np.pi * frequency * t)
    
    # 16-bit PCM形式に変換
    audio *= 32767 / np.max(np.abs(audio))
    audio = audio.astype(np.int16)
    return audio

def main():
    # RealSenseパイプラインの初期化
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 深度ストリームとカラーストリームを有効化
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    
    print("RealSenseカメラを起動中...")
    profile = pipeline.start(config)

    # 画面の中心座標を計算
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    
    print(f"画面中央 ({center_x}, {center_y}) の距離を監視します。")
    print("'q'キーを押すと終了します。")

    current_play_obj = None # 現在再生中のサウンドオブジェクト

    try:
        while True:
            # フレームセットを待機
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # --- 距離の取得 ---
            # 指定した中心ピクセルの距離（メートル）を取得
            distance = depth_frame.get_distance(center_x, center_y)

            # --- 周波数の計算 ---
            frequency = 0 # デフォルトは無音
            if distance > 0: # 距離が0より大きい（有効な）場合
                # 距離を [MIN_DIST, MAX_DIST] の範囲にクリップ（制限）
                clamped_dist = np.clip(distance, MIN_DIST, MAX_DIST)
                
                # 距離を 0.0 (近い) 〜 1.0 (遠い) の値に正規化
                norm_dist = (clamped_dist - MIN_DIST) / (MAX_DIST - MIN_DIST)
                
                # 距離が近い(0.0)ほどMAX_FREQ、遠い(1.0)ほどMIN_FREQになるように線形補間
                frequency = int(MAX_FREQ + (MIN_FREQ - MAX_FREQ) * norm_dist)

            # --- サウンドの再生 ---
            if frequency > 0:
                # 既に音が鳴っていれば停止（音が重ならないように）
                if current_play_obj and current_play_obj.is_playing():
                    current_play_obj.stop()
                    
                # 新しい周波数でビープ音を生成
                audio_data = generate_tone(frequency, BEEP_DURATION_MS, SAMPLE_RATE)
                # 再生（非同期）
                current_play_obj = sa.play_buffer(audio_data, 1, 2, SAMPLE_RATE)
            else:
                # 範囲外なら音を止める
                if current_play_obj and current_play_obj.is_playing():
                    current_play_obj.stop()

            # --- 映像の表示 ---
            color_image = np.asanyarray(color_frame.get_data())
            
            # 画面中央に赤い円を描画
            cv2.circle(color_image, (center_x, center_y), 8, (0, 0, 255), -1) # 赤い点
            
            # 距離と周波数を画面に表示
            text = f"Dist: {distance:.2f} m"
            text_freq = f"Freq: {frequency} Hz"
            cv2.putText(color_image, text, (center_x + 15, center_y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, text_freq, (center_x + 15, center_y + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("RealSense (Press 'q' to quit)", color_image)
            
            # 'q'キーでループを抜ける
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("終了します。")
                break
            
            # CPU負荷を少し下げる（ビープ音の長さに合わせる）
            time.sleep(BEEP_DURATION_MS / 1000.0)

    finally:
        # ストリーミングを停止し、ウィンドウを閉じる
        if current_play_obj:
            current_play_obj.stop()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("カメラを停止しました。")

if __name__ == "__main__":
    main()
