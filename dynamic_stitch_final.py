import json
import os
from pydub import AudioSegment
from pydub.scipy_effects import low_pass_filter
from pathlib import Path

def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def apply_spotify_transition(audio_A, seg_A, seg_B):
    """
    根據前後片段的 MIR 特徵，動態決定轉場模式
    """
    features_A = seg_A.get("features", {})
    features_B = seg_B.get("features", {})
    
    vocal_A = features_A.get("vocal_density", 0)
    vocal_B = features_B.get("vocal_density", 0)
    energy_lift = features_B.get("energy_lift", 0)
    
    current_section = seg_A.get("section", "").lower()
    next_section = seg_B.get("section", "").lower()
    
    print(f"\n[轉場分析] {seg_A['section']} ({vocal_A:.2f}) -> {seg_B['section']} ({vocal_B:.2f})")
    
    # ── Case1 ：低通濾波轉場  (情緒攀升爆發)──
    if next_section == "chorus" and current_section != "chorus" and energy_lift > 0.2:
        print(" Low-Pass Filter Burst)")
        fade_duration = 500
        normal_part = audio_A[:-fade_duration]
        muffled_part = low_pass_filter(audio_A[-fade_duration:], cutoff_freq=2500)
        audio_A = normal_part + muffled_part
        overlap_ms = 1500  
        
    # ── Case2：人聲防撞(動態權重) ──
    elif vocal_A > 0.5 and vocal_B > 0.5:
        print(" Dynamic Vocal Anti-Collision ")
        max_vocal = max(vocal_A, vocal_B)
        
        dynamic_gain = -20.0 - (max_vocal * 10.0)
        dynamic_duration = int(400 + (max_vocal * 1000))
        
        overlap_ms = int(1500 * (1.0 - max_vocal * 0.9))
        overlap_ms = max(overlap_ms, 1500)  
        
        audio_A = audio_A.fade(to_gain=dynamic_gain, end=len(audio_A), duration=dynamic_duration)
        print(f"     -> 衰減: {dynamic_gain:.1f}dB, 歷時: {dynamic_duration}ms, 重疊: {overlap_ms}ms")
        
    # ── Case3 ：動態權重平滑轉場 ──
    else:
        print(" Smooth Chordal Crossfade ")
        max_vocal = max(vocal_A, vocal_B)
        
        overlap_ms = int(2000 * (1.0 - max_vocal * 0.8))
        overlap_ms = max(overlap_ms, 500)  
        print(f"     -> 調整重疊時間為：{overlap_ms} ms")
        
    return audio_A, overlap_ms

def main(mp3_path, json_path, output_path):
    print("1. loading original MP3 file...")
    full_song = AudioSegment.from_file(mp3_path, format="mp3")
    
    global_target_dbfs = full_song.dBFS
    print(f"   -> 偵測到整曲全局平均音量基準：{global_target_dbfs:.2f} dBFS")
    
    print("2. 正在載入選定的精華片段 JSON...")
    candidates = load_json_data(json_path)
    
    if not candidates:
        print("❌ JSON 內無資料")
        return

    print("3. 開始智慧前瞻拼接與雙軌全域對齊流程...")
    
    # 初始化第一個片段
    seg_first = candidates[0]
    start_ms = int(seg_first["start_time"] * 1000)
    end_ms = int(seg_first["end_time"] * 1000)
    final_summary = full_song[start_ms:end_ms]
    
    # Global Loudness Balancing 全域音量平衡第一步 
    summary_norm_offset = global_target_dbfs - final_summary.dBFS
    final_summary = final_summary + summary_norm_offset
    
    for i in range(len(candidates) - 1):
        seg_A = candidates[i]
        seg_B = candidates[i+1]
        
        # 3a. 運算轉場模式與重疊秒數
        final_summary, overlap_ms = apply_spotify_transition(final_summary, seg_A, seg_B)
        
        # 3b. 切出即將進場的 B 片段
        start_ms_B = int(seg_B["start_time"] * 1000)
        end_ms_B = int(seg_B["end_time"] * 1000)
        audio_B = full_song[start_ms_B:end_ms_B]
        
        # 全域音量平衡第二步：確保 B 基底音量正常，不受到前人連鎖干擾
        b_norm_offset = global_target_dbfs - audio_B.dBFS
        audio_B = audio_B + b_norm_offset
        
        # 3c. 微觀邊界偵測
        check_window = max(overlap_ms, 100)
        loudness_tail_A = final_summary[-check_window:].dBFS
        loudness_head_B = audio_B[:check_window].dBFS
        
        if loudness_tail_A != float('-inf') and loudness_head_B != float('-inf'):
            db_delta = loudness_tail_A - loudness_head_B
            smooth_factor = 0.7  
            adjusted_db_delta = db_delta * smooth_factor
            
            # 終極優化：採用「漸變式增益（Gain Ramp）」代替全段推拉！
            # 讓 B 的開頭與 A 完美對齊，但在兩倍重疊時間內（例如 2~3 秒）絲滑地回歸標準全局音量
            ramp_duration = min(int(check_window * 2), len(audio_B))
            audio_B = audio_B.fade(from_gain=adjusted_db_delta, to_gain=0.0, start=0, duration=ramp_duration)
            print(f"     [智慧漸變對齊] A結尾: {loudness_tail_A:.1f}dB | B開頭: {loudness_head_B:.1f}dB -> 邊界初始補償: {adjusted_db_delta:+.1f}dB (於 {ramp_duration}ms 內歸零)")
        
        # 3d. 結尾優化
        if i == len(candidates) - 2:
            print(f"\n[智慧結尾] 正在分析最後一個片段的內容 ({seg_B['section']})...")
            is_high_energy = seg_B["section"].lower() in ["chorus", "bridge"] or seg_B["features"]["rms"] > 0.4
            
            if is_high_energy:
                print(" 偵測到強烈段落中斷！施加 2.5 秒『長度漸穩淡出』")
                audio_B = audio_B.fade(to_gain=-40.0, end=len(audio_B), duration=2500)
            else:
                print(" 偵測到平緩段落結尾，施加 2.0 秒常規淡出")
                audio_B = audio_B.fade(to_gain=-60.0, end=len(audio_B), duration=2000)
        
        # 3e. 正式合併
        final_summary = final_summary.append(audio_B, crossfade=overlap_ms)
            
    # 防爆音系統（Peak Limiter / Normalization）
    # 檢查整首摘要的最高峰值是否會衝破天花板，若是，則自動安全降載
    if final_summary.max_dBFS > -1.0:
        print(f"\n🛡️ [防爆音系統啟動] 偵測到音訊峰值過高 ({final_summary.max_dBFS:.2f} dBFS)，正在進行智慧安全降載...")
        final_summary = final_summary.normalize(headroom=1.0) # 強制將最高峰值定格在安全線 -1.0 dBFS
        
    print(f"\n4. 拼接完成！總長度：{len(final_summary)/1000:.2f} 秒")
    print(f"5. 正在匯出檔案至：{output_path}")
    final_summary.export(output_path, format="mp3", bitrate="192k")
    print("測試音訊智慧全域平衡 + 漸變對齊生成成功")
    
if __name__ == "__main__":
    __path_dir__ = Path(__file__).parent
    results_dir = __path_dir__ / "results" / "selected_candidates"
    MP3_FILE = results_dir / "Heavy Senerade.mp3"
    JSON_FILE = results_dir / "Heavy Senerade_selected_candidates.json"
    OUTPUT_FILE = results_dir / "dynamic_stitched_Heavy Senerade_summary.mp3"

    main(MP3_FILE, JSON_FILE, OUTPUT_FILE)

