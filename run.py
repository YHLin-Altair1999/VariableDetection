"""
Demo: end-to-end photometry pipeline.
"""
from pipeline import PhotometryPipeline
import os
import sys
from datetime import datetime

def find_unprocessed_folders(base_folder):
    """
    Find all subfolders that have hops processed data (REDUCED_DATA with alignment info)
    but NOT yet went through the photometry pipeline (which generates output_light_curves).
    
    Hops processing creates a REDUCED_DATA folder with FITS files containing alignment
    headers (ALIGN_X0, ALIGN_Y0, ALIGN_U0).
    """
    unprocessed = []
    
    if not os.path.isdir(base_folder):
        print(f"Error: Base folder not found: {base_folder}")
        return unprocessed
    
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        
        folder_exist = os.path.isdir(item_path)
        processed_by_hops = os.path.isdir(os.path.join(item_path, 'REDUCED_DATA'))
        output_path = os.path.join(item_path, 'output_light_curves')
        no_auto_lightcurve_pipeline_yet = not os.path.exists(output_path)
        
        if folder_exist and processed_by_hops and no_auto_lightcurve_pipeline_yet:
            unprocessed.append(item_path)
    
    return sorted(unprocessed)

def display_and_confirm(folders):
    """
    Display unprocessed folders and get user confirmation for processing.
    """
    if not folders:
        print("No unprocessed folders found.")
        return []
    
    print(f"\n{'='*60}")
    print(f"Found {len(folders)} unprocessed folder(s):")
    print(f"{'='*60}")
    
    for i, folder in enumerate(folders, 1):
        folder_name = os.path.basename(folder)
        print(f"{i}. {folder_name}")
    
    print(f"\n{'='*60}")
    user_input = input("Process all folders? (yes/no): ").strip().lower()
    
    if user_input in ['yes', 'y']:
        return folders
    else:
        print("User cancelled processing.")
        return []

def save_folders_log(folders, base_folder):
    """
    Save the list of confirmed folders to a log file with timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"processing_log_{timestamp}.txt"
    log_path = os.path.join(base_folder, log_filename)
    
    try:
        with open(log_path, 'w') as f:
            f.write(f"Processing Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Total folders to process: {len(folders)}\n")
            f.write(f"{'='*60}\n\n")
            
            for i, folder in enumerate(folders, 1):
                folder_name = os.path.basename(folder)
                f.write(f"{i}. {folder_name}\n")
                f.write(f"   Path: {folder}\n\n")
        
        print(f"Log saved: {log_path}")
        return log_path
    except Exception as e:
        print(f"Warning: Could not save log file: {e}")
        return None

def process_folders(folders, f_threshold=50.0, ref_frame=10, autocorr_threshold=5.0, autocorr_max_lag=5):
    """
    Process each folder through the photometry pipeline.
    """
    results = {}
    
    for working_dir in folders:
        folder_name = os.path.basename(working_dir)
        print(f"\n{'='*60}")
        print(f"Processing: {folder_name}")
        print(f"{'='*60}")
        
        try:
            pipe = PhotometryPipeline(
                working_dir=working_dir,
                output_dir=os.path.join(working_dir, './output_light_curves'),
                f_threshold=f_threshold,
                autocorr_threshold=autocorr_threshold,
                autocorr_max_lag=autocorr_max_lag,
            )
            
            # Full run: detect stars → extract → normalize → flag (slope OR nonlinear OR autocorr)
            nonlinear = pipe.run(ref_frame=ref_frame, skip_extraction=False, method='sep')
            
            # Store results
            results[folder_name] = {
                'stars_analysed': len(pipe.rel_lcs),
                'nonlinear_count': len(nonlinear),
                'nonlinear': nonlinear,
                'flag_reasons': {i: pipe.flag_reasons.get(i, []) for i in nonlinear},
                'slope_mean': pipe.slope_mean,
                'slope_std': pipe.slope_std,
            }
            
            print(f"Stars analysed : {len(pipe.rel_lcs)}")
            print(f"Flagged (slope/nonlinear/autocorr): {len(nonlinear)}  {nonlinear}")
            print(f"Slope mean={pipe.slope_mean:.6e}  std={pipe.slope_std:.6e}")
            
        except Exception as e:
            print(f"Error processing {folder_name}: {e}")
            results[folder_name] = {'error': str(e)}
    
    return results

def main():
    base_folder = r'F:\obsdata\LIGHT'
    
    # Find all unprocessed folders
    unprocessed_folders = find_unprocessed_folders(base_folder)
    
    # Display and get user confirmation
    folders_to_process = display_and_confirm(unprocessed_folders)
    
    if not folders_to_process:
        return
    
    # Save the confirmed folders to a log file
    save_folders_log(folders_to_process, base_folder)
    
    # Process the confirmed folders
    results = process_folders(folders_to_process)
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE - SUMMARY")
    print(f"{'='*60}")
    for folder_name, result in results.items():
        if 'error' in result:
            print(f"{folder_name}: ERROR - {result['error']}")
        else:
            print(f"{folder_name}: {result['stars_analysed']} stars, {result['nonlinear_count']} nonlinear")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
