"""
Convert a folder of .npz behavior cloning recordings into a single HDF5 dataset.
HDF5 is highly optimized for disk streaming and is required for Kaggle/Google Cloud
training where I/O bottlenecks will throttle the GPU.
"""
import os
import glob
import json
import argparse
import numpy as np
try:
    import h5py
except ImportError:
    print("❌ Error: h5py not installed. Run 'pip install h5py'")
    import sys
    sys.exit(1)
from tqdm import tqdm

def convert_to_hdf5(input_dir: str, output_file: str):
    npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return
        
    print(f"Found {len(npz_files)} .npz files. Compiling to HDF5: {output_file}")
    
    # Pre-calculate sizes
    total_frames = 0
    for f in npz_files:
        try:
            with np.load(f) as data:
                total_frames += len(data['features'])
        except: pass
        
    print(f"Total frames to process: {total_frames}")
    
    with h5py.File(output_file, 'w') as h5f:
        # Create datasets
        # We don't know the exact dims until we read the first file
        first_valid = None
        for f in npz_files:
            try:
                with np.load(f, allow_pickle=True) as data:
                    first_valid = data
                    break
            except: pass
            
        struct_dim = first_valid['features'].shape[1]
        cnn_shape = first_valid['cnn_frames'].shape[1:]
        
        dset_struct = h5f.create_dataset('features', shape=(total_frames, struct_dim), dtype='float32')
        dset_cnn = h5f.create_dataset('cnn_frames', shape=(total_frames, *cnn_shape), dtype='uint8')
        
        # Strings are tricky in HDF5, we'll store them as variable length strings
        dt_str = h5py.string_dtype(encoding='utf-8')
        dset_actions = h5f.create_dataset('actions', shape=(total_frames,), dtype=dt_str)
        
        idx = 0
        for f in tqdm(npz_files, desc="Converting"):
            try:
                with np.load(f, allow_pickle=True) as data:
                    features = data['features']
                    cnn_frames = data['cnn_frames']
                    actions = data['actions']
                    
                    n = len(features)
                    dset_struct[idx:idx+n] = features
                    dset_cnn[idx:idx+n] = cnn_frames
                    
                    # Store string actions
                    str_actions = [json.dumps(a) if isinstance(a, dict) else str(a) for a in actions]
                    dset_actions[idx:idx+n] = str_actions
                    
                    idx += n
            except Exception as e:
                print(f"\nSkipping {f} due to error: {e}")
                
    print(f"✅ HDF5 conversion complete! Saved to {output_file}")
    print("You can now upload this single file to Kaggle/Colab for high-speed training.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./data/recordings', help='Input directory with .npz files')
    parser.add_argument('--output', type=str, default='./data/dataset.h5', help='Output HDF5 file path')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    convert_to_hdf5(args.input, args.output)
