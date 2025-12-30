#!/usr/bin/env python3
"""
Stage 2: Merge Shards
This script merges the sharded .safetensors files created by preprocess.py
into the final, single-file .pth models required by the framework and
evaluation scripts.

This is run *after* preprocess.py is complete.

It merges one model at a time (first base_model, then final_model)
to be memory-efficient. It will still require enough RAM to hold
*one* full model, but not both at the same time.
"""

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Torch was not compiled with.*")
warnings.filterwarnings("ignore", message=".*You are using the default.*")
warnings.filterwarnings("ignore", message=".*AutoAWQ is officially deprecated.*")
warnings.filterwarnings("ignore", message=".*site-packages/torch/utils/cpp_extension.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*autoawq.*")
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)

import os
import torch
import json
import gc
import argparse
from safetensors.torch import load_file
from typing import Dict
import argcomplete

def merge_model(model_name: str, index_path: str, output_dir: str, prefix: str):
    """
    Loads all shards from an index.json file, merges them into a single
    dictionary in RAM, and saves them as a single .pth file.
    """
    print("="*80)
    print(f"🚀 Merging model: {model_name}")
    print(f"   Reading index: {index_path}")

    if not os.path.exists(index_path):
        print(f"⚠️ Index file not found, skipping merge for {model_name}.")
        print("="*80)
        return

    try:
        # Load the index file
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get("weight_map")
        if not weight_map:
            raise ValueError("Index file is valid JSON but contains no 'weight_map'.")

        # Find all unique shard files
        shard_files = sorted(list(set(weight_map.values())))
        print(f"   Found {len(shard_files)} shards to merge.")

        # This will hold the entire model in RAM
        full_model_dict = {}
        
        # Keep track of shard paths to delete *later*
        shard_paths_to_delete = []

        # Load one shard at a time
        for shard_file in shard_files:
            shard_path = os.path.join(output_dir, shard_file)
            if not os.path.exists(shard_path):
                print(f"   ❌ ERROR: Missing shard file: {shard_file}. Aborting.")
                return
            
            print(f"   ...loading shard: {shard_file}")
            # Load the shard (a dict of tensors) into CPU RAM
            shard_dict = load_file(shard_path, device="cpu")
            
            # Add its tensors to the main dictionary
            full_model_dict.update(shard_dict)
            
            # Add path to delete list
            shard_paths_to_delete.append(shard_path)

            # Clean up memory from this shard
            del shard_dict
            gc.collect()

        # Now, save the complete model to a single .pth file
        save_path = os.path.join(output_dir, f"{prefix}{model_name}.pth")
        print(f"\n   💾 Saving merged file to: {save_path}")
        
        torch.save(full_model_dict, save_path)
        
        print(f"   ✅ Successfully saved {model_name}.pth")
        del full_model_dict
        gc.collect()
        
        # --- NEW: Safe Deletion ---
        # Now that saving is successful, delete the shards
        print(f"   ...Cleaning up {len(shard_paths_to_delete)} shard files...")
        for shard_path in shard_paths_to_delete:
            try:
                os.remove(shard_path)
            except Exception as e:
                print(f"   ⚠️ Could not delete shard {shard_path}: {e}")
        
        # Also delete the index file
        try:
            os.remove(index_path)
            print(f"   ...Cleaned up index file: {os.path.basename(index_path)}")
        except Exception as e:
            print(f"   ⚠️ Could not delete index {index_path}: {e}")

        print("="*80)

    except Exception as e:
        print(f"\n❌ FAILED to merge {model_name}: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Merge sharded .safetensors files into single .pth files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output-dir",
        dest="output_dir",
        type=str,
        default=".",
        help="📁 Output directory where shards are located",
    )
    parser.add_argument(
        "--prefix",
        "--output-prefix",
        dest="output_prefix",
        type=str,
        default="",
        help="🏷️ Prefix used for the output files (must match preprocess.py)",
    )

    argcomplete.autocomplete(parser)
    
    args = parser.parse_args()

    # --- Create output directory if it doesn't exist ---
    if args.output_dir != '.':  # Only create if not current directory
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"📁 Output directory created: {args.output_dir}")

    # 1. Define paths
    base_index_path = os.path.join(args.output_dir, f"{args.output_prefix}base_model.safetensors.index.json")
    final_index_path = os.path.join(args.output_dir, f"{args.output_prefix}final_model.safetensors.index.json")
    
    # 2. Merge Base Model
    merge_model("base_model", base_index_path, args.output_dir, args.output_prefix)
    
    # 3. Merge Final Model
    merge_model("final_model", final_index_path, args.output_dir, args.output_prefix)
    
    print("\n🎉 Shard merging complete.")

if __name__ == "__main__":
    main()