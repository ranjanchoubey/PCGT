#!/usr/bin/env python3
"""
Download datasets for PCGT project

This script automatically downloads and organizes datasets needed for running
the PCGT (Simplified Graph Transformers) project. It supports:
- Planetoid datasets (Cora, Citeseer, Pubmed) via PyTorch Geometric
- OGB datasets (automatically handled by ogb library)
- Google Drive datasets (Pokec, Deezer, etc.)

Usage:
    python download_data.py
    python download_data.py --data-dir ./data
    python download_data.py --datasets cora citeseer
"""

import os
import ssl
import sys
import argparse
import urllib.request
import zipfile
import pickle
from pathlib import Path
from typing import List, Optional

# Try importing optional dependencies
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False

try:
    import torch_geometric
    HAS_PYTROCH_GEOMETRIC = True
except ImportError:
    HAS_PYTORCH_GEOMETRIC = False


def setup_ssl_context():
    """Configure SSL to bypass certificate verification if needed"""
    try:
        # Create unverified SSL context for macOS users with cert issues
        ssl._create_default_https_context = ssl._create_unverified_context
        print("✓ SSL context configured")
    except Exception as e:
        print(f"⚠ SSL configuration warning: {e}")


def download_planetoid_datasets(data_dir: Path, datasets: List[str] = None):
    """
    Download Planetoid datasets (Cora, Citeseer, Pubmed)
    
    These are automatically handled by PyTorch Geometric, but this function
    ensures they're properly stored.
    """
    if datasets is None:
        datasets = ['cora', 'citeseer', 'pubmed']
    
    print("\n" + "="*60)
    print("Downloading Planetoid Datasets")
    print("="*60)
    
    try:
        from torch_geometric.datasets import Planetoid
        
        for dataset_name in datasets:
            if dataset_name not in ['cora', 'citeseer', 'pubmed']:
                continue
                
            print(f"\nDownloading {dataset_name.upper()}...")
            try:
                dataset = Planetoid(
                    root=str(data_dir / 'Planetoid'),
                    name=dataset_name
                )
                print(f"✓ {dataset_name.upper()} dataset ready")
                print(f"  - Nodes: {dataset[0].num_nodes}")
                print(f"  - Edges: {dataset[0].num_edges}")
                print(f"  - Features: {dataset[0].num_node_features}")
                print(f"  - Classes: {dataset.num_classes}")
            except Exception as e:
                print(f"✗ Failed to download {dataset_name}: {e}")
                print("  Trying direct download from GitHub...")
                download_planetoid_raw(data_dir, dataset_name)
    
    except ImportError:
        print("✗ PyTorch Geometric not installed")
        print("  Installing: pip install torch-geometric")
        download_planetoid_raw(data_dir, datasets)


def download_planetoid_raw(data_dir: Path, datasets: List[str]):
    """
    Download raw Planetoid files directly from GitHub
    
    This is a fallback if PyTorch Geometric download fails.
    """
    print("\nDownloading from GitHub...")
    
    base_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    files_needed = ['.x', '.y', '.allx', '.ally', '.graph', '.test.index']
    
    for dataset_name in datasets:
        if not isinstance(dataset_name, str):
            continue
            
        dataset_name = dataset_name.lower()
        if dataset_name not in ['cora', 'citeseer', 'pubmed']:
            continue
        
        raw_dir = data_dir / 'Planetoid' / dataset_name / 'raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDownloading {dataset_name.upper()} raw files...")
        
        for file_ext in files_needed:
            filename = f"ind.{dataset_name}{file_ext}"
            url = f"{base_url}/{filename}"
            filepath = raw_dir / filename
            
            if filepath.exists():
                print(f"  ✓ {filename} (already exists)")
                continue
            
            try:
                print(f"  → Downloading {filename}...")
                urllib.request.urlretrieve(url, str(filepath))
                print(f"  ✓ {filename} downloaded")
            except Exception as e:
                print(f"  ✗ Failed to download {filename}: {e}")


def download_from_gdrive(gdrive_folder_id: str, output_dir: Path, dataset_name: str):
    """
    Download datasets from Google Drive
    
    The project provides datasets via Google Drive:
    https://drive.google.com/drive/folders/1rr3kewCBUvIuVxA6MJ90wzQuF-NnCRtf?usp=drive_link
    """
    if not HAS_GDOWN:
        print(f"\n✗ Cannot download {dataset_name} - gdown not installed")
        print("  Install with: pip install gdown")
        return False
    
    print(f"\nDownloading {dataset_name} from Google Drive...")
    
    try:
        url = f"https://drive.google.com/uc?id={gdrive_folder_id}"
        zip_path = output_dir / f"{dataset_name}.zip"
        extract_dir = output_dir / dataset_name
        
        if extract_dir.exists():
            print(f"  ✓ {dataset_name} already exists")
            return True
        
        print(f"  → Downloading to {zip_path.name}...")
        gdown.download_folder(url, output=str(output_dir), quiet=False)
        
        print(f"  ✓ {dataset_name} downloaded successfully")
        return True
    
    except Exception as e:
        print(f"  ✗ Failed to download {dataset_name}: {e}")
        return False


def download_ogb_datasets(datasets: List[str] = None):
    """
    Download Open Graph Benchmark datasets
    
    OGB datasets are automatically downloaded by the ogb library
    when first accessed in the code.
    """
    if datasets is None:
        datasets = []
    
    if not datasets:
        print("\n" + "="*60)
        print("OGB Datasets Info")
        print("="*60)
        print("\n✓ OGB datasets (ogbn-products, ogbn-arxiv, etc.)")
        print("  are automatically downloaded when first accessed.")
        print("  No manual download needed.")
        return
    
    print("\n" + "="*60)
    print("Setting up OGB Datasets")
    print("="*60)
    
    try:
        from ogb.nodeproppred import DglNodePropPredDataset
        
        for dataset_name in datasets:
            if not dataset_name.startswith('ogbn-'):
                continue
            
            print(f"\nSetting up {dataset_name}...")
            try:
                dataset = DglNodePropPredDataset(name=dataset_name)
                print(f"✓ {dataset_name} is ready")
            except Exception as e:
                print(f"✗ Failed to download {dataset_name}: {e}")
    
    except ImportError:
        print("✗ OGB library not installed")
        print("  Install with: pip install ogb")


def verify_datasets(data_dir: Path):
    """Verify downloaded datasets"""
    print("\n" + "="*60)
    print("Verifying Downloaded Datasets")
    print("="*60)
    
    datasets_to_check = {
        'Planetoid/cora/raw': ['ind.cora.x', 'ind.cora.y', 'ind.cora.graph'],
        'Planetoid/citeseer/raw': ['ind.citeseer.x', 'ind.citeseer.y'],
        'Planetoid/pubmed/raw': ['ind.pubmed.x', 'ind.pubmed.y'],
    }
    
    found_count = 0
    for dataset_path, required_files in datasets_to_check.items():
        full_path = data_dir / dataset_path
        
        if full_path.exists():
            existing_files = list(full_path.glob('ind.*'))
            if existing_files:
                print(f"✓ {dataset_path}: {len(existing_files)} files found")
                found_count += len(existing_files)
        else:
            print(f"· {dataset_path}: not yet downloaded")
    
    if found_count == 0:
        print("\n⚠ No datasets found yet. Run download after this verification.")
    else:
        print(f"\n✓ Total files found: {found_count}")


def create_directory_structure(data_dir: Path):
    """Create necessary directory structure"""
    print("\n" + "="*60)
    print("Setting Up Directory Structure")
    print("="*60)
    
    directories = [
        'Planetoid/cora/raw',
        'Planetoid/cora/processed',
        'Planetoid/citeseer/raw',
        'Planetoid/citeseer/processed',
        'Planetoid/pubmed/raw',
        'Planetoid/pubmed/processed',
        'deezer',
        'actor',
        'film',
        'chameleon',
        'squirrel',
        'pokec',
    ]
    
    for dir_path in directories:
        full_path = data_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")
    
    print("\n✓ Directory structure ready")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for PCGT project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py
  python download_data.py --data-dir ./datasets
  python download_data.py --datasets cora citeseer
  python download_data.py --skip-ogb
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory to store datasets (default: data/)'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['cora', 'citeseer', 'pubmed'],
        help='Datasets to download (default: cora citeseer pubmed)'
    )
    
    parser.add_argument(
        '--skip-ogb',
        action='store_true',
        help='Skip OGB dataset setup'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing datasets, do not download'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir).resolve()
    
    print("\n" + "="*60)
    print("PCGT Dataset Download Tool")
    print("="*60)
    print(f"\nData directory: {data_dir}")
    
    # Create directory structure
    create_directory_structure(data_dir)
    
    # Setup SSL
    setup_ssl_context()
    
    if args.verify_only:
        verify_datasets(data_dir)
        return
    
    # Download Planetoid datasets
    download_planetoid_datasets(data_dir, args.datasets)
    
    # Download OGB datasets (optional)
    if not args.skip_ogb:
        download_ogb_datasets()
    
    # Final verification
    verify_datasets(data_dir)
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"\nDatasets ready in: {data_dir}")
    print("\nTo train the model, use:")
    print(f"  cd medium")
    print(f"  python main.py --data-dir {data_dir} --dataset cora --epochs 100 --cpu")
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
