#!/usr/bin/env python3

import os
import sys
import json
import requests
from datetime import datetime
import argparse
from pathlib import Path

class LSLDataUploader:
    def __init__(self, upload_url=None):
        self.upload_url = upload_url or "https://your-server.com/api/upload"
        self.data_dir = "lsl_data"
        
    def list_data_files(self):
        """List all CSV files in the data directory"""
        if not os.path.exists(self.data_dir):
            print(f"Data directory '{self.data_dir}' not found.")
            return []
        
        csv_files = sorted(Path(self.data_dir).glob("*.csv"))
        return csv_files
    
    def read_metadata(self, filepath):
        """Extract metadata from CSV file headers"""
        metadata = {
            "filename": os.path.basename(filepath),
            "file_size": os.path.getsize(filepath),
            "upload_timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    if 'Session Start:' in line:
                        metadata['session_start'] = line.split(':', 1)[1].strip()
                    elif 'Stream Name:' in line:
                        metadata['stream_name'] = line.split(':', 1)[1].strip()
                    elif 'Stream Type:' in line:
                        metadata['stream_type'] = line.split(':', 1)[1].strip()
                    elif 'Sampling Rate:' in line:
                        metadata['sampling_rate'] = line.split(':', 1)[1].strip()
                    elif 'Channel Count:' in line:
                        metadata['channel_count'] = line.split(':', 1)[1].strip()
                else:
                    break
        
        return metadata
    
    def upload_file(self, filepath, server_url=None):
        """Upload a single file to the server"""
        url = server_url or self.upload_url
        
        try:
            metadata = self.read_metadata(filepath)
            
            print(f"\nUploading: {os.path.basename(filepath)}")
            print(f"  Session: {metadata.get('session_start', 'Unknown')}")
            print(f"  Size: {metadata['file_size'] / 1024:.2f} KB")
            
            with open(filepath, 'rb') as f:
                files = {'file': (os.path.basename(filepath), f, 'text/csv')}
                data = {'metadata': json.dumps(metadata)}
                
                # Simulated upload - replace with actual server endpoint
                # response = requests.post(url, files=files, data=data)
                
                # For demonstration, we'll save locally with metadata
                metadata_file = filepath.replace('.csv', '_metadata.json')
                with open(metadata_file, 'w') as mf:
                    json.dump(metadata, mf, indent=2)
                
                print(f"  ✓ Metadata saved to: {metadata_file}")
                print(f"  Note: Actual upload to {url} would happen here")
                
                # If using real upload:
                # if response.status_code == 200:
                #     print(f"  ✓ Successfully uploaded")
                #     return True
                # else:
                #     print(f"  ✗ Upload failed: {response.status_code}")
                #     return False
                
                return True
                
        except Exception as e:
            print(f"  ✗ Error uploading file: {e}")
            return False
    
    def upload_all(self, delete_after=False):
        """Upload all CSV files in the data directory"""
        files = self.list_data_files()
        
        if not files:
            print("No data files found to upload.")
            return
        
        print(f"Found {len(files)} file(s) to upload")
        
        successful = 0
        failed = 0
        
        for filepath in files:
            if self.upload_file(filepath):
                successful += 1
                if delete_after:
                    os.remove(filepath)
                    print(f"  Deleted local file: {filepath}")
            else:
                failed += 1
        
        print(f"\n{'='*40}")
        print(f"Upload Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"{'='*40}")
    
    def upload_latest(self):
        """Upload only the most recent file"""
        files = self.list_data_files()
        
        if not files:
            print("No data files found.")
            return
        
        latest_file = max(files, key=os.path.getctime)
        print(f"Uploading latest file: {latest_file}")
        self.upload_file(latest_file)
    
    def show_status(self):
        """Show status of local data files"""
        files = self.list_data_files()
        
        if not files:
            print("No data files found.")
            return
        
        print(f"\nLocal Data Files ({len(files)} total):")
        print("-" * 60)
        
        total_size = 0
        for filepath in files:
            size = os.path.getsize(filepath)
            total_size += size
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            # Check if metadata exists
            metadata_exists = os.path.exists(str(filepath).replace('.csv', '_metadata.json'))
            status = "✓ Has metadata" if metadata_exists else "○ No metadata"
            
            print(f"{os.path.basename(filepath):30} {size/1024:8.2f} KB  {mod_time.strftime('%Y-%m-%d %H:%M')}  {status}")
        
        print("-" * 60)
        print(f"Total size: {total_size / 1024 / 1024:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Upload LSL data files to server')
    parser.add_argument('--url', type=str, help='Upload server URL')
    parser.add_argument('--all', action='store_true', help='Upload all files')
    parser.add_argument('--latest', action='store_true', help='Upload only the latest file')
    parser.add_argument('--status', action='store_true', help='Show status of local files')
    parser.add_argument('--delete', action='store_true', help='Delete files after successful upload')
    parser.add_argument('--file', type=str, help='Upload specific file')
    
    args = parser.parse_args()
    
    uploader = LSLDataUploader(upload_url=args.url)
    
    if args.status:
        uploader.show_status()
    elif args.all:
        uploader.upload_all(delete_after=args.delete)
    elif args.latest:
        uploader.upload_latest()
    elif args.file:
        if os.path.exists(args.file):
            uploader.upload_file(args.file)
        else:
            print(f"File not found: {args.file}")
    else:
        # Interactive mode
        print("LSL Data Uploader")
        print("=" * 40)
        
        while True:
            print("\nOptions:")
            print("1. Show status of local files")
            print("2. Upload all files")
            print("3. Upload latest file")
            print("4. Upload specific file")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '1':
                uploader.show_status()
            elif choice == '2':
                delete = input("Delete files after upload? (y/n): ").lower() == 'y'
                uploader.upload_all(delete_after=delete)
            elif choice == '3':
                uploader.upload_latest()
            elif choice == '4':
                files = uploader.list_data_files()
                if files:
                    print("\nAvailable files:")
                    for i, f in enumerate(files):
                        print(f"{i+1}. {os.path.basename(f)}")
                    
                    try:
                        idx = int(input("Enter file number: ")) - 1
                        if 0 <= idx < len(files):
                            uploader.upload_file(files[idx])
                        else:
                            print("Invalid selection")
                    except ValueError:
                        print("Invalid input")
                else:
                    print("No files available")
            elif choice == '5':
                print("Exiting...")
                break
            else:
                print("Invalid choice")

if __name__ == "__main__":
    main()