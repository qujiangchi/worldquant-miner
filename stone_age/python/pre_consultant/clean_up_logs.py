import os
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

class LogCleaner:
    def __init__(self, results_dir: str, max_files: int = 200):
        self.results_dir = Path(results_dir)
        self.max_files = max_files
        
    def clean_old_files(self) -> None:
        """Delete oldest files if count exceeds max_files."""
        try:
            # Get all files in the results directory
            files = list(self.results_dir.glob('*'))
            file_count = len(files)
            
            if file_count <= self.max_files:
                logger.info(f"File count ({file_count}) within limit ({self.max_files}). No cleanup needed.")
                return
                
            # Sort files by modification time
            files.sort(key=lambda x: x.stat().st_mtime)
            
            # Calculate how many files to delete
            files_to_delete = file_count - self.max_files
            logger.info(f"Found {file_count} files, removing oldest {files_to_delete} files")
            
            # Delete oldest files
            for file in files[:files_to_delete]:
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                logger.info(f"Deleting {file.name} (modified: {file_time})")
                file.unlink()
                
            logger.info(f"Cleanup complete. Remaining files: {self.max_files}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            logger.exception("Full traceback:")

def main():
    parser = argparse.ArgumentParser(description='Clean up old result files')
    parser.add_argument('--results-dir', type=str, default='./results',
                      help='Results directory to monitor (default: ./results)')
    parser.add_argument('--max-files', type=int, default=200,
                      help='Maximum number of files to keep (default: 200)')
    parser.add_argument('--interval-hours', type=float, default=1,
                      help='Hours between cleanup checks (default: 1)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cleanup.log')
        ]
    )
    
    # Create results directory if it doesn't exist
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    cleaner = LogCleaner(results_dir, args.max_files)
    interval_seconds = args.interval_hours * 3600
    
    logger.info(f"Starting cleanup monitor for {args.results_dir}")
    logger.info(f"Max files: {args.max_files}")
    logger.info(f"Check interval: {args.interval_hours} hours")
    
    try:
        while True:
            logger.info(f"Running cleanup check at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            cleaner.clean_old_files()
            
            # Schedule next run
            next_run = time.time() + interval_seconds
            next_run_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                        time.localtime(next_run))
            logger.info(f"Next cleanup check scheduled for: {next_run_time}")
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, exiting gracefully...")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
