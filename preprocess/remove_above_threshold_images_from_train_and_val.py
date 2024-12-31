
import os
import shutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_dataset(input_file, output_file, threshold=192):
    logger.info(f"Starting to clean dataset: {input_file}")
    logger.info(f"Using threshold value: {threshold}")
    
    # Read the original annotation file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    logger.info(f"Read {len(lines)} lines from {input_file}")
    
    # Filter out empty lines and parse the data
    valid_lines = [line.strip() for line in lines if line.strip()]
    logger.info(f"Found {len(valid_lines)} valid lines after removing empty lines")
    
    # Keep only entries where label <= threshold
    filtered_lines = []
    removed_count = 0
    
    for line in valid_lines:
        try:
            image_path, label = line.split('\t')
            label = int(label)
            
            if label <= threshold:
                filtered_lines.append(line)
            else:
                removed_count += 1
                # Get image filename for removal
                image_name = image_path.strip()
                logger.info(f"Removing image with label {label}: {image_name}")
                
                # Remove the actual image file
                image_folder = 'dataset/trainset' if 'train.txt' in input_file else 'dataset/valset'
                image_path = f"{image_folder}/{image_name}"
                try:
                    os.remove(image_path)
                    logger.info(f"Deleted image file: {image_path}")
                except FileNotFoundError:
                    logger.warning(f"Could not find image file: {image_path}")
                except Exception as e:
                    logger.error(f"Error removing file {image_path}: {e}")
                
        except ValueError:
            # Keep empty lines or malformed lines as is
            filtered_lines.append(line)
            logger.warning(f"Found malformed line: {line}")
    
    # Write the filtered data back to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(filtered_lines) + '\n')
    
    logger.info(f"Wrote {len(filtered_lines)} lines to {output_file}")
    return removed_count

# Process training set
logger.info("Starting training set cleaning")
train_removed = clean_dataset('dataset/annotations/train.txt', 'dataset/annotations/train_cleaned.txt')
logger.info(f"Removed {train_removed} images from training set")

# Process validation set
logger.info("Starting validation set cleaning") 
val_removed = clean_dataset('dataset/annotations/val.txt', 'dataset/annotations/val_cleaned.txt')
logger.info(f"Removed {val_removed} images from validation set")

# Replace original files with cleaned versions
shutil.move('dataset/annotations/train_cleaned.txt', 'dataset/annotations/train.txt')
shutil.move('dataset/annotations/val_cleaned.txt', 'dataset/annotations/val.txt')

logger.info("Dataset cleaning completed!")
