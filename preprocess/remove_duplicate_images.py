import os
import imagehash
from PIL import Image
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('duplicate_removal.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_image_hash(image_path):
    try:
        return str(imagehash.average_hash(Image.open(image_path)))
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def remove_duplicates():
    train_dir = 'dataset/trainset'
    annotations_file = 'dataset/annotations/train.txt'
    
    # Read annotations
    with open(annotations_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    logger.info(f"Processing {len(lines)} images for duplicates")
    
    # Dictionary to store image hashes and their paths
    hash_dict = defaultdict(list)
    
    # Calculate hashes for all images
    for line in lines:
        image_name, label = line.split('\t')
        image_path = os.path.join(train_dir, image_name)
        
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
            
        image_hash = get_image_hash(image_path)
        if image_hash:
            hash_dict[image_hash].append((image_name, label))
    
    # Find and remove duplicates
    kept_lines = []
    removed_count = 0
    
    for image_hash, entries in hash_dict.items():
        if len(entries) > 1:
            # Keep the entry with the lowest label value
            entries.sort(key=lambda x: int(x[1]))
            kept_entry = entries[0]
            kept_lines.append(f"{kept_entry[0]}\t{kept_entry[1]}")
            
            # Remove other duplicate images
            for entry in entries[1:]:
                image_path = os.path.join(train_dir, entry[0])
                try:
                    os.remove(image_path)
                    removed_count += 1
                    logger.info(f"Removed duplicate image: {entry[0]} (label: {entry[1]})")
                except Exception as e:
                    logger.error(f"Error removing file {image_path}: {e}")
        else:
            # Keep single entries
            kept_lines.append(f"{entries[0][0]}\t{entries[0][1]}")
    
    # Write updated annotations
    with open(annotations_file, 'w') as f:
        f.write('\n'.join(kept_lines) + '\n')
    
    logger.info(f"Removed {removed_count} duplicate images")
    logger.info(f"Kept {len(kept_lines)} unique images")

if __name__ == "__main__":
    logger.info("Starting duplicate image removal process")
    remove_duplicates()
    logger.info("Duplicate removal completed!") 